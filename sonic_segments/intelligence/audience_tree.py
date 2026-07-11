"""Audience tree + blast planner: max-reach hyperpersonalization.

The tree mirrors the real media-planning funnel (see data/audience_tree.json
_meta for the research): Market (state/DMA) -> Age band -> Taste cluster.
Every impression lands in exactly one LEAF = (market, age, taste), and every
leaf resolves to a scored music brief — the taste cluster's direction
re-flavored by the market's geo pack. Personalization floor without any
listener login: a hip-hop 18-24 leaf sounds like trap in Atlanta and drill
in New York.

Blast planning separates PLANNING from RENDERING: thousands of leaves
deduplicate into few unique briefs (brief = region-pack x taste x mood), and
ad units aggregate leaves per (brief x age band x platform) — exactly how a
trafficker would group states into one ad set. Two music modes:

  rendered        each brief maps to the closest already-rendered cut
  platform_sound  ONE music-free master (voiceover preserved by the existing
                  separation stack) + a per-leaf sound pick from the
                  platform's PRECLEARED library (TikTok Commercial Music
                  Library / Meta Sound Collection — real songs, zero
                  clearance, no re-render; see data/platform_music.json)

Compliance: race/ethnicity is not targetable on any platform; leaves use
age, geography, language-market and music-affinity only.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from .demographics import DemographicsKB
from .rollout import (
    _SPEC_BUILDERS,
    _slug,
    GeoKB,
    RolloutPlanner,
    canonical_targeting,
    validate_plan,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TREE_PATH = DATA_DIR / "audience_tree.json"
PLATFORM_MUSIC_PATH = DATA_DIR / "platform_music.json"

SOUND_LIBRARY_PLATFORMS = {"tiktok", "meta"}
CLEAN_MASTER = "master_voiceover_only.mp4"


class AudienceTree:
    """Loads the funnel KB and expands selections into leaves and briefs."""

    def __init__(self, path: str | Path = TREE_PATH) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.meta = data.get("_meta", {})
        self.funnel = data["funnel"]
        self.age_bands: dict[str, dict[str, Any]] = {b["id"]: b for b in data["age_bands"]}
        self.taste_clusters: list[str] = data["taste_clusters"]
        self.us_states: dict[str, dict[str, Any]] = data["us_states"]
        self.dma_drilldown: dict[str, list[str]] = data.get("dma_drilldown", {})
        self.intl_markets: list[str] = data.get("intl_markets", [])
        self.geo_kb = GeoKB.default()
        self.demographics = DemographicsKB.default()

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls) -> "AudienceTree":
        return cls()

    # ---- UI description ----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        regions: dict[str, dict[str, Any]] = {}
        for code, st in self.us_states.items():
            r = regions.setdefault(st["region"], {"id": st["region"], "states": [], "pop_m": 0.0})
            r["states"].append(code)
            r["pop_m"] = round(r["pop_m"] + st["pop_m"], 1)
        for rid, r in regions.items():
            pack = self.geo_kb.geos.get(rid, {})
            r["label"] = pack.get("label", rid)
            r["genres"] = pack.get("music_bias", {}).get("genre_boosts", [])[:3]
        return {
            "funnel": self.funnel,
            "us_states": self.us_states,
            "regions": list(regions.values()),
            "dma_drilldown": self.dma_drilldown,
            "age_bands": list(self.age_bands.values()),
            "taste_clusters": [
                {
                    "id": t,
                    "label": self.demographics.segments[t]["label"],
                    "genres": self.demographics.segments[t].get("core_genres", [])[:3],
                }
                for t in self.taste_clusters
            ],
            "intl_markets": [
                {"id": g, "label": self.geo_kb.get(g)["label"],
                 "genres": self.geo_kb.get(g)["music_bias"]["genre_boosts"][:3]}
                for g in self.intl_markets
            ],
            "music_modes": [
                {"id": "rendered", "label": "Rendered cuts",
                 "note": "Each audience brief maps to the closest re-scored cut from this campaign."},
                {"id": "platform_sound", "label": "Platform-native sound (real songs, precleared)",
                 "note": "One music-free master + a precleared library pick per audience "
                         "(TikTok Commercial Music Library / Meta Sound Collection). "
                         "TikTok + Meta only."},
            ],
        }

    # ---- expansion -----------------------------------------------------------

    def expand(self, selection: dict[str, Any]) -> list[dict[str, Any]]:
        """Selection -> leaves. Empty age/taste lists mean 'all'; markets must
        be picked explicitly (that's the geo-first decision)."""
        states = [s for s in selection.get("us_states", []) if s in self.us_states]
        intl = [g for g in selection.get("intl", []) if g in self.intl_markets]
        if not states and not intl:
            raise ValueError("Pick at least one market (states and/or international).")
        bands = [b for b in selection.get("age_bands", []) if b in self.age_bands] or list(self.age_bands)
        tastes = [t for t in selection.get("tastes", []) if t in self.taste_clusters] or list(self.taste_clusters)
        moods = selection.get("moods") or ["hype"]

        leaves: list[dict[str, Any]] = []
        for mood in moods:
            for band_id in bands:
                share = self.age_bands[band_id]["total_pop_share"]
                for taste in tastes:
                    for code in states:
                        st = self.us_states[code]
                        leaves.append({
                            "market_type": "us_state", "market": code,
                            "market_label": st["name"], "geo_pack": st["region"],
                            "age_band": band_id, "taste": taste, "mood": mood,
                            "pop_m": round(st["pop_m"] * share, 3),
                        })
                    for gid in intl:
                        pack = self.geo_kb.get(gid)
                        leaves.append({
                            "market_type": "intl", "market": gid,
                            "market_label": pack["label"], "geo_pack": gid,
                            "age_band": band_id, "taste": taste, "mood": mood,
                            "pop_m": 0.0,  # census weighting is US-only; intl splits evenly
                        })
        return leaves

    def unique_briefs(self, leaves: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
        """Leaves collapse into unique music briefs: (geo_pack, taste, mood).
        This is the dedupe that makes blast affordable — thousands of leaves,
        dozens of briefs."""
        briefs: dict[tuple, dict[str, Any]] = {}
        for leaf in leaves:
            key = (leaf["geo_pack"], leaf["taste"], leaf["mood"])
            entry = briefs.get(key)
            if entry is None:
                geo = self.geo_kb.geos.get(leaf["geo_pack"])
                # Taste clusters keep their genre identity; the market flavors it.
                direction = self.demographics.direction(
                    leaf["taste"], leaf["mood"], geo=geo, geo_weight="flavor"
                )
                entry = briefs[key] = {"direction": direction, "leaves": []}
            entry["leaves"].append(leaf)
        return briefs


class BlastPlanner:
    """Expands a selection into platform-ready ad units, one per
    (brief x age band x platform), states aggregated like a trafficker would."""

    def __init__(self, tree: AudienceTree | None = None) -> None:
        self.tree = tree or AudienceTree.default()
        self.platform_music = json.loads(PLATFORM_MUSIC_PATH.read_text(encoding="utf-8"))
        self.rollout = RolloutPlanner()

    def plan(
        self,
        manifest: dict[str, Any],
        selection: dict[str, Any],
        platform_ids: list[str],
        music_mode: str = "rendered",
        total_budget: float = 5000.0,
        currency: str = "USD",
    ) -> dict[str, Any]:
        if music_mode not in {"rendered", "platform_sound"}:
            raise ValueError(f"Unknown music mode: {music_mode}")
        platform_ids = [p for p in platform_ids if p in self.rollout.platforms.platforms]
        skipped: list[dict[str, str]] = []
        if music_mode == "platform_sound":
            for pid in list(platform_ids):
                if pid not in SOUND_LIBRARY_PLATFORMS:
                    platform_ids.remove(pid)
                    skipped.append({
                        "variant": "platform_sound", "platform": pid, "geo": "-",
                        "reason": f"{pid} has no precleared ad music library; use rendered mode there",
                    })
        if not platform_ids:
            raise ValueError("No usable platforms for this music mode.")

        leaves = self.tree.expand(selection)
        briefs = self.tree.unique_briefs(leaves)
        brand = manifest.get("brand") or "BRAND"

        ad_units: list[dict[str, Any]] = []
        for (geo_pack_id, taste, mood), entry in briefs.items():
            direction = entry["direction"]
            segment = self.tree.demographics.get(taste)
            creative = self._creative_for(manifest, direction, music_mode)
            by_band: dict[str, list[dict[str, Any]]] = {}
            for leaf in entry["leaves"]:
                by_band.setdefault(leaf["age_band"], []).append(leaf)

            for band_id, band_leaves in by_band.items():
                band = self.tree.age_bands[band_id]
                geo = self._aggregate_geo(geo_pack_id, band_leaves)
                canon = canonical_targeting(segment, geo)
                canon["age_min"], canon["age_max"] = band["range"]
                pop = round(sum(l["pop_m"] for l in band_leaves), 2)
                for pid in platform_ids:
                    if not canon["geo"].get(pid):
                        skipped.append({
                            "variant": direction.segment_label, "platform": pid,
                            "geo": geo["label"],
                            "reason": f"{geo['label']} is not addressable on {pid}",
                        })
                        continue
                    name = (f"SS_{_slug(brand)}_{_slug(taste)}_{_slug(mood)}_"
                            f"{_slug(geo_pack_id)}_{band_id.upper()}_{_slug(pid)}")
                    unit = {
                        "name": name,
                        "platform": pid,
                        "platform_label": self.rollout.platforms.get(pid)["label"],
                        "variant_file": creative["file"],
                        "variant_label": creative["label"],
                        "segment_id": taste,
                        "segment_label": f"{direction.segment_label} · {band['label']}",
                        "mood": mood,
                        "geo_id": geo_pack_id,
                        "geo_label": geo["label"],
                        "track": creative["track"],
                        "match_score": creative["match_score"],
                        "leaf_count": len(band_leaves),
                        "addressable_pop_m": pop,
                        "canonical_targeting": canon,
                        "platform_spec": _SPEC_BUILDERS[pid](canon),
                        "export_format": self.rollout.platforms.get(pid).get("export_format", ""),
                        "why": (f"{direction.rationale} Age band {band['label']} "
                                f"across {len(band_leaves)} market(s)."),
                        "demo_only": creative["demo_only"],
                        "budget_weight": max(pop, 0.2) * max(creative["match_score"], 0.05),
                    }
                    if music_mode == "platform_sound":
                        unit["sound_brief"] = self._sound_brief(pid, direction, manifest)
                    ad_units.append(unit)

        if not ad_units:
            raise ValueError("Blast produced no ad units — check the selection.")

        wsum = sum(u["budget_weight"] for u in ad_units)
        for u in ad_units:
            u["budget"] = round(total_budget * u.pop("budget_weight") / wsum, 2)
        drift = round(total_budget - sum(u["budget"] for u in ad_units), 2)
        if abs(drift) >= 0.01:
            max(ad_units, key=lambda u: u["budget"])["budget"] += drift

        plan = {
            "campaign_id": manifest.get("id", ""),
            "brand": brand,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": f"blast/{music_mode}",
            "platforms": platform_ids,
            "geo_ids": sorted({u["geo_id"] for u in ad_units}),
            "total_budget": total_budget,
            "currency": currency,
            "ad_units": ad_units,
            "skipped": skipped,
            "summary": {
                "variants": len(manifest.get("variants", [])) or 1,
                "leaves": len(leaves),
                "unique_briefs": len(briefs),
                "ad_units": len(ad_units),
                "platforms": len(platform_ids),
                "geos": len({u["geo_id"] for u in ad_units}),
                # People, not leaves: the same (market, age) population is
                # shared across taste leaves, so dedupe before summing.
                "addressable_pop_m": round(sum(
                    {(l["market"], l["age_band"]): l["pop_m"] for l in leaves}.values()
                ), 1),
                "skipped": len(skipped),
            },
        }
        issues = validate_plan(plan)
        if music_mode == "platform_sound":
            # Library tracks are precleared by the platform; the demo-only
            # gate applies to baked-in audio, not to the music-free master.
            issues = [i for i in issues if "DEMO-ONLY" not in i]
        plan["issues"] = issues
        return plan

    # ---- helpers -------------------------------------------------------------

    def _aggregate_geo(self, geo_pack_id: str, leaves: list[dict[str, Any]]) -> dict[str, Any]:
        """One trafficking geo for a group of leaves: US states aggregate into
        state lists; intl markets use their pack's platform keys."""
        states = [l for l in leaves if l["market_type"] == "us_state"]
        if not states:
            return self.tree.geo_kb.get(geo_pack_id)
        names = [self.tree.us_states[l["market"]]["name"] for l in states]
        label = self.tree.geo_kb.geos.get(geo_pack_id, {}).get("label", geo_pack_id)
        return {
            "id": geo_pack_id,
            "label": f"{label} ({len(names)} state{'s' if len(names) > 1 else ''})",
            "music_bias": self.tree.geo_kb.geos.get(geo_pack_id, {}).get("music_bias", {}),
            "rationale": self.tree.geo_kb.geos.get(geo_pack_id, {}).get("rationale", ""),
            "platform_geo": {
                "meta": {"regions": names},
                "google_ads": [f"{n}, United States" for n in names],
                "dv360": names,
                "tiktok": names,
                "spotify": names,
            },
        }

    def _creative_for(self, manifest: dict[str, Any], direction, music_mode: str) -> dict[str, Any]:
        if music_mode == "platform_sound":
            return {
                "file": CLEAN_MASTER,
                "label": "Music-free master (voiceover preserved)",
                "track": {"name": "Platform library pick (see sound brief)", "source": "platform_library",
                          "license": "precleared by platform", "demo_only": False},
                "match_score": 0.75,
                "demo_only": False,
            }
        variants = manifest.get("variants", [])
        if not variants:
            raise ValueError("Rendered mode needs at least one rendered cut in the manifest.")
        want = set(direction.genres)

        def fit(v: dict[str, Any]) -> float:
            got = set(v.get("direction", {}).get("genres", []))
            genre_overlap = len(want & got) / max(1, len(want | got))
            mood_match = 1.0 if v.get("mood") == direction.mood else 0.0
            return mood_match * 2 + genre_overlap

        best = max(variants, key=fit)
        return {
            "file": best["file"],
            "label": best.get("label", ""),
            "track": best.get("track", {}),
            "match_score": float(best.get("match", {}).get("total", 0.5)),
            "demo_only": bool(best.get("track", {}).get("demo_only")),
        }

    def _sound_brief(self, platform_id: str, direction, manifest: dict[str, Any]) -> dict[str, Any]:
        library = next(l for l in self.platform_music["libraries"] if l["platform"] == platform_id)
        duration = int(float(manifest.get("duration") or 30))
        return {
            "library": library["label"],
            "browse": library["browse"],
            "launch_path": library["launch_paths"][0]["label"],
            "filters": {
                "genre": direction.genres[:3],
                "mood": direction.mood_label,
                "duration_sec": duration,
                "tempo_bpm": [round(direction.tempo_range[0]), round(direction.tempo_range[1])],
            },
            "search_queries": direction.search_terms[:3],
            "sound_direction": direction.prompt,
            "note": "Pick a track matching these filters from the precleared library; "
                    "the platform streams the licensed song over the music-free master.",
        }
