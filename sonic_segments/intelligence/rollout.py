"""Rollout planner: turn rendered variants into platform-ready ad units.

This is the distribution intelligence layer. A campaign gives us N re-scored
cuts, one per (demographic, mood[, geo]). Every major ad platform already
supports many creatives under one campaign with audience/geo splits at the
ad-set / ad-group / line-item level (see data/ad_platforms.json, sources in
_meta) — what no platform does is GENERATE the per-audience creative. We do,
so the planner's job is to emit, for each cut, the exact targeting spec each
platform needs to put that cut in front of the audience its music was scored
for: geo as tight as ZIP/DMA/pin-radius, age/gender from the segment, and
music-interest targeting (Meta interest nodes, Google affinity segments,
Spotify streamed-genre targeting).

Everything here is deterministic and offline: KB lookups + spec shaping,
no model calls, so it can run synchronously in the web layer and be
benchmarked exhaustively (eval/rollout_benchmark.py).
"""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from .demographics import DemographicsKB

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PLATFORMS_PATH = DATA_DIR / "ad_platforms.json"
GEO_PATH = DATA_DIR / "geo_music.json"

DEFAULT_BUDGET = 1000.0

# Core-genre -> Google/DV360 affinity segment names (public affinity taxonomy).
GENRE_AFFINITY = {
    "country": "Country Music Fans",
    "country pop": "Country Music Fans",
    "texas country": "Country Music Fans",
    "americana": "Country Music Fans",
    "hip hop": "Rap & Hip Hop Fans",
    "trap": "Rap & Hip Hop Fans",
    "southern hip hop": "Rap & Hip Hop Fans",
    "west coast hip hop": "Rap & Hip Hop Fans",
    "east coast hip hop": "Rap & Hip Hop Fans",
    "drill": "Rap & Hip Hop Fans",
    "uk drill": "Rap & Hip Hop Fans",
    "grime": "Rap & Hip Hop Fans",
    "pop rap": "Rap & Hip Hop Fans",
    "desi hip hop": "Rap & Hip Hop Fans",
    "arabic hip hop": "Rap & Hip Hop Fans",
    "edm": "Electronica & Dance Music Fans",
    "house": "Electronica & Dance Music Fans",
    "techno": "Electronica & Dance Music Fans",
    "melodic house": "Electronica & Dance Music Fans",
    "uk garage": "Electronica & Dance Music Fans",
    "festival edm": "Electronica & Dance Music Fans",
    "hyperpop": "Pop Music Fans",
    "pop": "Pop Music Fans",
    "dance pop": "Pop Music Fans",
    "k-pop": "Pop Music Fans",
    "j-pop": "Pop Music Fans",
    "latin pop": "Latino Music Fans",
    "reggaeton": "Latino Music Fans",
    "latin trap": "Latino Music Fans",
    "latin urbano": "Latino Music Fans",
    "regional mexican": "Latino Music Fans",
    "corridos tumbados": "Latino Music Fans",
    "cumbia": "Latino Music Fans",
    "funk carioca": "Latino Music Fans",
    "sertanejo": "Latino Music Fans",
    "rock": "Rock Music Fans",
    "indie rock": "Indie & Alternative Rock Fans",
    "indie pop": "Indie & Alternative Rock Fans",
    "indie folk": "Folk & Traditional Music Fans",
    "alternative": "Indie & Alternative Rock Fans",
    "pop punk": "Rock Music Fans",
    "metal": "Metal Fans",
    "hard rock": "Rock Music Fans",
    "southern rock": "Rock Music Fans",
    "r&b": "Rhythm & Blues Fans",
    "neo-soul": "Rhythm & Blues Fans",
    "gospel": "Christian & Gospel Music Fans",
    "christian contemporary": "Christian & Gospel Music Fans",
    "jazz": "Jazz Fans",
    "classical": "Classical Music Fans",
    "film score": "Classical Music Fans",
    "lo-fi": "Chill Music Listeners",
    "bedroom pop": "Indie & Alternative Rock Fans",
    "ambient": "Chill Music Listeners",
    "afrobeats": "World Music Fans",
    "amapiano": "World Music Fans",
    "afroswing": "World Music Fans",
    "bollywood": "World Music Fans",
    "punjabi pop": "World Music Fans",
    "indi-pop": "World Music Fans",
    "arabic pop": "World Music Fans",
    "khaleeji": "World Music Fans",
    "reggae": "Reggae & Caribbean Music Fans",
    "dancehall": "Reggae & Caribbean Music Fans",
}

# Core-genre -> Spotify Ads Manager genre-targeting buckets.
SPOTIFY_GENRE = {
    "country": "Country", "country pop": "Country", "texas country": "Country",
    "americana": "Country", "southern rock": "Rock",
    "hip hop": "Hip-Hop", "trap": "Hip-Hop", "southern hip hop": "Hip-Hop",
    "west coast hip hop": "Hip-Hop", "east coast hip hop": "Hip-Hop",
    "drill": "Hip-Hop", "uk drill": "Hip-Hop", "grime": "Hip-Hop",
    "pop rap": "Hip-Hop", "desi hip hop": "Hip-Hop", "phonk": "Hip-Hop",
    "edm": "Electronic/Dance", "house": "Electronic/Dance", "techno": "Electronic/Dance",
    "melodic house": "Electronic/Dance", "uk garage": "Electronic/Dance",
    "hyperpop": "Pop", "pop": "Pop", "dance pop": "Pop", "indie pop": "Indie",
    "k-pop": "K-Pop", "j-pop": "Pop", "city pop": "Pop",
    "latin pop": "Latin", "reggaeton": "Latin", "latin trap": "Latin",
    "latin urbano": "Latin", "regional mexican": "Latin", "corridos tumbados": "Latin",
    "cumbia": "Latin", "funk carioca": "Latin", "sertanejo": "Latin",
    "rock": "Rock", "indie rock": "Indie", "indie folk": "Folk & Acoustic",
    "alternative": "Indie", "pop punk": "Rock", "metal": "Metal", "hard rock": "Rock",
    "r&b": "R&B", "neo-soul": "R&B", "gospel": "Christian",
    "christian contemporary": "Christian", "jazz": "Jazz", "classical": "Classical",
    "film score": "Classical", "lo-fi": "Chill", "bedroom pop": "Indie",
    "ambient": "Chill", "afrobeats": "Afro", "amapiano": "Afro", "afroswing": "Afro",
    "bollywood": "Bollywood", "punjabi pop": "Bollywood", "indi-pop": "Bollywood",
    "arabic pop": "Arab", "khaleeji": "Arab", "arabic hip hop": "Arab",
    "reggae": "Reggae", "dancehall": "Reggae",
}


# Subgenres not in the exact maps resolve to a family by pattern, most
# specific first (so "jazz rap" -> hip hop before jazz, "country rock" ->
# country before rock). Keeps affinity coverage at 100% as segment/geo KBs
# evolve (enforced by eval/rollout_benchmark.py).
FAMILY_PATTERNS: list[tuple[str, str, str]] = [
    (r"hip hop|rap|trap|drill|crunk|boom bap|phonk|grime", "Rap & Hip Hop Fans", "Hip-Hop"),
    (r"reggaeton|cumbia|latin|corridos|mexican|sertanejo|funk carioca|funk brasileiro", "Latino Music Fans", "Latin"),
    (r"country|americana|heartland", "Country Music Fans", "Country"),
    (r"gospel|worship|ccm|christian", "Christian & Gospel Music Fans", "Christian"),
    (r"reggae|dancehall|soca|island", "Reggae & Caribbean Music Fans", "Reggae"),
    (r"afro|amapiano|highlife", "World Music Fans", "Afro"),
    (r"arabic|rai|khaleeji|maqam", "World Music Fans", "Arab"),
    (r"bollywood|punjabi|desi|hindustani|ghazal|indi-pop", "World Music Fans", "Bollywood"),
    (r"k-pop", "Pop Music Fans", "K-Pop"),
    (r"house|techno|edm|electro|rave|bass|big room|acid|minimal|dance", "Electronica & Dance Music Fans", "Electronic/Dance"),
    (r"metal|doom|hardcore", "Metal Fans", "Metal"),
    (r"punk|emo|easycore", "Rock Music Fans", "Rock"),
    (r"indie|alt |alt-|alternative|garage rock|psych|post-punk|art pop|chamber pop|bedroom", "Indie & Alternative Rock Fans", "Indie"),
    (r"rock|blues", "Rock Music Fans", "Rock"),
    (r"r&b|soul|neo-soul", "Rhythm & Blues Fans", "R&B"),
    (r"jazz", "Jazz Fans", "Jazz"),
    (r"classical|orchestral|cinematic|score|neo-classical|epic", "Classical Music Fans", "Classical"),
    (r"folk|acoustic|singer-songwriter", "Folk & Traditional Music Fans", "Folk & Acoustic"),
    (r"lo-fi|chill|ambient|downtempo|easy listening|soft jazz", "Chill Music Listeners", "Chill"),
    (r"oldies|throwback|y2k|2000s|classic pop|nostalgia", "Pop Music Fans", "Pop"),
    (r"pop|synth", "Pop Music Fans", "Pop"),
    (r"world", "World Music Fans", "Pop"),
]


def genre_to_affinity(genre: str) -> str | None:
    """Google/DV360 affinity segment for a genre (exact map, then family)."""
    g = genre.lower().strip()
    if g in GENRE_AFFINITY:
        return GENRE_AFFINITY[g]
    for pattern, affinity, _ in FAMILY_PATTERNS:
        if re.search(pattern, g):
            return affinity
    return None


def genre_to_spotify(genre: str) -> str | None:
    """Spotify Ads Manager genre bucket for a genre (exact map, then family)."""
    g = genre.lower().strip()
    if g in SPOTIFY_GENRE:
        return SPOTIFY_GENRE[g]
    for pattern, _, bucket in FAMILY_PATTERNS:
        if re.search(pattern, g):
            return bucket
    return None


class AdPlatformsKB:
    """Loads and queries the ad-platform capability knowledge base."""

    def __init__(self, path: str | Path = PLATFORMS_PATH) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.platforms: dict[str, dict[str, Any]] = {p["id"]: p for p in data["platforms"]}
        self.meta: dict[str, Any] = data.get("_meta", {})

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls) -> "AdPlatformsKB":
        return cls()

    def get(self, platform_id: str) -> dict[str, Any]:
        if platform_id not in self.platforms:
            raise KeyError(f"Unknown ad platform: {platform_id}")
        return self.platforms[platform_id]

    def list_platforms(self) -> list[dict[str, Any]]:
        return [
            {
                "id": p["id"],
                "label": p["label"],
                "channels": p.get("channels", []),
                "mechanism": p.get("variant_mechanism", {}).get("name", ""),
                "music_targeting": p.get("targeting", {}).get("music_specific", ""),
            }
            for p in self.platforms.values()
        ]


class GeoKB:
    """Loads and queries the geo music-affinity packs."""

    def __init__(self, path: str | Path = GEO_PATH) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.geos: dict[str, dict[str, Any]] = {g["id"]: g for g in data["geos"]}
        self.meta: dict[str, Any] = data.get("_meta", {})

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls) -> "GeoKB":
        return cls()

    def get(self, geo_id: str) -> dict[str, Any]:
        if geo_id not in self.geos:
            raise KeyError(f"Unknown geo pack: {geo_id}")
        return self.geos[geo_id]

    def list_geos(self) -> list[dict[str, Any]]:
        return [
            {
                "id": g["id"],
                "label": g["label"],
                "level": g.get("level", ""),
                "country": g.get("country", ""),
                "genres": g.get("music_bias", {}).get("genre_boosts", []),
                "affinity_segments": g.get("affinity_segments", []),
            }
            for g in self.geos.values()
        ]

    def for_segment(self, segment_id: str) -> list[dict[str, Any]]:
        """Geo packs whose streaming profile matches this demographic."""
        return [g for g in self.geos.values() if segment_id in g.get("affinity_segments", [])]


# --------------------------------------------------------------------------
# Canonical targeting derivation
# --------------------------------------------------------------------------

def parse_age_range(label: str) -> tuple[int, int]:
    """Pull an ad-targetable age window out of a segment label.

    '(16-24)' -> (18, 24) [ad platforms floor at 18 for brand safety],
    '(55+)' -> (55, 65). Taste clusters without an age carry (18, 65).
    """
    m = re.search(r"(\d{2})\s*[-–]\s*(\d{2})", label)
    if m:
        return (max(18, int(m.group(1))), int(m.group(2)))
    m = re.search(r"(\d{2})\s*\+", label)
    if m:
        return (max(18, int(m.group(1))), 65)
    return (18, 65)


def canonical_targeting(segment: dict[str, Any], geo: dict[str, Any] | None) -> dict[str, Any]:
    """Platform-agnostic targeting derived from a segment (+ optional geo pack).

    Only non-sensitive attributes are used: age, geography, and declared
    music/interest affinities — all standard, policy-safe targeting on every
    platform in the KB.
    """
    age_min, age_max = parse_age_range(segment.get("label", ""))
    genres = segment.get("core_genres", [])[:5]
    if geo:
        genres = list(dict.fromkeys(geo.get("music_bias", {}).get("genre_boosts", [])[:3] + genres))[:6]

    affinities = sorted({a for g in genres if (a := genre_to_affinity(g))})
    spotify_genres = sorted({s for g in genres if (s := genre_to_spotify(g))})
    artists = segment.get("example_artists", [])[:3]
    if geo:
        artists = list(dict.fromkeys(geo.get("music_bias", {}).get("example_artists", [])[:2] + artists))[:5]

    return {
        "age_min": age_min,
        "age_max": age_max,
        "genders": "all",
        "geo": (geo or {}).get("platform_geo", {}),
        "geo_id": (geo or {}).get("id"),
        "geo_label": (geo or {}).get("label"),
        "interest_keywords": genres,
        "artist_keywords": artists,
        "google_affinities": affinities,
        "spotify_genres": spotify_genres,
    }


# --------------------------------------------------------------------------
# Per-platform spec shaping
# --------------------------------------------------------------------------

def _meta_spec(canon: dict[str, Any]) -> dict[str, Any]:
    geo = canon["geo"].get("meta", {})
    geo_locations: dict[str, Any] = {}
    if geo.get("countries"):
        geo_locations["countries"] = geo["countries"]
    if geo.get("regions"):
        geo_locations["regions"] = [{"name": r} for r in geo["regions"]]
    if geo.get("cities"):
        geo_locations["cities"] = [{"name": c, "radius": 25, "distance_unit": "mile"} for c in geo["cities"]]
    if geo.get("custom_locations"):
        geo_locations["custom_locations"] = [
            {"name": loc["name"], "radius": loc.get("radius_miles", 15), "distance_unit": "mile"}
            for loc in geo["custom_locations"]
        ]
    if not geo_locations:
        geo_locations = {"countries": ["US"]}
    return {
        "targeting": {
            "age_min": canon["age_min"],
            "age_max": canon["age_max"],
            "geo_locations": geo_locations,
            "flexible_spec": [
                {"interests": [{"name": kw} for kw in (canon["interest_keywords"] + canon["artist_keywords"])[:8]]}
            ],
            "targeting_automation": {"advantage_audience": 0},
        },
        "optimization_goal": "THRUPLAY",
        "billing_event": "IMPRESSIONS",
        "status": "PAUSED",
    }


def _google_spec(canon: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign_type": "DEMAND_GEN",
        "locations": canon["geo"].get("google_ads", []) or ["United States"],
        "age_range": f"{canon['age_min']}-{canon['age_max']}",
        "gender": "ALL",
        "affinity_segments": canon["google_affinities"],
        "custom_segment_keywords": canon["artist_keywords"],
        "status": "PAUSED",
    }


def _dv360_spec(canon: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity": "LineItem + YouTube AdGroup (SDF)",
        "geography_targeting_include": canon["geo"].get("dv360", []) or ["United States"],
        "demographic_targeting_age": f"{canon['age_min']}-{canon['age_max']}",
        "audience_targeting": canon["google_affinities"],
        "device_targeting_include": "Connected TV; Smart Phone",
        "line_item_status": "Draft",
    }


def _tiktok_spec(canon: dict[str, Any]) -> dict[str, Any]:
    return {
        "location": canon["geo"].get("tiktok", []) or ["United States"],
        "age_groups": _tiktok_age_groups(canon["age_min"], canon["age_max"]),
        "gender": "GENDER_UNLIMITED",
        "interest_keywords": canon["interest_keywords"][:5],
        "operation_status": "DISABLE",
    }


def _tiktok_age_groups(age_min: int, age_max: int) -> list[str]:
    brackets = [(18, 24, "AGE_18_24"), (25, 34, "AGE_25_34"), (35, 44, "AGE_35_44"),
                (45, 54, "AGE_45_54"), (55, 100, "AGE_55_100")]
    return [code for lo, hi, code in brackets if lo <= age_max and hi >= age_min]


def _spotify_spec(canon: dict[str, Any]) -> dict[str, Any]:
    return {
        "locations": canon["geo"].get("spotify", []) or ["United States"],
        "age_range": f"{canon['age_min']}-{canon['age_max']}",
        "gender": "All",
        # Spotify allows ONE advanced targeting option per ad set; streamed
        # genre is our strongest lever (ad music == listener's session music).
        "advanced_targeting": {"type": "genre", "values": canon["spotify_genres"] or ["Pop"]},
        "creative_note": "Audio-only cut, <=30s, 44.1kHz, -16 LUFS / -2.0 dBTP (exported to spotify/audio/)",
    }


_SPEC_BUILDERS = {
    "meta": _meta_spec,
    "google_ads": _google_spec,
    "dv360": _dv360_spec,
    "tiktok": _tiktok_spec,
    "spotify": _spotify_spec,
}


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------

def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper()[:24] or "X"


class RolloutPlanner:
    def __init__(
        self,
        platforms: AdPlatformsKB | None = None,
        geos: GeoKB | None = None,
        demographics: DemographicsKB | None = None,
    ) -> None:
        self.platforms = platforms or AdPlatformsKB.default()
        self.geos = geos or GeoKB.default()
        self.demographics = demographics or DemographicsKB.default()

    def plan(
        self,
        manifest: dict[str, Any],
        platform_ids: list[str],
        geo_ids: list[str] | None = None,
        total_budget: float = DEFAULT_BUDGET,
        currency: str = "USD",
    ) -> dict[str, Any]:
        brand = manifest.get("brand") or "BRAND"
        variants = manifest.get("variants", [])
        if not variants:
            raise ValueError("Manifest has no variants to roll out.")
        platform_ids = [p for p in platform_ids if p in self.platforms.platforms]
        if not platform_ids:
            raise ValueError("No valid platforms selected.")

        ad_units: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []

        for variant in variants:
            segment_id = variant.get("segment_id", "personal")
            segment = self.demographics.segments.get(segment_id, {"label": variant.get("segment_label", "Personal"), "core_genres": variant.get("direction", {}).get("genres", [])})
            variant_geos = self._geos_for_variant(variant, segment_id, geo_ids)

            for geo in variant_geos:
                canon = canonical_targeting(segment, geo)
                for pid in platform_ids:
                    # A geo pack with no keys for a platform means that market
                    # can't be bought there (e.g. TikTok is banned in India).
                    if geo is not None and not canon["geo"].get(pid):
                        skipped.append({
                            "variant": variant.get("label", ""),
                            "platform": pid,
                            "geo": geo["label"],
                            "reason": f"{geo['label']} is not addressable on {self.platforms.get(pid)['label']}",
                        })
                        continue
                    ad_units.append(self._ad_unit(brand, variant, segment, geo, canon, pid))

        if not ad_units:
            raise ValueError("No ad units could be planned — every platform/geo pair was skipped.")

        self._allocate_budget(ad_units, total_budget)
        plan = {
            "campaign_id": manifest.get("id", ""),
            "brand": brand,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platforms": platform_ids,
            "geo_ids": geo_ids or [],
            "total_budget": total_budget,
            "currency": currency,
            "ad_units": ad_units,
            "skipped": skipped,
            "summary": {
                "variants": len(variants),
                "ad_units": len(ad_units),
                "platforms": len(platform_ids),
                "geos": len({u["geo_id"] for u in ad_units if u["geo_id"]}) or 1,
                "skipped": len(skipped),
            },
        }
        plan["issues"] = validate_plan(plan)
        return plan

    def _geos_for_variant(
        self, variant: dict[str, Any], segment_id: str, geo_ids: list[str] | None
    ) -> list[dict[str, Any] | None]:
        # A geo baked into the variant at render time (geo-flavored music) wins.
        baked = variant.get("geo_id")
        if baked and baked in self.geos.geos:
            return [self.geos.get(baked)]
        if geo_ids:
            chosen = [self.geos.get(g) for g in geo_ids if g in self.geos.geos]
            # Prefer geos whose streaming profile matches the segment; fall
            # back to everything the user picked.
            matched = [g for g in chosen if segment_id in g.get("affinity_segments", [])]
            return matched or chosen or [None]
        # No geo requested: suggest the top curated matches, else untargeted.
        curated = self.geos.for_segment(segment_id)[:2]
        return curated or [None]

    def _ad_unit(
        self,
        brand: str,
        variant: dict[str, Any],
        segment: dict[str, Any],
        geo: dict[str, Any] | None,
        canon: dict[str, Any],
        platform_id: str,
    ) -> dict[str, Any]:
        platform = self.platforms.get(platform_id)
        geo_part = _slug(geo["id"]) if geo else "ALL"
        name = f"SS_{_slug(brand)}_{_slug(variant.get('segment_id', 'PERSONAL'))}_{_slug(variant.get('mood', ''))}_{geo_part}_{_slug(platform_id)}"

        why = [
            variant.get("direction", {}).get("rationale", ""),
            (geo or {}).get("rationale", ""),
            platform.get("variant_mechanism", {}).get("how", "").split(".")[0] + ".",
        ]
        return {
            "name": name,
            "platform": platform_id,
            "platform_label": platform["label"],
            "variant_file": variant.get("file", ""),
            "variant_label": variant.get("label", ""),
            "segment_id": variant.get("segment_id", ""),
            "segment_label": variant.get("segment_label", ""),
            "mood": variant.get("mood", ""),
            "geo_id": canon["geo_id"],
            "geo_label": canon["geo_label"] or "No geo refinement",
            "track": variant.get("track", {}),
            "match_score": variant.get("match", {}).get("total", 0.5),
            "canonical_targeting": canon,
            "platform_spec": _SPEC_BUILDERS[platform_id](canon),
            "export_format": platform.get("export_format", ""),
            "why": " ".join(w for w in why if w).strip(),
            "demo_only": bool(variant.get("track", {}).get("demo_only")),
        }

    @staticmethod
    def _allocate_budget(ad_units: list[dict[str, Any]], total: float) -> None:
        """Split budget proportionally to each cut's music-match score."""
        weights = [max(0.05, float(u.get("match_score") or 0.5)) for u in ad_units]
        wsum = sum(weights)
        for unit, w in zip(ad_units, weights):
            unit["budget"] = round(total * w / wsum, 2)
        # Absorb rounding drift into the largest allocation.
        drift = round(total - sum(u["budget"] for u in ad_units), 2)
        if ad_units and abs(drift) >= 0.01:
            top = max(ad_units, key=lambda u: u["budget"])
            top["budget"] = round(top["budget"] + drift, 2)


# --------------------------------------------------------------------------
# Validation (shared by tests, benchmark, and the web layer)
# --------------------------------------------------------------------------

def validate_plan(plan: dict[str, Any]) -> list[str]:
    """Enterprise gate: every ad unit must be complete enough to traffic."""
    issues: list[str] = []
    names = set()
    for unit in plan.get("ad_units", []):
        label = unit.get("name", "<unnamed>")
        if unit["name"] in names:
            issues.append(f"{label}: duplicate ad unit name")
        names.add(unit["name"])
        if not unit.get("variant_file"):
            issues.append(f"{label}: no creative file attached")
        if unit.get("budget", 0) <= 0:
            issues.append(f"{label}: budget must be positive")
        canon = unit.get("canonical_targeting", {})
        if not (18 <= canon.get("age_min", 0) <= canon.get("age_max", 0) <= 100):
            issues.append(f"{label}: invalid age window {canon.get('age_min')}-{canon.get('age_max')}")
        spec = unit.get("platform_spec", {})
        if unit["platform"] == "meta":
            if not spec.get("targeting", {}).get("geo_locations"):
                issues.append(f"{label}: Meta ad set has no geo_locations")
            if spec.get("status") != "PAUSED":
                issues.append(f"{label}: Meta entities must be created PAUSED")
        if unit["platform"] == "spotify" and not spec.get("advanced_targeting", {}).get("values"):
            issues.append(f"{label}: Spotify ad set has no genre targeting values")
        if unit["platform"] == "tiktok" and not spec.get("age_groups"):
            issues.append(f"{label}: TikTok ad group resolved no age brackets")
        if unit.get("demo_only"):
            issues.append(f"{label}: DEMO-ONLY track — cannot be trafficked; render with a cleared source first")
    total = plan.get("total_budget", 0)
    allocated = round(sum(u.get("budget", 0) for u in plan.get("ad_units", [])), 2)
    if abs(allocated - total) > 0.05:
        issues.append(f"Budget drift: allocated {allocated} vs requested {total}")
    return issues
