"""Rollout export packager: one plan -> platform-ready trafficking bundle.

Writes rollout/ inside the campaign's job directory:

  rollout_plan.json          machine-readable plan (publish CLI input)
  rollout_plan.md            brand-manager brief: what runs where and why
  meta/adsets.json           Marketing API-ready ad set payloads
  google_ads/demand_gen.csv  Google Ads Editor-style bulk sheet
  dv360/sdf_line_items.csv   SDF-flavored line-item sheet (CTV/YouTube TV path)
  tiktok/adgroups.json       TikTok Marketing API ad group payloads
  spotify/adsets.json        Spotify Ads Manager briefs
  spotify/audio/*.mp3        audio-only mixes of each cut at Spotify delivery
                             spec (44.1kHz, -16 LUFS / -2 dBTP)
  assets/                    each variant cut copied under its trafficking name

then zips everything to rollout_bundle.zip. All file shaping is offline;
the only external tool used is ffmpeg for the Spotify audio extraction.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

SPOTIFY_AUDIO_ARGS = [
    "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k",
    "-af", "loudnorm=I=-16:TP=-2.0:LRA=11",
]


def build_rollout_bundle(job_dir: Path, manifest: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    job_dir = Path(job_dir)
    out = job_dir / "rollout"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    (out / "rollout_plan.json").write_text(json.dumps(plan, indent=2, default=str))
    _write_markdown(out / "rollout_plan.md", manifest, plan)

    by_platform: dict[str, list[dict[str, Any]]] = {}
    for unit in plan["ad_units"]:
        by_platform.setdefault(unit["platform"], []).append(unit)

    if "meta" in by_platform:
        _write_meta(out / "meta", by_platform["meta"])
    if "google_ads" in by_platform:
        _write_google(out / "google_ads", by_platform["google_ads"])
    if "dv360" in by_platform:
        _write_dv360(out / "dv360", by_platform["dv360"])
    if "tiktok" in by_platform:
        _write_tiktok(out / "tiktok", by_platform["tiktok"])
    spotify_audio: list[str] = []
    if "spotify" in by_platform:
        spotify_audio = _write_spotify(out / "spotify", by_platform["spotify"], job_dir)

    assets = _copy_assets(out / "assets", plan, job_dir)

    zip_path = job_dir / "rollout_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(job_dir))

    return {
        "bundle": "rollout_bundle.zip",
        "plan": "rollout/rollout_plan.json",
        "brief": "rollout/rollout_plan.md",
        "assets": assets,
        "spotify_audio": spotify_audio,
        "files": sorted(str(p.relative_to(job_dir)) for p in out.rglob("*") if p.is_file()),
    }


# ---------------------------------------------------------------------------
# Platform writers
# ---------------------------------------------------------------------------

def _write_meta(dir_: Path, units: list[dict[str, Any]]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    payloads = []
    for u in units:
        spec = u["platform_spec"]
        payloads.append({
            "adset": {
                "name": u["name"],
                "daily_budget_cents": int(round(u["budget"] * 100)),
                **spec,
            },
            "ad": {
                "name": f"{u['name']}_AD",
                "creative_video": f"assets/{Path(u['variant_file']).name}",
                "status": "PAUSED",
            },
            "notes": u["why"],
        })
    (dir_ / "adsets.json").write_text(json.dumps({
        "how_to_launch": (
            "Per ad set: 1) POST /act_{AD_ACCOUNT_ID}/advideos (upload the creative from assets/); "
            "2) POST /act_{AD_ACCOUNT_ID}/adsets with the 'adset' body under an existing campaign; "
            "3) POST /act_{AD_ACCOUNT_ID}/adcreatives referencing the video id; "
            "4) POST /act_{AD_ACCOUNT_ID}/ads tying creative to ad set. "
            "Everything is status=PAUSED — nothing spends until a human flips it on. "
            "Dry-run first: python -m sonic_segments.publish --job <id> --platform meta"
        ),
        "ad_sets": payloads,
    }, indent=2))


GOOGLE_COLUMNS = [
    "Campaign", "Ad group", "Status", "Budget", "Locations", "Age range", "Gender",
    "Affinity segments", "Custom segment keywords", "Video file", "Why this cut",
]


def _write_google(dir_: Path, units: list[dict[str, Any]]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    with open(dir_ / "demand_gen.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(GOOGLE_COLUMNS)
        for u in units:
            spec = u["platform_spec"]
            writer.writerow([
                f"SS_{u['segment_id']}_DEMANDGEN", u["name"], "Paused", f"{u['budget']:.2f}",
                "; ".join(spec["locations"]), spec["age_range"], spec["gender"],
                "; ".join(spec["affinity_segments"]), "; ".join(spec["custom_segment_keywords"]),
                f"assets/{Path(u['variant_file']).name}", u["why"],
            ])


SDF_COLUMNS = [
    "Line Item Name", "Type", "Status", "Budget Amount", "Geography Targeting - Include",
    "Demographic Targeting - Age", "Audience Targeting - Include", "Device Targeting - Include",
    "Creative Asset", "Notes",
]


def _write_dv360(dir_: Path, units: list[dict[str, Any]]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    with open(dir_ / "sdf_line_items.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(SDF_COLUMNS)
        for u in units:
            spec = u["platform_spec"]
            writer.writerow([
                u["name"], "Video", spec["line_item_status"], f"{u['budget']:.2f}",
                "; ".join(spec["geography_targeting_include"]),
                spec["demographic_targeting_age"],
                "; ".join(spec["audience_targeting"]),
                spec["device_targeting_include"],
                f"assets/{Path(u['variant_file']).name}", u["why"],
            ])


def _write_tiktok(dir_: Path, units: list[dict[str, Any]]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "adgroups.json").write_text(json.dumps({
        "how_to_launch": (
            "POST /open_api/v1.3/adgroup/create/ per entry (operation_status=DISABLE), "
            "then /ad/create/ with the video from assets/. Nothing serves until enabled."
        ),
        "ad_groups": [
            {
                "adgroup_name": u["name"],
                "budget": u["budget"],
                "budget_mode": "BUDGET_MODE_DAY",
                **u["platform_spec"],
                "creative_video": f"assets/{Path(u['variant_file']).name}",
                "notes": u["why"],
            }
            for u in units
        ],
    }, indent=2))


def _write_spotify(dir_: Path, units: list[dict[str, Any]], job_dir: Path) -> list[str]:
    dir_.mkdir(parents=True, exist_ok=True)
    audio_dir = dir_ / "audio"
    audio_dir.mkdir(exist_ok=True)

    exported: list[str] = []
    seen: set[str] = set()
    for u in units:
        src = job_dir / u["variant_file"]
        name = Path(u["variant_file"]).stem + ".mp3"
        if name in seen or not src.exists():
            continue
        seen.add(name)
        target = audio_dir / name
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), *SPOTIFY_AUDIO_ARGS, str(target)],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode == 0:
            exported.append(f"rollout/spotify/audio/{name}")

    (dir_ / "adsets.json").write_text(json.dumps({
        "how_to_launch": (
            "Spotify Ads Manager is self-serve today (API in closed beta): create one ad set per "
            "entry, pick the genre under Advanced targeting, upload the matching file from audio/. "
            "Audio is pre-mastered to Spotify delivery spec (44.1kHz stereo, -16 LUFS, -2.0 dBTP)."
        ),
        "ad_sets": [
            {
                "name": u["name"],
                "budget": u["budget"],
                **u["platform_spec"],
                "audio_asset": f"audio/{Path(u['variant_file']).stem}.mp3",
                "notes": u["why"],
            }
            for u in units
        ],
    }, indent=2))
    return exported


def _copy_assets(dir_: Path, plan: dict[str, Any], job_dir: Path) -> list[str]:
    dir_.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    seen: set[str] = set()
    for u in plan["ad_units"]:
        rel = u.get("variant_file", "")
        src = job_dir / rel
        name = Path(rel).name
        if not rel or name in seen or not src.exists():
            continue
        seen.add(name)
        shutil.copyfile(src, dir_ / name)
        copied.append(f"rollout/assets/{name}")
    return copied


# ---------------------------------------------------------------------------
# The brand-manager brief
# ---------------------------------------------------------------------------

def _write_markdown(path: Path, manifest: dict[str, Any], plan: dict[str, Any]) -> None:
    lines: list[str] = []
    add = lines.append
    brand = plan.get("brand") or "your brand"
    s = plan["summary"]

    add(f"# {brand} — Sonic Segments rollout plan")
    add("")
    add(f"Generated {plan['generated_at']} · campaign `{plan['campaign_id']}`")
    add("")
    add("## The pitch in one paragraph")
    add("")
    add(
        f"You shipped **one ad**. This plan ships **{s['variants']} re-scored cuts** of it as "
        f"**{s['ad_units']} targeted ad units** across **{s['platforms']} platform(s)** and "
        f"**{s['geos']} market(s)** — same footage, same voiceover, but the music each audience "
        "hears is the music that audience actually streams. Music is the half of the ad experience "
        "brands never A/B test; every unit below is a measurable experiment (each ad set / ad group "
        "isolates one music-per-audience hypothesis), delivered entirely inside the targeting "
        "controls Meta, Google, TikTok and Spotify already give you. No new ad tech required."
    )
    add("")
    if plan.get("issues"):
        add("## ⚠ Gate check — resolve before trafficking")
        add("")
        for issue in plan["issues"]:
            add(f"- {issue}")
        add("")
    add("## What runs where")
    add("")
    add("| Ad unit | Platform | Audience | Market | Music | Budget |")
    add("|---|---|---|---|---|---|")
    for u in plan["ad_units"]:
        track = u.get("track", {})
        track_name = track.get("name", "")
        if track.get("demo_only"):
            track_name += " (DEMO ONLY)"
        add(
            f"| `{u['name']}` | {u['platform_label']} | {u['segment_label']} ({u['mood']}) "
            f"| {u['geo_label']} | {track_name} | {plan['currency']} {u['budget']:.2f} |"
        )
    add("")
    if plan.get("skipped"):
        add("### Skipped combinations")
        add("")
        for skip in plan["skipped"]:
            add(f"- {skip['variant']} × {skip['platform']} × {skip['geo']}: {skip['reason']}")
        add("")
    add("## Why each unit exists")
    add("")
    for u in plan["ad_units"]:
        add(f"### `{u['name']}`")
        add("")
        add(f"- **Cut:** {u['variant_label']} → `{u['variant_file']}`")
        add(f"- **Targeting:** ages {u['canonical_targeting']['age_min']}–{u['canonical_targeting']['age_max']}, "
            f"{u['geo_label']}; interests: {', '.join(u['canonical_targeting']['interest_keywords'][:4])}")
        add(f"- **Why:** {u['why']}")
        add("")
    add("## How to launch (safely)")
    add("")
    add("1. Everything in this bundle is generated **paused/draft** — nothing spends on upload.")
    add("2. `meta/adsets.json`, `tiktok/adgroups.json`: ready-to-POST API payloads "
        "(`python -m sonic_segments.publish --job <id> --platform meta` prints the exact request "
        "sequence without sending anything).")
    add("3. `google_ads/demand_gen.csv`: import via Google Ads Editor, review, then enable.")
    add("4. `dv360/sdf_line_items.csv`: the CTV / YouTube TV path — hand to your DV360 trader for SDF upload.")
    add("5. `spotify/`: one ad set per genre-targeted brief; audio pre-mastered to Spotify spec.")
    add("6. Measure: each unit isolates one music hypothesis — compare CTR/thruplay/completion "
        "per unit against your control cut to prove the lift, then scale budget to winners.")
    add("")
    add("---")
    add("*Sonic Segments — audio is 50% of the ad experience. Give it more than 0% of the optimization.*")
    path.write_text("\n".join(lines))
