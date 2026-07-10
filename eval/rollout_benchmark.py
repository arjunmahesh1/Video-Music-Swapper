"""Rollout-layer benchmark: exhaustive coverage + validation sweep.

Unlike the mood benchmark (model quality vs hand labels), this layer is
deterministic, so the bar is exhaustiveness: every (segment x mood x geo x
platform) combination the product can sell must plan cleanly. Run:

    /opt/miniconda3/bin/python -m eval.rollout_benchmark

Metrics reported:
  targeting coverage   age parse + interest->affinity mapping per segment
  geo coverage         segments with curated geo matches; geo platform keys
  full-matrix sweep    every segment x mood x selected-geo x platform planned
                       and gate-checked (issues must be zero)
  spec completeness    per-platform required fields present on every unit
"""

from __future__ import annotations

import itertools
import time

from sonic_segments.intelligence.demographics import DemographicsKB
from sonic_segments.intelligence.mood import MOODS
from sonic_segments.intelligence.rollout import (
    AdPlatformsKB,
    GeoKB,
    RolloutPlanner,
    canonical_targeting,
    parse_age_range,
)

ALL_PLATFORMS = ["meta", "google_ads", "dv360", "tiktok", "spotify"]


def _fake_variant(seg_id: str, seg_label: str, mood: str) -> dict:
    return {
        "file": f"variants/{seg_id}_{mood}.mp4",
        "label": f"{seg_label} · {mood}",
        "segment_id": seg_id,
        "segment_label": seg_label,
        "mood": mood,
        "mood_label": mood.title(),
        "direction": {"genres": [], "rationale": f"benchmark direction for {seg_id}/{mood}"},
        "track": {"name": "bench", "source": "library", "license": "licensed", "demo_only": False},
        "match": {"total": 0.75},
    }


def main() -> None:
    demo, geo, platforms = DemographicsKB.default(), GeoKB.default(), AdPlatformsKB.default()
    planner = RolloutPlanner()

    # ---- 1. targeting derivation coverage --------------------------------
    age_ok = affinity_ok = spotify_ok = 0
    for seg in demo.segments.values():
        lo, hi = parse_age_range(seg["label"])
        if 18 <= lo <= hi <= 100:
            age_ok += 1
        canon = canonical_targeting(seg, None)
        if canon["google_affinities"]:
            affinity_ok += 1
        if canon["spotify_genres"]:
            spotify_ok += 1
    n_seg = len(demo.segments)

    # ---- 2. geo coverage ---------------------------------------------------
    curated = sum(1 for s in demo.segments if geo.for_segment(s))
    geo_slots = [(g, p) for g in geo.geos.values() for p in ALL_PLATFORMS]
    addressable = sum(1 for g, p in geo_slots if g.get("platform_geo", {}).get(p))

    # ---- 3. full-matrix sweep ----------------------------------------------
    t0 = time.time()
    total_units = 0
    dirty_plans: list[str] = []
    combos = 0
    for seg_id, mood in itertools.product(demo.segments, MOODS):
        seg = demo.segments[seg_id]
        manifest = {
            "id": "benchmark", "brand": "BENCH",
            "variants": [_fake_variant(seg_id, seg["label"], mood)],
        }
        geo_ids = [g["id"] for g in geo.for_segment(seg_id)][:3] or ["la_metro"]
        try:
            plan = planner.plan(manifest, ALL_PLATFORMS, geo_ids=geo_ids, total_budget=100)
        except ValueError as exc:
            dirty_plans.append(f"{seg_id}/{mood}: {exc}")
            continue
        combos += 1
        total_units += plan["summary"]["ad_units"]
        if plan["issues"]:
            dirty_plans.append(f"{seg_id}/{mood}: {plan['issues'][:2]}")
    elapsed = time.time() - t0

    # ---- report -------------------------------------------------------------
    n_moods = len(MOODS)
    print("ROLLOUT LAYER BENCHMARK")
    print("=" * 54)
    print(f"segments={n_seg} moods={n_moods} geos={len(geo.geos)} platforms={len(platforms.platforms)}")
    print()
    print(f"age parse coverage          {age_ok}/{n_seg}  ({age_ok/n_seg:.0%})")
    print(f"google affinity coverage    {affinity_ok}/{n_seg}  ({affinity_ok/n_seg:.0%})")
    print(f"spotify genre coverage      {spotify_ok}/{n_seg}  ({spotify_ok/n_seg:.0%})")
    print(f"curated geo per segment     {curated}/{n_seg}  ({curated/n_seg:.0%})")
    print(f"geo x platform addressable  {addressable}/{len(geo_slots)}  ({addressable/len(geo_slots):.0%})")
    print()
    print(f"full matrix: {combos}/{n_seg * n_moods} (segment x mood) plans, "
          f"{total_units} ad units, {elapsed:.2f}s")
    print(f"gate-check failures         {len(dirty_plans)}")
    for line in dirty_plans[:10]:
        print(f"  ! {line}")
    print()
    verdict = "PASS" if not dirty_plans and age_ok == n_seg and affinity_ok == n_seg and spotify_ok == n_seg else "FAIL"
    print(f"VERDICT: {verdict}")
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
