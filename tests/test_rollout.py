"""Rollout planner: KB integrity, targeting derivation, and plan quality."""

import pytest

from sonic_segments.intelligence.demographics import DemographicsKB
from sonic_segments.intelligence.rollout import (
    AdPlatformsKB,
    GeoKB,
    RolloutPlanner,
    canonical_targeting,
    parse_age_range,
    validate_plan,
)

ALL_PLATFORMS = ["meta", "google_ads", "dv360", "tiktok", "spotify"]


# ---------------------------------------------------------------------------
# Knowledge bases
# ---------------------------------------------------------------------------

def test_platform_kb_complete():
    kb = AdPlatformsKB.default()
    assert set(kb.platforms) == set(ALL_PLATFORMS)
    for p in kb.platforms.values():
        for key in ("variant_mechanism", "targeting", "creative_specs", "delivery", "export_format", "sources"):
            assert key in p, f"{p['id']} missing {key}"
        assert p["sources"], f"{p['id']} has no cited sources"
    assert kb.meta.get("sources"), "platform KB missing research citations"


def test_geo_kb_complete():
    kb = GeoKB.default()
    demo = DemographicsKB.default()
    assert len(kb.geos) >= 12
    for g in kb.geos.values():
        assert g.get("platform_geo"), f"{g['id']} has no platform geo keys"
        assert g.get("music_bias", {}).get("genre_boosts"), f"{g['id']} has no genre boosts"
        assert g.get("rationale") and g.get("sources"), f"{g['id']} is not research-grounded"
        for seg in g.get("affinity_segments", []):
            assert seg in demo.segments, f"{g['id']} references unknown segment {seg}"


def test_every_segment_has_curated_geo_or_parses():
    """Every demographic must be plannable: age parses and geo resolution works."""
    demo = DemographicsKB.default()
    geo = GeoKB.default()
    for seg_id, seg in demo.segments.items():
        age_min, age_max = parse_age_range(seg["label"])
        assert 18 <= age_min <= age_max <= 100, f"{seg_id}: bad age window"
        # for_segment may be empty (planner falls back to untargeted) but must not raise
        geo.for_segment(seg_id)


# ---------------------------------------------------------------------------
# Targeting derivation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("Gen-Z (16-24)", (18, 24)),          # floored to 18 for ad platforms
    ("Boomers (55+)", (55, 65)),
    ("Indie & Alt Tribe", (18, 65)),      # taste cluster, no age
    ("Luxury / Premium (35+)", (35, 65)),
])
def test_parse_age_range(label, expected):
    assert parse_age_range(label) == expected


def test_canonical_targeting_merges_geo():
    demo = DemographicsKB.default()
    geo = GeoKB.default().get("nashville_dma")
    canon = canonical_targeting(demo.get("country_heartland"), geo)
    assert canon["geo_id"] == "nashville_dma"
    assert "country" in canon["interest_keywords"]
    assert "Country Music Fans" in canon["google_affinities"]
    assert "Country" in canon["spotify_genres"]


def test_canonical_targeting_without_geo():
    demo = DemographicsKB.default()
    canon = canonical_targeting(demo.get("gen_z"), None)
    assert canon["geo"] == {} and canon["geo_id"] is None
    assert canon["google_affinities"], "genre->affinity map must cover gen_z core genres"


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def test_plan_full_matrix(sample_manifest):
    planner = RolloutPlanner()
    plan = planner.plan(sample_manifest, ALL_PLATFORMS, geo_ids=["atlanta_dma", "nashville_dma"], total_budget=500)
    assert plan["summary"]["ad_units"] > 0
    assert not plan["issues"], f"clean manifest must produce a clean plan: {plan['issues']}"
    # Affinity filter: gen_z maps to atlanta, country_heartland to nashville
    gen_z_geos = {u["geo_id"] for u in plan["ad_units"] if u["segment_id"] == "gen_z"}
    heartland_geos = {u["geo_id"] for u in plan["ad_units"] if u["segment_id"] == "country_heartland"}
    assert gen_z_geos == {"atlanta_dma"}
    assert heartland_geos == {"nashville_dma"}
    # Budget conservation
    assert abs(sum(u["budget"] for u in plan["ad_units"]) - 500) < 0.05
    # Higher match score -> more budget (same unit count per variant here)
    by_seg = {}
    for u in plan["ad_units"]:
        by_seg.setdefault(u["segment_id"], []).append(u["budget"])
    assert sum(by_seg["gen_z"]) > sum(by_seg["country_heartland"])


def test_plan_skips_unaddressable_markets(sample_manifest):
    """India geo pack has no TikTok keys (banned) -> planner must skip, not emit."""
    sample_manifest["variants"][0]["segment_id"] = "india_youth"
    sample_manifest["variants"][0]["segment_label"] = "India Youth (18-30)"
    plan = RolloutPlanner().plan(sample_manifest, ["tiktok", "meta"], geo_ids=["india_metro"])
    tiktok_india = [u for u in plan["ad_units"] if u["geo_id"] == "india_metro" and u["platform"] == "tiktok"]
    assert not tiktok_india
    assert any(s["geo"] == "India (metro north + west)" for s in plan["skipped"])
    # Meta can buy India, so those units must exist
    assert any(u["geo_id"] == "india_metro" and u["platform"] == "meta" for u in plan["ad_units"])
    # All-skipped plans must raise, not emit an empty rollout
    with pytest.raises(ValueError):
        RolloutPlanner().plan(
            {**sample_manifest, "variants": [sample_manifest["variants"][0]]},
            ["tiktok"], geo_ids=["india_metro"],
        )


def test_plan_names_unique_and_traffickable(sample_manifest):
    plan = RolloutPlanner().plan(sample_manifest, ALL_PLATFORMS, geo_ids=["la_metro"])
    names = [u["name"] for u in plan["ad_units"]]
    assert len(names) == len(set(names))
    for name in names:
        assert name.startswith("SS_GATORADE_")
        assert " " not in name


def test_demo_only_track_blocks_trafficking(sample_manifest):
    sample_manifest["variants"][0]["track"]["demo_only"] = True
    plan = RolloutPlanner().plan(sample_manifest, ["meta"])
    assert any("DEMO-ONLY" in i for i in plan["issues"])


def test_validate_plan_catches_corruption(sample_manifest):
    plan = RolloutPlanner().plan(sample_manifest, ["meta"])
    plan["ad_units"][0]["budget"] = -5
    plan["ad_units"][0]["platform_spec"]["status"] = "ACTIVE"
    issues = validate_plan(plan)
    assert any("budget" in i for i in issues)
    assert any("PAUSED" in i for i in issues)


def test_geo_tuned_direction_changes_the_brief():
    """Same segment+mood must produce a different music brief per market."""
    demo = DemographicsKB.default()
    geo_kb = GeoKB.default()
    base = demo.direction("gen_z", "hype")
    atlanta = demo.direction("gen_z", "hype", geo=geo_kb.get("atlanta_dma"))
    london = demo.direction("gen_z", "hype", geo=geo_kb.get("uk_london"))
    assert atlanta.geo_id == "atlanta_dma" and base.geo_id == ""
    assert atlanta.genres != london.genres
    assert atlanta.search_terms[0] != base.search_terms[0]
    assert "uk drill" in london.genres
    assert "Geo-tuned" in atlanta.rationale
    # geo fields survive the manifest round-trip
    assert atlanta.as_dict()["geo_label"].startswith("Atlanta")


def test_plan_rejects_empty_inputs(sample_manifest):
    with pytest.raises(ValueError):
        RolloutPlanner().plan({"variants": []}, ["meta"])
    with pytest.raises(ValueError):
        RolloutPlanner().plan(sample_manifest, ["not_a_platform"])
