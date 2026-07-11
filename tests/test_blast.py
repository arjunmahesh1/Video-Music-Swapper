"""Audience tree + blast planner: funnel integrity and max-reach planning."""

import json

import pytest

from sonic_segments.intelligence.audience_tree import (
    SOUND_LIBRARY_PLATFORMS,
    AudienceTree,
    BlastPlanner,
)
from sonic_segments.intelligence.demographics import DemographicsKB
from sonic_segments.intelligence.rollout import GeoKB
from sonic_segments.pipeline.export import build_rollout_bundle


@pytest.fixture()
def tree():
    return AudienceTree.default()


@pytest.fixture()
def planner():
    return BlastPlanner()


FULL_US = lambda tree: {"us_states": list(tree.us_states), "intl": [],
                        "age_bands": [], "tastes": [], "moods": ["hype"]}


# ---------------------------------------------------------------------------
# Tree integrity: the whole US must be addressable
# ---------------------------------------------------------------------------

def test_every_state_covered_and_mapped(tree):
    geo = GeoKB.default()
    assert len(tree.us_states) == 51  # 50 states + DC
    tiles = set()
    for code, st in tree.us_states.items():
        assert st["region"] in geo.geos, f"{code} maps to unknown region {st['region']}"
        assert st["pop_m"] > 0
        pos = tuple(st["tile"])
        assert pos not in tiles, f"tile collision at {pos}"
        tiles.add(pos)
    # census sanity: US total within 10% of ~331M
    assert abs(sum(s["pop_m"] for s in tree.us_states.values()) - 331) < 33


def test_tastes_and_bands_valid(tree):
    demo = DemographicsKB.default()
    for t in tree.taste_clusters:
        assert t in demo.segments
    lo = min(b["range"][0] for b in tree.age_bands.values())
    hi = max(b["range"][1] for b in tree.age_bands.values())
    assert (lo, hi) == (18, 65), "age bands must tile the targetable 18-65 universe"


def test_funnel_order_is_geo_first(tree):
    assert [f["level"] for f in tree.funnel] == ["geo", "age", "taste"]


# ---------------------------------------------------------------------------
# Expansion + dedupe
# ---------------------------------------------------------------------------

def test_expand_counts_and_dedupe(tree):
    sel = {"us_states": ["GA", "TN", "NY"], "intl": ["uk_london"],
           "age_bands": ["18_24", "25_34"], "tastes": ["urban_hiphop", "country_heartland"],
           "moods": ["hype"]}
    leaves = tree.expand(sel)
    assert len(leaves) == (3 + 1) * 2 * 2  # markets x bands x tastes
    briefs = tree.unique_briefs(leaves)
    # GA/TN/NY are 3 different regions + uk => 4 geo packs x 2 tastes
    assert len(briefs) == 4 * 2
    assert all(len(e["leaves"]) >= 1 for e in briefs.values())


def test_expand_requires_markets(tree):
    with pytest.raises(ValueError):
        tree.expand({"us_states": [], "intl": []})


def test_taste_leads_geo_flavors(tree):
    """A hip-hop leaf in the heartland stays hip-hop with southern flavor."""
    leaves = tree.expand({"us_states": ["TN"], "intl": [], "age_bands": ["18_24"],
                          "tastes": ["urban_hiphop"], "moods": ["hype"]})
    direction = next(iter(tree.unique_briefs(leaves).values()))["direction"]
    hiphop_family = {"hip hop", "trap", "drill", "southern hip hop", "phonk"}
    assert direction.genres[0] in hiphop_family, direction.genres
    assert "country" in direction.genres, "market flavor must still appear"


# ---------------------------------------------------------------------------
# Blast planning
# ---------------------------------------------------------------------------

def test_full_us_blast_rendered(planner, tree, sample_manifest):
    plan = planner.plan(sample_manifest, FULL_US(tree), ["meta", "google_ads"],
                        music_mode="rendered", total_budget=5000)
    s = plan["summary"]
    assert s["leaves"] == 51 * 5 * 18
    assert s["unique_briefs"] == 10 * 18   # regions x tastes
    assert s["ad_units"] == s["unique_briefs"] * 5 * 2
    assert 200 < s["addressable_pop_m"] < 300  # ~255M people 18-65, not leaf-inflated
    assert abs(sum(u["budget"] for u in plan["ad_units"]) - 5000) < 0.05
    assert not plan["issues"]
    # aggregated geo: heartland ad set carries all 8 heartland states
    unit = next(u for u in plan["ad_units"] if u["geo_id"] == "us_south_heartland")
    assert len(unit["platform_spec"]["targeting"]["geo_locations"]["regions"]) == 8 \
        or len(unit["platform_spec"]["locations"]) == 8


def test_platform_sound_mode(planner, tree, sample_manifest):
    plan = planner.plan(sample_manifest, FULL_US(tree),
                        ["tiktok", "meta", "dv360", "spotify"],
                        music_mode="platform_sound", total_budget=2000)
    assert set(plan["platforms"]) <= SOUND_LIBRARY_PLATFORMS
    assert any(s["reason"].endswith("use rendered mode there") for s in plan["skipped"])
    for u in plan["ad_units"]:
        assert u["variant_file"] == "master_voiceover_only.mp4"
        brief = u["sound_brief"]
        assert brief["filters"]["genre"] and brief["filters"]["tempo_bpm"]
        assert brief["library"]
        assert not u["demo_only"]
    assert not plan["issues"]


def test_platform_sound_ignores_demo_only_cuts(planner, tree, sample_manifest):
    """Library music is precleared — a demo-only rendered track must not block."""
    sample_manifest["variants"][0]["track"]["demo_only"] = True
    plan = planner.plan(sample_manifest, FULL_US(tree), ["tiktok"],
                        music_mode="platform_sound")
    assert not any("DEMO-ONLY" in i for i in plan["issues"])
    plan2 = planner.plan(sample_manifest, {"us_states": ["GA"], "intl": [], "age_bands": [],
                                           "tastes": ["urban_hiphop"], "moods": ["hype"]},
                         ["meta"], music_mode="rendered")
    assert any("DEMO-ONLY" in i for i in plan2["issues"])


def test_blast_bundle_writes_sound_briefs(planner, tree, job_dir, sample_manifest):
    (job_dir / "master_voiceover_only.mp4").write_bytes(b"\x00" * 64)
    plan = planner.plan(sample_manifest, {"us_states": ["GA", "TN"], "intl": [],
                                          "age_bands": ["18_24"], "tastes": ["urban_hiphop"],
                                          "moods": ["hype"]},
                        ["tiktok", "meta"], music_mode="platform_sound")
    result = build_rollout_bundle(job_dir, sample_manifest, plan)
    briefs = json.loads((job_dir / "rollout" / "platform_sound" / "sound_briefs.json").read_text())
    assert briefs["briefs"] and briefs["briefs"][0]["filters"]["genre"]
    assert any("master_voiceover_only" in a for a in result["assets"])


def test_blast_rejects_bad_inputs(planner, tree, sample_manifest):
    with pytest.raises(ValueError):
        planner.plan(sample_manifest, FULL_US(tree), ["meta"], music_mode="nope")
    with pytest.raises(ValueError):
        planner.plan(sample_manifest, FULL_US(tree), ["spotify"], music_mode="platform_sound")
