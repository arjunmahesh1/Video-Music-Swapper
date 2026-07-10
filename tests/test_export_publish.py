"""Export bundle + publish dry-run: the local deployment-mechanism tests."""

import csv
import json
import zipfile
from pathlib import Path

import pytest

from sonic_segments.intelligence.rollout import RolloutPlanner
from sonic_segments.pipeline.export import GOOGLE_COLUMNS, SDF_COLUMNS, build_rollout_bundle
from sonic_segments.publish import build_meta_requests, dry_run_report

ALL_PLATFORMS = ["meta", "google_ads", "dv360", "tiktok", "spotify"]


@pytest.fixture()
def plan(sample_manifest):
    return RolloutPlanner().plan(sample_manifest, ALL_PLATFORMS, geo_ids=["atlanta_dma", "nashville_dma"], total_budget=500)


def test_bundle_builds_every_platform_file(job_dir, sample_manifest, plan):
    result = build_rollout_bundle(job_dir, sample_manifest, plan)
    out = job_dir / "rollout"
    for rel in ("rollout_plan.json", "rollout_plan.md", "meta/adsets.json",
                "google_ads/demand_gen.csv", "dv360/sdf_line_items.csv",
                "tiktok/adgroups.json", "spotify/adsets.json"):
        assert (out / rel).is_file(), f"missing {rel}"
    assert (job_dir / "rollout_bundle.zip").is_file()
    assert result["assets"], "variant creatives must be copied into assets/"

    with zipfile.ZipFile(job_dir / "rollout_bundle.zip") as zf:
        names = zf.namelist()
    assert "rollout/rollout_plan.md" in names
    assert any(n.startswith("rollout/assets/") for n in names)


def test_meta_payloads_are_api_shaped(job_dir, sample_manifest, plan):
    build_rollout_bundle(job_dir, sample_manifest, plan)
    data = json.loads((job_dir / "rollout" / "meta" / "adsets.json").read_text())
    assert data["ad_sets"], "no Meta ad sets exported"
    for entry in data["ad_sets"]:
        adset = entry["adset"]
        assert adset["status"] == "PAUSED"
        assert adset["daily_budget_cents"] > 0
        t = adset["targeting"]
        assert t["geo_locations"] and t["age_min"] >= 18
        assert t["flexible_spec"][0]["interests"]


def test_csv_exports_parse_with_expected_headers(job_dir, sample_manifest, plan):
    build_rollout_bundle(job_dir, sample_manifest, plan)
    with open(job_dir / "rollout" / "google_ads" / "demand_gen.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == GOOGLE_COLUMNS and len(rows) > 1
    with open(job_dir / "rollout" / "dv360" / "sdf_line_items.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == SDF_COLUMNS and len(rows) > 1
    for row in rows[1:]:
        assert row[2] == "Draft", "DV360 line items must land as drafts"


def test_brief_reads_like_a_pitch(job_dir, sample_manifest, plan):
    build_rollout_bundle(job_dir, sample_manifest, plan)
    md = (job_dir / "rollout" / "rollout_plan.md").read_text()
    assert "Gatorade" in md
    assert "re-scored cuts" in md
    assert "What runs where" in md and "Why each unit exists" in md
    assert "paused" in md.lower()


def test_meta_request_sequence_is_complete_and_paused(plan):
    steps = build_meta_requests(plan)
    meta_units = [u for u in plan["ad_units"] if u["platform"] == "meta"]
    # 1 campaign + 4 steps per unit (video, adset, creative, ad)
    assert len(steps) == 1 + 4 * len(meta_units)
    for s in steps:
        assert s["url"].startswith("https://graph.facebook.com/")
        if "status" in s["body"]:
            assert s["body"]["status"] == "PAUSED"
    adset_steps = [s for s in steps if s["step"].startswith("adset_")]
    for s in adset_steps:
        assert s["body"]["campaign_id"] == "{campaign_id}"
        assert s["body"]["daily_budget"] > 0


def test_dry_run_never_touches_network(plan, monkeypatch):
    import requests

    def _boom(*a, **k):  # any network call fails the test
        raise AssertionError("dry run must not make network calls")

    monkeypatch.setattr(requests, "post", _boom)
    monkeypatch.setattr(requests, "get", _boom)
    for platform in ALL_PLATFORMS:
        report = dry_run_report(plan, platform)
        assert "DRY RUN" in report
    assert "graph.facebook.com" in dry_run_report(plan, "meta")


def test_spotify_audio_export(job_dir, sample_manifest, plan):
    """Stub media can't be transcoded — exporter must degrade gracefully."""
    result = build_rollout_bundle(job_dir, sample_manifest, plan)
    # ffmpeg fails on the 64-byte stubs; adsets.json must still exist
    data = json.loads((job_dir / "rollout" / "spotify" / "adsets.json").read_text())
    assert data["ad_sets"]
    for entry in data["ad_sets"]:
        assert entry["advanced_targeting"]["type"] == "genre"
        assert entry["advanced_targeting"]["values"]
    assert isinstance(result["spotify_audio"], list)
