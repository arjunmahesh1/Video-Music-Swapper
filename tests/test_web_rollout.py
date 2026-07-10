"""Rollout endpoints through the real FastAPI app (local deployment test)."""

import json
import zipfile

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(monkeypatch, tmp_path, job_dir):
    """App wired to a temp job store containing one finished campaign."""
    from sonic_segments.pipeline import jobs

    monkeypatch.setattr(jobs, "JOBS_ROOT", tmp_path)
    monkeypatch.setattr(jobs.JobStore, "_instance", None)

    from sonic_segments.web import main as web_main

    store = jobs.JobStore.get()
    job = jobs.Job(id=job_dir.name, kind="campaign", status="done")
    job._store = store
    store.jobs[job.id] = job
    return TestClient(web_main.app)


def test_rollout_meta_endpoint(client):
    resp = client.get("/api/rollout/meta")
    assert resp.status_code == 200
    data = resp.json()
    assert {p["id"] for p in data["platforms"]} == {"meta", "google_ads", "dv360", "tiktok", "spotify"}
    assert len(data["geos"]) >= 12


def test_rollout_build_end_to_end(client, job_dir):
    resp = client.post(
        f"/api/campaigns/{job_dir.name}/rollout",
        data={"platforms": "meta,google_ads,spotify", "geos": "atlanta_dma,nashville_dma", "budget": 750},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["summary"]["ad_units"] > 0
    assert data["issues"] == []
    assert data["bundle_url"].endswith("rollout_bundle.zip")

    # the bundle really exists and is servable through /media
    assert (job_dir / "rollout_bundle.zip").is_file()
    with zipfile.ZipFile(job_dir / "rollout_bundle.zip") as zf:
        assert "rollout/rollout_plan.md" in zf.namelist()
    media = client.get(data["bundle_url"])
    assert media.status_code == 200

    plan = json.loads((job_dir / "rollout" / "rollout_plan.json").read_text())
    assert abs(sum(u["budget"] for u in plan["ad_units"]) - 750) < 0.05


def test_rollout_rejects_bad_requests(client, job_dir):
    assert client.post("/api/campaigns/nope/rollout", data={"platforms": "meta"}).status_code == 404
    assert client.post(f"/api/campaigns/{job_dir.name}/rollout", data={"platforms": ""}).status_code == 400
