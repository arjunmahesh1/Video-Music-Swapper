import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def sample_manifest() -> dict:
    """A campaign manifest shaped exactly like pipeline/campaign.py writes."""
    return {
        "id": "testjob12345",
        "brand": "Gatorade",
        "vibe": "gritty training montage",
        "mode": "variants",
        "original": "source.mp4",
        "duration": 30.0,
        "quality_warnings": [],
        "analysis": {"mood": {"primary": "hype"}, "speech": {"coverage_ratio": 0.4}},
        "variants": [
            {
                "file": "variants/01_gen_z_hype.mp4",
                "label": "Gen-Z (16-24) · Hype",
                "segment_id": "gen_z",
                "segment_label": "Gen-Z (16-24)",
                "mood": "hype",
                "mood_label": "Hype",
                "direction": {"genres": ["trap", "phonk"], "rationale": "Gen-Z responds to trap when the goal is hype."},
                "track": {"name": "Test Track A", "artist": "A", "source": "library", "license": "licensed", "demo_only": False},
                "match": {"total": 0.82},
            },
            {
                "file": "variants/02_country_heartland_happy.mp4",
                "label": "Country & Heartland · Happy",
                "segment_id": "country_heartland",
                "segment_label": "Country & Heartland (25-55)",
                "mood": "happy",
                "mood_label": "Happy",
                "direction": {"genres": ["country"], "rationale": "Heartland responds to country."},
                "track": {"name": "Test Track B", "artist": "B", "source": "jamendo", "license": "CC-BY", "demo_only": False},
                "match": {"total": 0.71},
            },
        ],
    }


@pytest.fixture()
def job_dir(tmp_path: Path, sample_manifest: dict) -> Path:
    """A fake campaign job directory with stub media files."""
    d = tmp_path / "testjob12345"
    (d / "variants").mkdir(parents=True)
    (d / "source.mp4").write_bytes(b"\x00" * 64)
    for v in sample_manifest["variants"]:
        (d / v["file"]).write_bytes(b"\x00" * 64)
    (d / "manifest.json").write_text(json.dumps(sample_manifest))
    return d
