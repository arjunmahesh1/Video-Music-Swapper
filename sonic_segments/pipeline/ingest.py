"""Media intake: direct uploads (preferred) or pulls from hosted URLs.

Uploads keep full quality; URL pulls (YouTube / Drive / Meta Ads Library)
are best-effort via yt-dlp and get probed so we can flag sources that are
too compressed to re-score well.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..intelligence.quality import SourceQuality, probe_source_quality


@dataclass(slots=True)
class IngestResult:
    video_path: Path
    quality: SourceQuality
    pulled_from_url: bool = False
    notes: list[str] = field(default_factory=list)


def ingest_media(
    workspace: Path,
    file_path: str | Path | None = None,
    url: str | None = None,
) -> IngestResult:
    """Resolve the ad video from an upload or a URL, then probe its quality."""
    workspace.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    if file_path:
        video_path = Path(file_path)
        pulled = False
    elif url:
        video_path = _pull_url(url.strip(), workspace)
        pulled = True
        notes.append("Source was pulled from a hosted link; direct uploads preserve more quality.")
    else:
        raise ValueError("Provide either an uploaded file or a URL.")

    quality = probe_source_quality(video_path)
    if not quality.ok:
        raise ValueError("; ".join(quality.warnings) or "Source has no usable audio.")
    return IngestResult(video_path=video_path, quality=quality, pulled_from_url=pulled, notes=notes)


def _pull_url(url: str, workspace: Path) -> Path:
    target = workspace / "source.%(ext)s"
    cmd = [
        "yt-dlp",
        url,
        "--no-playlist",
        "-f", "mp4/bestvideo*+bestaudio/best",
        "--merge-output-format", "mp4",
        "--max-filesize", "500M",
        "-o", str(target),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    hits = sorted(workspace.glob("source.*"))
    if not hits:
        detail = (result.stderr or result.stdout or "").strip()[-400:]
        raise ValueError(f"Could not pull media from that link: {detail}")
    return hits[0]
