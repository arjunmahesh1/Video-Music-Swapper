"""Demo-only reference tracks pulled from public sources and watermarked.

For pitching only: 'imagine your ad with a track like this'. Every file
this source emits carries an audible periodic watermark and is flagged
demo_only so the pipeline can label it and keep it out of deliverables.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..intelligence.demographics import MusicDirection
from .base import MusicSource, make_candidate
from .watermark import watermark_audio


class ReferenceSource(MusicSource):
    id = "reference"
    label = "Reference tracks (demo-only, watermarked)"
    demo_only = True
    description = "Pulls uncleared reference music matching the direction. Watermarked; never deliverable."

    def available(self) -> tuple[bool, str]:
        if shutil.which("yt-dlp") is None:
            try:
                import yt_dlp  # noqa: F401
            except ImportError:
                return False, "yt-dlp not installed."
        return True, "Reference pulls enabled (watermarked, demo use only)."

    def candidates(
        self,
        direction: MusicDirection,
        workspace: Path,
        ad_duration: float,
        limit: int = 4,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        context = context or {}
        queries: list[str] = []
        if context.get("reference_query"):  # user asked for a specific song
            queries.append(str(context["reference_query"]))
        queries.extend(f"{term} song" for term in direction.search_terms[:2])

        results: list[dict[str, Any]] = []
        for query in queries:
            if len(results) >= limit:
                break
            raw = self._search_download(query, workspace)
            if raw is None:
                continue
            marked = workspace / f"{raw.stem}_demo_watermarked.mp3"
            try:
                watermark_audio(raw, marked)
                raw.unlink(missing_ok=True)
            except Exception as exc:
                print(f"[reference] watermark failed, dropping candidate: {exc}")
                continue
            results.append(
                make_candidate(
                    name=f"Reference: {query}",
                    audio_path=marked,
                    source=self.id,
                    license_note="UNCLEARED reference — watermarked, demo use only",
                    demo_only=True,
                    query=query,
                )
            )
        return results

    @staticmethod
    def _search_download(query: str, workspace: Path) -> Path | None:
        workspace.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", query).strip("_")[:50]
        target = workspace / f"ref_{slug}.%(ext)s"
        cmd = [
            "yt-dlp",
            f"ytsearch1:{query}",
            "-x", "--audio-format", "mp3",
            "--no-playlist",
            "--max-downloads", "1",
            "--match-filter", "duration < 420",
            "-o", str(target),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return None
        hits = sorted(workspace.glob(f"ref_{slug}*.mp3"))
        return hits[0] if hits else None
