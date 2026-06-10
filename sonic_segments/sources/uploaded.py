"""Customer-provided track: they upload the music they already licensed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..intelligence.demographics import MusicDirection
from .base import MusicSource, make_candidate


class UploadedTrackSource(MusicSource):
    id = "uploaded"
    label = "Your own track (upload)"
    description = "Use a track you already own or licensed; we re-score every cut with it."

    def available(self) -> tuple[bool, str]:
        return True, "Provide the file with your request."

    def candidates(
        self,
        direction: MusicDirection,
        workspace: Path,
        ad_duration: float,
        limit: int = 4,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        context = context or {}
        paths = context.get("uploaded_tracks") or []
        return [
            make_candidate(
                name=Path(p).stem,
                audio_path=Path(p),
                source=self.id,
                license_note="Customer-provided (assumed licensed)",
            )
            for p in paths
            if Path(p).exists()
        ]
