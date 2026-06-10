"""Unified MusicSource interface.

Every replacement-music origin (local library, Jamendo, personal Spotify,
MusicGen, demo-only reference pulls) implements the same contract: given a
MusicDirection, produce candidate audio files on disk. The pipeline then
ranks candidates with the matching engine and renders the winner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..intelligence.demographics import MusicDirection


class MusicSource(ABC):
    id: str = "base"
    label: str = "Base source"
    demo_only: bool = False
    description: str = ""

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """Whether this source can run right now, and why not if it can't."""

    @abstractmethod
    def candidates(
        self,
        direction: MusicDirection,
        workspace: Path,
        ad_duration: float,
        limit: int = 4,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Produce candidate dicts: {'name', 'artist', 'audio_path', 'license', ...}.

        Audio files must exist on disk (downloaded/generated into `workspace`).
        `context` carries flow-specific extras (e.g. Spotify session, uploads).
        """

    def info(self) -> dict[str, Any]:
        ok, reason = self.available()
        return {
            "id": self.id,
            "label": self.label,
            "demo_only": self.demo_only,
            "description": self.description,
            "available": ok,
            "reason": reason,
        }


def make_candidate(
    name: str,
    audio_path: Path,
    source: str,
    artist: str | None = None,
    license_note: str = "",
    demo_only: bool = False,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "artist": artist,
        "audio_path": str(audio_path),
        "source": source,
        "license": license_note,
        "demo_only": demo_only,
        "metadata": metadata,
    }
