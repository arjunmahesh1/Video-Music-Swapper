"""Local-folder catalog used for the first MVP before external API integration."""

from __future__ import annotations

from pathlib import Path

from ..models import TrackCandidate

SUPPORTED_AUDIO_SUFFIXES = {".mp3", ".wav", ".aac", ".m4a", ".flac", ".ogg"}


class LocalTrackCatalog:
    """Read replacement tracks from a local directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def list_tracks(self, limit: int | None = None) -> list[TrackCandidate]:
        """Return local files as track candidates."""
        tracks: list[TrackCandidate] = []
        for path in sorted(self.root.glob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
                continue
            tracks.append(
                TrackCandidate(
                    track_id=path.stem,
                    name=path.stem,
                    source_path=path,
                    source="local",
                )
            )
            if limit is not None and len(tracks) >= limit:
                break
        return tracks
