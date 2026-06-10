"""Local royalty-free catalog: tagged folders of tracks you own/licensed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..intelligence.demographics import MusicDirection
from .base import MusicSource, make_candidate

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
FEATURE_CACHE = Path(".cache/track_features.json")


class LocalCatalogSource(MusicSource):
    id = "library"
    label = "Royalty-free library (local)"
    description = "Tracks from your local licensed library folders (audio/, library/)."

    def __init__(self, roots: list[str | Path] | None = None) -> None:
        self.roots = [Path(r) for r in (roots or ["audio", "library"])]

    def available(self) -> tuple[bool, str]:
        files = self._list_files()
        if not files:
            return False, "No audio files found in local library folders (audio/, library/)."
        return True, f"{len(files)} local tracks indexed."

    def candidates(
        self,
        direction: MusicDirection,
        workspace: Path,
        ad_duration: float,
        limit: int = 4,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        cands = []
        for path in self._list_files():
            cands.append(
                make_candidate(
                    name=path.stem,
                    audio_path=path,
                    source=self.id,
                    license_note="Local library (assumed licensed/royalty-free)",
                    features_cached=True,
                )
            )
            cands[-1]["features"] = cached_features(path)
        return cands  # ranking happens downstream; return everything we have

    def _list_files(self) -> list[Path]:
        files: list[Path] = []
        for root in self.roots:
            if root.is_dir():
                files.extend(p for p in sorted(root.rglob("*")) if p.suffix.lower() in AUDIO_SUFFIXES)
        return files


def cached_features(audio_path: Path) -> dict[str, Any] | None:
    """Disk-cached librosa features keyed by path+mtime (analysis is ~5s/track)."""
    try:
        key = hashlib.sha1(f"{audio_path.resolve()}::{audio_path.stat().st_mtime}".encode()).hexdigest()
        cache: dict[str, Any] = {}
        if FEATURE_CACHE.exists():
            cache = json.loads(FEATURE_CACHE.read_text())
        if key in cache:
            return cache[key]

        from audio_analyzer import analyze_audio_features

        features = analyze_audio_features(str(audio_path))
        cache[key] = features
        FEATURE_CACHE.parent.mkdir(exist_ok=True)
        FEATURE_CACHE.write_text(json.dumps(cache))
        return features
    except Exception as exc:
        print(f"[catalog] feature extraction failed for {audio_path.name}: {exc}")
        return None
