"""Music source registry."""

from __future__ import annotations

from .base import MusicSource, make_candidate
from .jamendo import JamendoSource
from .local_catalog import LocalCatalogSource
from .musicgen import MusicGenSource
from .reference import ReferenceSource
from .spotify_personal import SpotifyPersonalSource
from .uploaded import UploadedTrackSource

_REGISTRY: dict[str, MusicSource] | None = None


def get_sources() -> dict[str, MusicSource]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = {
            src.id: src
            for src in (
                LocalCatalogSource(),
                JamendoSource(),
                UploadedTrackSource(),
                MusicGenSource(),
                SpotifyPersonalSource(),
                ReferenceSource(),
            )
        }
    return _REGISTRY


def sources_info() -> list[dict]:
    return [src.info() for src in get_sources().values()]


__all__ = ["MusicSource", "make_candidate", "get_sources", "sources_info"]
