"""Personal Spotify source (demo-only): re-score the ad with YOUR music.

Funnel: pull the connected user's combined library -> prefilter by artist
genre overlap with the direction -> download a small candidate pool ->
the matching engine picks the track whose audio actually fits the mood.
Downloads are cached so repeat demos are fast.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ..intelligence.demographics import MusicDirection
from .base import MusicSource, make_candidate

DOWNLOAD_CACHE = Path(".cache/spotify_tracks")
POOL_SIZE = 10
MAX_DOWNLOADS = 5


class SpotifyPersonalSource(MusicSource):
    id = "spotify"
    label = "My Spotify library (personal demo)"
    demo_only = True
    description = "Mood-matches a song you actually listen to. Demo use only — not cleared for ad delivery."

    def __init__(self, manager=None) -> None:
        self._manager = manager

    def _get_manager(self):
        if self._manager is None:
            from spotify_helper import SpotifyManager

            self._manager = SpotifyManager()
        return self._manager

    def available(self) -> tuple[bool, str]:
        try:
            manager = self._get_manager()
            if manager.sp is None and not manager.authenticate():
                return False, "Spotify not connected — connect it from the site header."
            return True, "Spotify connected."
        except Exception as exc:
            return False, f"Spotify unavailable: {exc}"

    def candidates(
        self,
        direction: MusicDirection,
        workspace: Path,
        ad_duration: float,
        limit: int = 4,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        manager = self._get_manager()
        if manager.sp is None and not manager.authenticate():
            return []

        library = manager.get_combined_library(liked_limit=200, top_limit=50)
        if not library:
            return []

        pool = self._prefilter(library, direction, manager)
        results: list[dict[str, Any]] = []
        DOWNLOAD_CACHE.mkdir(parents=True, exist_ok=True)

        for song in pool:
            if len(results) >= min(limit, MAX_DOWNLOADS):
                break
            audio_path = self._fetch(song)
            if audio_path is None:
                continue
            results.append(
                make_candidate(
                    name=song.get("name", "Unknown"),
                    artist=song.get("artist"),
                    audio_path=audio_path,
                    source=self.id,
                    license_note="From your Spotify library — DEMO USE ONLY (uncleared)",
                    demo_only=True,
                    spotify_id=song.get("id"),
                    listening_score=song.get("listening_score"),
                )
            )
        return results

    def _prefilter(self, library: list[dict], direction: MusicDirection, manager) -> list[dict]:
        """Rank by artist-genre overlap with the direction, then listening score."""
        target_terms = {g.lower() for g in direction.genres}
        for adjective_set in (direction.style_keywords,):
            target_terms.update(t.lower() for t in adjective_set)

        try:
            artist_ids = [s["artist_id"] for s in library if s.get("artist_id")]
            genre_map = manager.get_artist_genres(artist_ids) if artist_ids else {}
        except Exception:
            genre_map = {}

        def genre_overlap(song: dict) -> float:
            genres = [g.lower() for g in genre_map.get(song.get("artist_id"), [])]
            if not genres or not target_terms:
                return 0.0
            hits = sum(1 for g in genres for t in target_terms if t in g or g in t)
            return min(1.0, hits / 2)

        scored = [
            (0.6 * genre_overlap(s) + 0.4 * float(s.get("listening_score", 0.5)) + random.uniform(0, 0.05), s)
            for s in library
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [s for _, s in scored[:POOL_SIZE]]

    def _fetch(self, song: dict) -> Path | None:
        from spotify_helper import download_spotify_track

        track_id = song.get("id") or "unknown"
        cached = list(DOWNLOAD_CACHE.glob(f"{track_id}.*"))
        if cached:
            return cached[0]
        try:
            query = f"{song.get('artist', '')} {song.get('name', '')}".strip()
            out = download_spotify_track(
                song.get("spotify_url") or song.get("uri") or "",
                output_path=str(DOWNLOAD_CACHE / f"{track_id}.mp3"),
                search_query=query,
            )
            return Path(out) if out else None
        except Exception as exc:
            print(f"[spotify] download failed for {song.get('name')}: {exc}")
            return None
