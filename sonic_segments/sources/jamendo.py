"""Jamendo: ~600k Creative Commons tracks, searchable by tags/mood/speed.

Free client_id from https://devportal.jamendo.com — set JAMENDO_CLIENT_ID in .env.
This is the 'legal path' catalog for cleared, monetizable replacement music.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import requests

from ..intelligence.demographics import MusicDirection
from .base import MusicSource, make_candidate

API_URL = "https://api.jamendo.com/v3.0/tracks/"


class JamendoSource(MusicSource):
    id = "jamendo"
    label = "Jamendo (royalty-free API)"
    description = "Creative Commons catalog searched by the demographic/mood direction."

    def available(self) -> tuple[bool, str]:
        if not os.environ.get("JAMENDO_CLIENT_ID"):
            return False, "Set JAMENDO_CLIENT_ID in .env (free key at devportal.jamendo.com)."
        return True, "Jamendo API configured."

    def candidates(
        self,
        direction: MusicDirection,
        workspace: Path,
        ad_duration: float,
        limit: int = 4,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        client_id = os.environ.get("JAMENDO_CLIENT_ID", "")
        seen: set[str] = set()
        results: list[dict[str, Any]] = []

        queries = list(direction.search_terms[:3]) + [" ".join(direction.genres[:2])]
        for query in queries:
            if len(results) >= limit:
                break
            try:
                tracks = self._search(client_id, query, direction)
            except Exception as exc:
                print(f"[jamendo] search '{query}' failed: {exc}")
                continue
            for track in tracks:
                if track["id"] in seen or len(results) >= limit:
                    continue
                seen.add(track["id"])
                audio_path = self._download(track, workspace)
                if audio_path is None:
                    continue
                results.append(
                    make_candidate(
                        name=track.get("name", "Untitled"),
                        artist=track.get("artist_name"),
                        audio_path=audio_path,
                        source=self.id,
                        license_note=f"Creative Commons ({track.get('license_ccurl', 'jamendo')})",
                        jamendo_id=track["id"],
                        query=query,
                    )
                )
        return results

    def _search(self, client_id: str, query: str, direction: MusicDirection) -> list[dict[str, Any]]:
        lo, hi = direction.tempo_range
        speed = "high" if lo >= 120 else "low" if hi <= 90 else "medium"
        params = {
            "client_id": client_id,
            "format": "json",
            "limit": 8,
            "search": query,
            "speed": speed,
            "include": "musicinfo licenses",
            "audioformat": "mp32",
            "order": "popularity_month",
            "vocalinstrumental": "instrumental",
        }
        resp = requests.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        tracks = resp.json().get("results", [])
        if not tracks:  # instrumental-only can be too strict; retry without it
            params.pop("vocalinstrumental")
            resp = requests.get(API_URL, params=params, timeout=30)
            resp.raise_for_status()
            tracks = resp.json().get("results", [])
        return tracks

    def _download(self, track: dict[str, Any], workspace: Path) -> Path | None:
        url = track.get("audiodownload") or track.get("audio")
        if not url:
            return None
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", track.get("name", track["id"])).strip("_")[:60]
        out = workspace / f"jamendo_{track['id']}_{slug}.mp3"
        if out.exists():
            return out
        try:
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                workspace.mkdir(parents=True, exist_ok=True)
                with open(out, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
            return out
        except Exception as exc:
            print(f"[jamendo] download failed for {track.get('name')}: {exc}")
            return None
