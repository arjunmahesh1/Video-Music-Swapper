"""Demographic -> music-direction knowledge base.

The KB (data/demographics.json) encodes which sonic directions convert for
which audience segments, grounded in streaming-market and ad-music research.
A MusicDirection is the contract consumed by every music source: it carries
genres, search terms, a generative prompt, and tempo/energy targets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .mood import MOODS

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "demographics.json"


@dataclass(slots=True)
class MusicDirection:
    """Resolved music brief for one (demographic, mood) pair."""

    segment_id: str
    segment_label: str
    mood: str
    mood_label: str
    genres: list[str]
    search_terms: list[str]
    prompt: str
    tempo_range: tuple[float, float]
    energy_range: tuple[float, float]
    style_keywords: list[str] = field(default_factory=list)
    example_artists: list[str] = field(default_factory=list)
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "segment_label": self.segment_label,
            "mood": self.mood,
            "mood_label": self.mood_label,
            "genres": self.genres,
            "search_terms": self.search_terms,
            "prompt": self.prompt,
            "tempo_range": list(self.tempo_range),
            "energy_range": list(self.energy_range),
            "style_keywords": self.style_keywords,
            "example_artists": self.example_artists,
            "rationale": self.rationale,
        }


class DemographicsKB:
    """Loads and queries the curated demographic knowledge base."""

    def __init__(self, kb_path: str | Path = KB_PATH) -> None:
        data = json.loads(Path(kb_path).read_text(encoding="utf-8"))
        self.segments: dict[str, dict[str, Any]] = {s["id"]: s for s in data["segments"]}
        self.meta: dict[str, Any] = data.get("_meta", {})

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls) -> "DemographicsKB":
        return cls()

    def list_segments(self) -> list[dict[str, Any]]:
        return [
            {
                "id": s["id"],
                "label": s["label"],
                "region": s.get("region", ""),
                "description": s.get("description", ""),
                "platforms": s.get("platforms", []),
                "core_genres": s.get("core_genres", []),
            }
            for s in self.segments.values()
        ]

    def get(self, segment_id: str) -> dict[str, Any]:
        if segment_id not in self.segments:
            raise KeyError(f"Unknown demographic segment: {segment_id}")
        return self.segments[segment_id]

    def direction(self, segment_id: str, mood: str) -> MusicDirection:
        """Merge a segment's preferences with the requested mood."""
        segment = self.get(segment_id)
        if mood not in MOODS:
            raise KeyError(f"Unknown mood: {mood}")
        mood_spec = MOODS[mood]
        adjectives = mood_spec["adjectives"]

        override = segment.get("mood_overrides", {}).get(mood)
        if override:
            genres = override["genres"]
            search_terms = override["search_terms"]
            prompt = override["prompt"]
            rationale = (
                f"Curated direction: {segment['label']} responds to {', '.join(genres)} "
                f"when the goal is a {mood_spec['label'].lower()} feel."
            )
        else:
            core = segment.get("core_genres", [])[:3]
            genres = core
            search_terms = [f"{adjectives[0]} {g}" for g in core[:2]]
            search_terms.append(f"{adjectives[1]} {adjectives[2]} instrumental")
            keywords = ", ".join(segment.get("style_keywords", [])[:2])
            prompt = f"{adjectives[0]} {core[0] if core else 'pop'} track"
            if keywords:
                prompt += f" with {keywords}"
            prompt += f", {adjectives[1]} and {adjectives[2]}"
            rationale = (
                f"Derived direction: blended {segment['label']} core genres with "
                f"{mood_spec['label'].lower()} characteristics (no curated override for this pair)."
            )

        # Mood shifts the segment's tempo/energy window toward its own targets.
        seg_tempo = segment.get("tempo_range", [70, 140])
        seg_energy = segment.get("energy_range", [0.2, 0.8])
        tempo_range = _nudge_range(seg_tempo, mood_spec["tempo"], spread=22.0)
        energy_range = _nudge_range(seg_energy, mood_spec["energy"], spread=0.18)

        return MusicDirection(
            segment_id=segment_id,
            segment_label=segment["label"],
            mood=mood,
            mood_label=mood_spec["label"],
            genres=genres,
            search_terms=search_terms,
            prompt=prompt,
            tempo_range=tempo_range,
            energy_range=energy_range,
            style_keywords=segment.get("style_keywords", []),
            example_artists=segment.get("example_artists", []),
            rationale=rationale,
        )

    def mood_only_direction(self, mood: str) -> MusicDirection:
        """Direction for the personal-demo flow where no demographic is targeted."""
        if mood not in MOODS:
            raise KeyError(f"Unknown mood: {mood}")
        spec = MOODS[mood]
        return MusicDirection(
            segment_id="personal",
            segment_label="Personal taste",
            mood=mood,
            mood_label=spec["label"],
            genres=[],
            search_terms=[f"{spec['adjectives'][0]} {spec['adjectives'][1]} music"],
            prompt=spec["prompts"][0],
            tempo_range=(max(40.0, spec["tempo"] - 30), spec["tempo"] + 30),
            energy_range=(max(0.0, spec["energy"] - 0.25), min(1.0, spec["energy"] + 0.25)),
            rationale="Matched purely on mood against the listener's own library.",
        )


def _nudge_range(base: list[float], target: float, spread: float) -> tuple[float, float]:
    """Intersect a segment range with a mood-centred window; widen if disjoint."""
    lo, hi = float(base[0]), float(base[1])
    m_lo, m_hi = target - spread, target + spread
    new_lo, new_hi = max(lo, m_lo), min(hi, m_hi)
    if new_lo >= new_hi:  # mood target sits outside the segment window: meet halfway
        mid = (max(lo, min(hi, target)) + (lo + hi) / 2) / 2
        new_lo, new_hi = mid - spread / 2, mid + spread / 2
    return (round(new_lo, 3), round(new_hi, 3))
