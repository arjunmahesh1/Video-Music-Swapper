"""Product-facing dataclasses for the Sonic Segments MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BrandProfile:
    """Compact description of the brand and desired sonic direction."""

    name: str
    product_category: str | None = None
    target_demo: str | None = None
    vibe: str | None = None
    desired_energy_min: float | None = None
    desired_energy_max: float | None = None
    desired_tempo_min: float | None = None
    desired_tempo_max: float | None = None
    desired_genres: list[str] = field(default_factory=list)
    desired_moods: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VoiceoverAnalysis:
    """Summary of detected narration coverage within an ad."""

    backend_id: str
    speech_segments: list[tuple[float, float]]
    speech_coverage_seconds: float
    speech_coverage_ratio: float
    detected_segment_count: int


@dataclass(slots=True)
class AdAnalysis:
    """Top-level analysis result for one ad."""

    video_path: Path
    audio_path: Path
    features: dict[str, Any]
    inferred_genres: list[str]
    brand_profile: BrandProfile | None = None
    voiceover: VoiceoverAnalysis | None = None
    mismatches: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TrackCandidate:
    """Candidate replacement track from a local folder or later catalog API."""

    track_id: str
    name: str
    artist: str | None = None
    source_path: Path | None = None
    source: str = "local"
    genres: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VariantRender:
    """One rendered ad variant and the reasoning behind it."""

    track: TrackCandidate
    output_path: Path
    audio_output_path: Path
    preserve_voiceover: bool
    backend_id: str | None
    predicted_notes: list[str] = field(default_factory=list)
