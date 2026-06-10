"""Score candidate tracks against a MusicDirection.

Combines CLAP style similarity (audio vs. the direction's text prompt),
mood-profile agreement, and tempo/energy window fit into one ranked score.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .demographics import MusicDirection
from .mood import MOODS, MoodProfile, detect_mood


def score_track(
    audio_path: str | Path,
    direction: MusicDirection,
    mood_profile: MoodProfile | None = None,
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return component scores and a fused total in [0, 1]."""
    from .clap_model import ClapEngine

    if mood_profile is None:
        mood_profile = detect_mood(audio_path, features=features)

    mood_fit = mood_profile.scores.get(direction.mood, 0.0)
    # Mood scores are a distribution over 10 moods; rescale so a clear winner ~1.0.
    mood_fit = min(1.0, mood_fit / 0.35)

    tempo_fit = _range_fit(mood_profile.tempo, direction.tempo_range, sigma=25.0, octave_tolerant=True)
    energy_fit = _range_fit(mood_profile.energy, direction.energy_range, sigma=0.18)

    clap = ClapEngine.get()
    style_fit = None
    if clap is not None:
        try:
            prompts = [direction.prompt] + [f"{g} music" for g in direction.genres[:3]]
            raw = clap.text_similarity(audio_path, prompts)
            # CLAP cosine for a good match typically lands ~0.25-0.55.
            style_fit = max(0.0, min(1.0, (raw - 0.05) / 0.45))
        except Exception as exc:
            print(f"[matching] CLAP style scoring failed: {exc}")

    if style_fit is not None:
        total = 0.45 * style_fit + 0.30 * mood_fit + 0.15 * tempo_fit + 0.10 * energy_fit
    else:
        total = 0.55 * mood_fit + 0.27 * tempo_fit + 0.18 * energy_fit

    return {
        "total": round(float(total), 4),
        "style_fit": None if style_fit is None else round(style_fit, 4),
        "mood_fit": round(mood_fit, 4),
        "tempo_fit": round(tempo_fit, 4),
        "energy_fit": round(energy_fit, 4),
        "track_mood": mood_profile.primary,
        "track_tempo": round(mood_profile.tempo, 1),
        "track_energy": round(mood_profile.energy, 3),
        "explanation": _explain(direction, mood_profile, total),
    }


def rank_tracks(
    candidates: list[dict[str, Any]],
    direction: MusicDirection,
) -> list[dict[str, Any]]:
    """Score and sort candidates (each needs an 'audio_path'; mutated in place)."""
    scored = []
    for cand in candidates:
        try:
            cand["match"] = score_track(cand["audio_path"], direction, features=cand.get("features"))
            scored.append(cand)
        except Exception as exc:
            print(f"[matching] skipping {cand.get('name', cand.get('audio_path'))}: {exc}")
    scored.sort(key=lambda c: c["match"]["total"], reverse=True)
    return scored


def _range_fit(value: float, rng: tuple[float, float], sigma: float, octave_tolerant: bool = False) -> float:
    if value <= 0:
        return 0.5  # unknown — neutral
    candidates = [value]
    if octave_tolerant:
        candidates += [value * 2, value / 2]
    best = 0.0
    lo, hi = rng
    for v in candidates:
        if lo <= v <= hi:
            return 1.0
        edge = lo if v < lo else hi
        best = max(best, math.exp(-((v - edge) ** 2) / (2 * sigma**2)))
    return best


def _explain(direction: MusicDirection, profile: MoodProfile, total: float) -> str:
    mood_label = MOODS[profile.primary]["label"]
    verdict = "strong" if total >= 0.65 else "decent" if total >= 0.45 else "weak"
    return (
        f"Reads as {mood_label.lower()} at {profile.tempo:.0f} BPM, energy {profile.energy:.2f} — "
        f"a {verdict} fit for {direction.mood_label.lower()} aimed at {direction.segment_label}."
    )
