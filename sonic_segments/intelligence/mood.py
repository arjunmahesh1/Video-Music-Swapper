"""Mood profiling for ads and tracks.

Fuses CLAP zero-shot classification (when available) with librosa DSP
heuristics so mood detection works fully offline and degrades gracefully.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Canonical mood taxonomy. Each mood carries:
# - prompts: CLAP zero-shot text prompts
# - adjectives: used when composing search terms / MusicGen prompts
# - energy / tempo: heuristic targets in [0,1] and BPM for the DSP prior
MOODS: dict[str, dict[str, Any]] = {
    "happy": {
        "label": "Happy / Feel-good",
        "prompts": [
            "happy upbeat feel-good music",
            "cheerful bright pop song with major chords",
            "joyful sunny music that makes you smile",
        ],
        "adjectives": ["happy", "feel-good", "bright", "sunny"],
        "energy": 0.65, "tempo": 118, "valence": 0.9,
    },
    "hype": {
        "label": "Hype / High-energy",
        "prompts": [
            "high energy intense hype music",
            "aggressive powerful beat with heavy bass",
            "adrenaline pumping workout music",
        ],
        "adjectives": ["high-energy", "hard-hitting", "intense", "powerful"],
        "energy": 0.92, "tempo": 145, "valence": 0.6,
    },
    "calm": {
        "label": "Calm / Relaxed",
        "prompts": [
            "calm relaxing peaceful music",
            "soft gentle ambient background music",
            "slow soothing meditation music",
        ],
        "adjectives": ["calm", "relaxed", "gentle", "soothing"],
        "energy": 0.15, "tempo": 75, "valence": 0.6,
    },
    "epic": {
        "label": "Epic / Cinematic",
        "prompts": [
            "epic cinematic orchestral trailer music",
            "dramatic heroic film score with big drums",
            "grand sweeping orchestral build",
        ],
        "adjectives": ["epic", "cinematic", "dramatic", "heroic"],
        "energy": 0.8, "tempo": 110, "valence": 0.55,
    },
    "romantic": {
        "label": "Romantic / Intimate",
        "prompts": [
            "romantic intimate love song",
            "sensual slow music with warm vocals",
            "tender romantic ballad",
        ],
        "adjectives": ["romantic", "intimate", "sensual", "tender"],
        "energy": 0.3, "tempo": 85, "valence": 0.7,
    },
    "emotional": {
        "label": "Emotional / Moving",
        "prompts": [
            "emotional moving sad music",
            "melancholic heartfelt ballad with strings",
            "bittersweet nostalgic emotional song",
        ],
        "adjectives": ["emotional", "heartfelt", "moving", "nostalgic"],
        "energy": 0.35, "tempo": 80, "valence": 0.3,
    },
    "dark": {
        "label": "Dark / Intense",
        "prompts": [
            "dark intense brooding music",
            "ominous tense suspenseful soundtrack",
            "menacing gritty beat",
        ],
        "adjectives": ["dark", "brooding", "gritty", "tense"],
        "energy": 0.6, "tempo": 100, "valence": 0.15,
    },
    "luxurious": {
        "label": "Luxurious / Elegant",
        "prompts": [
            "elegant sophisticated luxurious music",
            "classy minimal piano and strings, premium feel",
            "smooth high-end fashion show music",
        ],
        "adjectives": ["elegant", "luxurious", "sophisticated", "premium"],
        "energy": 0.3, "tempo": 95, "valence": 0.6,
    },
    "playful": {
        "label": "Playful / Quirky",
        "prompts": [
            "playful quirky fun music",
            "bouncy whimsical lighthearted tune",
            "cute comedic upbeat jingle",
        ],
        "adjectives": ["playful", "quirky", "bouncy", "whimsical"],
        "energy": 0.55, "tempo": 120, "valence": 0.85,
    },
    "uplifting": {
        "label": "Uplifting / Inspiring",
        "prompts": [
            "uplifting inspiring motivational music",
            "hopeful soaring anthem that builds",
            "triumphant inspirational soundtrack",
        ],
        "adjectives": ["uplifting", "inspiring", "hopeful", "triumphant"],
        "energy": 0.7, "tempo": 120, "valence": 0.8,
    },
}


@dataclass(slots=True)
class MoodProfile:
    """Mood scores for one piece of audio."""

    scores: dict[str, float]
    primary: str
    secondary: str
    energy: float
    tempo: float
    method: str  # "clap+dsp", "clap", or "dsp"
    features: dict[str, Any] = field(default_factory=dict)

    def top(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "primary": self.primary,
            "primary_label": MOODS[self.primary]["label"],
            "secondary": self.secondary,
            "energy": round(self.energy, 3),
            "tempo": round(self.tempo, 1),
            "method": self.method,
        }


def _gauss(value: float, target: float, sigma: float) -> float:
    return math.exp(-((value - target) ** 2) / (2 * sigma**2))


def _tempo_affinity(tempo: float, target: float) -> float:
    """Tempo similarity tolerant of half/double-time octave errors."""
    candidates = [tempo, tempo * 2, tempo / 2]
    return max(_gauss(t, target, 28.0) for t in candidates if t > 0)


def _dsp_mood_scores(features: dict[str, Any]) -> dict[str, float]:
    """Heuristic mood prior from objective DSP features."""
    energy = float(features.get("energy", 0.5))
    tempo = float(features.get("tempo", 110.0))
    brightness = float(features.get("brightness", 2000.0))
    hpss_ratio = float(features.get("hpss_ratio", 1.0))
    bright_norm = max(0.0, min(1.0, (brightness - 800.0) / 3200.0))
    harmonic_norm = max(0.0, min(1.0, hpss_ratio / 3.0))

    scores: dict[str, float] = {}
    for mood, spec in MOODS.items():
        s = 0.5 * _gauss(energy, spec["energy"], 0.22) + 0.3 * _tempo_affinity(tempo, spec["tempo"])
        # Brightness correlates with positive valence; harmonic dominance with calm/elegant moods.
        valence = spec["valence"]
        s += 0.1 * (1.0 - abs(bright_norm - valence))
        if mood in ("calm", "luxurious", "romantic", "emotional"):
            s += 0.1 * harmonic_norm
        else:
            s += 0.1 * (1.0 - harmonic_norm * 0.5)
        scores[mood] = s

    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}


def detect_mood(audio_path: str | Path, features: dict[str, Any] | None = None) -> MoodProfile:
    """Profile the mood of an audio file (ad soundtrack, music stem, or track)."""
    from .clap_model import ClapEngine

    if features is None:
        from audio_analyzer import analyze_audio_features

        features = analyze_audio_features(str(audio_path))

    dsp_scores = _dsp_mood_scores(features)

    clap = ClapEngine.get()
    if clap is not None:
        try:
            clap_scores = clap.zero_shot(audio_path, {m: spec["prompts"] for m, spec in MOODS.items()})
            fused = {m: 0.75 * clap_scores[m] + 0.25 * dsp_scores[m] for m in MOODS}
            method = "clap+dsp"
        except Exception as exc:
            print(f"[mood] CLAP scoring failed, using DSP only: {exc}")
            fused, method = dsp_scores, "dsp"
    else:
        fused, method = dsp_scores, "dsp"

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return MoodProfile(
        scores=fused,
        primary=ranked[0][0],
        secondary=ranked[1][0],
        energy=float(features.get("energy", 0.5)),
        tempo=float(features.get("tempo", 0.0)),
        method=method,
        features=features,
    )
