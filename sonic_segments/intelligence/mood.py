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
            "a cheerful upbeat pop song with bright major-key melodies",
            "feel-good sunny music with a bouncy danceable rhythm",
            "joyful celebratory music that makes you smile",
        ],
        "adjectives": ["happy", "feel-good", "bright", "sunny"],
        "energy": 0.65, "tempo": 118, "valence": 0.9,
    },
    "hype": {
        "label": "Hype / High-energy",
        "prompts": [
            "an aggressive high-energy beat with heavy bass drops",
            "intense adrenaline-pumping workout music with pounding drums",
            "a powerful stadium anthem with explosive energy",
        ],
        "adjectives": ["high-energy", "hard-hitting", "intense", "powerful"],
        "energy": 0.92, "tempo": 145, "valence": 0.6,
    },
    "calm": {
        "label": "Calm / Relaxed",
        "prompts": [
            "soft calm ambient music with gentle warm textures",
            "a peaceful slow instrumental with soothing pads",
            "relaxing meditative background music",
        ],
        "adjectives": ["calm", "relaxed", "gentle", "soothing"],
        "energy": 0.15, "tempo": 75, "valence": 0.6,
    },
    "epic": {
        "label": "Epic / Cinematic",
        "prompts": [
            "epic cinematic orchestral trailer music with big percussion and brass",
            "a dramatic heroic film score building to a climax",
            "grand sweeping orchestral music with soaring strings",
        ],
        "adjectives": ["epic", "cinematic", "dramatic", "heroic"],
        "energy": 0.8, "tempo": 110, "valence": 0.55,
    },
    "romantic": {
        "label": "Romantic / Intimate",
        "prompts": [
            "a slow romantic love ballad with tender melodies",
            "sensual intimate music with soft warm instrumentation",
            "a dreamy romantic serenade with lush harmonies",
        ],
        "adjectives": ["romantic", "intimate", "sensual", "tender"],
        "energy": 0.3, "tempo": 85, "valence": 0.7,
    },
    "emotional": {
        "label": "Emotional / Moving",
        "prompts": [
            "a melancholic emotional ballad with sad piano and strings",
            "bittersweet nostalgic music that feels like longing",
            "a sorrowful slow song with a heartfelt melody",
        ],
        "adjectives": ["emotional", "heartfelt", "moving", "nostalgic"],
        "energy": 0.35, "tempo": 80, "valence": 0.3,
    },
    "dark": {
        "label": "Dark / Intense",
        "prompts": [
            "dark ominous suspenseful music with menacing tension",
            "a brooding sinister soundtrack with low drones",
            "a gritty menacing beat with a dark atmosphere",
        ],
        "adjectives": ["dark", "brooding", "gritty", "tense"],
        "energy": 0.6, "tempo": 100, "valence": 0.15,
    },
    "luxurious": {
        "label": "Luxurious / Elegant",
        "prompts": [
            "elegant sophisticated music with minimal piano and strings",
            "classy refined music for a luxury fashion show",
            "smooth premium downtempo with polished expensive production",
        ],
        "adjectives": ["elegant", "luxurious", "sophisticated", "premium"],
        "energy": 0.3, "tempo": 95, "valence": 0.6,
    },
    "playful": {
        "label": "Playful / Quirky",
        "prompts": [
            "a quirky playful tune with bouncy pizzicato and whimsy",
            "fun lighthearted comedic music with a skipping rhythm",
            "a cute cheerful jingle with toy-like sounds",
        ],
        "adjectives": ["playful", "quirky", "bouncy", "whimsical"],
        "energy": 0.55, "tempo": 120, "valence": 0.85,
    },
    "uplifting": {
        "label": "Uplifting / Inspiring",
        "prompts": [
            "an uplifting inspiring anthem that builds with hope",
            "triumphant motivational music with soaring melodies",
            "a hopeful inspirational soundtrack with rising energy",
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


# Krumhansl-Schmuckler key profiles for major/minor mode estimation.
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def estimate_mode_major(audio_path: str | Path) -> float:
    """Confidence in [0,1] that the piece is in a major key (valence proxy).

    Correlates the averaged chromagram against Krumhansl-Schmuckler profiles
    across all 12 rotations for both modes.
    """
    import librosa
    import numpy as np

    try:
        y, sr = librosa.load(str(audio_path), duration=30, mono=True)
        y_harm = librosa.effects.harmonic(y)
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr).mean(axis=1)
        chroma = (chroma - chroma.mean()) / (chroma.std() + 1e-9)

        def best_corr(profile: list[float]) -> float:
            prof = np.array(profile)
            prof = (prof - prof.mean()) / (prof.std() + 1e-9)
            return max(float(np.corrcoef(np.roll(prof, k), chroma)[0, 1]) for k in range(12))

        major, minor = best_corr(_KS_MAJOR), best_corr(_KS_MINOR)
        # Map the major-minus-minor margin to a soft 0..1 confidence.
        return 1.0 / (1.0 + math.exp(-(major - minor) * 8.0))
    except Exception as exc:
        print(f"[mood] mode estimation failed: {exc}")
        return 0.5


def _tempo_affinity(tempo: float, target: float) -> float:
    """Tempo similarity tolerant of half/double-time octave errors."""
    candidates = [tempo, tempo * 2, tempo / 2]
    return max(_gauss(t, target, 28.0) for t in candidates if t > 0)


def _dsp_mood_scores(features: dict[str, Any], mode_major: float | None = None) -> dict[str, float]:
    """Heuristic mood prior from objective DSP features."""
    energy = float(features.get("energy", 0.5))
    tempo = float(features.get("tempo", 110.0))
    brightness = float(features.get("brightness", 2000.0))
    hpss_ratio = float(features.get("hpss_ratio", 1.0))
    bright_norm = max(0.0, min(1.0, (brightness - 800.0) / 3200.0))
    harmonic_norm = max(0.0, min(1.0, hpss_ratio / 3.0))

    scores: dict[str, float] = {}
    for mood, spec in MOODS.items():
        s = 0.45 * _gauss(energy, spec["energy"], 0.22) + 0.25 * _tempo_affinity(tempo, spec["tempo"])
        # Brightness correlates with positive valence; harmonic dominance with calm/elegant moods.
        valence = spec["valence"]
        s += 0.1 * (1.0 - abs(bright_norm - valence))
        if mood in ("calm", "luxurious", "romantic", "emotional"):
            s += 0.1 * harmonic_norm
        else:
            s += 0.1 * (1.0 - harmonic_norm * 0.5)
        # Major/minor mode is the strongest objective valence signal we have.
        if mode_major is not None:
            s += 0.1 * (1.0 - abs(mode_major - valence))
        scores[mood] = s

    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}


def detect_mood(audio_path: str | Path, features: dict[str, Any] | None = None) -> MoodProfile:
    """Profile the mood of an audio file (ad soundtrack, music stem, or track)."""
    from .clap_model import ClapEngine

    if features is None:
        from audio_analyzer import analyze_audio_features

        features = analyze_audio_features(str(audio_path))

    mode_major = estimate_mode_major(audio_path)
    dsp_scores = _dsp_mood_scores(features, mode_major=mode_major)

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

    features = {**features, "mode_major": round(mode_major, 3)}
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
