"""Lazy singleton around LAION CLAP for zero-shot audio understanding.

CLAP embeds audio and text into a shared space, which gives us local,
state-of-the-art zero-shot mood/style classification without any online API.
Falls back gracefully (returns None) when transformers/torch or the model
weights are unavailable, in which case callers use DSP-only heuristics.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

DEFAULT_MODEL_ID = os.environ.get("CLAP_MODEL_ID", "laion/larger_clap_music_and_speech")
CLAP_SAMPLE_RATE = 48000
MAX_AUDIO_SECONDS = 30.0


class ClapEngine:
    _instance: "ClapEngine | None" = None
    _lock = threading.Lock()
    _failed = False

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        import torch
        from transformers import ClapModel, ClapProcessor

        self.torch = torch
        self.device = "cpu"  # CLAP inference on 30s clips is fast enough on CPU and avoids MPS op gaps
        self.model = ClapModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(model_id)
        self._text_cache: dict[tuple[str, ...], np.ndarray] = {}
        self._audio_cache: dict[str, np.ndarray] = {}

    @classmethod
    def get(cls) -> "ClapEngine | None":
        """Return the shared engine, or None when CLAP cannot be loaded."""
        if cls._failed:
            return None
        with cls._lock:
            if cls._instance is None:
                try:
                    cls._instance = ClapEngine()
                except Exception as exc:  # missing deps, no weights, offline first run
                    print(f"[clap] unavailable, falling back to DSP heuristics: {exc}")
                    cls._failed = True
                    return None
            return cls._instance

    def embed_audio(self, audio_path: str | Path, offset: float = 0.0) -> np.ndarray:
        import librosa

        key = f"{Path(audio_path).resolve()}::{offset}"
        cached = self._audio_cache.get(key)
        if cached is not None:
            return cached

        waveform, _ = librosa.load(
            str(audio_path), sr=CLAP_SAMPLE_RATE, mono=True, offset=offset, duration=MAX_AUDIO_SECONDS
        )
        inputs = self.processor(audios=waveform, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt")
        with self.torch.no_grad():
            emb = self.model.get_audio_features(**{k: v.to(self.device) for k, v in inputs.items()})
        vec = emb[0].cpu().numpy()
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        if len(self._audio_cache) > 256:
            self._audio_cache.clear()
        self._audio_cache[key] = vec
        return vec

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        key = tuple(texts)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached

        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        with self.torch.no_grad():
            emb = self.model.get_text_features(**{k: v.to(self.device) for k, v in inputs.items()})
        mat = emb.cpu().numpy()
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        self._text_cache[key] = mat
        return mat

    def zero_shot(self, audio_path: str | Path, label_prompts: dict[str, list[str]]) -> dict[str, float]:
        """Softmax similarity of audio against labelled prompt groups."""
        audio_vec = self.embed_audio(audio_path)
        labels = list(label_prompts.keys())
        flat_prompts: list[str] = []
        spans: list[tuple[int, int]] = []
        for label in labels:
            prompts = label_prompts[label]
            spans.append((len(flat_prompts), len(flat_prompts) + len(prompts)))
            flat_prompts.extend(prompts)

        text_mat = self.embed_texts(flat_prompts)
        sims = text_mat @ audio_vec
        label_sims = np.array([sims[a:b].max() for a, b in spans])

        # Temperature sharpens CLAP's narrow cosine range into usable probabilities.
        logits = label_sims * 25.0
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return {label: float(p) for label, p in zip(labels, probs)}

    def text_similarity(self, audio_path: str | Path, prompts: list[str]) -> float:
        """Max cosine similarity between the audio and any of the prompts (0..1-ish)."""
        audio_vec = self.embed_audio(audio_path)
        text_mat = self.embed_texts(prompts)
        return float((text_mat @ audio_vec).max())
