"""AI-generated music via Meta MusicGen, running fully locally.

Generates a bed from the MusicDirection's text prompt, then tiles it with
crossfades to cover the ad duration (MusicGen tops out around 30s/pass).
"""

from __future__ import annotations

import re
import subprocess
import threading
from pathlib import Path
from typing import Any

from ..intelligence.demographics import MusicDirection
from .base import MusicSource, make_candidate

MODEL_ID = "facebook/musicgen-small"
GEN_SECONDS = 25.0  # ~0.4s/token at 50 tokens/s frame rate; keep one pass


class MusicGenSource(MusicSource):
    id = "musicgen"
    label = "AI-generated (MusicGen, local)"
    description = "Generates an original royalty-free bed from the demographic/mood brief."

    _model = None
    _processor = None
    _lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False, "transformers/torch not installed."
        return True, "MusicGen ready (generation takes 1-4 min per variant on this machine)."

    def candidates(
        self,
        direction: MusicDirection,
        workspace: Path,
        ad_duration: float,
        limit: int = 4,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        n = min(limit, 2)  # generation is expensive; two prompt variants max
        prompts = self._build_prompts(direction)[:n]
        results = []
        for idx, prompt in enumerate(prompts, start=1):
            try:
                raw = self._generate(prompt, workspace, f"musicgen_{direction.segment_id}_{direction.mood}_{idx}")
                full = self._tile_to_duration(raw, ad_duration + 2.0)
                results.append(
                    make_candidate(
                        name=f"AI bed {idx}: {direction.mood_label}",
                        artist="MusicGen (generated)",
                        audio_path=full,
                        source=self.id,
                        license_note="AI-generated, royalty-free output",
                        prompt=prompt,
                    )
                )
            except Exception as exc:
                print(f"[musicgen] generation failed: {exc}")
        return results

    def _build_prompts(self, direction: MusicDirection) -> list[str]:
        lo, hi = direction.tempo_range
        bpm = int((lo + hi) / 2)
        base = f"{direction.prompt}, around {bpm} bpm, instrumental, professional advertising background music"
        alt = (
            f"{direction.mood_label.split('/')[0].strip().lower()} {direction.genres[0] if direction.genres else 'pop'} "
            f"instrumental, {', '.join(direction.style_keywords[:2])}, around {bpm} bpm, polished production"
        )
        return [base, alt]

    @classmethod
    def _load(cls):
        with cls._lock:
            if cls._model is None:
                from transformers import AutoProcessor, MusicgenForConditionalGeneration

                cls._processor = AutoProcessor.from_pretrained(MODEL_ID)
                cls._model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)
                cls._model.eval()
        return cls._model, cls._processor

    def _generate(self, prompt: str, workspace: Path, stem: str) -> Path:
        import torch
        import scipy.io.wavfile

        model, processor = self._load()
        inputs = processor(text=[prompt], padding=True, return_tensors="pt")
        max_new_tokens = int(GEN_SECONDS * model.config.audio_encoder.frame_rate)
        with torch.no_grad():
            audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=max_new_tokens)

        sr = model.config.audio_encoder.sampling_rate
        workspace.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_")
        out = workspace / f"{slug}.wav"
        scipy.io.wavfile.write(out, rate=sr, data=audio[0, 0].cpu().numpy())
        return out

    @staticmethod
    def _tile_to_duration(audio_path: Path, duration: float) -> Path:
        """Loop the generated bed with a short crossfade until it covers the ad."""
        if duration <= GEN_SECONDS:
            return audio_path
        out = audio_path.with_name(audio_path.stem + "_full.wav")
        loops = int(duration // GEN_SECONDS) + 1
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", str(loops), "-i", str(audio_path),
            "-t", f"{duration:.2f}",
            "-af", "afade=t=out:st=" + f"{max(0.0, duration - 1.5):.2f}" + ":d=1.5",
            str(out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Tile/loop failed: {(result.stderr or '')[-300:]}")
        return out
