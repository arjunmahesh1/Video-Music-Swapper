"""Dataset utilities for Stage 1 separator training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Stage1ManifestEntry:
    dialogue: Path
    music: Path
    effects: Path
    mixture: Path | None = None
    weight: float = 1.0


def load_manifest(manifest_path: str | Path) -> list[Stage1ManifestEntry]:
    manifest_path = Path(manifest_path)
    entries: list[Stage1ManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            entries.append(
                Stage1ManifestEntry(
                    dialogue=Path(raw["dialogue"]),
                    music=Path(raw["music"]),
                    effects=Path(raw["effects"]),
                    mixture=Path(raw["mixture"]) if raw.get("mixture") else None,
                    weight=float(raw.get("weight", 1.0)),
                )
            )
    return entries


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return np.asarray(audio, dtype=np.float32)


def _crop_or_pad(audio: np.ndarray, start: int, target_len: int) -> np.ndarray:
    if len(audio) < target_len:
        padded = np.zeros(target_len, dtype=np.float32)
        padded[: len(audio)] = audio
        return padded
    return audio[start:start + target_len]


def _pad_to_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) >= target_len:
        return audio[:target_len]
    padded = np.zeros(target_len, dtype=np.float32)
    padded[: len(audio)] = audio
    return padded


class Stage1StemDataset(Dataset):
    """Loads fixed-length training chunks from a JSONL manifest."""

    def __init__(
        self,
        manifest_entries: list[Stage1ManifestEntry],
        sample_rate: int = 32000,
        chunk_seconds: float = 6.0,
        random_crop: bool = True,
    ):
        self.entries = manifest_entries
        self.sample_rate = sample_rate
        self.chunk_len = int(sample_rate * chunk_seconds)
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self.entries[index]
        dialogue = _load_audio(entry.dialogue, self.sample_rate)
        music = _load_audio(entry.music, self.sample_rate)
        effects = _load_audio(entry.effects, self.sample_rate)

        total_len = max(len(dialogue), len(music), len(effects))
        if entry.mixture:
            mixture = _load_audio(entry.mixture, self.sample_rate)
            total_len = max(total_len, len(mixture))
        else:
            dialogue_mix = _pad_to_length(dialogue, total_len)
            music_mix = _pad_to_length(music, total_len)
            effects_mix = _pad_to_length(effects, total_len)
            mixture = dialogue_mix + music_mix + effects_mix

        if total_len > self.chunk_len and self.random_crop:
            start = random.randint(0, total_len - self.chunk_len)
        else:
            start = 0

        dialogue = _crop_or_pad(dialogue, start, self.chunk_len)
        music = _crop_or_pad(music, start, self.chunk_len)
        effects = _crop_or_pad(effects, start, self.chunk_len)
        mixture = _crop_or_pad(mixture, start, self.chunk_len)

        targets = np.stack([dialogue, music, effects], axis=0)
        return {
            "mixture": torch.from_numpy(mixture),
            "targets": torch.from_numpy(targets),
            "weight": torch.tensor(entry.weight, dtype=torch.float32),
        }
