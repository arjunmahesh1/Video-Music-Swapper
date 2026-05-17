"""Inference helpers for the Stage 1 dialogue backend."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.io import wavfile

from .model import Stage1DialogueSeparator, Stage1SeparatorConfig


COMMON_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def default_checkpoint_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "models" / "stage1_dialogue" / "best.pt"


def stage1_checkpoint_available(checkpoint_path: str | Path | None = None) -> bool:
    return Path(checkpoint_path or default_checkpoint_path()).exists()


def load_stage1_model(
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> tuple[Stage1DialogueSeparator, Stage1SeparatorConfig, torch.device]:
    checkpoint_path = Path(checkpoint_path or default_checkpoint_path())
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at {checkpoint_path}. "
            "Train it first or set STAGE1_VOICEOVER_CHECKPOINT."
        )

    payload = torch.load(checkpoint_path, map_location="cpu")
    config = Stage1SeparatorConfig.from_dict(payload["config"])
    model = Stage1DialogueSeparator(config)
    model.load_state_dict(payload["model_state"])

    if device:
        torch_device = torch.device(device)
    else:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(torch_device)
    model.eval()
    return model, config, torch_device


def _extract_audio_for_inference(input_path: Path, sample_rate: int) -> Path:
    if input_path.suffix.lower() in COMMON_AUDIO_SUFFIXES:
        return input_path

    wav_path = Path(tempfile.gettempdir()) / "stage1_input_audio.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffmpeg extract failed: {detail}")
    return wav_path


def _save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    wavfile.write(str(path), sample_rate, np.int16(audio * 32767))


def _compute_chunk_starts(total_len: int, chunk_len: int, hop_len: int) -> list[int]:
    if total_len <= chunk_len:
        return [0]

    starts = list(range(0, total_len - chunk_len + 1, hop_len))
    last_start = total_len - chunk_len
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


@torch.inference_mode()
def run_stage1_separation(
    input_path: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    chunk_seconds: float | None = None,
    overlap: float = 0.5,
) -> dict[str, Path]:
    model, config, torch_device = load_stage1_model(checkpoint_path=checkpoint_path, device=device)
    audio_path = _extract_audio_for_inference(Path(input_path), config.sample_rate)
    waveform, _ = librosa.load(str(audio_path), sr=config.sample_rate, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)

    chunk_seconds = chunk_seconds or config.chunk_seconds
    chunk_len = int(config.sample_rate * chunk_seconds)
    hop_len = max(1, int(chunk_len * (1.0 - overlap)))
    starts = _compute_chunk_starts(len(waveform), chunk_len, hop_len)

    accum = np.zeros((len(config.targets), len(waveform)), dtype=np.float32)
    weights = np.zeros(len(waveform), dtype=np.float32)
    hann = np.hanning(chunk_len).astype(np.float32)
    if not np.any(hann):
        hann = np.ones(chunk_len, dtype=np.float32)

    for start in starts:
        end = min(len(waveform), start + chunk_len)
        chunk = np.zeros(chunk_len, dtype=np.float32)
        current = waveform[start:end]
        chunk[: len(current)] = current

        tensor = torch.from_numpy(chunk).unsqueeze(0).to(torch_device)
        prediction = model(tensor)["stems"][0].detach().cpu().numpy()

        if len(current) < chunk_len:
            window = hann.copy()
            window[len(current):] = 0.0
        else:
            window = hann

        accum[:, start:end] += prediction[:, : len(current)] * window[: len(current)]
        weights[start:end] += window[: len(current)]

    weights = np.maximum(weights, 1e-6)
    stems = accum / weights[None, :]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_paths: dict[str, Path] = {}
    for idx, target_name in enumerate(config.targets):
        stem_path = output_dir / f"{target_name}.wav"
        _save_audio(stem_path, stems[idx], config.sample_rate)
        stem_paths[target_name] = stem_path
    return stem_paths
