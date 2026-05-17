"""Thin adapters over the current repo's legacy modules."""

from __future__ import annotations

import subprocess
from pathlib import Path

from audio_analyzer import analyze_audio_features, extract_audio_from_video, infer_video_genres
from voice_separator import cleanup_demucs_output, detect_speech_segments, separate_audio
from voiceover_backend import (
    get_default_voice_backend_id,
    get_voice_backend_options,
    separate_and_remix_with_backend,
)

COMMON_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def run_subprocess_checked(cmd: list[str], step_name: str, timeout: int = 600) -> None:
    """Run a command and raise a useful error when it fails."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{step_name} timed out after {timeout}s") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"{step_name} failed: {detail}")


def extract_voiceover_source_audio(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 32000,
) -> Path:
    """Extract mono PCM audio for voiceover analysis or remixing."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.suffix.lower() in COMMON_AUDIO_SUFFIXES:
        return input_path

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
        str(output_path),
    ]
    run_subprocess_checked(cmd, "Extract voiceover source audio", timeout=180)
    return output_path


def mux_audio_with_video(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    timeout: int = 900,
) -> Path:
    """Replace a video's soundtrack with the provided audio track."""
    output_path = Path(output_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(output_path),
    ]
    run_subprocess_checked(cmd, "Mux video and audio", timeout=timeout)
    return output_path


def calculate_speech_coverage(speech_segments: list[tuple[float, float]]) -> float:
    """Return the total seconds covered by narration windows."""
    return sum(max(0.0, end - start) for start, end in speech_segments)
