"""Benchmark swapped outputs for voice retention and music leakage.

This is for model development and offline evaluation, not the runtime app path.

Example:
    python voiceover_benchmark.py ^
      --original video/Gatorade.mp4 ^
      --candidate output/Gatorade_swapped.mp4 ^
      --transcript transcript.txt
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, correlate, sosfiltfilt


TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)")


@dataclass
class BenchmarkReport:
    duration_seconds: float
    mix_rms_dbfs: float
    peak_dbfs: float
    jumps_gt12db: int
    hf_ratio_ge6k: float
    voice_retention_db: float | None
    music_leak_corr: float | None
    transcript_coverage_seconds: float | None


def _run_ffmpeg_extract(input_path: Path, output_wav: Path, sr: int = 32000) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ac", "1",
        str(output_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffmpeg extract failed: {detail}")


def _load_mono_f32(path: Path) -> tuple[int, np.ndarray]:
    sr, y = wavfile.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    return sr, y.astype(np.float32) / 32768.0


def _timestamp_to_seconds(token: str) -> float | None:
    parts = token.strip().split(":")
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return None


def _parse_transcript_segments(text: str, duration_seconds: float) -> list[tuple[float, float]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    pairs: list[tuple[float, str]] = []
    used = set()

    for idx, line in enumerate(lines):
        match = TIMESTAMP_PATTERN.search(line)
        if not match:
            continue
        ts = _timestamp_to_seconds(match.group(1))
        if ts is None:
            continue
        before = line[:match.start()].strip(" -\t")
        after = line[match.end():].strip(" -\t")
        text_part = after if after else before
        if text_part:
            pairs.append((ts, text_part))
            used.add(idx)

    for idx, line in enumerate(lines):
        if idx in used:
            continue
        ts = _timestamp_to_seconds(line)
        if ts is None:
            continue
        text_part = ""
        if idx - 1 >= 0 and (idx - 1) not in used:
            prev = lines[idx - 1]
            if _timestamp_to_seconds(prev) is None and not TIMESTAMP_PATTERN.search(prev):
                text_part = prev
                used.add(idx - 1)
        if not text_part and idx + 1 < len(lines) and (idx + 1) not in used:
            nxt = lines[idx + 1]
            if _timestamp_to_seconds(nxt) is None and not TIMESTAMP_PATTERN.search(nxt):
                text_part = nxt
                used.add(idx + 1)
        if text_part:
            pairs.append((ts, text_part))
            used.add(idx)

    pairs.sort(key=lambda item: item[0])
    segments: list[tuple[float, float]] = []
    for idx, (start, text_part) in enumerate(pairs):
        words = max(1, len(re.findall(r"[A-Za-z']+", text_part)))
        est = min(3.2, max(0.55, 0.28 * words))
        if idx + 1 < len(pairs):
            next_start = pairs[idx + 1][0]
            end = min(max(start + 0.25, next_start - 0.05), start + est)
        else:
            end = min(duration_seconds, start + est)
        start = min(max(0.0, start - 0.01), duration_seconds)
        end = min(duration_seconds, max(start + 0.25, end))
        if end > start + 0.08:
            segments.append((start, end))
    return segments


def _segments_to_mask(
    segments: list[tuple[float, float]], sample_rate: int, length: int
) -> np.ndarray:
    mask = np.zeros(length, dtype=np.float32)
    for start, end in segments:
        a = max(0, int(start * sample_rate))
        b = min(length, int(end * sample_rate))
        if b > a:
            mask[a:b] = 1.0
    return mask


def _max_corr(x: np.ndarray, y: np.ndarray, sr: int, max_lag_seconds: float = 0.75) -> float:
    n = min(len(x), len(y))
    if n < 4096:
        return 0.0

    x = x[:n]
    y = y[:n]
    window = min(n, int(24 * sr))
    start = max(0, (n - window) // 2)
    x = x[start:start + window]
    y = y[start:start + window]

    decim = 4
    x = x[::decim] - np.mean(x[::decim])
    y = y[::decim] - np.mean(y[::decim])
    sr_d = max(1, sr // decim)
    denom = float(np.sqrt(np.sum(x ** 2) * np.sum(y ** 2)) + 1e-12)
    if denom <= 1e-10:
        return 0.0

    corr = correlate(x, y, mode="full", method="fft")
    lag = int(max_lag_seconds * sr_d)
    mid = len(corr) // 2
    corr = corr[mid - lag:mid + lag + 1]
    return float(np.max(np.abs(corr)) / denom)


def benchmark(
    original_path: Path,
    candidate_path: Path,
    transcript_text: str | None = None,
    music_stem_path: Path | None = None,
) -> BenchmarkReport:
    tmp_dir = Path(tempfile.gettempdir()) / "voiceover_benchmark"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    original_wav = tmp_dir / "original.wav"
    candidate_wav = tmp_dir / "candidate.wav"
    _run_ffmpeg_extract(original_path, original_wav)
    _run_ffmpeg_extract(candidate_path, candidate_wav)

    sr_o, y_o = _load_mono_f32(original_wav)
    sr_c, y_c = _load_mono_f32(candidate_wav)
    n = min(len(y_o), len(y_c))
    y_o = y_o[:n]
    y_c = y_c[:n]

    duration_seconds = n / sr_c
    rms = float(np.sqrt(np.mean(y_c ** 2) + 1e-12))
    peak = float(np.max(np.abs(y_c)) + 1e-12)

    frame = int(0.02 * sr_c)
    levels = []
    for i in range(0, max(0, len(y_c) - frame), frame):
        seg = y_c[i:i + frame]
        levels.append(20 * np.log10(np.sqrt(np.mean(seg ** 2) + 1e-12)))
    levels_arr = np.array(levels, dtype=np.float32) if levels else np.zeros(1, dtype=np.float32)
    jumps_gt12db = int(np.sum(np.abs(np.diff(levels_arr)) > 12.0))

    yf = np.fft.rfft(y_c)
    ff = np.fft.rfftfreq(len(y_c), d=1 / sr_c)
    power = np.abs(yf) ** 2 + 1e-18
    hf_ratio = float(power[ff >= 6000].sum() / power.sum())

    voice_retention_db = None
    transcript_coverage_seconds = None
    if transcript_text:
        segments = _parse_transcript_segments(transcript_text, duration_seconds)
        if segments:
            mask = _segments_to_mask(segments, sr_c, len(y_c))
            sos = butter(6, [120 / (sr_c * 0.5), 5000 / (sr_c * 0.5)], btype="bandpass", output="sos")
            y_ob = sosfiltfilt(sos, y_o)
            y_cb = sosfiltfilt(sos, y_c)
            rms_o = float(np.sqrt(np.mean((y_ob * mask) ** 2) + 1e-12))
            rms_c = float(np.sqrt(np.mean((y_cb * mask) ** 2) + 1e-12))
            voice_retention_db = 20 * np.log10((rms_c + 1e-12) / (rms_o + 1e-12))
            transcript_coverage_seconds = float(sum(end - start for start, end in segments))

    music_leak_corr = None
    if music_stem_path:
        music_wav = tmp_dir / "music.wav"
        _run_ffmpeg_extract(music_stem_path, music_wav)
        sr_m, y_m = _load_mono_f32(music_wav)
        m = min(len(y_c), len(y_m))
        y_cb = y_c[:m]
        y_mb = y_m[:m]
        sos = butter(6, [120 / (sr_c * 0.5), 5000 / (sr_c * 0.5)], btype="bandpass", output="sos")
        y_cb = sosfiltfilt(sos, y_cb)
        y_mb = sosfiltfilt(sos, y_mb)
        music_leak_corr = _max_corr(y_cb, y_mb, sr_m)

    return BenchmarkReport(
        duration_seconds=duration_seconds,
        mix_rms_dbfs=20 * np.log10(rms),
        peak_dbfs=20 * np.log10(peak),
        jumps_gt12db=jumps_gt12db,
        hf_ratio_ge6k=hf_ratio,
        voice_retention_db=voice_retention_db,
        music_leak_corr=music_leak_corr,
        transcript_coverage_seconds=transcript_coverage_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark swapped voiceover quality.")
    parser.add_argument("--original", required=True, help="Original source video/audio path")
    parser.add_argument("--candidate", required=True, help="Swapped output video/audio path")
    parser.add_argument("--transcript", help="Optional transcript text file with timestamps")
    parser.add_argument("--music-stem", help="Optional original music stem path for leakage scoring")
    parser.add_argument("--json-out", help="Optional JSON output path")
    args = parser.parse_args()

    transcript_text = None
    if args.transcript:
        transcript_text = Path(args.transcript).read_text(encoding="utf-8")

    report = benchmark(
        original_path=Path(args.original),
        candidate_path=Path(args.candidate),
        transcript_text=transcript_text,
        music_stem_path=Path(args.music_stem) if args.music_stem else None,
    )

    report_json = json.dumps(asdict(report), indent=2)
    print(report_json)

    if args.json_out:
        Path(args.json_out).write_text(report_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
