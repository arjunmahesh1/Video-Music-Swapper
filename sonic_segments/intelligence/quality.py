"""Probe ingest sources so we can flag audio too compressed to re-score well."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SourceQuality:
    ok: bool
    codec: str | None = None
    sample_rate: int | None = None
    bit_rate: int | None = None
    channels: int | None = None
    duration: float | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "codec": self.codec,
            "sample_rate": self.sample_rate,
            "bit_rate": self.bit_rate,
            "channels": self.channels,
            "duration": self.duration,
            "warnings": self.warnings,
        }


def probe_source_quality(media_path: str | Path) -> SourceQuality:
    """ffprobe the first audio stream and flag compression red-flags."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,bit_rate,channels",
        "-show_entries", "format=duration,bit_rate",
        "-of", "json", str(media_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        info = json.loads(result.stdout or "{}")
    except Exception as exc:
        return SourceQuality(ok=False, warnings=[f"Could not probe source: {exc}"])

    streams = info.get("streams") or []
    if not streams:
        return SourceQuality(ok=False, warnings=["No audio stream found in the source."])

    stream = streams[0]
    fmt = info.get("format", {})
    codec = stream.get("codec_name")
    sample_rate = int(stream.get("sample_rate") or 0) or None
    bit_rate = int(stream.get("bit_rate") or 0) or None
    duration = float(fmt.get("duration") or 0) or None
    channels = int(stream.get("channels") or 0) or None

    warnings: list[str] = []
    if sample_rate and sample_rate < 32000:
        warnings.append(
            f"Audio sample rate is {sample_rate} Hz (below 32 kHz) — separation quality will suffer; "
            "upload the original file if you have it."
        )
    if bit_rate and bit_rate < 96_000:
        warnings.append(
            f"Audio bitrate is only {bit_rate // 1000} kbps — this pull looks heavily compressed; "
            "the re-score will work but artifacts are likely. Prefer a direct file upload."
        )
    if duration and duration > 180:
        warnings.append("Source is longer than 3 minutes — ads are typically 6-90s; processing will be slow.")

    return SourceQuality(
        ok=True,
        codec=codec,
        sample_rate=sample_rate,
        bit_rate=bit_rate,
        channels=channels,
        duration=duration,
        warnings=warnings,
    )
