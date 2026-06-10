"""Audible watermarking for demo-only reference tracks.

Uncleared reference music may only be used to demo the re-score concept,
so we burn in a periodic soft tone that makes the file unusable as final
ad creative while keeping the musical idea fully auditible.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

BEEP_PERIOD_S = 7.0
BEEP_LENGTH_S = 0.22
BEEP_FREQ_HZ = 1000
BEEP_GAIN = 0.10


def watermark_audio(input_path: str | Path, output_path: str | Path) -> Path:
    """Overlay a soft periodic tone onto the track (demo-use marker)."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    beep_expr = (
        f"sin({BEEP_FREQ_HZ}*2*PI*t)*{BEEP_GAIN}*lt(mod(t\\,{BEEP_PERIOD_S})\\,{BEEP_LENGTH_S})"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-f", "lavfi", "-i", f"aevalsrc={beep_expr}:s=44100",
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[out]",
        "-map", "[out]",
        "-c:a", "mp3", "-b:a", "192k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Watermarking failed: {(result.stderr or '').strip()[-400:]}")
    return output_path
