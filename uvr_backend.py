"""SOTA vocal separation backend using UVR-family models (audio-separator).

Two-stage cascade, per current source-separation benchmarks where
BS-Roformer-class vocal models outperform Demucs for voice isolation:

1. Vocal model (BS-Roformer): split the mix into vocals vs instrumental.
2. Karaoke model on the vocals stem: split LEAD voice (the narrator) from
   BACKING vocals (e.g. sung chants in the original ad bed). The backing
   vocals are folded into the instrumental reference so downstream
   de-bleed targets them as well.

Falls back cleanly: callers catch exceptions and use the Demucs ladder.
Env switches: SONIC_UVR=0 disables, SONIC_LEAD_ISOLATION=0 skips stage 2,
SONIC_UVR_VOCAL_MODEL / SONIC_UVR_KARAOKE_MODEL override model files.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

VOCAL_MODEL = os.getenv("SONIC_UVR_VOCAL_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
KARAOKE_MODEL = os.getenv("SONIC_UVR_KARAOKE_MODEL", "UVR_MDXNET_KARA_2.onnx")

import re
import tempfile

WORK_ROOT = Path(tempfile.gettempdir()) / "uvr_work"

_separator_cache: dict[str, tuple] = {}


def is_available() -> bool:
    try:
        import audio_separator  # noqa: F401

        return True
    except ImportError:
        return False


def _get_separator(model_filename: str):
    """Separator instances honor output_dir only at construction time, so
    each model gets a fixed work dir; results are globbed and moved out."""
    cached = _separator_cache.get(model_filename)
    if cached is None:
        from audio_separator.separator import Separator

        slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_filename)[:40]
        work_dir = WORK_ROOT / slug
        work_dir.mkdir(parents=True, exist_ok=True)
        sep = Separator(log_level=logging.WARNING, output_format="WAV", output_dir=str(work_dir))
        sep.load_model(model_filename=model_filename)
        cached = (sep, work_dir)
        _separator_cache[model_filename] = cached
    return cached


def _run_model(model_filename: str, audio_path: Path, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sep, work_dir = _get_separator(model_filename)
    for stale in work_dir.glob("*"):
        stale.unlink(missing_ok=True)

    sep.separate(str(audio_path))

    found: dict[str, Path] = {}
    for produced in work_dir.glob("*"):
        lower = produced.name.lower()
        key = "vocals" if "(vocals)" in lower else "instrumental" if "(instrumental)" in lower else None
        if key:
            target = out_dir / f"{key}.wav"
            shutil.move(str(produced), str(target))
            found[key] = target
    if "vocals" not in found or "instrumental" not in found:
        raise RuntimeError(f"{model_filename} produced unexpected outputs: {sorted(found)}")
    return found


def separate_with_uvr(audio_path, output_dir, lead_isolation: bool = True) -> dict:
    """Return {'vocals','music','model_used'} matching the Demucs contract."""
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"UVR stage 1: {VOCAL_MODEL} (vocals vs instrumental)...")
    stage1 = _run_model(VOCAL_MODEL, audio_path, output_dir / "uvr_stage1")
    vocals, instrumental = stage1["vocals"], stage1["instrumental"]

    final_vocals = output_dir / "vocals.wav"
    final_music = output_dir / "music.wav"
    model_used = VOCAL_MODEL

    if lead_isolation:
        try:
            print(f"UVR stage 2: {KARAOKE_MODEL} (lead narrator vs backing vocals)...")
            stage2 = _run_model(KARAOKE_MODEL, vocals, output_dir / "uvr_stage2")
            shutil.copyfile(stage2["vocals"], final_vocals)
            # Fold backing vocals (chants/adlibs) into the accompaniment
            # reference so the de-bleed stage can suppress their residue.
            cmd = [
                "ffmpeg", "-y",
                "-i", str(instrumental),
                "-i", str(stage2["instrumental"]),
                "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0",
                str(final_music),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError((result.stderr or "")[-300:])
            return {"vocals": final_vocals, "music": final_music,
                    "model_used": f"{VOCAL_MODEL} + {KARAOKE_MODEL}"}
        except Exception as exc:
            print(f"UVR lead isolation failed ({exc}); using full vocal stem.")

    shutil.copyfile(vocals, final_vocals)
    shutil.copyfile(instrumental, final_music)
    return {"vocals": final_vocals, "music": final_music, "model_used": model_used}
