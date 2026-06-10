"""Benchmark the mood-detection layer against human labels for the sample ads.

Usage:
    python -m eval.mood_benchmark           # full-mix analysis (fast)
    python -m eval.mood_benchmark --stem    # Demucs music-stem analysis (product path)

Scores top-1 ("primary mood in accepted set") and top-3 hit rates.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Accepted moods per ad (any of these as primary counts as a top-1 hit).
EXPECTATIONS: dict[str, set[str]] = {
    "Gatorade.mp4": {"hype", "epic", "uplifting"},
    "So Win. Nike.mp4": {"hype", "epic", "dark", "uplifting"},
    "MISS DIOR.mp4": {"romantic", "luxurious", "emotional", "happy"},
    "Chanel No 5.mp4": {"luxurious", "romantic", "emotional"},
    "Starbucks.mp4": {"calm", "happy", "uplifting"},
    "Starbucks UK.mp4": {"calm", "happy", "uplifting"},
    "Coke.mp4": {"happy", "uplifting", "playful"},
    "Coca-Cola Masterpiece.mp4": {"happy", "uplifting", "epic", "emotional"},
    "Kawasaki.mp4": {"hype", "dark", "epic"},
    "Calvin Klein.mp4": {"luxurious", "dark", "romantic", "hype"},
    "Google Pixel 10.mp4": {"playful", "uplifting", "happy"},
    "Olipop.mp4": {"happy", "playful", "calm"},
    "Lexus December to Remembe.mp4": {"luxurious", "epic", "uplifting", "emotional", "happy"},
}


def main() -> None:
    from sonic_segments.adapters import extract_audio_from_video
    from sonic_segments.intelligence.mood import detect_mood

    use_stem = "--stem" in sys.argv
    video_dir = Path("video")
    top1_hits = 0
    top3_hits = 0
    rows = []
    start = time.time()

    for name, accepted in EXPECTATIONS.items():
        video = video_dir / name
        if not video.exists():
            print(f"skip (missing): {name}")
            continue
        if use_stem:
            audio = _music_stem(video)
        else:
            audio = Path(extract_audio_from_video(video))
        profile = detect_mood(audio)
        top3 = [m for m, _ in profile.top(3)]
        hit1 = profile.primary in accepted
        hit3 = bool(set(top3) & accepted)
        top1_hits += hit1
        top3_hits += hit3
        rows.append((name, profile.primary, top3, sorted(accepted), hit1, hit3, profile.method))

    n = len(rows)
    print(f"\n{'ad':<28} {'primary':<11} top3{'':<26} hit1 hit3")
    for name, primary, top3, accepted, hit1, hit3, method in rows:
        print(f"{name[:27]:<28} {primary:<11} {','.join(top3):<30} {str(hit1):<5}{str(hit3):<5} [{method}]")
    print(f"\ntop-1 accuracy: {top1_hits}/{n} ({top1_hits / n:.0%})")
    print(f"top-3 accuracy: {top3_hits}/{n} ({top3_hits / n:.0%})")
    print(f"elapsed: {time.time() - start:.0f}s")


def _music_stem(video: Path) -> Path:
    """Separate and cache the music stem for one ad (the real pipeline input)."""
    from sonic_segments.adapters import (
        cleanup_demucs_output,
        extract_voiceover_source_audio,
        separate_audio,
    )

    cache_dir = Path(".cache/stems") / video.stem
    cached = cache_dir / "music.wav"
    if cached.exists():
        return cached
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        source = extract_voiceover_source_audio(video, Path(tmp) / "src.wav")
        separated = separate_audio(source, output_dir=Path(tmp) / "sep")
        import shutil

        shutil.move(str(separated["music"]), str(cached))
    cleanup_demucs_output()
    return cached


if __name__ == "__main__":
    main()
