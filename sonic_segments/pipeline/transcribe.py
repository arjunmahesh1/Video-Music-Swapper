"""Auto-derive a timestamped narration transcript from the ad itself.

Same tech as YouTube captions (Whisper-class ASR on the full mix). The
resulting `M:SS text` lines feed the renderer's transcript-guided path,
which produces far more reliable speech windows than stem-based detection
(Gatorade: 4.1s detected acoustically vs 14.1s transcript-guided).
"""

from __future__ import annotations

from pathlib import Path


def auto_transcript(audio_path: str | Path) -> str | None:
    """Transcribe narration to `M:SS text` lines; None when nothing usable."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None

    try:
        from voice_separator import is_likely_music_lyrics
    except Exception:
        is_likely_music_lyrics = lambda text: False

    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _info = model.transcribe(
            str(audio_path),
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,
        )
        lines: list[str] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if len(text) < 3:
                continue
            if is_likely_music_lyrics(text):
                continue
            minutes, seconds = int(seg.start // 60), int(seg.start % 60)
            lines.append(f"{minutes}:{seconds:02d} {text}")
        return "\n".join(lines) if lines else None
    except Exception as exc:
        print(f"[transcribe] auto transcript failed: {exc}")
        return None
