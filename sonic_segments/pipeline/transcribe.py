"""Auto-derive a timestamped narration transcript from the ad itself.

Same tech as YouTube captions (Whisper-class ASR on the full mix). The
resulting `M:SS text` lines feed the renderer's transcript-guided path,
which produces far more reliable speech windows than stem-based detection
(Gatorade: 4.1s detected acoustically vs 14.1s transcript-guided).
"""

from __future__ import annotations

from pathlib import Path


WORD_MERGE_GAP_S = 0.8


def auto_speech_windows(audio_path: str | Path) -> tuple[str | None, list[tuple[float, float]]]:
    """Transcribe narration and return (transcript text, exact word-level windows).

    Word timestamps give true starts AND ends, so long narration lines with
    pauses don't get truncated by end-time estimation. Words separated by
    less than WORD_MERGE_GAP_S merge into one window.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None, []

    try:
        from voice_separator import is_likely_music_lyrics
    except Exception:
        is_likely_music_lyrics = lambda text: False

    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _info = model.transcribe(
            str(audio_path),
            vad_filter=True,
            # Sensitive VAD: whispered taglines ("Is it in you?") sit below
            # the default threshold; lyric filtering handles the extra recall.
            vad_parameters=dict(threshold=0.2, min_silence_duration_ms=250),
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
        )

        lines: list[str] = []
        words: list[tuple[float, float]] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if len(text) < 3 or is_likely_music_lyrics(text):
                continue
            minutes, seconds = int(seg.start // 60), int(seg.start % 60)
            lines.append(f"{minutes}:{seconds:02d} {text}")
            for word in seg.words or []:
                words.append((float(word.start), float(word.end)))

        windows: list[tuple[float, float]] = []
        for start, end in sorted(words):
            if windows and start - windows[-1][1] <= WORD_MERGE_GAP_S:
                windows[-1] = (windows[-1][0], max(windows[-1][1], end))
            else:
                windows.append((start, end))

        return ("\n".join(lines) if lines else None), windows
    except Exception as exc:
        print(f"[transcribe] auto transcript failed: {exc}")
        return None, []


def auto_transcript(audio_path: str | Path) -> str | None:
    """Back-compat wrapper returning only the transcript text."""
    text, _windows = auto_speech_windows(audio_path)
    return text
