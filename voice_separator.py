"""Voice/music separation and smart audio ducking for ads."""
import subprocess
import tempfile
from pathlib import Path
import shutil
import sys
import os
import re
import numpy as np
import torch
import librosa
from faster_whisper import WhisperModel


# Models to try in order of memory efficiency (quantized = smaller)
DEMUCS_MODELS = [
    ("mdx_q", "mdx_q"),           # Quantized hybrid - smallest memory footprint
    ("mdx_extra_q", "mdx_extra_q"), # Quantized with extra training
    ("mdx", "mdx"),               # Hybrid Transformer - less memory than htdemucs
    ("htdemucs", "htdemucs"),     # Default - highest quality but most memory
]

SEGMENT_MERGE_GAP_SECONDS = 0.22
SHORT_BURST_SECONDS = 0.35
ISOLATED_BURST_GAP_SECONDS = 0.70
MAX_SEGMENTS_BEFORE_CONSOLIDATION = 180
MAX_SEGMENTS_FOR_EXPRESSION = 260


def _run_checked(cmd, timeout=None, env=None, step_name="Command"):
    """Run subprocess and raise a detailed exception on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
    except subprocess.TimeoutExpired:
        raise Exception(f"{step_name} timed out after {timeout}s")

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr if stderr else stdout
        raise Exception(f"{step_name} failed: {details}")
    return result


def _probe_duration_seconds(audio_path):
    """Return media duration in seconds using ffprobe; None if unavailable."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return None
        return float(result.stdout.strip())
    except Exception:
        return None


def _merge_segments(segments, max_gap=SEGMENT_MERGE_GAP_SECONDS):
    """Merge adjacent segments separated by <= max_gap."""
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segments[0]]

    for start, end in sorted_segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _filter_short_isolated_segments(
    segments,
    min_duration=0.18,
    short_duration=SHORT_BURST_SECONDS,
    isolation_gap=ISOLATED_BURST_GAP_SECONDS
):
    """Remove very short isolated bursts that are usually music bleed artifacts."""
    if not segments:
        return []

    filtered = []
    for i, (start, end) in enumerate(segments):
        duration = end - start
        if duration < min_duration:
            continue

        if duration <= short_duration:
            prev_gap = start - segments[i - 1][1] if i > 0 else 999.0
            next_gap = segments[i + 1][0] - end if i + 1 < len(segments) else 999.0
            if prev_gap > isolation_gap and next_gap > isolation_gap:
                continue

        filtered.append((start, end))

    return filtered


def _is_probable_speech_text(text):
    """Fast textual gate to drop ad-libs/non-verbal fragments."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return False

    filler_words = {
        "yeah", "ayy", "uh", "oh", "ooh", "ah", "ahh", "huh", "nah", "yah",
        "woah", "whoa", "la", "na", "mm", "hm", "eh", "hey"
    }

    # Single filler token is almost always non-speech for our use case.
    if len(words) == 1 and words[0] in filler_words:
        return False

    # Tiny repeated filler snippets like "yeah yeah" should be rejected.
    if len(words) <= 3 and all(w in filler_words for w in words):
        return False

    return True


def separate_audio(audio_path, output_dir=None, model=None):
    """Separate audio into vocals and accompaniment using Demucs.

    Tries quantized/lighter models first to minimize memory usage.
    Falls back to heavier models if needed.

    Args:
        audio_path: Path to audio file (wav, mp3, etc.)
        output_dir: Directory for output files (uses temp dir if None)
        model: Specific model to use (None = auto-select lightest working model)

    Returns:
        Dict with paths: {'vocals': path, 'music': path}
    """
    audio_path = Path(audio_path)

    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "demucs_output"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    duration_seconds = _probe_duration_seconds(audio_path)

    # Use the same Python executable that's running this script
    python_exe = sys.executable

    # Set environment variables
    env = os.environ.copy()

    # Check if CUDA GPU is available
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)} - using GPU acceleration")
        env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"  # Reduce GPU memory caching
        # Don't set CUDA_VISIBLE_DEVICES - let it use GPU
    else:
        print("No GPU detected - using CPU (slower)")
        env["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only mode

    # If specific model requested, only try that one
    models_to_try = [(model, model)] if model else DEMUCS_MODELS

    last_error = None
    for model_name, model_id in models_to_try:
        print(f"Trying Demucs model: {model_name}...")
        print(f"Using Python: {python_exe}")

        # Build command - segment size balances memory vs speed
        # Larger segments = faster but more RAM.
        if torch.cuda.is_available():
            segment_size = "20"
        elif duration_seconds and duration_seconds <= 120:
            segment_size = "12"
        else:
            segment_size = "10"

        cmd = [
            python_exe, "-m", "demucs",
            "-v",                     # Verbose output
            "-n", model_id,           # Specify model
            "--two-stems=vocals",     # Only separate vocals/accompaniment
            "--segment", segment_size,
            "-o", str(output_dir),
            str(audio_path)
        ]

        print(f"Running: {' '.join(cmd)}")

        # Run with real-time output to see what's happening
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        # Dynamic timeout based on duration (capped to keep app responsive).
        if duration_seconds:
            demucs_timeout = int(min(900, max(240, duration_seconds * 2.5)))
        else:
            demucs_timeout = 600

        try:
            stdout_data, stderr_data = process.communicate(timeout=demucs_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout_data, stderr_data = process.communicate()
            raise Exception(
                f"Demucs timed out after {demucs_timeout}s."
                " Try a shorter clip or disable voice preservation."
            )

        # Create a result-like object
        class Result:
            pass
        result = Result()
        result.returncode = process.returncode
        result.stdout = stdout_data
        result.stderr = stderr_data

        # Combine stdout and stderr for full error context
        full_output = f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"

        if result.returncode == 0:
            # Success! Find the output files
            stem_name = audio_path.stem
            model_output = output_dir / model_id / stem_name

            vocals_path = model_output / "vocals.wav"
            music_path = model_output / "no_vocals.wav"

            if vocals_path.exists() and music_path.exists():
                print(f"Separation complete using {model_name}!")
                return {
                    'vocals': vocals_path,
                    'music': music_path,
                    'model_used': model_name
                }
            else:
                last_error = f"Output files not found at {model_output}"
                print(f"Warning: {last_error}")
        else:
            # Capture full error from both streams
            last_error = full_output
            print(f"Model {model_name} failed.")
            print(f"=== FULL OUTPUT ===")
            print(full_output)
            print(f"=== END OUTPUT ===")

            # Check if it's a memory error - if so, try next model
            combined_lower = full_output.lower()
            if "paging file" in combined_lower or "out of memory" in combined_lower or "memoryerror" in combined_lower:
                print("Memory error detected, trying lighter model...")
                continue
            # For other errors, also try next model
            continue

    # All models failed
    raise Exception(f"Demucs separation failed with all models. Last error: {last_error}")


def is_singing_not_speech(audio_segment, sr=16000, segment_duration=None):
    """Analyze if an audio segment is singing rather than speech.

    Uses adaptive thresholds:
    - Short segments (< 2s): More aggressive filtering (2/3 indicators) to catch bursts
    - Long segments (>= 2s): Conservative filtering (3/3) to preserve voiceover

    Args:
        audio_segment: Audio data (numpy array or torch tensor)
        sr: Sample rate
        segment_duration: Duration in seconds (if None, calculated from audio length)

    Returns:
        True if likely singing, False if likely speech
    """
    try:
        # Convert to numpy if it's a torch tensor
        if isinstance(audio_segment, torch.Tensor):
            audio_np = audio_segment.cpu().numpy()
        else:
            audio_np = audio_segment

        # Ensure float32
        audio_np = audio_np.astype(np.float32)

        # Calculate duration if not provided
        if segment_duration is None:
            segment_duration = len(audio_np) / sr

        # Calculate spectral features
        # 1. Spectral centroid - singing is usually brighter (higher frequency content)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_np, sr=sr)[0]
        mean_centroid = np.mean(spectral_centroid)

        # 2. Zero crossing rate - speech has more abrupt changes
        zcr = librosa.feature.zero_crossing_rate(audio_np)[0]
        mean_zcr = np.mean(zcr)

        # 3. Spectral flatness - speech is noisier (higher flatness)
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_np)[0]
        mean_flatness = np.mean(spectral_flatness)

        # Decision thresholds based on segment length
        # Short segments: likely music burps, use stricter thresholds
        # Long segments: likely voiceover, use conservative thresholds
        is_short_segment = segment_duration < 2.0

        if is_short_segment:
            # For short bursts: aggressive filtering to catch music burps, need 2/3
            is_high_centroid = mean_centroid > 2400  # Very bright
            is_low_zcr = mean_zcr < 0.08  # Very sustained
            is_low_flatness = mean_flatness < 0.025  # Very tonal
            required_score = 2  # Need 2 out of 3
        else:
            # For longer segments: MUST be very conservative to preserve narration
            # Voiceover narration: centroid ~1500-2000, zcr ~0.13-0.16, flatness ~0.02-0.04
            # Singing: centroid >2500, zcr <0.10, flatness <0.02
            # Require ALL 3 indicators to be very sure it's singing before filtering
            is_high_centroid = mean_centroid > 2500  # Significantly brighter than speech
            is_low_zcr = mean_zcr < 0.10  # More sustained than speech
            is_low_flatness = mean_flatness < 0.020  # More tonal than speech
            required_score = 3  # Need ALL 3 to filter (very conservative)

        singing_score = sum([is_high_centroid, is_low_zcr, is_low_flatness])

        # Debug output
        segment_type = "short burst" if is_short_segment else "long segment"
        print(f"  Spectral analysis ({segment_type}, {segment_duration:.1f}s): centroid={mean_centroid:.0f}Hz, zcr={mean_zcr:.3f}, flatness={mean_flatness:.3f}, score={singing_score}/{required_score}")

        # Filter if score meets threshold
        return singing_score >= required_score

    except Exception as e:
        print(f"  Warning: Spectral analysis failed: {e}, assuming speech")
        return False  # If analysis fails, assume it's speech (conservative)


def is_likely_advertising_voiceover(text):
    """Check if text contains advertising/commercial language patterns.

    Returns True if text appears to be advertising copy, False otherwise.
    """
    text_lower = text.lower().strip()

    # Commercial/advertising phrases
    ad_phrases = [
        'sales event', 'limited time', 'offer', 'financing', 'lease',
        'msrp', 'dealer', 'dealership', 'test drive', 'inventory',
        'see dealer', 'contact', 'visit us', 'call now', 'hurry',
        'ends soon', 'while supplies last', 'select models', 'terms apply',
        'learn more', 'details at', 'restrictions apply', 'warranty',
        'apr', 'down payment', 'per month', 'brand new', 'certified pre-owned',
    ]

    for phrase in ad_phrases:
        if phrase in text_lower:
            return True

    return False


def is_likely_music_lyrics(text):
    """Check if transcribed text is likely song lyrics rather than speech.

    Returns True if text appears to be music/singing, False if likely speech.
    """
    text_lower = text.lower().strip()

    # If it's advertising copy, it's NOT music lyrics
    if is_likely_advertising_voiceover(text):
        return False

    # Common music patterns - repeated syllables, ad-libs, vocalizations
    music_patterns = [
        # Trap/Hip-hop ad-libs and vocalizations
        'yeah', 'ayy', 'uh', 'oh', 'ooh', 'ahh', 'whoa', 'woah',
        'skrrt', 'brr', 'grr', 'yuh', 'huh', 'nah', 'yah',
        'la la', 'na na', 'da da', 'ba ba', 'sha la',
        # Common lyric fragments
        'fein', 'shawty', 'baby', 'bae', 'shorty',
        # Repetitive patterns
        'yeah yeah', 'no no', 'go go', 'hey hey',
    ]

    # Check for exact matches to common ad-libs
    if text_lower in music_patterns:
        return True

    # Check for very short utterances (likely ad-libs)
    if len(text_lower) <= 3 and text_lower in ['uh', 'oh', 'ah', 'mm', 'hm', 'eh']:
        return True

    # Remove punctuation and split into words for pattern analysis
    # This handles "rush," vs "rush" and "lot," vs "lot"
    text_clean = re.sub(r'[^\w\s]', '', text_lower)  # Remove all punctuation
    words = text_clean.split()

    if len(words) >= 2:
        # If 70%+ of words are the same, likely music
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        max_repeat = max(word_counts.values())
        if max_repeat / len(words) > 0.7:
            return True

        # Check for immediate word repetition (e.g., "I rush, I rush" or "rush rush")
        # Now works even with punctuation because we cleaned it
        for i in range(len(words) - 1):
            # Skip common filler words
            if words[i] not in ['the', 'a', 'an', 'and', 'or', 'but', 'to', 'of', 'in', 'i', 'you', 'we']:
                if words[i] == words[i + 1]:
                    print(f"    → Detected immediate repetition: '{words[i]} {words[i+1]}'")
                    return True

        # Check for A-B-A patterns (e.g., "a lot, a dream, a lot")
        # Very common in song lyrics, rare in speech
        if len(words) >= 5:
            for i in range(len(words) - 4):
                # Check if word at position i appears again within next 4 words
                # Excluding very common words that legitimately repeat
                if words[i] not in ['the', 'a', 'an', 'and', 'or', 'but', 'to', 'of', 'in', 'is', 'it', 'i', 'you', 'we', 'my', 'your']:
                    for j in range(i + 2, min(i + 5, len(words))):
                        if words[i] == words[j]:
                            print(f"    → Detected A-B-A pattern: word '{words[i]}' at positions {i} and {j}")
                            return True

    # Check for lyrical/poetic phrases (not conversational)
    # Add more Sweet Disposition specific patterns
    lyrical_phrases = [
        'wanna dream', 'on the momentum', 'dream for life',
        'in my heart', 'in my soul', 'feel the beat',
        'all night long', 'all day long', 'tonight',
        'let\'s go', 'lets go',  # Common in songs
    ]
    for phrase in lyrical_phrases:
        if phrase in text_lower:
            print(f"    → Detected lyrical phrase: '{phrase}'")
            return True

    # Check for all-caps screaming/singing patterns (FEIN!!!)
    if text.isupper() and len(text) >= 3:
        return True

    return False


def _finalize_speech_segments(raw_segments):
    """Post-process raw speech windows to remove artifacts and smooth timing."""
    merged_segments = _merge_segments(raw_segments, max_gap=SEGMENT_MERGE_GAP_SECONDS)
    cleaned_segments = _filter_short_isolated_segments(merged_segments)

    if len(cleaned_segments) > MAX_SEGMENTS_BEFORE_CONSOLIDATION:
        print(f"Too many speech fragments ({len(cleaned_segments)}), consolidating...")
        cleaned_segments = _merge_segments(cleaned_segments, max_gap=0.45)

    padded_segments = []
    for start, end in cleaned_segments:
        padded_start = max(0.0, start - 0.03)
        padded_end = end + 0.03
        padded_segments.append((padded_start, padded_end))

    return padded_segments


def detect_speech_segments(vocals_path, accompaniment_path=None):
    """Detect speech windows while filtering out singing/music bleed."""
    vocals_path = Path(vocals_path)
    duration_seconds = _probe_duration_seconds(vocals_path)

    # Use a lighter Whisper model for longer clips to keep runtime practical.
    whisper_model_name = "base" if (duration_seconds and duration_seconds <= 90) else "tiny"
    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_compute = "float16" if whisper_device == "cuda" else "int8"

    print(
        f"Loading faster-whisper ({whisper_model_name}, {whisper_device})"
        " for voiceover detection..."
    )

    try:
        try:
            model = WhisperModel(
                whisper_model_name,
                device=whisper_device,
                compute_type=whisper_compute
            )
        except Exception as init_error:
            if whisper_device == "cuda":
                print(f"CUDA Whisper init failed ({init_error}), retrying on CPU...")
                whisper_device = "cpu"
                whisper_compute = "int8"
                model = WhisperModel(
                    whisper_model_name,
                    device=whisper_device,
                    compute_type=whisper_compute
                )
            else:
                raise

        print("Transcribing audio to detect voiceover segments...")
        segments, _ = model.transcribe(
            str(vocals_path),
            language="en",
            word_timestamps=False,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            vad_filter=True
        )
        segments_list = list(segments)
    except Exception as e:
        print(f"Warning: faster-whisper failed: {e}")
        print("Falling back to VAD detection...")
        return _detect_speech_segments_fallback_vad(vocals_path)

    if not segments_list:
        print("Warning: Whisper found no speech segments")
        return []

    print(f"Whisper produced {len(segments_list)} candidate segments")

    # Load audio once for acoustic filters.
    print("Loading separated tracks for acoustic/music filtering...")
    try:
        vocals_audio, sr = librosa.load(str(vocals_path), sr=16000)
    except Exception as e:
        print(f"Warning: Could not load vocals audio: {e}")
        vocals_audio = None
        sr = 16000

    accompaniment_audio = None
    if accompaniment_path and Path(accompaniment_path).exists():
        try:
            accompaniment_audio, _ = librosa.load(str(accompaniment_path), sr=16000)
        except Exception as e:
            print(f"Warning: Could not load accompaniment: {e}")

    filtered_text_count = 0
    filtered_music_count = 0
    filtered_acoustic_count = 0
    filtered_prob_count = 0
    raw_segments = []

    for segment in segments_list:
        if not hasattr(segment, "start") or not hasattr(segment, "end"):
            continue

        seg_start = float(segment.start)
        seg_end = float(segment.end)
        seg_duration = max(0.0, seg_end - seg_start)
        segment_text = (segment.text or "").strip()

        if seg_duration < 0.12:
            continue

        if not _is_probable_speech_text(segment_text):
            filtered_text_count += 1
            continue

        if is_likely_music_lyrics(segment_text):
            filtered_text_count += 1
            print(f"  Filtered (text): \"{segment_text}\"")
            continue

        no_speech_prob = getattr(segment, "no_speech_prob", None)
        if no_speech_prob is not None and no_speech_prob > 0.72:
            filtered_prob_count += 1
            continue

        is_advertising = is_likely_advertising_voiceover(segment_text)
        start_sample = int(seg_start * sr)
        end_sample = int(seg_end * sr)

        # Reject segments where accompaniment dominates vocals
        if (
            not is_advertising
            and accompaniment_audio is not None
            and vocals_audio is not None
            and end_sample <= len(accompaniment_audio)
            and end_sample <= len(vocals_audio)
        ):
            acc_segment = accompaniment_audio[start_sample:end_sample]
            vocal_segment = vocals_audio[start_sample:end_sample]

            if len(acc_segment) > 0 and len(vocal_segment) > 0:
                acc_energy = float(np.sqrt(np.mean(acc_segment ** 2)))
                vocal_energy = float(np.sqrt(np.mean(vocal_segment ** 2)))
                if vocal_energy > 1e-7:
                    music_ratio = acc_energy / vocal_energy
                    ratio_threshold = 0.35 if seg_duration < 1.1 else 0.62
                    if music_ratio > ratio_threshold:
                        filtered_music_count += 1
                        print(
                            "  Filtered (music bleed):"
                            f" \"{segment_text}\" ratio={music_ratio:.2f}"
                        )
                        continue

        # Spectral singing check for any remaining ambiguous regions
        if vocals_audio is not None and end_sample <= len(vocals_audio):
            segment_audio = vocals_audio[start_sample:end_sample]
            if len(segment_audio) > sr * 0.08:
                if is_singing_not_speech(segment_audio, sr, segment_duration=seg_duration):
                    filtered_acoustic_count += 1
                    print(f"  Filtered (acoustic): \"{segment_text}\"")
                    continue

        raw_segments.append((seg_start, seg_end))

    total_candidates = len(segments_list)
    total_filtered = (
        filtered_text_count
        + filtered_music_count
        + filtered_acoustic_count
        + filtered_prob_count
    )

    if total_candidates > 0:
        filter_pct = 100.0 * total_filtered / total_candidates
        print(
            "Filtered segments:"
            f" text={filtered_text_count}, music={filtered_music_count},"
            f" acoustic={filtered_acoustic_count}, no_speech={filtered_prob_count}"
            f" ({filter_pct:.0f}% of candidates)"
        )

        # High rejection ratio + little usable speech usually means music-only content.
        if filter_pct > 85 and len(raw_segments) < 3:
            print("No reliable voiceover detected after filtering")
            return []

    speech_segments = _finalize_speech_segments(raw_segments)
    if not speech_segments:
        print("No speech segments survived post-processing")
        return []

    print(f"Final speech segments: {len(speech_segments)}")
    total_speech_time = sum(e - s for s, e in speech_segments)
    print(f"Total voiceover time: {total_speech_time:.1f}s")

    return speech_segments


def _detect_speech_segments_fallback_vad(vocals_path):
    """Fallback speech detection using simple silence detection."""
    cmd = [
        "ffmpeg",
        "-i", str(vocals_path),
        "-af", "silencedetect=noise=-40dB:d=0.15",
        "-f", "null",
        "-"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    silence_start_pattern = r"silence_start: ([\d.]+)"
    silence_end_pattern = r"silence_end: ([\d.]+)"

    silence_starts = [float(m) for m in re.findall(silence_start_pattern, result.stderr)]
    silence_ends = [float(m) for m in re.findall(silence_end_pattern, result.stderr)]

    duration_pattern = r"Duration: (\d{2}):(\d{2}):([\d.]+)"
    duration_match = re.search(duration_pattern, result.stderr)
    if duration_match:
        h, m, s = duration_match.groups()
        duration = int(h) * 3600 + int(m) * 60 + float(s)
    else:
        duration = 60

    all_segments = []
    if not silence_ends or (silence_starts and silence_starts[0] < silence_ends[0]):
        segment = (0.0, silence_starts[0] if silence_starts else duration)
        all_segments.append(segment)

    for i in range(len(silence_ends)):
        start = silence_ends[i]
        end = silence_starts[i + 1] if i + 1 < len(silence_starts) else duration
        if end > start:
            all_segments.append((start, end))

    MAX_SPEECH_DURATION = 4.0
    speech_segments = [(s, e) for s, e in all_segments if (e - s) <= MAX_SPEECH_DURATION]
    speech_segments = _finalize_speech_segments(speech_segments)

    print(f"Fallback: Detected {len(all_segments)} segments, filtered to {len(speech_segments)}")
    return speech_segments


def extract_speech_only(vocals_path, output_path):
    """Extract only speech from vocals track, completely removing singing.

    Uses speech segment detection to identify when someone is actually speaking,
    then keeps ONLY those segments and silences everything else.

    Args:
        vocals_path: Path to vocals audio (from Demucs)
        output_path: Path for speech-only output

    Returns:
        Path to speech-only audio
    """
    output_path = Path(output_path)

    # Detect when speech is happening
    speech_segments = detect_speech_segments(vocals_path)

    if not speech_segments:
        print("Warning: No speech detected, using silence")
        # Create silent audio
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", "60",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
        return output_path

    # Build FFmpeg filter to keep only speech segments
    # Create enable conditions for each speech segment
    enable_conditions = []
    for start, end in speech_segments:
        enable_conditions.append(f"between(t,{start},{end})")

    # Combine all conditions with OR
    enable_expression = "+".join(enable_conditions)

    # Apply volume filter that enables only during speech
    # During non-speech: volume=0, During speech: volume=1
    filter_complex = (
        f"volume=enable='{enable_expression}':volume=1,"
        f"volume=enable='not({enable_expression})':volume=0"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(vocals_path),
        "-af", filter_complex,
        str(output_path)
    ]

    print(f"Extracting speech from {len(speech_segments)} detected segments...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Speech extraction failed: {result.stderr}")
        # Fall back to aggressive gating
        fallback_filter = "highpass=f=100,lowpass=f=3500,agate=threshold=0.15:attack=5:release=50:range=0.001"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(vocals_path),
            "-af", fallback_filter,
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, text=True)

    return output_path


def mix_vocals_with_music_ducking(vocals_path, new_music_path, output_path,
                                   vocals_volume=1.0, music_volume=0.8,
                                   gate_threshold=0.035, process_timeout=600):
    """Mix vocals with new music using sidechain ducking and noise gate.

    The music volume automatically ducks (lowers) when voice is detected,
    and smoothly fades back up during silent parts - like a professional ad mix.

    A noise gate is applied to the vocals to suppress quiet background singing
    while keeping the louder voiceover.

    Args:
        vocals_path: Path to vocals audio file
        new_music_path: Path to new music to mix in
        output_path: Path for output mixed audio
        vocals_volume: Volume multiplier for vocals (1.0 = original)
        music_volume: Base volume for music when no voice (0.8 default)
        gate_threshold: Threshold for noise gate (0.05 = suppress quiet sounds)

    Returns:
        Path to mixed audio file
    """
    output_path = Path(output_path)

    filter_complex = (
        # Speech-focused filtering + gate to suppress residual singing/music.
        f"[0:a]highpass=f=110,lowpass=f=5200,"
        f"agate=threshold={gate_threshold}:ratio=8:attack=3:release=120:range=0.02,"
        f"volume={vocals_volume}[voice];"
        f"[1:a]volume={music_volume}[music_raw];"
        # Aggressive sidechain so music ducks clearly under voiceover.
        f"[music_raw][voice]sidechaincompress="
        f"threshold=0.015:"
        f"ratio=8:"
        f"attack=20:"
        f"release=350:"
        f"makeup=1"
        f"[music_ducked];"
        # Mix and limit final peaks.
        f"[voice][music_ducked]amix=inputs=2:duration=shortest:dropout_transition=0,"
        f"alimiter=limit=0.95"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(vocals_path),
        "-i", str(new_music_path),
        "-filter_complex", filter_complex,
        "-ac", "2",  # Stereo output
        str(output_path)
    ]

    print(f"Mixing with ducking (music auto-lowers during voiceover)...")
    _run_checked(cmd, timeout=process_timeout, step_name="Audio mixing")

    print(f"Mixed audio with ducking saved to: {output_path}")
    return output_path


def detect_song_start(audio_path, window_size=5.0, energy_threshold=1.5):
    """Detect where the actual song starts by finding energy increase.

    Skips quiet/minimal intros like Pink Floyd's long instrumental openings.

    Args:
        audio_path: Path to audio file
        window_size: Size of analysis window in seconds (default 5s)
        energy_threshold: How much higher energy must be vs minimum (1.5 = 50% higher)

    Returns:
        Start time in seconds (0 if no intro detected)
    """
    try:
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=22050, duration=120)  # Analyze first 2 minutes

        # Calculate RMS energy in windows
        hop_length = int(window_size * sr)
        frame_length = hop_length

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # Find minimum energy (intro baseline)
        min_energy = np.min(rms)

        # Find where energy first exceeds threshold (song starts)
        threshold = min_energy * energy_threshold

        for i, energy in enumerate(rms):
            if energy > threshold:
                start_time = i * window_size
                # Don't skip more than 60 seconds
                start_time = min(start_time, 60.0)
                if start_time > 10:  # Only skip if intro is >10 seconds
                    print(f"Detected {start_time:.1f}s intro, skipping to main section")
                    return start_time
                return 0.0

        return 0.0  # No intro detected
    except Exception as e:
        print(f"Warning: Intro detection failed: {e}, using full song")
        return 0.0


def separate_and_remix(video_audio_path, new_music_path, output_path,
                       vocals_volume=1.0, music_volume=0.7):
    """Full pipeline: replace background music while preserving voiceover.

    Creates a clean mix with:
    - New music playing throughout the entire duration
    - Original voiceover overlaid during speech moments only
    - Auto-ducking so music lowers when voiceover plays

    Args:
        video_audio_path: Path to extracted video audio
        new_music_path: Path to new music track
        output_path: Path for final mixed audio
        vocals_volume: Volume for preserved voiceover
        music_volume: Base volume for new music (auto-ducks when voice detected)

    Returns:
        Path to final mixed audio
    """
    video_audio_path = Path(video_audio_path)
    new_music_path = Path(new_music_path)
    output_path = Path(output_path)
    duration_seconds = _probe_duration_seconds(video_audio_path) or 120.0
    ffmpeg_timeout = int(min(1800, max(180, duration_seconds * 1.5)))

    # Step 1: Separate vocals from original using Demucs
    print("Step 1: Separating voiceover from background music...")
    separated = separate_audio(video_audio_path)

    # Step 1.5: Preprocess vocals to remove residual music artifacts
    print("Step 1.5: Cleaning vocals track (removing residual music)...")
    cleaned_vocals_path = Path(tempfile.gettempdir()) / "cleaned_vocals.wav"

    # Speech-focused cleanup to suppress residual music bleed from separation.
    preprocess_filter = (
        "highpass=f=110,"
        "lowpass=f=5200,"
        "afftdn=nf=-28:nt=w,"
        "agate=threshold=0.028:ratio=6:attack=3:release=120:range=0.02"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(separated['vocals']),
        "-af", preprocess_filter,
        str(cleaned_vocals_path)
    ]

    try:
        _run_checked(cmd, timeout=ffmpeg_timeout, step_name="Vocal cleanup")
    except Exception as e:
        print(f"Warning: Advanced vocal cleanup failed ({e}), retrying with compatibility filter")
        compatibility_filter = (
            "highpass=f=110,"
            "lowpass=f=5200,"
            "agate=threshold=0.03:ratio=6:attack=3:release=120:range=0.02"
        )
        fallback_cmd = [
            "ffmpeg", "-y",
            "-i", str(separated['vocals']),
            "-af", compatibility_filter,
            str(cleaned_vocals_path)
        ]
        _run_checked(fallback_cmd, timeout=ffmpeg_timeout, step_name="Vocal cleanup (compatibility)")

    # Step 2: Detect ONLY voiceover segments (not singing)
    print("Step 2: Detecting voiceover segments...")
    speech_segments = detect_speech_segments(cleaned_vocals_path, accompaniment_path=separated['music'])

    if not speech_segments:
        print("No voiceover detected - using new music only")
        # Just copy the new music as output
        shutil.copy(new_music_path, output_path)
        return output_path

    # Step 3: Extract voiceover using detected segments
    print("Step 3: Extracting voiceover segments...")
    clean_voiceover_path = Path(tempfile.gettempdir()) / "clean_voiceover.wav"

    if len(speech_segments) > MAX_SEGMENTS_FOR_EXPRESSION:
        print(
            f"{len(speech_segments)} speech segments detected; "
            "using aggressive full-track speech gate fallback."
        )
        fallback_filter = (
            "highpass=f=110,lowpass=f=5200,"
            "agate=threshold=0.032:ratio=8:attack=3:release=120:range=0.015"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(cleaned_vocals_path),
            "-af", fallback_filter,
            str(clean_voiceover_path)
        ]
        _run_checked(cmd, timeout=ffmpeg_timeout, step_name="Speech extraction (fallback gate)")
    else:
        # Build FFmpeg filter: keep ONLY detected speech windows.
        enable_conds = [f"between(t,{s:.3f},{e:.3f})" for s, e in speech_segments]
        enable_expr = "+".join(enable_conds)
        speech_mask_filter = (
            f"volume=enable='{enable_expr}':volume=1,"
            f"volume=enable='not({enable_expr})':volume=0,"
            "highpass=f=110,lowpass=f=5200,"
            "agate=threshold=0.028:ratio=6:attack=3:release=120:range=0.02"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(cleaned_vocals_path),
            "-af", speech_mask_filter,
            str(clean_voiceover_path)
        ]
        _run_checked(cmd, timeout=ffmpeg_timeout, step_name="Speech extraction")

    # Step 3.5: Detect and skip long instrumental intros (e.g., Pink Floyd)
    intro_skip = detect_song_start(new_music_path)
    music_to_use = new_music_path

    if intro_skip > 0:
        # Trim the intro from the music
        trimmed_music_path = Path(tempfile.gettempdir()) / "music_no_intro.wav"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(intro_skip),  # Skip intro
            "-i", str(new_music_path),
            "-acodec", "pcm_s16le",  # Re-encode audio (copy doesn't work with -ss)
            "-ar", "44100",
            "-ac", "2",
            str(trimmed_music_path)
        ]
        try:
            _run_checked(cmd, timeout=ffmpeg_timeout, step_name="Music intro trim")
            music_to_use = trimmed_music_path
        except Exception as e:
            print(f"Warning: Intro skip failed: {e}")
            music_to_use = new_music_path

    # Step 4: Mix clean voiceover with new music + ducking
    print("Step 4: Mixing voiceover with new music (auto-ducking enabled)...")
    mixed_path = mix_vocals_with_music_ducking(
        clean_voiceover_path,
        music_to_use,
        output_path,
        vocals_volume=vocals_volume,
        music_volume=music_volume,
        process_timeout=max(600, ffmpeg_timeout)
    )

    return mixed_path


def cleanup_demucs_output(output_dir=None):
    """Clean up temporary separation output files."""
    # Clean up both possible output directories
    dirs_to_clean = [
        Path(tempfile.gettempdir()) / "demucs_output",
        Path(tempfile.gettempdir()) / "voice_separation"
    ]

    if output_dir:
        dirs_to_clean.append(Path(output_dir))

    for dir_path in dirs_to_clean:
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"Cleaned up: {dir_path}")
            except:
                pass
