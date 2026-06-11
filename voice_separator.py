"""Voice/music separation and smart audio ducking for ads."""
import subprocess
import tempfile
from pathlib import Path
import shutil
import os
import re
import time
import gc
import numpy as np
import torch
import librosa
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, correlate

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))
os.environ.setdefault("NUMBA_NUM_THREADS", "1")


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
DISABLE_FASTER_WHISPER = False
TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)")


def _segment_size_for_model(model_id, duration_seconds=None, has_cuda=False):
    """Pick a safe demucs segment size for the specific model."""
    # htdemucs has a hard maximum ~7.8s; keep under that.
    if "htdemucs" in model_id:
        return "7"     

    if has_cuda:
        return "12"

    if duration_seconds and duration_seconds <= 120:
        return "8"

    return "6"


def _save_tensor_as_wav(audio_tensor, sample_rate, output_path):
    """Save torch tensor audio to WAV without torchaudio/torchcodec."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_np = audio_tensor.detach().cpu().numpy()
    if audio_np.ndim == 2:
        # Demucs tensors are [channels, samples]; scipy expects [samples, channels].
        audio_np = np.transpose(audio_np, (1, 0))

    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_i16 = np.int16(audio_np * 32767)
    wavfile.write(str(output_path), sample_rate, audio_i16)


def _load_demucs_model(model_id, device):
    """Load Demucs model for current run (no long-lived cache)."""
    from demucs.pretrained import get_model
    model = get_model(model_id)
    model.to(device)
    model.eval()
    return model


def _separate_audio_with_demucs_api(audio_path, output_dir, model_id, duration_seconds=None):
    """Run Demucs in-process and save stems via scipy to avoid torchcodec save issues."""
    from demucs.apply import apply_model
    from demucs.audio import AudioFile

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    segment_size = float(
        _segment_size_for_model(
            model_id,
            duration_seconds=duration_seconds,
            has_cuda=has_cuda
        )
    )

    start = time.time()
    model = None
    wav = None
    mix = None
    sources = None

    try:
        model = _load_demucs_model(model_id, device)
        wav = AudioFile(str(audio_path)).read(
            streams=0,
            samplerate=model.samplerate,
            channels=model.audio_channels
        )
        mix = wav.to(device)

        with torch.no_grad():
            sources = apply_model(
                model,
                mix[None],
                shifts=1 if has_cuda else 0,   # speed on CPU, mild quality boost on GPU
                split=True,
                overlap=0.25,
                progress=False,
                device=device,
                num_workers=0,
                segment=segment_size
            )[0].cpu()

        source_names = list(model.sources)
        if "vocals" not in source_names:
            raise Exception(f"Model {model_id} output has no vocals stem: {source_names}")

        vocals_idx = source_names.index("vocals")
        vocals = sources[vocals_idx]
        non_vocal_indices = [i for i, name in enumerate(source_names) if name != "vocals"]
        if non_vocal_indices:
            music = sources[non_vocal_indices].sum(dim=0)
        else:
            music = torch.zeros_like(vocals)

        stem_name = Path(audio_path).stem
        model_output = Path(output_dir) / model_id / stem_name
        vocals_path = model_output / "vocals.wav"
        music_path = model_output / "no_vocals.wav"

        _save_tensor_as_wav(vocals, model.samplerate, vocals_path)
        _save_tensor_as_wav(music, model.samplerate, music_path)

        elapsed = time.time() - start
        print(f"Demucs API separation using {model_id} finished in {elapsed:.1f}s")
        return vocals_path, music_path
    finally:
        # Keep memory stable across repeated Streamlit reruns.
        del sources
        del mix
        del wav
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _models_for_current_run(model=None):
    """Pick safe default model list based on available memory budget."""
    if model:
        return [(model, model)]

    if torch.cuda.is_available():
        models = list(DEMUCS_MODELS)
    else:
        # CPU-only default: skip heavier models to reduce OOM and latency.
        heavy_opt_in = os.getenv("DEMUCS_ENABLE_HEAVY_MODELS", "0").strip().lower() in {"1", "true", "yes"}
        models = list(DEMUCS_MODELS) if heavy_opt_in else list(DEMUCS_MODELS[:2])

    # DEMUCS_PREFERRED_MODEL jumps a specific model (e.g. htdemucs) to the
    # front of the ladder; the rest stay as fallbacks.
    preferred = os.getenv("DEMUCS_PREFERRED_MODEL", "").strip()
    if preferred:
        models = [(preferred, preferred)] + [m for m in models if m[0] != preferred]
    return models


def _run_checked(cmd, timeout=None, env=None, step_name="Command"):
    """Run subprocess and raise a detailed exception on failure."""
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            break
        except subprocess.TimeoutExpired:
            raise Exception(f"{step_name} timed out after {timeout}s")
        except OSError as e:
            if getattr(e, "winerror", None) == 1450 and attempt < max_attempts - 1:
                print(f"{step_name}: low system resources, retrying once...")
                gc.collect()
                time.sleep(1.0)
                continue
            raise Exception(f"{step_name} failed to start process: {e}")

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr if stderr else stdout
        raise Exception(f"{step_name} failed: {details}")
    return result


def _debleed_with_ffmpeg_sidechain(vocals_path, accompaniment_path, output_path, timeout=300):
    """Low-resource de-bleed fallback using ffmpeg sidechain compression."""
    filter_complex = (
        "[0:a][1:a]sidechaincompress="
        "threshold=0.010:ratio=10:attack=4:release=220,"
        "highpass=f=95,lowpass=f=6500,"
        "agate=threshold=0.004:ratio=1.6:attack=4:release=260:range=0.25"
        "[out]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(vocals_path),
        "-i", str(accompaniment_path),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        str(output_path)
    ]
    _run_checked(cmd, timeout=timeout, step_name="FFmpeg de-bleed fallback")


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


def suppress_music_bleed_with_reference(
    vocals_path,
    accompaniment_path,
    output_path,
    suppression_strength=1.25,
    residual_floor=0.06,
    sr=32000,
    speech_high_cut_hz=7800
):
    """Suppress residual original music in vocals using accompaniment reference."""
    vocals_path = Path(vocals_path)
    accompaniment_path = Path(accompaniment_path)
    output_path = Path(output_path)

    vocals, _ = librosa.load(str(vocals_path), sr=sr, mono=True)
    accompaniment, _ = librosa.load(str(accompaniment_path), sr=sr, mono=True)

    if len(vocals) == 0:
        raise Exception("Vocal track is empty")

    target_len = max(len(vocals), len(accompaniment))
    if len(vocals) < target_len:
        vocals = np.pad(vocals, (0, target_len - len(vocals)))
    if len(accompaniment) < target_len:
        accompaniment = np.pad(accompaniment, (0, target_len - len(accompaniment)))

    n_fft = 2048
    hop = 256

    v_stft = librosa.stft(vocals, n_fft=n_fft, hop_length=hop)
    a_stft = librosa.stft(accompaniment, n_fft=n_fft, hop_length=hop)

    v_mag = np.abs(v_stft)
    a_mag = np.abs(a_stft)
    v_pow = v_mag ** 2
    a_pow = a_mag ** 2

    # Base spectral subtraction in power domain.
    residual_pow = np.maximum(v_pow - suppression_strength * a_pow, residual_floor * v_pow)
    mask = np.sqrt(residual_pow / (v_pow + 1e-9))

    # Emphasize speech band and attenuate out-of-band residual music.
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    speech_band = (freqs >= 120) & (freqs <= speech_high_cut_hz)
    band_weights = np.where(speech_band, 1.0, 0.42).astype(np.float32)[:, None]
    mask *= band_weights

    # Frame-level accompaniment dominance penalty (strong for music-heavy bursts).
    frame_v = np.mean(v_mag, axis=0)
    frame_a = np.mean(a_mag, axis=0)
    frame_dom = frame_a / (frame_v + 1e-8)
    dom_penalty = 1.0 / (1.0 + np.maximum(0.0, frame_dom - 0.42) * 3.2)
    dom_penalty = np.clip(dom_penalty, 0.15, 1.0)
    mask *= dom_penalty[None, :]

    # Extra attenuation where accompaniment locally dominates magnitude.
    heavy_music_bins = a_mag > (0.92 * v_mag)
    mask[heavy_music_bins] *= 0.72

    # Light temporal smoothing to reduce musical bursts without pumping.
    smooth_kernel = np.array([0.15, 0.70, 0.15], dtype=np.float32)
    mask = np.apply_along_axis(lambda m: np.convolve(m, smooth_kernel, mode="same"), 1, mask)
    mask = np.clip(mask, max(0.03, residual_floor * 0.5), 1.0)

    enhanced_stft = v_stft * mask
    enhanced = librosa.istft(enhanced_stft, hop_length=hop, length=target_len)

    # Final bandpass in time domain for voiceover intelligibility.
    low_cut = 120.0
    high_cut = min(float(speech_high_cut_hz), sr * 0.45)
    if high_cut > low_cut + 50:
        sos = butter(6, [low_cut / (sr * 0.5), high_cut / (sr * 0.5)], btype="bandpass", output="sos")
        enhanced = sosfiltfilt(sos, enhanced)

    peak = float(np.max(np.abs(enhanced))) if len(enhanced) > 0 else 0.0
    if peak > 0.98:
        enhanced = enhanced / peak * 0.98

    enhanced_i16 = np.int16(np.clip(enhanced, -1.0, 1.0) * 32767)
    wavfile.write(str(output_path), sr, enhanced_i16)

    return output_path


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

    # Check if CUDA GPU is available
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)} - using GPU acceleration")
    else:
        print("No GPU detected - using CPU (slower)")
        # Restrict thread fan-out to avoid CPU-memory spikes on Windows.
        try:
            torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    print("Demucs backend: in-process API (low-memory mode)")

    models_to_try = _models_for_current_run(model=model)
    if not torch.cuda.is_available() and len(models_to_try) < len(DEMUCS_MODELS):
        print("CPU mode: trying compact Demucs models first (set DEMUCS_ENABLE_HEAVY_MODELS=1 to include all).")

    last_error = None
    for model_name, model_id in models_to_try:
        print(f"Trying Demucs model: {model_name}...")
        try:
            vocals_path, music_path = _separate_audio_with_demucs_api(
                audio_path=audio_path,
                output_dir=output_dir,
                model_id=model_id,
                duration_seconds=duration_seconds
            )
            print(f"Separation complete using {model_name}!")
            return {
                'vocals': vocals_path,
                'music': music_path,
                'model_used': model_name
            }
        except Exception as e:
            last_error = str(e)
            print(f"Model {model_name} failed: {e}")
            # Try the next model if available.
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
        # Keep generous padding so phrase edges are not cut.
        padded_start = max(0.0, start - 0.10)
        padded_end = end + 0.10
        padded_segments.append((padded_start, padded_end))

    return padded_segments


def _parse_version_tuple(version_text):
    """Parse version string into comparable integer tuple."""
    nums = re.findall(r"\d+", str(version_text or ""))
    if not nums:
        return (0,)
    return tuple(int(n) for n in nums[:3])


def _is_faster_whisper_environment_compatible():
    """Check local dependency compatibility before loading faster-whisper."""
    try:
        import huggingface_hub
        current = _parse_version_tuple(getattr(huggingface_hub, "__version__", "0"))
        required = (0, 23, 0)
        if current < required:
            return False, (
                f"huggingface_hub {getattr(huggingface_hub, '__version__', '?')} is too old; "
                f"need >= {'.'.join(map(str, required))}"
            )
        return True, ""
    except Exception as e:
        return False, f"dependency check failed: {e}"


def _parse_timestamp_seconds(token):
    """Parse mm:ss or hh:mm:ss into float seconds."""
    token = token.strip()
    if not token:
        return None
    parts = token.split(":")
    if len(parts) == 2:
        mm, ss = parts
        if mm.isdigit() and ss.isdigit():
            return int(mm) * 60 + int(ss)
    elif len(parts) == 3:
        hh, mm, ss = parts
        if hh.isdigit() and mm.isdigit() and ss.isdigit():
            return int(hh) * 3600 + int(mm) * 60 + int(ss)
    return None


def parse_transcript_speech_segments(transcript_hint_text, duration_seconds=None):
    """Build speech windows from a timestamped transcript.

    Accepts common formats:
    - `0:05 text...`
    - `text...` then next line `0:05`
    """
    if not transcript_hint_text:
        return []

    raw_lines = [ln.strip() for ln in str(transcript_hint_text).splitlines() if ln.strip()]
    if not raw_lines:
        return []

    used_line_idx = set()
    pairs = []

    # First pass: inline timestamp + text on the same line.
    for i, line in enumerate(raw_lines):
        ts_match = TIMESTAMP_PATTERN.search(line)
        if not ts_match:
            continue
        ts_text = ts_match.group(1)
        ts = _parse_timestamp_seconds(ts_text)
        if ts is None:
            continue

        before = line[:ts_match.start()].strip(" -\t")
        after = line[ts_match.end():].strip(" -\t")
        text = after if after else before
        if text:
            pairs.append((float(ts), text))
            used_line_idx.add(i)

    # Second pass: timestamp-only line paired with nearby text line.
    for i, line in enumerate(raw_lines):
        if i in used_line_idx:
            continue

        ts = _parse_timestamp_seconds(line)
        if ts is None:
            continue

        text = ""
        # Prefer previous line (matches common copied YouTube transcript format).
        if i - 1 >= 0 and (i - 1) not in used_line_idx:
            prev = raw_lines[i - 1]
            if _parse_timestamp_seconds(prev) is None and not TIMESTAMP_PATTERN.search(prev):
                text = prev
                used_line_idx.add(i - 1)
        if not text and i + 1 < len(raw_lines) and (i + 1) not in used_line_idx:
            nxt = raw_lines[i + 1]
            if _parse_timestamp_seconds(nxt) is None and not TIMESTAMP_PATTERN.search(nxt):
                text = nxt
                used_line_idx.add(i + 1)

        if text:
            pairs.append((float(ts), text))
            used_line_idx.add(i)

    if not pairs:
        return []

    # Sort by timestamp and de-duplicate near-identical entries.
    pairs.sort(key=lambda x: x[0])
    dedup = []
    for ts, txt in pairs:
        if dedup and abs(ts - dedup[-1][0]) < 0.02:
            # Keep longer textual line for same timestamp.
            if len(txt) > len(dedup[-1][1]):
                dedup[-1] = (ts, txt)
        else:
            dedup.append((ts, txt))

    segments = []
    for idx, (start_ts, txt) in enumerate(dedup):
        word_count = max(1, len(re.findall(r"[A-Za-z']+", txt)))
        estimated = min(3.2, max(0.55, 0.28 * word_count))

        if idx + 1 < len(dedup):
            next_ts = dedup[idx + 1][0]
            max_allowed = max(start_ts + 0.25, next_ts - 0.05)
            end_ts = min(max_allowed, start_ts + estimated)
            end_ts = max(start_ts + 0.25, end_ts)
        else:
            # Last segment: estimate duration from text length.
            end_ts = start_ts + estimated
            if duration_seconds is not None:
                end_ts = min(end_ts, float(duration_seconds))

        start = max(0.0, start_ts - 0.01)
        if duration_seconds is not None:
            start = min(start, float(duration_seconds))
        end = max(start + 0.25, end_ts)
        if duration_seconds is not None:
            end = min(end, float(duration_seconds))
        if end > start + 0.08:
            segments.append((start, end))

    return sorted(segments, key=lambda x: x[0])


def detect_speech_segments(vocals_path, accompaniment_path=None):
    """Detect speech windows while filtering out singing/music bleed."""
    global DISABLE_FASTER_WHISPER
    vocals_path = Path(vocals_path)
    duration_seconds = _probe_duration_seconds(vocals_path)

    if not DISABLE_FASTER_WHISPER:
        fw_ok, fw_reason = _is_faster_whisper_environment_compatible()
        if not fw_ok:
            DISABLE_FASTER_WHISPER = True
            print(f"Skipping faster-whisper ({fw_reason})")

    if DISABLE_FASTER_WHISPER:
        print("Skipping faster-whisper (disabled due previous environment error).")
        return _detect_speech_segments_fallback_vad(
            vocals_path,
            accompaniment_path=accompaniment_path,
            asr_failed=True
        )

    # Use a lighter Whisper model for longer clips to keep runtime practical.
    whisper_model_name = "base" if (duration_seconds and duration_seconds <= 90) else "tiny"
    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_compute = "float16" if whisper_device == "cuda" else "int8"

    print(
        f"Loading faster-whisper ({whisper_model_name}, {whisper_device})"
        " for voiceover detection..."
    )

    try:
        from faster_whisper import WhisperModel
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
        if "snapshot_download()" in str(e):
            DISABLE_FASTER_WHISPER = True
            print("Disabling faster-whisper for this process due dependency mismatch.")
        print("Falling back to VAD detection...")
        return _detect_speech_segments_fallback_vad(
            vocals_path,
            accompaniment_path=accompaniment_path,
            asr_failed=True
        )

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


def _derive_speech_segments_acoustic(vocals_path, accompaniment_path=None, duration=None):
    """Derive speech windows from acoustic features only (no ASR)."""
    try:
        vocals_audio, sr = librosa.load(str(vocals_path), sr=16000, mono=True)
    except Exception as e:
        print(f"Warning: Acoustic speech masking could not load vocals: {e}")
        return []

    if len(vocals_audio) < int(0.25 * sr):
        return []

    accompaniment_audio = None
    if accompaniment_path and Path(accompaniment_path).exists():
        try:
            accompaniment_audio, _ = librosa.load(str(accompaniment_path), sr=16000, mono=True)
        except Exception:
            accompaniment_audio = None

    n = len(vocals_audio)
    if accompaniment_audio is not None:
        if len(accompaniment_audio) < n:
            accompaniment_audio = np.pad(accompaniment_audio, (0, n - len(accompaniment_audio)))
        accompaniment_audio = accompaniment_audio[:n]

    # Residual suppresses background bed and keeps speech prominence.
    residual = vocals_audio.copy()
    if accompaniment_audio is not None:
        residual = residual - 0.92 * accompaniment_audio

    # Speech band emphasis.
    try:
        sos = butter(4, [100 / (sr * 0.5), 5000 / (sr * 0.5)], btype="bandpass", output="sos")
        residual = sosfiltfilt(sos, residual)
    except Exception:
        pass

    hop = int(0.01 * sr)   # 10 ms
    frame = int(0.03 * sr) # 30 ms
    if hop < 1 or frame <= hop:
        return []

    rms_v = librosa.feature.rms(y=vocals_audio, frame_length=frame, hop_length=hop)[0]
    rms_r = librosa.feature.rms(y=residual, frame_length=frame, hop_length=hop)[0]
    if accompaniment_audio is not None:
        rms_a = librosa.feature.rms(y=accompaniment_audio, frame_length=frame, hop_length=hop)[0]
    else:
        rms_a = np.zeros_like(rms_v)

    stft = librosa.stft(residual, n_fft=512, hop_length=hop, win_length=frame)
    mag = np.abs(stft) + 1e-9
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    speech_band = (freqs >= 120) & (freqs <= 4600)
    band_ratio = np.sum(mag[speech_band], axis=0) / np.sum(mag, axis=0)
    peakiness = np.max(mag, axis=0) / np.mean(mag, axis=0)

    # Align array lengths.
    k = min(len(rms_v), len(rms_r), len(rms_a), len(band_ratio), len(peakiness))
    rms_v = rms_v[:k]
    rms_r = rms_r[:k]
    rms_a = rms_a[:k]
    band_ratio = band_ratio[:k]
    peakiness = peakiness[:k]

    db = 20 * np.log10(rms_r + 1e-9)
    p25, p85 = np.percentile(db, [25, 85])
    norm_db = np.clip((db - p25) / (p85 - p25 + 1e-6), 0.0, 1.0)

    modulation = np.abs(np.diff(np.r_[db[0], db]))
    mod_norm = np.clip(modulation / 6.0, 0.0, 1.0)
    music_ratio = rms_a / (rms_v + 1e-8)
    music_pen = np.clip((music_ratio - 0.55) / 1.2, 0.0, 1.0)
    tonal_pen = np.clip((peakiness - 6.0) / 8.0, 0.0, 1.0)

    score = (
        0.52 * norm_db
        + 0.34 * band_ratio
        + 0.20 * mod_norm
        - 0.22 * music_pen
        - 0.20 * tonal_pen
    )
    score = np.clip(score, 0.0, 1.0)
    score = np.convolve(score, np.array([0.2, 0.6, 0.2], dtype=np.float32), mode="same")

    hi = max(0.38, float(np.percentile(score, 65)))
    lo = max(0.24, hi - 0.18)

    # Hysteresis mask for stable speech windows.
    segments = []
    active = False
    seg_start = 0
    for i, s in enumerate(score):
        if not active and s >= hi:
            active = True
            seg_start = i
        elif active and s < lo:
            active = False
            seg_end = i
            start_t = seg_start * hop / sr
            end_t = seg_end * hop / sr
            if end_t - start_t >= 0.18:
                segments.append((start_t, end_t))
    if active:
        start_t = seg_start * hop / sr
        end_t = k * hop / sr
        if end_t - start_t >= 0.18:
            segments.append((start_t, end_t))

    # Merge small pauses and pad phrase edges.
    segments = _merge_segments(segments, max_gap=0.36)
    padded = []
    max_dur = duration if duration is not None else (len(vocals_audio) / sr)
    for s, e in segments:
        s2 = max(0.0, s - 0.14)
        e2 = min(max_dur, e + 0.16)
        if e2 - s2 >= 0.14:
            padded.append((s2, e2))
    segments = padded

    # Short-burst cleanup: remove likely singing ad-libs.
    final_segments = []
    for s, e in segments:
        seg_dur = e - s
        a = max(0, int(s * sr))
        b = min(len(vocals_audio), int(e * sr))
        if b <= a:
            continue
        seg_audio = vocals_audio[a:b]

        if accompaniment_audio is not None:
            acc_seg = accompaniment_audio[a:b]
            v_rms = float(np.sqrt(np.mean(seg_audio ** 2) + 1e-12))
            a_rms = float(np.sqrt(np.mean(acc_seg ** 2) + 1e-12))
            ratio = a_rms / (v_rms + 1e-8)
            # Keep narration, reject only very short music-dominant bursts.
            if seg_dur < 0.45 and ratio > 0.90:
                if is_singing_not_speech(seg_audio, sr=sr, segment_duration=seg_dur):
                    continue

        final_segments.append((s, e))

    finalized = _finalize_speech_segments(final_segments)
    # Minimum coverage safeguard for spoken ads when ASR is unavailable.
    total = sum(e - s for s, e in finalized)
    max_dur = duration if duration is not None else (len(vocals_audio) / sr)
    if max_dur > 0 and total < max_dur * 0.12 and len(finalized) <= 3:
        expanded = []
        for s, e in finalized:
            expanded.append((max(0.0, s - 0.25), min(max_dur, e + 0.35)))
        if expanded:
            finalized = _merge_segments(expanded, max_gap=0.45)
    return finalized


def _detect_speech_segments_fallback_vad(vocals_path, accompaniment_path=None, asr_failed=False):
    """Fallback speech detection using silence detection.

    This path is intentionally voice-preserving when ASR is unavailable.
    """
    cmd = [
        "ffmpeg",
        "-i", str(vocals_path),
        "-af", "silencedetect=noise=-38dB:d=0.10",
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
        duration = _probe_duration_seconds(vocals_path) or 60.0

    all_segments = []
    if not silence_ends or (silence_starts and silence_starts[0] < silence_ends[0]):
        segment = (0.0, silence_starts[0] if silence_starts else duration)
        all_segments.append(segment)

    for i in range(len(silence_ends)):
        start = silence_ends[i]
        end = silence_starts[i + 1] if i + 1 < len(silence_starts) else duration
        if end > start:
            all_segments.append((start, end))

    # Voice-preserving fallback: keep meaningful non-silent regions and avoid over-pruning.
    MIN_SEGMENT_DURATION = 0.12
    MAX_SEGMENT_DURATION = 24.0
    speech_segments = []
    for s, e in all_segments:
        seg_dur = e - s
        if seg_dur < MIN_SEGMENT_DURATION:
            continue
        if seg_dur > MAX_SEGMENT_DURATION:
            # Split very long segments so downstream enable expressions stay stable.
            cursor = s
            while cursor < e:
                chunk_end = min(cursor + MAX_SEGMENT_DURATION, e)
                if chunk_end - cursor >= MIN_SEGMENT_DURATION:
                    speech_segments.append((cursor, chunk_end))
                cursor = chunk_end
            continue
        speech_segments.append((s, e))

    # Keep fallback less aggressive than ASR path: merge gaps and pad without burst dropping.
    speech_segments = _merge_segments(speech_segments, max_gap=0.75)
    padded = []
    for s, e in speech_segments:
        padded.append((max(0.0, s - 0.12), min(duration, e + 0.12)))
    speech_segments = padded

    # Additional cleanup pass for fallback mode:
    # remove short music-dominant bursts ("burps"/ad-libs) while preserving narration.
    filtered_segments = []
    try:
        vocals_audio, sr = librosa.load(str(vocals_path), sr=16000, mono=True)
        accompaniment_audio = None
        if accompaniment_path and Path(accompaniment_path).exists():
            accompaniment_audio, _ = librosa.load(str(accompaniment_path), sr=16000, mono=True)

        for s, e in speech_segments:
            seg_dur = e - s
            if seg_dur < 0.14:
                continue

            start_sample = max(0, int(s * sr))
            end_sample = min(len(vocals_audio), int(e * sr))
            if end_sample <= start_sample:
                continue

            vocal_segment = vocals_audio[start_sample:end_sample]
            if len(vocal_segment) < int(0.08 * sr):
                continue

            if accompaniment_audio is not None and end_sample <= len(accompaniment_audio):
                acc_segment = accompaniment_audio[start_sample:end_sample]
                vocal_energy = float(np.sqrt(np.mean(vocal_segment ** 2)))
                acc_energy = float(np.sqrt(np.mean(acc_segment ** 2)))
                if vocal_energy > 1e-7:
                    music_ratio = acc_energy / vocal_energy

                    # Reject only very short, heavily music-dominant bursts.
                    if seg_dur < 0.45 and music_ratio > 0.95:
                        if is_singing_not_speech(vocal_segment, sr=sr, segment_duration=seg_dur):
                            continue

            filtered_segments.append((s, e))
    except Exception as e:
        print(f"Warning: Fallback acoustic filtering failed: {e}")
        filtered_segments = speech_segments

    # If filtering removed everything, keep original fallback windows to preserve narration.
    if filtered_segments:
        speech_segments = filtered_segments

    total_speech = sum(e - s for s, e in speech_segments)
    if asr_failed and (len(speech_segments) <= 1 or total_speech < max(2.0, duration * 0.06)):
        print(
            "Fallback VAD produced low-confidence speech coverage; "
            "trying acoustic speech-mask refinement."
        )
        refined = _derive_speech_segments_acoustic(
            vocals_path,
            accompaniment_path=accompaniment_path,
            duration=duration
        )
        if refined:
            speech_segments = refined

    # Overly-broad fallback windows leak song vocals; refine when needed.
    total_speech = sum(e - s for s, e in speech_segments) if speech_segments else 0.0
    if asr_failed and speech_segments and total_speech > duration * 0.80:
        print("Fallback windows cover most of track; applying acoustic refinement.")
        refined = _derive_speech_segments_acoustic(
            vocals_path,
            accompaniment_path=accompaniment_path,
            duration=duration
        )
        if refined:
            speech_segments = refined

    # Final safety fallback: avoid hard failure, but do not default to full-track.
    if not speech_segments:
        print("No fallback speech windows survived; using non-silent windows as safety fallback.")
        speech_segments = _merge_segments(all_segments, max_gap=0.45)
        padded = []
        for s, e in speech_segments:
            if e - s < 0.18:
                continue
            padded.append((max(0.0, s - 0.10), min(duration, e + 0.10)))
        speech_segments = padded if padded else [(0.0, min(duration, 3.0))]

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
                                   gate_threshold=0.0028, process_timeout=600,
                                   voice_lowpass_hz=9800,
                                   sidechain_threshold=0.012,
                                   sidechain_ratio=7.0,
                                   output_makeup=1.2):
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
        gate_threshold: Threshold for noise gate
        voice_lowpass_hz: Voice low-pass cutoff (higher keeps brightness)
        sidechain_threshold: Sidechain trigger level
        sidechain_ratio: Sidechain compression ratio
        output_makeup: Final gain before limiter

    Returns:
        Path to mixed audio file
    """
    output_path = Path(output_path)

    filter_complex = (
        # Voice path: preserve speech continuity, reduce bleed, avoid muffling.
        f"[0:a]highpass=f=90,lowpass=f={voice_lowpass_hz},"
        f"agate=threshold={gate_threshold}:ratio=1.22:attack=7:release=440:range=0.70,"
        f"acompressor=threshold=0.105:ratio=1.75:attack=10:release=190:makeup=2.6,"
        f"volume={vocals_volume}[voice];"
        f"[1:a]volume={music_volume}[music_raw];"
        # Sidechain keeps replacement music under narration while avoiding pumping.
        f"[music_raw][voice]sidechaincompress="
        f"threshold={sidechain_threshold}:"
        f"ratio={sidechain_ratio}:"
        f"attack=18:"
        f"release=460:"
        f"makeup=1"
        f"[music_ducked];"
        # Final leveling so output is not too quiet.
        f"[voice][music_ducked]amix=inputs=2:duration=shortest:dropout_transition=0,"
        f"acompressor=threshold=0.18:ratio=2.0:attack=20:release=220:makeup={output_makeup},"
        f"alimiter=limit=0.96"
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


def mix_vocals_with_music_ducking_pro(vocals_path, new_music_path, output_path,
                                       vocals_volume=1.0, music_volume=0.8,
                                       gate_threshold=0.0028, process_timeout=600,
                                       voice_lowpass_hz=11500,
                                       sidechain_threshold=0.012,
                                       sidechain_ratio=6.0):
    """Production master chain: 48 kHz stereo with presence-band ducking.

    Differences vs the legacy mixer:
    - the whole graph runs at 48 kHz stereo, so the replacement music keeps
      its full bandwidth and stereo image (legacy collapsed to the 32 kHz
      mono voice rate)
    - the music bed is loudness-normalized to -18 LUFS before levelling, so
      ducking behaves identically for any source track
    - EQ-ducking: the bed splits into lows / presence (250-5200 Hz) / air,
      and only the presence band - the range that masks speech - ducks
      under the voice. Bass and sparkle ride through narration, which reads
      as a professional mix instead of full-band pumping.
    """
    output_path = Path(output_path)

    filter_complex = (
        # Voice: clean, compress, lift to 48k stereo (center image), split a sidechain key.
        f"[0:a]aresample=48000,highpass=f=90,lowpass=f={voice_lowpass_hz},"
        f"agate=threshold={gate_threshold}:ratio=1.22:attack=7:release=440:range=0.70,"
        f"acompressor=threshold=0.105:ratio=1.75:attack=10:release=190:makeup=2.6,"
        f"volume={vocals_volume},aformat=channel_layouts=stereo,asplit=2[voice][vkey];"
        # Music: full-bandwidth stereo, normalized bed, levelled, with a short
        # intro swell so the bed never drowns narration that starts at t=0.
        f"[1:a]aresample=48000,aformat=channel_layouts=stereo,"
        f"loudnorm=I=-18:TP=-2.0:LRA=11,aresample=48000,"
        f"volume={music_volume},afade=t=in:st=0:d=0.9[mn];"
        # Presence-band ducking: only the speech-masking band compresses.
        f"[mn]acrossover=split=250 5200:order=4th[ml][mm][mh];"
        f"[mm][vkey]sidechaincompress="
        f"threshold={sidechain_threshold}:ratio={sidechain_ratio}:"
        f"attack=18:release=420:makeup=1[mmd];"
        f"[ml][mmd][mh]amix=inputs=3:normalize=0,alimiter=limit=0.98[md];"
        # Sum and glue.
        f"[voice][md]amix=inputs=2:duration=shortest:normalize=0:dropout_transition=0,"
        f"acompressor=threshold=0.20:ratio=1.9:attack=20:release=220:makeup=1.15,"
        f"alimiter=limit=0.95"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(vocals_path),
        "-i", str(new_music_path),
        "-filter_complex", filter_complex,
        "-ar", "48000",
        "-ac", "2",
        str(output_path)
    ]

    print("Mixing with production chain (48 kHz stereo, presence-band ducking)...")
    _run_checked(cmd, timeout=process_timeout, step_name="Audio mixing (pro)")
    print(f"Pro mix saved to: {output_path}")
    return output_path


def _extract_mono_wav_for_analysis(source_path, output_path, sr=32000):
    """Convert any audio source to analysis-friendly mono PCM WAV."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ac", "1",
        str(output_path)
    ]
    _run_checked(cmd, timeout=180, step_name="Audio analysis extraction")
    sr_loaded, y = wavfile.read(str(output_path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32) / 32768.0
    return sr_loaded, y


def _max_corr_with_lag_limit(x, y, sr, max_lag_seconds=0.75):
    """Compute max normalized cross-correlation with limited lag."""
    n = min(len(x), len(y))
    if n < 4096:
        return 0.0

    # Keep memory bounded: center crop + cheap decimation.
    x = x[:n]
    y = y[:n]
    target_len = min(n, int(24 * sr))
    start = max(0, (n - target_len) // 2)
    x = x[start:start + target_len]
    y = y[start:start + target_len]

    decim = 4
    x = x[::decim]
    y = y[::decim]
    sr_d = max(1, sr // decim)

    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x ** 2) * np.sum(y ** 2)) + 1e-12)
    if denom <= 1e-10:
        return 0.0

    max_lag = int(max_lag_seconds * sr_d)
    corr = correlate(x, y, mode="full", method="fft")
    mid = len(corr) // 2
    corr = corr[mid - max_lag:mid + max_lag + 1]
    return float(np.max(np.abs(corr)) / denom)


def evaluate_mix_quality(mixed_audio_path, original_music_path):
    """Estimate whether a swap quality is acceptable and suggest adjustments."""
    mixed_audio_path = Path(mixed_audio_path)
    original_music_path = Path(original_music_path)

    tmp_dir = Path(tempfile.gettempdir()) / "mix_quality_eval"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mixed_wav = tmp_dir / "mixed_eval.wav"
    music_wav = tmp_dir / "orig_music_eval.wav"

    report = {
        "rms_dbfs": None,
        "peak_dbfs": None,
        "jumps_gt12db": None,
        "hf_ratio_ge6k": None,
        "music_leak_corr": None,
        "issues": [],
        "retry_recommended": False
    }

    try:
        sr_m, y_m = _extract_mono_wav_for_analysis(mixed_audio_path, mixed_wav, sr=32000)
        sr_o, y_o = _extract_mono_wav_for_analysis(original_music_path, music_wav, sr=32000)

        n = min(len(y_m), len(y_o))
        y_m = y_m[:n]
        y_o = y_o[:n]
        duration_seconds = n / sr_m if sr_m else 0.0

        rms = float(np.sqrt(np.mean(y_m ** 2) + 1e-12))
        peak = float(np.max(np.abs(y_m)) + 1e-12)
        report["rms_dbfs"] = 20 * np.log10(rms)
        report["peak_dbfs"] = 20 * np.log10(peak)

        frame = int(0.02 * sr_m)
        if frame > 0 and len(y_m) > frame * 2:
            levels = []
            for i in range(0, len(y_m) - frame, frame):
                seg = y_m[i:i + frame]
                levels.append(20 * np.log10(np.sqrt(np.mean(seg ** 2) + 1e-12)))
            levels = np.array(levels, dtype=np.float32)
            jumps = np.abs(np.diff(levels))
            report["jumps_gt12db"] = int(np.sum(jumps > 12.0))
        else:
            report["jumps_gt12db"] = 0

        yf = np.fft.rfft(y_m)
        ff = np.fft.rfftfreq(len(y_m), d=1 / sr_m)
        power = np.abs(yf) ** 2 + 1e-18
        report["hf_ratio_ge6k"] = float(power[ff >= 6000].sum() / power.sum())

        # Leakage proxy: only reliable on clips with enough duration.
        if duration_seconds >= 15.0:
            sos = butter(6, [120 / (sr_m * 0.5), 5000 / (sr_m * 0.5)], btype="bandpass", output="sos")
            y_m_band = sosfiltfilt(sos, y_m)
            y_o_band = sosfiltfilt(sos, y_o)
            report["music_leak_corr"] = _max_corr_with_lag_limit(y_m_band, y_o_band, sr_m)
        else:
            report["music_leak_corr"] = 0.0

        if report["rms_dbfs"] < -19.0:
            report["issues"].append("mix_too_quiet")
        if report["jumps_gt12db"] > 20:
            report["issues"].append("voice_choppy_or_pumping")
        if report["hf_ratio_ge6k"] < 0.010:
            report["issues"].append("output_too_muffled")
        if report["music_leak_corr"] > 0.075:
            report["issues"].append("possible_original_music_bleed")

        report["retry_recommended"] = len(report["issues"]) > 0
    except Exception as e:
        # Do not block output on QA failures.
        report["issues"].append(f"quality_eval_failed:{e}")
        report["retry_recommended"] = False

    return report


def _apply_post_gain_limiter(input_path, output_path, gain_db, timeout=300):
    """Apply a fixed gain then limiter to lift quiet outputs safely."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", f"volume={gain_db:.2f}dB,alimiter=limit=0.97",
        str(output_path)
    ]
    _run_checked(cmd, timeout=timeout, step_name="Post-gain loudness correction")


def _blend_voice_continuity(primary_voice_path, continuity_voice_path, output_path, timeout=300, continuity_gain=0.40):
    """Blend segmented voice with a low-level full-track continuity bed."""
    continuity_gain = float(np.clip(continuity_gain, 0.20, 0.65))
    filter_complex = (
        "[0:a]volume=1.00[seg];"
        "[1:a]highpass=f=85,lowpass=f=9500,"
        "agate=threshold=0.0018:ratio=1.20:attack=8:release=420:range=0.72,"
        "acompressor=threshold=0.11:ratio=1.55:attack=10:release=180:makeup=2.2,"
        f"volume={continuity_gain:.3f}[cont];"
        "[seg][cont]amix=inputs=2:duration=shortest:dropout_transition=0,"
        "alimiter=limit=0.97"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(primary_voice_path),
        "-i", str(continuity_voice_path),
        "-filter_complex", filter_complex,
        str(output_path)
    ]
    _run_checked(cmd, timeout=timeout, step_name="Voice continuity blend")


def _measure_rms_dbfs(audio_path):
    """Return RMS level in dBFS for a media/audio file."""
    tmp_dir = Path(tempfile.gettempdir()) / "voice_rms_eval"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = tmp_dir / "rms_probe.wav"
    sr, y = _extract_mono_wav_for_analysis(audio_path, wav_path, sr=32000)
    del sr
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    return 20 * np.log10(rms)


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
                       vocals_volume=1.0, music_volume=0.7,
                       transcript_hint_text=None,
                       speech_segments_override=None):
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
        transcript_hint_text: Optional timestamped transcript to force speech windows

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
        "highpass=f=95,"
        "lowpass=f=9000,"
        "afftdn=nf=-26:nt=w"
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
            "highpass=f=95,"
            "lowpass=f=9000"
        )
        fallback_cmd = [
            "ffmpeg", "-y",
            "-i", str(separated['vocals']),
            "-af", compatibility_filter,
            str(cleaned_vocals_path)
        ]
        _run_checked(fallback_cmd, timeout=ffmpeg_timeout, step_name="Vocal cleanup (compatibility)")

    # Step 1.6: Suppress residual original-music bleed using accompaniment reference
    print("Step 1.6: Suppressing residual original music from vocals...")
    debleed_vocals_path = Path(tempfile.gettempdir()) / "debleed_vocals.wav"
    speech_source_path = cleaned_vocals_path
    try:
        suppress_music_bleed_with_reference(
            cleaned_vocals_path,
            separated['music'],
            debleed_vocals_path,
            suppression_strength=1.12,
            residual_floor=0.10,
            sr=32000,
            speech_high_cut_hz=8200
        )
        speech_source_path = debleed_vocals_path
    except Exception as e:
        print(f"Warning: Music bleed suppression failed ({e}); trying ffmpeg fallback.")
        try:
            _debleed_with_ffmpeg_sidechain(
                cleaned_vocals_path,
                separated['music'],
                debleed_vocals_path,
                timeout=ffmpeg_timeout
            )
            speech_source_path = debleed_vocals_path
            print("Applied ffmpeg de-bleed fallback.")
        except Exception as fallback_error:
            print(
                f"Warning: FFmpeg de-bleed fallback failed ({fallback_error}); "
                "continuing with cleaned vocals."
            )

    # Step 2: Detect ONLY voiceover segments (not singing)
    print("Step 2: Detecting voiceover segments...")
    using_transcript_segments = False
    using_override_segments = False
    speech_segments = []
    if speech_segments_override:
        # Exact (start, end) windows from word-level ASR upstream: more
        # precise than text timestamps, which only carry line starts.
        speech_segments = [(float(s), float(e)) for s, e in speech_segments_override]
        using_transcript_segments = True
        using_override_segments = True
        print(f"Using word-level speech windows from upstream ASR: {len(speech_segments)}")
        # Generalization net: quiet conversational dialogue can slip past
        # Whisper's VAD. Union in acoustic speech windows from the stem,
        # but only those that read as speech (not singing).
        try:
            acoustic = _derive_speech_segments_acoustic(
                str(separated['vocals']), accompaniment_path=str(separated['music'])
            ) or []
            added = 0
            import librosa as _librosa

            y_voc, sr_voc = _librosa.load(str(separated['vocals']), sr=16000, mono=True)
            for a_start, a_end in acoustic:
                overlaps = any(not (a_end <= s or a_start >= e) for s, e in speech_segments)
                if overlaps:
                    continue
                clip = y_voc[int(a_start * sr_voc):int(a_end * sr_voc)]
                if clip.size and not is_singing_not_speech(clip, sr=sr_voc):
                    speech_segments.append((float(a_start), float(a_end)))
                    added += 1
            if added:
                speech_segments = _merge_segments(sorted(speech_segments))
                print(f"Added {added} acoustic speech window(s) Whisper missed.")
        except Exception as e:
            print(f"Warning: acoustic union skipped ({e}).")
    elif transcript_hint_text:
        parsed_segments = parse_transcript_speech_segments(
            transcript_hint_text,
            duration_seconds=duration_seconds
        )
        if parsed_segments:
            speech_segments = parsed_segments
            using_transcript_segments = True
            print(f"Using transcript-guided speech segments: {len(speech_segments)}")
        else:
            print("Transcript hint provided but no timestamps were parsed; using ASR/VAD detection.")

    if not speech_segments:
        speech_segments = detect_speech_segments(
            speech_source_path,
            accompaniment_path=separated['music']
        )
    force_full_track_gate = False
    total_detected_speech = sum((e - s) for s, e in speech_segments) if speech_segments else 0.0
    min_expected_speech = max(4.0, duration_seconds * 0.14)
    print(f"Detected speech coverage: {total_detected_speech:.1f}s / {duration_seconds:.1f}s")

    if not speech_segments:
        print("No speech segments detected; using full-track speech-gate fallback.")
        force_full_track_gate = True
    elif total_detected_speech < min_expected_speech:
        print(
            f"Low-confidence speech detection ({total_detected_speech:.1f}s total); "
            "using full-track speech-gate fallback to avoid dropping narration."
        )
        force_full_track_gate = True
    elif DISABLE_FASTER_WHISPER and not using_transcript_segments:
        print("ASR unavailable in this environment; using acoustic-only speech windows.")

    # Step 3: Extract voiceover using detected segments
    print("Step 3: Extracting voiceover segments...")
    clean_voiceover_path = Path(tempfile.gettempdir()) / "clean_voiceover.wav"

    if force_full_track_gate or len(speech_segments) > MAX_SEGMENTS_FOR_EXPRESSION:
        print(
            f"{len(speech_segments)} speech segments detected; "
            "using aggressive full-track speech gate fallback."
        )
        fallback_filter = (
            "highpass=f=95,lowpass=f=9000,"
            "agate=threshold=0.0022:ratio=1.18:attack=7:release=450:range=0.72,"
            "acompressor=threshold=0.11:ratio=1.7:attack=11:release=210:makeup=2.6"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(speech_source_path),
            "-af", fallback_filter,
            str(clean_voiceover_path)
        ]
        _run_checked(cmd, timeout=ffmpeg_timeout, step_name="Speech extraction (fallback gate)")
    else:
        # Build FFmpeg filter: keep ONLY detected speech windows.
        # Pad each window so words straddling a boundary are not clipped.
        pad_pre, pad_post = 0.30, 0.45
        enable_conds = [
            f"between(t,{max(0.0, s - pad_pre):.3f},{e + pad_post:.3f})" for s, e in speech_segments
        ]
        enable_expr = "+".join(enable_conds)
        # User-pasted transcript windows are trusted -> hard-mute between them.
        # Auto-ASR word windows are precise but can miss words -> small floor.
        # Acoustic-only windows are least reliable -> higher floor (~-10 dB).
        if using_override_segments:
            outside_floor = 0.18
        elif using_transcript_segments:
            outside_floor = 0.0
        else:
            outside_floor = 0.30
        speech_mask_filter = (
            f"volume=enable='{enable_expr}':volume=1,"
            f"volume=enable='not({enable_expr})':volume={outside_floor},"
            "highpass=f=95,lowpass=f=9000,"
            "agate=threshold=0.0024:ratio=1.22:attack=7:release=420:range=0.68"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(speech_source_path),
            "-af", speech_mask_filter,
            str(clean_voiceover_path)
        ]
        _run_checked(cmd, timeout=ffmpeg_timeout, step_name="Speech extraction")

    # Step 3.2: Final de-bleed pass on extracted voiceover.
    print("Step 3.2: Final voiceover de-bleed (removing residual original music)...")
    final_voiceover_path = Path(tempfile.gettempdir()) / "final_voiceover.wav"
    mix_voice_source = clean_voiceover_path
    try:
        pre_final_rms = _measure_rms_dbfs(clean_voiceover_path)
        suppress_music_bleed_with_reference(
            clean_voiceover_path,
            separated['music'],
            final_voiceover_path,
            suppression_strength=1.14,
            residual_floor=0.10,
            sr=32000,
            speech_high_cut_hz=7600
        )
        post_final_rms = _measure_rms_dbfs(final_voiceover_path)
        if post_final_rms < pre_final_rms - 2.4:
            print(
                f"Final de-bleed reduced voice too much ({pre_final_rms:.2f} -> {post_final_rms:.2f} dBFS); "
                "keeping pre-final voiceover."
            )
        else:
            mix_voice_source = final_voiceover_path
    except Exception as e:
        print(f"Warning: Final de-bleed pass failed ({e}); trying ffmpeg fallback.")
        try:
            _debleed_with_ffmpeg_sidechain(
                clean_voiceover_path,
                separated['music'],
                final_voiceover_path,
                timeout=ffmpeg_timeout
            )
            mix_voice_source = final_voiceover_path
            print("Applied final ffmpeg de-bleed fallback.")
        except Exception as fallback_error:
            print(
                f"Warning: Final ffmpeg de-bleed fallback failed ({fallback_error}); "
                "using pre-final voiceover track."
            )

    # ASR-unavailable continuity blend: keep missed narration from disappearing.
    if DISABLE_FASTER_WHISPER and not using_transcript_segments:
        try:
            speech_ratio = 0.0
            if duration_seconds > 0:
                speech_ratio = total_detected_speech / duration_seconds
            if speech_ratio < 0.18:
                continuity_gain = 0.52
            elif speech_ratio < 0.30:
                continuity_gain = 0.38
            else:
                continuity_gain = 0.28

            continuity_path = Path(tempfile.gettempdir()) / "continuity_voice.wav"
            _run_checked(
                [
                    "ffmpeg", "-y",
                    "-i", str(speech_source_path),
                    "-af",
                    "highpass=f=85,lowpass=f=9500,"
                    "agate=threshold=0.0018:ratio=1.20:attack=8:release=420:range=0.72,"
                    "acompressor=threshold=0.11:ratio=1.55:attack=10:release=180:makeup=2.2",
                    str(continuity_path)
                ],
                timeout=max(180, ffmpeg_timeout),
                step_name="Continuity voice preparation"
            )
            blended_voice_path = Path(tempfile.gettempdir()) / "blended_voiceover.wav"
            _blend_voice_continuity(
                mix_voice_source,
                continuity_path,
                blended_voice_path,
                timeout=max(240, ffmpeg_timeout),
                continuity_gain=continuity_gain
            )
            mix_voice_source = blended_voice_path
            print(
                "Applied voice continuity blend for ASR-unavailable mode "
                f"(speech_ratio={speech_ratio:.2f}, gain={continuity_gain:.2f})."
            )
        except Exception as e:
            print(f"Warning: Continuity blend failed ({e}); using primary voice track.")

    # If extracted narration is too quiet, boost before mixing.
    try:
        voice_rms_db = _measure_rms_dbfs(mix_voice_source)
        print(f"Voiceover RMS before mix: {voice_rms_db:.2f} dBFS")
        if voice_rms_db < -26.5:
            gain_db = min(9.0, -23.5 - voice_rms_db)
            boosted_voice_path = Path(tempfile.gettempdir()) / "boosted_voiceover.wav"
            _apply_post_gain_limiter(
                mix_voice_source,
                boosted_voice_path,
                gain_db=gain_db,
                timeout=max(180, ffmpeg_timeout)
            )
            mix_voice_source = boosted_voice_path
            print(f"Boosted quiet voiceover by +{gain_db:.1f} dB before mixing.")
    except Exception as e:
        print(f"Warning: Voiceover RMS check failed ({e}); continuing without pre-mix gain boost.")

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
    use_pro_mix = os.getenv("SONIC_PRO_MIX", "1").strip().lower() not in {"0", "false", "no"}
    mixed_path = None
    if use_pro_mix:
        try:
            mixed_path = mix_vocals_with_music_ducking_pro(
                mix_voice_source,
                music_to_use,
                output_path,
                vocals_volume=vocals_volume,
                music_volume=music_volume,
                process_timeout=max(600, ffmpeg_timeout)
            )
        except Exception as e:
            print(f"Warning: pro mix chain failed ({e}); falling back to legacy mixer.")
    if mixed_path is None:
        mixed_path = mix_vocals_with_music_ducking(
            mix_voice_source,
            music_to_use,
            output_path,
            vocals_volume=vocals_volume,
            music_volume=music_volume,
            process_timeout=max(600, ffmpeg_timeout)
        )

    # Step 4.5: Evaluate quality and auto-correct once when quality is poor.
    print("Step 4.5: Evaluating swap quality...")
    qa = evaluate_mix_quality(mixed_path, separated['music'])
    selected_qa = qa
    print(
        "Mix QA:"
        f" rms={qa.get('rms_dbfs')},"
        f" jumps12={qa.get('jumps_gt12db')},"
        f" hf6k={qa.get('hf_ratio_ge6k')},"
        f" leak={qa.get('music_leak_corr')},"
        f" issues={qa.get('issues')}"
    )

    if qa.get("retry_recommended"):
        print("Quality issues detected, remixing once with adjusted settings...")
        retry_output = output_path.with_name(f"{output_path.stem}_retry{output_path.suffix}")

        retry_vocals_volume = min(1.45, vocals_volume * 1.15)
        retry_music_volume = max(0.42, music_volume * 0.82)
        retry_gate_threshold = 0.0032
        retry_voice_lowpass = 10200
        retry_sidechain_threshold = 0.010
        retry_sidechain_ratio = 9.0
        retry_output_makeup = 1.45

        if "possible_original_music_bleed" in qa["issues"]:
            retry_music_volume = max(0.35, retry_music_volume * 0.90)
            retry_sidechain_ratio = max(retry_sidechain_ratio, 10.0)

        if "voice_choppy_or_pumping" in qa["issues"]:
            retry_gate_threshold = 0.0028
            retry_sidechain_ratio = 6.0
            retry_sidechain_threshold = 0.014
            retry_voice_lowpass = max(retry_voice_lowpass, 10800)

        if "output_too_muffled" in qa["issues"]:
            retry_voice_lowpass = max(retry_voice_lowpass, 11800)
            retry_gate_threshold = min(retry_gate_threshold, 0.0024)
            retry_output_makeup = max(retry_output_makeup, 1.55)

        if "mix_too_quiet" in qa["issues"]:
            retry_vocals_volume = min(1.60, retry_vocals_volume * 1.15)
            retry_music_volume = max(0.35, retry_music_volume * 0.85)

        mix_vocals_with_music_ducking(
            mix_voice_source,
            music_to_use,
            retry_output,
            vocals_volume=retry_vocals_volume,
            music_volume=retry_music_volume,
            gate_threshold=retry_gate_threshold,
            process_timeout=max(600, ffmpeg_timeout),
            voice_lowpass_hz=retry_voice_lowpass,
            sidechain_threshold=retry_sidechain_threshold,
            sidechain_ratio=retry_sidechain_ratio,
            output_makeup=retry_output_makeup
        )

        retry_qa = evaluate_mix_quality(retry_output, separated['music'])
        print(
            "Retry Mix QA:"
            f" rms={retry_qa.get('rms_dbfs')},"
            f" jumps12={retry_qa.get('jumps_gt12db')},"
            f" hf6k={retry_qa.get('hf_ratio_ge6k')},"
            f" leak={retry_qa.get('music_leak_corr')},"
            f" issues={retry_qa.get('issues')}"
        )

        # Prefer retry if it has fewer issues or less severe loudness/choppiness.
        prefer_retry = len(retry_qa.get("issues", [])) < len(qa.get("issues", []))
        if not prefer_retry:
            base_rms = qa.get("rms_dbfs") if qa.get("rms_dbfs") is not None else -99
            retry_rms = retry_qa.get("rms_dbfs") if retry_qa.get("rms_dbfs") is not None else -99
            base_jumps = qa.get("jumps_gt12db") if qa.get("jumps_gt12db") is not None else 999
            retry_jumps = retry_qa.get("jumps_gt12db") if retry_qa.get("jumps_gt12db") is not None else 999
            prefer_retry = retry_rms > base_rms + 1.0 or retry_jumps + 6 < base_jumps

        if prefer_retry:
            try:
                shutil.move(str(retry_output), str(output_path))
                print("Using remixed output after QA adaptation.")
                selected_qa = retry_qa
            except Exception:
                print("Warning: Could not promote retry output; keeping original mix.")
        else:
            print("Keeping original mix after QA comparison.")

    # Step 4.6: If still too quiet, apply a conservative loudness lift.
    if selected_qa.get("rms_dbfs") is not None and selected_qa["rms_dbfs"] < -19.0:
        target_rms_dbfs = -15.5
        gain_db = min(12.0, target_rms_dbfs - selected_qa["rms_dbfs"])
        if gain_db > 0.4:
            print(f"Step 4.6: Applying post-gain loudness correction (+{gain_db:.1f} dB)...")
            loud_tmp = output_path.with_name(f"{output_path.stem}_loud{output_path.suffix}")
            try:
                _apply_post_gain_limiter(
                    output_path,
                    loud_tmp,
                    gain_db=gain_db,
                    timeout=max(300, ffmpeg_timeout)
                )
                shutil.move(str(loud_tmp), str(output_path))
                final_qa = evaluate_mix_quality(output_path, separated['music'])
                print(
                    "Post-gain Mix QA:"
                    f" rms={final_qa.get('rms_dbfs')},"
                    f" jumps12={final_qa.get('jumps_gt12db')},"
                    f" hf6k={final_qa.get('hf_ratio_ge6k')},"
                    f" leak={final_qa.get('music_leak_corr')},"
                    f" issues={final_qa.get('issues')}"
                )
            except Exception as e:
                print(f"Warning: Post-gain loudness correction failed ({e}).")

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
