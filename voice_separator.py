"""Voice/music separation and smart audio ducking for ads."""
import subprocess
import tempfile
from pathlib import Path
import shutil
import sys
import os
import numpy as np
import torch
import torchaudio
import librosa
import whisper


# Models to try in order of memory efficiency (quantized = smaller)
DEMUCS_MODELS = [
    ("mdx_q", "mdx_q"),           # Quantized hybrid - smallest memory footprint
    ("mdx_extra_q", "mdx_extra_q"), # Quantized with extra training
    ("mdx", "mdx"),               # Hybrid Transformer - less memory than htdemucs
    ("htdemucs", "htdemucs"),     # Default - highest quality but most memory
]


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

    # Use the same Python executable that's running this script
    python_exe = sys.executable

    # Set environment variables to reduce memory usage
    env = os.environ.copy()
    env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"  # Reduce GPU memory caching
    env["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only mode (avoids GPU memory issues)

    # If specific model requested, only try that one
    models_to_try = [(model, model)] if model else DEMUCS_MODELS

    last_error = None
    for model_name, model_id in models_to_try:
        print(f"Trying Demucs model: {model_name} (lower memory)...")
        print(f"Using Python: {python_exe}")

        # Build command - use smallest segment size to minimize memory
        # Smaller segments = less RAM needed, but slower processing
        segment_size = "5"  # Very small chunks to reduce peak memory

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
        import io
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        # Wait for process with a long timeout (10 minutes for long audio)
        try:
            stdout_data, stderr_data = process.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout_data, stderr_data = process.communicate()
            raise Exception("Demucs timed out after 10 minutes")

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


def is_singing_not_speech(audio_segment, sr=16000):
    """Analyze if an audio segment is singing rather than speech.

    Singing has:
    - More harmonic/tonal content (cleaner spectrum)
    - Higher spectral centroid (brighter sound)
    - More sustained pitch

    Speech has:
    - Noisier spectrum
    - More irregular patterns
    - Lower spectral centroid

    Returns True if it's likely singing, False if it's likely speech.
    """
    try:
        # Convert to numpy if it's a torch tensor
        if isinstance(audio_segment, torch.Tensor):
            audio_np = audio_segment.cpu().numpy()
        else:
            audio_np = audio_segment

        # Ensure float32
        audio_np = audio_np.astype(np.float32)

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

        # Decision logic:
        # Singing indicators (if any 2 are true, it's probably singing):
        is_high_centroid = mean_centroid > 2000  # Bright/musical
        is_low_zcr = mean_zcr < 0.08  # Sustained tones
        is_low_flatness = mean_flatness < 0.05  # Tonal/harmonic

        singing_score = sum([is_high_centroid, is_low_zcr, is_low_flatness])

        # Debug output
        print(f"  Spectral analysis: centroid={mean_centroid:.0f}Hz, zcr={mean_zcr:.3f}, flatness={mean_flatness:.3f}, singing_score={singing_score}/3")

        # If 2 or more singing indicators, classify as singing
        return singing_score >= 2

    except Exception as e:
        print(f"  Warning: Spectral analysis failed: {e}, assuming speech")
        return False  # If analysis fails, assume it's speech (conservative)


def detect_speech_segments(vocals_path):
    """Detect when speech is happening using Whisper speech recognition.

    Uses Whisper's word-level timestamps to identify EXACTLY when voiceover is spoken.
    This approach is like Evernote AI - it transcribes only speech and ignores singing/music.

    Returns list of (start_time, end_time) tuples in seconds.
    """
    print("Loading Whisper model for voiceover detection...")

    try:
        # Load Whisper model (base model is good balance of speed/accuracy)
        # First run downloads ~150MB model
        model = whisper.load_model("base")
        print("Transcribing audio to detect voiceover...")

        # Transcribe with word-level timestamps
        result = model.transcribe(
            str(vocals_path),
            word_timestamps=True,  # Get exact timing for each word
            language="en",  # Assume English (adjust if needed)
            task="transcribe"
        )

        print(f"Detected speech: \"{result['text'].strip()}\"")

    except Exception as e:
        print(f"Warning: Whisper failed: {e}")
        print("Falling back to VAD detection...")
        return _detect_speech_segments_fallback_vad(vocals_path)

    # Extract word-level timestamps from Whisper result
    # Whisper groups words into segments, we need to extract all word timestamps
    all_words = []
    for segment in result.get('segments', []):
        for word in segment.get('words', []):
            all_words.append(word)

    if not all_words:
        print("Warning: No words detected in transcription")
        return []

    print(f"Detected {len(all_words)} words with timestamps")

    # Merge nearby words into phrases (words within 0.3s are part of same phrase)
    speech_segments = []
    current_start = None
    current_end = None

    for i, word in enumerate(all_words):
        word_start = word['start']
        word_end = word['end']

        if current_start is None:
            # Start new segment
            current_start = word_start
            current_end = word_end
        elif word_start - current_end < 0.3:
            # Word is close to previous - extend current segment
            current_end = word_end
        else:
            # Gap detected - save current segment and start new one
            speech_segments.append((current_start, current_end))
            current_start = word_start
            current_end = word_end

    # Don't forget the last segment
    if current_start is not None:
        speech_segments.append((current_start, current_end))

    print(f"Merged into {len(speech_segments)} voiceover phrases")

    # Add minimal padding to avoid cutting off beginnings/endings
    # Keep padding small to minimize residual music from Demucs separation
    # Crossfades will handle smooth transitions
    padded_segments = []
    for start, end in speech_segments:
        padded_start = max(0, start - 0.02)  # Minimal 20ms padding before
        padded_end = end + 0.02  # Minimal 20ms padding after
        padded_segments.append((padded_start, padded_end))
        print(f"  Phrase: {padded_start:.2f}s - {padded_end:.2f}s (duration: {padded_end - padded_start:.2f}s)")

    total_speech_time = sum(e - s for s, e in padded_segments)
    print(f"Total voiceover time: {total_speech_time:.1f}s")

    return padded_segments


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

    import re
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
                                   gate_threshold=0.05):
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

    # FFmpeg filter chain:
    # 1. Apply noise gate to vocals to suppress quiet singing (keeps loud voiceover)
    # 2. Apply sidechain compression so music ducks when voice is present
    #
    # The noise gate (agate) parameters:
    # - threshold: audio level below which sound is suppressed
    # - attack: how fast the gate opens (ms)
    # - release: how fast the gate closes (ms)
    # - range: how much to reduce volume when gated (dB, negative = quieter)
    filter_complex = (
        # Apply noise gate to vocals - suppress quiet background singing
        f"[0:a]agate=threshold={gate_threshold}:attack=10:release=100:range=0.1,"
        f"volume={vocals_volume}[voice];"
        f"[1:a]volume={music_volume}[music_raw];"
        # Sidechain compress: music ducked by voice
        f"[music_raw][voice]sidechaincompress="
        f"threshold=0.02:"      # Trigger at low voice level
        f"ratio=3:"             # Duck by 3:1
        f"attack=50:"           # 50ms attack (quick duck)
        f"release=400:"         # 400ms release (smooth fade up)
        f"makeup=1"             # No makeup gain
        f"[music_ducked];"
        # Mix the voice and ducked music together
        f"[voice][music_ducked]amix=inputs=2:duration=shortest:dropout_transition=0"
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
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"Audio mixing failed: {result.stderr}")

    print(f"Mixed audio with ducking saved to: {output_path}")
    return output_path


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
    # Step 1: Separate vocals from original using Demucs
    print("Step 1: Separating voiceover from background music...")
    separated = separate_audio(video_audio_path)

    # Step 1.5: Preprocess vocals to remove residual music artifacts
    print("Step 1.5: Cleaning vocals track (removing residual music)...")
    cleaned_vocals_path = Path(tempfile.gettempdir()) / "cleaned_vocals.wav"

    # Apply aggressive filtering to suppress residual music:
    # 1. High-pass filter at 80Hz (removes low-frequency music/bass)
    # 2. Noise gate to suppress quiet background music (threshold=-35dB)
    # 3. De-esser to reduce harsh frequencies from music
    preprocess_filter = (
        "highpass=f=80,"  # Remove bass/low-frequency music
        "agate=threshold=0.018:ratio=3:attack=5:release=50,"  # Suppress quiet sounds (-35dB)
        "equalizer=f=8000:t=q:w=2:g=-3"  # Reduce harsh high frequencies
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(separated['vocals']),
        "-af", preprocess_filter,
        str(cleaned_vocals_path)
    ]

    subprocess.run(cmd, capture_output=True, text=True)

    # Step 2: Detect ONLY voiceover segments (not singing)
    print("Step 2: Detecting voiceover segments...")
    speech_segments = detect_speech_segments(cleaned_vocals_path)

    if not speech_segments:
        print("No voiceover detected - using new music only")
        # Just copy the new music as output
        shutil.copy(new_music_path, output_path)
        return output_path

    # Step 3: Extract voiceover using detected segments
    print("Step 3: Extracting voiceover segments...")
    clean_voiceover_path = Path(tempfile.gettempdir()) / "clean_voiceover.wav"

    # Build FFmpeg filter: keep ONLY detected speech segments
    enable_conds = [f"between(t,{s},{e})" for s, e in speech_segments]
    enable_expr = "+".join(enable_conds)

    # Simple volume gating: volume=1 during speech, volume=0 elsewhere
    cmd = [
        "ffmpeg", "-y",
        "-i", str(cleaned_vocals_path),
        "-af", f"volume=enable='{enable_expr}':volume=1,volume=enable='not({enable_expr})':volume=0",
        str(clean_voiceover_path)
    ]

    subprocess.run(cmd, capture_output=True, text=True)

    # Step 4: Mix clean voiceover with new music + ducking
    print("Step 4: Mixing voiceover with new music (auto-ducking enabled)...")
    mixed_path = mix_vocals_with_music_ducking(
        clean_voiceover_path,
        new_music_path,
        output_path,
        vocals_volume=vocals_volume,
        music_volume=music_volume
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