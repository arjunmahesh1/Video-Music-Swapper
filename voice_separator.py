"""Voice/music separation and smart audio ducking for ads."""
import subprocess
import tempfile
from pathlib import Path
import shutil
import sys


def separate_audio(audio_path, output_dir=None):
    """Separate audio into vocals and accompaniment using Demucs.

    Uses the smaller 'htdemucs' model with segment processing for lower memory usage.

    Args:
        audio_path: Path to audio file (wav, mp3, etc.)
        output_dir: Directory for output files (uses temp dir if None)

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

    # Use smaller segments to reduce memory usage
    # htdemucs max segment is 7.8 seconds
    cmd = [
        python_exe, "-m", "demucs",
        "--two-stems=vocals",
        "--segment", "7",  # Process in 7-second chunks (within model limit)
        "-o", str(output_dir),
        str(audio_path)
    ]

    print(f"Running Demucs voice separation on {audio_path.name}...")
    print(f"Using Python: {python_exe}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Demucs stdout: {result.stdout}")
        print(f"Demucs stderr: {result.stderr}")
        raise Exception(f"Demucs separation failed: {result.stderr}")

    stem_name = audio_path.stem
    model_output = output_dir / "htdemucs" / stem_name

    vocals_path = model_output / "vocals.wav"
    music_path = model_output / "no_vocals.wav"

    if not vocals_path.exists() or not music_path.exists():
        raise Exception(f"Separation output not found at {model_output}")

    print(f"Separation complete!")
    return {
        'vocals': vocals_path,
        'music': music_path
    }


def mix_vocals_with_music_ducking(vocals_path, new_music_path, output_path,
                                   vocals_volume=1.0, music_volume=0.8):
    """Mix vocals with new music using sidechain ducking.

    The music volume automatically ducks (lowers) when voice is detected,
    and smoothly fades back up during silent parts - like a professional ad mix.

    Args:
        vocals_path: Path to vocals audio file
        new_music_path: Path to new music to mix in
        output_path: Path for output mixed audio
        vocals_volume: Volume multiplier for vocals (1.0 = original)
        music_volume: Base volume for music when no voice (0.8 default)

    Returns:
        Path to mixed audio file
    """
    output_path = Path(output_path)

    # FFmpeg sidechaincompress filter:
    # - Uses vocals as sidechain input to compress (duck) the music
    # - threshold: voice level that triggers ducking (-30dB)
    # - ratio: how much to compress (4:1 = significant ducking)
    # - attack: how fast to duck (20ms = quick response)
    # - release: how fast to fade back up (300ms = smooth)
    # - makeup: makeup gain
    filter_complex = (
        f"[0:a]volume={vocals_volume}[voice];"
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
    """Full pipeline: separate vocals from video, mix with new music + ducking.

    The final mix will automatically duck (lower) the music when voice is present,
    and smoothly fade back up during silent parts - like a professional ad.

    Args:
        video_audio_path: Path to extracted video audio
        new_music_path: Path to new music track
        output_path: Path for final mixed audio
        vocals_volume: Volume for preserved vocals
        music_volume: Base volume for new music (auto-ducks when voice detected)

    Returns:
        Path to final mixed audio
    """
    # Step 1: Separate vocals from original using Demucs
    print("Step 1: Extracting voice from original audio...")
    separated = separate_audio(video_audio_path)

    # Step 2: Mix vocals with new music using ducking
    print("Step 2: Mixing voice with new music (with auto-ducking)...")
    mixed_path = mix_vocals_with_music_ducking(
        separated['vocals'],
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
