import argparse
import subprocess
import sys
from pathlib import Path


def swap_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """Invoke FFmpeg to fuse `video_path` with `audio_path`."""
    cmd = [
        "ffmpeg",
        "-y",                    
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",         
        "-map", "0:v:0",        # map video stream from first input
        "-map", "1:a:0",        # map audio stream from second input
        "-shortest",            # stop when video ends (audio length > video length)
        str(output_path),
    ]

    print("▶", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode())
        raise SystemExit(f"FFmpeg failed on {video_path.name} x {audio_path.name}")

    try:
        rel = output_path.relative_to(Path.cwd())
    except ValueError:
        rel = output_path
    print(f"✅  Created {rel}\n")

def iter_media_files(directory: str, exts: set[str]):
    for f in Path(directory).glob("*"):
        if f.suffix.lower() in exts and f.is_file():
            yield f

def main():
    parser = argparse.ArgumentParser(
        description="Swap a video's audio track with a new one using FFmpeg.")
    parser.add_argument("--video", type=str,
                        help="Path to a video file (e.g. ./videos/clip.mp4)")
    parser.add_argument("--audio", type=str,
                        help="Path to an audio file (e.g. ./audio/song.mp3)")
    parser.add_argument("--output", type=str,
                        help="Path for the new video (default: <video>__<audio>.mp4)")
    parser.add_argument("--videos-dir", default="videos",
                        help="Folder to search for videos in batch mode (default: videos)")
    parser.add_argument("--audio-dir", default="audio",
                        help="Folder to search for audio tracks in batch mode (default: audio)")
    args = parser.parse_args()

    if args.video and args.audio:
        video = Path(args.video)
        audio = Path(args.audio)
        if not video.exists():
            raise SystemExit(f"❌ Video not found: {video}")
        if not audio.exists():
            raise SystemExit(f"❌ Audio not found: {audio}")

        out = Path(args.output) if args.output else Path(
            f"{video.stem}__{audio.stem}{video.suffix}")
        swap_audio(video, audio, out)
        return

    videos = list(iter_media_files(args.videos_dir, {".mp4", ".mov", ".mkv"}))
    audios = list(iter_media_files(args.audio_dir, {".mp3", ".wav", ".aac"}))

    if not videos:
        raise SystemExit(f"No videos found in ./{args.videos_dir} — add files or use --video.")
    if not audios:
        raise SystemExit(f"No audio tracks found in ./{args.audio_dir} — add files or use --audio.")

    Path("output").mkdir(exist_ok=True)

    for vid in videos:
        for aud in audios:
            out_path = Path("output") / f"{vid.stem}__{aud.stem}{vid.suffix}"
            swap_audio(vid, aud, out_path)


if __name__ == "__main__":
    main()