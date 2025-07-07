import streamlit as st
from pathlib import Path
import subprocess, tempfile, base64

VIDEO_DIR = Path(__file__).with_name("video")
AUDIO_DIR = Path(__file__).with_name("audio")
SAMPLE_VIDEO = VIDEO_DIR / "Gatorade.mp4"
SAMPLE_AUDIO = AUDIO_DIR / "Can't Hold Us - Macklemore & Ryan Lewis (feat. Ray Dalton).mp3"

st.title("🎵➡️🎬  Swap Video Audio Demo")

use_sample = st.checkbox("Try built-in demo assets instead", value=False)

if use_sample:
    vid_file_bytes  = SAMPLE_VIDEO.read_bytes()
    aud_file_bytes  = SAMPLE_AUDIO.read_bytes()
    vid_displayname = SAMPLE_VIDEO.name
    aud_displayname = SAMPLE_AUDIO.name
else:
    vid_uploader = st.file_uploader("Upload a video", type=["mp4","mov","mkv","webm","avi"])
    aud_uploader = st.file_uploader("Upload an audio track", type=["mp3","wav","aac","ogg","flac"])
    if vid_uploader:  vid_file_bytes, vid_displayname = vid_uploader.read(), vid_uploader.name
    else:             vid_file_bytes = vid_displayname = None
    if aud_uploader:  aud_file_bytes, aud_displayname = aud_uploader.read(), aud_uploader.name
    else:             aud_file_bytes = aud_displayname = None

if vid_file_bytes and aud_file_bytes:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎬 Original video")
        st.video(vid_file_bytes)
        st.caption(vid_displayname)
    with col2:
        st.subheader("🎧 New audio track")
        st.audio(aud_file_bytes)
        st.caption(aud_displayname)

if st.button("Swap!") and vid_file_bytes and aud_file_bytes:
    with tempfile.TemporaryDirectory() as tmp:
        v_path = Path(tmp, vid_displayname)
        a_path = Path(tmp, aud_displayname)
        out_path = Path(tmp, f"{v_path.stem}__{a_path.stem}{v_path.suffix}")
        v_path.write_bytes(vid_file_bytes)
        a_path.write_bytes(aud_file_bytes)

        cmd = [
            "ffmpeg","-y",
            "-i", str(v_path),
            "-i", str(a_path),
            "-c:v","copy","-map","0:v:0","-map","1:a:0","-shortest",
            str(out_path)
        ]
        result = subprocess.run(cmd, capture_output=True)

        if result.returncode:
            st.error(result.stderr.decode() or "FFmpeg failed")
        else:
            out_bytes = out_path.read_bytes()
            st.success("✅ Swapped!")
            st.video(out_bytes)
            st.download_button("Download result", out_bytes,
                               file_name=out_path.name, mime="video/mp4")
