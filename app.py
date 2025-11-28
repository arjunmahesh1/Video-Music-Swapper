import streamlit as st
from pathlib import Path
import subprocess
import tempfile
from spotify_helper import SpotifyManager, download_spotify_track
from audio_analyzer import (
    extract_audio_from_video,
    analyze_audio_features,
    calculate_similarity,
    select_song_probabilistic
)
from musicbrainz_features import batch_get_features
from voice_separator import separate_and_remix, cleanup_demucs_output

VIDEO_DIR = Path(__file__).with_name("video")
AUDIO_DIR = Path(__file__).with_name("audio")
SAMPLE_VIDEO = VIDEO_DIR / "Gatorade.mp4"
SAMPLE_AUDIO = AUDIO_DIR / "Can't Hold Us - Macklemore & Ryan Lewis (feat. Ray Dalton).mp3"

st.set_page_config(page_title="Video Music Swapper", page_icon="🎵")
st.title("🎵➡️🎬  Video Music Swapper")
st.caption("Upload a video → Auto-match with songs YOU like → Instant personalized content")

# Initialize session state for Spotify
if 'spotify_manager' not in st.session_state:
    st.session_state.spotify_manager = SpotifyManager()
if 'spotify_authenticated' not in st.session_state:
    st.session_state.spotify_authenticated = False
if 'user_library' not in st.session_state:
    st.session_state.user_library = []
if 'audio_features_cache' not in st.session_state:
    st.session_state.audio_features_cache = {}

# Check if .env file exists
if not Path(".env").exists():
    st.warning("⚠️ No .env file found! Please create one with your Spotify credentials.")
    st.info("""
    1. Copy `.env.example` to `.env`
    2. Get credentials from https://developer.spotify.com/dashboard
    3. Add your SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET
    4. Restart the app
    """)
    st.stop()

# Spotify Authentication Section
st.sidebar.header("🎧 Spotify Integration")

# Check if we're being redirected back from Spotify with an auth code
query_params = st.query_params
auth_code = query_params.get("code", None)

if auth_code and not st.session_state.spotify_authenticated:
    # We have an auth code from Spotify redirect - exchange it for a token
    with st.spinner("Completing authentication..."):
        if st.session_state.spotify_manager.handle_redirect_code(auth_code):
            # Now try to authenticate with the cached token
            if st.session_state.spotify_manager.authenticate():
                st.session_state.spotify_authenticated = True
                st.session_state.user_library = st.session_state.spotify_manager.get_combined_library(
                    liked_limit=200, top_limit=50
                )
                # Clear the code from URL
                st.query_params.clear()
                st.rerun()

if not st.session_state.spotify_authenticated:
    if st.sidebar.button("🔐 Connect Spotify", type="primary"):
        # Generate auth URL and redirect
        auth_url = st.session_state.spotify_manager.get_auth_url()
        st.markdown(f'<meta http-equiv="refresh" content="0;url={auth_url}">', unsafe_allow_html=True)
        st.stop()
else:
    st.sidebar.success(f"✅ Spotify Connected")
    st.sidebar.caption(f"{len(st.session_state.user_library)} songs in library")
    if st.sidebar.button("🔄 Refresh Library"):
        st.session_state.user_library = st.session_state.spotify_manager.get_combined_library(
            liked_limit=200, top_limit=50
        )
        st.session_state.audio_features_cache = {}  # Clear cache
        st.sidebar.success("Refreshed!")

st.sidebar.markdown("---")

# Mode Selection
st.sidebar.subheader("⚙️ Mode")
mode = st.sidebar.radio(
    "Select mode:",
    ["🤖 Auto-Match (Smart)", "🎯 Manual Select"],
    index=0
)

is_auto_mode = mode.startswith("🤖")

if is_auto_mode:
    st.sidebar.markdown("**Auto-Match Settings:**")
    distribution = st.sidebar.selectbox(
        "Selection style:",
        ["balanced", "conservative", "adventurous"],
        help="Conservative = heavily favor top match, Adventurous = more variety"
    )
    top_n = st.sidebar.slider("Consider top N matches:", 3, 10, 5)
else:
    distribution = None
    top_n = None

st.sidebar.markdown("---")

# Voice Preservation Option
st.sidebar.subheader("🎤 Voice Preservation")
preserve_voice = st.sidebar.checkbox(
    "Keep original voiceover",
    value=False,
    help="AI extracts voice, replaces only the music, auto-ducks music during speech"
)
if preserve_voice:
    st.sidebar.caption("AI separation + auto-ducking (music lowers when voice speaks)")
    st.sidebar.caption("⏱️ Takes ~30-60 sec for AI processing")
    voice_volume = st.sidebar.slider("Voice volume:", 0.5, 1.5, 1.0, 0.1)
    music_volume = st.sidebar.slider("Music volume (ducks during voice):", 0.3, 1.0, 0.7, 0.1)
else:
    voice_volume = 1.0
    music_volume = 0.8

st.sidebar.markdown("---")

# Video Source Selection
st.subheader("🎬 Select Video")
video_source = st.radio(
    "Choose video source:",
    ["Upload Video", "Use Sample"],
    horizontal=True
)

vid_file_bytes = None
vid_displayname = None

if video_source == "Upload Video":
    vid_uploader = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "webm", "avi"])
    if vid_uploader:
        vid_file_bytes = vid_uploader.read()
        vid_displayname = vid_uploader.name

elif video_source == "Use Sample":
    if SAMPLE_VIDEO.exists():
        vid_file_bytes = SAMPLE_VIDEO.read_bytes()
        vid_displayname = SAMPLE_VIDEO.name
    else:
        st.warning(f"Sample video not found at {SAMPLE_VIDEO}")

# Audio/Song Selection based on mode
aud_file_bytes = None
aud_displayname = None
spotify_url = None
selected_song = None

if is_auto_mode:
    st.subheader("🤖 Auto-Match Mode")
    st.info("🎵 Song will be automatically selected based on similarity to the video's audio")

else:
    # Manual mode
    st.subheader("🎵 Select Audio Source")
    audio_source = st.radio(
        "Choose how to provide audio:",
        ["Upload Audio File", "Pick from Spotify Library", "Use Sample"],
        horizontal=True
    )

    if audio_source == "Upload Audio File":
        aud_uploader = st.file_uploader("Upload an audio track", type=["mp3", "wav", "aac", "ogg", "flac"])
        if aud_uploader:
            aud_file_bytes = aud_uploader.read()
            aud_displayname = aud_uploader.name

    elif audio_source == "Pick from Spotify Library":
        if not st.session_state.spotify_authenticated:
            st.warning("⚠️ Please connect your Spotify account in the sidebar first!")
        elif not st.session_state.user_library:
            st.info("No songs found. Make sure you have liked songs or listening history.")
        else:
            song_options = [""] + [song['display_name'] for song in st.session_state.user_library]
            selected_song_name = st.selectbox(
                "Choose a song:",
                options=song_options,
                index=0
            )

            if selected_song_name:
                selected_song = next((s for s in st.session_state.user_library if s['display_name'] == selected_song_name), None)
                if selected_song:
                    spotify_url = selected_song['spotify_url']
                    aud_displayname = selected_song_name
                    st.success(f"🎵 Selected: {selected_song_name}")

    elif audio_source == "Use Sample":
        if SAMPLE_AUDIO.exists():
            aud_file_bytes = SAMPLE_AUDIO.read_bytes()
            aud_displayname = SAMPLE_AUDIO.name
        else:
            st.warning(f"Sample audio not found at {SAMPLE_AUDIO}")

# Preview Section
if vid_file_bytes:
    st.markdown("---")
    st.subheader("📋 Preview")

    if not is_auto_mode:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🎬 Video**")
            st.video(vid_file_bytes)
            st.caption(vid_displayname)
        with col2:
            if aud_file_bytes:
                st.markdown("**🎧 Audio**")
                st.audio(aud_file_bytes)
                st.caption(aud_displayname)
            elif spotify_url:
                st.markdown("**🎧 Audio (from Spotify)**")
                st.caption(aud_displayname)
    else:
        st.markdown("**🎬 Video**")
        st.video(vid_file_bytes)
        st.caption(vid_displayname)

# Main Action Button
st.markdown("---")

button_enabled = False
button_text = "🔄 Swap Audio"

if is_auto_mode:
    button_enabled = vid_file_bytes is not None and st.session_state.spotify_authenticated
    button_text = "🤖 Auto-Match & Swap"
    if not st.session_state.spotify_authenticated:
        st.warning("⚠️ Connect Spotify in sidebar to use Auto-Match mode")
else:
    button_enabled = vid_file_bytes is not None and (aud_file_bytes is not None or spotify_url is not None)

if st.button(button_text, type="primary", disabled=not button_enabled):

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Save video
        v_path = tmp_path / vid_displayname
        v_path.write_bytes(vid_file_bytes)

        # AUTO-MATCH MODE
        if is_auto_mode:
            st.info("🔍 Analyzing video audio...")

            # Extract audio from video
            try:
                video_audio_path = extract_audio_from_video(v_path)
            except Exception as e:
                st.error(f"❌ Failed to extract audio from video: {str(e)}")
                st.stop()

            # Analyze video audio features
            st.info("🎵 Extracting audio features...")
            try:
                video_features = analyze_audio_features(video_audio_path)
                st.success(f"✅ Detected: Tempo={video_features['tempo']:.0f} BPM, Energy={video_features['energy']:.2f}")
            except Exception as e:
                st.error(f"❌ Failed to analyze audio: {str(e)}")
                st.stop()

            # Get audio features using heuristic estimation (Spotify removed Audio Features API)
            if not st.session_state.audio_features_cache:
                st.info(f"📊 Analyzing {len(st.session_state.user_library)} songs from your library...")

                # Use fast heuristic estimation
                with st.spinner("Generating feature estimates..."):
                    st.session_state.audio_features_cache = batch_get_features(
                        st.session_state.user_library,
                        max_songs=None  # Process all songs
                    )

                st.success(f"✅ Analyzed {len(st.session_state.audio_features_cache)} songs!")

            # Calculate similarities
            st.info("🧮 Calculating similarity scores...")
            song_scores = []
            for song in st.session_state.user_library:
                if song['id'] in st.session_state.audio_features_cache:
                    song_features = st.session_state.audio_features_cache[song['id']]
                    listening_score = song.get('listening_score', 0.3)  # Default to low if not set
                    similarity_score = calculate_similarity(video_features, song_features, listening_score)
                    song_scores.append({
                        **song,
                        'similarity_score': similarity_score,
                        'features': song_features
                    })

            # Sort by similarity (lower score = more similar)
            song_scores.sort(key=lambda x: x['similarity_score'])

            if not song_scores:
                st.error("❌ No songs with features available for matching")
                st.info("💡 Switch to Manual mode to pick a song directly")
                st.stop()

            # Show top matches
            with st.expander("🎯 Top Matches", expanded=True):
                for i, song in enumerate(song_scores[:5], 1):
                    score_pct = (1 - song['similarity_score']) * 100  # Convert to similarity percentage
                    st.write(f"{i}. **{song['display_name']}** - {score_pct:.0f}% match")

            # Select song probabilistically
            selected_song = select_song_probabilistic(song_scores, top_n=min(top_n, len(song_scores)), distribution=distribution)

            if not selected_song:
                st.error("❌ No suitable songs found in your library")
                st.stop()

            st.success(f"🎵 Auto-selected: **{selected_song['display_name']}**")
            spotify_url = selected_song['spotify_url']
            aud_displayname = selected_song['display_name']

        # Download audio from Spotify if needed
        if spotify_url:
            st.info("⬇️ Downloading song from Spotify...")
            try:
                a_path = download_spotify_track(spotify_url, tmp_path / "downloaded_track.mp3")
                st.success(f"✅ Downloaded: {aud_displayname}")
            except Exception as e:
                st.error(f"❌ Download failed: {str(e)}")
                st.stop()
        else:
            a_path = tmp_path / aud_displayname
            a_path.write_bytes(aud_file_bytes)

        # Output path
        out_path = tmp_path / f"{v_path.stem}__swapped{v_path.suffix}"

        # Voice preservation mode
        if preserve_voice:
            st.info("🎤 Extracting voice from original audio...")

            # Extract original audio from video
            original_audio = tmp_path / "original_audio.wav"
            extract_cmd = [
                "ffmpeg", "-y",
                "-i", str(v_path),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "44100", "-ac", "2",
                str(original_audio)
            ]
            subprocess.run(extract_cmd, capture_output=True)

            # Separate voice and mix with new music
            try:
                mixed_audio = tmp_path / "mixed_audio.wav"
                with st.spinner("🎵 Separating voice and mixing with new music..."):
                    separate_and_remix(
                        original_audio,
                        a_path,
                        mixed_audio,
                        vocals_volume=voice_volume,
                        music_volume=music_volume
                    )
                st.success("✅ Voice preserved and mixed with new music!")

                # Use mixed audio for final video
                a_path = mixed_audio
            except Exception as e:
                st.error(f"❌ Voice separation failed: {str(e)}")
                # Show detailed error in expandable section
                with st.expander("🔍 Full Error Details (click to expand)"):
                    st.code(str(e), language=None)
                st.warning("⚠️ Falling back to full audio replacement...")

        # Run FFmpeg to create final video
        st.info("🔧 Swapping audio track...")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(v_path),
            "-i", str(a_path),
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(out_path)
        ]

        result = subprocess.run(cmd, capture_output=True)

        # Cleanup Demucs temp files
        if preserve_voice:
            try:
                cleanup_demucs_output()
            except:
                pass

        if result.returncode:
            st.error("❌ FFmpeg failed")
            st.code(result.stderr.decode() or "Unknown error")
        else:
            out_bytes = out_path.read_bytes()
            st.success("✅ Audio swapped successfully!")

            st.subheader("🎉 Result")
            if is_auto_mode and selected_song:
                st.info(f"🎵 Replaced with: **{selected_song['display_name']}**")
            st.video(out_bytes)

            st.download_button(
                "⬇️ Download Video",
                out_bytes,
                file_name=out_path.name,
                mime="video/mp4"
            )

# Footer
st.markdown("---")
st.caption("💡 **Note**: Auto-match uses video audio analysis + smart song matching. Enable 'Keep original voiceover' to preserve speech while replacing music.")
