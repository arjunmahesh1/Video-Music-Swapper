import streamlit as st
from pathlib import Path
import subprocess
import tempfile
from spotify_helper import SpotifyManager, download_spotify_track
from audio_analyzer import (
    extract_audio_from_video,
    analyze_audio_features,
    calculate_similarity,
    calculate_genre_similarity,
    infer_video_genres,
    select_song_probabilistic
)
from voice_separator import separate_and_remix, cleanup_demucs_output
from cache_manager import (
    load_recently_used_songs,
    save_recently_used_songs
)

VIDEO_DIR = Path(__file__).with_name("video")
AUDIO_DIR = Path(__file__).with_name("audio")
SAMPLE_VIDEO = VIDEO_DIR / "Gatorade.mp4"
SAMPLE_AUDIO = AUDIO_DIR / "Can't Hold Us - Macklemore & Ryan Lewis (feat. Ray Dalton).mp3"

def run_subprocess_checked(cmd, step_name, timeout=600):
    """Run subprocess and fail fast with a useful message."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise Exception(f"{step_name} timed out after {timeout}s")

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise Exception(f"{step_name} failed: {err}")

    return result

st.set_page_config(page_title="Video Music Swapper", page_icon="🎵", layout="centered")

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'spotify_manager' not in st.session_state:
    st.session_state.spotify_manager = SpotifyManager()
if 'spotify_authenticated' not in st.session_state:
    st.session_state.spotify_authenticated = False
if 'user_library' not in st.session_state:
    st.session_state.user_library = []
if 'genre_cache' not in st.session_state:
    st.session_state.genre_cache = {}
if 'audio_features_cache' not in st.session_state:
    st.session_state.audio_features_cache = {}
if 'video_file' not in st.session_state:
    st.session_state.video_file = None
if 'video_name' not in st.session_state:
    st.session_state.video_name = None
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'selected_song' not in st.session_state:
    st.session_state.selected_song = None
if 'preserve_voice' not in st.session_state:
    st.session_state.preserve_voice = False
if 'voiceover_transcript_hint' not in st.session_state:
    st.session_state.voiceover_transcript_hint = ""
if 'recently_used_songs' not in st.session_state:
    st.session_state.recently_used_songs = load_recently_used_songs()  # Load from cache

# Check if .env file exists
if not Path(".env").exists():
    st.error("No .env file found")
    st.info("""
    Please create a .env file with your Spotify credentials:
    1. Copy .env.example to .env
    2. Get credentials from https://developer.spotify.com/dashboard
    3. Add your SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET
    4. Restart the app
    """)
    st.stop()

# Handle Spotify OAuth redirect
query_params = st.query_params
auth_code = query_params.get("code", None)

if auth_code and not st.session_state.spotify_authenticated:
    with st.spinner("Completing authentication..."):
        if st.session_state.spotify_manager.handle_redirect_code(auth_code):
            if st.session_state.spotify_manager.authenticate():
                st.session_state.spotify_authenticated = True
                with st.spinner("Loading your music library..."):
                    st.session_state.user_library = st.session_state.spotify_manager.get_combined_library(
                        liked_limit=500, top_limit=50
                    )
                st.query_params.clear()
                st.rerun()

# Header
st.title("Video Music Swapper")
st.markdown("---")

# Progress indicator
progress_labels = {
    1: "Connect Spotify",
    2: "Upload Video",
    3: "Select Mode",
    4: "Process"
}

# Show current step
col1, col2, col3, col4 = st.columns(4)
for i, (step_num, label) in enumerate(progress_labels.items()):
    with [col1, col2, col3, col4][i]:
        if st.session_state.step == step_num:
            st.markdown(f"**→ {label}**")
        elif st.session_state.step > step_num:
            st.markdown(f"✓ {label}")
        else:
            st.markdown(f"{label}")

st.markdown("---")

# Callback function for "Process Another Video" button
def reset_to_upload():
    """Reset session state to go back to upload step."""
    st.session_state.step = 2
    st.session_state.video_file = None
    st.session_state.video_name = None
    st.session_state.selected_song = None
    st.session_state.mode = None

# STEP 1: Connect Spotify
if st.session_state.step == 1:
    st.subheader("Step 1: Connect Spotify")

    if not st.session_state.spotify_authenticated:
        st.write("Connect your Spotify account to access your music library.")

        if st.button("Connect Spotify Account", type="primary"):
            auth_url = st.session_state.spotify_manager.get_auth_url()
            st.markdown(f'<meta http-equiv="refresh" content="0;url={auth_url}">', unsafe_allow_html=True)
            st.stop()
    else:
        st.success(f"Connected • {len(st.session_state.user_library)} songs in library")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Continue", type="primary"):
                st.session_state.step = 2
                st.rerun()
        with col2:
            if st.button("Disconnect"):
                import os
                if os.path.exists(".spotify_cache"):
                    os.remove(".spotify_cache")
                st.session_state.spotify_authenticated = False
                st.session_state.user_library = []
                st.session_state.genre_cache = {}
                st.session_state.audio_features_cache = {}
                st.session_state.spotify_manager = SpotifyManager()
                st.rerun()

# STEP 2: Upload Video
elif st.session_state.step == 2:
    st.subheader("Step 2: Upload Video")

    video_source = st.radio(
        "Choose video source:",
        ["Upload File", "Use Sample Video"],
        horizontal=True
    )

    vid_file_bytes = None
    vid_displayname = None

    if video_source == "Upload File":
        vid_uploader = st.file_uploader("Select a video file", type=["mp4", "mov", "mkv", "webm", "avi"])
        if vid_uploader:
            vid_file_bytes = vid_uploader.read()
            vid_displayname = vid_uploader.name

    elif video_source == "Use Sample Video":
        if SAMPLE_VIDEO.exists():
            vid_file_bytes = SAMPLE_VIDEO.read_bytes()
            vid_displayname = SAMPLE_VIDEO.name
        else:
            st.warning(f"Sample video not found at {SAMPLE_VIDEO}")

    if vid_file_bytes:
        st.video(vid_file_bytes)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Back"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("Continue", type="primary"):
                st.session_state.video_file = vid_file_bytes
                st.session_state.video_name = vid_displayname
                st.session_state.step = 3
                st.rerun()

# STEP 3: Select Mode
elif st.session_state.step == 3:
    st.subheader("Step 3: Select Mode")

    mode = st.radio(
        "How would you like to select the music?",
        ["Auto-Match (Recommended)", "Manual Selection"],
        help="Auto-Match uses AI to find the best matching song from your library"
    )

    st.session_state.mode = mode

    # Voice preservation option
    st.markdown("---")
    preserve_voice = st.checkbox(
        "Preserve original voiceover",
        value=st.session_state.preserve_voice,
        help="AI extracts speech and mixes it with new music (often 5-10 minutes on CPU; faster with GPU)"
    )
    st.session_state.preserve_voice = preserve_voice

    if preserve_voice:
        st.info("Voice will be extracted and mixed with the new music track")
        with st.expander("Advanced (optional): transcript hint", expanded=False):
            transcript_hint = st.text_area(
                "Timestamped transcript (optional)",
                value=st.session_state.voiceover_transcript_hint,
                height=140,
                help=(
                    "Leave blank for fully automatic mode. "
                    "If provided, timestamps (e.g. `0:05`) can further reduce lyric bleed."
                ),
                placeholder=(
                    "Everything about the game has changed.\n"
                    "0:05\n"
                    "Except for the most important thing.\n"
                    "0:08\n"
                    "..."
                )
            )
            st.session_state.voiceover_transcript_hint = transcript_hint

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Continue", type="primary"):
            st.session_state.step = 4
            st.rerun()

# STEP 4: Process
elif st.session_state.step == 4:
    st.subheader("Step 4: Process")

    # Auto-Match Mode
    if st.session_state.mode == "Auto-Match (Recommended)":
        if st.button("Start Auto-Match", type="primary"):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)

                # Save video
                v_path = tmp_path / st.session_state.video_name
                v_path.write_bytes(st.session_state.video_file)

                # Extract and analyze video audio
                with st.spinner("Analyzing video audio..."):
                    try:
                        video_audio_path = extract_audio_from_video(v_path)
                        video_features = analyze_audio_features(video_audio_path)
                        video_genres = infer_video_genres(video_features)

                        st.success(f"Detected: {video_features['tempo']:.0f} BPM, Energy: {video_features['energy']:.2f}")
                        st.info(f"Inferred genres: {', '.join(video_genres)}")
                    except Exception as e:
                        st.error(f"Failed to analyze audio: {str(e)}")
                        st.stop()

                # Fetch genres for all songs if not cached
                if not st.session_state.genre_cache:
                    with st.spinner("Fetching genre data from Spotify..."):
                        artist_ids = list(set(song['artist_id'] for song in st.session_state.user_library if 'artist_id' in song))
                        st.session_state.genre_cache = st.session_state.spotify_manager.get_artist_genres(artist_ids)
                        st.success(f"Loaded genres for {len(st.session_state.genre_cache)} artists")

                # Get audio features using heuristic estimation
                if not st.session_state.audio_features_cache:
                    with st.spinner(f"Analyzing {len(st.session_state.user_library)} songs..."):
                        from musicbrainz_features import batch_get_features
                        st.session_state.audio_features_cache = batch_get_features(st.session_state.user_library)

                # Calculate similarities with genre matching
                with st.spinner("Finding best matches..."):
                    song_scores = []
                    for song in st.session_state.user_library:
                        if song['id'] in st.session_state.audio_features_cache:
                            song_features = st.session_state.audio_features_cache[song['id']]

                            # Get song genres from cache
                            song_genres = st.session_state.genre_cache.get(song.get('artist_id', ''), [])

                            # Calculate genre similarity
                            genre_sim = calculate_genre_similarity(video_genres, song_genres)

                            # Calculate overall similarity
                            listening_score = song.get('listening_score', 0.3)
                            similarity_score = calculate_similarity(
                                video_features,
                                song_features,
                                listening_score,
                                genre_sim
                            )

                            song_scores.append({
                                **song,
                                'similarity_score': similarity_score,
                                'genre_similarity': genre_sim,
                                'genres': song_genres,
                                'features': song_features
                            })

                    # Add diversity to scoring
                    import random
                    for song in song_scores:
                        # 1. Add small random variance (±5%) to shuffle similar songs
                        random_factor = random.uniform(0.95, 1.05)
                        song['similarity_score'] *= random_factor

                        # 2. Penalize recently used songs (last 10 selections)
                        if song['id'] in st.session_state.recently_used_songs:
                            # Add penalty based on how recently used (more recent = bigger penalty)
                            recency_index = st.session_state.recently_used_songs.index(song['id'])
                            penalty = 0.15 * (1.0 - recency_index / len(st.session_state.recently_used_songs))
                            song['similarity_score'] += penalty  # Increase score = worse match

                    # Sort by similarity (lower = better)
                    song_scores.sort(key=lambda x: x['similarity_score'])

                    if not song_scores:
                        st.error("No songs available for matching")
                        st.stop()

                    # Show top 5 matches
                    st.markdown("### Top Matches")
                    for i, song in enumerate(song_scores[:5], 1):
                        match_pct = (1 - song['similarity_score']) * 100
                        genre_pct = song['genre_similarity'] * 100
                        st.write(f"{i}. **{song['display_name']}** - {match_pct:.0f}% match (Genre: {genre_pct:.0f}%)")
                        if song['genres']:
                            st.caption(f"   Genres: {', '.join(song['genres'][:3])}")

                    # Select from top matches probabilistically (adds variety)
                    # 50% chance = top match, 30% = 2nd, 20% = 3rd
                    selected_song = select_song_probabilistic(song_scores, top_n=5, distribution='balanced')
                    st.session_state.selected_song = selected_song

                    # Track this selection for diversity in future runs
                    if selected_song['id'] not in st.session_state.recently_used_songs:
                        st.session_state.recently_used_songs.insert(0, selected_song['id'])
                        # Keep only last 10 selections
                        if len(st.session_state.recently_used_songs) > 10:
                            st.session_state.recently_used_songs = st.session_state.recently_used_songs[:10]

                    # Save to cache for persistence across app restarts
                    save_recently_used_songs(st.session_state.recently_used_songs)

                    st.success(f"Selected: **{selected_song['display_name']}**")

                # Download song
                with st.spinner("Downloading song..."):
                    try:
                        spotify_url = selected_song['spotify_url']
                        search_query = f"{selected_song.get('artist', '')} - {selected_song.get('name', '')}".strip(" -")
                        a_path = download_spotify_track(
                            spotify_url,
                            tmp_path / "downloaded_track.mp3",
                            search_query=search_query or selected_song.get('display_name')
                        )
                    except Exception as e:
                        st.error(f"Download failed: {str(e)}")
                        st.info(
                            "If this shows SABR/403, set YTDLP_PO_TOKEN plus "
                            "YTDLP_COOKIE_FILE in .env, or set "
                            "YTDLP_COOKIES_FROM_BROWSER=firefox. "
                            "You can also switch to Manual Selection -> Upload Audio File."
                        )
                        st.stop()

                # Voice preservation
                if st.session_state.preserve_voice:
                    with st.spinner("Extracting and mixing voiceover..."):
                        # Extract original audio
                        original_audio = tmp_path / "original_audio.wav"
                        extract_cmd = [
                            "ffmpeg", "-y",
                            "-i", str(v_path),
                            "-vn", "-acodec", "pcm_s16le",
                            "-ar", "32000", "-ac", "1",
                            str(original_audio)
                        ]
                        try:
                            run_subprocess_checked(
                                extract_cmd,
                                "Extract video audio for voice preservation",
                                timeout=180
                            )

                            # Separate and mix
                            mixed_audio = tmp_path / "mixed_audio.wav"
                            separate_and_remix(
                                original_audio,
                                a_path,
                                mixed_audio,
                                vocals_volume=1.0,
                                music_volume=0.7,
                                transcript_hint_text=(st.session_state.voiceover_transcript_hint or "").strip() or None
                            )
                            a_path = mixed_audio
                        except Exception as e:
                            st.error(f"Voice separation failed: {str(e)}")
                            st.warning("Falling back to full audio replacement")

                # Create final video
                with st.spinner("Creating final video..."):
                    out_path = tmp_path / f"{v_path.stem}_swapped{v_path.suffix}"
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
                    mux_error = None
                    try:
                        run_subprocess_checked(cmd, "Create final video", timeout=900)
                    except Exception as e:
                        mux_error = str(e)

                    # Cleanup
                    if st.session_state.preserve_voice:
                        try:
                            cleanup_demucs_output()
                        except:
                            pass

                    if mux_error:
                        st.error("Video processing failed")
                        st.code(mux_error)
                    else:
                        out_bytes = out_path.read_bytes()
                        st.success("Complete!")

                        st.markdown("### Result")
                        st.info(f"Replaced with: **{selected_song['display_name']}**")
                        st.video(out_bytes)

                        st.download_button(
                            "Download Video",
                            out_bytes,
                            file_name=out_path.name,
                            mime="video/mp4"
                        )

                        with st.expander("Show Original Video"):
                            st.video(st.session_state.video_file)

                        st.button("Process Another Video", key="auto_process_another", on_click=reset_to_upload)

    # Manual Mode
    else:
        st.write("Manual selection mode")

        audio_source = st.radio(
            "Audio source:",
            ["Upload Audio File", "Pick from Spotify Library", "Use Sample"],
            horizontal=True
        )

        aud_file_bytes = None
        aud_displayname = None
        spotify_url = None

        if audio_source == "Upload Audio File":
            aud_uploader = st.file_uploader("Upload audio", type=["mp3", "wav", "aac", "ogg", "flac"])
            if aud_uploader:
                aud_file_bytes = aud_uploader.read()
                aud_displayname = aud_uploader.name

        elif audio_source == "Pick from Spotify Library":
            if not st.session_state.user_library:
                st.warning("No songs in library")
            else:
                song_options = [song['display_name'] for song in st.session_state.user_library]
                selected_song_name = st.selectbox("Choose a song:", song_options)

                if selected_song_name:
                    selected_song = next((s for s in st.session_state.user_library if s['display_name'] == selected_song_name), None)
                    if selected_song:
                        spotify_url = selected_song['spotify_url']
                        aud_displayname = selected_song_name
                        st.success(f"Selected: {selected_song_name}")

        elif audio_source == "Use Sample":
            if SAMPLE_AUDIO.exists():
                aud_file_bytes = SAMPLE_AUDIO.read_bytes()
                aud_displayname = SAMPLE_AUDIO.name
            else:
                st.warning(f"Sample audio not found")

        if aud_file_bytes or spotify_url:
            if st.button("Process Video", type="primary"):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)

                    # Save video
                    v_path = tmp_path / st.session_state.video_name
                    v_path.write_bytes(st.session_state.video_file)

                    # Get audio
                    if spotify_url:
                        with st.spinner("Downloading song..."):
                            try:
                                a_path = download_spotify_track(
                                    spotify_url,
                                    tmp_path / "downloaded_track.mp3",
                                    search_query=aud_displayname
                                )
                            except Exception as e:
                                st.error(f"Download failed: {str(e)}")
                                st.info(
                                    "If this shows SABR/403, set YTDLP_PO_TOKEN plus "
                                    "YTDLP_COOKIE_FILE in .env, or set "
                                    "YTDLP_COOKIES_FROM_BROWSER=firefox. "
                                    "You can switch to Upload Audio File to skip Spotify download."
                                )
                                st.stop()
                    else:
                        a_path = tmp_path / aud_displayname
                        a_path.write_bytes(aud_file_bytes)

                    # Voice preservation
                    if st.session_state.preserve_voice:
                        with st.spinner("Extracting and mixing voiceover..."):
                            original_audio = tmp_path / "original_audio.wav"
                            extract_cmd = [
                                "ffmpeg", "-y",
                                "-i", str(v_path),
                                "-vn", "-acodec", "pcm_s16le",
                                "-ar", "32000", "-ac", "1",
                                str(original_audio)
                            ]
                            try:
                                run_subprocess_checked(
                                    extract_cmd,
                                    "Extract video audio for voice preservation",
                                    timeout=180
                                )

                                mixed_audio = tmp_path / "mixed_audio.wav"
                                separate_and_remix(
                                    original_audio,
                                    a_path,
                                    mixed_audio,
                                    vocals_volume=1.0,
                                    music_volume=0.7,
                                    transcript_hint_text=(st.session_state.voiceover_transcript_hint or "").strip() or None
                                )
                                a_path = mixed_audio
                            except Exception as e:
                                st.error(f"Voice separation failed: {str(e)}")
                                st.warning("Falling back to full audio replacement")

                    # Create final video
                    with st.spinner("Creating final video..."):
                        out_path = tmp_path / f"{v_path.stem}_swapped{v_path.suffix}"
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
                        mux_error = None
                        try:
                            run_subprocess_checked(cmd, "Create final video", timeout=900)
                        except Exception as e:
                            mux_error = str(e)

                        if st.session_state.preserve_voice:
                            try:
                                cleanup_demucs_output()
                            except:
                                pass

                        if mux_error:
                            st.error("Video processing failed")
                            st.code(mux_error)
                        else:
                            out_bytes = out_path.read_bytes()
                            st.success("Complete!")

                            st.markdown("### Result")
                            st.video(out_bytes)

                            st.download_button(
                                "Download Video",
                                out_bytes,
                                file_name=out_path.name,
                                mime="video/mp4"
                            )

                            with st.expander("Show Original Video"):
                                st.video(st.session_state.video_file)

                            st.button("Process Another Video", key="manual_process_another", on_click=reset_to_upload)

        if st.button("Back"):
            st.session_state.step = 3
            st.rerun()
