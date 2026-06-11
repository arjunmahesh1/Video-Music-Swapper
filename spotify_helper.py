"""Spotify integration for fetching user's liked songs and downloading tracks."""
import os
import shutil
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from pathlib import Path
import subprocess
import tempfile
import time

load_dotenv()
SPOTDL_PRIMARY_TIMEOUT_SECONDS = 240
SPOTDL_FALLBACK_TIMEOUT_SECONDS = 300
YTDLP_PRIMARY_TIMEOUT_SECONDS = 120
YTDLP_AUTH_TIMEOUT_SECONDS = 180
SUPPORTED_AUDIO_EXTENSIONS = (".mp3", ".m4a", ".webm", ".opus", ".ogg", ".wav", ".flac")

class SpotifyManager:
    def __init__(self):
        self.scope = "user-library-read user-top-read"
        self.sp = None

    def authenticate(self):
        """Authenticate with Spotify using OAuth."""
        try:
            auth_manager = SpotifyOAuth(
                client_id=os.getenv("SPOTIPY_CLIENT_ID"),
                client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
                redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
                scope=self.scope,
                cache_path=".spotify_cache",
                open_browser=False,  # Don't auto-open browser
                show_dialog=False     # Don't show dialog prompts
            )

            # Check if we have a cached token
            token_info = auth_manager.get_cached_token()

            if not token_info:
                # No token available - need to authenticate first
                return False

            self.sp = spotipy.Spotify(auth=token_info['access_token'], auth_manager=auth_manager)

            # Test the connection
            self.sp.current_user()
            return True
        except Exception as e:
            print(f"Authentication failed: {e}")
            return False

    def get_auth_url(self):
        """Get the Spotify authorization URL for manual authentication."""
        auth_manager = SpotifyOAuth(
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
            redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
            scope=self.scope,
            cache_path=".spotify_cache",
            open_browser=False,
            show_dialog=True  # Force login page to appear (allow account switching)
        )
        return auth_manager.get_authorize_url()

    def handle_redirect_code(self, code):
        """Handle the authorization code from redirect URL."""
        try:
            auth_manager = SpotifyOAuth(
                client_id=os.getenv("SPOTIPY_CLIENT_ID"),
                client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
                redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
                scope=self.scope,
                cache_path=".spotify_cache",
                open_browser=False,
                show_dialog=True  # Force login page to appear (allow account switching)
            )
            # Exchange code for token. check_cache=False is critical: with the
            # default, spotipy returns the previously cached account's token
            # and silently discards the new authorization code, making account
            # switching impossible.
            token_info = auth_manager.get_access_token(code, as_dict=True, check_cache=False)
            return token_info is not None
        except Exception as e:
            print(f"Failed to handle redirect code: {e}")
            return False

    def get_liked_songs(self, limit=50):
        """Fetch user's liked songs.

        Args:
            limit: Maximum number of songs to fetch (Spotify API max is 50 per request, will paginate)

        Returns:
            List of dicts with keys: name, artist, uri, spotify_url
        """
        if not self.sp:
            raise Exception("Not authenticated. Call authenticate() first.")

        songs = []
        offset = 0
        batch_size = 50  # Spotify's max per request

        while len(songs) < limit:
            # Fetch next batch (max 50 at a time)
            batch_limit = min(batch_size, limit - len(songs))
            results = self.sp.current_user_saved_tracks(limit=batch_limit, offset=offset)

            if not results['items']:
                break  # No more songs

            for item in results['items']:
                track = item['track']
                artist_id = track['artists'][0]['id']

                songs.append({
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'artist_id': artist_id,
                    'uri': track['uri'],
                    'spotify_url': track['external_urls']['spotify'],
                    'display_name': f"{track['artists'][0]['name']} - {track['name']}",
                    'id': track['id']
                })

            # Check if there are more songs available
            if not results['next']:
                break

            offset += batch_size

        return songs

    def get_top_tracks(self, limit=50, time_range='medium_term'):
        """Fetch user's top tracks.

        Args:
            limit: Maximum number of tracks (max 50)
            time_range: 'short_term' (4 weeks), 'medium_term' (6 months), 'long_term' (years)

        Returns:
            List of track dicts
        """
        if not self.sp:
            raise Exception("Not authenticated. Call authenticate() first.")

        songs = []
        results = self.sp.current_user_top_tracks(limit=limit, time_range=time_range)

        for track in results['items']:
            artist_id = track['artists'][0]['id']

            songs.append({
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'artist_id': artist_id,
                'uri': track['uri'],
                'spotify_url': track['external_urls']['spotify'],
                'display_name': f"{track['artists'][0]['name']} - {track['name']}",
                'id': track['id']
            })

        return songs

    def get_artist_genres(self, artist_ids):
        """Fetch genres for multiple artists at once.

        Args:
            artist_ids: List of Spotify artist IDs

        Returns:
            Dict mapping artist_id -> list of genres
        """
        if not self.sp:
            raise Exception("Not authenticated. Call authenticate() first.")

        genres_map = {}

        # Spotify allows max 50 artists per request
        for i in range(0, len(artist_ids), 50):
            batch = artist_ids[i:i+50]
            try:
                artists = self.sp.artists(batch)
                for artist in artists['artists']:
                    if artist:  # Sometimes can be None
                        genres_map[artist['id']] = artist.get('genres', [])
            except Exception as e:
                print(f"Warning: Failed to fetch genres for batch: {e}")
                # Fill with empty genres for failed artists
                for artist_id in batch:
                    if artist_id not in genres_map:
                        genres_map[artist_id] = []

        return genres_map

    def get_combined_library(self, liked_limit=200, top_limit=50):
        """Get combined set of liked songs and top tracks (deduplicated).

        Top tracks are heavily weighted as "frequently listened" songs.

        Args:
            liked_limit: Max liked songs to fetch
            top_limit: Max top tracks to fetch

        Returns:
            List of unique tracks with 'listening_score' (0-1, higher = more listened to)
        """
        # Get top tracks from different time periods - these are songs you ACTUALLY listen to
        top_recent = self.get_top_tracks(limit=top_limit, time_range='short_term')  # Last 4 weeks
        top_medium = self.get_top_tracks(limit=top_limit, time_range='medium_term')  # Last 6 months

        # Track URIs and assign listening scores
        track_scores = {}

        # Recent top tracks = highest priority
        for i, song in enumerate(top_recent):
            track_scores[song['uri']] = {
                'song': song,
                'listening_score': 1.0 - (i * 0.01)  # 1.0 for #1, decreasing slightly
            }

        # Medium-term top tracks = high priority (if not already in recent)
        for i, song in enumerate(top_medium):
            if song['uri'] not in track_scores:
                track_scores[song['uri']] = {
                    'song': song,
                    'listening_score': 0.7 - (i * 0.005)
                }

        # Liked songs = lower priority (may not actually listen much)
        liked = self.get_liked_songs(limit=liked_limit)
        for i, song in enumerate(liked):
            if song['uri'] not in track_scores:
                track_scores[song['uri']] = {
                    'song': song,
                    'listening_score': 0.3 - (i * 0.001)
                }

        # Build final list with listening scores
        combined = []
        for uri, data in track_scores.items():
            song = data['song']
            song['listening_score'] = data['listening_score']
            combined.append(song)

        # Sort by listening score (most listened first)
        combined.sort(key=lambda x: x.get('listening_score', 0), reverse=True)

        return combined

    def get_audio_features_batch(self, track_ids):
        """Get audio features for multiple tracks at once.

        Args:
            track_ids: List of Spotify track IDs

        Returns:
            Dict mapping track_id -> audio features
        """
        if not self.sp:
            raise Exception("Not authenticated. Call authenticate() first.")

        # Spotify API allows max 100 tracks per request
        features_map = {}

        # Try fetching one track first to test if API access works
        if len(track_ids) > 0:
            try:
                test_result = self.sp.audio_features([track_ids[0]])
                print(f"✓ Successfully fetched test audio features")
            except Exception as e:
                print(f"✗ Test audio features request failed: {str(e)}")
                # If even a single track fails, the API access is blocked
                # Return empty dict - the error will be caught upstream
                return {}

        # Process in smaller batches (50 instead of 100) to avoid issues
        for i in range(0, len(track_ids), 50):
            batch = track_ids[i:i+50]

            try:
                results = self.sp.audio_features(batch)

                for track_id, features in zip(batch, results):
                    if features:  # Some tracks may not have audio features
                        features_map[track_id] = {
                            'tempo': features['tempo'],
                            'energy': features['energy'],
                            'valence': features['valence'],
                            'danceability': features['danceability'],
                            'acousticness': features['acousticness'],
                            'instrumentalness': features['instrumentalness'],
                            'speechiness': features['speechiness']
                        }
                print(f"✓ Fetched features for batch {i//50 + 1}")
            except Exception as e:
                print(f"✗ Batch {i//50 + 1} failed: {str(e)}")
                # Continue with next batch instead of failing completely
                continue

        if not features_map:
            print(f"✗ CRITICAL: No audio features could be fetched for any songs!")
            print(f"✗ This indicates your Spotify app may not have API access enabled")

        return features_map

    def get_user_playlists(self):
        """Fetch user's playlists."""
        if not self.sp:
            raise Exception("Not authenticated. Call authenticate() first.")

        playlists = []
        results = self.sp.current_user_playlists()

        for item in results['items']:
            playlists.append({
                'name': item['name'],
                'id': item['id'],
                'tracks_count': item['tracks']['total']
            })

        return playlists

def _download_with_ytdlp_search(search_query, output_path, cookie_file=None, browser_sources=None):
    """Fallback downloader using direct yt-dlp search with multiple auth strategies."""
    if browser_sources is None:
        browser_sources = []

    def _find_audio_files(run_directory):
        candidates = []
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            candidates.extend(run_directory.rglob(f"*{ext}"))
        return [p for p in candidates if p.is_file()]

    ytdlp_po_token = os.getenv("YTDLP_PO_TOKEN", "").strip()
    # Detect placeholder values that were never configured
    if ytdlp_po_token and ("your_" in ytdlp_po_token.lower() or "xxx" in ytdlp_po_token.lower()):
        ytdlp_po_token = ""
    base_cmd = [
        "yt-dlp",
        f"ytsearch5:{search_query} official audio",
        "--no-playlist",
        "--no-progress",
        "--no-update",
        "--restrict-filenames",
        "--socket-timeout", "20",
        "--retries", "2",
        "--fragment-retries", "2",
        "--extractor-retries", "2",
        "--js-runtimes", "node",
        "--extractor-args", "youtube:player_client=web,default",
        "--match-filter", "duration>=60",
        "--playlist-items", "1",
        "-x",
        "--audio-format", "mp3",
    ]

    if ytdlp_po_token:
        base_cmd.extend([
            "--extractor-args", f"youtube:po_token=web+{ytdlp_po_token}"
        ])

    proxy = os.getenv("YTDLP_PROXY")
    if proxy:
        base_cmd.extend(["--proxy", proxy])

    attempts = []

    if cookie_file and Path(cookie_file).exists():
        attempts.append({
            "name": "yt-dlp-cookie-file",
            "timeout": YTDLP_AUTH_TIMEOUT_SECONDS,
            "extra": ["--cookies", cookie_file]
        })

    for browser in browser_sources:
        attempts.append({
            "name": f"yt-dlp-browser-{browser}",
            "timeout": YTDLP_AUTH_TIMEOUT_SECONDS,
            "extra": ["--cookies-from-browser", browser]
        })

    attempts.append({
        "name": "yt-dlp-anonymous",
        "timeout": YTDLP_PRIMARY_TIMEOUT_SECONDS,
        "extra": []
    })

    errors = []

    for attempt in attempts:
        run_dir = output_path.parent / f".ytdlp_run_{int(time.time() * 1000)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        output_template = run_dir / "%(title).90s.%(ext)s"

        cmd = base_cmd + attempt["extra"] + ["-o", str(output_template)]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=attempt["timeout"]
            )
        except subprocess.TimeoutExpired:
            shutil.rmtree(run_dir, ignore_errors=True)
            errors.append(f"{attempt['name']}: timed out after {attempt['timeout']}s")
            continue

        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip()[-500:]
            if not stderr_tail:
                stderr_tail = "yt-dlp failed"
            errors.append(f"{attempt['name']}: {stderr_tail}")
            shutil.rmtree(run_dir, ignore_errors=True)
            continue

        downloaded_files = _find_audio_files(run_dir)
        if not downloaded_files:
            errors.append(f"{attempt['name']}: succeeded but no audio file was created")
            shutil.rmtree(run_dir, ignore_errors=True)
            continue

        latest_file = max(downloaded_files, key=lambda p: p.stat().st_mtime)
        shutil.move(str(latest_file), str(output_path))
        shutil.rmtree(run_dir, ignore_errors=True)
        return output_path

    raise Exception(" | ".join(errors) if errors else "yt-dlp failed")


def download_spotify_track(spotify_url, output_path=None, search_query=None):
    """Download a Spotify track using yt-dlp first, then spotdl fallback.

    Args:
        spotify_url: Spotify track URL or URI
        output_path: Path to save the downloaded file (default: temp file)
        search_query: Artist/title text for yt-dlp search fallback

    Returns:
        Path to downloaded file
    """
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = Path(temp_dir) / "spotify_track.mp3"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    spotify_client_id = os.getenv("SPOTIPY_CLIENT_ID")
    spotify_client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    spotdl_cookie_file = os.getenv("SPOTDL_COOKIE_FILE")
    ytdlp_cookie_file = os.getenv("YTDLP_COOKIE_FILE") or spotdl_cookie_file
    ytdlp_po_token = os.getenv("YTDLP_PO_TOKEN", "").strip()
    # Detect placeholder values that were never configured
    if ytdlp_po_token and ("your_" in ytdlp_po_token.lower() or "xxx" in ytdlp_po_token.lower()):
        ytdlp_po_token = ""
    browsers_csv = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "chrome,edge,firefox")
    ytdlp_browser_sources = [b.strip() for b in browsers_csv.split(",") if b.strip()]
    errors = []

    def _find_audio_files(run_directory):
        candidates = []
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            candidates.extend(run_directory.rglob(f"*{ext}"))
        return [p for p in candidates if p.is_file()]

    # Fast and currently more reliable path for your setup.
    if search_query:
        try:
            return _download_with_ytdlp_search(
                search_query=search_query,
                output_path=output_path,
                cookie_file=ytdlp_cookie_file,
                browser_sources=ytdlp_browser_sources
            )
        except Exception as e:
            errors.append(f"yt-dlp-search: {e}")

    ytdlp_blocked = any(
        token in " ".join(errors).lower()
        for token in ["sabr", "403", "forbidden"]
    )
    spotdl_primary_timeout = SPOTDL_PRIMARY_TIMEOUT_SECONDS
    spotdl_fallback_timeout = SPOTDL_FALLBACK_TIMEOUT_SECONDS
    if ytdlp_blocked:
        # If yt-dlp clearly hit a provider block, avoid waiting many extra minutes.
        spotdl_primary_timeout = min(spotdl_primary_timeout, 90)
        spotdl_fallback_timeout = min(spotdl_fallback_timeout, 120)

    # Fallback: spotdl with multiple providers.
    run_dir = output_path.parent / f".spotdl_run_{int(time.time() * 1000)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_template = run_dir / "{artists} - {title}.{output-ext}"

    base_cmd = [
        "spotdl",
        "download",
        spotify_url,
        "--output", str(output_template),
        "--format", "mp3",
        "--max-retries", "1",
        "--threads", "2",
        "--restrict", "ascii",
        "--print-errors",
    ]

    ytdlp_args_parts = [
        "--no-update",
        "--js-runtimes", "node",
        "--extractor-args", "youtube:player_client=web,default",
    ]
    if ytdlp_po_token:
        ytdlp_args_parts.extend(["--extractor-args", f"youtube:po_token=web+{ytdlp_po_token}"])
    if ytdlp_cookie_file and Path(ytdlp_cookie_file).exists():
        cookie_path = str(Path(ytdlp_cookie_file))
        if " " in cookie_path:
            cookie_path = f"\"{cookie_path}\""
        ytdlp_args_parts.extend(["--cookies", cookie_path])
    base_cmd.extend(["--yt-dlp-args", " ".join(ytdlp_args_parts)])

    if spotify_client_id and spotify_client_secret:
        base_cmd.extend([
            "--client-id", spotify_client_id,
            "--client-secret", spotify_client_secret,
        ])

    if spotdl_cookie_file and Path(spotdl_cookie_file).exists():
        base_cmd.extend(["--cookie-file", spotdl_cookie_file])

    attempt_profiles = [
        {
            "name": "youtube-music-first",
            "timeout": spotdl_primary_timeout,
            "extra_args": ["--audio", "youtube-music", "youtube"]
        },
        {
            "name": "youtube-fallback",
            "timeout": spotdl_fallback_timeout,
            "extra_args": ["--audio", "youtube", "piped", "--dont-filter-results"]
        },
    ]

    for profile in attempt_profiles:
        cmd = base_cmd + profile["extra_args"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=profile["timeout"]
            )
        except subprocess.TimeoutExpired:
            errors.append(
                f"{profile['name']}: timed out after {profile['timeout']}s"
            )
            continue

        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip()[-400:]
            if not stderr_tail:
                stderr_tail = "unknown spotdl error"
            errors.append(f"{profile['name']}: {stderr_tail}")
            continue

        downloaded_files = _find_audio_files(run_dir)
        if not downloaded_files:
            stderr_tail = (result.stderr or "").strip()[-300:]
            stdout_tail = (result.stdout or "").strip()[-300:]
            details = stderr_tail or stdout_tail or "command succeeded but no audio file was created"
            errors.append(f"{profile['name']}: {details}")
            continue

        latest_file = max(downloaded_files, key=lambda p: p.stat().st_mtime)
        shutil.move(str(latest_file), str(output_path))
        shutil.rmtree(run_dir, ignore_errors=True)
        return output_path

    shutil.rmtree(run_dir, ignore_errors=True)
    joined_errors = " | ".join(errors) if errors else "unknown error"
    if "sabr" in joined_errors.lower() or "403" in joined_errors.lower():
        joined_errors += (
            " | hint: YouTube rejected anonymous requests."
            " Set YTDLP_PO_TOKEN and YTDLP_COOKIE_FILE in .env, or"
            " set YTDLP_COOKIES_FROM_BROWSER=firefox and retry."
        )
    raise Exception(
        "download failed after fallback attempts. "
        f"Details: {joined_errors}"
    )
