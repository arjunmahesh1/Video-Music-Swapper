"""Spotify integration for fetching user's liked songs and downloading tracks."""
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from pathlib import Path
import subprocess
import tempfile

load_dotenv()
SPOTDL_TIMEOUT_SECONDS = 420

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
            # Exchange code for token
            token_info = auth_manager.get_access_token(code, as_dict=True)
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

def download_spotify_track(spotify_url, output_path=None):
    """Download a Spotify track using spotdl.

    Args:
        spotify_url: Spotify track URL or URI
        output_path: Path to save the downloaded file (default: temp file)

    Returns:
        Path to downloaded file
    """
    if output_path is None:
        # Create temp file
        temp_dir = tempfile.gettempdir()
        output_path = Path(temp_dir) / "spotify_track.mp3"
    else:
        output_path = Path(output_path)

    # Remove existing file if it exists
    if output_path.exists():
        output_path.unlink()

    # Download using spotdl
    cmd = [
        "spotdl",
        "download",
        spotify_url,
        "--output", str(output_path.parent / "{artists} - {title}.{output-ext}"),
        "--format", "mp3"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SPOTDL_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        raise Exception(
            f"spotdl timed out after {SPOTDL_TIMEOUT_SECONDS}s. "
            "This usually means the source is unavailable or network is slow."
        )

    if result.returncode != 0:
        raise Exception(f"spotdl failed: {result.stderr}")

    # Find the downloaded file (spotdl uses artist - title format)
    downloaded_files = list(output_path.parent.glob("*.mp3"))
    if not downloaded_files:
        raise Exception("Download succeeded but file not found")

    # Get the most recently created file
    latest_file = max(downloaded_files, key=lambda p: p.stat().st_mtime)

    return latest_file
