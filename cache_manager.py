"""Persistent cache management for app state across restarts."""
import json
from pathlib import Path
from datetime import datetime, timedelta


CACHE_DIR = Path(__file__).parent / ".cache"
RECENTLY_USED_FILE = CACHE_DIR / "recently_used_songs.json"
SPOTIFY_TOKEN_FILE = CACHE_DIR / "spotify_token.json"


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(exist_ok=True)


def load_recently_used_songs(max_age_days=7):
    """Load recently used songs from cache.

    Args:
        max_age_days: Clear songs older than this many days

    Returns:
        List of song IDs
    """
    ensure_cache_dir()

    if not RECENTLY_USED_FILE.exists():
        return []

    try:
        with open(RECENTLY_USED_FILE, 'r') as f:
            data = json.load(f)

        # Filter out old entries
        cutoff = datetime.now() - timedelta(days=max_age_days)
        recent = [
            song_id for song_id, timestamp_str in data.items()
            if datetime.fromisoformat(timestamp_str) > cutoff
        ]

        return recent[:20]  # Keep last 20 songs
    except Exception as e:
        print(f"Warning: Could not load recently used songs: {e}")
        return []


def save_recently_used_songs(song_ids):
    """Save recently used songs to cache with timestamps.

    Args:
        song_ids: List of song IDs (most recent first)
    """
    ensure_cache_dir()

    try:
        # Load existing data to preserve timestamps
        existing = {}
        if RECENTLY_USED_FILE.exists():
            with open(RECENTLY_USED_FILE, 'r') as f:
                existing = json.load(f)

        # Add new entries with current timestamp
        now = datetime.now().isoformat()
        data = {}
        for song_id in song_ids[:20]:  # Keep last 20
            # Use existing timestamp if available, otherwise use now
            data[song_id] = existing.get(song_id, now)

        with open(RECENTLY_USED_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save recently used songs: {e}")


def load_spotify_token():
    """Load Spotify token from cache.

    Returns:
        Dict with token_info or None if not found/expired
    """
    ensure_cache_dir()

    if not SPOTIFY_TOKEN_FILE.exists():
        return None

    try:
        with open(SPOTIFY_TOKEN_FILE, 'r') as f:
            token_info = json.load(f)

        # Check if token is expired
        if 'expires_at' in token_info:
            expires_at = datetime.fromtimestamp(token_info['expires_at'])
            if expires_at < datetime.now():
                print("Spotify token expired, need to re-authenticate")
                return None

        return token_info
    except Exception as e:
        print(f"Warning: Could not load Spotify token: {e}")
        return None


def save_spotify_token(token_info):
    """Save Spotify token to cache.

    Args:
        token_info: Spotipy token info dict
    """
    ensure_cache_dir()

    try:
        with open(SPOTIFY_TOKEN_FILE, 'w') as f:
            json.dump(token_info, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save Spotify token: {e}")


def clear_cache():
    """Clear all cache files."""
    try:
        if RECENTLY_USED_FILE.exists():
            RECENTLY_USED_FILE.unlink()
        if SPOTIFY_TOKEN_FILE.exists():
            SPOTIFY_TOKEN_FILE.unlink()
    except Exception as e:
        print(f"Warning: Could not clear cache: {e}")
