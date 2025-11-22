"""Smart audio features estimation using genre and artist knowledge bases."""
import random


# Comprehensive artist genre database with characteristics
ARTIST_PROFILES = {
    # Hip-Hop/Rap (High energy, high tempo, high danceability)
    'travis scott': {'tempo': 140, 'energy': 0.85, 'valence': 0.65, 'danceability': 0.80},
    'playboi carti': {'tempo': 145, 'energy': 0.90, 'valence': 0.70, 'danceability': 0.85},
    'drake': {'tempo': 125, 'energy': 0.70, 'valence': 0.60, 'danceability': 0.75},
    'kanye west': {'tempo': 130, 'energy': 0.75, 'valence': 0.65, 'danceability': 0.70},
    'kendrick lamar': {'tempo': 135, 'energy': 0.80, 'valence': 0.60, 'danceability': 0.75},
    'future': {'tempo': 140, 'energy': 0.80, 'valence': 0.65, 'danceability': 0.80},
    '21 savage': {'tempo': 135, 'energy': 0.75, 'valence': 0.55, 'danceability': 0.75},
    'lil uzi vert': {'tempo': 145, 'energy': 0.85, 'valence': 0.70, 'danceability': 0.85},
    'juice wrld': {'tempo': 130, 'energy': 0.70, 'valence': 0.50, 'danceability': 0.70},
    'post malone': {'tempo': 120, 'energy': 0.65, 'valence': 0.60, 'danceability': 0.70},
    'asap rocky': {'tempo': 135, 'energy': 0.75, 'valence': 0.65, 'danceability': 0.75},
    'tyler the creator': {'tempo': 130, 'energy': 0.70, 'valence': 0.65, 'danceability': 0.70},
    'j cole': {'tempo': 125, 'energy': 0.70, 'valence': 0.60, 'danceability': 0.70},
    'metro boomin': {'tempo': 140, 'energy': 0.85, 'valence': 0.65, 'danceability': 0.80},
    'migos': {'tempo': 140, 'energy': 0.80, 'valence': 0.70, 'danceability': 0.85},
    'gunna': {'tempo': 135, 'energy': 0.75, 'valence': 0.65, 'danceability': 0.80},
    'lil baby': {'tempo': 140, 'energy': 0.80, 'valence': 0.65, 'danceability': 0.80},
    'polo g': {'tempo': 130, 'energy': 0.75, 'valence': 0.55, 'danceability': 0.70},

    # EDM/Electronic (Very high energy, fast tempo)
    'skrillex': {'tempo': 150, 'energy': 0.95, 'valence': 0.75, 'danceability': 0.90},
    'marshmello': {'tempo': 128, 'energy': 0.85, 'valence': 0.75, 'danceability': 0.85},
    'calvin harris': {'tempo': 128, 'energy': 0.85, 'valence': 0.80, 'danceability': 0.90},
    'martin garrix': {'tempo': 128, 'energy': 0.85, 'valence': 0.80, 'danceability': 0.90},
    'avicii': {'tempo': 128, 'energy': 0.80, 'valence': 0.75, 'danceability': 0.85},
    'deadmau5': {'tempo': 128, 'energy': 0.80, 'valence': 0.70, 'danceability': 0.80},
    'diplo': {'tempo': 130, 'energy': 0.85, 'valence': 0.75, 'danceability': 0.85},
    'kygo': {'tempo': 115, 'energy': 0.70, 'valence': 0.75, 'danceability': 0.80},

    # Rock/Alternative (Medium-high energy, variable tempo)
    'foo fighters': {'tempo': 135, 'energy': 0.85, 'valence': 0.70, 'danceability': 0.60},
    'imagine dragons': {'tempo': 130, 'energy': 0.85, 'valence': 0.70, 'danceability': 0.65},
    'twenty one pilots': {'tempo': 125, 'energy': 0.75, 'valence': 0.60, 'danceability': 0.70},
    'arctic monkeys': {'tempo': 130, 'energy': 0.80, 'valence': 0.65, 'danceability': 0.70},
    'the killers': {'tempo': 130, 'energy': 0.80, 'valence': 0.70, 'danceability': 0.75},
    'radiohead': {'tempo': 110, 'energy': 0.60, 'valence': 0.40, 'danceability': 0.50},
    'muse': {'tempo': 135, 'energy': 0.85, 'valence': 0.65, 'danceability': 0.65},

    # Pop (Medium energy, danceable)
    'taylor swift': {'tempo': 120, 'energy': 0.70, 'valence': 0.70, 'danceability': 0.75},
    'ariana grande': {'tempo': 120, 'energy': 0.75, 'valence': 0.70, 'danceability': 0.80},
    'billie eilish': {'tempo': 100, 'energy': 0.50, 'valence': 0.40, 'danceability': 0.70},
    'the weeknd': {'tempo': 115, 'energy': 0.70, 'valence': 0.50, 'danceability': 0.75},
    'dua lipa': {'tempo': 120, 'energy': 0.75, 'valence': 0.75, 'danceability': 0.85},
    'ed sheeran': {'tempo': 110, 'energy': 0.60, 'valence': 0.70, 'danceability': 0.65},
    'doja cat': {'tempo': 125, 'energy': 0.75, 'valence': 0.75, 'danceability': 0.85},

    # Indie/Acoustic (Lower energy, slower tempo)
    'bon iver': {'tempo': 95, 'energy': 0.40, 'valence': 0.40, 'danceability': 0.40},
    'phoebe bridgers': {'tempo': 95, 'energy': 0.35, 'valence': 0.35, 'danceability': 0.35},
    'mac demarco': {'tempo': 105, 'energy': 0.50, 'valence': 0.60, 'danceability': 0.55},
    'tame impala': {'tempo': 115, 'energy': 0.65, 'valence': 0.60, 'danceability': 0.70},
}


def estimate_features_from_metadata(track_name, artist_name):
    """Estimate audio features using artist/genre knowledge and smart heuristics.

    Args:
        track_name: Track title
        artist_name: Artist name

    Returns:
        Dict with estimated audio features
    """
    artist_lower = artist_name.lower()
    track_lower = track_name.lower()

    # Check if we have a profile for this artist
    base_profile = ARTIST_PROFILES.get(artist_lower)

    if base_profile:
        # Use artist profile as base
        tempo = base_profile['tempo']
        energy = base_profile['energy']
        valence = base_profile['valence']
        danceability = base_profile['danceability']
    else:
        # Default neutral values
        tempo = 120
        energy = 0.5
        valence = 0.5
        danceability = 0.5

        # Genre detection from artist name patterns
        text = f"{track_lower} {artist_lower}"

        # Hip-hop/Rap indicators
        rap_keywords = ['lil', 'yung', 'young', '$', 'feat.', 'ft.']
        if any(kw in artist_lower for kw in rap_keywords):
            tempo = 135
            energy = 0.80
            valence = 0.65
            danceability = 0.80

        # EDM/Electronic indicators
        edm_keywords = ['dj', 'producer']
        if any(kw in artist_lower for kw in edm_keywords):
            tempo = 128
            energy = 0.85
            valence = 0.75
            danceability = 0.85

    # Track name modifiers (fine-tuning based on track characteristics)

    # Remix/VIP versions tend to be higher energy
    if any(word in track_lower for word in ['remix', 'vip', 'vip edit', 'festival']):
        energy = min(1.0, energy + 0.10)
        tempo += 5
        danceability = min(1.0, danceability + 0.10)

    # Acoustic/Stripped versions are lower energy
    if any(word in track_lower for word in ['acoustic', 'stripped', 'unplugged', 'piano']):
        energy = max(0.1, energy - 0.25)
        tempo -= 15
        danceability = max(0.1, danceability - 0.20)

    # Explicit high-energy track indicators
    if any(word in track_lower for word in ['bangers', 'banger', 'hype', 'rage', 'wild']):
        energy = min(1.0, energy + 0.15)
        tempo += 10

    # Sad/emotional indicators
    if any(word in track_lower for word in ['sad', 'lonely', 'heartbreak', 'cry', 'tears', 'hurt']):
        valence = max(0.1, valence - 0.20)
        energy = max(0.2, energy - 0.15)

    # Party/upbeat indicators
    if any(word in track_lower for word in ['party', 'celebrate', 'dance', 'groove', 'bounce']):
        valence = min(1.0, valence + 0.15)
        danceability = min(1.0, danceability + 0.15)
        energy = min(1.0, energy + 0.10)

    # Add slight randomization to avoid identical scores
    tempo += random.uniform(-3, 3)
    energy += random.uniform(-0.03, 0.03)
    valence += random.uniform(-0.03, 0.03)
    danceability += random.uniform(-0.03, 0.03)

    # Clamp values to valid ranges
    tempo = max(60, min(180, tempo))
    energy = max(0, min(1, energy))
    valence = max(0, min(1, valence))
    danceability = max(0, min(1, danceability))

    return {
        'tempo': tempo,
        'energy': energy,
        'valence': valence,
        'danceability': danceability
    }


def batch_get_features(songs, max_songs=None):
    """Get estimated features for multiple songs instantly.

    Args:
        songs: List of song dicts with 'name', 'artist', 'display_name', 'id'
        max_songs: Optional limit on number of songs to process

    Returns:
        Dict mapping song_id -> features
    """
    features_map = {}
    songs_to_process = songs[:max_songs] if max_songs else songs

    print(f"\nEstimating features for {len(songs_to_process)} songs...")

    for song in songs_to_process:
        features = estimate_features_from_metadata(
            song['name'],
            song['artist']
        )
        features_map[song['id']] = features

    print(f"Generated feature estimates for {len(features_map)} songs\n")

    return features_map
