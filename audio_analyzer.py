"""Audio analysis and similarity matching for video soundtracks."""
import librosa
import numpy as np
from pathlib import Path
import subprocess
import tempfile


def extract_audio_from_video(video_path):
    """Extract audio from video file to temporary WAV file.

    Args:
        video_path: Path to video file

    Returns:
        Path to extracted audio file
    """
    temp_audio = Path(tempfile.gettempdir()) / "temp_video_audio.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM audio codec
        "-ar", "44100",  # Sample rate
        "-ac", "2",  # Stereo
        str(temp_audio)
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise Exception(f"Failed to extract audio: {result.stderr.decode()}")

    return temp_audio


def analyze_audio_features(audio_path):
    """Analyze audio file and extract features for similarity matching.

    Args:
        audio_path: Path to audio file (wav, mp3, etc.)

    Returns:
        Dict with extracted features optimized for genre detection
    """
    # Load audio file - analyze first 30 seconds
    y, sr = librosa.load(str(audio_path), duration=30)

    # Harmonic-percussive separation for better tempo detection
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Extract tempo using percussive component (more accurate)
    tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)

    # Calculate energy (RMS) - normalized
    rms = librosa.feature.rms(y=y)[0]
    energy = float(np.mean(rms))
    # Normalize energy to 0-1 range
    energy = min(1.0, energy * 10)  # Typical RMS values are 0-0.1

    # Spectral features for genre fingerprinting
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = float(np.mean(spectral_centroid))

    # Spectral contrast - distinguishes different genres
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast = float(np.mean(spectral_contrast))

    # Zero crossing rate (percussiveness/noisiness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    percussiveness = float(np.mean(zcr))

    # Chroma features - harmonic/melodic content
    chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
    chroma_mean = float(np.mean(chroma))
    chroma_std = float(np.std(chroma))

    # MFCCs - timbre/texture (first 13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = [float(np.mean(mfcc[i])) for i in range(13)]

    # Derive high-level characteristics from features
    # These help with genre classification
    is_harmonic = chroma_mean > 0.3  # Strong harmonic content
    is_percussive = percussiveness > 0.1  # Strong percussive elements
    is_bright = brightness > 2000  # Bright/sharp timbre

    return {
        'tempo': float(tempo),
        'energy': energy,
        'brightness': brightness,
        'percussiveness': percussiveness,
        'spectral_contrast': contrast,
        'chroma_mean': chroma_mean,
        'chroma_std': chroma_std,
        'mfcc_mean': mfcc_mean,
        'is_harmonic': is_harmonic,
        'is_percussive': is_percussive,
        'is_bright': is_bright
    }


def calculate_genre_similarity(video_genres, song_genres):
    """Calculate genre similarity between video and song.

    Args:
        video_genres: List of genre strings for video (can be empty/inferred)
        song_genres: List of genre strings for song from Spotify

    Returns:
        Float between 0 (no match) and 1 (perfect match)
    """
    if not song_genres:
        return 0.3  # Neutral score if no genre data

    if not video_genres:
        return 0.5  # Can't compare, give neutral score

    # Normalize genre strings (lowercase, remove extra spaces)
    video_genres_norm = [g.lower().strip() for g in video_genres]
    song_genres_norm = [g.lower().strip() for g in song_genres]

    # Check for exact matches
    exact_matches = len(set(video_genres_norm) & set(song_genres_norm))
    if exact_matches > 0:
        return 1.0  # Perfect match

    # Check for partial matches (genre name contains another)
    # e.g., "alt rock" matches "alternative rock"
    partial_score = 0.0
    for vg in video_genres_norm:
        for sg in song_genres_norm:
            # Check if either genre contains the other
            if vg in sg or sg in vg:
                partial_score = max(partial_score, 0.7)
            # Check for common genre family keywords
            genre_families = {
                'rock': ['rock', 'metal', 'punk', 'grunge', 'alternative'],
                'electronic': ['electronic', 'edm', 'techno', 'house', 'dubstep', 'trap'],
                'hip-hop': ['hip hop', 'rap', 'trap', 'drill'],
                'pop': ['pop', 'indie pop', 'synth pop'],
                'indie': ['indie', 'alternative', 'bedroom pop'],
                'r&b': ['r&b', 'soul', 'rnb', 'neo soul'],
                'jazz': ['jazz', 'blues', 'swing'],
                'classical': ['classical', 'orchestra', 'symphonic', 'orchestral', 'cinematic'],
                'acoustic': ['acoustic', 'piano', 'ballad', 'singer-songwriter', 'folk'],
                'country': ['country', 'folk', 'americana'],
                'latin': ['latin', 'reggaeton', 'salsa', 'bachata']
            }

            for _, keywords in genre_families.items():
                vg_in_family = any(kw in vg for kw in keywords)
                sg_in_family = any(kw in sg for kw in keywords)
                if vg_in_family and sg_in_family:
                    partial_score = max(partial_score, 0.5)

    return partial_score


def calculate_similarity(video_features, song_features, listening_score=0.5, genre_similarity=0.5):
    """Calculate similarity score between video audio and song.

    Lower score = more similar. Uses audio features, genre matching, and listening frequency.

    Args:
        video_features: Dict from analyze_audio_features()
        song_features: Dict with keys: tempo, energy, valence, danceability
        listening_score: 0-1 score indicating how often you listen to this song (1.0 = most)
        genre_similarity: 0-1 score from calculate_genre_similarity() (1.0 = perfect genre match)

    Returns:
        Float similarity score (0 = identical, higher = more different)
    """
    # Normalize tempo difference (BPM can range 60-180)
    tempo_diff = abs(video_features['tempo'] - song_features.get('tempo', 120)) / 180.0

    # Energy difference (both should be 0-1 scale)
    energy_diff = abs(video_features['energy'] - song_features.get('energy', 0.5))

    # For Spotify features we don't have from video, we use heuristics
    # Higher energy video → prefer higher valence (happy) and danceability
    valence_preference = video_features['energy']  # High energy → prefer happy songs
    valence_diff = abs(valence_preference - song_features.get('valence', 0.5))

    danceability_preference = video_features['energy']
    danceability_diff = abs(danceability_preference - song_features.get('danceability', 0.5))

    # Audio similarity with updated weights
    # Tempo (30%), Energy (25%), with remaining split between valence/danceability
    audio_similarity = (
        tempo_diff * 0.30 +        # Tempo is very important
        energy_diff * 0.25 +        # Energy is important
        valence_diff * 0.10 +       # Mood matters less
        danceability_diff * 0.10    # Rhythm matters less
    )
    # Total: 75% from audio features

    # Genre matching is CRITICAL (35% weight)
    # Convert genre_similarity (0-1) to a difference score (0-0.35)
    genre_diff = (1.0 - genre_similarity) * 0.35

    # Combine audio and genre
    combined_similarity = audio_similarity + genre_diff

    # Apply listening score bonus - favor songs you actually listen to (10% weight)
    # Songs with high listening_score get a boost (lower similarity score)
    listening_bonus = (1.0 - listening_score) * 0.10

    final_score = combined_similarity + listening_bonus

    return final_score


def infer_video_genres(video_features):
    """Infer likely genres from video audio features.

    Args:
        video_features: Dict from analyze_audio_features()

    Returns:
        List of likely genre strings
    """
    genres = []

    tempo = video_features['tempo']
    energy = video_features['energy']
    percussiveness = video_features.get('percussiveness', 0)
    is_harmonic = video_features.get('is_harmonic', False)
    chroma_mean = video_features.get('chroma_mean', 0)
    brightness = video_features.get('brightness', 0)

    # DEBUG: Print actual features being analyzed
    print(f"\nGenre Detection Debug:")
    print(f"  Tempo: {tempo:.1f} BPM")
    print(f"  Energy: {energy:.2f}")
    print(f"  Percussiveness: {percussiveness:.3f}")
    print(f"  Chroma (harmonic): {chroma_mean:.3f}")
    print(f"  Is harmonic: {is_harmonic}")
    print(f"  Brightness: {brightness:.0f} Hz")

    # Hip-hop/Rap - PRIORITIZE THIS (check first before other genres)
    # Very percussive, WIDER tempo range (rap can be slower!), lower harmonic content
    # EXPANDED: 85-160 BPM to catch slower trap/rap like FEIN
    if percussiveness > 0.1 and 85 <= tempo <= 160 and energy > 0.5:
        # Strong indicator: high percussiveness with lower harmonic content
        if not is_harmonic or chroma_mean < 0.4:
            print(f"  → Matched: Rap/Hip-hop (percussive + low harmonic)")
            genres.append('rap')
            genres.append('hip hop')
            # Skip other genre checks if this is clearly rap
            return genres

    # Trap/Modern Hip-hop - specific subgenre with distinct characteristics
    # EXPANDED: 70-170 to catch half-time trap
    if 70 <= tempo <= 170 and percussiveness > 0.11 and energy > 0.6:
        if not is_harmonic or chroma_mean < 0.35:
            print(f"  → Matched: Trap (very percussive + low harmonic)")
            genres.append('trap')
            genres.append('rap')
            genres.append('hip hop')
            return genres

    # Rock/Metal - high energy, high tempo, bright timbre, harmonic
    if energy > 0.7 and tempo > 120 and video_features.get('is_bright', False) and is_harmonic:
        genres.append('rock')
        if energy > 0.85:
            genres.append('metal')

    # Electronic/EDM - very high energy, steady tempo around 128, percussive
    # Higher harmonic content than rap, more consistent tempo
    if energy > 0.75 and 120 <= tempo <= 140 and percussiveness > 0.08 and is_harmonic:
        genres.append('electronic')
        genres.append('edm')

    # Piano/Acoustic - MUST be VERY harmonic and VERY low percussiveness
    # Stricter to avoid catching rap/trap that happens to be slow
    # Real piano: very harmonic (> 0.4), very low percussiveness (< 0.07)
    if tempo < 110 and is_harmonic and chroma_mean > 0.4 and percussiveness < 0.07:
        print(f"  → Matched: Piano/Acoustic (slow + very harmonic + very low percussive)")
        genres.append('piano')
        genres.append('acoustic')
        if tempo < 90:
            genres.append('ballad')
        # Early return - don't add conflicting genres
        return genres

    # Orchestral/Cinematic - wide range, very harmonic, slow-medium tempo
    if is_harmonic and chroma_mean > 0.4 and tempo < 120 and percussiveness < 0.06:
        genres.append('orchestral')
        genres.append('cinematic')
        genres.append('classical')
        return genres

    # Pop - medium energy and tempo, harmonic
    if 0.5 <= energy <= 0.75 and 100 <= tempo <= 130 and is_harmonic and chroma_mean > 0.35:
        genres.append('pop')

    # Indie/Alternative - medium energy, harmonic content
    # Only match if not already caught by more specific genres
    if 0.4 <= energy <= 0.7 and is_harmonic and chroma_mean > 0.3:
        genres.append('indie')
        genres.append('alternative')

    # Classical/Jazz - lower energy, high harmonic content
    if energy < 0.6 and is_harmonic and chroma_mean > 0.35:
        genres.append('classical')
        if percussiveness > 0.05:
            genres.append('jazz')

    # Default to general categories if nothing matched
    if not genres:
        if energy > 0.6:
            print(f"  → Matched: Energetic (fallback - high energy)")
            genres.append('energetic')
        else:
            print(f"  → Matched: Mellow (fallback - low energy)")
            genres.append('mellow')

    return genres


def select_song_probabilistic(ranked_songs, top_n=5, distribution='balanced'):
    """Select a song from top matches using probabilistic distribution.

    Args:
        ranked_songs: List of songs sorted by similarity (best first)
        top_n: Consider only top N matches
        distribution: 'conservative', 'balanced', or 'adventurous'

    Returns:
        Selected song dict
    """
    if not ranked_songs:
        return None

    # Take only top N
    candidates = ranked_songs[:min(top_n, len(ranked_songs))]

    # Define probability distributions
    distributions = {
        'conservative': [0.70, 0.20, 0.10],  # Heavily favor top match
        'balanced': [0.50, 0.30, 0.20],      # More balanced
        'adventurous': [0.40, 0.30, 0.20, 0.10]  # More variety
    }

    probs = distributions.get(distribution, distributions['balanced'])

    # Pad or trim probabilities to match candidates
    if len(candidates) < len(probs):
        probs = probs[:len(candidates)]
    elif len(candidates) > len(probs):
        # Extend with diminishing probabilities
        remaining = 1.0 - sum(probs)
        extra = [remaining / (len(candidates) - len(probs))] * (len(candidates) - len(probs))
        probs = probs + extra

    # Normalize to ensure sum = 1.0
    probs = np.array(probs)
    probs = probs / probs.sum()

    # Random selection based on probabilities
    selected_idx = np.random.choice(len(candidates), p=probs)

    return candidates[selected_idx]