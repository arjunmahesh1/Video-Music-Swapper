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
        Dict with extracted features: tempo, energy, spectral_centroid, zero_crossing_rate
    """
    # Load audio file
    y, sr = librosa.load(str(audio_path), duration=30)  # Analyze first 30 seconds

    # Extract tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Calculate energy (RMS)
    rms = librosa.feature.rms(y=y)[0]
    energy = float(np.mean(rms))

    # Spectral centroid (brightness/timbre)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = float(np.mean(spectral_centroid))

    # Zero crossing rate (noisiness/percussiveness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    percussiveness = float(np.mean(zcr))

    return {
        'tempo': float(tempo),
        'energy': energy,
        'brightness': brightness,
        'percussiveness': percussiveness
    }


def calculate_similarity(video_features, song_features, listening_score=0.5):
    """Calculate similarity score between video audio and song.

    Lower score = more similar. Incorporates listening frequency to favor songs
    you actually listen to.

    Args:
        video_features: Dict from analyze_audio_features()
        song_features: Dict with keys: tempo, energy, valence, danceability
        listening_score: 0-1 score indicating how often you listen to this song (1.0 = most)

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

    # Base similarity from audio features
    audio_similarity = (
        tempo_diff * 0.3 +        # Tempo is important
        energy_diff * 0.4 +        # Energy is very important
        valence_diff * 0.15 +      # Mood matters
        danceability_diff * 0.15   # Rhythm matters
    )

    # Apply listening score bonus - heavily favor songs you actually listen to
    # Songs with high listening_score get a significant boost (lower similarity score)
    listening_bonus = (1.0 - listening_score) * 0.4  # Up to 0.4 penalty for never-listened songs

    final_score = audio_similarity + listening_bonus

    return final_score


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
