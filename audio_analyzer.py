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

    # Bass energy - for detecting trap/rap with heavy 808s
    # Extract low-frequency content (20-250 Hz for bass/sub-bass)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_bins = np.where((freqs >= 20) & (freqs <= 250))[0]
    bass_energy = float(np.mean(S[bass_bins, :]))

    # Sub-bass ratio - trap has VERY heavy 20-60Hz vs 60-250Hz
    sub_bass_bins = np.where((freqs >= 20) & (freqs <= 60))[0]
    mid_bass_bins = np.where((freqs >= 60) & (freqs <= 250))[0]
    sub_bass_energy = float(np.mean(S[sub_bass_bins, :]))
    mid_bass_energy = float(np.mean(S[mid_bass_bins, :]))
    sub_bass_ratio = sub_bass_energy / (mid_bass_energy + 1e-8)  # Avoid division by zero

    # Onset strength - measures rhythmic hits (good for rap/trap detection)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_strength = float(np.mean(onset_env))

    # Onset rate (onsets per second) - high for complex rhythms
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
    onset_rate = len(onset_frames) / (len(y) / sr)

    # Beat strength stability - stable for EDM/house, variable for acoustic
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    beat_strength_variance = float(np.std(np.mean(tempogram, axis=0)))

    # Spectral rolloff (85%) - bright genres have high rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    spectral_rolloff = float(np.mean(rolloff))

    # Spectral bandwidth - wide for orchestral, narrow for synths
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_bandwidth = float(np.mean(bandwidth))

    # Chroma features - harmonic/melodic content (MUST come before chroma_entropy)
    chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
    chroma_mean = float(np.mean(chroma))
    chroma_std = float(np.std(chroma))

    # Harmonic-percussive ratio - very high for piano/classical, low for trap
    harmonic_energy = float(np.mean(np.abs(y_harmonic)))
    percussive_energy = float(np.mean(np.abs(y_percussive)))
    hpss_ratio = harmonic_energy / (percussive_energy + 1e-8)

    # Chroma entropy - complex harmony (jazz) vs static harmony (trap)
    chroma_entropy = float(-np.sum(chroma * np.log(chroma + 1e-8), axis=0).mean())

    # Dynamic range (crest factor) - high for rock/acoustic, low for EDM/pop
    crest_factor = float(np.max(np.abs(y)) / (np.mean(np.abs(y)) + 1e-8))

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
        'bass_energy': bass_energy,
        'sub_bass_ratio': sub_bass_ratio,
        'onset_strength': onset_strength,
        'onset_rate': onset_rate,
        'beat_strength_variance': beat_strength_variance,
        'spectral_rolloff': spectral_rolloff,
        'spectral_bandwidth': spectral_bandwidth,
        'hpss_ratio': hpss_ratio,
        'chroma_entropy': chroma_entropy,
        'crest_factor': crest_factor,
        'spectral_contrast': contrast,
        'chroma_mean': chroma_mean,
        'chroma_std': chroma_std,
        'mfcc_mean': mfcc_mean,
        'is_harmonic': is_harmonic,
        'is_percussive': is_percussive,
        'is_bright': is_bright
    }


def calculate_genre_similarity(video_genres, song_genres, debug=False):
    """Calculate genre similarity between video and song.

    Args:
        video_genres: List of genre strings for video (can be empty/inferred)
        song_genres: List of genre strings for song from Spotify
        debug: If True, print debug info about matching

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

    if debug:
        print(f"  Comparing video genres {video_genres_norm} with song genres {song_genres_norm}")

    # Check for exact matches
    exact_matches = len(set(video_genres_norm) & set(song_genres_norm))
    if exact_matches > 0:
        if debug:
            print(f"    → EXACT match! Score: 1.0")
        return 1.0  # Perfect match

    # Check for partial matches (genre name contains another)
    # e.g., "alt rock" matches "alternative rock"
    # BUT "piano" should NOT match "afropiano" - require word boundaries
    partial_score = 0.0
    for vg in video_genres_norm:
        for sg in song_genres_norm:
            # Split genres by spaces and hyphens to check for word-level matches
            vg_words = set(vg.replace('-', ' ').split())
            sg_words = set(sg.replace('-', ' ').split())

            # Check if any words overlap OR if one genre is contained in the other with word boundaries
            word_overlap = len(vg_words & sg_words) > 0

            # Check if one is a complete substring of the other (not just partial word)
            # "rock" should match "indie rock" but "piano" should NOT match "afropiano"
            is_complete_substring = (
                (vg in sg and (vg + ' ' in sg or ' ' + vg in sg or vg == sg)) or
                (sg in vg and (sg + ' ' in vg or ' ' + sg in vg or sg == vg))
            )

            if word_overlap or is_complete_substring:
                if debug:
                    print(f"    → Partial match: '{vg}' and '{sg}' (score=0.7)")
                partial_score = max(partial_score, 0.7)
            # Check for common genre family keywords
            # NOTE: Each genre family includes its key as a keyword for self-matching
            genre_families = {
                'rock': ['rock', 'metal', 'punk', 'grunge', 'alternative', 'indie rock', 'hard rock',
                         'punk rock', 'hardcore punk', 'blues rock', 'country rock'],
                'metal': ['metal', 'heavy metal', 'thrash metal', 'speed metal', 'doom metal',
                          'stoner metal', 'black metal', 'death metal', 'metalcore'],
                'electronic': ['electronic', 'edm', 'techno', 'house', 'dubstep', 'trap', 'ambient',
                              'trance', 'drum and bass', 'jungle', 'deep house', 'tech house',
                              'bass music', 'downtempo', 'idm'],
                'hip-hop': ['hip hop', 'hip-hop', 'rap', 'trap', 'drill', 'trip-hop', 'boom bap',
                           'conscious rap', 'gangsta rap'],
                'pop': ['pop', 'indie pop', 'synth pop', 'upbeat', 'dance pop', 'electropop'],
                'indie': ['indie', 'alternative', 'bedroom pop', 'lo-fi', 'indie rock', 'indie folk',
                         'indie pop', 'shoegaze'],
                'r&b': ['r&b', 'soul', 'rnb', 'neo soul', 'neo-soul', 'funk', 'funk soul'],
                'jazz': ['jazz', 'blues', 'swing', 'downtempo', 'smooth jazz', 'bebop', 'jazz fusion',
                        'acid jazz', 'nu jazz'],
                'blues': ['blues', 'blues rock', 'acoustic blues', 'delta blues', 'chicago blues'],
                'classical': ['classical', 'orchestra', 'symphonic', 'orchestral', 'cinematic',
                            'contemporary classical', 'baroque', 'romantic'],
                'acoustic': ['acoustic', 'piano', 'ballad', 'singer-songwriter', 'folk', 'folk rock',
                            'indie folk', 'acoustic blues'],
                'country': ['country', 'folk', 'americana', 'country rock', 'bluegrass', 'alt-country'],
                'latin': ['latin', 'reggaeton', 'salsa', 'bachata', 'cumbia', 'latin pop', 'latin trap'],
                'reggae': ['reggae', 'dub', 'roots reggae', 'dancehall', 'ska', 'reggae fusion'],
                'chill': ['chill', 'ambient', 'downtempo', 'lo-fi', 'trip-hop', 'ballad', 'chillhop',
                         'chillwave', 'dream pop'],
                'afrobeats': ['afrobeats', 'afrobeat', 'afropop', 'afro-pop', 'afro', 'afro fusion']
            }

            for _, keywords in genre_families.items():
                vg_in_family = any(kw in vg for kw in keywords)
                sg_in_family = any(kw in sg for kw in keywords)
                if vg_in_family and sg_in_family:
                    partial_score = max(partial_score, 0.5)
                    if debug:
                        print(f"    Genre family match (score=0.5): {vg} and {sg} both in family")

    if debug and partial_score == 0.0:
        print(f"    No genre match: video={video_genres_norm} vs song={song_genres_norm}")

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
    # Genre is now DOMINANT (55%), tempo secondary (20%), energy tertiary (15%)
    # This prevents piano songs from matching afrobeats just because tempo/energy align
    audio_similarity = (
        tempo_diff * 0.20 +        # Tempo is important but secondary to genre
        energy_diff * 0.15 +        # Energy matters but less than tempo
        valence_diff * 0.05 +       # Mood matters minimally
        danceability_diff * 0.05    # Rhythm matters minimally
    )
    # Total: 45% from audio features

    # Genre matching is DOMINANT (55% weight) - increased from 35%
    # When genres don't match (piano vs afrobeats), this ensures low similarity even if tempo/energy match
    # Convert genre_similarity (0-1) to a difference score (0-0.55)
    genre_diff = (1.0 - genre_similarity) * 0.55

    # Combine audio and genre
    combined_similarity = audio_similarity + genre_diff

    # Apply listening score bonus - favor songs you actually listen to (5% weight, reduced from 10%)
    # Songs with high listening_score get a boost (lower similarity score)
    listening_bonus = (1.0 - listening_score) * 0.05

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
    bass_energy = video_features.get('bass_energy', 0)
    sub_bass_ratio = video_features.get('sub_bass_ratio', 0)
    onset_strength = video_features.get('onset_strength', 0)
    hpss_ratio = video_features.get('hpss_ratio', 1.0)
    chroma_entropy = video_features.get('chroma_entropy', 0)
    beat_strength_variance = video_features.get('beat_strength_variance', 0)
    spectral_rolloff = video_features.get('spectral_rolloff', 0)
    spectral_bandwidth = video_features.get('spectral_bandwidth', 0)
    crest_factor = video_features.get('crest_factor', 0)
    onset_rate = video_features.get('onset_rate', 0)

    # DEBUG: Print actual features being analyzed
    print(f"\nGenre Detection Debug:")
    print(f"  Tempo: {tempo:.1f} BPM")
    print(f"  Energy: {energy:.2f}")
    print(f"  Percussiveness: {percussiveness:.3f}")
    print(f"  Bass energy: {bass_energy:.1f}")
    print(f"  Sub-bass ratio: {sub_bass_ratio:.2f}")
    print(f"  Onset strength: {onset_strength:.2f}")
    print(f"  HPSS ratio (harmonic/percussive): {hpss_ratio:.2f}")
    print(f"  Chroma (harmonic): {chroma_mean:.3f}")
    print(f"  Chroma entropy: {chroma_entropy:.3f}")
    print(f"  Is harmonic: {is_harmonic}")
    print(f"  Brightness: {brightness:.0f} Hz")

    # Hip-hop/Rap/Trap - CHECK FIRST! (moved before piano)
    # FEIN example: 99 BPM, bass=28.6, sub_bass_ratio=0.22, hpss=2.04
    # Even with high HPSS, very heavy bass indicates rap, not piano
    if 70 <= tempo <= 165 and energy > 0.5:
        # Trap-specific: very heavy sub-bass presence
        has_trap_bass = sub_bass_ratio > 0.6  # Trap 808s dominate sub-bass
        has_very_heavy_bass = bass_energy > 25  # VERY heavy bass (rap/trap signature)
        has_heavy_bass = bass_energy > 15  # General bass presence
        has_strong_hits = onset_strength > 0.3 and percussiveness > 0.05  # Percussive beats
        has_low_harmony = not is_harmonic or chroma_mean < 0.35
        has_moderate_hpss = hpss_ratio < 3.0  # Not overly harmonic (allows some harmony)

        # Priority 1: Very heavy bass (>25) = rap, regardless of HPSS
        # FEIN has bass=28.6, so this catches it even with high HPSS
        if has_very_heavy_bass and has_moderate_hpss:
            print(f"  → Matched: Rap/Trap (very heavy bass={bass_energy:.1f}, hpss={hpss_ratio:.2f})")
            genres.append('rap')
            genres.append('hip hop')
            if tempo < 110 or has_trap_bass:
                genres.append('trap')
            return genres

        # Priority 2: Trap bass signature or strong percussive + low harmony + low HPSS
        has_low_hpss = hpss_ratio < 2.0  # More percussive than harmonic (not piano)
        if (has_trap_bass or (has_heavy_bass and has_strong_hits)) and has_low_harmony and has_low_hpss:
            print(f"  → Matched: Rap/Trap (bass={bass_energy:.1f}, sub-bass-ratio={sub_bass_ratio:.2f}, hpss={hpss_ratio:.2f})")
            genres.append('rap')
            genres.append('hip hop')
            if tempo < 110 or has_trap_bass:
                genres.append('trap')
            return genres

    # Piano/Acoustic/Chill - Check after rap
    # Slow tempo, low percussiveness, NO heavy bass/sub-bass, HIGH harmonic ratio
    # Piano has VERY high hpss_ratio (>2.0) and low sub_bass_ratio
    if tempo < 110 and percussiveness < 0.10:
        # Piano-specific: high harmonic content, low sub-bass
        has_piano_harmony = hpss_ratio > 2.0  # Much more harmonic than percussive
        has_no_trap_bass = sub_bass_ratio < 0.4  # No trap 808s
        has_some_harmony = chroma_mean > 0.20 or is_harmonic

        if has_piano_harmony and has_no_trap_bass and has_some_harmony:
            print(f"  → Matched: Piano/Acoustic/Chill (slow + harmonic + no sub-bass, hpss={hpss_ratio:.2f})")
            genres.append('piano')
            genres.append('acoustic')
            genres.append('chill')
            if tempo < 90:
                genres.append('ballad')
            return genres

    # Hard Rock/Metal - Check BEFORE indie/alternative to prioritize harder genres
    # High energy, fast tempo, heavy bass, bright/aggressive sound
    # Kawasaki ad example: 152 BPM, energy 1.00, bass 31.9, brightness 2271
    if energy > 0.85 and tempo > 130 and bass_energy > 25:
        is_very_bright = spectral_rolloff > 8000  # Extremely distorted guitars (metal)
        is_bright = spectral_rolloff > 5000  # Bright guitars (hard rock)
        has_high_dynamics = crest_factor > 15  # Extreme dynamics
        has_moderate_dynamics = crest_factor > 10  # Good dynamics

        if is_very_bright:
            # Metal subgenres - extremely bright/aggressive
            if onset_rate > 8 and tempo > 160:  # Very fast + many hits
                print(f"  → Matched: Thrash/Speed Metal (very fast + aggressive)")
                genres.append('thrash metal')
                genres.append('speed metal')
            elif tempo < 90 and bass_energy > 20:  # Slow + heavy
                print(f"  → Matched: Doom/Stoner Metal (slow + heavy)")
                genres.append('doom metal')
                genres.append('stoner metal')
            else:
                print(f"  → Matched: Metal (high energy + very bright + heavy)")
                genres.append('metal')
                genres.append('heavy metal')
            genres.append('rock')
            return genres
        elif is_bright or has_moderate_dynamics:
            # Hard rock - bright and energetic but not metal-level aggressive
            print(f"  → Matched: Hard Rock (high energy + fast + heavy bass)")
            genres.append('hard rock')
            genres.append('rock')
            if tempo > 140:
                genres.append('punk rock')
            return genres

    # Generic Rock - high energy, high tempo, harmonic but less intense than hard rock
    # Use spectral_rolloff and crest_factor to distinguish subgenres
    if energy > 0.7 and tempo > 120 and is_harmonic:
        is_very_bright = spectral_rolloff > 8000  # Distorted guitars
        has_high_dynamics = crest_factor > 15  # Extreme dynamics

        if energy > 0.85 and is_very_bright:
            # Metal subgenres
            if onset_rate > 8 and tempo > 160:  # Very fast + many hits
                genres.append('thrash metal')
                genres.append('speed metal')
            elif tempo < 90 and bass_energy > 20:  # Slow + heavy
                genres.append('doom metal')
                genres.append('stoner metal')
            else:
                genres.append('metal')
                genres.append('heavy metal')
            genres.append('rock')
        elif has_high_dynamics and video_features.get('is_bright', False):
            # Rock subgenres
            if tempo > 140 and percussiveness > 0.12:
                genres.append('punk rock')
                genres.append('hardcore punk')
            else:
                genres.append('rock')
                genres.append('hard rock')
        else:
            genres.append('rock')

    # Orchestral/Cinematic - wide range, very harmonic, slow-medium tempo
    # Check slow orchestral pieces (distinct from piano by higher chroma)
    if tempo < 120 and is_harmonic and chroma_mean > 0.45 and percussiveness < 0.06:
        print(f"  → Matched: Orchestral/Cinematic (slow-medium + very harmonic + very low percussive)")
        genres.append('orchestral')
        genres.append('cinematic')
        genres.append('classical')
        return genres

    # Electronic/EDM - very high energy, steady tempo, percussive
    # Use beat_strength_variance to distinguish subgenres
    if energy > 0.7 and percussiveness > 0.08 and is_harmonic:
        is_very_stable = beat_strength_variance < 0.5  # Machine-like stability

        # House (118-130 BPM, very stable 4-on-the-floor)
        if 118 <= tempo <= 130 and is_very_stable:
            if bass_energy > 18:
                genres.append('deep house')
                genres.append('tech house')
            else:
                genres.append('house')
            genres.append('edm')
            return genres

        # Techno (125-145 BPM, extremely stable)
        elif 125 <= tempo <= 145 and is_very_stable and chroma_entropy < 1.5:
            genres.append('techno')
            genres.append('edm')
            return genres

        # Trance (130-145 BPM, harmonic pads)
        elif 130 <= tempo <= 145 and hpss_ratio > 1.5 and chroma_mean > 0.35:
            genres.append('trance')
            genres.append('edm')
            return genres

        # Drum & Bass (160-180 BPM, complex breaks)
        elif 160 <= tempo <= 180 and onset_rate > 6:
            genres.append('drum and bass')
            genres.append('jungle')
            genres.append('edm')
            return genres

        # Dubstep (140 BPM, half-time feel, heavy bass modulation)
        elif 135 <= tempo <= 145 and sub_bass_ratio > 0.5 and bass_energy > 20:
            genres.append('dubstep')
            genres.append('bass music')
            genres.append('edm')
            return genres

        # Generic EDM/Electronic
        else:
            genres.append('electronic')
            genres.append('edm')

    # Pop - medium energy and tempo, harmonic
    if 0.5 <= energy <= 0.75 and 100 <= tempo <= 130 and is_harmonic and chroma_mean > 0.35:
        genres.append('pop')

    # Indie/Alternative/Rock - broader matching for guitar-based rock
    # Medium-high energy, harmonic content, bright timbre
    # Sweet Disposition example: 129 BPM, energy 1.0, harmonic
    if energy > 0.6 and is_harmonic and tempo > 100:
        # Distinguish from electronic (which also has high energy + tempo)
        # Indie rock has less percussiveness than electronic, more guitar-like brightness
        if percussiveness < 0.10:  # Less percussive than EDM
            print(f"  → Matched: Indie/Alternative Rock (energetic + harmonic + guitar-based)")
            genres.append('indie')
            genres.append('alternative')
            genres.append('rock')
            if energy > 0.8:
                genres.append('indie rock')
            return genres

    # Funk/Soul/R&B - groove-based, moderate tempo, strong bass
    if 90 <= tempo <= 120 and bass_energy > 12 and is_harmonic:
        has_groove = onset_strength > 0.25 and percussiveness > 0.06
        if has_groove and chroma_entropy > 1.5:  # Complex harmony
            if tempo > 105:
                genres.append('funk')
                genres.append('soul')
            else:
                genres.append('neo soul')
                genres.append('r&b')
            return genres

    # Blues - slow-medium tempo, harmonic, moderate energy
    if 60 <= tempo <= 100 and is_harmonic and 0.4 <= energy <= 0.7:
        if chroma_mean > 0.30 and hpss_ratio > 1.8:
            if bass_energy > 10:
                genres.append('blues rock')
                genres.append('blues')
            else:
                genres.append('blues')
                genres.append('acoustic blues')
            return genres

    # Reggae/Dub - 60-90 BPM, off-beat emphasis, bass-heavy
    if 60 <= tempo <= 90 and bass_energy > 15:
        if beat_strength_variance > 0.6:  # Off-beat rhythms
            if sub_bass_ratio > 0.5:  # Very bass-heavy
                genres.append('dub')
                genres.append('reggae')
            else:
                genres.append('reggae')
                genres.append('roots reggae')
            return genres

    # Country/Folk - acoustic, moderate tempo, harmonic
    if 80 <= tempo <= 140 and hpss_ratio > 2.5 and 0.4 <= energy <= 0.7:
        if spectral_rolloff < 6000:  # Less bright than rock
            if tempo > 110:
                genres.append('country rock')
                genres.append('country')
            else:
                genres.append('folk')
                genres.append('acoustic')
                genres.append('singer-songwriter')
            return genres

    # Latin - 90-110 BPM, rhythmic, percussive
    if 90 <= tempo <= 110 and percussiveness > 0.09 and onset_rate > 4:
        if is_harmonic:
            genres.append('latin')
            genres.append('salsa')
            return genres
        else:
            genres.append('reggaeton')
            genres.append('latin')
            return genres

    # Lo-fi/Chillhop - low fidelity, chill tempo, noisy/warm
    if 70 <= tempo <= 95 and 0.3 <= energy <= 0.6:
        # Note: spectral_flatness is in the features but checking if it's available
        if hpss_ratio > 1.5:  # Harmonic with some noise
            genres.append('lo-fi')
            genres.append('chillhop')
            genres.append('chill')
            return genres

    # Classical/Jazz - lower energy, high harmonic content
    if energy < 0.6 and is_harmonic and chroma_mean > 0.35:
        if chroma_entropy > 2.0:  # Very complex harmony = Jazz
            if tempo < 100:
                genres.append('smooth jazz')
                genres.append('jazz')
            else:
                genres.append('bebop')
                genres.append('jazz')
        elif spectral_bandwidth > 1500:  # Wide spectrum = Orchestral
            genres.append('orchestral')
            genres.append('classical')
        else:
            genres.append('classical')
            if percussiveness > 0.05:
                genres.append('jazz')

    # Fallback categories - use BOTH tempo and energy for better matching
    # Don't just lump everything into "energetic"
    if not genres:
        print(f"  → No specific genre matched, using fallback based on tempo + energy")

        # Slow tempo categories (< 100 BPM)
        if tempo < 100:
            if is_harmonic:
                # Slow + harmonic = ambient/downtempo/ballad/chill
                genres.append('chill')
                genres.append('ambient')
                genres.append('downtempo')
                if tempo < 80:
                    genres.append('ballad')
                print(f"  → Matched: Chill/Ambient/Downtempo (slow + harmonic)")
            else:
                # Slow + not harmonic = trip-hop/lo-fi/chill
                genres.append('chill')
                genres.append('trip-hop')
                genres.append('lo-fi')
                print(f"  → Matched: Chill/Trip-hop/Lo-fi (slow + less harmonic)")

        # Medium tempo categories (100-120 BPM)
        elif 100 <= tempo <= 120:
            if is_harmonic:
                genres.append('soul')
                genres.append('r&b')
                print(f"  → Matched: Soul/R&B (medium tempo + harmonic)")
            else:
                genres.append('indie')
                genres.append('alternative')
                print(f"  → Matched: Indie/Alternative (medium tempo)")

        # Fast tempo categories (> 120 BPM)
        else:
            if energy > 0.7:
                # Fast + high energy
                genres.append('upbeat')
                genres.append('energetic')
                print(f"  → Matched: Upbeat/Energetic (fast + high energy)")
            else:
                # Fast but lower energy
                genres.append('indie')
                genres.append('pop')
                print(f"  → Matched: Indie/Pop (fast + medium energy)")

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