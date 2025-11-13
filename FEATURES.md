# Auto-Match Feature Overview

## What You Get

**One-Click Personalized Video Audio Replacement**

Upload any video → System automatically finds and swaps in a similar song from YOUR Spotify library.

## How It Works

### 1. Video Audio Analysis
Uses `librosa` to extract features from the video's original audio:
- **Tempo (BPM)**: Beat speed
- **Energy**: Overall intensity (0-1 scale)
- **Brightness**: Spectral centroid (timbre/tone)
- **Percussiveness**: Zero-crossing rate

### 2. Your Music Library
Fetches and combines:
- Your **liked songs** (last 50)
- Your **top tracks** (last 50 most played)
- Deduplicates to create your personal music pool

### 3. Spotify Audio Features
For each song in your library, fetches Spotify's audio analysis:
- Tempo (BPM)
- Energy (0-1)
- Valence (happiness/positivity)
- Danceability (0-1)
- Acousticness, instrumentalness, speechiness

### 4. Similarity Scoring Algorithm
Calculates a similarity score for each song:

```python
similarity_score = (
    tempo_difference * 0.3 +      # Matching beat speed
    energy_difference * 0.4 +      # Matching intensity
    valence_difference * 0.15 +    # Matching mood
    danceability_difference * 0.15 # Matching rhythm
)
```

Lower score = more similar

### 5. Probabilistic Selection
Instead of always picking #1, randomly selects from top matches:

**Distribution Types:**
- **Conservative**: 70% top match, 20% 2nd, 10% 3rd
- **Balanced**: 50% top, 30% 2nd, 20% 3rd
- **Adventurous**: 40% top, 30% 2nd, 20% 3rd, 10% 4th

This adds variety while staying similar!

### 6. Download & Swap
- Downloads selected song from Spotify (via YouTube using `spotdl`)
- Swaps original audio with new track using FFmpeg
- Returns personalized video

## User Experience

**Auto-Match Mode:**
```
Upload video → Click button → Done!
```

**Behind the scenes:**
1. Extracting audio... ✓
2. Analyzing features (Tempo: 128 BPM, Energy: 0.72)... ✓
3. Fetching Spotify audio features for 87 songs... ✓
4. Calculating similarities... ✓
5. Top matches: [shows top 5]
6. Auto-selected: "Artist - Song Name"
7. Downloading from Spotify... ✓
8. Swapping audio... ✓
9. Done!

## Configuration Options

**In Sidebar:**
- **Mode**: Auto-Match vs Manual
- **Selection Style**: Conservative/Balanced/Adventurous
- **Top N**: How many top songs to consider (3-10)

## Technical Details

**Libraries Used:**
- `librosa`: Audio analysis (tempo, energy, spectral features)
- `numpy/scipy`: Numerical calculations
- `spotipy`: Spotify API integration
- `spotdl`: Download tracks from Spotify
- `ffmpeg`: Audio/video manipulation

**Performance:**
- First run: ~10-15 seconds (fetches Spotify features)
- Subsequent runs: ~5-7 seconds (uses cached features)

**Caching:**
- Audio features are cached in session state
- No re-fetching unless you click "Refresh Library"

## Future Optimizations

1. **Genre Filtering**: Match by genre first, then similarity
2. **Beat Alignment**: Sync beat drops between original and replacement
3. **Tempo Stretching**: Match BPM by time-stretching audio
4. **Voice Separation**: Keep voiceover, replace only background music
5. **User Feedback Loop**: Learn from user preferences over time

## Example Use Cases

**Scenario 1: Hype Sports Ad**
- Video: High energy Nike ad (Tempo: 140 BPM, Energy: 0.85)
- System picks from your library: "Till I Collapse - Eminem" (Tempo: 138, Energy: 0.89)

**Scenario 2: Chill Product Demo**
- Video: Calm Apple product video (Tempo: 85 BPM, Energy: 0.35)
- System picks: "Weightless - Marconi Union" (Tempo: 82, Energy: 0.31)

**Scenario 3: Dance/Party Content**
- Video: Energetic party scene (Tempo: 125 BPM, Danceability: 0.82)
- System picks from your EDM collection based on danceability + energy
