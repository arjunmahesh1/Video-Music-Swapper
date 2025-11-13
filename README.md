_Motivation_: Was watching an ad, and I realized I didn't skip it and watched the whole ad because I liked the song. Thought to myself, in this day of hyper-tuned personalization, why isn't there a better and more certain way of ad engagement?

# Video-Music-Swapper
Swap any video's soundtrack with a new audio track - upload a file, or pick from your Spotify liked songs!

**Features:**
- **🤖 Auto-Match Mode**: Upload video → Algorithm analyzes audio → Auto-selects similar song from YOUR library
- **🎯 Manual Mode**: Pick specific songs from Spotify or upload files
- Smart similarity matching using tempo, energy, valence, and danceability
- Probabilistic selection (balanced, conservative, or adventurous)
- Fetches both liked songs + top tracks for better variety
- Automatic audio download from Spotify (via YouTube)
- GUI (Streamlit) and CLI support

**How Auto-Match Works:**
1. Analyzes original video audio (tempo, energy, brightness, percussiveness)
2. Fetches Spotify audio features for your entire music library
3. Calculates similarity scores for each song
4. Ranks songs by similarity
5. Randomly selects from top matches using configurable probability distribution
6. Downloads and swaps automatically

**Future Enhancements:**
- Genre-based filtering for better matching
- Voice separation (keep voiceover, replace only music)
- Beat alignment and tempo matching
- Advanced audio optimization




# Setup

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

You'll also need FFmpeg installed on your system:
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `winget install ffmpeg`
- **Mac**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

## 2. Spotify Setup (for Spotify integration)

1. Go to https://developer.spotify.com/dashboard
2. Log in and click **"Create app"**
3. Fill in:
   - App name: "Video Music Swapper"
   - Redirect URI: `http://localhost:8501`
4. Copy your **Client ID** and **Client Secret**
5. Create a `.env` file in the project root:
```bash
cp .env.example .env
```
6. Add your credentials to `.env`:
```
SPOTIPY_CLIENT_ID=your_client_id_here
SPOTIPY_CLIENT_SECRET=your_client_secret_here
SPOTIPY_REDIRECT_URI=http://localhost:8501
```

## 3. Run the App

### GUI APP (Streamlit):
```bash
streamlit run app.py
```

Then:
1. Click "Connect Spotify" in the sidebar (first time only)
2. Authenticate with your Spotify account

**For Auto-Match Mode (Recommended):**
3. Ensure "🤖 Auto-Match (Smart)" is selected in sidebar
4. Upload a video (or use sample)
5. Click "🤖 Auto-Match & Swap"
6. Watch it analyze, match, and swap automatically!
7. Download your personalized result

**For Manual Mode:**
3. Switch to "🎯 Manual Select" in sidebar
4. Upload video and pick a specific song (or upload audio)
5. Click "Swap Audio"
6. Download result

**Settings:**
- **Selection Style**: Conservative (favor top match) | Balanced | Adventurous (more variety)
- **Top N Matches**: How many top songs to consider (3-10)



## CLI USAGE:

Save audios in /audio, save video to swap in /video

> pip install ffmpeg

  _Single pair_:

  > python swap_audio.py --video video/myclip.mp4 --audio "audio/my song.mp3" --output out.mp4



  _Batch‑process every video×audio permutation in the default folders_:

  > python swap_audio.py


  Can point to any folder/filename

  > python swap_audio.py --video C:/stuff/v1.mp4 --audio C:/stuff/sound.wav

-> out.mp4





#### Advanced

The CLI has --videos-dir and --audio-dir flags to point batch mode anywhere.

To fade the new track out, edit swap_audio.py and add an audio‑filter: -af "afade=t=out:st=<sec>:d=2".

Re‑encode video for wider compatibility by swapping -c:v copy for e.g. -c:v libx264 -crf 20.