# Video-Music-Swapper
Swap any video's soundtrack with a new audio track - upload a file, or pick from your Spotify liked songs!

# Setup

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```
FFmpeg install on your system:
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `winget install ffmpeg`
- **Mac**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

## 2. Spotify Setup (for Spotify integration)

.env: ```
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
