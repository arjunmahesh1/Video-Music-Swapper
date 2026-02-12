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
# Optional download hardening (YouTube/SABR issues):
# YTDLP_COOKIE_FILE=C:\path\to\cookies.txt
# YTDLP_COOKIES_FROM_BROWSER=firefox
# YTDLP_PO_TOKEN=your_ios_gvs_po_token_without_prefix
# SPOTDL_COOKIE_FILE=C:\path\to\cookies.txt
```

## 3. Run the App

### GUI APP (Streamlit):
```bash
streamlit run app.py
```

Then:
1. Click "Connect Spotify" in the sidebar (first time only)
2. Authenticate with your Spotify account

## Troubleshooting Spotify Song Download

If song download fails with SABR/403 errors:

1. Update yt-dlp:
```bash
python -m pip install -U yt-dlp
```
2. Set one of these in `.env`:
   - `YTDLP_COOKIE_FILE=...` (Netscape cookies file), or
   - `YTDLP_COOKIES_FROM_BROWSER=firefox` (or `chrome`, `edge`)
3. If error mentions PO token, also set:
   - `YTDLP_PO_TOKEN=...` (token value; code prepends `ios.gvs+`)
4. Retry, or use **Manual Selection -> Upload Audio File** to bypass online search.
