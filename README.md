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

## Accuracy Workflow

There are two separate loops in this project:

1. the product loop: upload video -> swap background music -> render output
2. the development loop: measure output quality -> improve backend -> retest

The current app uses the heuristic backend in `voice_separator.py`.
The evaluation tools in `eval/` are for development only. End users should not need to paste transcripts.

### Step 1: Render a baseline output

Run the app and produce a swapped output as usual:

```bash
streamlit run app.py
```

### Step 2: Benchmark that output

For the Gatorade test clip:

```powershell
powershell -ExecutionPolicy Bypass -File .\eval\run_gatorade_benchmark.ps1 -Candidate "C:\Users\Arjun\Downloads\Gatorade_swapped.mp4"
```

This scores:

- `voice_retention_db`
- `music_leak_corr`
- `jumps_gt12db`
- `hf_ratio_ge6k`

Use that as the baseline for the current heuristic backend.

### Step 3: Build a better backend offline

The target architecture is documented in `VOICEOVER_ACCURACY_PLAN.md`.

The planned sequence is:

1. train a `dialogue/music/effects` separator
2. run the same eval clips through that backend offline
3. benchmark the new outputs against the old baseline
4. only after it wins, wire it into `app.py`
5. later, add narrator-conditioned extraction to suppress sung lyrics even more

### Stage 1 commands

Build manifests from a stem dataset:

```powershell
python -m stage1_dialogue.build_manifest --root D:\datasets\dnr_train --output manifests\dnr_train.jsonl
python -m stage1_dialogue.build_manifest --root D:\datasets\dnr_val --output manifests\dnr_val.jsonl
```

Train the Stage 1 backend:

```powershell
python -m stage1_dialogue.train --train-manifest manifests\dnr_train.jsonl --val-manifest manifests\dnr_val.jsonl --output-dir models\stage1_dialogue\gatorade_v1 --epochs 30 --batch-size 4
```

Run offline inference before touching the app:

```powershell
python -m stage1_dialogue.infer --checkpoint models\stage1_dialogue\gatorade_v1\best.pt --input video\Gatorade.mp4 --output-dir output\stage1_gatorade
```

Then point the app at the checkpoint:

```powershell
$env:STAGE1_VOICEOVER_CHECKPOINT = "C:\Users\Arjun\OneDrive\Coding\Video-Music-Swapper\models\stage1_dialogue\gatorade_v1\best.pt"
$env:VOICEOVER_BACKEND = "stage1_dialogue"
streamlit run app.py
```

### Step 4: Wire the best backend into the app

Once a trained backend beats the current scores:

1. make it selectable in `app.py`
2. keep the heuristic path as fallback
3. render the same clip in both modes
4. benchmark both outputs

### Current state

Today, this repo can:

- render with the current heuristic backend
- render with a Stage 1 checkpoint if one exists
- benchmark rendered outputs

The Stage 1 code scaffold lives in `stage1_dialogue/`.

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
