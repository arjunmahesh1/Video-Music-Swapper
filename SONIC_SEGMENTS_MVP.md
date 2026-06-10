# Sonic Segments MVP

> **The product site is live.** Run it with:
>
> ```bash
> python -m uvicorn sonic_segments.web.main:app --port 8000
> ```
>
> then open http://localhost:8000. See "The Platform" below.

## The Platform

The repo now hosts the full Sonic Segments product on top of the original engine:

```
sonic_segments/
  intelligence/   # local SOTA reasoning (no online APIs)
    clap_model.py   - LAION CLAP zero-shot audio-text engine (lazy singleton)
    mood.py         - 10-mood taxonomy; CLAP + DSP fusion; major/minor valence prior
    demographics.py - 16-segment demographic -> music-direction knowledge base
    matching.py     - track-vs-brief scoring (style/mood/tempo/energy)
    quality.py      - ffprobe compression red-flags for pulled sources
  data/demographics.json  # research-grounded KB (sources cited in _meta)
  sources/        # unified MusicSource interface
    local_catalog.py  - your licensed folders (audio/, library/)   [CLEARED]
    jamendo.py        - Creative Commons API catalog               [CLEARED, needs JAMENDO_CLIENT_ID]
    uploaded.py       - customer-provided track                    [CLEARED]
    musicgen.py       - local Meta MusicGen generation             [CLEARED]
    spotify_personal.py - songs YOU listen to, mood-matched        [DEMO ONLY]
    reference.py      - uncleared reference pulls                  [DEMO ONLY, watermarked]
  pipeline/
    ingest.py   - upload or yt-dlp URL pull + quality flagging
    jobs.py     - persistent job store, serial heavy-work executor
    campaign.py - analyze once -> direction per (demographic, mood) -> source/rank/render
  web/          - FastAPI site: intake one-pager, status page, A/B preview
```

### The two flows

1. **Personal demo** — connect Spotify, pick a mood; the ad is re-scored with a
   track from *your* library whose audio actually matches that mood
   (metadata prefilter -> download pool -> CLAP verification). Demo-only.
2. **Scored variants** — pick up to 5 target demographics plus mood(s); one
   re-scored cut per audience, music direction resolved by the demographics KB
   (e.g. India Youth + happy -> festive indi-pop with dhol percussion).

Every render preserves the original voiceover via the existing Demucs
separation + ducking engine. Results land on a shareable `/preview/{id}` page
with an instant A/B audio flip.

### Environment keys (.env)

- `SPOTIPY_CLIENT_ID` / `SPOTIPY_CLIENT_SECRET` / `SPOTIPY_REDIRECT_URI` — Spotify (existing)
- `JAMENDO_CLIENT_ID` — free key from devportal.jamendo.com; enables the legal catalog source
- `CLAP_MODEL_ID` — optional override (default `laion/larger_clap_music_and_speech`)

### Benchmarks

`python -m eval.mood_benchmark` scores mood detection against 13 hand-labeled
sample ads; `--stem` runs the production path (Demucs music stem). Current
full-mix numbers: 54% top-1, 92% top-3 (CLAP+DSP) vs 23%/85% DSP-only.

---

## Original layering notes

This repo has two layers:

1. the current app layer
2. the MVP/service layer

The goal is to organize the work without breaking the current Streamlit product.

## Current App Layer

These files remain the working engine:

- `app.py`
- `audio_analyzer.py`
- `voice_separator.py`
- `voiceover_backend.py`
- `stage1_dialogue_backend.py`
- `swap_audio.py`

This layer is still responsible for:

- media upload
- Spotify selection
- feature extraction
- voiceover preservation
- final rendering

## New MVP Layer

The new package is `sonic_segments/`.

It is additive and wraps the existing modules instead of replacing them.

### Package structure

- `sonic_segments/models.py`
  - shared product-facing dataclasses
- `sonic_segments/adapters.py`
  - thin wrappers around the current repo functions
- `sonic_segments/service.py`
  - analysis, rendering, and audit orchestration
- `sonic_segments/audit.py`
  - command-line entrypoint for manual pilot audits
- `sonic_segments/reports.py`
  - markdown audit generation
- `sonic_segments/catalogs/manual.py`
  - local folder track catalog for the first MVP

## Safe Partitioning

To keep the repo stable, use this boundary:

### Legacy modules own:

- low-level DSP
- Demucs / Stage 1 backend behavior
- speech detection
- final audio mux logic

### `sonic_segments` owns:

- brand brief inputs
- ad-level analysis results
- report generation
- local-catalog variant generation
- future external catalog integrations
- future API/server entrypoints

That means we can evolve the MVP UX without constantly editing `app.py`.

## Recommended Build Order

### Phase 1: foundation

Done in this pass:

- added a product-oriented service layer
- added report generation
- added a local track catalog

### Phase 2: manual pilot workflow

Done in this pass:

- added a CLI that runs `SonicSegmentsService.create_audit_bundle(...)`
- supports analysis-only audits
- supports local-folder variant renders
- keeps voice preservation opt-in because it is slower

### Phase 3: productization

Next after that:

1. add a brand brief form
2. add a catalog adapter for a licensed music source
3. add persistent output folders per campaign
4. add benchmark logging so we can compare variant performance later

## Example usage

```python
from sonic_segments import BrandProfile, SonicSegmentsService
from sonic_segments.catalogs import LocalTrackCatalog

service = SonicSegmentsService()
brand = BrandProfile(
    name="Starbucks",
    product_category="coffee",
    target_demo="broad consumer",
    vibe="calm, premium, cozy",
    desired_energy_min=0.15,
    desired_energy_max=0.40,
    desired_tempo_min=60,
    desired_tempo_max=95,
    desired_genres=["piano", "acoustic", "ambient"],
)
tracks = LocalTrackCatalog("audio").list_tracks(limit=3)

analysis, variants, report_path = service.create_audit_bundle(
    video_path="video/Starbucks.mp4",
    output_dir="output/starbucks_audit",
    brand_profile=brand,
    tracks=tracks,
    preserve_voiceover=True,
)
```

## CLI usage

Install the project dependencies before running audio jobs:

```bash
python -m pip install -r requirements.txt
```

Fast analysis-only audit:

```bash
python -m sonic_segments.audit \
  --video "video/Starbucks.mp4" \
  --output-dir "output/starbucks_audit" \
  --brand-name "Starbucks" \
  --category "coffee" \
  --vibe "calm, premium, cozy" \
  --energy-min 0.15 \
  --energy-max 0.40 \
  --tempo-min 60 \
  --tempo-max 95 \
  --genres "piano, acoustic, ambient"
```

Render local-track variants without preserving original voiceover:

```bash
python -m sonic_segments.audit \
  --video "video/Starbucks.mp4" \
  --output-dir "output/starbucks_audit" \
  --tracks-dir "audio" \
  --variant-limit 3 \
  --brand-name "Starbucks" \
  --vibe "calm, premium, cozy"
```

Render variants with voiceover preservation:

```bash
python -m sonic_segments.audit \
  --video "video/Gatorade.mp4" \
  --output-dir "output/gatorade_audit" \
  --tracks-dir "audio" \
  --variant-limit 1 \
  --preserve-voiceover \
  --brand-name "Gatorade" \
  --vibe "high-energy sports"
```

## Why this organization works

- The current app keeps running the way it already does.
- New MVP work goes into a clean package instead of bloating `app.py`.
- We can later swap internals from heuristic Demucs to Stage 1 or narrator-conditioned models without changing the product-facing API.
