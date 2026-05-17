# Sonic Segments MVP

This repo now has two layers:

1. the current app layer
2. the new MVP/service layer

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

Next:

1. add a small CLI or admin page that runs `SonicSegmentsService.create_audit_bundle(...)`
2. point it at a local folder of royalty-cleared tracks
3. generate 3 to 5 variants plus `sonic_audit.md`

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

## Why this organization works

- The current app keeps running the way it already does.
- New MVP work goes into a clean package instead of bloating `app.py`.
- We can later swap internals from heuristic Demucs to Stage 1 or narrator-conditioned models without changing the product-facing API.
