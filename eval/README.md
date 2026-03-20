# Eval Workflow

This folder is for offline evaluation of swapped outputs.

It is not part of the end-user app flow. The app should remain:

1. upload video
2. choose replacement music
3. render output

We use evaluation offline to tell whether model changes are actually making the result better.

## What you run today

Right now, the app still uses the heuristic backend in `voice_separator.py`.

So the workflow today is:

1. render an output in the app
2. score that output with `voiceover_benchmark.py`
3. use the score as the baseline before adding a trained backend

## Gatorade benchmark

Use:

```powershell
powershell -ExecutionPolicy Bypass -File .\eval\run_gatorade_benchmark.ps1 -Candidate "C:\Users\Arjun\Downloads\Gatorade_swapped.mp4"
```

Optional:

```powershell
powershell -ExecutionPolicy Bypass -File .\eval\run_gatorade_benchmark.ps1 -Candidate "C:\Users\Arjun\Downloads\Gatorade_swapped.mp4" -MusicStem "C:\path\to\original_music.wav"
```

## What the metrics mean

- `voice_retention_db`: how much narration survived inside known speech windows
- `music_leak_corr`: how much old music appears to remain
- `jumps_gt12db`: how choppy or pumpy the audio is
- `hf_ratio_ge6k`: simple proxy for muffling

Use these targets:

- `voice_retention_db > -3`
- `music_leak_corr < 0.03`
- `jumps_gt12db` as low as practical
- `hf_ratio_ge6k` should not collapse toward zero

## How this fits the build plan

### Stage 0: baseline

Current state:

- app renders with heuristic backend
- benchmark scores current quality

### Stage 1: train dialogue separator

Next code step:

- add a trainable `dialogue/music/effects` backend
- train it offline
- export a checkpoint

Then:

1. run the same input through old backend
2. run the same input through new backend
3. benchmark both
4. keep the new backend only if scores improve

### Stage 2: add narrator-conditioned extractor

After Stage 1 is stable:

- add target-speaker extraction on top of dialogue output
- benchmark again on the same clips

### Stage 3: wire best backend into app

Once the trained backend beats the heuristic baseline:

1. make it selectable in `app.py`
2. default to the better backend
3. keep the old backend as fallback

## Important point

The transcript file in this folder is only for evaluation.

Users will not paste transcripts into the app. We use transcripts offline to measure whether the app kept the right speech.
