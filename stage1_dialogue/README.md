# Stage 1 Dialogue Backend

This is the first trainable backend for voiceover preservation.

Its job is to separate:

- `dialogue`
- `music`
- `effects`

It is not the final narrator-conditioned system. It is the first trainable replacement for the heuristic Demucs path.

## Expected directory layout

Each training scene should contain:

- `dialogue.wav`
- `music.wav`
- `effects.wav`
- optional `mixture.wav`

If `mixture.wav` is missing, the training loader will sum the three stems.

## 1. Build manifests

Example:

```powershell
python -m stage1_dialogue.build_manifest --root D:\datasets\dnr_train --output manifests\dnr_train.jsonl
python -m stage1_dialogue.build_manifest --root D:\datasets\dnr_val --output manifests\dnr_val.jsonl
```

## 2. Train

Example:

```powershell
python -m stage1_dialogue.train --train-manifest manifests\dnr_train.jsonl --val-manifest manifests\dnr_val.jsonl --output-dir models\stage1_dialogue\gatorade_v1 --epochs 30 --batch-size 4
```

Important outputs:

- `models\stage1_dialogue\gatorade_v1\latest.pt`
- `models\stage1_dialogue\gatorade_v1\best.pt`
- `models\stage1_dialogue\gatorade_v1\metrics.jsonl`

## 3. Run offline inference

Example:

```powershell
python -m stage1_dialogue.infer --checkpoint models\stage1_dialogue\gatorade_v1\best.pt --input video\Gatorade.mp4 --output-dir output\stage1_gatorade
```

## 4. Wire it into the app

Set:

```powershell
$env:STAGE1_VOICEOVER_CHECKPOINT = "C:\Users\Arjun\OneDrive\Coding\Video-Music-Swapper\models\stage1_dialogue\gatorade_v1\best.pt"
$env:VOICEOVER_BACKEND = "stage1_dialogue"
streamlit run app.py
```

Then enable:

- `Preserve original voiceover`
- `Voiceover backend -> Stage 1 dialogue model`

## 5. Benchmark it

After rendering:

```powershell
powershell -ExecutionPolicy Bypass -File .\eval\run_gatorade_benchmark.ps1 -Candidate "C:\Users\Arjun\Downloads\Gatorade_stage1.mp4"
```

Compare the Stage 1 output against the heuristic baseline before keeping it.
