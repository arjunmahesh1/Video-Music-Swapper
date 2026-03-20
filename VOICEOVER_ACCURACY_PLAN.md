# Voiceover Accuracy Plan

This app's hard problem is not generic "vocal separation". It is closer to cinematic dialogue extraction:

- keep spoken narration intact
- remove background music
- remove sung lyrics and ad-libs from the original soundtrack
- preserve timing and natural speech continuity

The current heuristic pipeline can only approximate that. It uses music-source-separation models plus speech masking, which is why it can alternate between two failure modes:

- voice drops out because speech windows are too sparse
- lyrics leak through because music-vocal models do not know which voice is the narrator

## Recommended backend

Use a two-stage trainable backend, then keep the current heuristic path only as a fallback.

### Stage A: Dialogue / Music / Effects separator

Train a separator for `dialogue`, `music`, and `effects`, not `vocals` vs `accompaniment`.

Recommended family:

- Bandsplit RNN or related bandsplit-transformer style model
- dataset target: Divide and Remaster v3

Why:

- Divide and Remaster v3 is explicitly built for `dialogue`, `music`, and `effects`
- its dialogue stem removes speech/vocals from music/effects and better matches ad voiceover than music-only demixing

Primary sources:

- Divide and Remaster v3 dataset: https://github.com/kwatcharasupat/divide-and-remaster-v3
- Divide and Remaster summary (dialogue stem details): https://zenodo.org/records/12659887
- Generalized Bandsplit model on DnR: https://huggingface.co/papers/2309.02539

### Stage B: Target narrator extractor

Run a target-speaker extractor on top of the Stage A dialogue stem.

Recommended family:

- VoiceFilter / VoiceFilter-Lite style speaker-conditioned separator

Why:

- the core ambiguity is not "speech vs music" only
- it is "narrator speech vs sung vocals / hype vocals / crowd vocals"
- a target-speaker model can keep the narrator once we estimate a speaker embedding from the cleanest narrator regions

Primary sources:

- VoiceFilter: https://research.google/pubs/voicefilter-targeted-voice-separation-by-speaker-conditioned-spectrogram-masking/
- VoiceFilter-Lite: https://research.google/pubs/voicefilter-lite-streaming-targeted-voice-separation-for-on-device-speech-recognition/

### Stage C: Soft confidence fusion

Do not hard-gate the final voice track.

Instead:

- fuse Stage A dialogue stem and Stage B target-speaker stem
- drive fusion with a frame-level confidence mask
- keep a low-level continuity bed to avoid missing phrase tails

This is the main architectural fix for "voice starts okay, then disappears".

## Automatic inference path

The production path should stay upload-only. No transcript paste should be required.

Proposed automatic inference:

1. Stage A separator predicts `dialogue`, `music`, `effects`.
2. Run VAD + diarization on the predicted dialogue stem.
3. Pick the dominant recurring speaker as the narrator.
4. Build a speaker embedding from the cleanest narrator segments.
5. Run Stage B target-speaker extraction on the dialogue stem.
6. Soft-fuse `dialogue` and `target_narrator`.
7. Mix fused narration over replacement music.

Useful building blocks:

- Silero VAD for cheap robust speech timestamps:
  https://github.com/snakers4/silero-vad
- pyannote.audio for speaker segmentation and embeddings:
  https://github.com/pyannote/pyannote-audio

## Training data

Accuracy will depend more on data design than on a single architecture choice.

### Supervised data

Use:

- Divide and Remaster v3 for dialogue/music/effects separation
- MUSDB or similar only as auxiliary music separation data, not as the primary objective

### Synthetic ad mixtures

Create a synthetic corpus that matches the real problem:

- clean voiceover speech
- full songs with vocals
- sound effects
- ad-like mastering, compression, sidechain, ducking, loudness normalization

Train with mixtures where the target is strictly narrator-only speech.

This synthetic corpus should intentionally include:

- rap vocals under narration
- shouted ad-libs
- crowd chants
- compression and limiter artifacts
- phone/commercial VO timbre

### Weakly supervised real ads

Use transcript services only offline for training/evaluation, not at runtime.

For real ads:

- collect timestamped transcripts where available
- align them to audio
- use them as weak speech-presence labels
- mine narrator reference segments for speaker embeddings

This is a good use of transcript data because it improves the model without adding user friction to the product.

## Loss design

For best accuracy, train with multi-objective losses:

- separation loss:
  SI-SDR or multi-resolution STFT loss on dialogue target
- speaker consistency loss:
  cosine loss between output speech embedding and narrator embedding
- intelligibility loss:
  ASR consistency or CTC-style loss using a frozen speech model on narrator regions
- anti-leakage loss:
  penalize correlation between output speech and music stem
- continuity loss:
  encourage stable energy through known speech spans so phrase tails do not vanish

## What to build first

### Phase 1

Train only Stage A:

- DnR-v3 dialogue/music/effects separator
- replace current Demucs-first heuristic for ads

This alone should be a big accuracy jump because the task matches the dataset.

### Phase 2

Add Stage B target narrator extraction:

- estimate narrator embedding from Stage A dialogue output
- train VoiceFilter-style extractor on synthetic ad mixtures

This is the part that should finally suppress FEIN-like sung vocals while keeping the narrator.

### Phase 3

Quantize / distill after accuracy is acceptable.

Do not optimize for instant speed until:

- narrator retention is reliably high
- lyric leakage is reliably low
- the benchmark stops failing on representative ads

## Benchmark targets

Use the benchmark script in this repo to score each experiment.

Minimum useful metrics:

- voice retention dB delta inside known speech windows
- music leakage correlation against original music stem
- choppiness count (`jumps_gt12db`)
- high-frequency retention (proxy for muffling)

Success criteria for this app should be closer to:

- voice retention better than `-3 dB`
- music leakage correlation under `0.03`
- low choppiness on narration windows

## Practical recommendation

If the goal is "best accuracy first", the highest-leverage route is:

1. Train a DnR-v3 dialogue/music/effects separator.
2. Add a target-speaker extractor conditioned on narrator embeddings.
3. Evaluate every run with objective voice-retention + leakage metrics.

That is the first path here that is structurally aligned with the real problem instead of trying to coerce a music-vocal separator into acting like a narrator isolator.
