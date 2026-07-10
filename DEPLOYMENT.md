# Sonic Segments — testing regime & deployment mechanism

How to verify the platform locally, and exactly how a rollout bundle goes live
on each ad platform. Everything below is spend-safe by construction: every
generated entity is **paused/draft**, demo-only tracks are gate-checked out of
trafficking, and the publish CLI defaults to dry-run (zero network).

## 1. Local testing regime (run these constantly)

All commands use the conda python (`/opt/miniconda3/bin/python`).

### Tier 1 — unit + integration tests (~1s, no models)

```bash
/opt/miniconda3/bin/python -m pytest tests -q
```

Covers: KB integrity (every geo pack research-cited, every platform spec
complete), targeting derivation for all 30 segments, planner behavior
(affinity matching, market-ban skips, budget conservation, demo-only gating),
export bundle round-trips (Meta payload shape, Editor CSV / SDF headers,
zip contents), publish request-sequence correctness, a network-call tripwire
for dry-run, and the rollout endpoints through the real FastAPI app.

### Tier 2 — exhaustive benchmark (~0.1s, deterministic layer)

```bash
/opt/miniconda3/bin/python -m eval.rollout_benchmark
```

Sweeps the full sellable matrix — every (segment × mood) with its curated
geos across all 5 platforms — and gate-checks every plan. Must print
`VERDICT: PASS` (currently: age parse 30/30, affinity 30/30, spotify genre
30/30, 300/300 plans, ~2,700 ad units, 0 failures). Run after ANY edit to
`data/*.json` or `intelligence/rollout.py` and quote the numbers in the
commit message. (The render-quality benchmark is separate:
`python -m eval.mood_benchmark`.)

### Tier 3 — live local end-to-end

```bash
/opt/miniconda3/bin/python -m uvicorn sonic_segments.web.main:app --port 8000
```

1. Open http://localhost:8000, drop an ad, pick audiences (+ optional geo
   focus), render variants (UVR render stack, unchanged).
2. On the preview page: **Roll it out** → pick platforms/markets/budget →
   plan + download `rollout_bundle.zip`.
3. Simulate the deployment (prints the exact API request sequence, sends
   nothing):

```bash
/opt/miniconda3/bin/python -m sonic_segments.publish --job <campaign_id> --platform meta
/opt/miniconda3/bin/python -m sonic_segments.publish --job <campaign_id> --platform dv360
```

4. Spot-check bundle quality: `rollout/spotify/audio/*.mp3` should measure
   ≈ -16 LUFS / 44.1kHz stereo (`ffmpeg -af loudnorm=print_format=json`).

## 2. Deployment mechanism per platform

The bundle maps 1:1 onto each platform's existing bulk/variant machinery —
that is the product thesis: micro-demographic distribution using only the
controls advertisers already have.

| Platform | Bundle file | Upload path | Lands as |
|---|---|---|---|
| Meta | `meta/adsets.json` | Marketing API (`publish.py` emits the exact POST sequence) | ad sets `PAUSED` |
| Google Ads (Demand Gen / YouTube) | `google_ads/demand_gen.csv` | Google Ads Editor → import | campaigns `Paused` |
| DV360 (CTV / **YouTube TV**) | `dv360/sdf_line_items.csv` | Advertiser → Insertion Orders → Upload SDF | line items `Draft` |
| TikTok | `tiktok/adgroups.json` | Marketing API `/adgroup/create/` | `operation_status=DISABLE` |
| Spotify | `spotify/adsets.json` + `spotify/audio/` | Ads Manager self-serve (API is closed beta) | drafts |

### Free, spend-proof rehearsal accounts

- **Meta sandbox ad account**: created from any Meta developer app
  (Dashboard → Marketing API → Sandbox). Full API surface, cannot deliver or
  spend. Then: `export META_ACCESS_TOKEN=... META_AD_ACCOUNT_ID=act_<sandbox>
  META_PAGE_ID=...` and run publish with `--live`. The CLI refuses to run live
  while the plan has gate-check issues, refuses non-Meta platforms, and
  creates everything `PAUSED` even in the sandbox.
- **Google Ads test manager account**: ads.google.com/nocache → test account
  from a manager account; API calls validate fully, nothing serves.
- **DV360**: SDF upload validates the file before anything is created; drafts
  never serve. TikTok offers sandbox ad accounts via its developer portal.

### The measurement loop (what the brand manager buys)

Each ad unit isolates ONE music-per-audience hypothesis (same footage, same
voiceover, same targeting except the split dimension). Compare per-unit
CTR / thruplay / completion against the original-score control; scale budget
to winning cuts. That per-audience lift report is the renewal pitch.

## 3. Hard safety rails (do not remove)

- `validate_plan` blocks demo-only (uncleared) tracks from every export path.
- Every writer stamps paused/draft/disabled status; tests assert it.
- `publish.py` is dry-run by default; live mode is double-gated (`--live`
  flag AND env credentials) and Meta-sandbox-scoped.
- Only non-sensitive targeting attributes are ever emitted: age, geography,
  music/interest affinities. No race, religion, health, or politics proxies —
  keeps every plan inside Meta/Google restricted-category policy.

## 4. Where this goes next (the YouTube TV vision)

Today's bundle already buys living-room screens per-DMA via DV360 (YouTube TV
inventory, ConnectedTV device targeting). The path to "every viewer hears
their own score" is server-side ad insertion (the same mechanism that already
geo-swaps podcast ads inside one Spotify stream): our per-segment cuts become
the creative pool an SSAI decision engine picks from per household. The
per-DMA lift data generated by today's rollouts is the evidence that earns
that integration conversation.
