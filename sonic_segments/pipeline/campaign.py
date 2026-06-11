"""Campaign orchestration: one re-scored cut per (demographic, mood) target.

Flow per campaign:
1. analyze the ad once (Demucs stems -> mood profile on the music bed,
   speech coverage on the vocal bed)
2. resolve a MusicDirection per target
3. pull candidates from every selected source, rank with the matching
   engine, pick the winner
4. render with voiceover preservation and write a manifest the preview
   page can serve
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..intelligence import DemographicsKB, detect_mood, rank_tracks
from ..intelligence.demographics import MusicDirection
from ..sources import get_sources
from .jobs import Job

MAX_CUTS = 5
CANDIDATES_PER_SOURCE = 3


def prepare_and_run(job: Job) -> dict[str, Any]:
    """Web entrypoint: resolve the media (upload or URL), then run the campaign."""
    from .ingest import ingest_media

    params = job.params
    job.log("Ingesting media...", pct=2)
    result = ingest_media(job.dir, file_path=params.get("staged_file"), url=params.get("url"))

    source = job.dir / "source.mp4"
    if result.video_path.resolve() != source.resolve():
        if result.video_path.suffix.lower() == ".mp4":
            shutil.move(str(result.video_path), str(source))
        else:  # remux to mp4 so browsers can play it
            import subprocess

            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", str(result.video_path), "-c", "copy", str(source)],
                capture_output=True, text=True, timeout=300,
            )
            if proc.returncode != 0:  # codec not mp4-compatible: re-encode
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(result.video_path), "-c:v", "libx264", "-c:a", "aac", str(source)],
                    capture_output=True, text=True, timeout=900,
                )
            if result.video_path.exists() and "uploads" in str(result.video_path):
                result.video_path.unlink(missing_ok=True)

    params["video_path"] = str(source)
    params["duration"] = result.quality.duration or 30.0
    params["quality_warnings"] = result.quality.warnings + result.notes
    for warning in params["quality_warnings"]:
        job.log(f"Quality note: {warning}")

    return run_campaign(job)


def run_campaign(job: Job) -> dict[str, Any]:
    params = job.params
    video_path = Path(params["video_path"])
    mode = params.get("mode", "variants")  # "variants" | "demo"
    moods: list[str] = params.get("moods") or ["happy"]
    demographics: list[str] = params.get("demographics") or []
    source_ids: list[str] = params.get("sources") or (["spotify"] if mode == "demo" else ["library"])
    duration = float(params.get("duration") or 30.0)

    kb = DemographicsKB.default()

    # ---- 1. Analyze the ad ------------------------------------------------
    job.log("Separating voiceover and music bed (Demucs)...", pct=5)
    stems = _separate_stems(video_path, job.dir / "stems")

    job.log("Profiling the ad's current sound (CLAP + DSP)...", pct=15)
    # Benchmarked on 13 labeled ads: full-mix mood detection beats the
    # separated music stem (54% vs 38% top-1) — stems lose sung vocals
    # and add artifacts, and our CLAP variant handles speech fine.
    ad_mood = detect_mood(_full_mix(video_path, job.dir / "stems"))
    speech = _speech_summary(stems, duration)
    job.log(
        f"Current bed reads {ad_mood.primary}/{ad_mood.secondary} at {ad_mood.tempo:.0f} BPM, "
        f"energy {ad_mood.energy:.2f}; narration covers {speech['coverage_ratio']:.0%} of the ad."
    )

    # ---- 2. Build render targets ------------------------------------------
    targets: list[MusicDirection] = []
    if mode == "demo":
        targets = [kb.mood_only_direction(m) for m in moods[:3]]
    else:
        for segment_id in demographics[:MAX_CUTS]:
            for mood in moods:
                targets.append(kb.direction(segment_id, mood))
    targets = targets[:MAX_CUTS]
    if not targets:
        raise ValueError("No targets: pick at least one mood (and demographics for scored variants).")

    # ---- 3+4. Source, rank, render per target ------------------------------
    sources = get_sources()
    context = {
        "uploaded_tracks": params.get("uploaded_tracks") or [],
        "reference_query": params.get("reference_query"),
    }
    variants: list[dict[str, Any]] = []
    base_pct = 20
    span = 75 // max(1, len(targets))

    for idx, direction in enumerate(targets, start=1):
        pct = base_pct + (idx - 1) * span
        label = f"{direction.segment_label} · {direction.mood_label}"
        job.log(f"[{idx}/{len(targets)}] Sourcing music for {label}...", pct=pct)

        candidates: list[dict[str, Any]] = []
        for sid in source_ids:
            src = sources.get(sid)
            if src is None:
                continue
            ok, reason = src.available()
            if not ok:
                job.log(f"  - {src.label}: skipped ({reason})")
                continue
            try:
                found = src.candidates(
                    direction, job.dir / "tracks", duration, limit=CANDIDATES_PER_SOURCE, context=context
                )
                job.log(f"  - {src.label}: {len(found)} candidate(s)")
                candidates.extend(found)
            except Exception as exc:
                job.log(f"  - {src.label}: failed ({exc})")

        if not candidates:
            job.log(f"  ! No candidates for {label}; skipping this cut.")
            continue

        job.log(f"  Scoring {len(candidates)} candidate(s) against the brief...", pct=pct + span // 3)
        ranked = rank_tracks(candidates, direction)
        if not ranked:
            job.log(f"  ! All candidates failed analysis for {label}; skipping.")
            continue
        winner = ranked[0]
        job.log(
            f"  Winner: {winner['name']}"
            + (f" — {winner['artist']}" if winner.get("artist") else "")
            + f" (score {winner['match']['total']:.2f}, via {winner['source']})"
        )

        out_name = f"{idx:02d}_{direction.segment_id}_{direction.mood}.mp4"
        job.log(f"  Rendering re-scored cut (separation + ducking + mux)...", pct=pct + span // 2)
        try:
            render = _render(
                video_path, Path(winner["audio_path"]), job.dir / "variants" / out_name,
                transcript=params.get("transcript"),
            )
        except Exception as exc:
            job.log(f"  ! Render failed for {label}: {exc}")
            continue

        variants.append(
            {
                "file": f"variants/{out_name}",
                "label": label,
                "segment_id": direction.segment_id,
                "segment_label": direction.segment_label,
                "mood": direction.mood,
                "mood_label": direction.mood_label,
                "direction": direction.as_dict(),
                "track": {
                    "name": winner["name"],
                    "artist": winner.get("artist"),
                    "source": winner["source"],
                    "license": winner.get("license", ""),
                    "demo_only": bool(winner.get("demo_only")),
                },
                "match": winner["match"],
                "runner_ups": [
                    {"name": r["name"], "artist": r.get("artist"), "score": r["match"]["total"], "source": r["source"]}
                    for r in ranked[1:4]
                ],
            }
        )

    if not variants:
        raise ValueError("No variants could be rendered — check source availability in the job log.")

    # ---- 5. Manifest --------------------------------------------------------
    manifest = {
        "id": job.id,
        "brand": params.get("brand", ""),
        "vibe": params.get("vibe", ""),
        "mode": mode,
        "original": "source.mp4",
        "duration": duration,
        "quality_warnings": params.get("quality_warnings") or [],
        "analysis": {
            "mood": ad_mood.as_dict(),
            "speech": speech,
            "mode_major": ad_mood.features.get("mode_major"),
        },
        "variants": variants,
    }
    (job.dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def _full_mix(video_path: Path, stems_dir: Path) -> Path:
    from ..adapters import extract_voiceover_source_audio

    stems_dir.mkdir(parents=True, exist_ok=True)
    target = stems_dir / "fullmix.wav"
    if not target.exists():
        extract_voiceover_source_audio(video_path, target)
    return target


def _separate_stems(video_path: Path, stems_dir: Path) -> dict[str, Path]:
    """Run Demucs once per campaign and keep the stems in the job directory."""
    from ..adapters import cleanup_demucs_output, extract_voiceover_source_audio, separate_audio

    music, vocals = stems_dir / "music.wav", stems_dir / "vocals.wav"
    if music.exists() and vocals.exists():
        return {"music": music, "vocals": vocals}

    stems_dir.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        source = extract_voiceover_source_audio(video_path, Path(tmp) / "src.wav")
        separated = separate_audio(source, output_dir=Path(tmp) / "sep")
        shutil.move(str(separated["music"]), str(music))
        shutil.move(str(separated["vocals"]), str(vocals))
    cleanup_demucs_output()
    return {"music": music, "vocals": vocals}


def _speech_summary(stems: dict[str, Path], duration: float) -> dict[str, Any]:
    from ..adapters import calculate_speech_coverage, detect_speech_segments

    try:
        segments = detect_speech_segments(str(stems["vocals"]), accompaniment_path=str(stems["music"]))
    except Exception as exc:
        print(f"[campaign] speech detection failed: {exc}")
        segments = []
    coverage = calculate_speech_coverage(segments)
    return {
        "segments": [[round(a, 2), round(b, 2)] for a, b in segments],
        "coverage_seconds": round(coverage, 2),
        "coverage_ratio": round(coverage / duration, 3) if duration else 0.0,
    }


def _render(video_path: Path, music_path: Path, output_path: Path, transcript: str | None = None):
    from ..service import SonicSegmentsService

    service = SonicSegmentsService()
    return service.render_variant(
        video_path=video_path,
        music_path=music_path,
        output_path=output_path,
        preserve_voiceover=True,
        transcript_hint_text=transcript,
    )
