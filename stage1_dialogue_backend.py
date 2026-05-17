"""App-facing wrapper for the trainable Stage 1 dialogue backend."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from stage1_dialogue.inference import default_checkpoint_path, run_stage1_separation, stage1_checkpoint_available
from voice_separator import (
    DISABLE_FASTER_WHISPER,
    MAX_SEGMENTS_FOR_EXPRESSION,
    _apply_post_gain_limiter,
    _blend_voice_continuity,
    _debleed_with_ffmpeg_sidechain,
    _measure_rms_dbfs,
    _probe_duration_seconds,
    _run_checked,
    detect_song_start,
    detect_speech_segments,
    evaluate_mix_quality,
    mix_vocals_with_music_ducking,
    parse_transcript_speech_segments,
    suppress_music_bleed_with_reference,
)


def resolve_stage1_checkpoint_path() -> Path:
    return Path(os.getenv("STAGE1_VOICEOVER_CHECKPOINT", str(default_checkpoint_path())))


def stage1_backend_available(checkpoint_path: str | Path | None = None) -> bool:
    return stage1_checkpoint_available(checkpoint_path or resolve_stage1_checkpoint_path())


def _run_stage1_postprocess(
    video_audio_path: Path,
    separated: dict,
    new_music_path: Path,
    output_path: Path,
    vocals_volume: float,
    music_volume: float,
    transcript_hint_text: str | None,
):
    duration_seconds = _probe_duration_seconds(video_audio_path) or 120.0
    ffmpeg_timeout = int(min(1800, max(180, duration_seconds * 1.5)))

    print("Step 1.5: Cleaning vocals track (removing residual music)...")
    cleaned_vocals_path = Path(tempfile.gettempdir()) / "cleaned_vocals.wav"
    preprocess_filter = "highpass=f=95,lowpass=f=9000,afftdn=nf=-26:nt=w"
    try:
        _run_checked(
            [
                "ffmpeg", "-y",
                "-i", str(separated["vocals"]),
                "-af", preprocess_filter,
                str(cleaned_vocals_path),
            ],
            timeout=ffmpeg_timeout,
            step_name="Vocal cleanup",
        )
    except Exception as e:
        print(f"Warning: Advanced vocal cleanup failed ({e}), retrying with compatibility filter")
        _run_checked(
            [
                "ffmpeg", "-y",
                "-i", str(separated["vocals"]),
                "-af", "highpass=f=95,lowpass=f=9000",
                str(cleaned_vocals_path),
            ],
            timeout=ffmpeg_timeout,
            step_name="Vocal cleanup (compatibility)",
        )

    print("Step 1.6: Suppressing residual original music from vocals...")
    debleed_vocals_path = Path(tempfile.gettempdir()) / "debleed_vocals.wav"
    speech_source_path = cleaned_vocals_path
    try:
        suppress_music_bleed_with_reference(
            cleaned_vocals_path,
            separated["music"],
            debleed_vocals_path,
            suppression_strength=1.12,
            residual_floor=0.10,
            sr=32000,
            speech_high_cut_hz=8200,
        )
        speech_source_path = debleed_vocals_path
    except Exception as e:
        print(f"Warning: Music bleed suppression failed ({e}); trying ffmpeg fallback.")
        try:
            _debleed_with_ffmpeg_sidechain(
                cleaned_vocals_path,
                separated["music"],
                debleed_vocals_path,
                timeout=ffmpeg_timeout,
            )
            speech_source_path = debleed_vocals_path
            print("Applied ffmpeg de-bleed fallback.")
        except Exception as fallback_error:
            print(
                f"Warning: FFmpeg de-bleed fallback failed ({fallback_error}); "
                "continuing with cleaned vocals."
            )

    print("Step 2: Detecting voiceover segments...")
    using_transcript_segments = False
    speech_segments = []
    if transcript_hint_text:
        parsed_segments = parse_transcript_speech_segments(
            transcript_hint_text,
            duration_seconds=duration_seconds,
        )
        if parsed_segments:
            speech_segments = parsed_segments
            using_transcript_segments = True
            print(f"Using transcript-guided speech segments: {len(speech_segments)}")
        else:
            print("Transcript hint provided but no timestamps were parsed; using ASR/VAD detection.")

    if not speech_segments:
        speech_segments = detect_speech_segments(
            speech_source_path,
            accompaniment_path=separated["music"],
        )

    total_detected_speech = sum((end - start) for start, end in speech_segments) if speech_segments else 0.0
    min_expected_speech = max(4.0, duration_seconds * 0.14)
    force_full_track_gate = not speech_segments or total_detected_speech < min_expected_speech
    print(f"Detected speech coverage: {total_detected_speech:.1f}s / {duration_seconds:.1f}s")

    if not speech_segments:
        print("No speech segments detected; using full-track speech-gate fallback.")
    elif total_detected_speech < min_expected_speech:
        print(
            f"Low-confidence speech detection ({total_detected_speech:.1f}s total); "
            "using full-track speech-gate fallback to avoid dropping narration."
        )
    elif DISABLE_FASTER_WHISPER and not using_transcript_segments:
        print("ASR unavailable in this environment; using acoustic-only speech windows.")

    print("Step 3: Extracting voiceover segments...")
    clean_voiceover_path = Path(tempfile.gettempdir()) / "clean_voiceover.wav"
    if force_full_track_gate or len(speech_segments) > MAX_SEGMENTS_FOR_EXPRESSION:
        print(
            f"{len(speech_segments)} speech segments detected; "
            "using aggressive full-track speech gate fallback."
        )
        fallback_filter = (
            "highpass=f=95,lowpass=f=9000,"
            "agate=threshold=0.0022:ratio=1.18:attack=7:release=450:range=0.72,"
            "acompressor=threshold=0.11:ratio=1.7:attack=11:release=210:makeup=2.6"
        )
        _run_checked(
            [
                "ffmpeg", "-y",
                "-i", str(speech_source_path),
                "-af", fallback_filter,
                str(clean_voiceover_path),
            ],
            timeout=ffmpeg_timeout,
            step_name="Speech extraction (fallback gate)",
        )
    else:
        enable_conds = [f"between(t,{start:.3f},{end:.3f})" for start, end in speech_segments]
        enable_expr = "+".join(enable_conds)
        speech_mask_filter = (
            f"volume=enable='{enable_expr}':volume=1,"
            f"volume=enable='not({enable_expr})':volume=0,"
            "highpass=f=95,lowpass=f=9000,"
            "agate=threshold=0.0024:ratio=1.22:attack=7:release=420:range=0.68"
        )
        _run_checked(
            [
                "ffmpeg", "-y",
                "-i", str(speech_source_path),
                "-af", speech_mask_filter,
                str(clean_voiceover_path),
            ],
            timeout=ffmpeg_timeout,
            step_name="Speech extraction",
        )

    print("Step 3.2: Final voiceover de-bleed (removing residual original music)...")
    final_voiceover_path = Path(tempfile.gettempdir()) / "final_voiceover.wav"
    mix_voice_source = clean_voiceover_path
    try:
        pre_final_rms = _measure_rms_dbfs(clean_voiceover_path)
        suppress_music_bleed_with_reference(
            clean_voiceover_path,
            separated["music"],
            final_voiceover_path,
            suppression_strength=1.14,
            residual_floor=0.10,
            sr=32000,
            speech_high_cut_hz=7600,
        )
        post_final_rms = _measure_rms_dbfs(final_voiceover_path)
        if post_final_rms < pre_final_rms - 2.4:
            print(
                f"Final de-bleed reduced voice too much ({pre_final_rms:.2f} -> {post_final_rms:.2f} dBFS); "
                "keeping pre-final voiceover."
            )
        else:
            mix_voice_source = final_voiceover_path
    except Exception as e:
        print(f"Warning: Final de-bleed pass failed ({e}); trying ffmpeg fallback.")
        try:
            _debleed_with_ffmpeg_sidechain(
                clean_voiceover_path,
                separated["music"],
                final_voiceover_path,
                timeout=ffmpeg_timeout,
            )
            mix_voice_source = final_voiceover_path
            print("Applied final ffmpeg de-bleed fallback.")
        except Exception as fallback_error:
            print(
                f"Warning: Final ffmpeg de-bleed fallback failed ({fallback_error}); "
                "using pre-final voiceover track."
            )

    if DISABLE_FASTER_WHISPER and not using_transcript_segments:
        try:
            speech_ratio = total_detected_speech / duration_seconds if duration_seconds > 0 else 0.0
            continuity_gain = 0.52 if speech_ratio < 0.18 else 0.38 if speech_ratio < 0.30 else 0.28
            continuity_path = Path(tempfile.gettempdir()) / "continuity_voice.wav"
            _run_checked(
                [
                    "ffmpeg", "-y",
                    "-i", str(speech_source_path),
                    "-af",
                    "highpass=f=85,lowpass=f=9500,"
                    "agate=threshold=0.0018:ratio=1.20:attack=8:release=420:range=0.72,"
                    "acompressor=threshold=0.11:ratio=1.55:attack=10:release=180:makeup=2.2",
                    str(continuity_path),
                ],
                timeout=max(180, ffmpeg_timeout),
                step_name="Continuity voice preparation",
            )
            blended_voice_path = Path(tempfile.gettempdir()) / "blended_voiceover.wav"
            _blend_voice_continuity(
                mix_voice_source,
                continuity_path,
                blended_voice_path,
                timeout=max(240, ffmpeg_timeout),
                continuity_gain=continuity_gain,
            )
            mix_voice_source = blended_voice_path
            print(
                "Applied voice continuity blend for ASR-unavailable mode "
                f"(speech_ratio={speech_ratio:.2f}, gain={continuity_gain:.2f})."
            )
        except Exception as e:
            print(f"Warning: Continuity blend failed ({e}); using primary voice track.")

    try:
        voice_rms_db = _measure_rms_dbfs(mix_voice_source)
        print(f"Voiceover RMS before mix: {voice_rms_db:.2f} dBFS")
        if voice_rms_db < -26.5:
            gain_db = min(9.0, -23.5 - voice_rms_db)
            boosted_voice_path = Path(tempfile.gettempdir()) / "boosted_voiceover.wav"
            _apply_post_gain_limiter(
                mix_voice_source,
                boosted_voice_path,
                gain_db=gain_db,
                timeout=max(180, ffmpeg_timeout),
            )
            mix_voice_source = boosted_voice_path
            print(f"Boosted quiet voiceover by +{gain_db:.1f} dB before mixing.")
    except Exception as e:
        print(f"Warning: Voiceover RMS check failed ({e}); continuing without pre-mix gain boost.")

    intro_skip = detect_song_start(new_music_path)
    music_to_use = new_music_path
    if intro_skip > 0:
        trimmed_music_path = Path(tempfile.gettempdir()) / "music_no_intro.wav"
        try:
            _run_checked(
                [
                    "ffmpeg", "-y",
                    "-ss", str(intro_skip),
                    "-i", str(new_music_path),
                    "-acodec", "pcm_s16le",
                    "-ar", "44100",
                    "-ac", "2",
                    str(trimmed_music_path),
                ],
                timeout=ffmpeg_timeout,
                step_name="Music intro trim",
            )
            music_to_use = trimmed_music_path
        except Exception as e:
            print(f"Warning: Intro skip failed: {e}")

    print("Step 4: Mixing voiceover with new music (auto-ducking enabled)...")
    mix_vocals_with_music_ducking(
        mix_voice_source,
        music_to_use,
        output_path,
        vocals_volume=vocals_volume,
        music_volume=music_volume,
        process_timeout=max(600, ffmpeg_timeout),
    )

    print("Step 4.5: Evaluating swap quality...")
    qa = evaluate_mix_quality(output_path, separated["music"])
    selected_qa = qa
    print(
        "Mix QA:"
        f" rms={qa.get('rms_dbfs')},"
        f" jumps12={qa.get('jumps_gt12db')},"
        f" hf6k={qa.get('hf_ratio_ge6k')},"
        f" leak={qa.get('music_leak_corr')},"
        f" issues={qa.get('issues')}"
    )

    if qa.get("retry_recommended"):
        print("Quality issues detected, remixing once with adjusted settings...")
        retry_output = output_path.with_name(f"{output_path.stem}_retry{output_path.suffix}")
        retry_vocals_volume = min(1.45, vocals_volume * 1.15)
        retry_music_volume = max(0.42, music_volume * 0.82)
        retry_gate_threshold = 0.0032
        retry_voice_lowpass = 10200
        retry_sidechain_threshold = 0.010
        retry_sidechain_ratio = 9.0
        retry_output_makeup = 1.45

        if "possible_original_music_bleed" in qa["issues"]:
            retry_music_volume = max(0.35, retry_music_volume * 0.90)
            retry_sidechain_ratio = max(retry_sidechain_ratio, 10.0)
        if "voice_choppy_or_pumping" in qa["issues"]:
            retry_gate_threshold = 0.0028
            retry_sidechain_ratio = 6.0
            retry_sidechain_threshold = 0.014
            retry_voice_lowpass = max(retry_voice_lowpass, 10800)
        if "output_too_muffled" in qa["issues"]:
            retry_voice_lowpass = max(retry_voice_lowpass, 11800)
            retry_gate_threshold = min(retry_gate_threshold, 0.0024)
            retry_output_makeup = max(retry_output_makeup, 1.55)
        if "mix_too_quiet" in qa["issues"]:
            retry_vocals_volume = min(1.60, retry_vocals_volume * 1.15)
            retry_music_volume = max(0.35, retry_music_volume * 0.85)

        mix_vocals_with_music_ducking(
            mix_voice_source,
            music_to_use,
            retry_output,
            vocals_volume=retry_vocals_volume,
            music_volume=retry_music_volume,
            gate_threshold=retry_gate_threshold,
            process_timeout=max(600, ffmpeg_timeout),
            voice_lowpass_hz=retry_voice_lowpass,
            sidechain_threshold=retry_sidechain_threshold,
            sidechain_ratio=retry_sidechain_ratio,
            output_makeup=retry_output_makeup,
        )

        retry_qa = evaluate_mix_quality(retry_output, separated["music"])
        print(
            "Retry Mix QA:"
            f" rms={retry_qa.get('rms_dbfs')},"
            f" jumps12={retry_qa.get('jumps_gt12db')},"
            f" hf6k={retry_qa.get('hf_ratio_ge6k')},"
            f" leak={retry_qa.get('music_leak_corr')},"
            f" issues={retry_qa.get('issues')}"
        )

        prefer_retry = len(retry_qa.get("issues", [])) < len(qa.get("issues", []))
        if not prefer_retry:
            base_rms = qa.get("rms_dbfs") if qa.get("rms_dbfs") is not None else -99
            retry_rms = retry_qa.get("rms_dbfs") if retry_qa.get("rms_dbfs") is not None else -99
            base_jumps = qa.get("jumps_gt12db") if qa.get("jumps_gt12db") is not None else 999
            retry_jumps = retry_qa.get("jumps_gt12db") if retry_qa.get("jumps_gt12db") is not None else 999
            prefer_retry = retry_rms > base_rms + 1.0 or retry_jumps + 6 < base_jumps

        if prefer_retry:
            try:
                shutil.move(str(retry_output), str(output_path))
                print("Using remixed output after QA adaptation.")
                selected_qa = retry_qa
            except Exception:
                print("Warning: Could not promote retry output; keeping original mix.")
        else:
            print("Keeping original mix after QA comparison.")

    if selected_qa.get("rms_dbfs") is not None and selected_qa["rms_dbfs"] < -19.0:
        target_rms_dbfs = -15.5
        gain_db = min(12.0, target_rms_dbfs - selected_qa["rms_dbfs"])
        if gain_db > 0.4:
            print(f"Step 4.6: Applying post-gain loudness correction (+{gain_db:.1f} dB)...")
            loud_tmp = output_path.with_name(f"{output_path.stem}_loud{output_path.suffix}")
            try:
                _apply_post_gain_limiter(
                    output_path,
                    loud_tmp,
                    gain_db=gain_db,
                    timeout=max(300, ffmpeg_timeout),
                )
                shutil.move(str(loud_tmp), str(output_path))
                final_qa = evaluate_mix_quality(output_path, separated["music"])
                print(
                    "Post-gain Mix QA:"
                    f" rms={final_qa.get('rms_dbfs')},"
                    f" jumps12={final_qa.get('jumps_gt12db')},"
                    f" hf6k={final_qa.get('hf_ratio_ge6k')},"
                    f" leak={final_qa.get('music_leak_corr')},"
                    f" issues={final_qa.get('issues')}"
                )
            except Exception as e:
                print(f"Warning: Post-gain loudness correction failed ({e}).")

    return output_path


def separate_and_remix_stage1(
    video_audio_path,
    new_music_path,
    output_path,
    vocals_volume=1.0,
    music_volume=0.7,
    transcript_hint_text=None,
    checkpoint_path: str | Path | None = None,
):
    checkpoint_path = Path(checkpoint_path or resolve_stage1_checkpoint_path())
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at {checkpoint_path}. "
            "Train it first or set STAGE1_VOICEOVER_CHECKPOINT."
        )

    stage1_output_dir = Path(tempfile.gettempdir()) / "stage1_dialogue_output"
    stage1_output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Separating dialogue/music/effects with Stage 1 backend...")
    separated_stems = run_stage1_separation(
        input_path=video_audio_path,
        output_dir=stage1_output_dir,
        checkpoint_path=checkpoint_path,
    )
    print(f"Stage 1 separation complete using checkpoint: {checkpoint_path.name}")

    separated = {
        "vocals": separated_stems["dialogue"],
        "music": separated_stems["music"],
        "effects": separated_stems.get("effects"),
        "model_used": checkpoint_path.stem,
    }
    return _run_stage1_postprocess(
        video_audio_path=Path(video_audio_path),
        separated=separated,
        new_music_path=Path(new_music_path),
        output_path=Path(output_path),
        vocals_volume=vocals_volume,
        music_volume=music_volume,
        transcript_hint_text=transcript_hint_text,
    )


def cleanup_stage1_output(output_dir: str | Path | None = None) -> None:
    dirs_to_clean = [Path(tempfile.gettempdir()) / "stage1_dialogue_output"]
    if output_dir:
        dirs_to_clean.append(Path(output_dir))

    for dir_path in dirs_to_clean:
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"Cleaned up: {dir_path}")
            except Exception:
                pass
