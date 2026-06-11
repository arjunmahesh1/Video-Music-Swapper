"""High-level orchestration for the Sonic Segments MVP."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from .adapters import (
    analyze_audio_features,
    calculate_speech_coverage,
    cleanup_demucs_output,
    detect_speech_segments,
    extract_audio_from_video,
    extract_voiceover_source_audio,
    get_default_voice_backend_id,
    infer_video_genres,
    mux_audio_with_video,
    separate_and_remix_with_backend,
    separate_audio,
)
from .models import AdAnalysis, BrandProfile, TrackCandidate, VariantRender, VoiceoverAnalysis
from .reports import write_sonic_audit_report


class SonicSegmentsService:
    """Product-oriented service layer around the existing repo pipeline."""

    def __init__(self, voice_backend_id: str | None = None) -> None:
        self.voice_backend_id = voice_backend_id or get_default_voice_backend_id()

    def analyze_ad(
        self,
        video_path: str | Path,
        brand_profile: BrandProfile | None = None,
        include_voiceover: bool = True,
    ) -> AdAnalysis:
        """Analyze an ad and optionally estimate narration coverage."""
        video_path = Path(video_path)
        audio_path = Path(extract_audio_from_video(video_path))
        features = analyze_audio_features(audio_path)
        inferred_genres = infer_video_genres(features)

        voiceover = self._analyze_voiceover(video_path) if include_voiceover else None
        analysis = AdAnalysis(
            video_path=video_path,
            audio_path=audio_path,
            features=features,
            inferred_genres=inferred_genres,
            brand_profile=brand_profile,
            voiceover=voiceover,
        )
        analysis.mismatches = self._build_mismatches(analysis)
        analysis.recommendations = self._build_recommendations(analysis)
        return analysis

    def render_variant(
        self,
        video_path: str | Path,
        music_path: str | Path,
        output_path: str | Path,
        preserve_voiceover: bool = True,
        transcript_hint_text: str | None = None,
        speech_segments_override: list | None = None,
        vocals_volume: float = 1.0,
        music_volume: float = 0.7,
        track: TrackCandidate | None = None,
    ) -> VariantRender:
        """Render one re-scored ad variant."""
        video_path = Path(video_path)
        music_path = Path(music_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            audio_output_path = output_path.with_suffix(".wav")

            if preserve_voiceover:
                source_audio = extract_voiceover_source_audio(
                    video_path,
                    tmp_path / "voiceover_source.wav",
                )
                separate_and_remix_with_backend(
                    video_audio_path=source_audio,
                    new_music_path=music_path,
                    output_path=audio_output_path,
                    backend_id=self.voice_backend_id,
                    vocals_volume=vocals_volume,
                    music_volume=music_volume,
                    transcript_hint_text=transcript_hint_text,
                    speech_segments_override=speech_segments_override,
                )
            else:
                audio_output_path = music_path

            mux_audio_with_video(video_path, audio_output_path, output_path)

        if preserve_voiceover:
            cleanup_demucs_output()

        return VariantRender(
            track=track or TrackCandidate(track_id=music_path.stem, name=music_path.stem, source_path=music_path),
            output_path=output_path,
            audio_output_path=audio_output_path,
            preserve_voiceover=preserve_voiceover,
            backend_id=self.voice_backend_id if preserve_voiceover else None,
            predicted_notes=self._build_variant_notes(track, preserve_voiceover),
        )

    def generate_variants(
        self,
        video_path: str | Path,
        tracks: list[TrackCandidate],
        output_dir: str | Path,
        preserve_voiceover: bool = True,
        transcript_hint_text: str | None = None,
    ) -> list[VariantRender]:
        """Render multiple candidate variants into one output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        variants: list[VariantRender] = []
        for idx, track in enumerate(tracks, start=1):
            if not track.source_path:
                continue
            output_name = f"{idx:02d}_{_slugify(track.name)}.mp4"
            variant = self.render_variant(
                video_path=video_path,
                music_path=track.source_path,
                output_path=output_dir / output_name,
                preserve_voiceover=preserve_voiceover,
                transcript_hint_text=transcript_hint_text,
                track=track,
            )
            variants.append(variant)
        return variants

    def create_audit_bundle(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        brand_profile: BrandProfile | None = None,
        tracks: list[TrackCandidate] | None = None,
        preserve_voiceover: bool = True,
        transcript_hint_text: str | None = None,
    ) -> tuple[AdAnalysis, list[VariantRender], Path]:
        """Analyze an ad, optionally render variants, and write a markdown audit."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis = self.analyze_ad(video_path, brand_profile=brand_profile, include_voiceover=preserve_voiceover)
        variants = self.generate_variants(
            video_path=video_path,
            tracks=tracks or [],
            output_dir=output_dir / "variants",
            preserve_voiceover=preserve_voiceover,
            transcript_hint_text=transcript_hint_text,
        ) if tracks else []
        report_path = write_sonic_audit_report(output_dir / "sonic_audit.md", analysis, variants=variants)
        return analysis, variants, report_path

    def _analyze_voiceover(self, video_path: Path) -> VoiceoverAnalysis:
        """Estimate narration windows by reusing the current separation stack."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_audio = extract_voiceover_source_audio(video_path, tmp_path / "voice_analysis.wav")
            separated = separate_audio(source_audio, output_dir=tmp_path / "voice_separation")
            speech_segments = detect_speech_segments(
                separated["vocals"],
                accompaniment_path=separated["music"],
            )

        cleanup_demucs_output()
        coverage_seconds = calculate_speech_coverage(speech_segments)
        duration_seconds = float(self._safe_duration_seconds(video_path))
        coverage_ratio = coverage_seconds / duration_seconds if duration_seconds > 0 else 0.0
        return VoiceoverAnalysis(
            backend_id=self.voice_backend_id,
            speech_segments=speech_segments,
            speech_coverage_seconds=coverage_seconds,
            speech_coverage_ratio=coverage_ratio,
            detected_segment_count=len(speech_segments),
        )

    def _build_mismatches(self, analysis: AdAnalysis) -> list[str]:
        """Flag simple brand-to-audio mismatches for the first MVP."""
        if not analysis.brand_profile:
            return []

        mismatches: list[str] = []
        energy = float(analysis.features.get("energy", 0.0))
        tempo = float(analysis.features.get("tempo", 0.0))
        brand = analysis.brand_profile

        if brand.desired_energy_max is not None and energy > brand.desired_energy_max:
            mismatches.append(
                f"Energy is high at {energy:.2f}, above the desired ceiling of {brand.desired_energy_max:.2f}."
            )
        if brand.desired_energy_min is not None and energy < brand.desired_energy_min:
            mismatches.append(
                f"Energy is low at {energy:.2f}, below the desired floor of {brand.desired_energy_min:.2f}."
            )
        if brand.desired_tempo_max is not None and tempo > brand.desired_tempo_max:
            mismatches.append(
                f"Tempo is fast at {tempo:.0f} BPM, above the desired ceiling of {brand.desired_tempo_max:.0f} BPM."
            )
        if brand.desired_tempo_min is not None and tempo < brand.desired_tempo_min:
            mismatches.append(
                f"Tempo is slow at {tempo:.0f} BPM, below the desired floor of {brand.desired_tempo_min:.0f} BPM."
            )
        if brand.desired_genres:
            overlap = {genre.lower() for genre in analysis.inferred_genres} & {
                genre.lower() for genre in brand.desired_genres
            }
            if not overlap:
                mismatches.append(
                    "Inferred genres do not overlap with the desired sonic direction: "
                    + ", ".join(brand.desired_genres)
                    + "."
                )

        return mismatches

    def _build_recommendations(self, analysis: AdAnalysis) -> list[str]:
        """Generate simple next-step recommendations from the analysis."""
        recommendations: list[str] = []
        energy = float(analysis.features.get("energy", 0.0))
        tempo = float(analysis.features.get("tempo", 0.0))

        if energy >= 0.75:
            recommendations.append("Test calmer replacement music with lower sustained energy.")
        elif energy <= 0.30:
            recommendations.append("Test a version with more rhythmic lift to increase momentum.")

        if tempo >= 120:
            recommendations.append("Try alternatives in the 70-110 BPM range for a less rushed feel.")
        elif tempo <= 75:
            recommendations.append("Try a slightly faster bed to improve pacing and urgency.")

        if analysis.voiceover and analysis.voiceover.speech_coverage_ratio >= 0.30:
            recommendations.append("Prioritize voice-preserving renders because narration drives a large share of the ad.")
        elif analysis.voiceover:
            recommendations.append("Narration coverage is light, so you can test more assertive music swaps safely.")

        if analysis.brand_profile and analysis.brand_profile.desired_genres:
            recommendations.append(
                "Source variants from tracks tagged close to: "
                + ", ".join(analysis.brand_profile.desired_genres)
                + "."
            )

        return recommendations

    @staticmethod
    def _safe_duration_seconds(video_path: Path) -> float:
        """Use ffprobe to estimate the duration of the source asset."""
        from subprocess import run

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = run(cmd, capture_output=True, text=True)
        try:
            return float((result.stdout or "").strip())
        except ValueError:
            return 0.0

    @staticmethod
    def _build_variant_notes(track: TrackCandidate | None, preserve_voiceover: bool) -> list[str]:
        """Generate short human-readable notes for a rendered variant."""
        notes: list[str] = []
        if preserve_voiceover:
            notes.append("Original narration preserved over replacement music.")
        else:
            notes.append("Full soundtrack replacement without narration preservation.")

        if track and track.genres:
            notes.append("Candidate genres: " + ", ".join(track.genres[:3]) + ".")
        if track and track.artist:
            notes.append(f"Selected artist: {track.artist}.")
        return notes


def _slugify(value: str) -> str:
    """Create filesystem-safe output names for variants."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "track"
