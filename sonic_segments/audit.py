"""CLI entrypoint for creating Sonic Segments audit bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

from .models import BrandProfile


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_optional_text(path: str | None) -> str | None:
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8").strip() or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze an ad, optionally render re-scored variants, and write a Sonic Audit bundle.",
    )
    parser.add_argument("--video", required=True, help="Path to the video ad to analyze.")
    parser.add_argument("--output-dir", required=True, help="Directory for the report and rendered variants.")
    parser.add_argument("--tracks-dir", help="Local folder of replacement tracks to render as variants.")
    parser.add_argument("--variant-limit", type=int, default=3, help="Maximum number of local tracks to render.")
    parser.add_argument(
        "--preserve-voiceover",
        action="store_true",
        help="Run the slower voice-preserving remix path for rendered variants and voiceover coverage.",
    )
    parser.add_argument(
        "--transcript-hint-file",
        help="Optional timestamped transcript file used by the voiceover backend.",
    )
    parser.add_argument(
        "--voice-backend",
        help="Optional voice backend id, such as heuristic or stage1_dialogue.",
    )

    parser.add_argument("--brand-name", help="Brand/client name for the audit report.")
    parser.add_argument("--category", help="Product category.")
    parser.add_argument("--target-demo", help="Target audience or demographic.")
    parser.add_argument("--vibe", help="Desired brand vibe or sonic direction.")
    parser.add_argument("--energy-min", type=float, help="Desired minimum energy score.")
    parser.add_argument("--energy-max", type=float, help="Desired maximum energy score.")
    parser.add_argument("--tempo-min", type=float, help="Desired minimum tempo in BPM.")
    parser.add_argument("--tempo-max", type=float, help="Desired maximum tempo in BPM.")
    parser.add_argument("--genres", help="Comma-separated desired genres.")
    parser.add_argument("--moods", help="Comma-separated desired moods.")
    return parser


def _build_brand_profile(args: argparse.Namespace) -> BrandProfile | None:
    if not any(
        [
            args.brand_name,
            args.category,
            args.target_demo,
            args.vibe,
            args.energy_min is not None,
            args.energy_max is not None,
            args.tempo_min is not None,
            args.tempo_max is not None,
            args.genres,
            args.moods,
        ]
    ):
        return None

    video_stem = Path(args.video).stem
    return BrandProfile(
        name=args.brand_name or video_stem,
        product_category=args.category,
        target_demo=args.target_demo,
        vibe=args.vibe,
        desired_energy_min=args.energy_min,
        desired_energy_max=args.energy_max,
        desired_tempo_min=args.tempo_min,
        desired_tempo_max=args.tempo_max,
        desired_genres=_split_csv(args.genres),
        desired_moods=_split_csv(args.moods),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    video_path = Path(args.video)
    if not video_path.exists():
        parser.error(f"Video not found: {video_path}")

    tracks = []
    if args.tracks_dir:
        from .catalogs import LocalTrackCatalog

        tracks_dir = Path(args.tracks_dir)
        if not tracks_dir.exists():
            parser.error(f"Tracks directory not found: {tracks_dir}")
        tracks = LocalTrackCatalog(tracks_dir).list_tracks(limit=args.variant_limit)
        if not tracks:
            parser.error(f"No supported audio files found in: {tracks_dir}")

    transcript_hint_text = _read_optional_text(args.transcript_hint_file)
    from .service import SonicSegmentsService

    service = SonicSegmentsService(voice_backend_id=args.voice_backend)
    brand_profile = _build_brand_profile(args)

    analysis, variants, report_path = service.create_audit_bundle(
        video_path=video_path,
        output_dir=args.output_dir,
        brand_profile=brand_profile,
        tracks=tracks,
        preserve_voiceover=args.preserve_voiceover,
        transcript_hint_text=transcript_hint_text,
    )

    print(f"Sonic audit written to: {report_path}")
    print(f"Tempo: {analysis.features.get('tempo', 0.0):.0f} BPM")
    print(f"Energy: {analysis.features.get('energy', 0.0):.2f}")
    print(f"Inferred genres: {', '.join(analysis.inferred_genres) or 'unknown'}")

    if analysis.voiceover:
        print(
            "Voiceover coverage: "
            f"{analysis.voiceover.speech_coverage_seconds:.1f}s "
            f"({analysis.voiceover.speech_coverage_ratio * 100:.0f}%)"
        )

    if variants:
        print("Rendered variants:")
        for variant in variants:
            print(f"- {variant.output_path}")
    else:
        print("Rendered variants: none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
