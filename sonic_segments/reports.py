"""Simple report generation for the Sonic Segments MVP."""

from __future__ import annotations

from pathlib import Path

from .models import AdAnalysis, VariantRender


def build_sonic_audit_markdown(
    analysis: AdAnalysis,
    variants: list[VariantRender] | None = None,
) -> str:
    """Render a markdown summary that can be shared with a client."""
    lines: list[str] = []
    brand_name = analysis.brand_profile.name if analysis.brand_profile else analysis.video_path.stem

    lines.append(f"# SONIC AUDIT: {brand_name}")
    lines.append("")
    lines.append(f"- Video: `{analysis.video_path.name}`")
    lines.append(f"- Tempo: {analysis.features.get('tempo', 0.0):.0f} BPM")
    lines.append(f"- Energy: {analysis.features.get('energy', 0.0):.2f}")
    lines.append(f"- Inferred genres: {', '.join(analysis.inferred_genres) or 'unknown'}")

    if analysis.voiceover:
        lines.append(
            "- Voiceover coverage: "
            f"{analysis.voiceover.speech_coverage_seconds:.1f}s "
            f"({analysis.voiceover.speech_coverage_ratio * 100:.0f}% of ad)"
        )

    if analysis.brand_profile:
        lines.append("")
        lines.append("## Brand Brief")
        lines.append(f"- Brand: {analysis.brand_profile.name}")
        if analysis.brand_profile.product_category:
            lines.append(f"- Category: {analysis.brand_profile.product_category}")
        if analysis.brand_profile.target_demo:
            lines.append(f"- Target demo: {analysis.brand_profile.target_demo}")
        if analysis.brand_profile.vibe:
            lines.append(f"- Vibe: {analysis.brand_profile.vibe}")

    if analysis.mismatches:
        lines.append("")
        lines.append("## Mismatch Signals")
        for mismatch in analysis.mismatches:
            lines.append(f"- {mismatch}")

    if analysis.recommendations:
        lines.append("")
        lines.append("## Recommendations")
        for recommendation in analysis.recommendations:
            lines.append(f"- {recommendation}")

    if variants:
        lines.append("")
        lines.append("## Rendered Variants")
        for idx, variant in enumerate(variants, start=1):
            artist = f" - {variant.track.artist}" if variant.track.artist else ""
            lines.append(f"- Variant {idx}: `{variant.output_path.name}` using `{variant.track.name}{artist}`")
            for note in variant.predicted_notes:
                lines.append(f"  - {note}")

    return "\n".join(lines) + "\n"


def write_sonic_audit_report(
    output_path: str | Path,
    analysis: AdAnalysis,
    variants: list[VariantRender] | None = None,
) -> Path:
    """Write the markdown report to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_sonic_audit_markdown(analysis, variants=variants), encoding="utf-8")
    return output_path
