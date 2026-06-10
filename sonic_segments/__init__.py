"""Sonic Segments MVP service layer."""

from .models import AdAnalysis, BrandProfile, TrackCandidate, VariantRender, VoiceoverAnalysis

__all__ = [
    "AdAnalysis",
    "BrandProfile",
    "SonicSegmentsService",
    "TrackCandidate",
    "VariantRender",
    "VoiceoverAnalysis",
]


def __getattr__(name: str):
    """Load heavy audio dependencies only when the service is requested."""
    if name == "SonicSegmentsService":
        from .service import SonicSegmentsService

        return SonicSegmentsService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
