"""Sonic Segments MVP service layer.

This package wraps the current repo's audio-analysis, voice-preservation,
and rendering code in a cleaner product-oriented API.
"""

from .models import AdAnalysis, BrandProfile, TrackCandidate, VariantRender, VoiceoverAnalysis
from .service import SonicSegmentsService

__all__ = [
    "AdAnalysis",
    "BrandProfile",
    "SonicSegmentsService",
    "TrackCandidate",
    "VariantRender",
    "VoiceoverAnalysis",
]
