"""Local (offline) intelligence layers for Sonic Segments.

- clap_model: zero-shot audio-text understanding (LAION CLAP)
- mood: ad/track mood profiling (CLAP + DSP fusion)
- demographics: curated demographic -> music-direction knowledge base
- matching: track-vs-direction scoring and ranking
- quality: ingest source quality probing
"""

from .demographics import DemographicsKB, MusicDirection
from .mood import MOODS, MoodProfile, detect_mood
from .matching import rank_tracks, score_track
from .quality import SourceQuality, probe_source_quality

__all__ = [
    "DemographicsKB",
    "MusicDirection",
    "MOODS",
    "MoodProfile",
    "detect_mood",
    "rank_tracks",
    "score_track",
    "SourceQuality",
    "probe_source_quality",
]
