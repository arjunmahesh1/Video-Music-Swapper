"""Backend selection wrapper for voiceover preservation."""

from __future__ import annotations

import os
from dataclasses import dataclass

from stage1_dialogue_backend import cleanup_stage1_output, resolve_stage1_checkpoint_path, separate_and_remix_stage1, stage1_backend_available
from voice_separator import cleanup_demucs_output, separate_and_remix


@dataclass
class VoiceBackendOption:
    backend_id: str
    label: str
    description: str
    available: bool
    availability_reason: str | None = None


def get_voice_backend_options() -> list[VoiceBackendOption]:
    options = [
        VoiceBackendOption(
            backend_id="heuristic",
            label="Heuristic baseline",
            description="Current Demucs + cleanup + ducking pipeline.",
            available=True,
        )
    ]

    checkpoint_path = resolve_stage1_checkpoint_path()
    if stage1_backend_available(checkpoint_path):
        options.append(
            VoiceBackendOption(
                backend_id="stage1_dialogue",
                label="Stage 1 dialogue model",
                description="Trainable dialogue/music/effects separator checkpoint.",
                available=True,
            )
        )
    else:
        options.append(
            VoiceBackendOption(
                backend_id="stage1_dialogue",
                label="Stage 1 dialogue model",
                description="Trainable dialogue/music/effects separator checkpoint.",
                available=False,
                availability_reason=(
                    f"No checkpoint found at {checkpoint_path}. "
                    "Train Stage 1 first, then set STAGE1_VOICEOVER_CHECKPOINT."
                ),
            )
        )
    return options


def get_default_voice_backend_id() -> str:
    preferred = os.getenv("VOICEOVER_BACKEND", "heuristic").strip() or "heuristic"
    available = {option.backend_id for option in get_voice_backend_options() if option.available}
    return preferred if preferred in available else "heuristic"


def separate_and_remix_with_backend(
    video_audio_path,
    new_music_path,
    output_path,
    backend_id: str | None = None,
    vocals_volume: float = 1.0,
    music_volume: float = 0.7,
    transcript_hint_text: str | None = None,
    speech_segments_override: list | None = None,
):
    backend_id = backend_id or get_default_voice_backend_id()
    if backend_id == "stage1_dialogue":
        return separate_and_remix_stage1(
            video_audio_path=video_audio_path,
            new_music_path=new_music_path,
            output_path=output_path,
            vocals_volume=vocals_volume,
            music_volume=music_volume,
            transcript_hint_text=transcript_hint_text,
        )

    return separate_and_remix(
        video_audio_path=video_audio_path,
        new_music_path=new_music_path,
        output_path=output_path,
        vocals_volume=vocals_volume,
        music_volume=music_volume,
        transcript_hint_text=transcript_hint_text,
        speech_segments_override=speech_segments_override,
    )


def cleanup_voice_backend_outputs() -> None:
    cleanup_demucs_output()
    cleanup_stage1_output()
