"""Build a JSONL manifest for Stage 1 separator training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_stem_folders(
    root: Path,
    dialogue_name: str,
    music_name: str,
    effects_name: str,
    mixture_name: str | None,
):
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        dialogue = directory / dialogue_name
        music = directory / music_name
        effects = directory / effects_name
        if not (dialogue.exists() and music.exists() and effects.exists()):
            continue

        record = {
            "dialogue": str(dialogue.resolve()),
            "music": str(music.resolve()),
            "effects": str(effects.resolve()),
        }
        if mixture_name:
            mixture = directory / mixture_name
            if mixture.exists():
                record["mixture"] = str(mixture.resolve())
        yield record


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 1 training manifest.")
    parser.add_argument("--root", required=True, help="Root directory containing stem folders")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path")
    parser.add_argument("--dialogue-name", default="dialogue.wav")
    parser.add_argument("--music-name", default="music.wav")
    parser.add_argument("--effects-name", default="effects.wav")
    parser.add_argument("--mixture-name", default="mixture.wav")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    records = list(
        iter_stem_folders(
            root=root,
            dialogue_name=args.dialogue_name,
            music_name=args.music_name,
            effects_name=args.effects_name,
            mixture_name=args.mixture_name,
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(f"Wrote {len(records)} manifest entries to {output}")


if __name__ == "__main__":
    main()
