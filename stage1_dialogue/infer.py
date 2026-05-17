"""CLI wrapper for Stage 1 separator inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from .inference import run_stage1_separation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 dialogue separation.")
    parser.add_argument("--checkpoint", help="Checkpoint path; defaults to models/stage1_dialogue/best.pt")
    parser.add_argument("--input", required=True, help="Input audio or video path")
    parser.add_argument("--output-dir", required=True, help="Output directory for stems")
    parser.add_argument("--device", help="torch device, e.g. cpu or cuda")
    parser.add_argument("--chunk-seconds", type=float, help="Override chunk length for inference")
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()

    outputs = run_stage1_separation(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        device=args.device,
        chunk_seconds=args.chunk_seconds,
        overlap=args.overlap,
    )
    for stem_name, stem_path in outputs.items():
        print(f"{stem_name}: {stem_path}")


if __name__ == "__main__":
    main()
