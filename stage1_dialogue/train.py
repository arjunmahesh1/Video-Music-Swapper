"""Train the Stage 1 dialogue/music/effects separator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .dataset import Stage1StemDataset, load_manifest
from .model import Stage1DialogueSeparator, Stage1SeparatorConfig, stage1_separator_loss


def _select_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_loader(
    manifest_path: Path,
    config: Stage1SeparatorConfig,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    entries = load_manifest(manifest_path)
    dataset = Stage1StemDataset(
        manifest_entries=entries,
        sample_rate=config.sample_rate,
        chunk_seconds=config.chunk_seconds,
        random_crop=shuffle,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def _save_checkpoint(
    output_dir: Path,
    name: str,
    model: Stage1DialogueSeparator,
    optimizer: torch.optim.Optimizer,
    config: Stage1SeparatorConfig,
    epoch: int,
    train_loss: float,
    val_loss: float | None,
) -> Path:
    payload = {
        "epoch": epoch,
        "config": config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    output_path = output_dir / name
    torch.save(payload, output_path)
    return output_path


def train_epoch(
    model: Stage1DialogueSeparator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_weight = 0

    for batch in loader:
        mixture = batch["mixture"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and device.type == "cuda"
        with autocast(enabled=use_amp):
            prediction = model(mixture)["stems"]
            loss = stage1_separator_loss(prediction, targets)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = mixture.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_weight += batch_size

    return total_loss / max(1, total_weight)


@torch.inference_mode()
def evaluate(
    model: Stage1DialogueSeparator,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_weight = 0

    for batch in loader:
        mixture = batch["mixture"].to(device)
        targets = batch["targets"].to(device)
        prediction = model(mixture)["stems"]
        loss = stage1_separator_loss(prediction, targets)
        batch_size = mixture.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_weight += batch_size

    return total_loss / max(1, total_weight)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 1 dialogue separator.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--chunk-seconds", type=float, default=6.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--resume")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Stage1SeparatorConfig(
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
    )
    model = Stage1DialogueSeparator(config)
    device = _select_device(args.device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scaler = GradScaler(enabled=device.type == "cuda")

    start_epoch = 1
    best_val_loss = None
    if args.resume:
        resume_path = Path(args.resume)
        payload = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_val_loss = payload.get("val_loss")
        config = Stage1SeparatorConfig.from_dict(payload["config"])
        model.config = config

    train_loader = _build_loader(
        manifest_path=Path(args.train_manifest),
        config=config,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if args.val_manifest:
        val_loader = _build_loader(
            manifest_path=Path(args.val_manifest),
            config=config,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        val_loss = evaluate(model, val_loader, device) if val_loader is not None else None

        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")

        latest_path = _save_checkpoint(
            output_dir=output_dir,
            name="latest.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
        )
        print(f"Epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss}")
        print(f"Saved latest checkpoint to {latest_path}")

        should_write_best = val_loss is None or best_val_loss is None or val_loss < best_val_loss
        if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
            best_val_loss = val_loss
        if should_write_best:
            best_path = _save_checkpoint(
                output_dir=output_dir,
                name="best.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )
            print(f"New best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
