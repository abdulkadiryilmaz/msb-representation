"""Train the Stage 1A CE-only baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger
from torch.utils.data import DataLoader

from msb_repr.data.normalizer import Normalizer
from msb_repr.stage1a.dataset import Stage1ADualWindowDataset, collate_fn_stage1a
from msb_repr.stage1a.model import Stage1AModel, get_device
from msb_repr.stage1a.trainer import Stage1ATrainer, Stage1ATrainerConfig


def _apply_normalizers(
    dataset: Stage1ADualWindowDataset,
    short_norm: Normalizer,
    long_norm: Normalizer,
) -> None:
    dataset.short_windows = short_norm.transform(dataset.short_windows).astype("float32")
    dataset.long_windows = long_norm.transform(dataset.long_windows).astype("float32")


def _label_counts(labels) -> dict[str, int]:
    counts = {}
    for label in range(3):
        counts[str(label)] = int((labels == label).sum())
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Stage 1A CE-only baseline")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--z-short", type=int, default=64)
    parser.add_argument("--z-long", type=int, default=32)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    args = parser.parse_args()

    train_dataset = Stage1ADualWindowDataset.from_root(args.dataset_root, split="train")
    val_dataset = Stage1ADualWindowDataset.from_root(args.dataset_root, split="val")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Train/val split is empty. Rebuild dataset with --train-end and --val-end.")

    short_norm = Normalizer().fit(train_dataset.short_windows)
    long_norm = Normalizer().fit(train_dataset.long_windows)
    _apply_normalizers(train_dataset, short_norm, long_norm)
    _apply_normalizers(val_dataset, short_norm, long_norm)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_stage1a,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_stage1a,
    )

    short_batch, long_batch, _, _ = next(iter(train_loader))
    model = Stage1AModel(
        short_input_channels=short_batch.shape[1],
        long_input_channels=long_batch.shape[1],
        z_short=args.z_short,
        z_long=args.z_long,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    )
    device = get_device()
    model = model.to(device)

    checkpoint_dir = args.checkpoint_dir or (args.dataset_root / "checkpoints" / "ce_only")
    trainer = Stage1ATrainer(
        model=model,
        device=device,
        cfg=Stage1ATrainerConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            checkpoint_dir=checkpoint_dir,
        ),
    )
    trainer.save_config()
    history = trainer.train(train_loader, val_loader)

    short_norm.save(checkpoint_dir / "short_norm_params.json")
    long_norm.save(checkpoint_dir / "long_norm_params.json")
    metadata = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "short_input_channels": int(short_batch.shape[1]),
        "long_input_channels": int(long_batch.shape[1]),
        "z_short": args.z_short,
        "z_long": args.z_long,
        "projection_dim": args.projection_dim,
        "train_label_counts": _label_counts(train_dataset.labels),
        "val_label_counts": _label_counts(val_dataset.labels),
        "best_val_loss": min(history["val_loss"]),
        "best_val_macro_f1": max(history["val_macro_f1"]),
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(
        "Stage 1A CE-only training finished -> {} | train={} val={} best_f1={:.4f}",
        checkpoint_dir,
        len(train_dataset),
        len(val_dataset),
        max(history["val_macro_f1"]),
    )


if __name__ == "__main__":
    main()
