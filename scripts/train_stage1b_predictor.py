"""Train Stage 1B H16 forward-structure predictor from frozen Stage 1A latents."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from msb_repr.stage1a.model import get_device
from msb_repr.stage1b.dataset import Stage1BForwardDataset, collate_fn_stage1b
from msb_repr.stage1b.model import Stage1BForwardPredictor
from msb_repr.stage1b.trainer import Stage1BTrainer, Stage1BTrainerConfig


def _set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def _label_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(label): int(count) for label, count in zip(unique, counts, strict=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-latent-path", type=Path, required=True)
    parser.add_argument("--train-label-path", type=Path, required=True)
    parser.add_argument("--val-latent-path", type=Path, required=True)
    parser.add_argument("--val-label-path", type=Path, required=True)
    parser.add_argument("--test-latent-path", type=Path, default=None)
    parser.add_argument("--test-label-path", type=Path, default=None)
    parser.add_argument("--feature-keys", nargs="+", default=["z_fused"])
    parser.add_argument("--target-column", type=str, default="h16_future_break_direction")
    parser.add_argument("--target-columns", nargs="*", default=None)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--break-loss-weight", type=float, default=1.0)
    parser.add_argument("--direction-loss-weight", type=float, default=1.0)
    parser.add_argument("--break-threshold", type=float, default=0.5)
    parser.add_argument("--primary-horizon-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()

    generator = _set_seed(args.seed)
    train_dataset = Stage1BForwardDataset.from_paths(
        latent_path=args.train_latent_path,
        label_path=args.train_label_path,
        feature_keys=args.feature_keys,
        target_column=args.target_column,
        target_columns=args.target_columns,
    )
    val_dataset = Stage1BForwardDataset.from_paths(
        latent_path=args.val_latent_path,
        label_path=args.val_label_path,
        feature_keys=args.feature_keys,
        target_column=args.target_column,
        target_columns=args.target_columns,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Stage 1B train/val dataset is empty")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn_stage1b,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_stage1b,
    )

    model = Stage1BForwardPredictor(
        input_dim=train_dataset.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_horizons=len(train_dataset.target_columns),
    )
    device = get_device()
    model = model.to(device)
    logger.info("Stage 1B training device: {}", device)
    logger.info(
        "Stage 1B dataset | train={} val={} input_dim={} features={}",
        len(train_dataset),
        len(val_dataset),
        train_dataset.input_dim,
        args.feature_keys,
    )

    trainer = Stage1BTrainer(
        model=model,
        device=device,
        cfg=Stage1BTrainerConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            break_loss_weight=args.break_loss_weight,
            direction_loss_weight=args.direction_loss_weight,
            break_threshold=args.break_threshold,
            primary_horizon_index=args.primary_horizon_index,
            checkpoint_dir=args.checkpoint_dir,
        ),
    )
    trainer.save_config()
    history = trainer.train(train_loader, val_loader)
    best_checkpoint = torch.load(
        args.checkpoint_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    val_metrics = trainer.evaluate(val_loader)

    test_metrics = None
    test_samples = None
    if args.test_latent_path is not None and args.test_label_path is not None:
        test_dataset = Stage1BForwardDataset.from_paths(
            latent_path=args.test_latent_path,
            label_path=args.test_label_path,
            feature_keys=args.feature_keys,
            target_column=args.target_column,
            target_columns=args.target_columns,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_stage1b,
        )
        test_metrics = trainer.evaluate(test_loader)
        test_samples = len(test_dataset)

    metadata = {
        "target_column": args.target_column,
        "target_columns": train_dataset.target_columns,
        "feature_keys": args.feature_keys,
        "input_dim": train_dataset.input_dim,
        "num_horizons": len(train_dataset.target_columns),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "seed": args.seed,
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": test_samples,
        "train_latent_path": str(args.train_latent_path),
        "train_label_path": str(args.train_label_path),
        "val_latent_path": str(args.val_latent_path),
        "val_label_path": str(args.val_label_path),
        "test_latent_path": None if args.test_latent_path is None else str(args.test_latent_path),
        "test_label_path": None if args.test_label_path is None else str(args.test_label_path),
        "train_break_target_counts": _label_counts(train_dataset.break_targets),
        "train_recon_target_counts": _label_counts(train_dataset.recon_targets),
        "val_break_target_counts": _label_counts(val_dataset.break_targets),
        "val_recon_target_counts": _label_counts(val_dataset.recon_targets),
        "primary_horizon_index": args.primary_horizon_index,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_val_recon_macro_f1": max(history["val_recon_macro_f1"]),
        "best_val_break_macro_f1": max(history["val_break_macro_f1"]),
        "best_val_direction_macro_f1": max(history["val_direction_macro_f1"]),
        "final_val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(
        "Stage 1B training finished -> {} | best_val_recon_f1={:.4f}",
        args.checkpoint_dir,
        metadata["best_val_recon_macro_f1"],
    )


if __name__ == "__main__":
    main()
