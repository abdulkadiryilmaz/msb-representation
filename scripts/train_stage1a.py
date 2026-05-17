"""Train the Stage 1A Stage 1A baseline with CE or CE+SupCon."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from msb_repr.data.normalizer import Normalizer
from msb_repr.stage1a.analysis import build_domain_summary, build_structural_training_targets
from msb_repr.stage1a.config import Stage1ADatasetSpec
from msb_repr.stage1a.dataset import (
    Stage1ADualWindowDataset,
    Stage1ASymbolBalancedBatchSampler,
    collate_fn_stage1a,
)
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


def _set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def _attach_structural_targets(
    dataset: Stage1ADualWindowDataset,
    spec: Stage1ADatasetSpec,
) -> None:
    domain_summary = build_domain_summary(dataset, spec)
    targets = build_structural_training_targets(
        labels=dataset.labels,
        domain_summary=domain_summary,
        min_recent_break_bars=spec.labels.min_recent_break_bars,
    )
    dataset.factor_targets = targets["factor_targets"]
    dataset.pressure_targets = targets["pressure_targets"]
    dataset.maturity_targets = targets["maturity_targets"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Stage 1A baseline")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epoch-checkpoints", nargs="*", type=int, default=[])
    parser.add_argument("--z-short", type=int, default=64)
    parser.add_argument("--z-long", type=int, default=32)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--long-projection-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-supcon", action="store_true")
    parser.add_argument("--ce-warmup-weight", type=float, default=None)
    parser.add_argument("--ce-warmup-epochs", type=int, default=0)
    parser.add_argument("--supcon-weight", type=float, default=0.05)
    parser.add_argument("--supcon-frontload-weight", type=float, default=None)
    parser.add_argument("--supcon-frontload-epochs", type=int, default=0)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument(
        "--supcon-embedding-key",
        choices=["z_proj", "z_long_proj"],
        default="z_proj",
    )
    parser.add_argument(
        "--supcon-positive-mode",
        choices=["label", "factor", "label_diff_symbol", "label_diff_symbol_neutral_same_symbol"],
        default="label",
    )
    parser.add_argument("--pressure-loss-weight", type=float, default=0.0)
    parser.add_argument("--maturity-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--long-aux-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary CE loss weight for a classifier attached directly to z_long.",
    )
    parser.add_argument(
        "--pressure-head-input",
        choices=["z_fused", "z_short"],
        default="z_fused",
        help="Latent view used by the pressure auxiliary head.",
    )
    parser.add_argument("--supcon-balance-symbols", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    args = parser.parse_args()
    data_loader_generator = _set_seed(args.seed)
    long_projection_dim = args.long_projection_dim
    if args.supcon_embedding_key == "z_long_proj" and long_projection_dim is None:
        long_projection_dim = args.z_long

    spec = Stage1ADatasetSpec.load(args.dataset_root / "spec.json")
    train_dataset = Stage1ADualWindowDataset.from_root(args.dataset_root, split="train")
    val_dataset = Stage1ADualWindowDataset.from_root(args.dataset_root, split="val")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Train/val split is empty. Rebuild dataset with --train-end and --val-end.")

    use_structural_targets = (
        args.supcon_positive_mode == "factor"
        or args.pressure_loss_weight > 0.0
        or args.maturity_loss_weight > 0.0
    )
    if use_structural_targets:
        _attach_structural_targets(train_dataset, spec)
        _attach_structural_targets(val_dataset, spec)

    short_norm = Normalizer().fit(train_dataset.short_windows)
    long_norm = Normalizer().fit(train_dataset.long_windows)
    _apply_normalizers(train_dataset, short_norm, long_norm)
    _apply_normalizers(val_dataset, short_norm, long_norm)

    if args.use_supcon and args.supcon_balance_symbols:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=Stage1ASymbolBalancedBatchSampler(
                symbols=train_dataset.symbols,
                batch_size=args.batch_size,
                shuffle=True,
                seed=args.seed,
            ),
            collate_fn=collate_fn_stage1a,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=data_loader_generator,
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
        num_pressure_classes=5 if args.pressure_loss_weight > 0.0 else None,
        num_maturity_classes=4 if args.maturity_loss_weight > 0.0 else None,
        num_long_aux_classes=3 if args.long_aux_loss_weight > 0.0 else None,
        pressure_head_input=args.pressure_head_input,
        projection_dim=args.projection_dim,
        long_projection_dim=long_projection_dim,
        dropout=args.dropout,
    )
    device = get_device()
    model = model.to(device)
    logger.info("Stage 1A training device: {}", device)

    if args.use_supcon and args.supcon_positive_mode == "factor":
        default_checkpoint_name = "ce_supcon_long_factor_v1"
    elif args.use_supcon and args.supcon_embedding_key == "z_long_proj":
        default_checkpoint_name = "ce_supcon_long_v1"
    elif args.use_supcon:
        default_checkpoint_name = "ce_supcon_v1"
    else:
        default_checkpoint_name = "ce_only"
    checkpoint_dir = args.checkpoint_dir or (args.dataset_root / "checkpoints" / default_checkpoint_name)
    trainer = Stage1ATrainer(
        model=model,
        device=device,
        cfg=Stage1ATrainerConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            use_supcon=args.use_supcon,
            ce_warmup_weight=args.ce_warmup_weight,
            ce_warmup_epochs=args.ce_warmup_epochs,
            supcon_weight=args.supcon_weight,
            supcon_frontload_weight=args.supcon_frontload_weight,
            supcon_frontload_epochs=args.supcon_frontload_epochs,
            supcon_temperature=args.supcon_temperature,
            supcon_embedding_key=args.supcon_embedding_key,
            supcon_positive_mode=args.supcon_positive_mode,
            pressure_loss_weight=args.pressure_loss_weight,
            maturity_loss_weight=args.maturity_loss_weight,
            long_aux_loss_weight=args.long_aux_loss_weight,
            seed=args.seed,
            epoch_checkpoints=tuple(args.epoch_checkpoints),
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
        "long_projection_dim": long_projection_dim,
        "num_pressure_classes": 5 if args.pressure_loss_weight > 0.0 else None,
        "num_maturity_classes": 4 if args.maturity_loss_weight > 0.0 else None,
        "num_long_aux_classes": 3 if args.long_aux_loss_weight > 0.0 else None,
        "pressure_head_input": args.pressure_head_input,
        "seed": args.seed,
        "device": str(device),
        "use_supcon": args.use_supcon,
        "ce_warmup_weight": args.ce_warmup_weight,
        "ce_warmup_epochs": args.ce_warmup_epochs,
        "supcon_weight": args.supcon_weight,
        "supcon_frontload_weight": args.supcon_frontload_weight,
        "supcon_frontload_epochs": args.supcon_frontload_epochs,
        "supcon_temperature": args.supcon_temperature,
        "supcon_embedding_key": args.supcon_embedding_key,
        "supcon_positive_mode": args.supcon_positive_mode,
        "pressure_loss_weight": args.pressure_loss_weight,
        "maturity_loss_weight": args.maturity_loss_weight,
        "long_aux_loss_weight": args.long_aux_loss_weight,
        "supcon_balance_symbols": args.supcon_balance_symbols,
        "epoch_checkpoints": args.epoch_checkpoints,
        "train_label_counts": _label_counts(train_dataset.labels),
        "val_label_counts": _label_counts(val_dataset.labels),
        "best_val_loss": min(history["val_loss"]),
        "best_val_macro_f1": max(history["val_macro_f1"]),
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(
        "Stage 1A training finished -> {} | train={} val={} best_f1={:.4f} supcon={}",
        checkpoint_dir,
        len(train_dataset),
        len(val_dataset),
        max(history["val_macro_f1"]),
        args.use_supcon,
    )


if __name__ == "__main__":
    main()
