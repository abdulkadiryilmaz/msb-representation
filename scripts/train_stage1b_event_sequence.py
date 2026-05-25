"""Train Stage 1B event-sequence predictor from frozen Stage 1A latents."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from msb_repr.stage1a.model import get_device
from msb_repr.stage1b.dataset import (
    DIRECTION_LABEL_TO_ID,
    EVENT_TYPE_ID_TO_LABEL,
    EVENT_TYPE_LABEL_TO_ID,
    Stage1BEventSequenceDataset,
    collate_fn_stage1b_event,
)
from msb_repr.stage1b.model import Stage1BEventSequencePredictor


DIRECTION_ID_TO_LABEL = {idx: label for label, idx in DIRECTION_LABEL_TO_ID.items()}
DIRECTION_ID_TO_LABEL[-1] = "none"


def _set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> tuple[float, dict[str, float]]:
    per_label: dict[str, float] = {}
    scores = []
    for label in labels:
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        per_label[f"class_{label}_f1"] = float(f1)
        scores.append(f1)
    return float(np.mean(scores)), per_label


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> list[list[int]]:
    return [
        [int(np.sum((y_true == true_label) & (y_pred == pred_label))) for pred_label in labels]
        for true_label in labels
    ]


def _balanced_weights(values: np.ndarray, labels: list[int]) -> list[float]:
    counts = np.array([max(1, int(np.sum(values == label))) for label in labels], dtype=np.float64)
    weights = counts.sum() / (len(labels) * counts)
    return [float(weight) for weight in weights]


def _label_counts(values: np.ndarray, label_map: dict[int, str]) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {label_map[int(label)]: int(count) for label, count in zip(unique, counts, strict=True)}


def _run_epoch(
    model: Stage1BEventSequencePredictor,
    loader: DataLoader,
    device: torch.device,
    event_type_criterion: nn.CrossEntropyLoss,
    direction_criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer | None,
    event_type_loss_weight: float,
    direction_loss_weight: float,
    include_confusion: bool = False,
) -> dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_type_loss = 0.0
    total_direction_loss = 0.0
    total_samples = 0
    total_direction_samples = 0
    type_true = []
    type_pred = []
    direction_true = []
    direction_pred = []

    with torch.set_grad_enabled(train):
        for features, event_type_targets, event_direction_targets, _metas in loader:
            features = features.to(device)
            event_type_targets = event_type_targets.to(device)
            event_direction_targets = event_direction_targets.to(device)
            outputs = model(features)
            type_logits = outputs["event_type_logits"]
            direction_logits = outputs["event_direction_logits"]
            type_loss = event_type_criterion(type_logits, event_type_targets)
            direction_mask = event_direction_targets >= 0
            if bool(direction_mask.any()):
                direction_loss = direction_criterion(
                    direction_logits[direction_mask],
                    event_direction_targets[direction_mask],
                )
            else:
                direction_loss = type_loss.new_zeros(())
            loss = event_type_loss_weight * type_loss + direction_loss_weight * direction_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = int(features.size(0))
            direction_count = int(direction_mask.sum().item())
            total_samples += batch_size
            total_direction_samples += direction_count
            total_loss += float(loss.item()) * batch_size
            total_type_loss += float(type_loss.item()) * batch_size
            total_direction_loss += float(direction_loss.item()) * direction_count

            batch_type_pred = type_logits.argmax(dim=1)
            batch_direction_pred = direction_logits.argmax(dim=1)
            type_true.append(event_type_targets.detach().cpu().numpy())
            type_pred.append(batch_type_pred.detach().cpu().numpy())
            if direction_count:
                direction_true.append(event_direction_targets[direction_mask].detach().cpu().numpy())
                direction_pred.append(batch_direction_pred[direction_mask].detach().cpu().numpy())

    y_type_true = np.concatenate(type_true)
    y_type_pred = np.concatenate(type_pred)
    event_type_macro_f1, event_type_per_class = _macro_f1(y_type_true, y_type_pred, labels=[0, 1, 2, 3])
    if direction_true:
        y_direction_true = np.concatenate(direction_true)
        y_direction_pred = np.concatenate(direction_pred)
        direction_macro_f1, direction_per_class = _macro_f1(y_direction_true, y_direction_pred, labels=[0, 1])
        direction_accuracy = float(np.mean(y_direction_true == y_direction_pred))
    else:
        y_direction_true = np.array([], dtype=np.int64)
        y_direction_pred = np.array([], dtype=np.int64)
        direction_macro_f1 = 0.0
        direction_accuracy = 0.0
        direction_per_class = {"class_0_f1": 0.0, "class_1_f1": 0.0}

    metrics: dict[str, Any] = {
        "loss": total_loss / max(total_samples, 1),
        "event_type_loss": total_type_loss / max(total_samples, 1),
        "event_direction_loss": total_direction_loss / max(total_direction_samples, 1),
        "event_type_macro_f1": event_type_macro_f1,
        "event_type_per_class_f1": event_type_per_class,
        "event_type_accuracy": float(np.mean(y_type_true == y_type_pred)),
        "event_direction_macro_f1": direction_macro_f1,
        "event_direction_per_class_f1": direction_per_class,
        "event_direction_accuracy": direction_accuracy,
        "direction_samples": int(total_direction_samples),
    }
    if include_confusion:
        metrics["event_type_confusion_matrix"] = {
            "labels": [EVENT_TYPE_ID_TO_LABEL[idx] for idx in [0, 1, 2, 3]],
            "matrix": _confusion_matrix(y_type_true, y_type_pred, labels=[0, 1, 2, 3]),
        }
        metrics["event_direction_confusion_matrix"] = {
            "labels": [DIRECTION_ID_TO_LABEL[idx] for idx in [0, 1]],
            "matrix": _confusion_matrix(y_direction_true, y_direction_pred, labels=[0, 1])
            if len(y_direction_true)
            else [[0, 0], [0, 0]],
        }
    return metrics


def _collect_predictions(
    model: Stage1BEventSequencePredictor,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    with torch.no_grad():
        for features, event_type_targets, event_direction_targets, metas in loader:
            features = features.to(device)
            outputs = model(features)
            type_probs = torch.softmax(outputs["event_type_logits"], dim=1).cpu().numpy()
            direction_probs = torch.softmax(outputs["event_direction_logits"], dim=1).cpu().numpy()
            type_pred = type_probs.argmax(axis=1)
            direction_pred = direction_probs.argmax(axis=1)
            for i, meta in enumerate(metas):
                pred_type_id = int(type_pred[i])
                true_type_id = int(event_type_targets[i].item())
                pred_direction_id = int(direction_pred[i])
                true_direction_id = int(event_direction_targets[i].item())
                row = {
                    "index": int(meta.index),
                    "symbol": meta.symbol,
                    "timestamp": int(meta.timestamp),
                    "true_event_type": EVENT_TYPE_ID_TO_LABEL[true_type_id],
                    "pred_event_type": EVENT_TYPE_ID_TO_LABEL[pred_type_id],
                    "event_type_confidence": float(type_probs[i, pred_type_id]),
                    "true_event_direction": DIRECTION_ID_TO_LABEL[true_direction_id],
                    "pred_event_direction": DIRECTION_ID_TO_LABEL[pred_direction_id],
                    "event_direction_confidence": float(direction_probs[i, pred_direction_id]),
                    "is_event_type_correct": bool(true_type_id == pred_type_id),
                    "is_event_direction_correct": bool(
                        true_direction_id < 0 or true_direction_id == pred_direction_id
                    ),
                    "p_no_event": float(type_probs[i, EVENT_TYPE_LABEL_TO_ID["no_event"]]),
                    "p_fresh_break": float(type_probs[i, EVENT_TYPE_LABEL_TO_ID["fresh_break"]]),
                    "p_continuation": float(type_probs[i, EVENT_TYPE_LABEL_TO_ID["continuation"]]),
                    "p_reversal": float(type_probs[i, EVENT_TYPE_LABEL_TO_ID["reversal"]]),
                    "p_bullish": float(direction_probs[i, DIRECTION_LABEL_TO_ID["bullish"]]),
                    "p_bearish": float(direction_probs[i, DIRECTION_LABEL_TO_ID["bearish"]]),
                }
                rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-latent-path", type=Path, required=True)
    parser.add_argument("--train-label-path", type=Path, required=True)
    parser.add_argument("--val-latent-path", type=Path, required=True)
    parser.add_argument("--val-label-path", type=Path, required=True)
    parser.add_argument("--test-latent-path", type=Path, default=None)
    parser.add_argument("--test-label-path", type=Path, default=None)
    parser.add_argument("--feature-keys", nargs="+", default=["z_fused"])
    parser.add_argument("--context-feature-keys", nargs="*", default=[])
    parser.add_argument("--event-type-column", type=str, default="h16_event_type")
    parser.add_argument("--event-direction-column", type=str, default="h16_event_direction")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--event-type-loss-weight", type=float, default=1.0)
    parser.add_argument("--direction-loss-weight", type=float, default=1.0)
    parser.add_argument("--balanced-event-type-loss", action="store_true")
    parser.add_argument("--balanced-direction-loss", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()

    generator = _set_seed(args.seed)
    train_dataset = Stage1BEventSequenceDataset.from_paths(
        latent_path=args.train_latent_path,
        label_path=args.train_label_path,
        feature_keys=args.feature_keys,
        context_feature_keys=args.context_feature_keys,
        event_type_column=args.event_type_column,
        event_direction_column=args.event_direction_column,
    )
    val_dataset = Stage1BEventSequenceDataset.from_paths(
        latent_path=args.val_latent_path,
        label_path=args.val_label_path,
        feature_keys=args.feature_keys,
        context_feature_keys=args.context_feature_keys,
        event_type_column=args.event_type_column,
        event_direction_column=args.event_direction_column,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Stage 1B event train/val dataset is empty")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn_stage1b_event,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_stage1b_event,
    )

    device = get_device()
    model = Stage1BEventSequencePredictor(
        input_dim=train_dataset.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_event_types=len(EVENT_TYPE_LABEL_TO_ID),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    event_type_weights = (
        torch.tensor(
            _balanced_weights(train_dataset.event_type_targets, labels=[0, 1, 2, 3]),
            dtype=torch.float32,
            device=device,
        )
        if args.balanced_event_type_loss
        else None
    )
    direction_values = train_dataset.event_direction_targets[train_dataset.event_direction_targets >= 0]
    direction_weights = (
        torch.tensor(_balanced_weights(direction_values, labels=[0, 1]), dtype=torch.float32, device=device)
        if args.balanced_direction_loss
        else None
    )
    event_type_criterion = nn.CrossEntropyLoss(weight=event_type_weights)
    direction_criterion = nn.CrossEntropyLoss(weight=direction_weights)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_config = vars(args).copy()
    for key, value in list(train_config.items()):
        if isinstance(value, Path):
            train_config[key] = str(value)
    (args.checkpoint_dir / "train_config.json").write_text(json.dumps(train_config, indent=2))

    logger.info("Stage 1B event training device: {}", device)
    logger.info(
        "Stage 1B event dataset | train={} val={} input_dim={} features={} context={}",
        len(train_dataset),
        len(val_dataset),
        train_dataset.input_dim,
        args.feature_keys,
        args.context_feature_keys,
    )

    best_score = -float("inf")
    best_epoch = 0
    patience_counter = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_event_type_macro_f1": [],
        "train_event_direction_macro_f1": [],
        "val_loss": [],
        "val_event_type_macro_f1": [],
        "val_event_direction_macro_f1": [],
    }
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            event_type_criterion=event_type_criterion,
            direction_criterion=direction_criterion,
            optimizer=optimizer,
            event_type_loss_weight=args.event_type_loss_weight,
            direction_loss_weight=args.direction_loss_weight,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            event_type_criterion=event_type_criterion,
            direction_criterion=direction_criterion,
            optimizer=None,
            event_type_loss_weight=args.event_type_loss_weight,
            direction_loss_weight=args.direction_loss_weight,
        )
        for prefix, metrics in [("train", train_metrics), ("val", val_metrics)]:
            history[f"{prefix}_loss"].append(float(metrics["loss"]))
            history[f"{prefix}_event_type_macro_f1"].append(float(metrics["event_type_macro_f1"]))
            history[f"{prefix}_event_direction_macro_f1"].append(float(metrics["event_direction_macro_f1"]))

        logger.info(
            "Epoch {}/{} | train_loss={:.5f} val_loss={:.5f} "
            "val_event_type_f1={:.4f} val_event_dir_f1={:.4f}",
            epoch,
            args.epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["event_type_macro_f1"],
            val_metrics["event_direction_macro_f1"],
        )

        score = float(val_metrics["event_type_macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                },
                args.checkpoint_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch {}", epoch)
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": len(history["val_loss"]),
            "metrics": val_metrics,
        },
        args.checkpoint_dir / "final.pt",
    )
    (args.checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))

    best_checkpoint = torch.load(args.checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_val_metrics = _run_epoch(
        model=model,
        loader=val_loader,
        device=device,
        event_type_criterion=event_type_criterion,
        direction_criterion=direction_criterion,
        optimizer=None,
        event_type_loss_weight=args.event_type_loss_weight,
        direction_loss_weight=args.direction_loss_weight,
        include_confusion=True,
    )

    test_metrics = None
    test_samples = None
    if args.test_latent_path is not None and args.test_label_path is not None:
        test_dataset = Stage1BEventSequenceDataset.from_paths(
            latent_path=args.test_latent_path,
            label_path=args.test_label_path,
            feature_keys=args.feature_keys,
            context_feature_keys=args.context_feature_keys,
            event_type_column=args.event_type_column,
            event_direction_column=args.event_direction_column,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_stage1b_event,
        )
        test_metrics = _run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            event_type_criterion=event_type_criterion,
            direction_criterion=direction_criterion,
            optimizer=None,
            event_type_loss_weight=args.event_type_loss_weight,
            direction_loss_weight=args.direction_loss_weight,
            include_confusion=True,
        )
        test_samples = len(test_dataset)
        prediction_rows = _collect_predictions(model, test_loader, device)
        _write_csv(args.checkpoint_dir / "test_event_predictions.csv", prediction_rows)

    metadata = {
        "event_type_column": args.event_type_column,
        "event_direction_column": args.event_direction_column,
        "feature_keys": args.feature_keys,
        "context_feature_keys": args.context_feature_keys,
        "input_dim": train_dataset.input_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "seed": args.seed,
        "device": str(device),
        "balanced_event_type_loss": bool(args.balanced_event_type_loss),
        "balanced_direction_loss": bool(args.balanced_direction_loss),
        "event_type_loss_weight": args.event_type_loss_weight,
        "direction_loss_weight": args.direction_loss_weight,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": test_samples,
        "train_latent_path": str(args.train_latent_path),
        "train_label_path": str(args.train_label_path),
        "val_latent_path": str(args.val_latent_path),
        "val_label_path": str(args.val_label_path),
        "test_latent_path": None if args.test_latent_path is None else str(args.test_latent_path),
        "test_label_path": None if args.test_label_path is None else str(args.test_label_path),
        "train_event_type_counts": _label_counts(train_dataset.event_type_targets, EVENT_TYPE_ID_TO_LABEL),
        "val_event_type_counts": _label_counts(val_dataset.event_type_targets, EVENT_TYPE_ID_TO_LABEL),
        "best_epoch": int(best_epoch),
        "best_val_event_type_macro_f1": float(best_score),
        "best_val_event_direction_macro_f1": max(history["val_event_direction_macro_f1"]),
        "final_val_metrics": final_val_metrics,
        "test_metrics": test_metrics,
    }
    (args.checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(
        "Stage 1B event training finished -> {} | best_val_event_type_f1={:.4f}",
        args.checkpoint_dir,
        best_score,
    )


if __name__ == "__main__":
    main()
