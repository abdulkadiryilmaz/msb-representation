"""Train Stage 1B joint event-sequence predictor from frozen Stage 1A latents."""

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
    JOINT_EVENT_ID_TO_LABEL,
    JOINT_EVENT_LABEL_TO_ID,
    Stage1BJointEventSequenceDataset,
    collate_fn_stage1b_joint_event,
)
from msb_repr.stage1b.model import Stage1BJointEventSequencePredictor


def _set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def _event_type_from_joint(label: str) -> str:
    if label == "no_event":
        return "no_event"
    return label.rsplit("_", 1)[0]


def _direction_from_joint(label: str) -> str:
    if label == "no_event":
        return "none"
    return label.rsplit("_", 1)[1]


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


def _label_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {JOINT_EVENT_ID_TO_LABEL[int(label)]: int(count) for label, count in zip(unique, counts, strict=True)}


def _derived_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    type_names = ["no_event", "fresh_break", "continuation", "reversal"]
    dir_names = ["bullish", "bearish", "none"]
    type_to_id = {label: idx for idx, label in enumerate(type_names)}
    dir_to_id = {label: idx for idx, label in enumerate(dir_names)}

    true_labels = [JOINT_EVENT_ID_TO_LABEL[int(value)] for value in y_true]
    pred_labels = [JOINT_EVENT_ID_TO_LABEL[int(value)] for value in y_pred]
    type_true = np.array([type_to_id[_event_type_from_joint(label)] for label in true_labels], dtype=np.int64)
    type_pred = np.array([type_to_id[_event_type_from_joint(label)] for label in pred_labels], dtype=np.int64)
    direction_mask = np.array([_direction_from_joint(label) != "none" for label in true_labels], dtype=bool)
    direction_true = np.array(
        [dir_to_id[_direction_from_joint(label)] for label, keep in zip(true_labels, direction_mask, strict=True) if keep],
        dtype=np.int64,
    )
    direction_pred = np.array(
        [dir_to_id[_direction_from_joint(label)] for label, keep in zip(pred_labels, direction_mask, strict=True) if keep],
        dtype=np.int64,
    )
    event_type_f1, event_type_per_class = _macro_f1(type_true, type_pred, labels=[0, 1, 2, 3])
    direction_f1, direction_per_class = _macro_f1(direction_true, direction_pred, labels=[0, 1])
    return {
        "event_type_macro_f1": event_type_f1,
        "event_type_per_class_f1": event_type_per_class,
        "event_type_accuracy": float(np.mean(type_true == type_pred)),
        "event_direction_macro_f1": direction_f1,
        "event_direction_per_class_f1": direction_per_class,
        "event_direction_accuracy": float(np.mean(direction_true == direction_pred)),
        "event_type_confusion_matrix": {
            "labels": type_names,
            "matrix": _confusion_matrix(type_true, type_pred, labels=[0, 1, 2, 3]),
        },
        "event_direction_confusion_matrix": {
            "labels": dir_names,
            "matrix": _confusion_matrix(direction_true, direction_pred, labels=[0, 1, 2]),
        },
    }


def _run_epoch(
    model: Stage1BJointEventSequencePredictor,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer | None,
    include_confusion: bool = False,
) -> dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_samples = 0
    y_true_parts = []
    y_pred_parts = []
    with torch.set_grad_enabled(train):
        for features, targets, _metas in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)["logits"]
            loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_size = int(features.size(0))
            total_samples += batch_size
            total_loss += float(loss.item()) * batch_size
            y_true_parts.append(targets.detach().cpu().numpy())
            y_pred_parts.append(logits.argmax(dim=1).detach().cpu().numpy())

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    labels = list(range(len(JOINT_EVENT_LABEL_TO_ID)))
    joint_f1, joint_per_class = _macro_f1(y_true, y_pred, labels=labels)
    metrics: dict[str, Any] = {
        "loss": total_loss / max(total_samples, 1),
        "joint_macro_f1": joint_f1,
        "joint_per_class_f1": joint_per_class,
        "joint_accuracy": float(np.mean(y_true == y_pred)),
        **_derived_metrics(y_true, y_pred),
    }
    if include_confusion:
        metrics["joint_confusion_matrix"] = {
            "labels": [JOINT_EVENT_ID_TO_LABEL[idx] for idx in labels],
            "matrix": _confusion_matrix(y_true, y_pred, labels=labels),
        }
    return metrics


def _collect_predictions(
    model: Stage1BJointEventSequencePredictor,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    with torch.no_grad():
        for features, targets, metas in loader:
            features = features.to(device)
            probs = torch.softmax(model(features)["logits"], dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            for i, meta in enumerate(metas):
                true_label = JOINT_EVENT_ID_TO_LABEL[int(targets[i].item())]
                pred_label = JOINT_EVENT_ID_TO_LABEL[int(preds[i])]
                pred_event_type = _event_type_from_joint(pred_label)
                joint_confidence = float(probs[i, int(preds[i])])
                candidate_gate = pred_label != "no_event" and (
                    joint_confidence >= 0.50 or (pred_event_type == "reversal" and joint_confidence >= 0.45)
                )
                row = {
                    "index": int(meta.index),
                    "symbol": meta.symbol,
                    "timestamp": int(meta.timestamp),
                    "true_joint_event": true_label,
                    "pred_joint_event": pred_label,
                    "joint_confidence": joint_confidence,
                    "candidate_strength": joint_confidence,
                    "candidate_gate": bool(candidate_gate),
                    "true_event_type": _event_type_from_joint(true_label),
                    "pred_event_type": pred_event_type,
                    "true_event_direction": _direction_from_joint(true_label),
                    "pred_event_direction": _direction_from_joint(pred_label),
                    "is_joint_correct": bool(true_label == pred_label),
                    "is_event_type_correct": bool(_event_type_from_joint(true_label) == _event_type_from_joint(pred_label)),
                    "is_event_direction_correct": bool(
                        _direction_from_joint(true_label) == "none"
                        or _direction_from_joint(true_label) == _direction_from_joint(pred_label)
                    ),
                }
                for label, label_id in JOINT_EVENT_LABEL_TO_ID.items():
                    row[f"p_{label}"] = float(probs[i, label_id])
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
    parser.add_argument("--outcome-column", type=str, default=None)
    parser.add_argument("--dominant-direction-column", type=str, default=None)
    parser.add_argument("--target-mode", choices=["event", "outcome"], default="event")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--balanced-loss", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()

    generator = _set_seed(args.seed)
    train_dataset = Stage1BJointEventSequenceDataset.from_paths(
        latent_path=args.train_latent_path,
        label_path=args.train_label_path,
        feature_keys=args.feature_keys,
        context_feature_keys=args.context_feature_keys,
        event_type_column=args.event_type_column,
        event_direction_column=args.event_direction_column,
        outcome_column=args.outcome_column,
        dominant_direction_column=args.dominant_direction_column,
        target_mode=args.target_mode,
    )
    val_dataset = Stage1BJointEventSequenceDataset.from_paths(
        latent_path=args.val_latent_path,
        label_path=args.val_label_path,
        feature_keys=args.feature_keys,
        context_feature_keys=args.context_feature_keys,
        event_type_column=args.event_type_column,
        event_direction_column=args.event_direction_column,
        outcome_column=args.outcome_column,
        dominant_direction_column=args.dominant_direction_column,
        target_mode=args.target_mode,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Stage 1B joint event train/val dataset is empty")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn_stage1b_joint_event,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_stage1b_joint_event,
    )

    device = get_device()
    model = Stage1BJointEventSequencePredictor(
        input_dim=train_dataset.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=len(JOINT_EVENT_LABEL_TO_ID),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = (
        torch.tensor(
            _balanced_weights(train_dataset.joint_targets, labels=list(range(len(JOINT_EVENT_LABEL_TO_ID)))),
            dtype=torch.float32,
            device=device,
        )
        if args.balanced_loss
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_config = vars(args).copy()
    for key, value in list(train_config.items()):
        if isinstance(value, Path):
            train_config[key] = str(value)
    (args.checkpoint_dir / "train_config.json").write_text(json.dumps(train_config, indent=2))

    logger.info("Stage 1B joint event training device: {}", device)
    logger.info(
        "Stage 1B joint event dataset | train={} val={} input_dim={} features={} context={}",
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
        "train_joint_macro_f1": [],
        "train_event_type_macro_f1": [],
        "train_event_direction_macro_f1": [],
        "val_loss": [],
        "val_joint_macro_f1": [],
        "val_event_type_macro_f1": [],
        "val_event_direction_macro_f1": [],
    }
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics = _run_epoch(model, val_loader, device, criterion, None)
        for prefix, metrics in [("train", train_metrics), ("val", val_metrics)]:
            history[f"{prefix}_loss"].append(float(metrics["loss"]))
            history[f"{prefix}_joint_macro_f1"].append(float(metrics["joint_macro_f1"]))
            history[f"{prefix}_event_type_macro_f1"].append(float(metrics["event_type_macro_f1"]))
            history[f"{prefix}_event_direction_macro_f1"].append(float(metrics["event_direction_macro_f1"]))
        logger.info(
            "Epoch {}/{} | train_loss={:.5f} val_loss={:.5f} "
            "val_joint_f1={:.4f} val_type_f1={:.4f} val_dir_f1={:.4f}",
            epoch,
            args.epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["joint_macro_f1"],
            val_metrics["event_type_macro_f1"],
            val_metrics["event_direction_macro_f1"],
        )

        score = float(val_metrics["joint_macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "metrics": val_metrics},
                args.checkpoint_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch {}", epoch)
                break

    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": len(history["val_loss"]), "metrics": val_metrics},
        args.checkpoint_dir / "final.pt",
    )
    (args.checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))

    best_checkpoint = torch.load(args.checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_val_metrics = _run_epoch(model, val_loader, device, criterion, None, include_confusion=True)

    test_metrics = None
    test_samples = None
    if args.test_latent_path is not None and args.test_label_path is not None:
        test_dataset = Stage1BJointEventSequenceDataset.from_paths(
            latent_path=args.test_latent_path,
            label_path=args.test_label_path,
            feature_keys=args.feature_keys,
            context_feature_keys=args.context_feature_keys,
            event_type_column=args.event_type_column,
            event_direction_column=args.event_direction_column,
            outcome_column=args.outcome_column,
            dominant_direction_column=args.dominant_direction_column,
            target_mode=args.target_mode,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_stage1b_joint_event,
        )
        test_metrics = _run_epoch(model, test_loader, device, criterion, None, include_confusion=True)
        test_samples = len(test_dataset)
        _write_csv(args.checkpoint_dir / "test_joint_event_predictions.csv", _collect_predictions(model, test_loader, device))

    metadata = {
        "event_type_column": args.event_type_column,
        "event_direction_column": args.event_direction_column,
        "outcome_column": args.outcome_column,
        "dominant_direction_column": args.dominant_direction_column,
        "target_mode": args.target_mode,
        "feature_keys": args.feature_keys,
        "context_feature_keys": args.context_feature_keys,
        "input_dim": train_dataset.input_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "seed": args.seed,
        "device": str(device),
        "balanced_loss": bool(args.balanced_loss),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": test_samples,
        "train_latent_path": str(args.train_latent_path),
        "train_label_path": str(args.train_label_path),
        "val_latent_path": str(args.val_latent_path),
        "val_label_path": str(args.val_label_path),
        "test_latent_path": None if args.test_latent_path is None else str(args.test_latent_path),
        "test_label_path": None if args.test_label_path is None else str(args.test_label_path),
        "train_joint_counts": _label_counts(train_dataset.joint_targets),
        "val_joint_counts": _label_counts(val_dataset.joint_targets),
        "best_epoch": int(best_epoch),
        "best_val_joint_macro_f1": float(best_score),
        "best_val_event_type_macro_f1": max(history["val_event_type_macro_f1"]),
        "best_val_event_direction_macro_f1": max(history["val_event_direction_macro_f1"]),
        "final_val_metrics": final_val_metrics,
        "test_metrics": test_metrics,
    }
    (args.checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(
        "Stage 1B joint event training finished -> {} | best_val_joint_f1={:.4f}",
        args.checkpoint_dir,
        best_score,
    )


if __name__ == "__main__":
    main()
