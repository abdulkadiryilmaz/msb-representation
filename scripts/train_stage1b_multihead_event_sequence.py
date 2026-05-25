"""Train Stage 1B multi-head event-sequence predictor from frozen Stage 1A latents."""

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
    DIRECTION3_ID_TO_LABEL,
    DIRECTION3_LABEL_TO_ID,
    JOINT_EVENT_ID_TO_LABEL,
    JOINT_EVENT_LABEL_TO_ID,
    Stage1BMultiHeadEventSequenceDataset,
    collate_fn_stage1b_multihead_event,
)
from msb_repr.stage1b.model import Stage1BMultiHeadEventSequencePredictor


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


def _joint_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {JOINT_EVENT_ID_TO_LABEL[int(label)]: int(count) for label, count in zip(unique, counts, strict=True)}


def _direction_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {DIRECTION3_ID_TO_LABEL[int(label)]: int(count) for label, count in zip(unique, counts, strict=True)}


def _joint_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, Any]:
    labels = list(range(len(JOINT_EVENT_LABEL_TO_ID)))
    f1, per_class = _macro_f1(y_true, y_pred, labels)
    return {
        f"{prefix}_macro_f1": f1,
        f"{prefix}_per_class_f1": per_class,
        f"{prefix}_accuracy": float(np.mean(y_true == y_pred)),
    }


def _direction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = list(range(len(DIRECTION3_LABEL_TO_ID)))
    f1, per_class = _macro_f1(y_true, y_pred, labels)
    return {
        "dominant_direction_macro_f1": f1,
        "dominant_direction_per_class_f1": per_class,
        "dominant_direction_accuracy": float(np.mean(y_true == y_pred)),
    }


def _run_epoch(
    model: Stage1BMultiHeadEventSequencePredictor,
    loader: DataLoader,
    device: torch.device,
    criteria: dict[str, nn.CrossEntropyLoss],
    loss_weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None,
    include_confusion: bool = False,
) -> dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_samples = 0
    parts: dict[str, list[np.ndarray]] = {
        "first_true": [],
        "first_pred": [],
        "outcome_true": [],
        "outcome_pred": [],
        "dominant_true": [],
        "dominant_pred": [],
    }
    with torch.set_grad_enabled(train):
        for features, first_targets, outcome_targets, dominant_targets, _metas in loader:
            features = features.to(device)
            first_targets = first_targets.to(device)
            outcome_targets = outcome_targets.to(device)
            dominant_targets = dominant_targets.to(device)
            outputs = model(features)
            loss = (
                loss_weights["first_event"] * criteria["first_event"](outputs["first_event_logits"], first_targets)
                + loss_weights["outcome"] * criteria["outcome"](outputs["outcome_logits"], outcome_targets)
                + loss_weights["dominant_direction"]
                * criteria["dominant_direction"](outputs["dominant_direction_logits"], dominant_targets)
            )
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_size = int(features.size(0))
            total_samples += batch_size
            total_loss += float(loss.item()) * batch_size
            parts["first_true"].append(first_targets.detach().cpu().numpy())
            parts["first_pred"].append(outputs["first_event_logits"].argmax(dim=1).detach().cpu().numpy())
            parts["outcome_true"].append(outcome_targets.detach().cpu().numpy())
            parts["outcome_pred"].append(outputs["outcome_logits"].argmax(dim=1).detach().cpu().numpy())
            parts["dominant_true"].append(dominant_targets.detach().cpu().numpy())
            parts["dominant_pred"].append(outputs["dominant_direction_logits"].argmax(dim=1).detach().cpu().numpy())

    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    metrics: dict[str, Any] = {
        "loss": total_loss / max(total_samples, 1),
        **_joint_metrics(arrays["first_true"], arrays["first_pred"], "first_event_joint"),
        **_joint_metrics(arrays["outcome_true"], arrays["outcome_pred"], "outcome_joint"),
        **_direction_metrics(arrays["dominant_true"], arrays["dominant_pred"]),
    }
    metrics["selection_score"] = float(
        0.45 * metrics["first_event_joint_macro_f1"]
        + 0.35 * metrics["outcome_joint_macro_f1"]
        + 0.20 * metrics["dominant_direction_macro_f1"]
    )
    if include_confusion:
        joint_labels = list(range(len(JOINT_EVENT_LABEL_TO_ID)))
        direction_labels = list(range(len(DIRECTION3_LABEL_TO_ID)))
        metrics["first_event_joint_confusion_matrix"] = {
            "labels": [JOINT_EVENT_ID_TO_LABEL[idx] for idx in joint_labels],
            "matrix": _confusion_matrix(arrays["first_true"], arrays["first_pred"], joint_labels),
        }
        metrics["outcome_joint_confusion_matrix"] = {
            "labels": [JOINT_EVENT_ID_TO_LABEL[idx] for idx in joint_labels],
            "matrix": _confusion_matrix(arrays["outcome_true"], arrays["outcome_pred"], joint_labels),
        }
        metrics["dominant_direction_confusion_matrix"] = {
            "labels": [DIRECTION3_ID_TO_LABEL[idx] for idx in direction_labels],
            "matrix": _confusion_matrix(arrays["dominant_true"], arrays["dominant_pred"], direction_labels),
        }
    return metrics


def _collect_predictions(
    model: Stage1BMultiHeadEventSequencePredictor,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    with torch.no_grad():
        for features, first_targets, outcome_targets, dominant_targets, metas in loader:
            features = features.to(device)
            outputs = model(features)
            first_probs = torch.softmax(outputs["first_event_logits"], dim=1).cpu().numpy()
            outcome_probs = torch.softmax(outputs["outcome_logits"], dim=1).cpu().numpy()
            dominant_probs = torch.softmax(outputs["dominant_direction_logits"], dim=1).cpu().numpy()
            first_preds = first_probs.argmax(axis=1)
            outcome_preds = outcome_probs.argmax(axis=1)
            dominant_preds = dominant_probs.argmax(axis=1)
            for i, meta in enumerate(metas):
                true_first = JOINT_EVENT_ID_TO_LABEL[int(first_targets[i].item())]
                pred_first = JOINT_EVENT_ID_TO_LABEL[int(first_preds[i])]
                true_outcome = JOINT_EVENT_ID_TO_LABEL[int(outcome_targets[i].item())]
                pred_outcome = JOINT_EVENT_ID_TO_LABEL[int(outcome_preds[i])]
                true_dominant = DIRECTION3_ID_TO_LABEL[int(dominant_targets[i].item())]
                pred_dominant = DIRECTION3_ID_TO_LABEL[int(dominant_preds[i])]
                row = {
                    "index": int(meta.index),
                    "symbol": meta.symbol,
                    "timestamp": int(meta.timestamp),
                    "true_first_event_joint": true_first,
                    "pred_first_event_joint": pred_first,
                    "first_event_confidence": float(first_probs[i, int(first_preds[i])]),
                    "true_first_event_type": _event_type_from_joint(true_first),
                    "pred_first_event_type": _event_type_from_joint(pred_first),
                    "true_first_event_direction": _direction_from_joint(true_first),
                    "pred_first_event_direction": _direction_from_joint(pred_first),
                    "true_outcome_joint": true_outcome,
                    "pred_outcome_joint": pred_outcome,
                    "outcome_confidence": float(outcome_probs[i, int(outcome_preds[i])]),
                    "true_outcome_type": _event_type_from_joint(true_outcome),
                    "pred_outcome_type": _event_type_from_joint(pred_outcome),
                    "true_outcome_direction": _direction_from_joint(true_outcome),
                    "pred_outcome_direction": _direction_from_joint(pred_outcome),
                    "true_dominant_direction": true_dominant,
                    "pred_dominant_direction": pred_dominant,
                    "dominant_direction_confidence": float(dominant_probs[i, int(dominant_preds[i])]),
                    "is_first_event_correct": bool(true_first == pred_first),
                    "is_outcome_correct": bool(true_outcome == pred_outcome),
                    "is_dominant_direction_correct": bool(true_dominant == pred_dominant),
                    "raw_first_event_type": meta.raw_first_event_type,
                    "raw_first_event_direction": meta.raw_first_event_direction,
                    "raw_outcome": meta.raw_outcome,
                    "raw_dominant_direction": meta.raw_dominant_direction,
                }
                for label, label_id in JOINT_EVENT_LABEL_TO_ID.items():
                    row[f"p_first_{label}"] = float(first_probs[i, label_id])
                    row[f"p_outcome_{label}"] = float(outcome_probs[i, label_id])
                for label, label_id in DIRECTION3_LABEL_TO_ID.items():
                    row[f"p_dominant_{label}"] = float(dominant_probs[i, label_id])
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
    parser.add_argument("--outcome-column", type=str, default="h16_post_break_outcome")
    parser.add_argument("--dominant-direction-column", type=str, default="h16_dominant_forward_direction")
    parser.add_argument("--first-event-loss-weight", type=float, default=1.0)
    parser.add_argument("--outcome-loss-weight", type=float, default=0.7)
    parser.add_argument("--dominant-direction-loss-weight", type=float, default=0.4)
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
    train_dataset = Stage1BMultiHeadEventSequenceDataset.from_paths(
        latent_path=args.train_latent_path,
        label_path=args.train_label_path,
        feature_keys=args.feature_keys,
        context_feature_keys=args.context_feature_keys,
        event_type_column=args.event_type_column,
        event_direction_column=args.event_direction_column,
        outcome_column=args.outcome_column,
        dominant_direction_column=args.dominant_direction_column,
    )
    val_dataset = Stage1BMultiHeadEventSequenceDataset.from_paths(
        latent_path=args.val_latent_path,
        label_path=args.val_label_path,
        feature_keys=args.feature_keys,
        context_feature_keys=args.context_feature_keys,
        event_type_column=args.event_type_column,
        event_direction_column=args.event_direction_column,
        outcome_column=args.outcome_column,
        dominant_direction_column=args.dominant_direction_column,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Stage 1B multi-head event train/val dataset is empty")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn_stage1b_multihead_event,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_stage1b_multihead_event,
    )

    device = get_device()
    model = Stage1BMultiHeadEventSequencePredictor(
        input_dim=train_dataset.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_joint_classes=len(JOINT_EVENT_LABEL_TO_ID),
        num_direction_classes=len(DIRECTION3_LABEL_TO_ID),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    joint_labels = list(range(len(JOINT_EVENT_LABEL_TO_ID)))
    direction_labels = list(range(len(DIRECTION3_LABEL_TO_ID)))
    criteria = {
        "first_event": nn.CrossEntropyLoss(
            weight=(
                torch.tensor(_balanced_weights(train_dataset.first_event_targets, joint_labels), dtype=torch.float32).to(
                    device
                )
                if args.balanced_loss
                else None
            )
        ),
        "outcome": nn.CrossEntropyLoss(
            weight=(
                torch.tensor(_balanced_weights(train_dataset.outcome_targets, joint_labels), dtype=torch.float32).to(device)
                if args.balanced_loss
                else None
            )
        ),
        "dominant_direction": nn.CrossEntropyLoss(
            weight=(
                torch.tensor(
                    _balanced_weights(train_dataset.dominant_direction_targets, direction_labels), dtype=torch.float32
                ).to(device)
                if args.balanced_loss
                else None
            )
        ),
    }
    loss_weights = {
        "first_event": args.first_event_loss_weight,
        "outcome": args.outcome_loss_weight,
        "dominant_direction": args.dominant_direction_loss_weight,
    }

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_config = vars(args).copy()
    for key, value in list(train_config.items()):
        if isinstance(value, Path):
            train_config[key] = str(value)
    (args.checkpoint_dir / "train_config.json").write_text(json.dumps(train_config, indent=2))

    logger.info("Stage 1B multi-head event training device: {}", device)
    logger.info(
        "Stage 1B multi-head event dataset | train={} val={} input_dim={} features={} context={}",
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
        "train_selection_score": [],
        "train_first_event_joint_macro_f1": [],
        "train_outcome_joint_macro_f1": [],
        "train_dominant_direction_macro_f1": [],
        "val_loss": [],
        "val_selection_score": [],
        "val_first_event_joint_macro_f1": [],
        "val_outcome_joint_macro_f1": [],
        "val_dominant_direction_macro_f1": [],
    }
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, criteria, loss_weights, optimizer)
        val_metrics = _run_epoch(model, val_loader, device, criteria, loss_weights, None)
        for prefix, metrics in [("train", train_metrics), ("val", val_metrics)]:
            history[f"{prefix}_loss"].append(float(metrics["loss"]))
            history[f"{prefix}_selection_score"].append(float(metrics["selection_score"]))
            history[f"{prefix}_first_event_joint_macro_f1"].append(float(metrics["first_event_joint_macro_f1"]))
            history[f"{prefix}_outcome_joint_macro_f1"].append(float(metrics["outcome_joint_macro_f1"]))
            history[f"{prefix}_dominant_direction_macro_f1"].append(float(metrics["dominant_direction_macro_f1"]))
        logger.info(
            "Epoch {}/{} | train_loss={:.5f} val_loss={:.5f} val_score={:.4f} "
            "val_first_f1={:.4f} val_outcome_f1={:.4f} val_dir_f1={:.4f}",
            epoch,
            args.epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["selection_score"],
            val_metrics["first_event_joint_macro_f1"],
            val_metrics["outcome_joint_macro_f1"],
            val_metrics["dominant_direction_macro_f1"],
        )

        score = float(val_metrics["selection_score"])
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
    final_val_metrics = _run_epoch(model, val_loader, device, criteria, loss_weights, None, include_confusion=True)

    test_metrics = None
    test_samples = None
    if args.test_latent_path is not None and args.test_label_path is not None:
        test_dataset = Stage1BMultiHeadEventSequenceDataset.from_paths(
            latent_path=args.test_latent_path,
            label_path=args.test_label_path,
            feature_keys=args.feature_keys,
            context_feature_keys=args.context_feature_keys,
            event_type_column=args.event_type_column,
            event_direction_column=args.event_direction_column,
            outcome_column=args.outcome_column,
            dominant_direction_column=args.dominant_direction_column,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_stage1b_multihead_event,
        )
        test_metrics = _run_epoch(model, test_loader, device, criteria, loss_weights, None, include_confusion=True)
        test_samples = len(test_dataset)
        _write_csv(args.checkpoint_dir / "test_multihead_event_predictions.csv", _collect_predictions(model, test_loader, device))

    metadata = {
        "event_type_column": args.event_type_column,
        "event_direction_column": args.event_direction_column,
        "outcome_column": args.outcome_column,
        "dominant_direction_column": args.dominant_direction_column,
        "feature_keys": args.feature_keys,
        "context_feature_keys": args.context_feature_keys,
        "input_dim": train_dataset.input_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "seed": args.seed,
        "device": str(device),
        "balanced_loss": bool(args.balanced_loss),
        "loss_weights": loss_weights,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": test_samples,
        "train_latent_path": str(args.train_latent_path),
        "train_label_path": str(args.train_label_path),
        "val_latent_path": str(args.val_latent_path),
        "val_label_path": str(args.val_label_path),
        "test_latent_path": None if args.test_latent_path is None else str(args.test_latent_path),
        "test_label_path": None if args.test_label_path is None else str(args.test_label_path),
        "train_first_event_counts": _joint_counts(train_dataset.first_event_targets),
        "train_outcome_counts": _joint_counts(train_dataset.outcome_targets),
        "train_dominant_direction_counts": _direction_counts(train_dataset.dominant_direction_targets),
        "val_first_event_counts": _joint_counts(val_dataset.first_event_targets),
        "val_outcome_counts": _joint_counts(val_dataset.outcome_targets),
        "val_dominant_direction_counts": _direction_counts(val_dataset.dominant_direction_targets),
        "best_epoch": int(best_epoch),
        "best_val_selection_score": float(best_score),
        "best_val_first_event_joint_macro_f1": max(history["val_first_event_joint_macro_f1"]),
        "best_val_outcome_joint_macro_f1": max(history["val_outcome_joint_macro_f1"]),
        "best_val_dominant_direction_macro_f1": max(history["val_dominant_direction_macro_f1"]),
        "final_val_metrics": final_val_metrics,
        "test_metrics": test_metrics,
    }
    (args.checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(
        "Stage 1B multi-head event training finished -> {} | best_val_score={:.4f}",
        args.checkpoint_dir,
        best_score,
    )


if __name__ == "__main__":
    main()
