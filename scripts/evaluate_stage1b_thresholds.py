"""Sweep Stage 1B break thresholds on validation and confirm on test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from msb_repr.stage1a.model import get_device
from msb_repr.stage1b.dataset import Stage1BForwardDataset, collate_fn_stage1b
from msb_repr.stage1b.model import Stage1BForwardPredictor


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> tuple[float, dict[str, float]]:
    per_label = {}
    scores = []
    for label in labels:
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        per_label[f"class_{label}_f1"] = f1
        scores.append(f1)
    return float(np.mean(scores)), per_label


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> list[list[int]]:
    return [
        [int(np.sum((y_true == true_label) & (y_pred == pred_label))) for pred_label in labels]
        for true_label in labels
    ]


def _collect_predictions(
    model: Stage1BForwardPredictor,
    loader: DataLoader,
    device: torch.device,
    horizon_index: int,
) -> dict[str, np.ndarray]:
    model.eval()
    break_probs = []
    direction_preds = []
    break_targets = []
    direction_targets = []
    recon_targets = []
    with torch.no_grad():
        for features, batch_break_targets, batch_direction_targets, batch_recon_targets, _metas in loader:
            features = features.to(device)
            outputs = model(features)
            break_probs.append(torch.softmax(outputs["break_logits"], dim=2)[:, horizon_index, 1].cpu().numpy())
            direction_preds.append(outputs["direction_logits"].argmax(dim=2)[:, horizon_index].cpu().numpy())
            break_targets.append(batch_break_targets[:, horizon_index].numpy())
            direction_targets.append(batch_direction_targets[:, horizon_index].numpy())
            recon_targets.append(batch_recon_targets[:, horizon_index].numpy())
    return {
        "break_probs": np.concatenate(break_probs),
        "direction_preds": np.concatenate(direction_preds),
        "break_targets": np.concatenate(break_targets),
        "direction_targets": np.concatenate(direction_targets),
        "recon_targets": np.concatenate(recon_targets),
    }


def _evaluate_at_threshold(predictions: dict[str, np.ndarray], threshold: float) -> dict[str, Any]:
    break_pred = (predictions["break_probs"] >= threshold).astype(np.int64)
    recon_pred = np.zeros_like(predictions["recon_targets"], dtype=np.int64)
    recon_pred[break_pred == 1] = predictions["direction_preds"][break_pred == 1] + 1

    break_f1, break_per_class = _macro_f1(predictions["break_targets"], break_pred, labels=[0, 1])
    recon_f1, recon_per_class = _macro_f1(predictions["recon_targets"], recon_pred, labels=[0, 1, 2])
    direction_mask = predictions["direction_targets"] >= 0
    direction_f1, direction_per_class = _macro_f1(
        predictions["direction_targets"][direction_mask],
        predictions["direction_preds"][direction_mask],
        labels=[0, 1],
    )
    return {
        "threshold": float(threshold),
        "recon_macro_f1": recon_f1,
        "recon_per_class_f1": recon_per_class,
        "recon_accuracy": float(np.mean(predictions["recon_targets"] == recon_pred)),
        "break_macro_f1": break_f1,
        "break_per_class_f1": break_per_class,
        "break_accuracy": float(np.mean(predictions["break_targets"] == break_pred)),
        "direction_macro_f1": direction_f1,
        "direction_per_class_f1": direction_per_class,
        "direction_accuracy": float(
            np.mean(predictions["direction_targets"][direction_mask] == predictions["direction_preds"][direction_mask])
        ),
        "break_pred_rate": float(np.mean(break_pred)),
        "recon_confusion_matrix": {
            "labels": ["none", "bullish", "bearish"],
            "matrix": _confusion_matrix(predictions["recon_targets"], recon_pred, labels=[0, 1, 2]),
        },
        "break_confusion_matrix": {
            "labels": ["none", "break"],
            "matrix": _confusion_matrix(predictions["break_targets"], break_pred, labels=[0, 1]),
        },
    }


def _load_dataset(metadata: dict[str, Any], split: str) -> Stage1BForwardDataset:
    return Stage1BForwardDataset.from_paths(
        latent_path=Path(metadata[f"{split}_latent_path"]),
        label_path=Path(metadata[f"{split}_label_path"]),
        feature_keys=list(metadata["feature_keys"]),
        target_column=str(metadata["target_column"]),
        target_columns=metadata.get("target_columns"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    metadata = json.loads((args.checkpoint_dir / "metadata.json").read_text())
    device = get_device()
    model = Stage1BForwardPredictor(
        input_dim=int(metadata["input_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        dropout=float(metadata["dropout"]),
        num_horizons=int(metadata.get("num_horizons", 1)),
    ).to(device)
    checkpoint = torch.load(args.checkpoint_dir / args.checkpoint_name, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_dataset = _load_dataset(metadata, "val")
    test_dataset = _load_dataset(metadata, "test")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_stage1b)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_stage1b)

    horizon_index = int(metadata.get("primary_horizon_index", 0))
    val_predictions = _collect_predictions(model, val_loader, device, horizon_index=horizon_index)
    test_predictions = _collect_predictions(model, test_loader, device, horizon_index=horizon_index)

    thresholds = np.arange(args.threshold_min, args.threshold_max + 0.5 * args.threshold_step, args.threshold_step)
    val_results = [_evaluate_at_threshold(val_predictions, float(threshold)) for threshold in thresholds]
    best_val = max(val_results, key=lambda row: (row["recon_macro_f1"], row["break_macro_f1"]))
    test_at_best = _evaluate_at_threshold(test_predictions, float(best_val["threshold"]))
    test_at_default = _evaluate_at_threshold(test_predictions, float(metadata.get("break_threshold", 0.5)))

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_name": args.checkpoint_name,
        "target_column": metadata["target_column"],
        "feature_keys": metadata["feature_keys"],
        "selection_metric": "val_recon_macro_f1",
        "best_val": best_val,
        "test_at_best_val_threshold": test_at_best,
        "test_at_default_threshold": test_at_default,
        "val_sweep": [
            {
                "threshold": row["threshold"],
                "recon_macro_f1": row["recon_macro_f1"],
                "break_macro_f1": row["break_macro_f1"],
                "break_pred_rate": row["break_pred_rate"],
            }
            for row in val_results
        ],
    }
    output_path = args.output_path or (args.checkpoint_dir / "threshold_sweep.json")
    output_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "Threshold sweep complete -> {} | best_val_threshold={:.2f} val_recon_f1={:.4f} "
        "test_recon_f1={:.4f}",
        output_path,
        best_val["threshold"],
        best_val["recon_macro_f1"],
        test_at_best["recon_macro_f1"],
    )


if __name__ == "__main__":
    main()
