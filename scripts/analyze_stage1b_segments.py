"""Analyze Stage 1B multi-horizon prediction segments."""

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


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> float:
    scores = []
    for label in labels:
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        scores.append(2.0 * precision * recall / (precision + recall + 1e-8))
    return float(np.mean(scores))


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _safe_rate(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else 0.0


def _label_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(label): int(count) for label, count in zip(unique, counts, strict=True)}


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> list[list[int]]:
    return [
        [int(np.sum((y_true == true_label) & (y_pred == pred_label))) for pred_label in labels]
        for true_label in labels
    ]


def _collect_predictions(
    model: Stage1BForwardPredictor,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    break_probs = []
    direction_probs = []
    direction_preds = []
    break_targets = []
    direction_targets = []
    recon_targets = []
    symbols = []
    timestamps = []
    indices = []
    with torch.no_grad():
        for features, batch_break_targets, batch_direction_targets, batch_recon_targets, metas in loader:
            features = features.to(device)
            outputs = model(features)
            break_probs.append(torch.softmax(outputs["break_logits"], dim=2)[:, :, 1].cpu().numpy())
            direction_probs.append(torch.softmax(outputs["direction_logits"], dim=2).cpu().numpy())
            direction_preds.append(outputs["direction_logits"].argmax(dim=2).cpu().numpy())
            break_targets.append(batch_break_targets.numpy())
            direction_targets.append(batch_direction_targets.numpy())
            recon_targets.append(batch_recon_targets.numpy())
            symbols.extend(meta.symbol for meta in metas)
            timestamps.extend(meta.timestamp for meta in metas)
            indices.extend(meta.index for meta in metas)
    return {
        "break_probs": np.concatenate(break_probs),
        "direction_probs": np.concatenate(direction_probs),
        "direction_preds": np.concatenate(direction_preds),
        "break_targets": np.concatenate(break_targets),
        "direction_targets": np.concatenate(direction_targets),
        "recon_targets": np.concatenate(recon_targets),
        "symbols": np.array(symbols),
        "timestamps": np.array(timestamps, dtype=np.int64),
        "indices": np.array(indices, dtype=np.int64),
    }


def _segment_summary(
    predictions: dict[str, np.ndarray],
    break_threshold: float,
    h16_idx: int,
    h32_idx: int,
) -> dict[str, Any]:
    h16_true_none = predictions["break_targets"][:, h16_idx] == 0
    h32_true_break = predictions["break_targets"][:, h32_idx] == 1
    true_segment = h16_true_none & h32_true_break

    h16_pred_none = predictions["break_probs"][:, h16_idx] < break_threshold
    h32_pred_break = predictions["break_probs"][:, h32_idx] >= break_threshold
    pred_segment = h16_pred_none & h32_pred_break

    h32_direction_true = predictions["direction_targets"][:, h32_idx]
    h32_direction_pred = predictions["direction_preds"][:, h32_idx]
    direction_mask = true_segment & (h32_direction_true >= 0)

    segment_support = int(np.sum(true_segment))
    predicted_support = int(np.sum(pred_segment))
    intersection_support = int(np.sum(true_segment & pred_segment))
    total = int(len(true_segment))

    precision = intersection_support / max(predicted_support, 1)
    recall = intersection_support / max(segment_support, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    true_h32_direction_counts = _label_counts(h32_direction_true[direction_mask])
    pred_h32_direction_counts = _label_counts(h32_direction_pred[direction_mask])
    h32_direction_accuracy = (
        float(np.mean(h32_direction_true[direction_mask] == h32_direction_pred[direction_mask]))
        if np.any(direction_mask)
        else 0.0
    )
    h32_direction_macro_f1 = (
        _macro_f1(h32_direction_true[direction_mask], h32_direction_pred[direction_mask], labels=[0, 1])
        if np.any(direction_mask)
        else 0.0
    )

    symbol_summary = {}
    for symbol in sorted(np.unique(predictions["symbols"])):
        symbol_mask = predictions["symbols"] == symbol
        symbol_segment = true_segment & symbol_mask
        symbol_pred_segment = pred_segment & symbol_mask
        symbol_intersection = symbol_segment & symbol_pred_segment
        symbol_summary[str(symbol)] = {
            "samples": int(np.sum(symbol_mask)),
            "true_segment": int(np.sum(symbol_segment)),
            "pred_segment": int(np.sum(symbol_pred_segment)),
            "segment_recall": float(np.sum(symbol_intersection) / max(np.sum(symbol_segment), 1)),
        }

    return {
        "samples": total,
        "true_segment_samples": segment_support,
        "true_segment_rate": segment_support / max(total, 1),
        "predicted_segment_samples": predicted_support,
        "predicted_segment_rate": predicted_support / max(total, 1),
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": f1,
        "h16_no_break_recall_inside_true_segment": _safe_rate(h16_pred_none[true_segment]),
        "h32_break_recall_inside_true_segment": _safe_rate(h32_pred_break[true_segment]),
        "h32_direction_accuracy_inside_true_segment": h32_direction_accuracy,
        "h32_direction_macro_f1_inside_true_segment": h32_direction_macro_f1,
        "h32_direction_confusion_inside_true_segment": {
            "labels": ["bullish", "bearish"],
            "matrix": _confusion_matrix(
                h32_direction_true[direction_mask],
                h32_direction_pred[direction_mask],
                labels=[0, 1],
            )
            if np.any(direction_mask)
            else [[0, 0], [0, 0]],
        },
        "h32_true_direction_counts_inside_true_segment": true_h32_direction_counts,
        "h32_pred_direction_counts_inside_true_segment": pred_h32_direction_counts,
        "mean_p_break_h16_inside_true_segment": _safe_mean(predictions["break_probs"][:, h16_idx][true_segment]),
        "mean_p_break_h32_inside_true_segment": _safe_mean(predictions["break_probs"][:, h32_idx][true_segment]),
        "mean_p_break_h16_inside_pred_segment": _safe_mean(predictions["break_probs"][:, h16_idx][pred_segment]),
        "mean_p_break_h32_inside_pred_segment": _safe_mean(predictions["break_probs"][:, h32_idx][pred_segment]),
        "symbol_summary": symbol_summary,
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
    parser.add_argument("--break-threshold", type=float, default=0.5)
    parser.add_argument("--h16-index", type=int, default=0)
    parser.add_argument("--h32-index", type=int, default=1)
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

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_name": args.checkpoint_name,
        "break_threshold": args.break_threshold,
        "segment_definition": "true H16=none and true H32=bullish/bearish",
        "target_columns": metadata.get("target_columns"),
        "splits": {},
    }
    for split in ["val", "test"]:
        dataset = _load_dataset(metadata, split)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_stage1b)
        predictions = _collect_predictions(model, loader, device)
        summary["splits"][split] = _segment_summary(
            predictions,
            break_threshold=args.break_threshold,
            h16_idx=args.h16_index,
            h32_idx=args.h32_index,
        )

    output_path = args.output_path or (args.checkpoint_dir / "segment_h16_none_h32_break.json")
    output_path.write_text(json.dumps(summary, indent=2))
    test = summary["splits"]["test"]
    logger.info(
        "Segment analysis complete -> {} | test_support={} precision={:.4f} recall={:.4f} "
        "h32_dir_f1={:.4f}",
        output_path,
        test["true_segment_samples"],
        test["segment_precision"],
        test["segment_recall"],
        test["h32_direction_macro_f1_inside_true_segment"],
    )


if __name__ == "__main__":
    main()
