"""Audit Stage 1B predictor errors against forward label metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader

from msb_repr.stage1a.model import get_device
from msb_repr.stage1b.dataset import Stage1BForwardDataset, collate_fn_stage1b
from msb_repr.stage1b.model import Stage1BForwardPredictor


RECON_ID_TO_LABEL = {0: "none", 1: "bullish", 2: "bearish"}
DIRECTION_ID_TO_LABEL = {0: "bullish", 1: "bearish", -1: "none"}


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _safe_median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else 0.0


def _counts(values: np.ndarray | list[str]) -> dict[str, int]:
    if len(values) == 0:
        return {}
    unique, counts = np.unique(np.asarray(values), return_counts=True)
    return {str(value): int(count) for value, count in zip(unique, counts, strict=True)}


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def _collect_predictions(
    model: Stage1BForwardPredictor,
    loader: DataLoader,
    device: torch.device,
    horizon_index: int,
) -> dict[str, np.ndarray]:
    model.eval()
    break_probs = []
    direction_probs = []
    direction_preds = []
    break_targets = []
    direction_targets = []
    recon_targets = []
    indices = []
    symbols = []
    timestamps = []
    with torch.no_grad():
        for features, batch_break_targets, batch_direction_targets, batch_recon_targets, metas in loader:
            features = features.to(device)
            outputs = model(features)
            break_probs.append(torch.softmax(outputs["break_logits"], dim=2)[:, horizon_index, 1].cpu().numpy())
            direction_probs.append(torch.softmax(outputs["direction_logits"], dim=2)[:, horizon_index].cpu().numpy())
            direction_preds.append(outputs["direction_logits"].argmax(dim=2)[:, horizon_index].cpu().numpy())
            break_targets.append(batch_break_targets[:, horizon_index].numpy())
            direction_targets.append(batch_direction_targets[:, horizon_index].numpy())
            recon_targets.append(batch_recon_targets[:, horizon_index].numpy())
            indices.extend(meta.index for meta in metas)
            symbols.extend(meta.symbol for meta in metas)
            timestamps.extend(meta.timestamp for meta in metas)
    return {
        "break_probs": np.concatenate(break_probs),
        "direction_probs": np.concatenate(direction_probs),
        "direction_preds": np.concatenate(direction_preds),
        "break_targets": np.concatenate(break_targets),
        "direction_targets": np.concatenate(direction_targets),
        "recon_targets": np.concatenate(recon_targets),
        "indices": np.array(indices, dtype=np.int64),
        "symbols": np.array(symbols),
        "timestamps": np.array(timestamps, dtype=np.int64),
    }


def _load_dataset(metadata: dict[str, Any], split: str) -> Stage1BForwardDataset:
    return Stage1BForwardDataset.from_paths(
        latent_path=Path(metadata[f"{split}_latent_path"]),
        label_path=Path(metadata[f"{split}_label_path"]),
        feature_keys=list(metadata["feature_keys"]),
        context_feature_keys=list(metadata.get("context_feature_keys", [])),
        target_column=str(metadata["target_column"]),
        target_columns=metadata.get("target_columns"),
    )


def _metadata_by_index(label_path: Path, indices: np.ndarray) -> dict[int, dict[str, Any]]:
    label_df = pl.read_parquet(label_path)
    wanted = set(int(idx) for idx in indices)
    rows = label_df.filter(pl.col("index").is_in(list(wanted))).to_dicts()
    return {int(row["index"]): row for row in rows}


def _build_rows(
    predictions: dict[str, np.ndarray],
    label_rows: dict[int, dict[str, Any]],
    threshold: float,
    target_prefix: str,
    secondary_prefix: str | None,
) -> list[dict[str, Any]]:
    break_pred = (predictions["break_probs"] >= threshold).astype(np.int64)
    direction_pred = predictions["direction_preds"]
    recon_pred = np.zeros_like(predictions["recon_targets"])
    recon_pred[break_pred == 1] = direction_pred[break_pred == 1] + 1
    direction_conf = np.max(predictions["direction_probs"], axis=1)
    break_conf = np.where(break_pred == 1, predictions["break_probs"], 1.0 - predictions["break_probs"])

    rows = []
    for i, index in enumerate(predictions["indices"]):
        label_row = label_rows[int(index)]
        true_recon = int(predictions["recon_targets"][i])
        pred_recon = int(recon_pred[i])
        true_break = int(predictions["break_targets"][i])
        pred_break = int(break_pred[i])
        row = {
            "index": int(index),
            "symbol": str(predictions["symbols"][i]),
            "timestamp": int(predictions["timestamps"][i]),
            "true_label": RECON_ID_TO_LABEL[true_recon],
            "pred_label": RECON_ID_TO_LABEL[pred_recon],
            "true_break": true_break,
            "pred_break": pred_break,
            "p_break": float(predictions["break_probs"][i]),
            "break_confidence": float(break_conf[i]),
            "pred_direction": DIRECTION_ID_TO_LABEL[int(direction_pred[i])],
            "direction_confidence": float(direction_conf[i]),
            "is_correct": bool(true_recon == pred_recon),
            "error_type": _error_type(true_break, pred_break, true_recon, pred_recon),
            f"{target_prefix}_time_to_break": int(label_row.get(f"{target_prefix}_time_to_break", -1)),
            f"{target_prefix}_bull_close_count": int(label_row.get(f"{target_prefix}_bull_close_count", 0)),
            f"{target_prefix}_bear_close_count": int(label_row.get(f"{target_prefix}_bear_close_count", 0)),
            f"{target_prefix}_bull_confirm_bar": int(label_row.get(f"{target_prefix}_bull_confirm_bar", -1)),
            f"{target_prefix}_bear_confirm_bar": int(label_row.get(f"{target_prefix}_bear_confirm_bar", -1)),
        }
        if secondary_prefix:
            row[f"{secondary_prefix}_label"] = str(label_row.get(f"{secondary_prefix}_future_break_direction", ""))
            row[f"{secondary_prefix}_time_to_break"] = int(label_row.get(f"{secondary_prefix}_time_to_break", -1))
        rows.append(row)
    return rows


def _error_type(true_break: int, pred_break: int, true_recon: int, pred_recon: int) -> str:
    if true_recon == pred_recon:
        return "correct"
    if true_break == 0 and pred_break == 1:
        return "false_positive_break"
    if true_break == 1 and pred_break == 0:
        return "false_negative_break"
    if true_break == 1 and pred_break == 1:
        return "wrong_direction"
    return "other"


def _segment_summary(rows: list[dict[str, Any]], name: str, target_prefix: str, secondary_prefix: str | None) -> dict[str, Any]:
    if not rows:
        return {"name": name, "support": 0}
    p_break = np.array([row["p_break"] for row in rows], dtype=np.float64)
    break_conf = np.array([row["break_confidence"] for row in rows], dtype=np.float64)
    direction_conf = np.array([row["direction_confidence"] for row in rows], dtype=np.float64)
    time_to_break = np.array([row[f"{target_prefix}_time_to_break"] for row in rows], dtype=np.float64)
    valid_ttb = time_to_break[time_to_break >= 0]
    bull_close_count = np.array([row[f"{target_prefix}_bull_close_count"] for row in rows], dtype=np.float64)
    bear_close_count = np.array([row[f"{target_prefix}_bear_close_count"] for row in rows], dtype=np.float64)
    out = {
        "name": name,
        "support": len(rows),
        "symbol_counts": _counts([row["symbol"] for row in rows]),
        "true_label_counts": _counts([row["true_label"] for row in rows]),
        "pred_label_counts": _counts([row["pred_label"] for row in rows]),
        "p_break_mean": _safe_mean(p_break),
        "p_break_median": _safe_median(p_break),
        "break_confidence_mean": _safe_mean(break_conf),
        "direction_confidence_mean": _safe_mean(direction_conf),
        "time_to_break": _quantiles(valid_ttb),
        "bull_close_count": _quantiles(bull_close_count),
        "bear_close_count": _quantiles(bear_close_count),
    }
    if secondary_prefix:
        secondary_labels = [str(row.get(f"{secondary_prefix}_label", "")) for row in rows]
        out[f"{secondary_prefix}_label_counts"] = _counts(secondary_labels)
    return out


def _confidence_buckets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    out = []
    for lo, hi in buckets:
        bucket = [row for row in rows if lo <= float(row["break_confidence"]) < hi]
        out.append(
            {
                "break_confidence_min": lo,
                "break_confidence_max": min(hi, 1.0),
                "support": len(bucket),
                "accuracy": _safe_mean(np.array([row["is_correct"] for row in bucket], dtype=np.float64)),
                "error_type_counts": _counts([row["error_type"] for row in bucket]),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _top_rows(rows: list[dict[str, Any]], error_type: str, limit: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["error_type"] == error_type]
    return sorted(selected, key=lambda row: float(row["break_confidence"]), reverse=True)[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--break-threshold", type=float, default=0.5)
    parser.add_argument("--target-prefix", type=str, default="h8")
    parser.add_argument("--secondary-prefix", type=str, default="h16")
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=None)
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

    dataset = _load_dataset(metadata, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_stage1b)
    horizon_index = int(metadata.get("primary_horizon_index", 0))
    predictions = _collect_predictions(model, loader, device, horizon_index=horizon_index)
    label_rows = _metadata_by_index(Path(metadata[f"{args.split}_label_path"]), predictions["indices"])
    rows = _build_rows(
        predictions=predictions,
        label_rows=label_rows,
        threshold=args.break_threshold,
        target_prefix=args.target_prefix,
        secondary_prefix=args.secondary_prefix,
    )

    output_dir = args.output_dir or (args.checkpoint_dir / f"error_audit_{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = {
        "all": rows,
        "correct": [row for row in rows if row["error_type"] == "correct"],
        "false_positive_break": [row for row in rows if row["error_type"] == "false_positive_break"],
        "false_negative_break": [row for row in rows if row["error_type"] == "false_negative_break"],
        "wrong_direction": [row for row in rows if row["error_type"] == "wrong_direction"],
    }
    if args.secondary_prefix:
        segments[f"{args.target_prefix}_none_{args.secondary_prefix}_break"] = [
            row
            for row in rows
            if row["true_label"] == "none"
            and str(row.get(f"{args.secondary_prefix}_label", "")) in {"bullish", "bearish"}
        ]

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_name": args.checkpoint_name,
        "split": args.split,
        "target_prefix": args.target_prefix,
        "secondary_prefix": args.secondary_prefix,
        "break_threshold": args.break_threshold,
        "target_column": metadata["target_column"],
        "feature_keys": metadata["feature_keys"],
        "segment_summaries": {
            name: _segment_summary(segment_rows, name, args.target_prefix, args.secondary_prefix)
            for name, segment_rows in segments.items()
        },
        "confidence_buckets": _confidence_buckets(rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_csv(output_dir / "all_predictions.csv", rows)
    for error_type in ["false_positive_break", "false_negative_break", "wrong_direction"]:
        _write_csv(output_dir / f"top_{error_type}.csv", _top_rows(rows, error_type, args.sample_limit))

    logger.info(
        "Stage 1B error audit complete -> {} | fp={} fn={} wrong_dir={}",
        output_dir,
        len(segments["false_positive_break"]),
        len(segments["false_negative_break"]),
        len(segments["wrong_direction"]),
    )


if __name__ == "__main__":
    main()
