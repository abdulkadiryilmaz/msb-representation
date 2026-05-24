"""Audit Stage 1B predictions by proximity to structural break levels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from msb_repr.stage1a.features import STAGE1A_SHORT_FEATURE_COLUMNS


def _safe_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else 0.0


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


def _counts(values: list[str]) -> dict[str, int]:
    if not values:
        return {}
    unique, counts = np.unique(np.asarray(values), return_counts=True)
    return {str(value): int(count) for value, count in zip(unique, counts, strict=True)}


def _load_anchor_closes(dataset_root: Path) -> dict[tuple[str, int], float]:
    close_idx = STAGE1A_SHORT_FEATURE_COLUMNS.index("close")
    out: dict[tuple[str, int], float] = {}
    for path in sorted((dataset_root / "symbols").glob("*.npz")):
        symbol = path.stem
        arrays = np.load(path)
        timestamps = arrays["timestamps"].astype(np.int64)
        closes = arrays["short_windows"][:, close_idx, -1].astype(np.float64)
        out.update({(symbol, int(ts)): float(close) for ts, close in zip(timestamps, closes, strict=True)})
    return out


def _distance_fields(row: dict[str, Any], current_close: float) -> dict[str, float | str]:
    bull_level = row.get("bull_level")
    bear_level = row.get("bear_level")
    effective_break_pct = float(row.get("effective_break_pct") or 0.0)
    pred_label = str(row["pred_label"])
    true_label = str(row["true_label"])

    bull_distance_pct = np.nan
    bear_distance_pct = np.nan
    if bull_level is not None:
        bull_distance_pct = (float(bull_level) - current_close) / max(current_close, 1e-12)
    if bear_level is not None:
        bear_distance_pct = (current_close - float(bear_level)) / max(current_close, 1e-12)

    pred_distance_pct = np.nan
    true_distance_pct = np.nan
    if pred_label == "bullish":
        pred_distance_pct = bull_distance_pct
    elif pred_label == "bearish":
        pred_distance_pct = bear_distance_pct
    if true_label == "bullish":
        true_distance_pct = bull_distance_pct
    elif true_label == "bearish":
        true_distance_pct = bear_distance_pct

    distances = np.array([bull_distance_pct, bear_distance_pct], dtype=np.float64)
    nearest_distance_pct = np.nan if np.all(np.isnan(distances)) else np.nanmin(distances)
    return {
        "current_close": float(current_close),
        "bull_distance_pct": float(bull_distance_pct),
        "bear_distance_pct": float(bear_distance_pct),
        "nearest_distance_pct": float(nearest_distance_pct),
        "pred_direction_distance_pct": float(pred_distance_pct),
        "true_direction_distance_pct": float(true_distance_pct),
        "nearest_distance_over_break_pct": float(nearest_distance_pct / max(effective_break_pct, 1e-12)),
        "pred_distance_over_break_pct": float(pred_distance_pct / max(effective_break_pct, 1e-12)),
        "true_distance_over_break_pct": float(true_distance_pct / max(effective_break_pct, 1e-12)),
    }


def _enrich_rows(
    prediction_rows: list[dict[str, Any]],
    label_rows: dict[int, dict[str, Any]],
    anchor_closes: dict[tuple[str, int], float],
) -> list[dict[str, Any]]:
    out = []
    for row in prediction_rows:
        index = int(row["index"])
        symbol = str(row["symbol"])
        timestamp = int(row["timestamp"])
        label_row = label_rows[index]
        current_close = anchor_closes[(symbol, timestamp)]
        enriched = {
            **label_row,
            **row,
            "bull_level": label_row.get("bull_level"),
            "bear_level": label_row.get("bear_level"),
            "effective_break_pct": label_row.get("effective_break_pct"),
            **_distance_fields({**row, **label_row}, current_close),
        }
        out.append(enriched)
    return out


def _segment_summary(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    if not rows:
        return {"name": name, "support": 0}
    nearest = np.array([row["nearest_distance_pct"] for row in rows], dtype=np.float64)
    pred_dist = np.array([row["pred_direction_distance_pct"] for row in rows], dtype=np.float64)
    true_dist = np.array([row["true_direction_distance_pct"] for row in rows], dtype=np.float64)
    nearest_norm = np.array([row["nearest_distance_over_break_pct"] for row in rows], dtype=np.float64)
    pred_norm = np.array([row["pred_distance_over_break_pct"] for row in rows], dtype=np.float64)
    true_norm = np.array([row["true_distance_over_break_pct"] for row in rows], dtype=np.float64)
    return {
        "name": name,
        "support": len(rows),
        "error_type_counts": _counts([str(row["error_type"]) for row in rows]),
        "true_label_counts": _counts([str(row["true_label"]) for row in rows]),
        "pred_label_counts": _counts([str(row["pred_label"]) for row in rows]),
        "nearest_distance_pct": _quantiles(nearest),
        "pred_direction_distance_pct": _quantiles(pred_dist),
        "true_direction_distance_pct": _quantiles(true_dist),
        "nearest_distance_over_break_pct": _quantiles(nearest_norm),
        "pred_distance_over_break_pct": _quantiles(pred_norm),
        "true_distance_over_break_pct": _quantiles(true_norm),
        "mean_p_break": _safe_mean(np.array([row["p_break"] for row in rows], dtype=np.float64)),
    }


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
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--label-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.prediction_csv.open() as fh:
        prediction_rows = list(csv.DictReader(fh))
    indices = [int(row["index"]) for row in prediction_rows]
    label_df = pl.read_parquet(args.label_path).filter(pl.col("index").is_in(indices))
    label_rows = {int(row["index"]): row for row in label_df.to_dicts()}
    anchor_closes = _load_anchor_closes(args.dataset_root)
    rows = _enrich_rows(prediction_rows, label_rows, anchor_closes)

    segments = {
        "all": rows,
        "correct": [row for row in rows if row["error_type"] == "correct"],
        "false_positive_break": [row for row in rows if row["error_type"] == "false_positive_break"],
        "false_negative_break": [row for row in rows if row["error_type"] == "false_negative_break"],
        "wrong_direction": [row for row in rows if row["error_type"] == "wrong_direction"],
        "high_confidence": [row for row in rows if float(row["break_confidence"]) >= 0.9],
        "high_confidence_wrong": [
            row for row in rows if float(row["break_confidence"]) >= 0.9 and row["error_type"] != "correct"
        ],
    }
    summary = {
        "prediction_csv": str(args.prediction_csv),
        "label_path": str(args.label_path),
        "dataset_root": str(args.dataset_root),
        "segment_summaries": {name: _segment_summary(segment, name) for name, segment in segments.items()},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_csv(args.output_dir / "predictions_with_proximity.csv", rows)
    _write_csv(args.output_dir / "high_confidence_wrong.csv", segments["high_confidence_wrong"])
    logger.info("Stage 1B proximity audit written to {}", args.output_dir)


if __name__ == "__main__":
    main()
