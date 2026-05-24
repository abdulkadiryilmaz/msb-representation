"""Audit Stage 1B labels/predictions for fresh-break vs already-broken semantics."""

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


def _load_anchor_closes(dataset_root: Path) -> dict[tuple[str, int], float]:
    close_idx = STAGE1A_SHORT_FEATURE_COLUMNS.index("close")
    closes_by_key: dict[tuple[str, int], float] = {}
    for path in sorted((dataset_root / "symbols").glob("*.npz")):
        symbol = path.stem
        arrays = np.load(path)
        timestamps = arrays["timestamps"].astype(np.int64)
        closes = arrays["short_windows"][:, close_idx, -1].astype(np.float64)
        closes_by_key.update(
            {
                (symbol, int(timestamp)): float(close)
                for timestamp, close in zip(timestamps, closes, strict=True)
            }
        )
    return closes_by_key


def _side_status(label: str, current_close: float, bull_level: float | None, bear_level: float | None) -> str:
    if label == "bullish":
        if bull_level is None or not np.isfinite(bull_level):
            return "missing_level"
        return "already_broken" if current_close >= bull_level else "fresh_candidate"
    if label == "bearish":
        if bear_level is None or not np.isfinite(bear_level):
            return "missing_level"
        return "already_broken" if current_close <= bear_level else "fresh_candidate"
    return "not_directional"


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    values = [str(row.get(key, "")) for row in rows]
    if not values:
        return {}
    unique, counts = np.unique(np.asarray(values), return_counts=True)
    return {str(value): int(count) for value, count in zip(unique, counts, strict=True)}


def _directional_summary(rows: list[dict[str, Any]], label_key: str, status_key: str) -> dict[str, Any]:
    directional = [row for row in rows if row.get(label_key) in {"bullish", "bearish"}]
    already = [row for row in directional if row.get(status_key) == "already_broken"]
    fresh = [row for row in directional if row.get(status_key) == "fresh_candidate"]
    return {
        "support": len(rows),
        "directional_support": len(directional),
        "label_counts": _counts(rows, label_key),
        "status_counts": _counts(rows, status_key),
        "directional_status_counts": _counts(directional, status_key),
        "already_broken_rate_directional": float(len(already) / len(directional)) if directional else 0.0,
        "fresh_candidate_rate_directional": float(len(fresh) / len(directional)) if directional else 0.0,
        "already_broken_by_label": {
            label: _counts([row for row in directional if row.get(label_key) == label], status_key)
            for label in ["bullish", "bearish"]
        },
    }


def _label_rows(label_path: Path, dataset_root: Path, horizon_prefix: str) -> list[dict[str, Any]]:
    label_df = pl.read_parquet(label_path)
    anchor_closes = _load_anchor_closes(dataset_root)
    rows: list[dict[str, Any]] = []
    target_col = f"{horizon_prefix}_future_break_direction"
    for row in label_df.to_dicts():
        symbol = str(row["symbol"])
        timestamp = int(row["timestamp"])
        label = str(row[target_col])
        current_close = anchor_closes[(symbol, timestamp)]
        bull_level = row.get("bull_level")
        bear_level = row.get("bear_level")
        status = _side_status(
            label,
            current_close,
            None if bull_level is None else float(bull_level),
            None if bear_level is None else float(bear_level),
        )
        rows.append(
            {
                "index": int(row["index"]),
                "symbol": symbol,
                "timestamp": timestamp,
                "label": label,
                "current_close": current_close,
                "bull_level": bull_level,
                "bear_level": bear_level,
                "anchor_break_status": status,
            }
        )
    return rows


def _prediction_rows(prediction_csv: Path) -> list[dict[str, Any]]:
    with prediction_csv.open() as fh:
        reader = csv.DictReader(fh)
        rows: list[dict[str, Any]] = []
        for row in reader:
            pred_label = str(row.get("pred_label", ""))
            current_close = float(row["current_close"])
            bull_level = _maybe_float(row.get("bull_level"))
            bear_level = _maybe_float(row.get("bear_level"))
            rows.append(
                {
                    **row,
                    "label": pred_label,
                    "anchor_break_status": _side_status(pred_label, current_close, bull_level, bear_level),
                }
            )
    return rows


def _maybe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--horizon-prefix", type=str, default="h8")
    parser.add_argument("--prediction-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = _label_rows(args.label_path, args.dataset_root, args.horizon_prefix)
    summary: dict[str, Any] = {
        "label_path": str(args.label_path),
        "dataset_root": str(args.dataset_root),
        "horizon_prefix": args.horizon_prefix,
        "label_semantics": _directional_summary(labels, "label", "anchor_break_status"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "label_anchor_break_status.csv", labels)

    if args.prediction_csv:
        preds = _prediction_rows(args.prediction_csv)
        summary["prediction_csv"] = str(args.prediction_csv)
        summary["prediction_semantics"] = _directional_summary(preds, "label", "anchor_break_status")
        _write_csv(args.output_dir / "prediction_anchor_break_status.csv", preds)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Stage 1B fresh-break semantics audit written to {}", args.output_dir)


if __name__ == "__main__":
    main()
