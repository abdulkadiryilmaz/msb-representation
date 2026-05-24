"""Audit rule-based Stage 2 actionability labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return np.nan


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    values = [str(row.get(key, "")) for row in rows]
    if not values:
        return {}
    unique, counts = np.unique(np.asarray(values), return_counts=True)
    return {str(value): int(count) for value, count in zip(unique, counts, strict=True)}


def _quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "max": 0.0}
    return {
        "min": float(np.min(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def _class_summary(rows: list[dict[str, Any]], label_name: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("label_name") == label_name]
    return {
        "support": len(selected),
        "symbol_counts": _counts(selected, "symbol"),
        "reason_code_counts": _counts(selected, "reason_code"),
        "distance_bucket_counts": _counts(selected, "distance_bucket"),
        "stage1b_error_type_counts": _counts(selected, "stage1b_error_type"),
        "break_confidence": _quantiles([_float(row, "break_confidence") for row in selected]),
        "distance_to_trigger": _quantiles([_float(row, "distance_to_trigger") for row in selected]),
        "mfe_r": _quantiles([_float(row, "mfe_r") for row in selected]),
        "mae_r": _quantiles([_float(row, "mae_r") for row in selected]),
        "break_time_bars": _quantiles([_float(row, "break_time_bars") for row in selected]),
    }


def _confidence_buckets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = [(0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    out = []
    for lo, hi in buckets:
        selected = [row for row in rows if lo <= _float(row, "break_confidence") < hi]
        out.append(
            {
                "break_confidence_min": lo,
                "break_confidence_max": min(hi, 1.0),
                "support": len(selected),
                "label_counts": _counts(selected, "label_name"),
                "reason_code_counts": _counts(selected, "reason_code"),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=100)
    args = parser.parse_args()

    rows = _load_rows(args.label_csv)
    labels = ["no_trade", "wait_for_break", "wait_for_retest", "actionable_break_candidate"]
    false_positive_stage1b = [row for row in rows if row.get("stage1b_error_type") == "false_positive_break"]
    high_confidence = [row for row in rows if _float(row, "break_confidence") >= 0.9]
    actionable = [row for row in rows if row.get("label_name") == "actionable_break_candidate"]
    summary = {
        "label_csv": str(args.label_csv),
        "support": len(rows),
        "label_counts": _counts(rows, "label_name"),
        "reason_code_counts": _counts(rows, "reason_code"),
        "symbol_counts": _counts(rows, "symbol"),
        "distance_bucket_counts": _counts(rows, "distance_bucket"),
        "class_summaries": {label: _class_summary(rows, label) for label in labels},
        "confidence_buckets": _confidence_buckets(rows),
        "stage1b_false_positive_label_counts": _counts(false_positive_stage1b, "label_name"),
        "high_confidence_label_counts": _counts(high_confidence, "label_name"),
        "actionable_stage1b_error_type_counts": _counts(actionable, "stage1b_error_type"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_csv(
        args.output_dir / "sample_actionable.csv",
        sorted(actionable, key=lambda row: _float(row, "break_confidence"), reverse=True)[: args.sample_limit],
    )
    _write_csv(
        args.output_dir / "sample_wait_for_retest.csv",
        [row for row in rows if row.get("label_name") == "wait_for_retest"][: args.sample_limit],
    )
    logger.info("Stage 2 actionability label audit written to {}", args.output_dir)


if __name__ == "__main__":
    main()
