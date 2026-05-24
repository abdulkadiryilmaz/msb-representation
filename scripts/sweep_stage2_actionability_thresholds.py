"""Sweep Stage 2 actionability label thresholds and summarize label behavior."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

from loguru import logger

from generate_stage2_actionability_labels import (
    _counts,
    _label_row,
    _load_prediction_rows,
    _load_symbol_bars,
    _parse_float,
)


def _parse_float_list(values: list[str]) -> list[float]:
    out: list[float] = []
    for value in values:
        out.extend(float(part) for part in value.split(",") if part)
    return out


def _fraction(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _row_for_combo(
    rows: list[dict[str, Any]],
    *,
    min_confidence: float,
    near_threshold: float,
    watch_threshold: float,
    min_mfe_r: float,
) -> dict[str, Any]:
    label_counts = _counts(rows, "label_name")
    fp_rows = [row for row in rows if row.get("stage1b_error_type") == "false_positive_break"]
    fp_label_counts = _counts(fp_rows, "label_name")
    actionable_rows = [row for row in rows if row.get("label_name") == "actionable_break_candidate"]
    actionable_error_counts = _counts(actionable_rows, "stage1b_error_type")
    high_conf_rows = [row for row in rows if _parse_float(row.get("break_confidence")) >= 0.9]
    high_conf_label_counts = _counts(high_conf_rows, "label_name")
    return {
        "min_confidence": min_confidence,
        "near_threshold": near_threshold,
        "watch_threshold": watch_threshold,
        "min_mfe_r": min_mfe_r,
        "support": len(rows),
        "no_trade": label_counts.get("no_trade", 0),
        "wait_for_break": label_counts.get("wait_for_break", 0),
        "wait_for_retest": label_counts.get("wait_for_retest", 0),
        "actionable_break_candidate": label_counts.get("actionable_break_candidate", 0),
        "actionable_rate": _fraction(label_counts.get("actionable_break_candidate", 0), len(rows)),
        "fp_total": len(fp_rows),
        "fp_to_no_trade": fp_label_counts.get("no_trade", 0),
        "fp_to_wait_for_break": fp_label_counts.get("wait_for_break", 0),
        "fp_to_wait_for_retest": fp_label_counts.get("wait_for_retest", 0),
        "fp_to_actionable": fp_label_counts.get("actionable_break_candidate", 0),
        "fp_no_trade_rate": _fraction(fp_label_counts.get("no_trade", 0), len(fp_rows)),
        "actionable_correct": actionable_error_counts.get("correct", 0),
        "actionable_false_positive": actionable_error_counts.get("false_positive_break", 0),
        "actionable_wrong_direction": actionable_error_counts.get("wrong_direction", 0),
        "actionable_correct_rate": _fraction(actionable_error_counts.get("correct", 0), len(actionable_rows)),
        "high_conf_total": len(high_conf_rows),
        "high_conf_actionable": high_conf_label_counts.get("actionable_break_candidate", 0),
        "high_conf_wait_for_retest": high_conf_label_counts.get("wait_for_retest", 0),
        "high_conf_no_trade": high_conf_label_counts.get("no_trade", 0),
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
    parser.add_argument("--raw-data-root", type=Path, default=Path("data/raw/binance"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decision-horizon", type=int, default=8)
    parser.add_argument("--observation-horizon", type=int, default=16)
    parser.add_argument("--short-window", type=int, default=48)
    parser.add_argument("--min-confidence", nargs="+", default=["0.60", "0.70", "0.80", "0.90"])
    parser.add_argument("--near-threshold", nargs="+", default=["0.0025", "0.005", "0.01"])
    parser.add_argument("--watch-threshold", nargs="+", default=["0.01", "0.015", "0.02"])
    parser.add_argument("--min-mfe-r", nargs="+", default=["1.0", "1.5", "2.0"])
    parser.add_argument("--retest-tolerance", type=float, default=0.0025)
    parser.add_argument("--large-displacement", type=float, default=0.005)
    args = parser.parse_args()

    prediction_rows = _load_prediction_rows(args.prediction_csv)
    symbols = sorted({str(row["symbol"]) for row in prediction_rows})
    bars_by_symbol = {symbol: _load_symbol_bars(args.raw_data_root, symbol) for symbol in symbols}

    min_confidences = _parse_float_list(args.min_confidence)
    near_thresholds = _parse_float_list(args.near_threshold)
    watch_thresholds = _parse_float_list(args.watch_threshold)
    min_mfe_rs = _parse_float_list(args.min_mfe_r)

    results: list[dict[str, Any]] = []
    for min_confidence, near_threshold, watch_threshold, min_mfe_r in itertools.product(
        min_confidences,
        near_thresholds,
        watch_thresholds,
        min_mfe_rs,
    ):
        if watch_threshold < near_threshold:
            continue
        labeled = [
            _label_row(
                row,
                bars_by_symbol[str(row["symbol"])],
                decision_horizon=args.decision_horizon,
                observation_horizon=args.observation_horizon,
                short_window=args.short_window,
                min_confidence=min_confidence,
                near_threshold=near_threshold,
                watch_threshold=watch_threshold,
                min_mfe_r=min_mfe_r,
                retest_tolerance=args.retest_tolerance,
                large_displacement=args.large_displacement,
            )
            for row in prediction_rows
        ]
        results.append(
            _row_for_combo(
                labeled,
                min_confidence=min_confidence,
                near_threshold=near_threshold,
                watch_threshold=watch_threshold,
                min_mfe_r=min_mfe_r,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "sweep_results.csv", results)
    ranked = sorted(
        results,
        key=lambda row: (
            row["fp_to_actionable"],
            -row["actionable_correct_rate"],
            -row["actionable_break_candidate"],
        ),
    )
    summary = {
        "prediction_csv": str(args.prediction_csv),
        "raw_data_root": str(args.raw_data_root),
        "decision_horizon": args.decision_horizon,
        "observation_horizon": args.observation_horizon,
        "num_combinations": len(results),
        "top_by_low_fp_actionable": ranked[:10],
        "default_like": [
            row
            for row in results
            if row["min_confidence"] == 0.7
            and row["near_threshold"] == 0.005
            and row["watch_threshold"] == 0.015
            and row["min_mfe_r"] == 1.0
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Stage 2 actionability sweep written to {}", args.output_dir)


if __name__ == "__main__":
    main()
