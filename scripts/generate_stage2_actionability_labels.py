"""Generate rule-based Stage 2 actionability labels from Stage 1B predictions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger


LABEL_TO_ID = {
    "no_trade": 0,
    "wait_for_break": 1,
    "wait_for_retest": 2,
    "actionable_break_candidate": 3,
}


@dataclass(frozen=True)
class SymbolBars:
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    timestamp_to_pos: dict[int, int]


def _parse_float(value: Any, default: float = np.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _load_symbol_bars(raw_data_root: Path, symbol: str) -> SymbolBars:
    path = raw_data_root / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Raw OHLCV file not found for {symbol}: {path}")
    df = pl.read_parquet(path).sort("timestamp")
    timestamps = (df["timestamp"].dt.epoch("s")).to_numpy().astype(np.int64)
    return SymbolBars(
        timestamps=timestamps,
        open=df["open"].to_numpy().astype(np.float64),
        high=df["high"].to_numpy().astype(np.float64),
        low=df["low"].to_numpy().astype(np.float64),
        close=df["close"].to_numpy().astype(np.float64),
        timestamp_to_pos={int(ts): int(i) for i, ts in enumerate(timestamps)},
    )


def _distance_bucket(side: str, current_price: float, trigger_level: float, near: float, watch: float) -> str:
    if side == "long" and current_price >= trigger_level:
        return "at_or_beyond_trigger"
    if side == "short" and current_price <= trigger_level:
        return "at_or_beyond_trigger"
    distance = abs(trigger_level - current_price) / max(abs(current_price), 1e-12)
    if distance <= near:
        return "near_trigger"
    if distance <= watch:
        return "watchable"
    return "far"


def _first_expected_break(
    *,
    side: str,
    closes: np.ndarray,
    trigger_level: float,
    effective_break_pct: float,
    horizon: int,
) -> tuple[bool, int, float]:
    for offset, close in enumerate(closes[:horizon], start=1):
        if side == "long" and close > trigger_level * (1.0 + effective_break_pct):
            return True, offset, float(close)
        if side == "short" and close < trigger_level * (1.0 - effective_break_pct):
            return True, offset, float(close)
    return False, -1, np.nan


def _first_target_time(side: str, highs: np.ndarray, lows: np.ndarray, target_level: float) -> int:
    for offset, (high, low) in enumerate(zip(highs, lows, strict=True), start=1):
        if side == "long" and high >= target_level:
            return offset
        if side == "short" and low <= target_level:
            return offset
    return -1


def _first_invalidation_time(side: str, highs: np.ndarray, lows: np.ndarray, invalidation_level: float) -> int:
    for offset, (high, low) in enumerate(zip(highs, lows, strict=True), start=1):
        if side == "long" and low <= invalidation_level:
            return offset
        if side == "short" and high >= invalidation_level:
            return offset
    return -1


def _first_retest_time(
    *,
    side: str,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    trigger_level: float,
    break_time: int,
    tolerance: float,
) -> int:
    if break_time < 1:
        return -1
    for offset in range(break_time + 1, len(closes) + 1):
        high = highs[offset - 1]
        low = lows[offset - 1]
        close = closes[offset - 1]
        if side == "long" and low <= trigger_level * (1.0 + tolerance) and close >= trigger_level:
            return offset
        if side == "short" and high >= trigger_level * (1.0 - tolerance) and close <= trigger_level:
            return offset
    return -1


def _safe_before(first: int, second: int) -> bool:
    return first >= 0 and (second < 0 or first < second)


def _label_row(
    row: dict[str, Any],
    bars: SymbolBars,
    *,
    decision_horizon: int,
    observation_horizon: int,
    short_window: int,
    min_confidence: float,
    near_threshold: float,
    watch_threshold: float,
    min_mfe_r: float,
    retest_tolerance: float,
    large_displacement: float,
) -> dict[str, Any]:
    symbol = str(row["symbol"])
    timestamp = int(row["timestamp"])
    pos = bars.timestamp_to_pos.get(timestamp)
    if pos is None:
        raise ValueError(f"Anchor timestamp not found in raw bars: {symbol} {timestamp}")

    side = {"bullish": "long", "bearish": "short"}.get(str(row.get("pred_label")), "none")
    current_price = _parse_float(row.get("current_close"))
    p_break = _parse_float(row.get("p_break"))
    break_confidence = _parse_float(row.get("break_confidence"))
    direction_confidence = _parse_float(row.get("direction_confidence"))
    bull_level = _parse_float(row.get("bull_level"))
    bear_level = _parse_float(row.get("bear_level"))
    effective_break_pct = _parse_float(row.get("effective_break_pct"), default=0.0)

    future_start = pos + 1
    future_end = min(pos + observation_horizon + 1, len(bars.close))
    future_highs = bars.high[future_start:future_end]
    future_lows = bars.low[future_start:future_end]
    future_closes = bars.close[future_start:future_end]

    base = {
        "index": int(row["index"]),
        "symbol": symbol,
        "timestamp": timestamp,
        "side": side,
        "stage1b_pred_label": str(row.get("pred_label", "")),
        "stage1b_true_label": str(row.get("true_label", "")),
        "stage1b_error_type": str(row.get("error_type", "")),
        "p_break": p_break,
        "break_confidence": break_confidence,
        "direction_confidence": direction_confidence,
        "current_price": current_price,
        "bull_level": bull_level,
        "bear_level": bear_level,
        "effective_break_pct": effective_break_pct,
        "distance_bucket": "not_applicable",
    }

    if side == "none":
        return _finalize(base, "no_trade", "stage1b_no_break")
    if len(future_closes) < observation_horizon:
        return _finalize(base, "no_trade", "insufficient_future")

    trigger_level = bull_level if side == "long" else bear_level
    opposite_level = bear_level if side == "long" else bull_level
    if not np.isfinite(trigger_level) or trigger_level <= 0:
        return _finalize(base, "no_trade", "missing_trigger_level")

    distance_to_trigger = abs(trigger_level - current_price) / max(abs(current_price), 1e-12)
    distance_bucket = _distance_bucket(side, current_price, trigger_level, near_threshold, watch_threshold)

    past_start = max(0, pos - short_window + 1)
    past_lows = bars.low[past_start : pos + 1]
    past_highs = bars.high[past_start : pos + 1]
    if side == "long":
        primary_invalidation = float(np.min(past_lows)) if len(past_lows) else np.nan
        fallback_invalidation = opposite_level
        invalidation_level = primary_invalidation if primary_invalidation < trigger_level else fallback_invalidation
        risk = trigger_level - invalidation_level
        mfe = max(0.0, float(np.max(future_highs) - trigger_level))
        mae = max(0.0, float(trigger_level - np.min(future_lows)))
        target_level = trigger_level + min_mfe_r * risk
    else:
        primary_invalidation = float(np.max(past_highs)) if len(past_highs) else np.nan
        fallback_invalidation = opposite_level
        invalidation_level = primary_invalidation if primary_invalidation > trigger_level else fallback_invalidation
        risk = invalidation_level - trigger_level
        mfe = max(0.0, float(trigger_level - np.min(future_lows)))
        mae = max(0.0, float(np.max(future_highs) - trigger_level))
        target_level = trigger_level - min_mfe_r * risk

    if not np.isfinite(invalidation_level):
        return _finalize(base, "no_trade", "missing_invalidation", trigger_level=trigger_level)

    mfe_r = mfe / max(risk, 1e-12)
    mae_r = mae / max(risk, 1e-12)
    break_h8, break_time_h8, break_price_h8 = _first_expected_break(
        side=side,
        closes=future_closes,
        trigger_level=trigger_level,
        effective_break_pct=effective_break_pct,
        horizon=decision_horizon,
    )
    break_h16, break_time_h16, break_price_h16 = _first_expected_break(
        side=side,
        closes=future_closes,
        trigger_level=trigger_level,
        effective_break_pct=effective_break_pct,
        horizon=observation_horizon,
    )
    break_time = break_time_h8 if break_h8 else break_time_h16
    break_price = break_price_h8 if break_h8 else break_price_h16
    first_target_time = _first_target_time(side, future_highs, future_lows, target_level)
    first_invalidation_time = _first_invalidation_time(side, future_highs, future_lows, invalidation_level)
    first_retest_time = _first_retest_time(
        side=side,
        highs=future_highs,
        lows=future_lows,
        closes=future_closes,
        trigger_level=trigger_level,
        break_time=break_time,
        tolerance=retest_tolerance,
    )
    break_displacement = abs(break_price - trigger_level) / max(abs(trigger_level), 1e-12) if break_h16 else np.nan

    features = {
        "trigger_level": float(trigger_level),
        "opposite_level": float(opposite_level),
        "primary_invalidation_level": float(primary_invalidation),
        "invalidation_level": float(invalidation_level),
        "target_level": float(target_level),
        "risk": float(risk),
        "distance_to_trigger": float(distance_to_trigger),
        "distance_bucket": distance_bucket,
        "break_in_h8": bool(break_h8),
        "break_in_h16": bool(break_h16),
        "break_time_bars": int(break_time),
        "break_price": float(break_price),
        "first_target_time": int(first_target_time),
        "first_invalidation_time": int(first_invalidation_time),
        "first_retest_time": int(first_retest_time),
        "break_displacement": float(break_displacement),
        "mfe": float(mfe),
        "mae": float(mae),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
    }
    out = {**base, **features}

    if break_confidence < min_confidence:
        return _finalize(out, "no_trade", "low_confidence")
    if risk <= 0 or not np.isfinite(risk):
        return _finalize(out, "no_trade", "invalid_risk")
    if distance_bucket == "far":
        return _finalize(out, "no_trade", "too_far_from_trigger")
    if not break_h16:
        return _finalize(out, "no_trade", "no_break_in_observation_horizon")
    if _safe_before(first_invalidation_time, break_time):
        return _finalize(out, "no_trade", "invalidated_before_break")

    target_before_invalidation = _safe_before(first_target_time, first_invalidation_time)
    retest_observed = first_retest_time >= 0
    large_break_displacement = bool(np.isfinite(break_displacement) and break_displacement >= large_displacement)

    if (
        distance_bucket in {"at_or_beyond_trigger", "near_trigger"}
        and break_h8
        and target_before_invalidation
        and mfe_r >= min_mfe_r
    ):
        return _finalize(out, "actionable_break_candidate", "target_before_invalidation")

    if (
        break_h8
        and (distance_bucket == "at_or_beyond_trigger" or large_break_displacement)
        and retest_observed
        and not _safe_before(first_target_time, first_retest_time)
    ):
        return _finalize(out, "wait_for_retest", "retest_after_break")

    if distance_bucket in {"near_trigger", "watchable"} and break_h16 and not _safe_before(first_invalidation_time, break_time):
        return _finalize(out, "wait_for_break", "break_expected_within_observation")

    return _finalize(out, "no_trade", "fallback_no_actionability")


def _finalize(row: dict[str, Any], label_name: str, reason_code: str, **extra: Any) -> dict[str, Any]:
    out = {**row, **extra}
    out["label"] = LABEL_TO_ID[label_name]
    out["label_name"] = label_name
    out["reason_code"] = reason_code
    return out


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    values = [str(row.get(key, "")) for row in rows]
    unique, counts = np.unique(np.asarray(values), return_counts=True)
    return {str(value): int(count) for value, count in zip(unique, counts, strict=True)}


def _summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    high_conf = [row for row in rows if _parse_float(row.get("break_confidence")) >= 0.9]
    actionable = [row for row in rows if row["label_name"] == "actionable_break_candidate"]
    return {
        "prediction_csv": str(args.prediction_csv),
        "raw_data_root": str(args.raw_data_root),
        "decision_horizon": args.decision_horizon,
        "observation_horizon": args.observation_horizon,
        "min_confidence": args.min_confidence,
        "near_threshold": args.near_threshold,
        "watch_threshold": args.watch_threshold,
        "min_mfe_r": args.min_mfe_r,
        "retest_tolerance": args.retest_tolerance,
        "large_displacement": args.large_displacement,
        "support": len(rows),
        "label_counts": _counts(rows, "label_name"),
        "reason_code_counts": _counts(rows, "reason_code"),
        "symbol_counts": _counts(rows, "symbol"),
        "distance_bucket_counts": _counts(rows, "distance_bucket"),
        "high_confidence_label_counts": _counts(high_conf, "label_name"),
        "actionable_support": len(actionable),
    }


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
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--raw-data-root", type=Path, default=Path("data/raw/binance"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decision-horizon", type=int, default=8)
    parser.add_argument("--observation-horizon", type=int, default=16)
    parser.add_argument("--short-window", type=int, default=48)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--near-threshold", type=float, default=0.005)
    parser.add_argument("--watch-threshold", type=float, default=0.015)
    parser.add_argument("--min-mfe-r", type=float, default=1.0)
    parser.add_argument("--retest-tolerance", type=float, default=0.0025)
    parser.add_argument("--large-displacement", type=float, default=0.005)
    args = parser.parse_args()

    prediction_rows = _load_prediction_rows(args.prediction_csv)
    symbols = sorted({str(row["symbol"]) for row in prediction_rows})
    bars_by_symbol = {symbol: _load_symbol_bars(args.raw_data_root, symbol) for symbol in symbols}

    rows = [
        _label_row(
            row,
            bars_by_symbol[str(row["symbol"])],
            decision_horizon=args.decision_horizon,
            observation_horizon=args.observation_horizon,
            short_window=args.short_window,
            min_confidence=args.min_confidence,
            near_threshold=args.near_threshold,
            watch_threshold=args.watch_threshold,
            min_mfe_r=args.min_mfe_r,
            retest_tolerance=args.retest_tolerance,
            large_displacement=args.large_displacement,
        )
        for row in prediction_rows
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "stage2_actionability_labels.csv", rows)
    (args.output_dir / "summary.json").write_text(json.dumps(_summary(rows, args), indent=2))
    logger.info("Stage 2 actionability labels written to {}", args.output_dir)


if __name__ == "__main__":
    main()
