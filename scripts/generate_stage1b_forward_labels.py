"""Generate analysis-only Stage 1B forward-structure labels from Stage 1A exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from msb_repr.stage1a.analysis import load_latent_export
from msb_repr.stage1a.config import Stage1ADatasetSpec
from msb_repr.stage1a.features import STAGE1A_SHORT_FEATURE_COLUMNS
from msb_repr.stage1a.labels import find_last_pivot_high, find_last_pivot_low

LABEL_NAMES = {0: "intact", 1: "bullish", 2: "bearish"}


def _first_confirm_bar(excess: np.ndarray, min_count: int) -> int | None:
    hits = np.flatnonzero(excess > 0.0)
    if len(hits) < min_count:
        return None
    return int(hits[min_count - 1] + 1)


def _direction_label(
    bull_bar: int | None,
    bear_bar: int | None,
    ambiguity_bars: int,
) -> tuple[str, int]:
    if bull_bar is None and bear_bar is None:
        return "none", -1
    if bull_bar is not None and bear_bar is None:
        return "bullish", bull_bar
    if bear_bar is not None and bull_bar is None:
        return "bearish", bear_bar
    assert bull_bar is not None and bear_bar is not None
    if abs(bull_bar - bear_bar) <= ambiguity_bars:
        return "ambiguous", min(bull_bar, bear_bar)
    if bull_bar < bear_bar:
        return "bullish", bull_bar
    return "bearish", bear_bar


def _levels_from_short_window(
    short_window: np.ndarray,
    label_config: Any,
) -> tuple[float | None, float | None, float]:
    feature_index = {name: idx for idx, name in enumerate(STAGE1A_SHORT_FEATURE_COLUMNS)}
    highs = short_window[feature_index["high"]]
    lows = short_window[feature_index["low"]]
    closes = short_window[feature_index["close"]]

    struct_end = int(label_config.structure_bars)
    struct_highs = highs[:struct_end]
    struct_lows = lows[:struct_end]
    atr = float(np.mean(struct_highs - struct_lows)) if len(struct_highs) else 0.0
    ref_close = float(closes[struct_end - 1]) if closes[struct_end - 1] > 0 else 1.0
    effective_break_pct = max(
        float(label_config.min_break_pct),
        float(label_config.atr_factor) * atr / ref_close,
    )
    swing_high = find_last_pivot_high(struct_highs, int(label_config.pivot_n))
    swing_low = find_last_pivot_low(struct_lows, int(label_config.pivot_n))
    bull_level = None if swing_high is None else float(swing_high * (1.0 + effective_break_pct))
    bear_level = None if swing_low is None else float(swing_low * (1.0 - effective_break_pct))
    return bull_level, bear_level, float(effective_break_pct)


def _load_raw_symbol(raw_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pl.read_parquet(raw_path).sort("timestamp")
    timestamps = df["timestamp"].dt.epoch(time_unit="s").to_numpy().astype(np.int64)
    closes = df["close"].to_numpy().astype(np.float32)
    return timestamps, closes


def _symbol_dataset_path(dataset_root: Path, symbol: str) -> Path:
    return dataset_root / "symbols" / f"{symbol}.npz"


def _raw_symbol_path(data_root: Path, exchange: str, symbol: str) -> Path:
    return data_root / "raw" / exchange / f"{symbol}.parquet"


def _build_rows(
    arrays: dict[str, np.ndarray],
    dataset_root: Path,
    data_root: Path,
    spec: Stage1ADatasetSpec,
    horizons: list[int],
    ambiguity_bars: int,
) -> list[dict[str, object]]:
    symbols = arrays["symbols"].astype(str)
    timestamps = arrays["timestamps"].astype(np.int64)
    rows_by_index: dict[int, dict[str, object]] = {}

    for symbol in sorted(set(symbols.tolist())):
        symbol_indices = np.flatnonzero(symbols == symbol)
        ds_path = _symbol_dataset_path(dataset_root, symbol)
        raw_path = _raw_symbol_path(data_root, spec.exchange, symbol)
        if not ds_path.exists():
            raise FileNotFoundError(f"Missing Stage 1A symbol dataset: {ds_path}")
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw OHLCV parquet: {raw_path}")

        ds = np.load(ds_path)
        # Materialize compressed NPZ arrays once. Indexing directly into NpzFile
        # inside the sample loop repeatedly decompresses data and is very slow.
        ds_timestamps = ds["timestamps"].astype(np.int64)
        ds_short_windows = ds["short_windows"].astype(np.float32)
        ds_ts = ds_timestamps
        ds_ts_to_idx = {int(ts): idx for idx, ts in enumerate(ds_ts)}
        raw_ts, raw_closes = _load_raw_symbol(raw_path)
        raw_ts_to_idx = {int(ts): idx for idx, ts in enumerate(raw_ts)}

        for export_idx in symbol_indices:
            ts = int(timestamps[export_idx])
            if ts not in ds_ts_to_idx:
                raise ValueError(f"{symbol}: export timestamp not found in dataset: {ts}")
            if ts not in raw_ts_to_idx:
                raise ValueError(f"{symbol}: export timestamp not found in raw parquet: {ts}")

            short_window = ds_short_windows[ds_ts_to_idx[ts]]
            bull_level, bear_level, effective_break_pct = _levels_from_short_window(short_window, spec.labels)
            raw_idx = raw_ts_to_idx[ts]
            row: dict[str, object] = {
                "index": int(export_idx),
                "symbol": symbol,
                "timestamp": ts,
                "current_label": LABEL_NAMES[int(arrays["labels"][export_idx])],
                "has_bull_level": bull_level is not None,
                "has_bear_level": bear_level is not None,
                "bull_level": None if bull_level is None else float(bull_level),
                "bear_level": None if bear_level is None else float(bear_level),
                "effective_break_pct": effective_break_pct,
            }

            for horizon in horizons:
                future = raw_closes[raw_idx + 1: raw_idx + 1 + horizon]
                prefix = f"h{horizon}"
                if len(future) < horizon:
                    row[f"{prefix}_future_break_direction"] = "insufficient_future"
                    row[f"{prefix}_future_break_occurs"] = False
                    row[f"{prefix}_time_to_break"] = -1
                    row[f"{prefix}_bull_confirm_bar"] = -1
                    row[f"{prefix}_bear_confirm_bar"] = -1
                    row[f"{prefix}_bull_close_count"] = 0
                    row[f"{prefix}_bear_close_count"] = 0
                    continue

                bull_excess = np.zeros_like(future, dtype=np.float32)
                bear_excess = np.zeros_like(future, dtype=np.float32)
                if bull_level is not None:
                    bull_excess = future - float(bull_level)
                if bear_level is not None:
                    bear_excess = float(bear_level) - future

                bull_bar = None if bull_level is None else _first_confirm_bar(
                    bull_excess,
                    int(spec.labels.min_recent_break_bars),
                )
                bear_bar = None if bear_level is None else _first_confirm_bar(
                    bear_excess,
                    int(spec.labels.min_recent_break_bars),
                )
                direction, time_to_break = _direction_label(bull_bar, bear_bar, ambiguity_bars)
                row[f"{prefix}_future_break_direction"] = direction
                row[f"{prefix}_future_break_occurs"] = direction in {"bullish", "bearish", "ambiguous"}
                row[f"{prefix}_time_to_break"] = int(time_to_break)
                row[f"{prefix}_bull_confirm_bar"] = -1 if bull_bar is None else int(bull_bar)
                row[f"{prefix}_bear_confirm_bar"] = -1 if bear_bar is None else int(bear_bar)
                row[f"{prefix}_bull_close_count"] = int(np.sum(bull_excess > 0.0))
                row[f"{prefix}_bear_close_count"] = int(np.sum(bear_excess > 0.0))

            rows_by_index[int(export_idx)] = row

    return [rows_by_index[idx] for idx in range(len(arrays["labels"]))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[16, 32, 48])
    parser.add_argument("--ambiguity-bars", type=int, default=1)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    arrays, export_metadata = load_latent_export(args.latent_path)
    spec = Stage1ADatasetSpec.load(args.dataset_root / "spec.json")
    horizons = sorted(set(args.horizons))
    rows = _build_rows(
        arrays=arrays,
        dataset_root=args.dataset_root,
        data_root=args.data_root,
        spec=spec,
        horizons=horizons,
        ambiguity_bars=args.ambiguity_bars,
    )

    output_path = args.output_path or (args.latent_path.with_suffix("") / "stage1b_forward_labels.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(rows)
    frame.write_parquet(output_path)

    summary = {
        "latent_path": str(args.latent_path),
        "dataset_root": str(args.dataset_root),
        "data_root": str(args.data_root),
        "output_path": str(output_path),
        "horizons": horizons,
        "ambiguity_bars": int(args.ambiguity_bars),
        "num_rows": int(frame.height),
        "export_metadata": export_metadata,
        "label_counts": {
            f"h{horizon}_future_break_direction": frame.group_by(f"h{horizon}_future_break_direction")
            .len()
            .sort(f"h{horizon}_future_break_direction")
            .to_dicts()
            for horizon in horizons
        },
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Stage 1B forward labels written to {} (rows={})", output_path, frame.height)


if __name__ == "__main__":
    main()
