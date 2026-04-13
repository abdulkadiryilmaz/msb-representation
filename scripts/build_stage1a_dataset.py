"""Build the Stage 1A dual-window dataset.

Output structure:
    data/stage1a/<exchange>/<timeframe>/
      spec.json
      symbols/
        BTC_USDT_15m.npz
        XRP_USDT_15m.npz

Each symbol NPZ contains:
    - short_windows: (N, F_short, 48)
    - long_windows:  (N, F_long, 288)
    - labels:        (N,)
    - timestamps:    (N,)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from msb_repr.data.features import compute_features
from msb_repr.data.windows import WindowConfig, build_windows
from msb_repr.stage1a.config import (
    Stage1ADatasetSpec,
    Stage1ALabelConfig,
    Stage1ASplitConfig,
    Stage1AWindowConfig,
)
from msb_repr.stage1a.dataset import align_windows_by_timestamp
from msb_repr.stage1a.features import STAGE1A_LONG_FEATURE_COLUMNS, STAGE1A_SHORT_FEATURE_COLUMNS
from msb_repr.stage1a.labels import label_state_window
from msb_repr.stage1a.profiles import STAGE1A_DATASET_PROFILES, get_stage1a_profile


def _build_label_windows(df: pl.DataFrame, cfg: WindowConfig) -> tuple[np.ndarray, np.ndarray]:
    data = df.select(["high", "low", "close"]).to_numpy().T
    timestamps = df["timestamp"].dt.epoch(time_unit="s").to_numpy().astype(np.int64)
    windows = []
    window_timestamps = []
    for start in range(0, data.shape[1] - cfg.window_size + 1, cfg.step_size):
        end = start + cfg.window_size
        windows.append(data[:, start:end])
        window_timestamps.append(int(timestamps[end - 1]))
    if not windows:
        return np.empty((0, 3, cfg.window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(windows).astype(np.float32), np.array(window_timestamps, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aligned Stage 1A symbol datasets")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--bars-per-day", type=int, default=96)
    parser.add_argument("--short-window-size", type=int, default=48)
    parser.add_argument("--long-window-size", type=int, default=288)
    parser.add_argument("--step-size", type=int, default=4)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument(
        "--dataset-profile",
        choices=sorted(STAGE1A_DATASET_PROFILES),
        default=None,
        help="Named Stage 1A symbol allowlist. Preferred over repeating --symbols in every run.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Explicit symbol stems, e.g. BTC_USDT_15m XRP_USDT_15m. "
             "Use when you need a one-off fixed allowlist instead of a named profile.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    raw_dir = args.data_root / "raw" / args.exchange
    out_root = args.output_root or (args.data_root / "stage1a" / args.exchange / args.timeframe)
    symbol_dir = out_root / "symbols"
    symbol_dir.mkdir(parents=True, exist_ok=True)

    short_cfg = WindowConfig(
        window_size=args.short_window_size,
        step_size=args.step_size,
        min_coverage=args.min_coverage,
    )
    long_cfg = WindowConfig(
        window_size=args.long_window_size,
        step_size=args.step_size,
        min_coverage=args.min_coverage,
    )

    if bool(args.dataset_profile) == bool(args.symbols):
        raise ValueError("Pass exactly one of --dataset-profile or --symbols.")

    if args.dataset_profile:
        requested_symbols = get_stage1a_profile(args.dataset_profile)
        logger.info("Using Stage 1A dataset profile '{}' -> {}", args.dataset_profile, requested_symbols)
    else:
        requested_symbols = list(dict.fromkeys(args.symbols))

    parquet_files = [raw_dir / f"{symbol}.parquet" for symbol in requested_symbols]
    missing = [p.stem for p in parquet_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw parquet files for requested symbols: {missing} under {raw_dir}"
        )

    built_symbols: list[str] = []
    for pf in parquet_files:
        df = compute_features(pl.read_parquet(pf), bars_per_day=args.bars_per_day)
        symbol = pf.stem

        tmp_short = out_root / ".tmp_short.npz"
        tmp_long = out_root / ".tmp_long.npz"
        build_windows(
            df=df,
            symbol=symbol,
            cfg=short_cfg,
            output_path=tmp_short,
            feature_columns=STAGE1A_SHORT_FEATURE_COLUMNS,
        )
        build_windows(
            df=df,
            symbol=symbol,
            cfg=long_cfg,
            output_path=tmp_long,
            feature_columns=STAGE1A_LONG_FEATURE_COLUMNS,
        )
        short_npz = np.load(tmp_short)
        long_npz = np.load(tmp_long)
        label_windows, label_timestamps = _build_label_windows(df, short_cfg)

        short_windows, long_windows, timestamps = align_windows_by_timestamp(
            short_npz["windows"],
            short_npz["timestamps"],
            long_npz["windows"],
            long_npz["timestamps"],
        )
        short_windows, label_aligned, label_ts = align_windows_by_timestamp(
            short_windows,
            timestamps,
            label_windows,
            label_timestamps,
        )
        _, long_windows, _ = align_windows_by_timestamp(
            label_aligned,
            label_ts,
            long_windows,
            timestamps,
        )
        if len(short_windows) == 0 or len(label_aligned) == 0:
            logger.warning(f"{symbol}: no aligned Stage 1A windows found")
            continue

        labels = np.array(
            [
                label_state_window(
                    highs=window[0],
                    lows=window[1],
                    closes=window[2],
                )
                for window in label_aligned
            ],
            dtype=np.int64,
        )

        out_path = symbol_dir / f"{symbol}.npz"
        np.savez_compressed(
            out_path,
            short_windows=short_windows.astype(np.float32),
            long_windows=long_windows.astype(np.float32),
            labels=labels,
            timestamps=label_ts.astype(np.int64),
        )
        built_symbols.append(symbol)
        logger.info(f"{symbol}: {len(labels)} aligned Stage 1A samples -> {out_path}")

        tmp_short.unlink(missing_ok=True)
        tmp_long.unlink(missing_ok=True)

    spec = Stage1ADatasetSpec(
        exchange=args.exchange,
        timeframe=args.timeframe,
        symbols=built_symbols,
        windows=Stage1AWindowConfig(
            short_window_size=args.short_window_size,
            long_window_size=args.long_window_size,
            step_size=args.step_size,
            min_coverage=args.min_coverage,
        ),
        labels=Stage1ALabelConfig(),
        splits=Stage1ASplitConfig(train_end=args.train_end, val_end=args.val_end)
        if args.train_end and args.val_end
        else None,
    )
    spec.save(out_root / "spec.json")
    if args.dataset_profile:
        (out_root / "dataset_profile.txt").write_text(f"{args.dataset_profile}\n")
    logger.info(f"Stage 1A spec written -> {out_root / 'spec.json'}")


if __name__ == "__main__":
    main()
