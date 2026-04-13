"""Sliding window builder: feature DataFrame → fixed-size numpy windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from msb_repr.data.features import FEATURE_COLUMNS, TREND_FEATURE_COLUMNS


@dataclass
class WindowConfig:
    window_size: int = 180    # bars per window: 30 days × 6 bars/day (4h timeframe)
    step_size: int = 6        # bars per step: 1 day at 4h timeframe
    min_coverage: float = 0.95  # drop windows with too many NaNs


def build_windows(
    df: pl.DataFrame,
    symbol: str,
    cfg: WindowConfig,
    output_path: Path,
    feature_columns: list[str] | None = None,
) -> int:
    """Slice a feature-enriched DataFrame into overlapping windows and save.

    Output: npz file with:
        - windows: float32 array of shape (N, num_features, window_size)
        - timestamps: int64 array of window-end Unix timestamps (seconds)

    Args:
        feature_columns: Which feature columns to include. Defaults to FEATURE_COLUMNS
                         (short window, 16 channels). Pass TREND_FEATURE_COLUMNS for
                         long trend windows (5 channels).

    Returns:
        Number of windows written.
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    df = df.sort("timestamp")
    data = df.select(feature_columns).to_numpy().T  # (num_features, T)
    # Normalize to Unix seconds using Polars native epoch conversion (handles ms/μs/ns)
    timestamps = df["timestamp"].dt.epoch(time_unit="s").to_numpy()

    W = cfg.window_size
    S = cfg.step_size
    T = data.shape[1]

    windows = []
    window_timestamps = []

    for start in range(0, T - W + 1, S):
        end = start + W
        window = data[:, start:end]  # (num_features, W)
        coverage = np.isfinite(window).mean()
        if coverage < cfg.min_coverage:
            continue
        window = np.where(np.isfinite(window), window, 0.0)
        windows.append(window)
        window_timestamps.append(int(timestamps[end - 1]))

    if not windows:
        logger.warning(f"No valid windows for {symbol}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        windows=np.stack(windows).astype(np.float32),
        timestamps=np.array(window_timestamps, dtype=np.int64),
    )
    logger.info(f"{symbol}: {len(windows)} windows → {output_path}")
    return len(windows)
