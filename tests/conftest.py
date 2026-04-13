"""Shared test fixtures and helpers for coin-oracle tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from msb_repr.data.features import FEATURE_COLUMNS


# ── OHLCV helpers ─────────────────────────────────────────────────────────────

def make_ohlcv_df(
    n_bars: int,
    base_price: float = 100.0,
    timeframe_minutes: int = 15,
    start: datetime | None = None,
) -> pl.DataFrame:
    """Synthetic OHLCV DataFrame with a Polars Datetime timestamp column."""
    if start is None:
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=timeframe_minutes * i) for i in range(n_bars)]
    rng = np.random.default_rng(42)
    closes = base_price + np.cumsum(rng.normal(0, 0.5, n_bars))
    return pl.DataFrame({
        "timestamp": timestamps,
        "open":   closes * (1 + rng.uniform(-0.002, 0.002, n_bars)),
        "high":   closes * (1 + rng.uniform(0.001, 0.005, n_bars)),
        "low":    closes * (1 - rng.uniform(0.001, 0.005, n_bars)),
        "close":  closes,
        "volume": rng.uniform(500, 2000, n_bars),
    })


def make_feature_df(n_bars: int, start: datetime | None = None) -> pl.DataFrame:
    """DataFrame with FEATURE_COLUMNS + timestamp — suitable for build_windows."""
    if start is None:
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=15 * i) for i in range(n_bars)]
    rng = np.random.default_rng(0)
    data: dict = {"timestamp": timestamps}
    for col in FEATURE_COLUMNS:
        data[col] = rng.standard_normal(n_bars).tolist()
    return pl.DataFrame(data)


def make_ohlcv_polars_ts(
    n_bars: int,
    base_price: float = 100.0,
    entry_ts_s: int = 1_000_000,
    bar_seconds: int = 900,
) -> pl.DataFrame:
    """OHLCV DataFrame where timestamps are Unix seconds encoded as Polars Datetime(us).

    All bars start strictly after entry_ts_s.  Useful for PositionTracker tests.
    """
    timestamps_s = [entry_ts_s + (i + 1) * bar_seconds for i in range(n_bars)]
    closes = np.full(n_bars, base_price)
    highs  = closes + 1.0
    lows   = closes - 1.0
    return pl.DataFrame({
        "timestamp": pl.from_epoch(pl.Series(timestamps_s, dtype=pl.Int64), time_unit="s"),
        "open":   closes.tolist(),
        "high":   highs.tolist(),
        "low":    lows.tolist(),
        "close":  closes.tolist(),
        "volume": [1000.0] * n_bars,
    })
