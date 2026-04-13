"""Technical feature computation from raw OHLCV data."""

from __future__ import annotations

import math

import polars as pl


def compute_features(
    df: pl.DataFrame,
    bars_per_day: int = 6,  # 6 for 4h, 24 for 1h, 1 for 1d
    rsi_period: int = 14,
) -> pl.DataFrame:
    """Compute derived features on top of raw OHLCV.

    Input columns expected: timestamp, open, high, low, close, volume.
    Output adds: log_return, hl_range, vwap_dev, volume_zscore,
                 rsi, macd, macd_signal, bb_position,
                 open_close_return, atr_norm, volume_ratio, price_position,
                 hour_sin, hour_cos, dow_sin, dow_cos.

    Args:
        df: Raw OHLCV DataFrame sorted by timestamp ascending.
            timestamp must be a Polars Datetime column.
        bars_per_day: Number of bars per day for the chosen timeframe.
                      6 for 4h candles, 24 for 1h, 1 for daily.
        rsi_period: Lookback for RSI calculation (in bars).

    Returns:
        DataFrame with all original columns plus derived features.
    """
    df = df.sort("timestamp")

    vwap_window = bars_per_day           # 1-day rolling VWAP
    vol_zscore_window = bars_per_day * 7 # 7-day rolling volume z-score

    # Log return (close-to-close)
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return"),
        # High-low range normalised by close
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("hl_range"),
        # Intrabar return: open-to-close
        (pl.col("close") / pl.col("open")).log().alias("open_close_return"),
    )

    # VWAP deviation (rolling 1-day window)
    df = df.with_columns(
        (
            pl.col("close") / (
                (pl.col("close") * pl.col("volume")).rolling_mean(vwap_window) /
                pl.col("volume").rolling_mean(vwap_window)
            ) - 1
        ).alias("vwap_dev"),
    )

    # Volume z-score (rolling 7-day window)
    df = df.with_columns(
        (
            (pl.col("volume") - pl.col("volume").rolling_mean(vol_zscore_window)) /
            (pl.col("volume").rolling_std(vol_zscore_window) + 1e-8)
        ).alias("volume_zscore"),
    )

    # Volume ratio: current bar vs rolling ~4h max (scaled with bars_per_day)
    vol_ratio_window = max(4, bars_per_day // 6)  # ~4h window; floor at 4 bars
    df = df.with_columns(
        (pl.col("volume") / (pl.col("volume").rolling_max(vol_ratio_window) + 1e-8)).alias("volume_ratio"),
    )

    # RSI(14) — short-window momentum
    delta = pl.col("close").diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0).rolling_mean(rsi_period)
    loss = pl.when(delta < 0).then(-delta).otherwise(0).rolling_mean(rsi_period)
    df = df.with_columns(
        (100 - 100 / (1 + gain / (loss + 1e-8))).alias("rsi"),
    )

    # RSI(42) — slow RSI for trend-window AE; fewer oscillations over 288-bar window
    rsi_slow_period = rsi_period * 3  # 42 bars = ~10.5h at 15m
    gain_slow = pl.when(delta > 0).then(delta).otherwise(0).rolling_mean(rsi_slow_period)
    loss_slow = pl.when(delta < 0).then(-delta).otherwise(0).rolling_mean(rsi_slow_period)
    df = df.with_columns(
        (100 - 100 / (1 + gain_slow / (loss_slow + 1e-8))).alias("rsi_42"),
    )

    # MACD (12/26 EMA diff) and signal (9 EMA of MACD)
    ema12 = pl.col("close").ewm_mean(span=12)
    ema26 = pl.col("close").ewm_mean(span=26)
    df = df.with_columns(
        (ema12 - ema26).alias("macd"),
    )
    df = df.with_columns(
        pl.col("macd").ewm_mean(span=9).alias("macd_signal"),
    )
    # MACD histogram — trend acceleration; directly encodes momentum divergence
    df = df.with_columns(
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist"),
    )

    # Bollinger band position: (close - lower) / (upper - lower)
    roll_mean = pl.col("close").rolling_mean(20)
    roll_std = pl.col("close").rolling_std(20)
    df = df.with_columns(
        (
            (pl.col("close") - (roll_mean - 2 * roll_std)) /
            (4 * roll_std + 1e-8)
        ).alias("bb_position"),
    )

    # BB position (50-bar) — structural position for trend-window AE
    roll_mean_50 = pl.col("close").rolling_mean(50)
    roll_std_50 = pl.col("close").rolling_std(50)
    df = df.with_columns(
        (
            (pl.col("close") - (roll_mean_50 - 2 * roll_std_50)) /
            (4 * roll_std_50 + 1e-8)
        ).alias("bb_position_50"),
    )

    # Price position in 20-bar range: 0 = at rolling low, 1 = at rolling high
    roll_low = pl.col("close").rolling_min(20)
    roll_high = pl.col("close").rolling_max(20)
    df = df.with_columns(
        (
            (pl.col("close") - roll_low) / (roll_high - roll_low + 1e-8)
        ).alias("price_position"),
    )

    # ATR(14) normalised by close — smoothed volatility across sessions
    # TR = max(high-low, |high-prev_close|, |low-prev_close|)
    # fill_null(0.0) for the first bar where prev_close is unavailable
    hl = pl.col("high") - pl.col("low")
    hpc = (pl.col("high") - pl.col("close").shift(1)).abs().fill_null(0.0)
    lpc = (pl.col("low") - pl.col("close").shift(1)).abs().fill_null(0.0)
    df = df.with_columns(
        pl.max_horizontal(hl, hpc, lpc).alias("_tr"),
    )
    df = df.with_columns(
        (pl.col("_tr").rolling_mean(14) / (pl.col("close") + 1e-8)).alias("atr_norm"),
        # ATR(28) — slower volatility regime indicator for trend-window AE
        (pl.col("_tr").rolling_mean(28) / (pl.col("close") + 1e-8)).alias("atr_norm_28"),
    )
    df = df.drop("_tr")

    # Cyclical time encoding — captures session and weekly calendar effects
    # hour: UTC hour of bar open (4h bars land at 0, 4, 8, 12, 16, 20)
    # weekday: ISO weekday (Mon=1 .. Sun=7), shifted to 0-indexed before encoding
    df = df.with_columns(
        (pl.col("timestamp").dt.hour() * (2 * math.pi / 24)).sin().alias("hour_sin"),
        (pl.col("timestamp").dt.hour() * (2 * math.pi / 24)).cos().alias("hour_cos"),
        ((pl.col("timestamp").dt.weekday() - 1) * (2 * math.pi / 7)).sin().alias("dow_sin"),
        ((pl.col("timestamp").dt.weekday() - 1) * (2 * math.pi / 7)).cos().alias("dow_cos"),
    )

    # ── Trend features (long-window AE input) ────────────────────────────────
    # close_norm: 3-day log return — captures medium-term price trend direction
    long_return_period = bars_per_day * 3
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(long_return_period)).log().alias("close_norm"),
    )

    # volume_norm: log deviation from 3-day rolling volume mean (same horizon as close_norm)
    vol_long_window = max(bars_per_day * 3, 20)
    df = df.with_columns(
        (pl.col("volume") / (pl.col("volume").rolling_mean(vol_long_window) + 1e-8)).log().alias("volume_norm"),
    )

    # ema_slope: log-return of 1-day EMA — smoothed trend momentum
    df = df.with_columns(
        pl.col("close").ewm_mean(span=bars_per_day).log().diff().alias("ema_slope"),
    )

    return df


# Ordered list of feature columns fed to the short AE (16 channels)
FEATURE_COLUMNS: list[str] = [
    # Price dynamics
    "log_return",
    "open_close_return",
    # Volatility / range
    "hl_range",
    "atr_norm",
    # VWAP & mean-reversion
    "vwap_dev",
    "bb_position",
    "price_position",
    # Volume
    "volume_zscore",
    "volume_ratio",
    # Momentum
    "rsi",
    "macd",
    "macd_signal",
    # Time encoding (cyclical)
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

# Ordered list of feature columns for the long-window trend AE (7 channels)
# Slow-moving trend signals — used with 288-bar (72h) windows.
# Deliberately excludes rsi(14) and atr_norm(14) — those are in FEATURE_COLUMNS
# and overlap with z_short; these slower variants encode trend-regime context instead.
TREND_FEATURE_COLUMNS: list[str] = [
    "close_norm",     # 3-day log return: medium-term price trend direction
    "ema_slope",      # 1-day EMA log-return: smoothed trend momentum
    "rsi_42",         # RSI(42): slow momentum — fewer oscillations over 288-bar window
    "atr_norm_28",    # ATR(28)/close: slower volatility regime (vs ATR(14) in short AE)
    "bb_position_50", # BB(50) position: structural price level over 72h window
    "macd_hist",      # MACD histogram: trend acceleration / divergence signal
]
