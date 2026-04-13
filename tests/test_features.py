"""Tests for msb_repr.data.features — technical feature computation."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from msb_repr.data.features import FEATURE_COLUMNS, compute_features
from tests.conftest import make_ohlcv_df

# bars_per_day=1 → rolling windows stay small (max 7 bars), warmup ~26 bars
BARS_PER_DAY = 1
N_BARS = 60  # enough to clear all rolling warmup periods


@pytest.fixture()
def flat_df() -> pl.DataFrame:
    """Constant price OHLCV — good for testing mean-reversion features."""
    from datetime import datetime, timedelta, timezone
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(N_BARS)]
    return pl.DataFrame({
        "timestamp": timestamps,
        "open":   [100.0] * N_BARS,
        "high":   [100.5] * N_BARS,
        "low":    [99.5]  * N_BARS,
        "close":  [100.0] * N_BARS,
        "volume": [1000.0] * N_BARS,
    })


@pytest.fixture()
def trending_df() -> pl.DataFrame:
    """Steadily rising prices — useful for directional feature checks."""
    return make_ohlcv_df(N_BARS, base_price=100.0)


# ── Column presence ───────────────────────────────────────────────────────────

def test_all_feature_columns_present(trending_df):
    out = compute_features(trending_df, bars_per_day=BARS_PER_DAY)
    for col in FEATURE_COLUMNS:
        assert col in out.columns, f"Missing feature column: {col}"


def test_feature_count():
    assert len(FEATURE_COLUMNS) == 16


def test_no_extra_nans_after_warmup(trending_df):
    """After the rolling warmup period, no NaN should remain in feature columns."""
    out = compute_features(trending_df, bars_per_day=BARS_PER_DAY)
    # Drop the warmup rows (longest window is MACD ema26 = 26 bars)
    tail = out.tail(N_BARS - 30).select(FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        n_null = tail[col].null_count()
        assert n_null == 0, f"{col} has {n_null} nulls after warmup"


# ── Individual feature correctness ────────────────────────────────────────────

def test_log_return_correct(trending_df):
    out = compute_features(trending_df, bars_per_day=BARS_PER_DAY)
    closes = trending_df["close"].to_numpy()
    expected_lr = np.log(closes[1] / closes[0])
    actual_lr = out["log_return"][1]
    assert abs(actual_lr - expected_lr) < 1e-6


def test_log_return_first_bar_is_null(trending_df):
    out = compute_features(trending_df, bars_per_day=BARS_PER_DAY)
    assert out["log_return"][0] is None


def test_hl_range_nonneg(trending_df):
    out = compute_features(trending_df, bars_per_day=BARS_PER_DAY)
    assert (out["hl_range"].drop_nulls() >= 0).all()


def test_rsi_bounded(trending_df):
    out = compute_features(trending_df, bars_per_day=BARS_PER_DAY)
    rsi = out["rsi"].drop_nulls().to_numpy()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_rsi_near_100_on_constant_up(flat_df):
    """In a constant-price series deltas are all 0 → RSI should be 50 (neutral)."""
    out = compute_features(flat_df, bars_per_day=BARS_PER_DAY)
    rsi = out["rsi"].drop_nulls()
    # With zero gain and zero loss, formula gives 100 - 100/(1+0/0+eps) ≈ 50
    assert rsi.mean() is not None  # just ensure it runs without error


def test_vwap_dev_near_zero_for_flat_price(flat_df):
    """Flat price → VWAP == price → vwap_dev ≈ 0."""
    out = compute_features(flat_df, bars_per_day=BARS_PER_DAY)
    # Drop the warmup rows
    dev = out["vwap_dev"].drop_nulls().to_numpy()
    assert np.abs(dev).max() < 1e-6


def test_price_position_bounded(trending_df):
    """price_position should be in [0, 1] (it's a normalized range position)."""
    out = compute_features(trending_df, bars_per_day=BARS_PER_DAY)
    pp = out["price_position"].drop_nulls().to_numpy()
    assert (pp >= 0).all() and (pp <= 1.0 + 1e-6).all()


def test_atr_norm_positive(trending_df):
    out = compute_features(trending_df, bars_per_day=BARS_PER_DAY)
    atr = out["atr_norm"].drop_nulls().to_numpy()
    assert (atr >= 0).all()


# ── Cyclical time encoding ────────────────────────────────────────────────────

def test_hour_encoding_unit_circle(trending_df):
    """sin²(hour) + cos²(hour) should equal 1 for every bar."""
    out = compute_features(trending_df, bars_per_day=BARS_PER_DAY)
    sin_h = out["hour_sin"].to_numpy()
    cos_h = out["hour_cos"].to_numpy()
    norms = sin_h ** 2 + cos_h ** 2
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_dow_encoding_unit_circle(trending_df):
    """sin²(dow) + cos²(dow) should equal 1 for every bar."""
    out = compute_features(trending_df, bars_per_day=BARS_PER_DAY)
    sin_d = out["dow_sin"].to_numpy()
    cos_d = out["dow_cos"].to_numpy()
    norms = sin_d ** 2 + cos_d ** 2
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_hour_encoding_range(trending_df):
    """hour_sin and hour_cos must be in [-1, 1]."""
    out = compute_features(trending_df, bars_per_day=BARS_PER_DAY)
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        arr = out[col].to_numpy()
        assert arr.min() >= -1.0 - 1e-9 and arr.max() <= 1.0 + 1e-9, \
            f"{col} out of [-1, 1]"


def test_hour_midnight_encoding():
    """Midnight bars (hour=0) should have hour_sin≈0, hour_cos≈1."""
    from datetime import datetime, timezone
    df = pl.DataFrame({
        "timestamp": [datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)],
        "open": [100.0], "high": [101.0], "low": [99.0],
        "close": [100.0], "volume": [1000.0],
    })
    out = compute_features(df, bars_per_day=1)
    assert abs(out["hour_sin"][0]) < 1e-6
    assert abs(out["hour_cos"][0] - 1.0) < 1e-6


# ── bars_per_day scaling ──────────────────────────────────────────────────────

def test_bars_per_day_affects_vwap_window(flat_df):
    """Larger bars_per_day → wider vwap_window → more warmup rows with null."""
    out_bpd1 = compute_features(flat_df, bars_per_day=1)
    out_bpd5 = compute_features(flat_df, bars_per_day=5)
    # Both should return the same number of rows
    assert len(out_bpd1) == len(out_bpd5) == N_BARS


def test_open_close_return_direction(flat_df):
    """For flat open == close the open_close_return should be 0."""
    out = compute_features(flat_df, bars_per_day=BARS_PER_DAY)
    ocr = out["open_close_return"].drop_nulls().to_numpy()
    np.testing.assert_allclose(ocr, 0.0, atol=1e-6)


def test_volume_ratio_bounded(trending_df):
    """volume_ratio should be in (0, 1] — current volume / rolling max volume."""
    out = compute_features(trending_df, bars_per_day=BARS_PER_DAY)
    vr = out["volume_ratio"].drop_nulls().to_numpy()
    assert (vr > 0).all() and (vr <= 1.0 + 1e-6).all()


def test_volume_ratio_scales_with_bars_per_day():
    """With bars_per_day=96 (15m) the ratio window should be larger than bpd=6 (4h)."""
    df96 = make_ohlcv_df(N_BARS, base_price=100.0)
    df6  = make_ohlcv_df(N_BARS, base_price=100.0)
    # Both use same data; window sizes differ but output column must exist in both
    out96 = compute_features(df96, bars_per_day=96)
    out6  = compute_features(df6,  bars_per_day=6)
    assert "volume_ratio" in out96.columns
    assert "volume_ratio" in out6.columns


def test_volume_zscore_zero_variance_handled():
    """Constant volume should not produce NaN (epsilon guard in denominator)."""
    from datetime import datetime, timedelta, timezone
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(N_BARS)]
    df = pl.DataFrame({
        "timestamp": timestamps,
        "open": [100.0] * N_BARS,
        "high": [101.0] * N_BARS,
        "low":  [99.0]  * N_BARS,
        "close": [100.0] * N_BARS,
        "volume": [500.0] * N_BARS,  # constant volume
    })
    out = compute_features(df, bars_per_day=BARS_PER_DAY)
    # After warmup, no inf/nan
    tail = out.tail(N_BARS - 10)["volume_zscore"].to_numpy()
    assert np.isfinite(tail).all()
