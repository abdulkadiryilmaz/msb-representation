"""Tests for msb_repr.data.windows — sliding window builder."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from msb_repr.data.features import FEATURE_COLUMNS
from msb_repr.data.windows import WindowConfig, build_windows
from tests.conftest import make_feature_df


WINDOW_SIZE = 48
STEP_SIZE   = 4
N_BARS      = 300  # enough for multiple complete windows


@pytest.fixture()
def feat_df():
    return make_feature_df(N_BARS)


@pytest.fixture()
def cfg():
    return WindowConfig(window_size=WINDOW_SIZE, step_size=STEP_SIZE, min_coverage=0.95)


# ── output shape ──────────────────────────────────────────────────────────────

def test_windows_shape(feat_df, cfg, tmp_path):
    out = tmp_path / "w.npz"
    n = build_windows(feat_df, "TEST_USDT", cfg, out)
    data = np.load(out)
    assert data["windows"].shape == (n, len(FEATURE_COLUMNS), WINDOW_SIZE)


def test_timestamps_shape_matches_windows(feat_df, cfg, tmp_path):
    out = tmp_path / "w.npz"
    n = build_windows(feat_df, "TEST_USDT", cfg, out)
    data = np.load(out)
    assert data["timestamps"].shape == (n,)


def test_windows_dtype_float32(feat_df, cfg, tmp_path):
    out = tmp_path / "w.npz"
    build_windows(feat_df, "TEST_USDT", cfg, out)
    data = np.load(out)
    assert data["windows"].dtype == np.float32


def test_timestamps_dtype_int64(feat_df, cfg, tmp_path):
    out = tmp_path / "w.npz"
    build_windows(feat_df, "TEST_USDT", cfg, out)
    data = np.load(out)
    assert data["timestamps"].dtype == np.int64


# ── window count ──────────────────────────────────────────────────────────────

def test_window_count_correct(feat_df, cfg, tmp_path):
    """Number of windows = floor((T - W) / S) + 1."""
    out = tmp_path / "w.npz"
    n = build_windows(feat_df, "TEST_USDT", cfg, out)
    expected = (N_BARS - WINDOW_SIZE) // STEP_SIZE + 1
    assert n == expected


def test_step_size_one_maximum_windows(feat_df, tmp_path):
    cfg = WindowConfig(window_size=WINDOW_SIZE, step_size=1)
    out = tmp_path / "w.npz"
    n = build_windows(feat_df, "TEST_USDT", cfg, out)
    expected = N_BARS - WINDOW_SIZE + 1
    assert n == expected


def test_no_windows_if_too_short(tmp_path):
    """If T < window_size, no windows should be built."""
    df = make_feature_df(WINDOW_SIZE - 1)
    cfg = WindowConfig(window_size=WINDOW_SIZE)
    out = tmp_path / "w.npz"
    n = build_windows(df, "SHORT", cfg, out)
    assert n == 0
    assert not out.exists()


# ── timestamps ────────────────────────────────────────────────────────────────

def test_timestamp_is_window_end(feat_df, cfg, tmp_path):
    """Each timestamp should equal the Unix epoch of the last bar in the window."""
    out = tmp_path / "w.npz"
    build_windows(feat_df, "TEST_USDT", cfg, out)
    data = np.load(out)
    all_ts = feat_df["timestamp"].dt.epoch(time_unit="s").to_numpy()

    for i, ts in enumerate(data["timestamps"]):
        start = i * STEP_SIZE
        expected_end_ts = all_ts[start + WINDOW_SIZE - 1]
        assert ts == expected_end_ts, f"Window {i}: expected {expected_end_ts}, got {ts}"


def test_timestamps_monotonically_increasing(feat_df, cfg, tmp_path):
    out = tmp_path / "w.npz"
    build_windows(feat_df, "TEST_USDT", cfg, out)
    ts = np.load(out)["timestamps"]
    assert (np.diff(ts) > 0).all()


# ── NaN handling ──────────────────────────────────────────────────────────────

def test_nan_replaced_with_zero(tmp_path):
    """NaN values inside a valid window should be replaced with 0."""
    from datetime import datetime, timedelta, timezone
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=15 * i) for i in range(N_BARS)]
    data = {col: [float("nan")] * 3 + [1.0] * (N_BARS - 3) for col in FEATURE_COLUMNS}
    data["timestamp"] = timestamps
    df = pl.DataFrame(data)

    cfg = WindowConfig(window_size=WINDOW_SIZE, step_size=STEP_SIZE, min_coverage=0.9)
    out = tmp_path / "w.npz"
    build_windows(df, "NAN_TEST", cfg, out)
    if out.exists():
        windows = np.load(out)["windows"]
        # No NaN should survive in the output
        assert not np.isnan(windows).any()


def test_min_coverage_drops_nan_heavy_windows(tmp_path):
    """Windows with too many NaN (< min_coverage) should be dropped."""
    from datetime import datetime, timedelta, timezone
    n = 200
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=15 * i) for i in range(n)]
    # First 100 bars: all NaN
    data = {col: [float("nan")] * 100 + [1.0] * 100 for col in FEATURE_COLUMNS}
    data["timestamp"] = timestamps
    df = pl.DataFrame(data)

    cfg_strict = WindowConfig(window_size=WINDOW_SIZE, step_size=STEP_SIZE, min_coverage=0.95)
    cfg_loose  = WindowConfig(window_size=WINDOW_SIZE, step_size=STEP_SIZE, min_coverage=0.0)
    out_strict = tmp_path / "strict.npz"
    out_loose  = tmp_path / "loose.npz"

    n_strict = build_windows(df, "STRICT", cfg_strict, out_strict)
    n_loose  = build_windows(df, "LOOSE",  cfg_loose,  out_loose)

    # Strict should drop NaN-heavy early windows; loose keeps them
    assert n_strict < n_loose


# ── output file ───────────────────────────────────────────────────────────────

def test_output_file_created(feat_df, cfg, tmp_path):
    out = tmp_path / "sub" / "w.npz"
    n = build_windows(feat_df, "TEST_USDT", cfg, out)
    assert n > 0
    assert out.exists()


def test_no_output_file_if_no_windows(tmp_path):
    df = make_feature_df(WINDOW_SIZE - 1)
    cfg = WindowConfig(window_size=WINDOW_SIZE)
    out = tmp_path / "empty.npz"
    build_windows(df, "SHORT", cfg, out)
    assert not out.exists()


def test_return_value_matches_file(feat_df, cfg, tmp_path):
    out = tmp_path / "w.npz"
    n = build_windows(feat_df, "TEST_USDT", cfg, out)
    data = np.load(out)
    assert data["windows"].shape[0] == n
