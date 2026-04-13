"""Tests for msb_repr.data.normalizer — per-feature z-score normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from msb_repr.data.normalizer import Normalizer


def make_windows(n: int = 100, f: int = 8, t: int = 48, seed: int = 0) -> np.ndarray:
    """Random windows of shape (N, F, T)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, f, t)).astype(np.float32)


# ── fit ───────────────────────────────────────────────────────────────────────

def test_fit_returns_self():
    norm = Normalizer()
    windows = make_windows()
    result = norm.fit(windows)
    assert result is norm


def test_fit_mean_shape():
    n, f, t = 100, 8, 48
    norm = Normalizer().fit(make_windows(n, f, t))
    assert norm.mean_.shape == (f,)


def test_fit_std_shape():
    n, f, t = 100, 8, 48
    norm = Normalizer().fit(make_windows(n, f, t))
    assert norm.std_.shape == (f,)


def test_fit_std_positive():
    """std_ must be strictly positive (epsilon guard applied)."""
    norm = Normalizer().fit(make_windows())
    assert (norm.std_ > 0).all()


def test_fit_constant_feature_std_is_epsilon():
    """A constant feature across all windows → std = 0 + 1e-8."""
    windows = make_windows()
    windows[:, 3, :] = 5.0  # channel 3 is constant
    norm = Normalizer().fit(windows)
    assert abs(norm.std_[3] - 1e-8) < 1e-12


# ── transform ─────────────────────────────────────────────────────────────────

def test_transform_3d_shape():
    windows = make_windows()
    norm = Normalizer().fit(windows)
    out = norm.transform(windows)
    assert out.shape == windows.shape


def test_transform_zero_mean_on_training_data():
    """Transformed training data should have ≈ zero mean per feature."""
    windows = make_windows(500)
    norm = Normalizer().fit(windows)
    transformed = norm.transform(windows)
    # Mean over N and T axes per feature
    per_feature_mean = transformed.mean(axis=(0, 2))
    np.testing.assert_allclose(per_feature_mean, 0.0, atol=1e-3)


def test_transform_unit_std_on_training_data():
    """Transformed training data should have ≈ unit std per feature."""
    windows = make_windows(500)
    norm = Normalizer().fit(windows)
    transformed = norm.transform(windows)
    per_feature_std = transformed.reshape(500, 8, -1).std(axis=(0, 2))
    np.testing.assert_allclose(per_feature_std, 1.0, atol=1e-2)


def test_transform_2d_input():
    """transform should accept 2D input (F, T) for single-window inference."""
    windows = make_windows()
    norm = Normalizer().fit(windows)
    single = make_windows(n=1)[0]  # shape (F, T)
    out = norm.transform(single)
    assert out.shape == single.shape


def test_transform_before_fit_raises():
    norm = Normalizer()
    with pytest.raises(AssertionError):
        norm.transform(make_windows())


# ── inverse_transform ─────────────────────────────────────────────────────────

def test_inverse_transform_round_trip_3d():
    windows = make_windows()
    norm = Normalizer().fit(windows)
    reconstructed = norm.inverse_transform(norm.transform(windows))
    np.testing.assert_allclose(reconstructed, windows, atol=1e-4)


def test_inverse_transform_round_trip_2d():
    windows = make_windows()
    norm = Normalizer().fit(windows)
    single = windows[0]  # (F, T)
    reconstructed = norm.inverse_transform(norm.transform(single))
    np.testing.assert_allclose(reconstructed, single, atol=1e-4)


# ── save / load ───────────────────────────────────────────────────────────────

def test_save_load_mean_preserved(tmp_path):
    windows = make_windows()
    norm = Normalizer().fit(windows)
    path = tmp_path / "norm_params.json"
    norm.save(path)
    loaded = Normalizer.load(path)
    np.testing.assert_array_equal(loaded.mean_, norm.mean_)


def test_save_load_std_preserved(tmp_path):
    windows = make_windows()
    norm = Normalizer().fit(windows)
    path = tmp_path / "norm_params.json"
    norm.save(path)
    loaded = Normalizer.load(path)
    np.testing.assert_array_equal(loaded.std_, norm.std_)


def test_save_creates_parent_dirs(tmp_path):
    norm = Normalizer().fit(make_windows())
    nested = tmp_path / "a" / "b" / "norm.json"
    norm.save(nested)
    assert nested.exists()


def test_load_transform_matches_original(tmp_path):
    windows = make_windows()
    norm = Normalizer().fit(windows)
    path = tmp_path / "norm.json"
    norm.save(path)
    loaded = Normalizer.load(path)
    np.testing.assert_allclose(loaded.transform(windows), norm.transform(windows), atol=1e-6)


# ── edge cases ────────────────────────────────────────────────────────────────

def test_transform_does_not_mutate_input():
    windows = make_windows()
    original = windows.copy()
    norm = Normalizer().fit(windows)
    norm.transform(windows)
    np.testing.assert_array_equal(windows, original)


def test_fit_with_different_num_features():
    for f in (4, 8, 16):
        norm = Normalizer().fit(make_windows(f=f))
        assert norm.mean_.shape == (f,)
        assert norm.std_.shape == (f,)
