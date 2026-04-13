from __future__ import annotations

import numpy as np

from msb_repr.stage1a.labels import label_state_window


def test_label_state_window_bullish_state():
    highs = np.array(
        [1.00, 1.05, 1.10, 1.15, 1.20, 1.16, 1.12] + [1.08] * 29 + [1.21, 1.23] + [1.24] * 10,
        dtype=np.float32,
    )
    lows = highs - 0.04
    closes = np.array(
        [1.00, 1.03, 1.07, 1.11, 1.18, 1.14, 1.10] + [1.07] * 29 + [1.22, 1.23] + [1.25] * 10,
        dtype=np.float32,
    )
    assert len(highs) == 48
    assert label_state_window(highs, lows, closes) == 1


def test_label_state_window_intact_when_no_break():
    highs = np.array([1.00 + 0.01 * (i % 5) for i in range(48)], dtype=np.float32)
    lows = highs - 0.05
    closes = np.array([1.00 + 0.005 * (i % 5) for i in range(48)], dtype=np.float32)
    assert label_state_window(highs, lows, closes) == 0
