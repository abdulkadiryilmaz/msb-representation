"""State-style MSB labels for Stage 1A representation learning."""

from __future__ import annotations

import numpy as np


def find_last_pivot_high(highs: np.ndarray, n: int) -> float | None:
    """Return the most recent confirmed pivot high inside the structure zone."""
    for i in range(len(highs) - n - 1, n - 1, -1):
        val = highs[i]
        if val > highs[i - n:i].max() and val > highs[i + 1:i + n + 1].max():
            return float(val)
    return None


def find_last_pivot_low(lows: np.ndarray, n: int) -> float | None:
    """Return the most recent confirmed pivot low inside the structure zone."""
    for i in range(len(lows) - n - 1, n - 1, -1):
        val = lows[i]
        if val < lows[i - n:i].min() and val < lows[i + 1:i + n + 1].min():
            return float(val)
    return None


def label_state_window(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    structure_bars: int = 36,
    recent_bars: int = 12,
    pivot_n: int = 3,
    min_break_pct: float = 0.002,
    atr_factor: float = 0.5,
    min_recent_break_bars: int = 2,
) -> int:
    """Label the current structural state of a short window.

    Labels:
        0 = intact
        1 = bullish structure state
        2 = bearish structure state

    The label is state-style, not event-style: if the last `recent_bars` section
    still holds beyond the latest confirmed swing level, the state can remain
    bullish or bearish for several adjacent anchors.
    """
    if len(highs) != len(lows) or len(highs) != len(closes):
        raise ValueError("highs, lows, closes must have identical lengths")
    if structure_bars + recent_bars > len(closes):
        raise ValueError("structure_bars + recent_bars exceeds window length")

    struct_highs = highs[:structure_bars]
    struct_lows = lows[:structure_bars]
    recent_closes = closes[structure_bars:structure_bars + recent_bars]

    atr = float(np.mean(struct_highs - struct_lows)) if len(struct_highs) else 0.0
    ref_close = float(closes[structure_bars - 1]) if closes[structure_bars - 1] > 0 else 1.0
    effective_break_pct = max(min_break_pct, atr_factor * atr / ref_close)

    swing_high = find_last_pivot_high(struct_highs, pivot_n)
    swing_low = find_last_pivot_low(struct_lows, pivot_n)

    bull_strength = -np.inf
    bear_strength = -np.inf

    if swing_high is not None:
        bull_level = swing_high * (1.0 + effective_break_pct)
        bull_excess = recent_closes - bull_level
        if int(np.sum(bull_excess > 0.0)) >= min_recent_break_bars:
            bull_strength = float(np.max(bull_excess / np.maximum(bull_level, 1e-8)))

    if swing_low is not None:
        bear_level = swing_low * (1.0 - effective_break_pct)
        bear_excess = bear_level - recent_closes
        if int(np.sum(bear_excess > 0.0)) >= min_recent_break_bars:
            bear_strength = float(np.max(bear_excess / np.maximum(bear_level, 1e-8)))

    if bull_strength == -np.inf and bear_strength == -np.inf:
        return 0
    if bull_strength >= bear_strength:
        return 1
    return 2
