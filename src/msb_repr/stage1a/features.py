"""Feature definitions for the Stage 1A representation pipeline."""

from __future__ import annotations

# Shared-core feature family chosen in the Stage 1A design worklog.
# Short and long branches intentionally speak the same feature language;
# the main difference should come from temporal scale, not a different modality.
STAGE1A_SHORT_FEATURE_COLUMNS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "log_return",
    "open_close_return",
    "hl_range",
    "atr_norm",
    "price_position",
    "bb_position",
]

# V1 keeps the same semantic family for long context. Later iterations may add
# carefully scoped scale-specific extras after the shared-core baseline is stable.
STAGE1A_LONG_FEATURE_COLUMNS: list[str] = STAGE1A_SHORT_FEATURE_COLUMNS.copy()

# Raw price columns required for state-style label generation.
STAGE1A_LABEL_PRICE_COLUMNS: list[str] = ["high", "low", "close"]
