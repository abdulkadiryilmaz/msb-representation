"""Config objects and dataset spec I/O for Stage 1A."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from msb_repr.stage1a.features import (
    STAGE1A_LONG_FEATURE_COLUMNS,
    STAGE1A_SHORT_FEATURE_COLUMNS,
)


@dataclass
class Stage1AWindowConfig:
    short_window_size: int = 48
    long_window_size: int = 288
    step_size: int = 4
    min_coverage: float = 0.95


@dataclass
class Stage1ALabelConfig:
    structure_bars: int = 36
    recent_bars: int = 12
    pivot_n: int = 3
    min_break_pct: float = 0.002
    atr_factor: float = 0.5
    min_recent_break_bars: int = 2


@dataclass
class Stage1ASplitConfig:
    train_end: str
    val_end: str


@dataclass
class Stage1ADatasetSpec:
    exchange: str
    timeframe: str
    symbols: list[str]
    windows: Stage1AWindowConfig = field(default_factory=Stage1AWindowConfig)
    labels: Stage1ALabelConfig = field(default_factory=Stage1ALabelConfig)
    splits: Stage1ASplitConfig | None = None
    short_feature_columns: list[str] = field(default_factory=lambda: STAGE1A_SHORT_FEATURE_COLUMNS.copy())
    long_feature_columns: list[str] = field(default_factory=lambda: STAGE1A_LONG_FEATURE_COLUMNS.copy())

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Stage1ADatasetSpec":
        data = json.loads(path.read_text())
        return cls(
            exchange=data["exchange"],
            timeframe=data["timeframe"],
            symbols=data["symbols"],
            windows=Stage1AWindowConfig(**data["windows"]),
            labels=Stage1ALabelConfig(**data["labels"]),
            splits=Stage1ASplitConfig(**data["splits"]) if data.get("splits") else None,
            short_feature_columns=data.get("short_feature_columns", STAGE1A_SHORT_FEATURE_COLUMNS.copy()),
            long_feature_columns=data.get("long_feature_columns", STAGE1A_LONG_FEATURE_COLUMNS.copy()),
        )
