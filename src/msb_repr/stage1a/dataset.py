"""Dataset helpers for Stage 1A dual-window training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from msb_repr.stage1a.config import Stage1ADatasetSpec


@dataclass
class Stage1AMeta:
    symbol: str
    timestamp: int


def _date_to_unix(date_str: str) -> int:
    return int(datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc).timestamp())


def align_windows_by_timestamp(
    short_windows: np.ndarray,
    short_timestamps: np.ndarray,
    long_windows: np.ndarray,
    long_timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align short and long window tensors on common end timestamps."""
    common_ts, short_idx, long_idx = np.intersect1d(
        short_timestamps,
        long_timestamps,
        assume_unique=False,
        return_indices=True,
    )
    return short_windows[short_idx], long_windows[long_idx], common_ts.astype(np.int64)


class Stage1ADualWindowDataset(Dataset):
    """Aligned short/long window dataset for Stage 1A."""

    def __init__(
        self,
        short_windows: np.ndarray,
        long_windows: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray,
        symbols: list[str],
    ) -> None:
        if not (len(short_windows) == len(long_windows) == len(labels) == len(timestamps) == len(symbols)):
            raise ValueError("All Stage 1A dataset arrays must have the same length")
        self.short_windows = short_windows.astype(np.float32)
        self.long_windows = long_windows.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.timestamps = timestamps.astype(np.int64)
        self.symbols = symbols

    @classmethod
    def from_root(
        cls,
        root: Path,
        split: str | None = None,
        symbols: list[str] | None = None,
    ) -> "Stage1ADualWindowDataset":
        spec = Stage1ADatasetSpec.load(root / "spec.json")
        symbol_dir = root / "symbols"
        short_parts = []
        long_parts = []
        label_parts = []
        ts_parts = []
        symbol_parts: list[str] = []

        selected_symbols = symbols or spec.symbols
        for symbol in selected_symbols:
            path = symbol_dir / f"{symbol}.npz"
            data = np.load(path)
            ts = data["timestamps"].astype(np.int64)
            mask = np.ones(len(ts), dtype=bool)

            if split and spec.splits is not None:
                train_end = _date_to_unix(spec.splits.train_end)
                val_end = _date_to_unix(spec.splits.val_end)
                if split == "train":
                    mask = ts <= train_end
                elif split == "val":
                    mask = (ts > train_end) & (ts <= val_end)
                elif split == "test":
                    mask = ts > val_end
                else:
                    raise ValueError(f"Unknown split: {split}")

            short_parts.append(data["short_windows"][mask])
            long_parts.append(data["long_windows"][mask])
            label_parts.append(data["labels"][mask])
            ts_parts.append(ts[mask])
            symbol_parts.extend([symbol] * int(mask.sum()))

        if not short_parts:
            raise ValueError(f"No Stage 1A symbol files found in {symbol_dir}")

        return cls(
            short_windows=np.concatenate(short_parts, axis=0),
            long_windows=np.concatenate(long_parts, axis=0),
            labels=np.concatenate(label_parts, axis=0),
            timestamps=np.concatenate(ts_parts, axis=0),
            symbols=symbol_parts,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Stage1AMeta]:
        return (
            torch.from_numpy(self.short_windows[idx]),
            torch.from_numpy(self.long_windows[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
            Stage1AMeta(symbol=self.symbols[idx], timestamp=int(self.timestamps[idx])),
        )


def collate_fn_stage1a(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, Stage1AMeta]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[Stage1AMeta]]:
    shorts, longs, labels, metas = zip(*batch)
    return torch.stack(shorts), torch.stack(longs), torch.stack(labels), list(metas)
