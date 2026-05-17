"""Dataset helpers for Stage 1A dual-window training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from msb_repr.stage1a.config import Stage1ADatasetSpec


@dataclass
class Stage1AMeta:
    symbol: str
    timestamp: int
    factor_target: int | None = None
    pressure_target: int | None = None
    maturity_target: int | None = None


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
        factor_targets: np.ndarray | None = None,
        pressure_targets: np.ndarray | None = None,
        maturity_targets: np.ndarray | None = None,
    ) -> None:
        if not (len(short_windows) == len(long_windows) == len(labels) == len(timestamps) == len(symbols)):
            raise ValueError("All Stage 1A dataset arrays must have the same length")
        for name, values in {
            "factor_targets": factor_targets,
            "pressure_targets": pressure_targets,
            "maturity_targets": maturity_targets,
        }.items():
            if values is not None and len(values) != len(labels):
                raise ValueError(f"{name} must have the same length as labels")
        self.short_windows = short_windows.astype(np.float32)
        self.long_windows = long_windows.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.timestamps = timestamps.astype(np.int64)
        self.symbols = symbols
        self.factor_targets = None if factor_targets is None else factor_targets.astype(np.int64)
        self.pressure_targets = None if pressure_targets is None else pressure_targets.astype(np.int64)
        self.maturity_targets = None if maturity_targets is None else maturity_targets.astype(np.int64)

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
            Stage1AMeta(
                symbol=self.symbols[idx],
                timestamp=int(self.timestamps[idx]),
                factor_target=None if self.factor_targets is None else int(self.factor_targets[idx]),
                pressure_target=None if self.pressure_targets is None else int(self.pressure_targets[idx]),
                maturity_target=None if self.maturity_targets is None else int(self.maturity_targets[idx]),
            ),
        )


def collate_fn_stage1a(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, Stage1AMeta]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[Stage1AMeta]]:
    shorts, longs, labels, metas = zip(*batch)
    return torch.stack(shorts), torch.stack(longs), torch.stack(labels), list(metas)


class Stage1ASymbolBalancedBatchSampler(Sampler[list[int]]):
    """Round-robin batch sampler that keeps symbol coverage mixed within a batch."""

    def __init__(
        self,
        symbols: list[str],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.symbols = symbols
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._symbol_values = list(dict.fromkeys(symbols))
        self._indices_by_symbol: dict[str, list[int]] = {symbol: [] for symbol in self._symbol_values}
        for idx, symbol in enumerate(symbols):
            self._indices_by_symbol[symbol].append(idx)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        pools = {symbol: indices.copy() for symbol, indices in self._indices_by_symbol.items()}
        symbol_order = self._symbol_values.copy()
        if self.shuffle:
            rng.shuffle(symbol_order)
            for indices in pools.values():
                rng.shuffle(indices)

        offsets = {symbol: 0 for symbol in symbol_order}
        active_symbols = [symbol for symbol in symbol_order if pools[symbol]]
        batch: list[int] = []

        while active_symbols:
            next_active: list[str] = []
            for symbol in active_symbols:
                pos = offsets[symbol]
                if pos >= len(pools[symbol]):
                    continue
                batch.append(pools[symbol][pos])
                offsets[symbol] += 1
                if offsets[symbol] < len(pools[symbol]):
                    next_active.append(symbol)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            active_symbols = next_active

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total = len(self.symbols)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size
