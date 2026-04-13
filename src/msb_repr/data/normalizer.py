"""Per-feature normalizer: fit on training set, apply at inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Normalizer:
    """Z-score normalizer fitted per feature channel.

    Stores mean and std per feature so they can be saved and reloaded
    without touching the training data again.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, windows: np.ndarray) -> "Normalizer":
        """Fit on training windows of shape (N, num_features, T)."""
        # Compute stats across N and T, keep per-feature
        flat = windows.reshape(windows.shape[0], windows.shape[1], -1)
        self.mean_ = flat.mean(axis=(0, 2))   # (num_features,)
        self.std_ = flat.std(axis=(0, 2)) + 1e-8
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Normalize windows of shape (N, num_features, T) or (num_features, T)."""
        assert self.mean_ is not None, "Call fit() first"
        if windows.ndim == 3:
            return (windows - self.mean_[None, :, None]) / self.std_[None, :, None]
        return (windows - self.mean_[:, None]) / self.std_[:, None]

    def inverse_transform(self, windows: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None
        if windows.ndim == 3:
            return windows * self.std_[None, :, None] + self.mean_[None, :, None]
        return windows * self.std_[:, None] + self.mean_[:, None]

    def save(self, path: Path) -> None:
        assert self.mean_ is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"mean": self.mean_.tolist(), "std": self.std_.tolist()}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        with open(path) as f:
            data = json.load(f)
        n = cls()
        n.mean_ = np.array(data["mean"], dtype=np.float32)
        n.std_ = np.array(data["std"], dtype=np.float32)
        return n
