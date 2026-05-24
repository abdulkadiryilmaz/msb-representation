"""Dataset helpers for Stage 1B forward-structure prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


DIRECTION_LABEL_TO_ID = {"bullish": 0, "bearish": 1}
RECON_LABEL_TO_ID = {"none": 0, "bullish": 1, "bearish": 2}
RECON_ID_TO_LABEL = {idx: label for label, idx in RECON_LABEL_TO_ID.items()}
DEFAULT_RECENT_BARS = 12


def _finite_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _float_column(df: pl.DataFrame, column: str) -> np.ndarray:
    return df.get_column(column).cast(pl.Float64, strict=False).fill_null(np.nan).to_numpy()


def _context_feature_matrix(
    df: pl.DataFrame,
    latent: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    feature_keys: list[str],
) -> np.ndarray:
    current_close = _float_column(df, "current_close")
    bull_level = _float_column(df, "bull_level")
    bear_level = _float_column(df, "bear_level")
    valid_close = np.isfinite(current_close) & (current_close > 0)
    bull_distance_pct = np.zeros(len(df), dtype=np.float32)
    bear_distance_pct = np.zeros(len(df), dtype=np.float32)
    bull_mask = valid_close & np.isfinite(bull_level)
    bear_mask = valid_close & np.isfinite(bear_level)
    bull_distance_pct[bull_mask] = ((bull_level[bull_mask] - current_close[bull_mask]) / current_close[bull_mask]).astype(
        np.float32
    )
    bear_distance_pct[bear_mask] = ((current_close[bear_mask] - bear_level[bear_mask]) / current_close[bear_mask]).astype(
        np.float32
    )
    nearest_distance_pct = np.minimum(np.abs(bull_distance_pct), np.abs(bear_distance_pct)).astype(np.float32)
    nearest_side_signed = np.where(np.abs(bull_distance_pct) <= np.abs(bear_distance_pct), 1.0, -1.0).astype(np.float32)
    current_labels = df.get_column("current_label").cast(pl.String).to_numpy()

    values: dict[str, np.ndarray] = {
        "bull_distance_pct_scaled": bull_distance_pct * 100.0,
        "bear_distance_pct_scaled": bear_distance_pct * 100.0,
        "nearest_distance_pct_scaled": nearest_distance_pct * 100.0,
        "nearest_side_signed": nearest_side_signed,
        "bull_close_count_norm": latent["bull_close_count"][indices].astype(np.float32) / DEFAULT_RECENT_BARS,
        "bear_close_count_norm": latent["bear_close_count"][indices].astype(np.float32) / DEFAULT_RECENT_BARS,
        "bull_wick_count_norm": latent["bull_wick_count"][indices].astype(np.float32) / DEFAULT_RECENT_BARS,
        "bear_wick_count_norm": latent["bear_wick_count"][indices].astype(np.float32) / DEFAULT_RECENT_BARS,
        "bull_final_excess_scaled": np.nan_to_num(latent["bull_final_excess"][indices].astype(np.float32)) * 100.0,
        "bear_final_excess_scaled": np.nan_to_num(latent["bear_final_excess"][indices].astype(np.float32)) * 100.0,
        "current_label_intact": (current_labels == "intact").astype(np.float32),
        "current_label_bullish": (current_labels == "bullish").astype(np.float32),
        "current_label_bearish": (current_labels == "bearish").astype(np.float32),
    }
    return np.stack([values[key] for key in feature_keys], axis=1).astype(np.float32)


@dataclass(frozen=True)
class Stage1BMeta:
    index: int
    symbol: str
    timestamp: int
    raw_direction: str


class Stage1BForwardDataset(Dataset):
    """Frozen Stage 1A latent dataset with Stage 1B forward labels."""

    def __init__(
        self,
        features: np.ndarray,
        break_targets: np.ndarray,
        direction_targets: np.ndarray,
        recon_targets: np.ndarray,
        metas: list[Stage1BMeta],
        feature_keys: list[str],
        target_columns: list[str],
        latent_path: Path,
        label_path: Path,
        context_feature_keys: list[str] | None = None,
    ) -> None:
        if not (
            len(features)
            == len(break_targets)
            == len(direction_targets)
            == len(recon_targets)
            == len(metas)
        ):
            raise ValueError("All Stage 1B dataset arrays must have the same length")
        self.features = features.astype(np.float32)
        self.break_targets = break_targets.astype(np.int64)
        self.direction_targets = direction_targets.astype(np.int64)
        self.recon_targets = recon_targets.astype(np.int64)
        self.metas = metas
        self.feature_keys = feature_keys
        self.context_feature_keys = context_feature_keys or []
        self.target_columns = target_columns
        self.latent_path = latent_path
        self.label_path = label_path

    @classmethod
    def from_paths(
        cls,
        latent_path: Path,
        label_path: Path,
        feature_keys: list[str],
        target_column: str = "h16_future_break_direction",
        target_columns: list[str] | None = None,
        context_feature_keys: list[str] | None = None,
        drop_labels: set[str] | None = None,
    ) -> "Stage1BForwardDataset":
        drop_labels = drop_labels or {"ambiguous", "insufficient_future"}
        target_columns = target_columns or [target_column]
        latent = np.load(latent_path, allow_pickle=False)
        label_df = pl.read_parquet(label_path)
        valid_labels = [label for label in RECON_LABEL_TO_ID if label not in drop_labels]
        valid_df = label_df
        for column in target_columns:
            valid_df = valid_df.filter(pl.col(column).is_in(valid_labels))
        indices = valid_df["index"].to_numpy().astype(np.int64)

        feature_parts = []
        for key in feature_keys:
            if key not in latent.files:
                raise ValueError(f"Feature key not found in latent export: {key}")
            values = latent[key][indices]
            if values.ndim == 1:
                values = values[:, None]
            feature_parts.append(values.astype(np.float32))
        context_feature_keys = context_feature_keys or []
        if context_feature_keys:
            context_features = _context_feature_matrix(valid_df, latent, indices, context_feature_keys)
            feature_parts.append(context_features)
        features = np.concatenate(feature_parts, axis=1)

        raw_label_columns = [valid_df[column].to_numpy().astype(str) for column in target_columns]
        break_targets = np.stack(
            [
                np.array([0 if label == "none" else 1 for label in raw_labels], dtype=np.int64)
                for raw_labels in raw_label_columns
            ],
            axis=1,
        )
        direction_targets = np.stack(
            [
                np.array(
                    [DIRECTION_LABEL_TO_ID[label] if label in DIRECTION_LABEL_TO_ID else -1 for label in raw_labels],
                    dtype=np.int64,
                )
                for raw_labels in raw_label_columns
            ],
            axis=1,
        )
        recon_targets = np.stack(
            [
                np.array([RECON_LABEL_TO_ID[label] for label in raw_labels], dtype=np.int64)
                for raw_labels in raw_label_columns
            ],
            axis=1,
        )

        metas = [
            Stage1BMeta(
                index=int(row["index"]),
                symbol=str(row["symbol"]),
                timestamp=int(row["timestamp"]),
                raw_direction=str(row[target_columns[0]]),
            )
            for row in valid_df.to_dicts()
        ]
        return cls(
            features=features,
            break_targets=break_targets,
            direction_targets=direction_targets,
            recon_targets=recon_targets,
            metas=metas,
            feature_keys=feature_keys,
            context_feature_keys=context_feature_keys,
            target_columns=target_columns,
            latent_path=latent_path,
            label_path=label_path,
        )

    def __len__(self) -> int:
        return len(self.features)

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[1])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Stage1BMeta]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.break_targets[idx]),
            torch.from_numpy(self.direction_targets[idx]),
            torch.from_numpy(self.recon_targets[idx]),
            self.metas[idx],
        )


def collate_fn_stage1b(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Stage1BMeta]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[Stage1BMeta]]:
    features, break_targets, direction_targets, recon_targets, metas = zip(*batch)
    return (
        torch.stack(features),
        torch.stack(break_targets),
        torch.stack(direction_targets),
        torch.stack(recon_targets),
        list(metas),
    )
