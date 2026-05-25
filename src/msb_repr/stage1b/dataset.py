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
DIRECTION3_LABEL_TO_ID = {"none": 0, "bullish": 1, "bearish": 2}
DIRECTION3_ID_TO_LABEL = {idx: label for label, idx in DIRECTION3_LABEL_TO_ID.items()}
EVENT_TYPE_LABEL_TO_ID = {
    "no_event": 0,
    "fresh_break": 1,
    "continuation": 2,
    "reversal": 3,
}
EVENT_TYPE_ID_TO_LABEL = {idx: label for label, idx in EVENT_TYPE_LABEL_TO_ID.items()}
JOINT_EVENT_LABEL_TO_ID = {
    "no_event": 0,
    "fresh_break_bullish": 1,
    "fresh_break_bearish": 2,
    "continuation_bullish": 3,
    "continuation_bearish": 4,
    "reversal_bullish": 5,
    "reversal_bearish": 6,
}
JOINT_EVENT_ID_TO_LABEL = {idx: label for label, idx in JOINT_EVENT_LABEL_TO_ID.items()}
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


@dataclass(frozen=True)
class Stage1BEventMeta:
    index: int
    symbol: str
    timestamp: int
    raw_event_type: str
    raw_event_direction: str


@dataclass(frozen=True)
class Stage1BMultiHeadEventMeta:
    index: int
    symbol: str
    timestamp: int
    raw_first_event_type: str
    raw_first_event_direction: str
    raw_outcome: str
    raw_dominant_direction: str


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


class Stage1BEventSequenceDataset(Dataset):
    """Frozen Stage 1A latent dataset with event-sequence Stage 1B labels."""

    def __init__(
        self,
        features: np.ndarray,
        event_type_targets: np.ndarray,
        event_direction_targets: np.ndarray,
        metas: list[Stage1BEventMeta],
        feature_keys: list[str],
        event_type_column: str,
        event_direction_column: str,
        latent_path: Path,
        label_path: Path,
        context_feature_keys: list[str] | None = None,
    ) -> None:
        if not (len(features) == len(event_type_targets) == len(event_direction_targets) == len(metas)):
            raise ValueError("All Stage 1B event dataset arrays must have the same length")
        self.features = features.astype(np.float32)
        self.event_type_targets = event_type_targets.astype(np.int64)
        self.event_direction_targets = event_direction_targets.astype(np.int64)
        self.metas = metas
        self.feature_keys = feature_keys
        self.context_feature_keys = context_feature_keys or []
        self.event_type_column = event_type_column
        self.event_direction_column = event_direction_column
        self.latent_path = latent_path
        self.label_path = label_path

    @classmethod
    def from_paths(
        cls,
        latent_path: Path,
        label_path: Path,
        feature_keys: list[str],
        event_type_column: str = "h16_event_type",
        event_direction_column: str = "h16_event_direction",
        context_feature_keys: list[str] | None = None,
        drop_labels: set[str] | None = None,
    ) -> "Stage1BEventSequenceDataset":
        drop_labels = drop_labels or {"ambiguous", "insufficient_future"}
        latent = np.load(latent_path, allow_pickle=False)
        label_df = pl.read_parquet(label_path)
        valid_event_types = [label for label in EVENT_TYPE_LABEL_TO_ID if label not in drop_labels]
        valid_df = label_df.filter(pl.col(event_type_column).is_in(valid_event_types))
        valid_df = valid_df.filter(pl.col(event_direction_column).is_in(["none", "bullish", "bearish"]))
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

        raw_event_types = valid_df[event_type_column].to_numpy().astype(str)
        raw_event_directions = valid_df[event_direction_column].to_numpy().astype(str)
        event_type_targets = np.array([EVENT_TYPE_LABEL_TO_ID[label] for label in raw_event_types], dtype=np.int64)
        event_direction_targets = np.array(
            [DIRECTION_LABEL_TO_ID[label] if label in DIRECTION_LABEL_TO_ID else -1 for label in raw_event_directions],
            dtype=np.int64,
        )
        metas = [
            Stage1BEventMeta(
                index=int(row["index"]),
                symbol=str(row["symbol"]),
                timestamp=int(row["timestamp"]),
                raw_event_type=str(row[event_type_column]),
                raw_event_direction=str(row[event_direction_column]),
            )
            for row in valid_df.to_dicts()
        ]
        return cls(
            features=features,
            event_type_targets=event_type_targets,
            event_direction_targets=event_direction_targets,
            metas=metas,
            feature_keys=feature_keys,
            context_feature_keys=context_feature_keys,
            event_type_column=event_type_column,
            event_direction_column=event_direction_column,
            latent_path=latent_path,
            label_path=label_path,
        )

    def __len__(self) -> int:
        return len(self.features)

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[1])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Stage1BEventMeta]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.event_type_targets[idx], dtype=torch.long),
            torch.tensor(self.event_direction_targets[idx], dtype=torch.long),
            self.metas[idx],
        )


def collate_fn_stage1b_event(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, Stage1BEventMeta]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[Stage1BEventMeta]]:
    features, event_type_targets, event_direction_targets, metas = zip(*batch)
    return (
        torch.stack(features),
        torch.stack(event_type_targets),
        torch.stack(event_direction_targets),
        list(metas),
    )


class Stage1BJointEventSequenceDataset(Dataset):
    """Frozen Stage 1A latent dataset with joint event-direction labels."""

    def __init__(
        self,
        features: np.ndarray,
        joint_targets: np.ndarray,
        metas: list[Stage1BEventMeta],
        feature_keys: list[str],
        event_type_column: str,
        event_direction_column: str,
        latent_path: Path,
        label_path: Path,
        context_feature_keys: list[str] | None = None,
    ) -> None:
        if not (len(features) == len(joint_targets) == len(metas)):
            raise ValueError("All Stage 1B joint event dataset arrays must have the same length")
        self.features = features.astype(np.float32)
        self.joint_targets = joint_targets.astype(np.int64)
        self.metas = metas
        self.feature_keys = feature_keys
        self.context_feature_keys = context_feature_keys or []
        self.event_type_column = event_type_column
        self.event_direction_column = event_direction_column
        self.latent_path = latent_path
        self.label_path = label_path

    @classmethod
    def from_paths(
        cls,
        latent_path: Path,
        label_path: Path,
        feature_keys: list[str],
        event_type_column: str = "h16_event_type",
        event_direction_column: str = "h16_event_direction",
        outcome_column: str | None = None,
        dominant_direction_column: str | None = None,
        target_mode: str = "event",
        context_feature_keys: list[str] | None = None,
        drop_labels: set[str] | None = None,
    ) -> "Stage1BJointEventSequenceDataset":
        if target_mode not in {"event", "outcome"}:
            raise ValueError("target_mode must be either 'event' or 'outcome'")
        drop_labels = drop_labels or {"ambiguous", "insufficient_future"}
        latent = np.load(latent_path, allow_pickle=False)
        label_df = pl.read_parquet(label_path)
        if target_mode == "event":
            valid_df = label_df.filter(~pl.col(event_type_column).is_in(list(drop_labels)))
            valid_df = valid_df.filter(pl.col(event_type_column).is_in(list(EVENT_TYPE_LABEL_TO_ID)))
            valid_df = valid_df.filter(pl.col(event_direction_column).is_in(["none", "bullish", "bearish"]))
        else:
            if outcome_column is None or dominant_direction_column is None:
                raise ValueError("outcome mode requires outcome_column and dominant_direction_column")
            valid_df = label_df.filter(~pl.col(outcome_column).is_in(list(drop_labels)))
            valid_df = valid_df.filter(pl.col(dominant_direction_column).is_in(["none", "bullish", "bearish"]))

        joint_labels = []
        keep_mask = []
        if target_mode == "event":
            rows = valid_df.select([event_type_column, event_direction_column]).iter_rows(named=True)
            for row in rows:
                event_type = str(row[event_type_column])
                direction = str(row[event_direction_column])
                if event_type == "no_event":
                    joint_label = "no_event"
                elif direction in DIRECTION_LABEL_TO_ID:
                    joint_label = f"{event_type}_{direction}"
                else:
                    joint_label = ""
                keep_mask.append(joint_label in JOINT_EVENT_LABEL_TO_ID)
                joint_labels.append(joint_label)
        else:
            assert outcome_column is not None
            assert dominant_direction_column is not None
            rows = valid_df.select([outcome_column, dominant_direction_column]).iter_rows(named=True)
            for row in rows:
                outcome = str(row[outcome_column])
                direction = str(row[dominant_direction_column])
                if outcome in {"none", "failed_bullish_break", "failed_bearish_break"}:
                    joint_label = "no_event"
                elif outcome == "bullish_continuation":
                    joint_label = "continuation_bullish"
                elif outcome == "bearish_continuation":
                    joint_label = "continuation_bearish"
                elif outcome == "bullish_to_bearish_reversal":
                    joint_label = "reversal_bearish"
                elif outcome == "bearish_to_bullish_reversal":
                    joint_label = "reversal_bullish"
                elif direction in DIRECTION_LABEL_TO_ID:
                    joint_label = f"fresh_break_{direction}"
                else:
                    joint_label = ""
                keep_mask.append(joint_label in JOINT_EVENT_LABEL_TO_ID)
                joint_labels.append(joint_label)

        valid_df = valid_df.with_columns(pl.Series("_joint_event_label", joint_labels))
        valid_df = valid_df.filter(pl.Series(keep_mask))
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

        raw_joint_labels = valid_df["_joint_event_label"].to_numpy().astype(str)
        joint_targets = np.array([JOINT_EVENT_LABEL_TO_ID[label] for label in raw_joint_labels], dtype=np.int64)
        metas = [
            Stage1BEventMeta(
                index=int(row["index"]),
                symbol=str(row["symbol"]),
                timestamp=int(row["timestamp"]),
                raw_event_type=str(row[event_type_column]),
                raw_event_direction=str(row[event_direction_column]),
            )
            for row in valid_df.to_dicts()
        ]
        return cls(
            features=features,
            joint_targets=joint_targets,
            metas=metas,
            feature_keys=feature_keys,
            context_feature_keys=context_feature_keys,
            event_type_column=event_type_column,
            event_direction_column=event_direction_column,
            latent_path=latent_path,
            label_path=label_path,
        )

    def __len__(self) -> int:
        return len(self.features)

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[1])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, Stage1BEventMeta]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.joint_targets[idx], dtype=torch.long),
            self.metas[idx],
        )


def collate_fn_stage1b_joint_event(
    batch: list[tuple[torch.Tensor, torch.Tensor, Stage1BEventMeta]],
) -> tuple[torch.Tensor, torch.Tensor, list[Stage1BEventMeta]]:
    features, joint_targets, metas = zip(*batch)
    return (
        torch.stack(features),
        torch.stack(joint_targets),
        list(metas),
    )


def _joint_label_from_event(event_type: str, direction: str) -> str:
    if event_type == "no_event":
        return "no_event"
    if direction in DIRECTION_LABEL_TO_ID:
        return f"{event_type}_{direction}"
    return ""


def _joint_label_from_outcome(outcome: str, direction: str) -> str:
    if outcome in {"none", "failed_bullish_break", "failed_bearish_break"}:
        return "no_event"
    if outcome == "bullish_continuation":
        return "continuation_bullish"
    if outcome == "bearish_continuation":
        return "continuation_bearish"
    if outcome == "bullish_to_bearish_reversal":
        return "reversal_bearish"
    if outcome == "bearish_to_bullish_reversal":
        return "reversal_bullish"
    if direction in DIRECTION_LABEL_TO_ID:
        return f"fresh_break_{direction}"
    return ""


class Stage1BMultiHeadEventSequenceDataset(Dataset):
    """Frozen Stage 1A latent dataset with first-event, outcome, and direction labels."""

    def __init__(
        self,
        features: np.ndarray,
        first_event_targets: np.ndarray,
        outcome_targets: np.ndarray,
        dominant_direction_targets: np.ndarray,
        metas: list[Stage1BMultiHeadEventMeta],
        feature_keys: list[str],
        latent_path: Path,
        label_path: Path,
        context_feature_keys: list[str] | None = None,
    ) -> None:
        if not (
            len(features)
            == len(first_event_targets)
            == len(outcome_targets)
            == len(dominant_direction_targets)
            == len(metas)
        ):
            raise ValueError("All Stage 1B multi-head event dataset arrays must have the same length")
        self.features = features.astype(np.float32)
        self.first_event_targets = first_event_targets.astype(np.int64)
        self.outcome_targets = outcome_targets.astype(np.int64)
        self.dominant_direction_targets = dominant_direction_targets.astype(np.int64)
        self.metas = metas
        self.feature_keys = feature_keys
        self.context_feature_keys = context_feature_keys or []
        self.latent_path = latent_path
        self.label_path = label_path

    @classmethod
    def from_paths(
        cls,
        latent_path: Path,
        label_path: Path,
        feature_keys: list[str],
        event_type_column: str = "h16_event_type",
        event_direction_column: str = "h16_event_direction",
        outcome_column: str = "h16_post_break_outcome",
        dominant_direction_column: str = "h16_dominant_forward_direction",
        context_feature_keys: list[str] | None = None,
        drop_labels: set[str] | None = None,
    ) -> "Stage1BMultiHeadEventSequenceDataset":
        drop_labels = drop_labels or {"ambiguous", "insufficient_future"}
        latent = np.load(latent_path, allow_pickle=False)
        label_df = pl.read_parquet(label_path)
        valid_df = label_df.filter(~pl.col(event_type_column).is_in(list(drop_labels)))
        valid_df = valid_df.filter(~pl.col(outcome_column).is_in(list(drop_labels)))
        valid_df = valid_df.filter(pl.col(event_type_column).is_in(list(EVENT_TYPE_LABEL_TO_ID)))
        valid_df = valid_df.filter(pl.col(event_direction_column).is_in(list(DIRECTION3_LABEL_TO_ID)))
        valid_df = valid_df.filter(pl.col(dominant_direction_column).is_in(list(DIRECTION3_LABEL_TO_ID)))

        first_labels = []
        outcome_labels = []
        keep_mask = []
        rows = valid_df.select(
            [event_type_column, event_direction_column, outcome_column, dominant_direction_column]
        ).iter_rows(named=True)
        for row in rows:
            first_label = _joint_label_from_event(str(row[event_type_column]), str(row[event_direction_column]))
            outcome_label = _joint_label_from_outcome(str(row[outcome_column]), str(row[dominant_direction_column]))
            keep = first_label in JOINT_EVENT_LABEL_TO_ID and outcome_label in JOINT_EVENT_LABEL_TO_ID
            first_labels.append(first_label)
            outcome_labels.append(outcome_label)
            keep_mask.append(keep)

        valid_df = valid_df.with_columns(
            [
                pl.Series("_first_event_joint_label", first_labels),
                pl.Series("_outcome_joint_label", outcome_labels),
            ]
        )
        valid_df = valid_df.filter(pl.Series(keep_mask))
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

        first_event_targets = np.array(
            [JOINT_EVENT_LABEL_TO_ID[label] for label in valid_df["_first_event_joint_label"].to_numpy().astype(str)],
            dtype=np.int64,
        )
        outcome_targets = np.array(
            [JOINT_EVENT_LABEL_TO_ID[label] for label in valid_df["_outcome_joint_label"].to_numpy().astype(str)],
            dtype=np.int64,
        )
        dominant_direction_targets = np.array(
            [DIRECTION3_LABEL_TO_ID[label] for label in valid_df[dominant_direction_column].to_numpy().astype(str)],
            dtype=np.int64,
        )
        metas = [
            Stage1BMultiHeadEventMeta(
                index=int(row["index"]),
                symbol=str(row["symbol"]),
                timestamp=int(row["timestamp"]),
                raw_first_event_type=str(row[event_type_column]),
                raw_first_event_direction=str(row[event_direction_column]),
                raw_outcome=str(row[outcome_column]),
                raw_dominant_direction=str(row[dominant_direction_column]),
            )
            for row in valid_df.to_dicts()
        ]
        return cls(
            features=features,
            first_event_targets=first_event_targets,
            outcome_targets=outcome_targets,
            dominant_direction_targets=dominant_direction_targets,
            metas=metas,
            feature_keys=feature_keys,
            context_feature_keys=context_feature_keys,
            latent_path=latent_path,
            label_path=label_path,
        )

    def __len__(self) -> int:
        return len(self.features)

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[1])

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Stage1BMultiHeadEventMeta]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.first_event_targets[idx], dtype=torch.long),
            torch.tensor(self.outcome_targets[idx], dtype=torch.long),
            torch.tensor(self.dominant_direction_targets[idx], dtype=torch.long),
            self.metas[idx],
        )


def collate_fn_stage1b_multihead_event(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Stage1BMultiHeadEventMeta]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[Stage1BMultiHeadEventMeta]]:
    features, first_event_targets, outcome_targets, dominant_direction_targets, metas = zip(*batch)
    return (
        torch.stack(features),
        torch.stack(first_event_targets),
        torch.stack(outcome_targets),
        torch.stack(dominant_direction_targets),
        list(metas),
    )
