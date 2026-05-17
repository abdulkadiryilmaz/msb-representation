"""Checkpoint loading and latent analysis helpers for Stage 1A."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from msb_repr.data.normalizer import Normalizer
from msb_repr.stage1a.config import Stage1ADatasetSpec
from msb_repr.stage1a.dataset import Stage1ADualWindowDataset, collate_fn_stage1a
from msb_repr.stage1a.labels import find_last_pivot_high, find_last_pivot_low
from msb_repr.stage1a.model import Stage1AModel, get_device

LABEL_NAMES = {0: "intact", 1: "bullish", 2: "bearish"}
PRESSURE_LABEL_NAMES = {
    -1: "non_intact",
    0: "neutral",
    1: "up_pressure",
    2: "down_pressure",
    3: "mixed_pressure",
    4: "non_intact",
}
FACTOR_TARGET_NAMES = {
    0: "clean_intact",
    1: "wick_sweep_up",
    2: "wick_sweep_down",
    3: "borderline_up",
    4: "borderline_down",
    5: "borderline_mixed",
    6: "bullish_confirmed",
    7: "bearish_confirmed",
}
MATURITY_TARGET_NAMES = {0: "clean", 1: "wick_sweep", 2: "borderline", 3: "confirmed"}


@dataclass
class Stage1ACheckpointBundle:
    checkpoint_dir: Path
    checkpoint_path: Path
    model: Stage1AModel
    short_norm: Normalizer
    long_norm: Normalizer
    metadata: dict[str, Any]
    device: torch.device


def apply_saved_normalizers(
    dataset: Stage1ADualWindowDataset,
    short_norm: Normalizer,
    long_norm: Normalizer,
) -> Stage1ADualWindowDataset:
    dataset.short_windows = short_norm.transform(dataset.short_windows).astype(np.float32)
    dataset.long_windows = long_norm.transform(dataset.long_windows).astype(np.float32)
    return dataset


def load_checkpoint_bundle(
    checkpoint_dir: Path,
    checkpoint_name: str = "best.pt",
    device: torch.device | None = None,
) -> Stage1ACheckpointBundle:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / checkpoint_name
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
    train_config_path = checkpoint_dir / "train_config.json"
    train_config = json.loads(train_config_path.read_text()) if train_config_path.exists() else {}

    model = Stage1AModel(
        short_input_channels=int(metadata["short_input_channels"]),
        long_input_channels=int(metadata["long_input_channels"]),
        z_short=int(metadata["z_short"]),
        z_long=int(metadata["z_long"]),
        num_pressure_classes=(
            int(metadata["num_pressure_classes"]) if metadata.get("num_pressure_classes") is not None else None
        ),
        num_maturity_classes=(
            int(metadata["num_maturity_classes"]) if metadata.get("num_maturity_classes") is not None else None
        ),
        num_long_aux_classes=(
            int(metadata["num_long_aux_classes"]) if metadata.get("num_long_aux_classes") is not None else None
        ),
        pressure_head_input=str(metadata.get("pressure_head_input", "z_fused")),
        projection_dim=int(metadata.get("projection_dim", 64)),
        long_projection_dim=(
            int(metadata["long_projection_dim"]) if metadata.get("long_projection_dim") is not None else None
        ),
        dropout=float(train_config.get("dropout", 0.1)),
    )
    model_device = device or get_device()
    state = torch.load(checkpoint_path, map_location=model_device)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(model_device)
    model.eval()

    return Stage1ACheckpointBundle(
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        model=model,
        short_norm=Normalizer.load(checkpoint_dir / "short_norm_params.json"),
        long_norm=Normalizer.load(checkpoint_dir / "long_norm_params.json"),
        metadata=metadata,
        device=model_device,
    )


def load_split_dataset(
    dataset_root: Path,
    split: str,
    bundle: Stage1ACheckpointBundle,
    symbols: list[str] | None = None,
    normalize: bool = True,
) -> Stage1ADualWindowDataset:
    dataset = Stage1ADualWindowDataset.from_root(dataset_root, split=split, symbols=symbols)
    if normalize:
        return apply_saved_normalizers(dataset, bundle.short_norm, bundle.long_norm)
    return dataset


def build_domain_summary(
    dataset: Stage1ADualWindowDataset,
    spec: Stage1ADatasetSpec,
) -> dict[str, np.ndarray]:
    feature_index = {name: idx for idx, name in enumerate(spec.short_feature_columns)}
    required = {"open", "high", "low", "close", "hl_range", "atr_norm"}
    missing = required.difference(feature_index)
    if missing:
        raise ValueError(f"Stage1A short feature columns missing required fields: {sorted(missing)}")

    labels_cfg = spec.labels
    short_windows = dataset.short_windows
    opens = short_windows[:, feature_index["open"], :]
    highs = short_windows[:, feature_index["high"], :]
    lows = short_windows[:, feature_index["low"], :]
    closes = short_windows[:, feature_index["close"], :]
    hl_range = short_windows[:, feature_index["hl_range"], :]
    atr_norm = short_windows[:, feature_index["atr_norm"], :]

    struct_end = labels_cfg.structure_bars
    recent_end = struct_end + labels_cfg.recent_bars

    struct_highs = highs[:, :struct_end]
    struct_lows = lows[:, :struct_end]
    recent_opens = opens[:, struct_end:recent_end]
    recent_highs = highs[:, struct_end:recent_end]
    recent_lows = lows[:, struct_end:recent_end]
    recent_closes = closes[:, struct_end:recent_end]
    recent_hl = hl_range[:, struct_end:recent_end]
    recent_atr = atr_norm[:, struct_end:recent_end]

    num_samples = len(dataset)
    bull_levels = np.full(num_samples, np.nan, dtype=np.float32)
    bear_levels = np.full(num_samples, np.nan, dtype=np.float32)
    bull_close_count = np.zeros(num_samples, dtype=np.int64)
    bear_close_count = np.zeros(num_samples, dtype=np.int64)
    bull_wick_count = np.zeros(num_samples, dtype=np.int64)
    bear_wick_count = np.zeros(num_samples, dtype=np.int64)
    bull_final_excess = np.full(num_samples, np.nan, dtype=np.float32)
    bear_final_excess = np.full(num_samples, np.nan, dtype=np.float32)
    bull_max_excess = np.full(num_samples, np.nan, dtype=np.float32)
    bear_max_excess = np.full(num_samples, np.nan, dtype=np.float32)

    for idx in range(num_samples):
        atr = float(np.mean(struct_highs[idx] - struct_lows[idx])) if struct_end else 0.0
        ref_close = float(closes[idx, struct_end - 1]) if closes[idx, struct_end - 1] > 0 else 1.0
        effective_break_pct = max(labels_cfg.min_break_pct, labels_cfg.atr_factor * atr / ref_close)

        swing_high = find_last_pivot_high(struct_highs[idx], labels_cfg.pivot_n)
        if swing_high is not None:
            bull_level = swing_high * (1.0 + effective_break_pct)
            bull_levels[idx] = bull_level
            bull_close_excess = recent_closes[idx] - bull_level
            bull_high_excess = recent_highs[idx] - bull_level
            bull_close_count[idx] = int(np.sum(bull_close_excess > 0.0))
            bull_wick_count[idx] = int(np.sum((bull_high_excess > 0.0) & (bull_close_excess <= 0.0)))
            bull_final_excess[idx] = float(bull_close_excess[-1] / max(abs(bull_level), 1e-8))
            bull_max_excess[idx] = float(np.max(bull_high_excess) / max(abs(bull_level), 1e-8))

        swing_low = find_last_pivot_low(struct_lows[idx], labels_cfg.pivot_n)
        if swing_low is not None:
            bear_level = swing_low * (1.0 - effective_break_pct)
            bear_levels[idx] = bear_level
            bear_close_excess = bear_level - recent_closes[idx]
            bear_low_excess = bear_level - recent_lows[idx]
            bear_close_count[idx] = int(np.sum(bear_close_excess > 0.0))
            bear_wick_count[idx] = int(np.sum((bear_low_excess > 0.0) & (bear_close_excess <= 0.0)))
            bear_final_excess[idx] = float(bear_close_excess[-1] / max(abs(bear_level), 1e-8))
            bear_max_excess[idx] = float(np.max(bear_low_excess) / max(abs(bear_level), 1e-8))

    recent_body = np.abs(recent_closes - recent_opens)
    recent_range = recent_highs - recent_lows
    return {
        "recent_atr_mean": recent_atr.mean(axis=1).astype(np.float32),
        "recent_hl_mean": recent_hl.mean(axis=1).astype(np.float32),
        "recent_body_mean": recent_body.mean(axis=1).astype(np.float32),
        "recent_range_mean": recent_range.mean(axis=1).astype(np.float32),
        "recent_close_change": (recent_closes[:, -1] - recent_closes[:, 0]).astype(np.float32),
        "recent_close_std": recent_closes.std(axis=1).astype(np.float32),
        "bull_level": bull_levels,
        "bear_level": bear_levels,
        "bull_close_count": bull_close_count,
        "bear_close_count": bear_close_count,
        "bull_wick_count": bull_wick_count,
        "bear_wick_count": bear_wick_count,
        "bull_final_excess": bull_final_excess,
        "bear_final_excess": bear_final_excess,
        "bull_max_excess": bull_max_excess,
        "bear_max_excess": bear_max_excess,
    }


def _nanquantile(values: np.ndarray, q: float, fallback: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return fallback
    return float(np.quantile(finite, q))


def _factor_thresholds(domain_summary: dict[str, np.ndarray]) -> dict[str, float]:
    directional_excess = np.concatenate(
        [
            np.abs(domain_summary["bull_final_excess"][np.isfinite(domain_summary["bull_final_excess"])]),
            np.abs(domain_summary["bear_final_excess"][np.isfinite(domain_summary["bear_final_excess"])]),
        ]
    )
    return {
        "high_vol_atr": _nanquantile(domain_summary["recent_atr_mean"], 0.85),
        "high_vol_hl": _nanquantile(domain_summary["recent_hl_mean"], 0.85),
        "true_break_excess": _nanquantile(directional_excess, 0.75, fallback=0.0),
    }


def build_structural_training_targets(
    labels: np.ndarray,
    domain_summary: dict[str, np.ndarray],
    min_recent_break_bars: int,
) -> dict[str, np.ndarray]:
    """Build factor-aware training targets from Stage 1A domain summary arrays."""
    thresholds = _factor_thresholds(domain_summary)
    factor = np.zeros(len(labels), dtype=np.int64)
    pressure = np.full(len(labels), 4, dtype=np.int64)  # non_intact
    maturity = np.full(len(labels), 3, dtype=np.int64)  # confirmed

    for idx, raw_label in enumerate(labels.astype(np.int64)):
        bull_close_count = int(domain_summary["bull_close_count"][idx])
        bear_close_count = int(domain_summary["bear_close_count"][idx])
        bull_wick_count = int(domain_summary["bull_wick_count"][idx])
        bear_wick_count = int(domain_summary["bear_wick_count"][idx])
        bull_final_excess = (
            float(domain_summary["bull_final_excess"][idx])
            if np.isfinite(domain_summary["bull_final_excess"][idx])
            else np.nan
        )
        bear_final_excess = (
            float(domain_summary["bear_final_excess"][idx])
            if np.isfinite(domain_summary["bear_final_excess"][idx])
            else np.nan
        )
        high_vol = (
            float(domain_summary["recent_atr_mean"][idx]) >= thresholds["high_vol_atr"]
            or float(domain_summary["recent_hl_mean"][idx]) >= thresholds["high_vol_hl"]
        )

        if raw_label == 0:
            bull_near = bull_close_count == min_recent_break_bars - 1
            bear_near = bear_close_count == min_recent_break_bars - 1
            bull_active = bull_near or bull_wick_count > 0
            bear_active = bear_near or bear_wick_count > 0

            if bull_active and bear_active:
                direction = "mixed"
            elif bull_active:
                direction = "up"
            elif bear_active:
                direction = "down"
            else:
                direction = "none"

            if direction == "up":
                pressure[idx] = 1
            elif direction == "down":
                pressure[idx] = 2
            elif direction == "mixed":
                pressure[idx] = 3
            else:
                pressure[idx] = 0

            if bull_near or bear_near:
                maturity[idx] = 2
                if direction == "up":
                    factor[idx] = 3
                elif direction == "down":
                    factor[idx] = 4
                else:
                    factor[idx] = 5
            elif bull_wick_count > 0 or bear_wick_count > 0:
                maturity[idx] = 1
                if direction == "up":
                    factor[idx] = 1
                elif direction == "down":
                    factor[idx] = 2
                else:
                    factor[idx] = 5
            elif high_vol and direction in {"up", "down"}:
                maturity[idx] = 1
                factor[idx] = 1 if direction == "up" else 2
            else:
                maturity[idx] = 0
                factor[idx] = 0
        elif raw_label == 1:
            pressure[idx] = 4
            maturity[idx] = 3
            factor[idx] = 6
            if bull_close_count < min_recent_break_bars or (
                np.isfinite(bull_final_excess) and bull_final_excess < thresholds["true_break_excess"]
            ):
                factor[idx] = 6
        elif raw_label == 2:
            pressure[idx] = 4
            maturity[idx] = 3
            factor[idx] = 7
            if bear_close_count < min_recent_break_bars or (
                np.isfinite(bear_final_excess) and bear_final_excess < thresholds["true_break_excess"]
            ):
                factor[idx] = 7

    return {
        "factor_targets": factor,
        "pressure_targets": pressure,
        "maturity_targets": maturity,
    }


def export_latents(
    model: Stage1AModel,
    dataset: Stage1ADualWindowDataset,
    device: torch.device,
    batch_size: int = 256,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_stage1a,
    )
    outputs_acc: dict[str, list[np.ndarray]] = {
        "z_short": [],
        "z_long": [],
        "z_fused": [],
        "z_proj": [],
        "logits": [],
        "probs": [],
        "labels": [],
        "preds": [],
        "timestamps": [],
        "symbols": [],
    }

    with torch.no_grad():
        for short_x, long_x, labels, metas in loader:
            short_x = short_x.to(device)
            long_x = long_x.to(device)
            batch_outputs = model(short_x, long_x)
            probs = torch.softmax(batch_outputs["logits"], dim=1)
            preds = probs.argmax(dim=1)

            embedding_keys = ["z_short", "z_long", "z_fused", "z_proj", "logits"]
            if "z_long_proj" in batch_outputs:
                outputs_acc.setdefault("z_long_proj", [])
                embedding_keys.append("z_long_proj")
            for key in embedding_keys:
                outputs_acc[key].append(batch_outputs[key].detach().cpu().numpy().astype(np.float32))
            outputs_acc["probs"].append(probs.detach().cpu().numpy().astype(np.float32))
            outputs_acc["labels"].append(labels.numpy().astype(np.int64))
            outputs_acc["preds"].append(preds.detach().cpu().numpy().astype(np.int64))
            outputs_acc["timestamps"].append(np.array([meta.timestamp for meta in metas], dtype=np.int64))
            outputs_acc["symbols"].append(np.array([meta.symbol for meta in metas]))

    exported = {key: np.concatenate(parts, axis=0) for key, parts in outputs_acc.items()}
    if extra_arrays is not None:
        exported.update(extra_arrays)
    exported["label_names"] = np.array([LABEL_NAMES[idx] for idx in range(len(LABEL_NAMES))])
    return exported


def save_latent_export(
    output_path: Path,
    exported: dict[str, np.ndarray],
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    payload = dict(exported)
    if extra_metadata is not None:
        payload["export_metadata_json"] = np.array(json.dumps(extra_metadata, indent=2))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def load_latent_export(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    loaded = np.load(path, allow_pickle=False)
    arrays = {key: loaded[key] for key in loaded.files if key != "export_metadata_json"}
    metadata: dict[str, Any] = {}
    if "export_metadata_json" in loaded.files:
        metadata = json.loads(loaded["export_metadata_json"].item())
    return arrays, metadata


def _require_pressure_inputs(arrays: dict[str, np.ndarray]) -> None:
    required = {
        "labels",
        "timestamps",
        "symbols",
        "bull_close_count",
        "bear_close_count",
        "bull_wick_count",
        "bear_wick_count",
        "bull_final_excess",
        "bear_final_excess",
        "bull_max_excess",
        "bear_max_excess",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"Pressure labels require latent export domain summary arrays: {missing}")


def _pressure_component_score(
    close_count: np.ndarray,
    wick_count: np.ndarray,
    final_excess: np.ndarray,
    max_excess: np.ndarray,
    min_recent_break_bars: int,
    recent_bars: int,
    near_margin: float,
) -> np.ndarray:
    close_target = max(min_recent_break_bars - 1, 1)
    close_ratio = np.clip(close_count.astype(np.float32) / float(close_target), 0.0, 1.0)
    wick_ratio = np.clip(wick_count.astype(np.float32) / float(max(recent_bars, 1)), 0.0, 1.0)

    final_proximity = np.zeros_like(close_ratio, dtype=np.float32)
    max_proximity = np.zeros_like(close_ratio, dtype=np.float32)

    finite_final = np.isfinite(final_excess)
    finite_max = np.isfinite(max_excess)
    final_proximity[finite_final] = np.clip(1.0 + (final_excess[finite_final] / near_margin), 0.0, 1.0)
    max_proximity[finite_max] = np.clip(1.0 + (max_excess[finite_max] / near_margin), 0.0, 1.0)

    return (
        0.40 * close_ratio
        + 0.15 * wick_ratio
        + 0.25 * final_proximity
        + 0.20 * max_proximity
    ).astype(np.float32)


def compute_intact_pressure_labels(
    arrays: dict[str, np.ndarray],
    label_config: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    _require_pressure_inputs(arrays)
    cfg = label_config or {}
    min_recent_break_bars = int(cfg.get("min_recent_break_bars", 2))
    recent_bars = int(cfg.get("recent_bars", 12))
    near_margin = float(cfg.get("pressure_near_margin", max(float(cfg.get("min_break_pct", 0.002)) * 1.5, 0.0015)))
    min_pressure_score = float(cfg.get("pressure_min_score", 0.55))
    dominance_margin = float(cfg.get("pressure_dominance_margin", 0.10))

    up_score = _pressure_component_score(
        close_count=arrays["bull_close_count"],
        wick_count=arrays["bull_wick_count"],
        final_excess=arrays["bull_final_excess"],
        max_excess=arrays["bull_max_excess"],
        min_recent_break_bars=min_recent_break_bars,
        recent_bars=recent_bars,
        near_margin=near_margin,
    )
    down_score = _pressure_component_score(
        close_count=arrays["bear_close_count"],
        wick_count=arrays["bear_wick_count"],
        final_excess=arrays["bear_final_excess"],
        max_excess=arrays["bear_max_excess"],
        min_recent_break_bars=min_recent_break_bars,
        recent_bars=recent_bars,
        near_margin=near_margin,
    )

    labels = arrays["labels"].astype(np.int64)
    intact_mask = labels == 0
    dominant_score = np.maximum(up_score, down_score)
    score_gap = np.abs(up_score - down_score).astype(np.float32)

    pressure_label_ids = np.full(len(labels), -1, dtype=np.int64)
    neutral_mask = intact_mask & ((dominant_score < min_pressure_score) | (score_gap < dominance_margin))
    pressure_label_ids[neutral_mask] = 0
    pressure_label_ids[intact_mask & ~neutral_mask & (up_score > down_score)] = 1
    pressure_label_ids[intact_mask & ~neutral_mask & (down_score > up_score)] = 2

    pressure_label_names = np.array([PRESSURE_LABEL_NAMES[int(idx)] for idx in pressure_label_ids], dtype="<U16")

    return {
        "intact_mask": intact_mask.astype(bool),
        "pressure_label_id": pressure_label_ids,
        "pressure_label": pressure_label_names,
        "up_pressure_score": up_score,
        "down_pressure_score": down_score,
        "pressure_score_gap": score_gap,
        "pressure_near_margin": np.full(len(labels), near_margin, dtype=np.float32),
        "pressure_min_score": np.full(len(labels), min_pressure_score, dtype=np.float32),
        "pressure_dominance_margin": np.full(len(labels), dominance_margin, dtype=np.float32),
    }


def build_pressure_label_frame(
    arrays: dict[str, np.ndarray],
    label_config: dict[str, Any] | None = None,
) -> pl.DataFrame:
    derived = compute_intact_pressure_labels(arrays, label_config=label_config)
    labels = arrays["labels"].astype(np.int64)
    return pl.DataFrame(
        {
            "index": np.arange(len(labels), dtype=np.int64),
            "symbol": arrays["symbols"].astype(str),
            "timestamp": arrays["timestamps"].astype(np.int64),
            "base_label": labels,
            "base_label_name": np.array([LABEL_NAMES[int(idx)] for idx in labels], dtype="<U8"),
            "is_intact_base_label": derived["intact_mask"],
            "pressure_label_id": derived["pressure_label_id"].astype(np.int64),
            "pressure_label": derived["pressure_label"].astype(str),
            "up_pressure_score": derived["up_pressure_score"].astype(np.float32),
            "down_pressure_score": derived["down_pressure_score"].astype(np.float32),
            "pressure_score_gap": derived["pressure_score_gap"].astype(np.float32),
            "bull_close_count": arrays["bull_close_count"].astype(np.int64),
            "bear_close_count": arrays["bear_close_count"].astype(np.int64),
            "bull_wick_count": arrays["bull_wick_count"].astype(np.int64),
            "bear_wick_count": arrays["bear_wick_count"].astype(np.int64),
            "bull_final_excess": arrays["bull_final_excess"].astype(np.float32),
            "bear_final_excess": arrays["bear_final_excess"].astype(np.float32),
            "bull_max_excess": arrays["bull_max_excess"].astype(np.float32),
            "bear_max_excess": arrays["bear_max_excess"].astype(np.float32),
        }
    )


def compute_pca_projection(embeddings: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise RuntimeError("PCA için scikit-learn gerekli. `uv pip install -e \".[viz]\"` çalıştırın.") from exc

    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)
    info = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    return coords.astype(np.float32), info


def compute_umap_projection(
    embeddings: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("UMAP için `umap-learn` gerekli. `uv pip install -e \".[viz]\"` çalıştırın.") from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings).astype(np.float32)


def compute_cosine_neighbors(
    embeddings: np.ndarray,
    top_k: int = 10,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape (N, D)")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    normalized = embeddings.astype(np.float32)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    normalized = normalized / np.clip(norms, a_min=1e-8, a_max=None)

    num_samples = normalized.shape[0]
    effective_k = min(top_k + 1, num_samples)
    neighbor_indices = np.empty((num_samples, effective_k - 1), dtype=np.int64)
    neighbor_scores = np.empty((num_samples, effective_k - 1), dtype=np.float32)

    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        sims = normalized[start:end] @ normalized.T
        sims[np.arange(end - start), np.arange(start, end)] = -np.inf
        top_idx = np.argpartition(-sims, kth=effective_k - 1, axis=1)[:, :effective_k]
        top_scores = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)[:, : effective_k - 1]
        neighbor_indices[start:end] = np.take_along_axis(top_idx, order, axis=1)
        neighbor_scores[start:end] = np.take_along_axis(top_scores, order, axis=1)

    return neighbor_indices, neighbor_scores


def compute_hard_case_scores(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    max_prob = sorted_probs[:, -1]
    second_prob = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros_like(max_prob)
    margin = max_prob - second_prob
    neighbor_agreement = (labels[neighbor_indices] == labels[:, None]).mean(axis=1)
    misclassified = (preds != labels).astype(np.float32)
    return misclassified * 2.0 + (1.0 - margin.astype(np.float32)) + (1.0 - neighbor_agreement.astype(np.float32))


def write_rows_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
