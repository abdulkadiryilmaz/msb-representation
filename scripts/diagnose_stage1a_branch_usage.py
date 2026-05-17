"""Diagnose Stage 1A short/long branch usage after a completed run."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from msb_repr.stage1a.analysis import load_checkpoint_bundle, load_latent_export


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _mean_l2_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.linalg.norm(values.astype(np.float32), axis=1).mean())


def _mean_feature_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.astype(np.float32).std(axis=0).mean())


def _classifier_input_weight_norms(model: torch.nn.Module) -> dict[str, float]:
    first_linear = model.classifier[0]
    if not isinstance(first_linear, torch.nn.Linear):
        raise TypeError("Expected model.classifier[0] to be torch.nn.Linear")

    z_short_dim = int(model.short_encoder.proj.out_features)
    z_long_dim = int(model.long_encoder.proj.out_features)
    weights = first_linear.weight.detach().cpu().numpy().astype(np.float32)
    short_weights = weights[:, :z_short_dim]
    long_weights = weights[:, z_short_dim : z_short_dim + z_long_dim]
    short_norm = float(np.linalg.norm(short_weights))
    long_norm = float(np.linalg.norm(long_weights))
    return {
        "classifier_short_weight_norm": short_norm,
        "classifier_long_weight_norm": long_norm,
        "classifier_long_short_weight_ratio": _safe_ratio(long_norm, short_norm),
    }


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, a_min=1e-8, a_max=None)


def _mean_pairwise_distance(centroids: list[np.ndarray]) -> float:
    if len(centroids) < 2:
        return 0.0
    distances: list[float] = []
    for left, right in combinations(centroids, 2):
        distances.append(float(np.linalg.norm(left - right)))
    return float(np.mean(np.array(distances, dtype=np.float32)))


def _centroid_separation(
    embeddings: np.ndarray,
    groups: np.ndarray,
) -> float:
    normalized = _normalize_rows(embeddings)
    centroids: list[np.ndarray] = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        if np.any(mask):
            centroids.append(normalized[mask].mean(axis=0))
    return _mean_pairwise_distance(centroids)


def branch_usage_summary(
    arrays: dict[str, np.ndarray],
    model: torch.nn.Module,
    history: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    z_short_norm = _mean_l2_norm(arrays["z_short"])
    z_long_norm = _mean_l2_norm(arrays["z_long"])
    z_short_feature_std = _mean_feature_std(arrays["z_short"])
    z_long_feature_std = _mean_feature_std(arrays["z_long"])
    weight_stats = _classifier_input_weight_norms(model)

    effective_ratio = _safe_ratio(
        weight_stats["classifier_long_weight_norm"] * z_long_norm,
        weight_stats["classifier_short_weight_norm"] * z_short_norm,
    )

    rows: dict[str, Any] = {
        "z_short_norm": z_short_norm,
        "z_long_norm": z_long_norm,
        "z_long_short_norm_ratio": _safe_ratio(z_long_norm, z_short_norm),
        "z_short_feature_std": z_short_feature_std,
        "z_long_feature_std": z_long_feature_std,
        **weight_stats,
        "effective_long_short_ratio": effective_ratio,
    }

    labels = arrays["labels"].astype(np.int64)
    symbols = arrays["symbols"].astype(str)
    for embedding_key in ["z_long", "z_long_proj", "z_fused"]:
        if embedding_key not in arrays:
            continue
        label_sep = _centroid_separation(arrays[embedding_key], labels)
        symbol_sep = _centroid_separation(arrays[embedding_key], symbols)
        rows[f"{embedding_key}_label_centroid_distance"] = label_sep
        rows[f"{embedding_key}_symbol_centroid_distance"] = symbol_sep
        rows[f"{embedding_key}_symbol_label_centroid_ratio"] = _safe_ratio(symbol_sep, label_sep)

    if history:
        train_supcon = history.get("train_supcon_loss", [])
        val_macro_f1 = history.get("val_macro_f1", [])
        if train_supcon:
            rows["train_supcon_first"] = float(train_supcon[0])
            rows["train_supcon_min"] = float(min(train_supcon))
            rows["train_supcon_last"] = float(train_supcon[-1])
            if val_macro_f1:
                best_idx = int(np.argmax(np.array(val_macro_f1, dtype=np.float32)))
                if best_idx < len(train_supcon):
                    rows["train_supcon_at_best_val_f1"] = float(train_supcon[best_idx])
                    rows["best_val_f1_epoch"] = best_idx + 1

    return rows


def _write_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    arrays, export_metadata = load_latent_export(args.latent_path)
    bundle = load_checkpoint_bundle(args.checkpoint_dir, checkpoint_name=args.checkpoint_name)
    history_path = args.checkpoint_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else None

    summary = branch_usage_summary(arrays, bundle.model, history=history)
    summary.update(
        {
            "latent_path": str(args.latent_path),
            "checkpoint_dir": str(args.checkpoint_dir),
            "checkpoint_name": args.checkpoint_name,
            "split": export_metadata.get("split", ""),
        }
    )

    output_dir = args.output_dir or args.latent_path.with_suffix("") / "branch_usage_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_single_row_csv(output_dir / "summary.csv", summary)

    print(f"Wrote branch usage diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
