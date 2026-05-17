"""Diagnose domain bucket centroid geometry for Stage 1A latents."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from msb_repr.stage1a.analysis import load_latent_export

try:
    from scripts.analyze_stage1a_latents import _build_bucket_thresholds, _has_domain_summary, _structural_state_fields
except ModuleNotFoundError:  # Direct execution via `python scripts/...`.
    from analyze_stage1a_latents import _build_bucket_thresholds, _has_domain_summary, _structural_state_fields


FOCUS_BUCKET_PAIR_GROUPS = [
    {"group": "clean_vs_borderline", "left": "clean_intact", "right": "borderline_down"},
    {"group": "clean_vs_borderline", "left": "clean_intact", "right": "borderline_up"},
    {"group": "borderline_vs_confirmed", "left": "borderline_down", "right": "bearish_confirmed"},
    {"group": "borderline_vs_confirmed", "left": "borderline_up", "right": "bullish_confirmed"},
    {"group": "wick_sweep_vs_confirmed", "left": "wick_sweep_down", "right": "bearish_confirmed"},
    {"group": "wick_sweep_vs_confirmed", "left": "wick_sweep_up", "right": "bullish_confirmed"},
]

HOLDING_BUCKET_PAIR_GROUPS = [
    {"group": "borderline_holding_vs_confirmed", "left": "borderline_down_holding", "right": "bearish_confirmed"},
    {"group": "borderline_holding_vs_confirmed", "left": "borderline_down_reverted", "right": "bearish_confirmed"},
    {"group": "borderline_holding_vs_confirmed", "left": "borderline_up_holding", "right": "bullish_confirmed"},
    {"group": "borderline_holding_vs_confirmed", "left": "borderline_up_reverted", "right": "bullish_confirmed"},
]


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, a_min=1e-8, a_max=None)


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    return value / max(float(np.linalg.norm(value)), 1e-8)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(1.0 - np.dot(left, right))


def structural_bucket_labels(arrays: dict[str, np.ndarray], min_recent_break_bars: int = 2) -> np.ndarray:
    if not _has_domain_summary(arrays):
        raise ValueError("Domain summary fields are required for bucket geometry diagnostics.")

    thresholds = _build_bucket_thresholds(arrays)
    labels: list[str] = []
    for idx in range(len(arrays["labels"])):
        fields = _structural_state_fields(
            arrays,
            idx,
            min_recent_break_bars=min_recent_break_bars,
            thresholds=thresholds,
            pressure_labels=None,
        )
        confirmed = fields["confirmed_state"]
        maturity = fields["break_maturity"]
        direction = fields["structural_direction"]

        if confirmed == "intact" and maturity == "clean":
            bucket = "clean_intact"
        elif confirmed == "intact" and maturity == "borderline" and direction in {"up", "down"}:
            bucket = f"borderline_{direction}"
        elif confirmed == "intact" and maturity == "wick_sweep" and direction in {"up", "down"}:
            bucket = f"wick_sweep_{direction}"
        elif confirmed == "bullish":
            bucket = "bullish_confirmed"
        elif confirmed == "bearish":
            bucket = "bearish_confirmed"
        else:
            bucket = "other"
        labels.append(bucket)
    return np.array(labels, dtype=object)


def structural_bucket_labels_with_holding(
    arrays: dict[str, np.ndarray],
    min_recent_break_bars: int = 2,
) -> np.ndarray:
    if not _has_domain_summary(arrays):
        raise ValueError("Domain summary fields are required for bucket geometry diagnostics.")

    thresholds = _build_bucket_thresholds(arrays)
    labels: list[str] = []
    for idx in range(len(arrays["labels"])):
        fields = _structural_state_fields(
            arrays,
            idx,
            min_recent_break_bars=min_recent_break_bars,
            thresholds=thresholds,
            pressure_labels=None,
        )
        confirmed = fields["confirmed_state"]
        maturity = fields["break_maturity"]
        direction = fields["structural_direction"]
        holding = fields["holding_status"]

        if confirmed == "intact" and maturity == "borderline" and direction in {"up", "down"}:
            bucket = f"borderline_{direction}_{holding}"
        elif confirmed == "bullish":
            bucket = "bullish_confirmed"
        elif confirmed == "bearish":
            bucket = "bearish_confirmed"
        else:
            bucket = "other"
        labels.append(bucket)
    return np.array(labels, dtype=object)


def bucket_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    min_count: int = 20,
) -> dict[str, np.ndarray]:
    normalized = _normalize_rows(embeddings)
    centroids: dict[str, np.ndarray] = {}
    for bucket in sorted(set(labels.tolist())):
        mask = labels == bucket
        if int(mask.sum()) < min_count:
            continue
        centroids[str(bucket)] = _normalize_vector(normalized[mask].mean(axis=0))
    return centroids


def bucket_distance_rows(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    embedding_keys: list[str],
    pair_groups: list[dict[str, str]],
    min_count: int = 20,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for embedding_key in embedding_keys:
        if embedding_key not in arrays:
            continue
        centroids = bucket_centroids(arrays[embedding_key], labels, min_count=min_count)
        for pair in pair_groups:
            left = pair["left"]
            right = pair["right"]
            if left not in centroids or right not in centroids:
                continue
            rows.append(
                {
                    "embedding_key": embedding_key,
                    "group": pair["group"],
                    "left": left,
                    "right": right,
                    "pair": f"{left} vs {right}",
                    "cosine_distance": cosine_distance(centroids[left], centroids[right]),
                    "left_count": int((labels == left).sum()),
                    "right_count": int((labels == right).sum()),
                }
            )
    return rows


def bucket_count_rows(labels: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    total = len(labels)
    for bucket in sorted(set(labels.tolist())):
        count = int((labels == bucket).sum())
        rows.append({"bucket": str(bucket), "count": count, "share": count / total if total else 0.0})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        if not rows:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for row in rows:
        key = f"{row['embedding_key']}:{row['group']}:{row['pair']}"
        summary[key] = float(row["cosine_distance"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--embedding-keys", nargs="+", default=["z_long", "z_long_proj", "z_fused"])
    parser.add_argument("--min-count", type=int, default=20)
    args = parser.parse_args()

    arrays, export_metadata = load_latent_export(args.latent_path)
    label_config = export_metadata.get("label_config", {})
    min_recent_break_bars = int(label_config.get("min_recent_break_bars", 2))

    labels = structural_bucket_labels(arrays, min_recent_break_bars=min_recent_break_bars)
    holding_labels = structural_bucket_labels_with_holding(arrays, min_recent_break_bars=min_recent_break_bars)
    focus_rows = bucket_distance_rows(
        arrays,
        labels,
        embedding_keys=args.embedding_keys,
        pair_groups=FOCUS_BUCKET_PAIR_GROUPS,
        min_count=args.min_count,
    )
    holding_rows = bucket_distance_rows(
        arrays,
        holding_labels,
        embedding_keys=args.embedding_keys,
        pair_groups=HOLDING_BUCKET_PAIR_GROUPS,
        min_count=args.min_count,
    )

    output_dir = args.output_dir or args.latent_path.with_suffix("") / "bucket_geometry_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "bucket_counts.csv", bucket_count_rows(labels))
    _write_csv(output_dir / "bucket_distance_focus.csv", focus_rows)
    _write_csv(output_dir / "bucket_distance_holding.csv", holding_rows)

    summary = {
        "latent_path": str(args.latent_path),
        "split": export_metadata.get("split", ""),
        "embedding_keys": args.embedding_keys,
        "min_count": args.min_count,
        "focus_distances": _summary_from_rows(focus_rows),
        "holding_distances": _summary_from_rows(holding_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote bucket geometry diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
