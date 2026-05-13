"""Diagnose Stage 1A fused-latent geometry vs classifier-head behavior."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from msb_repr.stage1a.analysis import LABEL_NAMES, compute_cosine_neighbors, load_latent_export

try:
    from scripts.analyze_stage1a_latents import (
        _assign_domain_bucket,
        _build_bucket_thresholds,
        _structural_state_fields,
    )
except ModuleNotFoundError:  # Direct execution via `python scripts/...`.
    from analyze_stage1a_latents import (
        _assign_domain_bucket,
        _build_bucket_thresholds,
        _structural_state_fields,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        if not rows:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _prob_margin(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    return (sorted_probs[:, -1] - sorted_probs[:, -2]).astype(np.float32)


def _family_key(fields: dict[str, str]) -> str:
    return " / ".join(
        [
            fields["confirmed_state"],
            fields["pressure_state"],
            fields["break_maturity"],
            fields["structural_direction"],
            fields["holding_status"],
        ]
    )


def _row_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.array(values, dtype=np.float32)))


def _build_sample_rows(
    arrays: dict[str, np.ndarray],
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    min_recent_break_bars: int,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    margins = _prob_margin(arrays["probs"])
    rows: list[dict[str, Any]] = []
    for idx in range(len(arrays["labels"])):
        fields = _structural_state_fields(arrays, idx, min_recent_break_bars, thresholds)
        if fields["confirmed_state"] != "intact" or fields["break_maturity"] != "borderline":
            continue

        label = int(arrays["labels"][idx])
        pred = int(arrays["preds"][idx])
        neighbor_fields = [
            _structural_state_fields(arrays, int(nn_idx), min_recent_break_bars, thresholds)
            for nn_idx in neighbor_indices[idx]
        ]
        neighbor_labels = arrays["labels"][neighbor_indices[idx]]
        neighbor_symbols = arrays["symbols"][neighbor_indices[idx]].astype(str)
        same_symbol = neighbor_symbols == str(arrays["symbols"][idx])
        intact_neighbors = sum(1 for item in neighbor_fields if item["confirmed_state"] == "intact")
        confirmed_neighbors = sum(1 for item in neighbor_fields if item["confirmed_state"] in {"bullish", "bearish"})
        same_direction_neighbors = sum(
            1
            for item in neighbor_fields
            if fields["structural_direction"] in {"up", "down"}
            and item["structural_direction"] == fields["structural_direction"]
        )
        family_counts = Counter(_family_key(item) for item in neighbor_fields)
        top_family, top_family_count = family_counts.most_common(1)[0]

        rows.append(
            {
                "index": idx,
                "symbol": str(arrays["symbols"][idx]),
                "timestamp": int(arrays["timestamps"][idx]),
                "label_name": LABEL_NAMES[label],
                "pred_name": LABEL_NAMES[pred],
                "is_correct": label == pred,
                "confidence": float(arrays["probs"][idx].max()),
                "prob_margin": float(margins[idx]),
                "p_intact": float(arrays["probs"][idx, 0]),
                "p_bullish": float(arrays["probs"][idx, 1]),
                "p_bearish": float(arrays["probs"][idx, 2]),
                "bucket": _assign_domain_bucket(arrays, idx, min_recent_break_bars, thresholds)[0],
                **fields,
                "bull_close_count": int(arrays["bull_close_count"][idx]),
                "bear_close_count": int(arrays["bear_close_count"][idx]),
                "bull_wick_count": int(arrays["bull_wick_count"][idx]),
                "bear_wick_count": int(arrays["bear_wick_count"][idx]),
                "bull_final_excess": (
                    "" if not np.isfinite(arrays["bull_final_excess"][idx]) else float(arrays["bull_final_excess"][idx])
                ),
                "bear_final_excess": (
                    "" if not np.isfinite(arrays["bear_final_excess"][idx]) else float(arrays["bear_final_excess"][idx])
                ),
                "nn_label_agreement": float((neighbor_labels == label).mean()),
                "nn_same_symbol_rate": float(same_symbol.mean()),
                "nn_intact_rate": intact_neighbors / neighbor_indices.shape[1],
                "nn_confirmed_rate": confirmed_neighbors / neighbor_indices.shape[1],
                "nn_same_direction_rate": same_direction_neighbors / neighbor_indices.shape[1],
                "nn_top_family": top_family,
                "nn_top_family_share": top_family_count / neighbor_indices.shape[1],
                "nn_top_indices": "|".join(str(int(v)) for v in neighbor_indices[idx]),
                "nn_top_scores": "|".join(f"{float(v):.6f}" for v in neighbor_scores[idx]),
            }
        )
    return rows


def _group_rows(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        key = (
            "correct" if row["is_correct"] else "misclassified",
            str(row["pressure_state"]),
            str(row["structural_direction"]),
            str(row["holding_status"]),
            str(row["pred_name"]),
        )
        groups[key].append(row)

    rows: list[dict[str, Any]] = []
    for (status, pressure, direction, holding, pred_name), items in sorted(groups.items()):
        rows.append(
            {
                "status": status,
                "pressure_state": pressure,
                "structural_direction": direction,
                "holding_status": holding,
                "pred_name": pred_name,
                "count": len(items),
                "avg_confidence": _row_mean([float(item["confidence"]) for item in items]),
                "avg_prob_margin": _row_mean([float(item["prob_margin"]) for item in items]),
                "avg_p_intact": _row_mean([float(item["p_intact"]) for item in items]),
                "avg_p_bullish": _row_mean([float(item["p_bullish"]) for item in items]),
                "avg_p_bearish": _row_mean([float(item["p_bearish"]) for item in items]),
                "avg_nn_label_agreement": _row_mean([float(item["nn_label_agreement"]) for item in items]),
                "avg_nn_intact_rate": _row_mean([float(item["nn_intact_rate"]) for item in items]),
                "avg_nn_confirmed_rate": _row_mean([float(item["nn_confirmed_rate"]) for item in items]),
                "avg_nn_same_direction_rate": _row_mean([float(item["nn_same_direction_rate"]) for item in items]),
                "avg_nn_same_symbol_rate": _row_mean([float(item["nn_same_symbol_rate"]) for item in items]),
            }
        )
    rows.sort(key=lambda row: (row["status"], row["pressure_state"], row["pred_name"], -int(row["count"])))
    return rows


def _neighbor_family_rows(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()
    for row in sample_rows:
        status = "correct" if row["is_correct"] else "misclassified"
        family = str(row["nn_top_family"])
        counts[(status, family)] += 1
        totals[status] += 1

    rows = []
    for (status, family), count in sorted(counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1])):
        rows.append(
            {
                "status": status,
                "nn_top_family": family,
                "count": count,
                "share": count / totals[status],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-key", default="z_fused")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    arrays, export_metadata = load_latent_export(args.latent_path)
    label_config = export_metadata.get("label_config", {})
    min_recent_break_bars = int(label_config.get("min_recent_break_bars", 2))
    thresholds = _build_bucket_thresholds(arrays)
    neighbor_indices, neighbor_scores = compute_cosine_neighbors(arrays[args.embedding_key], top_k=args.top_k)

    sample_rows = _build_sample_rows(
        arrays,
        neighbor_indices,
        neighbor_scores,
        min_recent_break_bars=min_recent_break_bars,
        thresholds=thresholds,
    )
    group_rows = _group_rows(sample_rows)
    family_rows = _neighbor_family_rows(sample_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "borderline_samples.csv", sample_rows)
    _write_csv(args.output_dir / "head_group_summary.csv", group_rows)
    _write_csv(args.output_dir / "neighbor_family_summary.csv", family_rows)

    total = len(sample_rows)
    misclassified = sum(1 for row in sample_rows if not row["is_correct"])
    summary = {
        "latent_path": str(args.latent_path),
        "embedding_key": args.embedding_key,
        "top_k": args.top_k,
        "borderline_count": total,
        "misclassified_count": misclassified,
        "misclassified_rate": misclassified / total if total else 0.0,
        "avg_confidence": _row_mean([float(row["confidence"]) for row in sample_rows]),
        "avg_prob_margin": _row_mean([float(row["prob_margin"]) for row in sample_rows]),
        "avg_nn_intact_rate": _row_mean([float(row["nn_intact_rate"]) for row in sample_rows]),
        "avg_nn_confirmed_rate": _row_mean([float(row["nn_confirmed_rate"]) for row in sample_rows]),
        "avg_nn_same_symbol_rate": _row_mean([float(row["nn_same_symbol_rate"]) for row in sample_rows]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote fusion/head diagnosis to {args.output_dir}")


if __name__ == "__main__":
    main()
