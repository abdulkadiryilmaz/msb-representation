"""Compare Stage 1A embedding views by bucket."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from msb_repr.stage1a.analysis import LABEL_NAMES, compute_cosine_neighbors, load_latent_export, write_rows_csv
try:
    from analyze_stage1a_latents import (
        _assign_domain_bucket,
        _build_bucket_thresholds,
        _has_domain_summary,
    )
except ModuleNotFoundError:
    from scripts.analyze_stage1a_latents import (
        _assign_domain_bucket,
        _build_bucket_thresholds,
        _has_domain_summary,
    )


def _default_embedding_keys(arrays: dict[str, np.ndarray]) -> list[str]:
    keys = ["z_short", "z_long", "z_fused"]
    if "z_long_proj" in arrays:
        keys.append("z_long_proj")
    return keys


def _bucket_assignments(
    arrays: dict[str, np.ndarray],
    min_recent_break_bars: int,
    thresholds: dict[str, float],
) -> tuple[list[str], list[str]]:
    buckets: list[str] = []
    reasons: list[str] = []
    for idx in range(len(arrays["labels"])):
        bucket, reason = _assign_domain_bucket(arrays, idx, min_recent_break_bars, thresholds)
        buckets.append(bucket)
        reasons.append(reason)
    return buckets, reasons


def _embedding_bucket_rows(
    arrays: dict[str, np.ndarray],
    embedding_key: str,
    buckets: list[str],
    bucket_reasons: list[str],
    top_k: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    embeddings = arrays[embedding_key]
    neighbor_indices, _ = compute_cosine_neighbors(embeddings, top_k=top_k)

    grouped: dict[tuple[str, str, str], list[int]] = {}
    symbol_totals: dict[str, int] = {}
    for idx, bucket in enumerate(buckets):
        symbol = str(arrays["symbols"][idx])
        grouped.setdefault((symbol, bucket, bucket_reasons[idx]), []).append(idx)
        symbol_totals[symbol] = symbol_totals.get(symbol, 0) + 1

    rows: list[dict[str, object]] = []
    top1 = neighbor_indices[:, 0]
    same_label_neighbors = (arrays["labels"][neighbor_indices] == arrays["labels"][:, None]).mean(axis=1)
    same_symbol_neighbors = (arrays["symbols"][neighbor_indices] == arrays["symbols"][:, None]).mean(axis=1)
    top1_label_match = (arrays["labels"][top1] == arrays["labels"]).astype(np.float32)
    top1_symbol_match = (arrays["symbols"][top1] == arrays["symbols"]).astype(np.float32)

    for (symbol, bucket, reason), indices in sorted(grouped.items()):
        idx_arr = np.array(indices, dtype=np.int64)
        labels = arrays["labels"][idx_arr]
        preds = arrays["preds"][idx_arr]
        probs = arrays["probs"][idx_arr]
        count = len(indices)
        misclassified = int(np.sum(labels != preds))

        rows.append(
            {
                "embedding_key": embedding_key,
                "symbol": symbol,
                "bucket": bucket,
                "count": count,
                "share_within_symbol": count / symbol_totals[symbol],
                "misclassified_count": misclassified,
                "misclassified_rate": misclassified / count,
                "avg_confidence": float(probs.max(axis=1).mean()),
                "avg_nn_label_agreement": float(same_label_neighbors[idx_arr].mean()),
                "avg_nn_symbol_agreement": float(same_symbol_neighbors[idx_arr].mean()),
                "top1_label_match_rate": float(top1_label_match[idx_arr].mean()),
                "top1_symbol_match_rate": float(top1_symbol_match[idx_arr].mean()),
                "label_names": "|".join(sorted({LABEL_NAMES[int(label)] for label in labels})),
                "pred_names": "|".join(sorted({LABEL_NAMES[int(pred)] for pred in preds})),
                "bucket_reason": reason,
            }
        )

    rows.sort(key=lambda row: (row["embedding_key"], row["symbol"], -int(row["count"]), row["bucket"]))
    summary = {
        "embedding_key": embedding_key,
        "mean_nn_label_agreement": float(same_label_neighbors.mean()),
        "mean_nn_symbol_agreement": float(same_symbol_neighbors.mean()),
        "top1_label_match_rate": float(top1_label_match.mean()),
        "top1_symbol_match_rate": float(top1_symbol_match.mean()),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Stage 1A embedding views by domain bucket")
    parser.add_argument("--latent-path", type=Path, required=True)
    parser.add_argument(
        "--embedding-keys",
        nargs="+",
        default=None,
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    arrays, export_metadata = load_latent_export(args.latent_path)
    output_dir = args.output_dir or (args.latent_path.with_suffix("") / "embedding_compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    label_config = export_metadata.get("label_config", {})
    min_recent_break_bars = int(label_config.get("min_recent_break_bars", 2))
    if _has_domain_summary(arrays):
        thresholds = _build_bucket_thresholds(arrays)
    else:
        thresholds = {}
        logger.warning("Domain summary arrays missing in latent export. Buckets will be unclassified.")

    buckets, bucket_reasons = _bucket_assignments(arrays, min_recent_break_bars, thresholds)

    embedding_keys = args.embedding_keys or _default_embedding_keys(arrays)
    comparison_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []
    for embedding_key in embedding_keys:
        if embedding_key not in arrays:
            raise ValueError(f"Embedding key not found in latent export: {embedding_key}")
        rows, summary = _embedding_bucket_rows(
            arrays=arrays,
            embedding_key=embedding_key,
            buckets=buckets,
            bucket_reasons=bucket_reasons,
            top_k=args.top_k,
        )
        comparison_rows.extend(rows)
        overall_rows.append(summary)

    write_rows_csv(
        output_dir / "embedding_bucket_comparison.csv",
        list(comparison_rows[0].keys()) if comparison_rows else [],
        comparison_rows,
    )
    write_rows_csv(
        output_dir / "embedding_overall_summary.csv",
        list(overall_rows[0].keys()) if overall_rows else [],
        overall_rows,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "latent_path": str(args.latent_path),
                "embedding_keys": embedding_keys,
                "top_k": args.top_k,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    logger.info("Embedding comparison written to {}", output_dir)


if __name__ == "__main__":
    main()
