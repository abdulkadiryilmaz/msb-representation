"""Analyze exported Stage 1A latents with PCA/UMAP, nearest neighbors and hard cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from msb_repr.stage1a.analysis import (
    LABEL_NAMES,
    compute_cosine_neighbors,
    compute_hard_case_scores,
    compute_pca_projection,
    compute_umap_projection,
    load_latent_export,
    write_rows_csv,
)


def _nanquantile(values: np.ndarray, q: float, fallback: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return fallback
    return float(np.quantile(finite, q))


def _build_bucket_thresholds(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    directional_excess = np.concatenate(
        [
            np.abs(arrays["bull_final_excess"][np.isfinite(arrays["bull_final_excess"])]),
            np.abs(arrays["bear_final_excess"][np.isfinite(arrays["bear_final_excess"])]),
        ]
    )
    return {
        "high_vol_atr": _nanquantile(arrays["recent_atr_mean"], 0.85),
        "high_vol_hl": _nanquantile(arrays["recent_hl_mean"], 0.85),
        "true_break_excess": _nanquantile(directional_excess, 0.75, fallback=0.0),
        "chop_std": _nanquantile(arrays["recent_close_std"], 0.80),
    }


def _has_domain_summary(arrays: dict[str, np.ndarray]) -> bool:
    required = {
        "recent_atr_mean",
        "recent_hl_mean",
        "recent_close_std",
        "bull_close_count",
        "bear_close_count",
        "bull_wick_count",
        "bear_wick_count",
        "bull_final_excess",
        "bear_final_excess",
    }
    return required.issubset(arrays.keys())


def _assign_domain_bucket(
    arrays: dict[str, np.ndarray],
    idx: int,
    min_recent_break_bars: int,
    thresholds: dict[str, float],
) -> tuple[str, str]:
    if not _has_domain_summary(arrays):
        return "unclassified", "domain summary missing; rerun latent export with the updated exporter"

    label = int(arrays["labels"][idx])
    bull_close_count = int(arrays["bull_close_count"][idx])
    bear_close_count = int(arrays["bear_close_count"][idx])
    bull_wick_count = int(arrays["bull_wick_count"][idx])
    bear_wick_count = int(arrays["bear_wick_count"][idx])
    recent_atr_mean = float(arrays["recent_atr_mean"][idx])
    recent_hl_mean = float(arrays["recent_hl_mean"][idx])
    recent_close_std = float(arrays["recent_close_std"][idx])
    bull_final_excess = float(arrays["bull_final_excess"][idx]) if np.isfinite(arrays["bull_final_excess"][idx]) else np.nan
    bear_final_excess = float(arrays["bear_final_excess"][idx]) if np.isfinite(arrays["bear_final_excess"][idx]) else np.nan

    high_vol = recent_atr_mean >= thresholds["high_vol_atr"] or recent_hl_mean >= thresholds["high_vol_hl"]
    choppy = recent_close_std >= thresholds["chop_std"]

    if label == 0:
        if high_vol and (bull_wick_count > 0 or bear_wick_count > 0):
            return "high_vol_intact_wick_sweep", "intact label with high volatility and failed wick break attempts"
        if high_vol:
            return "high_vol_intact", "intact label under elevated volatility regime"
        if bull_close_count == min_recent_break_bars - 1 or bear_close_count == min_recent_break_bars - 1:
            return "borderline_intact_break", "intact label but close count is one step below break confirmation"
        if bull_wick_count > 0:
            return "wick_sweep_up", "upside wick exceeded structure level without close confirmation"
        if bear_wick_count > 0:
            return "wick_sweep_down", "downside wick exceeded structure level without close confirmation"
        return "clean_intact", "intact label without obvious failed break signature"

    if label == 1:
        if bull_wick_count > 0 and bull_close_count < min_recent_break_bars:
            return "wick_sweep_up", "bullish context but upside breaks are wick-led without enough close confirmation"
        if bull_close_count >= min_recent_break_bars and np.isfinite(bull_final_excess):
            if bull_final_excess >= thresholds["true_break_excess"]:
                return "true_break_up", "bullish label with strong close displacement beyond break level"
            return "close_confirmed_break_up", "bullish label confirmed by close count beyond structure level"
        if high_vol or choppy:
            return "borderline_bullish_break", "bullish label under volatile or choppy confirmation regime"
        return "borderline_bullish_break", "bullish label with weak displacement beyond break level"

    if label == 2:
        if bear_wick_count > 0 and bear_close_count < min_recent_break_bars:
            return "wick_sweep_down", "bearish context but downside breaks are wick-led without enough close confirmation"
        if bear_close_count >= min_recent_break_bars and np.isfinite(bear_final_excess):
            if bear_final_excess >= thresholds["true_break_excess"]:
                return "true_break_down", "bearish label with strong close displacement beyond break level"
            return "close_confirmed_break_down", "bearish label confirmed by close count beyond structure level"
        if high_vol or choppy:
            return "borderline_bearish_break", "bearish label under volatile or choppy confirmation regime"
        return "borderline_bearish_break", "bearish label with weak displacement beyond break level"

    return "unclassified", "bucket rules did not match the sample"


def _projection_rows(arrays: dict[str, np.ndarray], coords: np.ndarray, source: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(len(coords)):
        rows.append(
            {
                "index": idx,
                "source": source,
                "x": float(coords[idx, 0]),
                "y": float(coords[idx, 1]),
                "label": int(arrays["labels"][idx]),
                "label_name": LABEL_NAMES[int(arrays["labels"][idx])],
                "pred": int(arrays["preds"][idx]),
                "pred_name": LABEL_NAMES[int(arrays["preds"][idx])],
                "symbol": str(arrays["symbols"][idx]),
                "timestamp": int(arrays["timestamps"][idx]),
                "confidence": float(arrays["probs"][idx].max()),
            }
        )
    return rows


def _neighbor_rows(
    arrays: dict[str, np.ndarray],
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(len(neighbor_indices)):
        row: dict[str, object] = {
            "index": idx,
            "symbol": str(arrays["symbols"][idx]),
            "timestamp": int(arrays["timestamps"][idx]),
            "label": int(arrays["labels"][idx]),
            "label_name": LABEL_NAMES[int(arrays["labels"][idx])],
            "pred": int(arrays["preds"][idx]),
            "pred_name": LABEL_NAMES[int(arrays["preds"][idx])],
        }
        for rank in range(neighbor_indices.shape[1]):
            neighbor_idx = int(neighbor_indices[idx, rank])
            row[f"nn{rank + 1}_index"] = neighbor_idx
            row[f"nn{rank + 1}_score"] = float(neighbor_scores[idx, rank])
            row[f"nn{rank + 1}_label_name"] = LABEL_NAMES[int(arrays["labels"][neighbor_idx])]
            row[f"nn{rank + 1}_symbol"] = str(arrays["symbols"][neighbor_idx])
            row[f"nn{rank + 1}_timestamp"] = int(arrays["timestamps"][neighbor_idx])
        rows.append(row)
    return rows


def _hard_case_rows(
    arrays: dict[str, np.ndarray],
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    scores: np.ndarray,
    limit: int,
    min_recent_break_bars: int,
    thresholds: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    order = np.argsort(-scores)[:limit]
    for idx in order:
        top_neighbors = neighbor_indices[idx, : min(3, neighbor_indices.shape[1])]
        top_scores = neighbor_scores[idx, : min(3, neighbor_scores.shape[1])]
        bucket, bucket_reason = _assign_domain_bucket(arrays, int(idx), min_recent_break_bars, thresholds)
        rows.append(
            {
                "index": int(idx),
                "hard_score": float(scores[idx]),
                "bucket": bucket,
                "bucket_reason": bucket_reason,
                "symbol": str(arrays["symbols"][idx]),
                "timestamp": int(arrays["timestamps"][idx]),
                "label": int(arrays["labels"][idx]),
                "label_name": LABEL_NAMES[int(arrays["labels"][idx])],
                "pred": int(arrays["preds"][idx]),
                "pred_name": LABEL_NAMES[int(arrays["preds"][idx])],
                "confidence": float(arrays["probs"][idx].max()),
                "nn_label_agreement": float((arrays["labels"][neighbor_indices[idx]] == arrays["labels"][idx]).mean()),
                "recent_atr_mean": float(arrays["recent_atr_mean"][idx]) if "recent_atr_mean" in arrays else "",
                "recent_hl_mean": float(arrays["recent_hl_mean"][idx]) if "recent_hl_mean" in arrays else "",
                "bull_close_count": int(arrays["bull_close_count"][idx]) if "bull_close_count" in arrays else "",
                "bear_close_count": int(arrays["bear_close_count"][idx]) if "bear_close_count" in arrays else "",
                "bull_wick_count": int(arrays["bull_wick_count"][idx]) if "bull_wick_count" in arrays else "",
                "bear_wick_count": int(arrays["bear_wick_count"][idx]) if "bear_wick_count" in arrays else "",
                "bull_final_excess": (
                    float(arrays["bull_final_excess"][idx])
                    if "bull_final_excess" in arrays and np.isfinite(arrays["bull_final_excess"][idx])
                    else ""
                ),
                "bear_final_excess": (
                    float(arrays["bear_final_excess"][idx])
                    if "bear_final_excess" in arrays and np.isfinite(arrays["bear_final_excess"][idx])
                    else ""
                ),
                "top_neighbor_indices": "|".join(str(int(v)) for v in top_neighbors),
                "top_neighbor_scores": "|".join(f"{float(v):.6f}" for v in top_scores),
                "top_neighbor_labels": "|".join(LABEL_NAMES[int(arrays["labels"][v])] for v in top_neighbors),
                "top_neighbor_symbols": "|".join(str(arrays["symbols"][v]) for v in top_neighbors),
            }
        )
    return rows


def _symbol_bucket_summary_rows(hard_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    symbol_totals: dict[str, int] = {}

    for row in hard_rows:
        symbol = str(row["symbol"])
        bucket = str(row["bucket"])
        grouped.setdefault((symbol, bucket), []).append(row)
        symbol_totals[symbol] = symbol_totals.get(symbol, 0) + 1

    rows: list[dict[str, object]] = []
    for (symbol, bucket), samples in sorted(grouped.items()):
        count = len(samples)
        symbol_total = symbol_totals[symbol]
        confidences = np.array([float(sample["confidence"]) for sample in samples], dtype=np.float32)
        hard_scores = np.array([float(sample["hard_score"]) for sample in samples], dtype=np.float32)
        nn_agreement = np.array([float(sample["nn_label_agreement"]) for sample in samples], dtype=np.float32)
        misclassified = sum(1 for sample in samples if sample["label"] != sample["pred"])
        label_names = sorted({str(sample["label_name"]) for sample in samples})
        pred_names = sorted({str(sample["pred_name"]) for sample in samples})

        rows.append(
            {
                "symbol": symbol,
                "bucket": bucket,
                "count": count,
                "share_within_symbol_hard_cases": count / symbol_total,
                "misclassified_count": misclassified,
                "misclassified_rate": misclassified / count,
                "avg_hard_score": float(hard_scores.mean()),
                "avg_confidence": float(confidences.mean()),
                "avg_nn_label_agreement": float(nn_agreement.mean()),
                "label_names": "|".join(label_names),
                "pred_names": "|".join(pred_names),
                "bucket_reason": str(samples[0]["bucket_reason"]),
            }
        )

    rows.sort(key=lambda row: (row["symbol"], -int(row["count"]), row["bucket"]))
    return rows


def _full_distribution_rows(
    arrays: dict[str, np.ndarray],
    min_recent_break_bars: int,
    thresholds: dict[str, float],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[int]] = {}
    symbol_totals: dict[str, int] = {}

    for idx in range(len(arrays["labels"])):
        symbol = str(arrays["symbols"][idx])
        bucket, bucket_reason = _assign_domain_bucket(arrays, idx, min_recent_break_bars, thresholds)
        grouped.setdefault((symbol, bucket, bucket_reason), []).append(idx)
        symbol_totals[symbol] = symbol_totals.get(symbol, 0) + 1

    rows: list[dict[str, object]] = []
    for (symbol, bucket, bucket_reason), indices in sorted(grouped.items()):
        idx_arr = np.array(indices, dtype=np.int64)
        labels = arrays["labels"][idx_arr]
        preds = arrays["preds"][idx_arr]
        probs = arrays["probs"][idx_arr]

        count = len(indices)
        misclassified = int(np.sum(labels != preds))
        rows.append(
            {
                "symbol": symbol,
                "bucket": bucket,
                "count": count,
                "share_within_symbol": count / symbol_totals[symbol],
                "misclassified_count": misclassified,
                "misclassified_rate": misclassified / count,
                "avg_confidence": float(probs.max(axis=1).mean()),
                "label_names": "|".join(sorted({LABEL_NAMES[int(label)] for label in labels})),
                "pred_names": "|".join(sorted({LABEL_NAMES[int(pred)] for pred in preds})),
                "bucket_reason": bucket_reason,
            }
        )

    rows.sort(key=lambda row: (row["symbol"], -int(row["count"]), row["bucket"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exported Stage 1A latent representations")
    parser.add_argument("--latent-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--embedding-key", type=str, default="z_fused")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hard-case-limit", type=int, default=200)
    parser.add_argument("--skip-umap", action="store_true")
    args = parser.parse_args()

    arrays, export_metadata = load_latent_export(args.latent_path)
    embeddings = arrays[args.embedding_key]
    output_dir = args.output_dir or args.latent_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)
    label_config = export_metadata.get("label_config", {})
    min_recent_break_bars = int(label_config.get("min_recent_break_bars", 2))
    has_domain_summary = _has_domain_summary(arrays)
    if has_domain_summary:
        bucket_thresholds = _build_bucket_thresholds(arrays)
    else:
        bucket_thresholds = {}
        logger.warning("Domain summary arrays missing in latent export. Re-run export to enable domain buckets.")

    pca_coords, pca_info = compute_pca_projection(embeddings)
    write_rows_csv(
        output_dir / "pca_projection.csv",
        [
            "index",
            "source",
            "x",
            "y",
            "label",
            "label_name",
            "pred",
            "pred_name",
            "symbol",
            "timestamp",
            "confidence",
        ],
        _projection_rows(arrays, pca_coords, source="pca"),
    )

    umap_status = {"created": False}
    if not args.skip_umap:
        try:
            umap_coords = compute_umap_projection(embeddings)
        except RuntimeError as exc:
            logger.warning(str(exc))
            umap_status = {"created": False, "reason": str(exc)}
        else:
            write_rows_csv(
                output_dir / "umap_projection.csv",
                [
                    "index",
                    "source",
                    "x",
                    "y",
                    "label",
                    "label_name",
                    "pred",
                    "pred_name",
                    "symbol",
                    "timestamp",
                    "confidence",
                ],
                _projection_rows(arrays, umap_coords, source="umap"),
            )
            umap_status = {"created": True}

    neighbor_indices, neighbor_scores = compute_cosine_neighbors(embeddings, top_k=args.top_k)
    neighbor_rows = _neighbor_rows(arrays, neighbor_indices, neighbor_scores)
    neighbor_fields = list(neighbor_rows[0].keys()) if neighbor_rows else []
    if neighbor_fields:
        write_rows_csv(output_dir / "nearest_neighbors.csv", neighbor_fields, neighbor_rows)

    hard_scores = compute_hard_case_scores(
        labels=arrays["labels"],
        preds=arrays["preds"],
        probs=arrays["probs"],
        neighbor_indices=neighbor_indices,
    )
    hard_rows = _hard_case_rows(
        arrays=arrays,
        neighbor_indices=neighbor_indices,
        neighbor_scores=neighbor_scores,
        scores=hard_scores,
        limit=args.hard_case_limit,
        min_recent_break_bars=min_recent_break_bars,
        thresholds=bucket_thresholds,
    )
    hard_fields = list(hard_rows[0].keys()) if hard_rows else []
    if hard_fields:
        write_rows_csv(output_dir / "hard_cases.csv", hard_fields, hard_rows)
        symbol_bucket_rows = _symbol_bucket_summary_rows(hard_rows)
        write_rows_csv(
            output_dir / "symbol_bucket_summary.csv",
            list(symbol_bucket_rows[0].keys()) if symbol_bucket_rows else [],
            symbol_bucket_rows,
        )
    else:
        symbol_bucket_rows = []

    full_distribution_rows = _full_distribution_rows(arrays, min_recent_break_bars, bucket_thresholds)
    write_rows_csv(
        output_dir / "symbol_bucket_full_distribution.csv",
        list(full_distribution_rows[0].keys()) if full_distribution_rows else [],
        full_distribution_rows,
    )

    summary = {
        "latent_path": str(args.latent_path),
        "output_dir": str(output_dir),
        "embedding_key": args.embedding_key,
        "num_samples": int(len(embeddings)),
        "pca": pca_info,
        "umap": umap_status,
        "top_k": args.top_k,
        "hard_case_limit": args.hard_case_limit,
        "domain_summary_present": has_domain_summary,
        "bucket_thresholds": bucket_thresholds,
        "accuracy": float((arrays["labels"] == arrays["preds"]).mean()),
        "mean_confidence": float(arrays["probs"].max(axis=1).mean()),
        "mean_neighbor_label_agreement": float(
            (arrays["labels"][neighbor_indices] == arrays["labels"][:, None]).mean()
        ),
        "hard_case_bucket_counts": {
            row["bucket"]: sum(1 for sample in hard_rows if sample["bucket"] == row["bucket"])
            for row in hard_rows
        },
        "symbol_bucket_counts": {
            row["symbol"]: sum(1 for sample in symbol_bucket_rows if sample["symbol"] == row["symbol"])
            for row in symbol_bucket_rows
        },
        "full_distribution_bucket_counts": {
            row["symbol"]: sum(1 for sample in full_distribution_rows if sample["symbol"] == row["symbol"])
            for row in full_distribution_rows
        },
        "symbols": sorted({str(symbol) for symbol in arrays["symbols"]}),
        "export_metadata": export_metadata,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Latent analysis written to {}", output_dir)


if __name__ == "__main__":
    main()
