"""Create manual nearest-neighbor price-window review panels for Stage 1A latents."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from msb_repr.stage1a.analysis import (
    LABEL_NAMES,
    compute_cosine_neighbors,
    compute_hard_case_scores,
    load_latent_export,
)
from msb_repr.stage1a.config import Stage1ADatasetSpec
from msb_repr.stage1a.dataset import Stage1ADualWindowDataset
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


@dataclass(frozen=True)
class SampleView:
    index: int
    role: str
    rank: int | None
    score: float | None
    bucket: str
    bucket_reason: str


def _require_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "msb_repr_matplotlib"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional viz extra
        raise SystemExit(
            "matplotlib is required for PNG panels. Install the viz extra first: pip install -e '.[viz]'"
        ) from exc
    return plt


def _iso_utc(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _feature_indices(feature_columns: list[str]) -> dict[str, int]:
    required = {"open", "high", "low", "close"}
    missing = required.difference(feature_columns)
    if missing:
        raise ValueError(f"Missing OHLC feature columns: {sorted(missing)}")
    return {name: feature_columns.index(name) for name in required}


def _validate_dataset_alignment(
    dataset: Stage1ADualWindowDataset,
    arrays: dict[str, np.ndarray],
) -> None:
    if len(dataset) != len(arrays["labels"]):
        raise ValueError(
            f"Dataset/latent length mismatch: dataset={len(dataset)} latent={len(arrays['labels'])}"
        )
    latent_symbols = arrays["symbols"].astype(str)
    dataset_symbols = np.array(dataset.symbols).astype(str)
    if not np.array_equal(dataset.timestamps, arrays["timestamps"]):
        raise ValueError("Dataset timestamps do not match latent export order")
    if not np.array_equal(dataset.labels, arrays["labels"]):
        raise ValueError("Dataset labels do not match latent export order")
    if not np.array_equal(dataset_symbols, latent_symbols):
        raise ValueError("Dataset symbols do not match latent export order")


def _bucket_for(
    arrays: dict[str, np.ndarray],
    idx: int,
    min_recent_break_bars: int,
    thresholds: dict[str, float],
) -> tuple[str, str]:
    return _assign_domain_bucket(arrays, idx, min_recent_break_bars, thresholds)


def _select_candidates(
    arrays: dict[str, np.ndarray],
    neighbor_indices: np.ndarray,
    hard_scores: np.ndarray,
    buckets: set[str] | None,
    pred_filter: str,
    min_recent_break_bars: int,
    thresholds: dict[str, float],
    limit: int,
) -> list[int]:
    selected: list[int] = []
    order = np.argsort(-hard_scores)
    for idx in order:
        idx_int = int(idx)
        bucket, _ = _bucket_for(arrays, idx_int, min_recent_break_bars, thresholds)
        if buckets is not None and bucket not in buckets:
            continue
        is_error = int(arrays["labels"][idx_int]) != int(arrays["preds"][idx_int])
        if pred_filter == "errors" and not is_error:
            continue
        if pred_filter == "correct" and is_error:
            continue
        # Skip isolated points if nearest-neighbor computation returned no usable neighbors.
        if neighbor_indices.shape[1] == 0:
            continue
        selected.append(idx_int)
        if len(selected) >= limit:
            break
    return selected


def _plot_price_window(
    ax,
    window: np.ndarray,
    feature_idx: dict[str, int],
    title: str,
    meta_text: str,
    structure_bars: int | None,
    bull_level: float,
    bear_level: float,
) -> None:
    x = np.arange(window.shape[1])
    highs = window[feature_idx["high"]]
    lows = window[feature_idx["low"]]
    closes = window[feature_idx["close"]]
    opens = window[feature_idx["open"]]

    ax.fill_between(x, lows, highs, color="#d8dee9", alpha=0.55, linewidth=0)
    ax.plot(x, closes, color="#1f2937", linewidth=1.3, label="close")
    ax.plot(x, opens, color="#6b7280", linewidth=0.8, alpha=0.55, label="open")
    if structure_bars is not None and 0 < structure_bars < window.shape[1]:
        ax.axvline(structure_bars - 0.5, color="#7c3aed", linewidth=0.9, alpha=0.7)
    if np.isfinite(bull_level):
        ax.axhline(float(bull_level), color="#059669", linewidth=0.9, linestyle="--", alpha=0.85)
    if np.isfinite(bear_level):
        ax.axhline(float(bear_level), color="#dc2626", linewidth=0.9, linestyle="--", alpha=0.85)
    ax.set_title(title, fontsize=8, loc="left")
    if meta_text:
        ax.text(
            0.01,
            0.97,
            meta_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.82},
        )
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.35)


def _sample_text(
    arrays: dict[str, np.ndarray],
    view: SampleView,
    min_recent_break_bars: int,
    thresholds: dict[str, float],
) -> tuple[str, str]:
    idx = view.index
    state_fields = _structural_state_fields(arrays, idx, min_recent_break_bars, thresholds)
    prefix = f"{view.role}"
    if view.rank is not None:
        prefix = f"nn{view.rank}"
    title = (
        f"{prefix} idx={idx} {LABEL_NAMES[int(arrays['labels'][idx])]}->{LABEL_NAMES[int(arrays['preds'][idx])]}"
    )
    if view.score is not None:
        title = f"{title} cos={view.score:.4f}"
    meta_lines = [
        str(arrays["symbols"][idx]),
        _iso_utc(int(arrays["timestamps"][idx])),
        f"bucket={view.bucket}",
        (
            f"cs={state_fields['confirmed_state']} ps={state_fields['pressure_state']}\n"
            f"mat={state_fields['break_maturity']} dir={state_fields['structural_direction']} "
            f"hold={state_fields['holding_status']}"
        ),
    ]
    parts = [
        "\n".join(meta_lines),
    ]
    return title, "\n".join(parts)


def _write_panel(
    output_path: Path,
    dataset: Stage1ADualWindowDataset,
    arrays: dict[str, np.ndarray],
    spec: Stage1ADatasetSpec,
    views: list[SampleView],
    min_recent_break_bars: int,
    thresholds: dict[str, float],
) -> None:
    plt = _require_matplotlib()
    short_idx = _feature_indices(spec.short_feature_columns)
    long_idx = _feature_indices(spec.long_feature_columns)

    rows = len(views)
    fig, axes = plt.subplots(rows, 2, figsize=(15, max(2.6 * rows, 4.2)), squeeze=False)
    fig.suptitle(output_path.stem, fontsize=11)

    for row, view in enumerate(views):
        idx = view.index
        title, meta_text = _sample_text(arrays, view, min_recent_break_bars, thresholds)
        bull_level = float(arrays["bull_level"][idx]) if "bull_level" in arrays else np.nan
        bear_level = float(arrays["bear_level"][idx]) if "bear_level" in arrays else np.nan
        _plot_price_window(
            axes[row, 0],
            dataset.short_windows[idx],
            short_idx,
            f"short | {title}",
            meta_text,
            spec.labels.structure_bars,
            bull_level,
            bear_level,
        )
        _plot_price_window(
            axes[row, 1],
            dataset.long_windows[idx],
            long_idx,
            f"long | {title}",
            meta_text,
            None,
            bull_level,
            bear_level,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _candidate_row(
    arrays: dict[str, np.ndarray],
    view: SampleView,
    image_path: Path,
    min_recent_break_bars: int,
    thresholds: dict[str, float],
) -> dict[str, object]:
    idx = view.index
    state_fields = _structural_state_fields(arrays, idx, min_recent_break_bars, thresholds)
    return {
        "query_image": image_path.name,
        "role": view.role,
        "rank": "" if view.rank is None else view.rank,
        "index": idx,
        "score": "" if view.score is None else f"{view.score:.8f}",
        "bucket": view.bucket,
        "bucket_reason": view.bucket_reason,
        **state_fields,
        "symbol": str(arrays["symbols"][idx]),
        "timestamp": int(arrays["timestamps"][idx]),
        "timestamp_utc": _iso_utc(int(arrays["timestamps"][idx])),
        "label": int(arrays["labels"][idx]),
        "label_name": LABEL_NAMES[int(arrays["labels"][idx])],
        "pred": int(arrays["preds"][idx]),
        "pred_name": LABEL_NAMES[int(arrays["preds"][idx])],
        "confidence": float(arrays["probs"][idx].max()),
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
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_review_notes(path: Path, args: argparse.Namespace, candidate_count: int) -> None:
    bucket_text = ", ".join(args.bucket) if args.bucket else "all"
    text = f"""# Stage 1A Manual NN Review

Embedding: `{args.embedding_key}`

Split: `{args.split}`

Checkpoint / latent path:

```text
{args.latent_path}
```

Filters:

- buckets: `{bucket_text}`
- prediction filter: `{args.pred_filter}`
- candidates: `{candidate_count}`
- top-k neighbors per panel: `{args.top_k}`

Review rubric:

- Do query and neighbors share the same directional pressure?
- Compare `confirmed_state`, `pressure_state`, `break_maturity`, `structural_direction`, and `holding_status`.
- Are neighbors structurally similar across symbols, or mostly same-symbol shortcuts?
- Is the query closer to confirmed break examples because the structure is genuinely near-confirmed?
- If neighbors are confirmed breaks, check whether they are `holding` or `reverted`.
- Is similarity driven by wick/close behavior, broader long context, volatility, or coin identity?

Suggested note format per image:

```text
image:
- query factorized state: confirmed_state / pressure_state / break_maturity / structural_direction / holding_status
- neighbor factorized states:
- structural similarity: yes / partial / no
- shortcut suspicion: none / same-symbol / same-period / volatility-only
- borderline interpretation: pressure-continuum / classifier-boundary error / noisy sample
- decision impact:
```
"""
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-path", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-key", default="z_long")
    parser.add_argument("--bucket", action="append", default=None)
    parser.add_argument("--pred-filter", choices=["all", "errors", "correct"], default="all")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args()

    spec = Stage1ADatasetSpec.load(args.dataset_root / "spec.json")
    dataset = Stage1ADualWindowDataset.from_root(args.dataset_root, split=args.split, symbols=args.symbols)
    arrays, export_metadata = load_latent_export(args.latent_path)
    _validate_dataset_alignment(dataset, arrays)
    if args.embedding_key not in arrays:
        raise ValueError(f"Embedding key {args.embedding_key!r} not found in latent export")

    label_config = export_metadata.get("label_config", {})
    min_recent_break_bars = int(label_config.get("min_recent_break_bars", spec.labels.min_recent_break_bars))
    thresholds = _build_bucket_thresholds(arrays)

    neighbor_indices, neighbor_scores = compute_cosine_neighbors(arrays[args.embedding_key], top_k=args.top_k)
    hard_scores = compute_hard_case_scores(
        labels=arrays["labels"],
        preds=arrays["preds"],
        probs=arrays["probs"],
        neighbor_indices=neighbor_indices,
    )
    bucket_filter = set(args.bucket) if args.bucket else None
    candidate_indices = _select_candidates(
        arrays=arrays,
        neighbor_indices=neighbor_indices,
        hard_scores=hard_scores,
        buckets=bucket_filter,
        pred_filter=args.pred_filter,
        min_recent_break_bars=min_recent_break_bars,
        thresholds=thresholds,
        limit=args.limit,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for query_idx in candidate_indices:
        query_bucket, query_reason = _bucket_for(arrays, query_idx, min_recent_break_bars, thresholds)
        views = [
            SampleView(
                index=query_idx,
                role="query",
                rank=None,
                score=None,
                bucket=query_bucket,
                bucket_reason=query_reason,
            )
        ]
        for rank, (neighbor_idx, score) in enumerate(
            zip(neighbor_indices[query_idx], neighbor_scores[query_idx], strict=True),
            start=1,
        ):
            neighbor_bucket, neighbor_reason = _bucket_for(
                arrays, int(neighbor_idx), min_recent_break_bars, thresholds
            )
            views.append(
                SampleView(
                    index=int(neighbor_idx),
                    role="neighbor",
                    rank=rank,
                    score=float(score),
                    bucket=neighbor_bucket,
                    bucket_reason=neighbor_reason,
                )
            )

        image_path = args.output_dir / f"index_{query_idx}_{query_bucket}.png"
        _write_panel(
            image_path,
            dataset,
            arrays,
            spec,
            views,
            min_recent_break_bars,
            thresholds,
        )
        rows.extend(
            _candidate_row(arrays, view, image_path, min_recent_break_bars, thresholds)
            for view in views
        )

    _write_csv(args.output_dir / "candidates.csv", rows)
    _write_review_notes(args.output_dir / "review_notes.md", args, len(candidate_indices))
    print(f"Wrote {len(candidate_indices)} review panels to {args.output_dir}")


if __name__ == "__main__":
    main()
