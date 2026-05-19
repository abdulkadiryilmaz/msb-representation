"""Audit Stage 1B forward label distributions and horizon consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger


LABEL_ORDER = ["none", "bullish", "bearish", "ambiguous", "insufficient_future"]


def _counts(df: pl.DataFrame, column: str) -> dict[str, int]:
    rows = df.group_by(column).len().to_dicts()
    counts = {str(row[column]): int(row["len"]) for row in rows}
    return {label: counts.get(label, 0) for label in LABEL_ORDER if counts.get(label, 0) > 0}


def _rates(counts: dict[str, int]) -> dict[str, float]:
    total = max(sum(counts.values()), 1)
    return {label: count / total for label, count in counts.items()}


def _crosstab(df: pl.DataFrame, row_col: str, col_col: str) -> dict[str, dict[str, int]]:
    rows = (
        df.group_by(row_col, col_col)
        .len()
        .sort(row_col, col_col)
        .to_dicts()
    )
    table = {label: {inner: 0 for inner in LABEL_ORDER} for label in LABEL_ORDER}
    for row in rows:
        table[str(row[row_col])][str(row[col_col])] = int(row["len"])
    return {
        label: {inner: count for inner, count in inner_counts.items() if count > 0}
        for label, inner_counts in table.items()
        if any(count > 0 for count in inner_counts.values())
    }


def _time_to_break_summary(df: pl.DataFrame, horizon: int) -> dict[str, Any]:
    prefix = f"h{horizon}"
    direction_col = f"{prefix}_future_break_direction"
    time_col = f"{prefix}_time_to_break"
    break_df = df.filter(pl.col(direction_col).is_in(["bullish", "bearish", "ambiguous"]))
    if break_df.is_empty():
        return {"support": 0}
    quantiles = break_df.select(
        pl.col(time_col).min().alias("min"),
        pl.col(time_col).quantile(0.25).alias("q25"),
        pl.col(time_col).median().alias("median"),
        pl.col(time_col).quantile(0.75).alias("q75"),
        pl.col(time_col).max().alias("max"),
    ).to_dicts()[0]
    return {
        "support": int(break_df.height),
        **{key: float(value) for key, value in quantiles.items()},
    }


def _delayed_segment_summary(df: pl.DataFrame, short_h: int, long_h: int) -> dict[str, Any]:
    short_col = f"h{short_h}_future_break_direction"
    long_col = f"h{long_h}_future_break_direction"
    segment = df.filter(
        (pl.col(short_col) == "none")
        & pl.col(long_col).is_in(["bullish", "bearish"])
    )
    return {
        "support": int(segment.height),
        "rate": float(segment.height / max(df.height, 1)),
        "direction_counts": _counts(segment, long_col) if not segment.is_empty() else {},
        "by_symbol": (
            segment.group_by("symbol").len().sort("symbol").to_dicts()
            if not segment.is_empty()
            else []
        ),
    }


def _audit_file(path: Path, horizons: list[int]) -> dict[str, Any]:
    df = pl.read_parquet(path)
    summary: dict[str, Any] = {
        "path": str(path),
        "rows": int(df.height),
        "horizons": {},
        "cross_horizon": {},
        "delayed_segments": {},
        "suspicious": {},
    }
    for horizon in horizons:
        direction_col = f"h{horizon}_future_break_direction"
        counts = _counts(df, direction_col)
        summary["horizons"][f"h{horizon}"] = {
            "direction_counts": counts,
            "direction_rates": _rates(counts),
            "time_to_break": _time_to_break_summary(df, horizon),
        }

    for short_h, long_h in zip(horizons, horizons[1:], strict=False):
        short_col = f"h{short_h}_future_break_direction"
        long_col = f"h{long_h}_future_break_direction"
        key = f"h{short_h}_to_h{long_h}"
        summary["cross_horizon"][key] = _crosstab(df, short_col, long_col)
        summary["delayed_segments"][key] = _delayed_segment_summary(df, short_h, long_h)
        impossible = df.filter(
            pl.col(short_col).is_in(["bullish", "bearish", "ambiguous"])
            & (pl.col(long_col) == "none")
        )
        direction_flip = df.filter(
            ((pl.col(short_col) == "bullish") & (pl.col(long_col) == "bearish"))
            | ((pl.col(short_col) == "bearish") & (pl.col(long_col) == "bullish"))
        )
        summary["suspicious"][key] = {
            "short_break_long_none": int(impossible.height),
            "short_break_long_none_rate": float(impossible.height / max(df.height, 1)),
            "direction_flip": int(direction_flip.height),
            "direction_flip_rate": float(direction_flip.height / max(df.height, 1)),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--split-names", nargs="*", default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=[16, 32, 48])
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    split_names = args.split_names or [path.stem for path in args.label_paths]
    if len(split_names) != len(args.label_paths):
        raise ValueError("--split-names must match --label-paths length")

    summary = {
        "horizons": args.horizons,
        "splits": {
            split: _audit_file(path, args.horizons)
            for split, path in zip(split_names, args.label_paths, strict=True)
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, indent=2))
    logger.info("Stage 1B label audit written to {}", args.output_path)


if __name__ == "__main__":
    main()
