"""Create stratified manual price-window review panels for pressure/maturity labels."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from msb_repr.stage1a.analysis import LABEL_NAMES, load_latent_export
from msb_repr.stage1a.config import Stage1ADatasetSpec
from msb_repr.stage1a.dataset import Stage1ADualWindowDataset

try:
    from scripts.analyze_stage1a_latents import _assign_domain_bucket, _build_bucket_thresholds, _structural_state_fields
    from scripts.inspect_stage1a_nn_windows import (
        SampleView,
        _candidate_row,
        _validate_dataset_alignment,
        _write_panel,
    )
except ModuleNotFoundError:  # Direct execution via `python scripts/...`.
    from analyze_stage1a_latents import _assign_domain_bucket, _build_bucket_thresholds, _structural_state_fields
    from inspect_stage1a_nn_windows import SampleView, _candidate_row, _validate_dataset_alignment, _write_panel


@dataclass(frozen=True)
class ReviewStratum:
    name: str
    description: str


STRATA = [
    ReviewStratum("clean_neutral", "intact clean neutral"),
    ReviewStratum("wick_sweep_up", "intact upside wick sweep"),
    ReviewStratum("wick_sweep_down", "intact downside wick sweep"),
    ReviewStratum("borderline_up_holding", "intact borderline up pressure still holding"),
    ReviewStratum("borderline_up_reverted", "intact borderline up pressure reverted by anchor"),
    ReviewStratum("borderline_down_holding", "intact borderline down pressure still holding"),
    ReviewStratum("borderline_down_reverted", "intact borderline down pressure reverted by anchor"),
    ReviewStratum("mixed_borderline", "intact mixed borderline pressure"),
]


def _stratum_for(fields: dict[str, str]) -> str | None:
    if fields["confirmed_state"] != "intact":
        return None
    if (
        fields["pressure_state"] == "neutral"
        and fields["break_maturity"] == "clean"
        and fields["structural_direction"] == "none"
    ):
        return "clean_neutral"
    if fields["break_maturity"] == "wick_sweep" and fields["structural_direction"] == "up":
        return "wick_sweep_up"
    if fields["break_maturity"] == "wick_sweep" and fields["structural_direction"] == "down":
        return "wick_sweep_down"
    if fields["break_maturity"] == "borderline":
        if fields["structural_direction"] == "mixed":
            return "mixed_borderline"
        if fields["structural_direction"] == "up" and fields["holding_status"] == "holding":
            return "borderline_up_holding"
        if fields["structural_direction"] == "up" and fields["holding_status"] == "reverted":
            return "borderline_up_reverted"
        if fields["structural_direction"] == "down" and fields["holding_status"] == "holding":
            return "borderline_down_holding"
        if fields["structural_direction"] == "down" and fields["holding_status"] == "reverted":
            return "borderline_down_reverted"
    return None


def _select_stratified_indices(
    arrays: dict[str, np.ndarray],
    min_recent_break_bars: int,
    thresholds: dict[str, float],
    per_stratum: int,
    seed: int,
) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx in range(len(arrays["labels"])):
        fields = _structural_state_fields(arrays, idx, min_recent_break_bars, thresholds)
        stratum = _stratum_for(fields)
        if stratum is not None:
            grouped[stratum].append(idx)

    selected: dict[str, list[int]] = {}
    for stratum in [item.name for item in STRATA]:
        indices = np.array(grouped.get(stratum, []), dtype=np.int64)
        if len(indices) == 0:
            selected[stratum] = []
            continue
        rng.shuffle(indices)
        selected[stratum] = [int(idx) for idx in indices[:per_stratum]]
    return selected


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_review_notes(
    path: Path,
    args: argparse.Namespace,
    selected: dict[str, list[int]],
    available_counts: dict[str, int],
) -> None:
    rows = "\n".join(
        f"| `{item.name}` | {available_counts.get(item.name, 0)} | {len(selected.get(item.name, []))} | {item.description} |"
        for item in STRATA
    )
    text = f"""# Stage 1A Pressure/Maturity Manual Review

Latent path:

```text
{args.latent_path}
```

Split: `{args.split}`

Selection:

- per stratum: `{args.per_stratum}`
- random seed: `{args.seed}`

| Stratum | Available | Selected | Description |
|---|---:|---:|---|
{rows}

Review question:

```text
Does the visible short/long price window support the assigned pressure and maturity label?
```

Use this decision vocabulary:

- `accept`: label is visually and structurally reasonable
- `questionable`: rule is technically explainable, but the semantic read is weak or ambiguous
- `reject`: label is misleading or contradicts the visible window

Recommended columns to fill in `candidates.csv`:

- `manual_decision`
- `manual_reason`

Group-specific cues:

- `clean_neutral`: no obvious close/wick pressure near structural levels
- `wick_sweep_up/down`: wick exceeds the relevant level, but close confirmation is absent
- `borderline_*_holding`: close pressure is one step below confirmation and final excess is still holding
- `borderline_*_reverted`: close pressure occurred, but final excess reverted by anchor
- `mixed_borderline`: both directions are active enough that a single directional pressure label would be misleading
"""
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-path", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args()

    spec = Stage1ADatasetSpec.load(args.dataset_root / "spec.json")
    dataset = Stage1ADualWindowDataset.from_root(args.dataset_root, split=args.split, symbols=args.symbols)
    arrays, export_metadata = load_latent_export(args.latent_path)
    _validate_dataset_alignment(dataset, arrays)

    label_config = export_metadata.get("label_config", {})
    min_recent_break_bars = int(label_config.get("min_recent_break_bars", spec.labels.min_recent_break_bars))
    thresholds = _build_bucket_thresholds(arrays)

    selected = _select_stratified_indices(
        arrays=arrays,
        min_recent_break_bars=min_recent_break_bars,
        thresholds=thresholds,
        per_stratum=args.per_stratum,
        seed=args.seed,
    )
    available = {
        stratum.name: len(
            [
                idx
                for idx in range(len(arrays["labels"]))
                if _stratum_for(_structural_state_fields(arrays, idx, min_recent_break_bars, thresholds))
                == stratum.name
            ]
        )
        for stratum in STRATA
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for stratum in [item.name for item in STRATA]:
        for idx in selected[stratum]:
            bucket, reason = _assign_domain_bucket(arrays, idx, min_recent_break_bars, thresholds)
            view = SampleView(index=idx, role="query", rank=None, score=None, bucket=bucket, bucket_reason=reason)
            image_path = args.output_dir / f"{stratum}_index_{idx}_{bucket}.png"
            _write_panel(
                image_path,
                dataset,
                arrays,
                spec,
                [view],
                min_recent_break_bars,
                thresholds,
            )
            row = _candidate_row(arrays, view, image_path, min_recent_break_bars, thresholds)
            row = {
                "stratum": stratum,
                "manual_decision": "",
                "manual_reason": "",
                **row,
            }
            rows.append(row)

    _write_csv(args.output_dir / "candidates.csv", rows)
    _write_review_notes(args.output_dir / "review_notes.md", args, selected, available)
    print(f"Wrote {len(rows)} stratified review panels to {args.output_dir}")


if __name__ == "__main__":
    main()
