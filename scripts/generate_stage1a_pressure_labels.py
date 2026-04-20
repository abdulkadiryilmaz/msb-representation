"""Generate analysis-only pressure labels from an exported Stage 1A latent file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from msb_repr.stage1a.analysis import build_pressure_label_frame, load_latent_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 1A intact-pressure analysis labels")
    parser.add_argument("--latent-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--version", type=str, default="v1")
    args = parser.parse_args()

    arrays, export_metadata = load_latent_export(args.latent_path)
    label_config = export_metadata.get("label_config", {})
    frame = build_pressure_label_frame(arrays, label_config=label_config)

    output_path = args.output_path or (args.latent_path.with_suffix("") / f"pressure_labels_{args.version}.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)

    intact_frame = frame.filter(frame["is_intact_base_label"])
    summary = {
        "latent_path": str(args.latent_path),
        "output_path": str(output_path),
        "version": args.version,
        "num_rows": int(frame.height),
        "num_intact_rows": int(intact_frame.height),
        "label_config": label_config,
        "pressure_label_counts": frame.group_by("pressure_label").len().sort("pressure_label").to_dicts(),
        "intact_pressure_label_counts": intact_frame.group_by("pressure_label").len().sort("pressure_label").to_dicts(),
    }
    (output_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
    logger.info("Pressure labels written to {} (rows={}, intact_rows={})", output_path, frame.height, intact_frame.height)


if __name__ == "__main__":
    main()
