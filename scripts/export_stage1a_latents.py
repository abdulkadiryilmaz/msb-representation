"""Export Stage 1A latents and logits for a dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from msb_repr.stage1a.analysis import (
    build_domain_summary,
    export_latents,
    load_checkpoint_bundle,
    load_split_dataset,
    save_latent_export,
)
from msb_repr.stage1a.config import Stage1ADatasetSpec


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Stage 1A latents from a saved checkpoint")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args()

    bundle = load_checkpoint_bundle(args.checkpoint_dir, checkpoint_name=args.checkpoint_name)
    spec = Stage1ADatasetSpec.load(args.dataset_root / "spec.json")
    raw_dataset = load_split_dataset(
        args.dataset_root,
        split=args.split,
        bundle=bundle,
        symbols=args.symbols,
        normalize=False,
    )
    domain_summary = build_domain_summary(raw_dataset, spec)
    dataset = load_split_dataset(args.dataset_root, split=args.split, bundle=bundle, symbols=args.symbols)
    exported = export_latents(
        bundle.model,
        dataset,
        device=bundle.device,
        batch_size=args.batch_size,
        extra_arrays=domain_summary,
    )

    output_path = args.output_path or (args.checkpoint_dir / "analysis" / f"{args.split}_latents.npz")
    save_latent_export(
        output_path,
        exported,
        extra_metadata={
            "dataset_root": str(args.dataset_root),
            "checkpoint_dir": str(args.checkpoint_dir),
            "checkpoint_name": args.checkpoint_name,
            "split": args.split,
            "num_samples": int(len(dataset)),
            "symbols": sorted(set(dataset.symbols)),
            "model_metadata": bundle.metadata,
            "label_config": spec.labels.to_dict() if hasattr(spec.labels, "to_dict") else spec.labels.__dict__,
        },
    )
    logger.info("Latent export written to {} (samples={})", output_path, len(dataset))


if __name__ == "__main__":
    main()
