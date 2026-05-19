from __future__ import annotations

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from msb_repr.stage1b.dataset import Stage1BForwardDataset, collate_fn_stage1b
from msb_repr.stage1b.model import Stage1BForwardPredictor
from msb_repr.stage1b.trainer import Stage1BTrainer, Stage1BTrainerConfig


def test_stage1b_forward_dataset_maps_targets(tmp_path):
    latent_path = tmp_path / "latents.npz"
    label_path = tmp_path / "labels.parquet"
    np.savez(
        latent_path,
        z_fused=np.arange(20, dtype=np.float32).reshape(5, 4),
        logits=np.ones((5, 3), dtype=np.float32),
    )
    pl.DataFrame(
        {
            "index": [0, 1, 2, 3, 4],
            "symbol": ["BTC"] * 5,
            "timestamp": [1, 2, 3, 4, 5],
            "h16_future_break_direction": [
                "none",
                "bullish",
                "bearish",
                "ambiguous",
                "insufficient_future",
            ],
        }
    ).write_parquet(label_path)

    dataset = Stage1BForwardDataset.from_paths(
        latent_path=latent_path,
        label_path=label_path,
        feature_keys=["z_fused", "logits"],
        target_column="h16_future_break_direction",
    )

    assert len(dataset) == 3
    assert dataset.input_dim == 7
    assert dataset.break_targets.tolist() == [[0], [1], [1]]
    assert dataset.direction_targets.tolist() == [[-1], [0], [1]]
    assert dataset.recon_targets.tolist() == [[0], [1], [2]]


def test_stage1b_forward_predictor_shapes():
    model = Stage1BForwardPredictor(input_dim=6, hidden_dim=8)
    outputs = model(torch.randn(4, 6))
    assert outputs["break_logits"].shape == (4, 1, 2)
    assert outputs["direction_logits"].shape == (4, 1, 2)


def test_stage1b_trainer_runs_one_epoch(tmp_path):
    latent_path = tmp_path / "latents.npz"
    label_path = tmp_path / "labels.parquet"
    features = np.random.default_rng(42).normal(size=(12, 4)).astype(np.float32)
    labels = ["none", "bullish", "bearish"] * 4
    np.savez(latent_path, z_fused=features)
    pl.DataFrame(
        {
            "index": list(range(12)),
            "symbol": ["BTC"] * 12,
            "timestamp": list(range(12)),
            "h16_future_break_direction": labels,
        }
    ).write_parquet(label_path)
    dataset = Stage1BForwardDataset.from_paths(
        latent_path=latent_path,
        label_path=label_path,
        feature_keys=["z_fused"],
        target_column="h16_future_break_direction",
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_stage1b)
    model = Stage1BForwardPredictor(input_dim=4, hidden_dim=8)
    trainer = Stage1BTrainer(
        model=model,
        device=torch.device("cpu"),
        cfg=Stage1BTrainerConfig(epochs=1, patience=2, checkpoint_dir=tmp_path / "ckpt"),
    )

    history = trainer.train(loader, loader)
    metrics = trainer.evaluate(loader)

    assert len(history["val_recon_macro_f1"]) == 1
    assert "recon_confusion_matrix" in metrics
    assert (tmp_path / "ckpt" / "best.pt").exists()
