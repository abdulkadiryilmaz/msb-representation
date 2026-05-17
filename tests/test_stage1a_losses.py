from __future__ import annotations

import torch

from msb_repr.stage1a.losses import SupConLoss
from msb_repr.stage1a.model import Stage1AModel
from msb_repr.stage1a.trainer import Stage1ATrainer, Stage1ATrainerConfig


def test_supcon_loss_smaller_for_well_grouped_embeddings():
    loss_fn = SupConLoss(temperature=0.1)

    grouped = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    mixed = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    grouped_loss = loss_fn(grouped, labels)
    mixed_loss = loss_fn(mixed, labels)

    assert grouped_loss.item() < mixed_loss.item()


def test_supcon_loss_zero_when_no_positive_pairs():
    loss_fn = SupConLoss(temperature=0.1)
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)

    loss = loss_fn(features, labels)

    assert loss.item() == 0.0


def test_supcon_loss_label_diff_symbol_ignores_same_symbol_pairs():
    loss_fn = SupConLoss(temperature=0.1)
    features = torch.tensor(
        [
            [1.0, 0.0],  # label 0, symbol A
            [0.95, 0.05],  # label 0, symbol A
            [0.9, 0.1],  # label 0, symbol B
            [0.0, 1.0],  # label 1, symbol C
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    symbol_ids = torch.tensor([0, 0, 1, 2], dtype=torch.long)

    label_loss = loss_fn(features, labels, positive_mode="label")
    diff_symbol_loss = loss_fn(features, labels, symbol_ids=symbol_ids, positive_mode="label_diff_symbol")

    assert diff_symbol_loss.item() > 0.0
    assert diff_symbol_loss.item() != label_loss.item()


def test_supcon_loss_neutral_same_symbol_is_lower_than_diff_symbol_mode():
    loss_fn = SupConLoss(temperature=0.1)
    features = torch.tensor(
        [
            [1.0, 0.0],   # label 0, symbol A
            [0.98, 0.02], # label 0, symbol A
            [0.9, 0.1],   # label 0, symbol B
            [0.0, 1.0],   # label 1, symbol C
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    symbol_ids = torch.tensor([0, 0, 1, 2], dtype=torch.long)

    diff_symbol_loss = loss_fn(features, labels, symbol_ids=symbol_ids, positive_mode="label_diff_symbol")
    neutral_loss = loss_fn(
        features,
        labels,
        symbol_ids=symbol_ids,
        positive_mode="label_diff_symbol_neutral_same_symbol",
    )

    assert neutral_loss.item() > 0.0
    assert neutral_loss.item() < diff_symbol_loss.item()


def test_trainer_supcon_frontload_weight_schedule():
    model = Stage1AModel(short_input_channels=2, long_input_channels=2)
    cfg = Stage1ATrainerConfig(
        use_supcon=True,
        supcon_weight=0.05,
        supcon_frontload_weight=0.10,
        supcon_frontload_epochs=3,
    )
    trainer = Stage1ATrainer(model=model, device=torch.device("cpu"), cfg=cfg)

    assert trainer._supcon_weight_for_epoch(1) == 0.10
    assert trainer._supcon_weight_for_epoch(3) == 0.10
    assert trainer._supcon_weight_for_epoch(4) == 0.05


def test_trainer_ce_warmup_weight_schedule():
    model = Stage1AModel(short_input_channels=2, long_input_channels=2)
    cfg = Stage1ATrainerConfig(
        ce_warmup_weight=0.0,
        ce_warmup_epochs=5,
    )
    trainer = Stage1ATrainer(model=model, device=torch.device("cpu"), cfg=cfg)

    assert trainer._ce_weight_for_epoch(1) == 0.0
    assert trainer._ce_weight_for_epoch(5) == 0.0
    assert trainer._ce_weight_for_epoch(6) == 1.0
