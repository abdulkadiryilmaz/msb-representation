from __future__ import annotations

import torch

from msb_repr.stage1a.losses import SupConLoss


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
