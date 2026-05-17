from __future__ import annotations

import numpy as np

from msb_repr.stage1a.dataset import (
    Stage1ADualWindowDataset,
    Stage1ASymbolBalancedBatchSampler,
    align_windows_by_timestamp,
)
from msb_repr.stage1a.model import Stage1AModel


def test_align_windows_by_timestamp_inner_join():
    short = np.arange(5 * 2 * 4, dtype=np.float32).reshape(5, 2, 4)
    long = np.arange(3 * 2 * 8, dtype=np.float32).reshape(3, 2, 8)
    short_ts = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    long_ts = np.array([20, 40, 60], dtype=np.int64)

    short_aligned, long_aligned, ts = align_windows_by_timestamp(short, short_ts, long, long_ts)

    assert short_aligned.shape[0] == 2
    assert long_aligned.shape[0] == 2
    assert ts.tolist() == [20, 40]


def test_stage1a_dataset_len_and_item():
    dataset = Stage1ADualWindowDataset(
        short_windows=np.ones((3, 10, 48), dtype=np.float32),
        long_windows=np.ones((3, 10, 288), dtype=np.float32),
        labels=np.array([0, 1, 2], dtype=np.int64),
        timestamps=np.array([100, 200, 300], dtype=np.int64),
        symbols=["BTC_USDT_15m", "BTC_USDT_15m", "ETH_USDT_15m"],
    )

    short_x, long_x, y, meta = dataset[1]
    assert len(dataset) == 3
    assert tuple(short_x.shape) == (10, 48)
    assert tuple(long_x.shape) == (10, 288)
    assert int(y.item()) == 1
    assert meta.timestamp == 200
    assert meta.symbol == "BTC_USDT_15m"


def test_stage1a_dataset_item_includes_optional_factor_targets():
    dataset = Stage1ADualWindowDataset(
        short_windows=np.ones((2, 10, 48), dtype=np.float32),
        long_windows=np.ones((2, 10, 288), dtype=np.float32),
        labels=np.array([0, 1], dtype=np.int64),
        timestamps=np.array([100, 200], dtype=np.int64),
        symbols=["BTC_USDT_15m", "ETH_USDT_15m"],
        factor_targets=np.array([3, 6], dtype=np.int64),
        pressure_targets=np.array([1, 4], dtype=np.int64),
        maturity_targets=np.array([2, 3], dtype=np.int64),
    )

    _, _, _, meta = dataset[0]

    assert meta.factor_target == 3
    assert meta.pressure_target == 1
    assert meta.maturity_target == 2


def test_stage1a_model_forward_shapes():
    model = Stage1AModel(short_input_channels=10, long_input_channels=10)
    short_x = np.ones((4, 10, 48), dtype=np.float32)
    long_x = np.ones((4, 10, 288), dtype=np.float32)

    outputs = model(
        model.short_encoder.stem[0].weight.new_tensor(short_x),
        model.long_encoder.stem[0].weight.new_tensor(long_x),
    )

    assert tuple(outputs["z_short"].shape) == (4, 64)
    assert tuple(outputs["z_long"].shape) == (4, 32)
    assert tuple(outputs["z_fused"].shape) == (4, 96)
    assert tuple(outputs["z_proj"].shape) == (4, 64)
    assert tuple(outputs["logits"].shape) == (4, 3)


def test_stage1a_model_forward_shapes_with_long_projection():
    model = Stage1AModel(short_input_channels=10, long_input_channels=10, long_projection_dim=16)
    short_x = np.ones((4, 10, 48), dtype=np.float32)
    long_x = np.ones((4, 10, 288), dtype=np.float32)

    outputs = model(
        model.short_encoder.stem[0].weight.new_tensor(short_x),
        model.long_encoder.stem[0].weight.new_tensor(long_x),
    )

    assert tuple(outputs["z_long_proj"].shape) == (4, 16)


def test_stage1a_model_forward_shapes_with_auxiliary_heads():
    model = Stage1AModel(
        short_input_channels=10,
        long_input_channels=10,
        num_pressure_classes=5,
        num_maturity_classes=4,
        num_long_aux_classes=3,
    )
    short_x = np.ones((4, 10, 48), dtype=np.float32)
    long_x = np.ones((4, 10, 288), dtype=np.float32)

    outputs = model(
        model.short_encoder.stem[0].weight.new_tensor(short_x),
        model.long_encoder.stem[0].weight.new_tensor(long_x),
    )

    assert tuple(outputs["pressure_logits"].shape) == (4, 5)
    assert tuple(outputs["maturity_logits"].shape) == (4, 4)
    assert tuple(outputs["long_aux_logits"].shape) == (4, 3)


def test_stage1a_model_pressure_head_can_use_short_latent():
    model = Stage1AModel(
        short_input_channels=10,
        long_input_channels=10,
        num_pressure_classes=5,
        pressure_head_input="z_short",
    )
    short_x = np.ones((4, 10, 48), dtype=np.float32)
    long_x = np.ones((4, 10, 288), dtype=np.float32)

    outputs = model(
        model.short_encoder.stem[0].weight.new_tensor(short_x),
        model.long_encoder.stem[0].weight.new_tensor(long_x),
    )

    assert model.pressure_head_input == "z_short"
    assert tuple(outputs["pressure_logits"].shape) == (4, 5)


def test_symbol_balanced_batch_sampler_mixes_symbols():
    symbols = ["BTC", "BTC", "ETH", "ETH", "SOL", "SOL"]
    sampler = Stage1ASymbolBalancedBatchSampler(symbols=symbols, batch_size=3, shuffle=False)

    batches = list(iter(sampler))

    assert len(batches) == 2
    for batch in batches:
        batch_symbols = {symbols[idx] for idx in batch}
        assert len(batch_symbols) == 3


def test_symbol_balanced_batch_sampler_seed_is_reproducible():
    symbols = ["BTC", "BTC", "ETH", "ETH", "SOL", "SOL"]
    first = Stage1ASymbolBalancedBatchSampler(symbols=symbols, batch_size=3, seed=7)
    second = Stage1ASymbolBalancedBatchSampler(symbols=symbols, batch_size=3, seed=7)

    assert list(iter(first)) == list(iter(second))
