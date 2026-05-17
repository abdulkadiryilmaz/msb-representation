from __future__ import annotations

import json

import numpy as np
import torch

from msb_repr.data.normalizer import Normalizer
from msb_repr.stage1a.analysis import (
    build_domain_summary,
    build_pressure_label_frame,
    build_structural_training_targets,
    compute_intact_pressure_labels,
    compute_cosine_neighbors,
    compute_hard_case_scores,
    export_latents,
    load_checkpoint_bundle,
)
from msb_repr.stage1a.config import Stage1ADatasetSpec
from msb_repr.stage1a.dataset import Stage1ADualWindowDataset
from msb_repr.stage1a.model import Stage1AModel
from scripts.analyze_stage1a_latents import (
    _assign_domain_bucket,
    _build_bucket_thresholds,
    _full_distribution_rows,
    _structural_state_fields,
)
from scripts.compare_stage1a_embedding_views import _embedding_bucket_rows
from scripts.diagnose_stage1a_branch_usage import branch_usage_summary
from scripts.diagnose_stage1a_bucket_geometry import (
    bucket_centroids,
    bucket_distance_rows,
    structural_bucket_labels,
)


def test_load_checkpoint_bundle_and_export_latents(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    metadata = {
        "short_input_channels": 2,
        "long_input_channels": 3,
        "z_short": 4,
        "z_long": 5,
        "projection_dim": 6,
        "long_projection_dim": 3,
        "pressure_head_input": "z_fused",
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata))

    short_norm = Normalizer()
    short_norm.mean_ = np.array([0.0, 0.0], dtype=np.float32)
    short_norm.std_ = np.array([1.0, 1.0], dtype=np.float32)
    short_norm.save(checkpoint_dir / "short_norm_params.json")

    long_norm = Normalizer()
    long_norm.mean_ = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    long_norm.std_ = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    long_norm.save(checkpoint_dir / "long_norm_params.json")

    model = Stage1AModel(
        short_input_channels=2,
        long_input_channels=3,
        z_short=4,
        z_long=5,
        projection_dim=6,
        long_projection_dim=3,
    )
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_dir / "best.pt")

    bundle = load_checkpoint_bundle(checkpoint_dir, device=torch.device("cpu"))
    dataset = Stage1ADualWindowDataset(
        short_windows=np.ones((3, 2, 48), dtype=np.float32),
        long_windows=np.ones((3, 3, 288), dtype=np.float32),
        labels=np.array([0, 1, 2], dtype=np.int64),
        timestamps=np.array([10, 20, 30], dtype=np.int64),
        symbols=["BTC_USDT_15m", "ETH_USDT_15m", "SOL_USDT_15m"],
    )

    exported = export_latents(bundle.model, dataset, device=torch.device("cpu"), batch_size=2)

    assert tuple(exported["z_short"].shape) == (3, 4)
    assert tuple(exported["z_long"].shape) == (3, 5)
    assert tuple(exported["z_fused"].shape) == (3, 9)
    assert tuple(exported["z_proj"].shape) == (3, 6)
    assert tuple(exported["z_long_proj"].shape) == (3, 3)
    assert tuple(exported["logits"].shape) == (3, 3)
    assert exported["timestamps"].tolist() == [10, 20, 30]
    assert exported["symbols"].tolist() == ["BTC_USDT_15m", "ETH_USDT_15m", "SOL_USDT_15m"]


def test_compute_cosine_neighbors_returns_expected_order():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    indices, scores = compute_cosine_neighbors(embeddings, top_k=2, chunk_size=2)

    assert indices.shape == (3, 2)
    assert scores.shape == (3, 2)
    assert indices[0, 0] == 1
    assert indices[2, 0] == 1
    assert scores[0, 0] > scores[0, 1]


def test_branch_usage_summary_reports_effective_long_short_ratio():
    model = Stage1AModel(short_input_channels=2, long_input_channels=2, z_short=2, z_long=2)
    with torch.no_grad():
        first_linear = model.classifier[0]
        first_linear.weight[:, :2] = 1.0
        first_linear.weight[:, 2:] = 2.0

    arrays = {
        "z_short": np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32),
        "z_long": np.array([[0.0, 10.0], [6.0, 8.0]], dtype=np.float32),
        "z_long_proj": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "z_fused": np.array([[3.0, 4.0, 0.0, 10.0], [0.0, 5.0, 6.0, 8.0]], dtype=np.float32),
        "labels": np.array([0, 1], dtype=np.int64),
        "symbols": np.array(["BTC", "ETH"]),
    }

    summary = branch_usage_summary(arrays, model, history={"train_supcon_loss": [4.0, 3.0]})

    assert summary["z_long_short_norm_ratio"] == 2.0
    assert summary["classifier_long_short_weight_ratio"] == 2.0
    assert summary["effective_long_short_ratio"] == 4.0
    assert summary["train_supcon_first"] == 4.0
    assert summary["train_supcon_last"] == 3.0


def test_bucket_geometry_diagnostics_build_focus_distances():
    arrays = {
        "z_long": np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        ),
        "labels": np.array([0, 0, 1, 1], dtype=np.int64),
        "bull_close_count": np.array([0, 1, 2, 2], dtype=np.int64),
        "bear_close_count": np.array([0, 0, 0, 0], dtype=np.int64),
        "bull_wick_count": np.array([0, 0, 0, 0], dtype=np.int64),
        "bear_wick_count": np.array([0, 0, 0, 0], dtype=np.int64),
        "recent_atr_mean": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "recent_hl_mean": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "recent_close_std": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "bull_final_excess": np.array([np.nan, -0.01, 0.02, 0.03], dtype=np.float32),
        "bear_final_excess": np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32),
    }

    labels = structural_bucket_labels(arrays, min_recent_break_bars=2)
    centroids = bucket_centroids(arrays["z_long"], labels, min_count=1)
    rows = bucket_distance_rows(
        arrays,
        labels,
        embedding_keys=["z_long"],
        pair_groups=[
            {"group": "clean_vs_borderline", "left": "clean_intact", "right": "borderline_up"},
            {"group": "borderline_vs_confirmed", "left": "borderline_up", "right": "bullish_confirmed"},
        ],
        min_count=1,
    )

    assert set(labels.tolist()) == {"clean_intact", "borderline_up", "bullish_confirmed"}
    assert {"clean_intact", "borderline_up", "bullish_confirmed"}.issubset(centroids.keys())
    assert len(rows) == 2
    assert rows[0]["cosine_distance"] >= 0.0


def test_compute_hard_case_scores_prioritizes_errors():
    labels = np.array([0, 1, 1], dtype=np.int64)
    preds = np.array([0, 0, 1], dtype=np.int64)
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.45, 0.4, 0.15],
            [0.34, 0.33, 0.33],
        ],
        dtype=np.float32,
    )
    neighbor_indices = np.array(
        [
            [1, 2],
            [0, 2],
            [1, 0],
        ],
        dtype=np.int64,
    )

    scores = compute_hard_case_scores(labels, preds, probs, neighbor_indices)

    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_build_domain_summary_extracts_break_signals():
    spec = Stage1ADatasetSpec(exchange="binance", timeframe="15m", symbols=["BTC_USDT_15m"])
    window = np.zeros((10, 48), dtype=np.float32)
    open_idx, high_idx, low_idx, close_idx = 0, 1, 2, 3
    hl_idx, atr_idx = 6, 7

    window[open_idx, :] = 100.0
    window[high_idx, :36] = np.array(
        [100.0] * 16 + [101.0, 102.0, 103.0, 104.0, 106.0, 104.0, 103.0, 102.0, 101.0] + [100.0] * 11,
        dtype=np.float32,
    )
    window[low_idx, :36] = np.array([99.0] * 36, dtype=np.float32)
    window[close_idx, :36] = np.array([100.0] * 36, dtype=np.float32)
    window[high_idx, 36:] = np.array([108.2] * 12, dtype=np.float32)
    window[low_idx, 36:] = np.array([106.9] * 12, dtype=np.float32)
    window[close_idx, 36:] = np.array([107.8] * 12, dtype=np.float32)
    window[open_idx, 36:] = np.array([107.0] * 12, dtype=np.float32)
    window[hl_idx, :] = 0.02
    window[atr_idx, :] = 0.03

    dataset = Stage1ADualWindowDataset(
        short_windows=np.expand_dims(window, axis=0),
        long_windows=np.ones((1, 10, 288), dtype=np.float32),
        labels=np.array([1], dtype=np.int64),
        timestamps=np.array([1], dtype=np.int64),
        symbols=["BTC_USDT_15m"],
    )

    summary = build_domain_summary(dataset, spec)

    assert int(summary["bull_close_count"][0]) >= 2
    assert float(summary["bull_final_excess"][0]) > 0.0
    assert int(summary["bull_wick_count"][0]) == 0


def test_domain_bucket_assignment_uses_bucket_names():
    arrays = {
        "labels": np.array([0, 1, 2], dtype=np.int64),
        "bull_close_count": np.array([1, 3, 0], dtype=np.int64),
        "bear_close_count": np.array([0, 0, 3], dtype=np.int64),
        "bull_wick_count": np.array([1, 0, 0], dtype=np.int64),
        "bear_wick_count": np.array([0, 0, 0], dtype=np.int64),
        "recent_atr_mean": np.array([0.06, 0.02, 0.02], dtype=np.float32),
        "recent_hl_mean": np.array([0.07, 0.02, 0.02], dtype=np.float32),
        "recent_close_std": np.array([0.04, 0.01, 0.01], dtype=np.float32),
        "bull_final_excess": np.array([np.nan, 0.05, np.nan], dtype=np.float32),
        "bear_final_excess": np.array([np.nan, np.nan, 0.05], dtype=np.float32),
    }
    thresholds = _build_bucket_thresholds(arrays)

    bucket0, _ = _assign_domain_bucket(arrays, 0, min_recent_break_bars=2, thresholds=thresholds)
    bucket1, _ = _assign_domain_bucket(arrays, 1, min_recent_break_bars=2, thresholds=thresholds)
    bucket2, _ = _assign_domain_bucket(arrays, 2, min_recent_break_bars=2, thresholds=thresholds)

    assert bucket0 == "high_vol_intact_wick_sweep"
    assert bucket1 in {"true_break_up", "close_confirmed_break_up"}
    assert bucket2 in {"true_break_down", "close_confirmed_break_down"}


def test_domain_bucket_assignment_splits_borderline_intact_direction():
    arrays = {
        "labels": np.array([0, 0, 0], dtype=np.int64),
        "bull_close_count": np.array([1, 0, 1], dtype=np.int64),
        "bear_close_count": np.array([0, 1, 1], dtype=np.int64),
        "bull_wick_count": np.array([0, 0, 0], dtype=np.int64),
        "bear_wick_count": np.array([0, 0, 0], dtype=np.int64),
        "recent_atr_mean": np.array([0.01, 0.01, 0.01], dtype=np.float32),
        "recent_hl_mean": np.array([0.01, 0.01, 0.01], dtype=np.float32),
        "recent_close_std": np.array([0.01, 0.01, 0.01], dtype=np.float32),
        "bull_final_excess": np.array([np.nan, np.nan, np.nan], dtype=np.float32),
        "bear_final_excess": np.array([np.nan, np.nan, np.nan], dtype=np.float32),
    }
    thresholds = {
        "high_vol_atr": 0.05,
        "high_vol_hl": 0.05,
        "true_break_excess": 0.1,
        "chop_std": 0.05,
    }

    bucket0, _ = _assign_domain_bucket(arrays, 0, min_recent_break_bars=2, thresholds=thresholds)
    bucket1, _ = _assign_domain_bucket(arrays, 1, min_recent_break_bars=2, thresholds=thresholds)
    bucket2, _ = _assign_domain_bucket(arrays, 2, min_recent_break_bars=2, thresholds=thresholds)

    assert bucket0 == "borderline_intact_break_up"
    assert bucket1 == "borderline_intact_break_down"
    assert bucket2 == "borderline_intact_break_mixed"


def test_structural_state_fields_factorize_confirmed_pressure_and_holding():
    arrays = {
        "labels": np.array([0, 1, 1, 2], dtype=np.int64),
        "preds": np.array([2, 1, 1, 2], dtype=np.int64),
        "probs": np.array(
            [
                [0.4, 0.1, 0.5],
                [0.1, 0.8, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ],
            dtype=np.float32,
        ),
        "symbols": np.array(["BTC", "ETH", "SOL", "XRP"]),
        "recent_atr_mean": np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float32),
        "recent_hl_mean": np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float32),
        "recent_close_std": np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float32),
        "bull_close_count": np.array([0, 3, 3, 0], dtype=np.int64),
        "bear_close_count": np.array([1, 0, 0, 4], dtype=np.int64),
        "bull_wick_count": np.array([0, 0, 0, 0], dtype=np.int64),
        "bear_wick_count": np.array([1, 0, 0, 0], dtype=np.int64),
        "bull_final_excess": np.array([-0.01, 0.02, -0.01, np.nan], dtype=np.float32),
        "bear_final_excess": np.array([-0.002, np.nan, np.nan, 0.02], dtype=np.float32),
    }
    thresholds = {
        "high_vol_atr": 0.05,
        "high_vol_hl": 0.05,
        "true_break_excess": 0.015,
        "chop_std": 0.05,
    }

    intact = _structural_state_fields(arrays, 0, min_recent_break_bars=2, thresholds=thresholds)
    bullish_holding = _structural_state_fields(arrays, 1, min_recent_break_bars=2, thresholds=thresholds)
    bullish_reverted = _structural_state_fields(arrays, 2, min_recent_break_bars=2, thresholds=thresholds)
    bearish_holding = _structural_state_fields(arrays, 3, min_recent_break_bars=2, thresholds=thresholds)

    assert intact["confirmed_state"] == "intact"
    assert intact["pressure_state"] == "down_pressure"
    assert intact["break_maturity"] == "borderline"
    assert intact["holding_status"] == "reverted"
    assert bullish_holding["break_maturity"] == "strong_confirmed"
    assert bullish_holding["holding_status"] == "holding"
    assert bullish_reverted["holding_status"] == "reverted"
    assert bearish_holding["structural_direction"] == "down"
    assert bearish_holding["holding_status"] == "holding"


def test_full_distribution_rows_split_confirmed_break_holding_status():
    arrays = {
        "labels": np.array([1, 1], dtype=np.int64),
        "preds": np.array([1, 1], dtype=np.int64),
        "probs": np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32),
        "symbols": np.array(["BTC", "BTC"]),
        "recent_atr_mean": np.array([0.01, 0.01], dtype=np.float32),
        "recent_hl_mean": np.array([0.01, 0.01], dtype=np.float32),
        "recent_close_std": np.array([0.01, 0.01], dtype=np.float32),
        "bull_close_count": np.array([3, 3], dtype=np.int64),
        "bear_close_count": np.array([0, 0], dtype=np.int64),
        "bull_wick_count": np.array([0, 0], dtype=np.int64),
        "bear_wick_count": np.array([0, 0], dtype=np.int64),
        "bull_final_excess": np.array([0.02, -0.01], dtype=np.float32),
        "bear_final_excess": np.array([np.nan, np.nan], dtype=np.float32),
    }
    thresholds = {
        "high_vol_atr": 0.05,
        "high_vol_hl": 0.05,
        "true_break_excess": 0.015,
        "chop_std": 0.05,
    }

    rows = _full_distribution_rows(arrays, min_recent_break_bars=2, thresholds=thresholds)

    assert {row["holding_status"] for row in rows} == {"holding", "reverted"}


def test_embedding_bucket_rows_capture_neighbor_quality():
    arrays = {
        "z_short": np.array([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]], dtype=np.float32),
        "labels": np.array([0, 0, 1, 1], dtype=np.int64),
        "preds": np.array([0, 0, 1, 0], dtype=np.int64),
        "probs": np.array(
            [
                [0.9, 0.05, 0.05],
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.45, 0.45, 0.1],
            ],
            dtype=np.float32,
        ),
        "symbols": np.array(["BTC", "BTC", "ETH", "ETH"]),
    }
    buckets = ["clean_intact", "clean_intact", "close_confirmed_break_up", "close_confirmed_break_up"]
    reasons = ["x", "x", "y", "y"]

    rows, summary = _embedding_bucket_rows(arrays, "z_short", buckets, reasons, top_k=2)

    assert len(rows) == 2
    assert summary["embedding_key"] == "z_short"
    assert summary["mean_nn_label_agreement"] >= 0.5


def test_compute_intact_pressure_labels_separates_up_down_and_neutral():
    arrays = {
        "labels": np.array([0, 0, 0, 1], dtype=np.int64),
        "timestamps": np.array([1, 2, 3, 4], dtype=np.int64),
        "symbols": np.array(["BTC", "ETH", "SOL", "XRP"]),
        "bull_close_count": np.array([1, 0, 0, 2], dtype=np.int64),
        "bear_close_count": np.array([0, 1, 0, 0], dtype=np.int64),
        "bull_wick_count": np.array([2, 0, 0, 0], dtype=np.int64),
        "bear_wick_count": np.array([0, 2, 0, 0], dtype=np.int64),
        "bull_final_excess": np.array([-0.0005, -0.0040, -0.0100, 0.0100], dtype=np.float32),
        "bear_final_excess": np.array([-0.0050, -0.0004, -0.0100, np.nan], dtype=np.float32),
        "bull_max_excess": np.array([0.0002, -0.0020, -0.0100, 0.0200], dtype=np.float32),
        "bear_max_excess": np.array([-0.0040, 0.0003, -0.0100, np.nan], dtype=np.float32),
    }
    derived = compute_intact_pressure_labels(arrays, label_config={"min_break_pct": 0.002, "recent_bars": 12})

    assert derived["pressure_label"].tolist() == ["up_pressure", "down_pressure", "neutral", "non_intact"]
    assert derived["up_pressure_score"][0] > derived["down_pressure_score"][0]
    assert derived["down_pressure_score"][1] > derived["up_pressure_score"][1]


def test_build_pressure_label_frame_includes_join_keys():
    arrays = {
        "labels": np.array([0, 2], dtype=np.int64),
        "timestamps": np.array([100, 200], dtype=np.int64),
        "symbols": np.array(["BTC_USDT_15m", "ETH_USDT_15m"]),
        "bull_close_count": np.array([0, 0], dtype=np.int64),
        "bear_close_count": np.array([0, 3], dtype=np.int64),
        "bull_wick_count": np.array([0, 0], dtype=np.int64),
        "bear_wick_count": np.array([0, 0], dtype=np.int64),
        "bull_final_excess": np.array([-0.01, np.nan], dtype=np.float32),
        "bear_final_excess": np.array([-0.01, 0.01], dtype=np.float32),
        "bull_max_excess": np.array([-0.01, np.nan], dtype=np.float32),
        "bear_max_excess": np.array([-0.01, 0.02], dtype=np.float32),
    }

    frame = build_pressure_label_frame(arrays, label_config={"min_break_pct": 0.002})

    assert frame.columns[:5] == ["index", "symbol", "timestamp", "base_label", "base_label_name"]
    assert frame["pressure_label"].to_list() == ["neutral", "non_intact"]


def test_build_structural_training_targets_maps_factor_pressure_and_maturity():
    labels = np.array([0, 0, 0, 1, 2], dtype=np.int64)
    domain_summary = {
        "recent_atr_mean": np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        "recent_hl_mean": np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        "bull_close_count": np.array([0, 1, 0, 2, 0], dtype=np.int64),
        "bear_close_count": np.array([0, 0, 1, 0, 2], dtype=np.int64),
        "bull_wick_count": np.array([0, 0, 1, 0, 0], dtype=np.int64),
        "bear_wick_count": np.array([0, 0, 1, 0, 0], dtype=np.int64),
        "bull_final_excess": np.array([-0.01, 0.01, -0.01, 0.02, -0.01], dtype=np.float32),
        "bear_final_excess": np.array([-0.01, -0.01, 0.01, -0.01, 0.02], dtype=np.float32),
        "bull_max_excess": np.array([0.0, 0.01, 0.01, 0.02, 0.0], dtype=np.float32),
        "bear_max_excess": np.array([0.0, 0.0, 0.01, 0.0, 0.02], dtype=np.float32),
    }

    targets = build_structural_training_targets(labels, domain_summary, min_recent_break_bars=2)

    assert targets["factor_targets"].tolist() == [0, 3, 5, 6, 7]
    assert targets["pressure_targets"].tolist() == [0, 1, 3, 4, 4]
    assert targets["maturity_targets"].tolist() == [0, 2, 2, 3, 3]
