"""Probe level-proximity labels from frozen Stage 1A latent exports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from msb_repr.stage1a.analysis import load_latent_export, write_rows_csv

TARGET_LABEL_ORDER = {
    "nearest_level_side": ["bull", "bear"],
    "nearest_distance_bucket": ["very_near", "near", "watch", "far"],
    "wick_breach_side": ["none", "bull", "bear", "both"],
}


def _finite_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _distance_bucket(distance: float) -> str:
    if distance <= 0.001:
        return "very_near"
    if distance <= 0.002:
        return "near"
    if distance <= 0.005:
        return "watch"
    return "far"


def _build_rows(arrays: dict[str, np.ndarray], label_path: Path, target: str, intact_only: bool) -> list[dict[str, Any]]:
    labels_df = pl.read_parquet(label_path)
    rows: list[dict[str, Any]] = []

    for row in labels_df.iter_rows(named=True):
        idx = int(row["index"])
        current_label = str(row.get("current_label", ""))
        if intact_only and current_label != "intact":
            continue

        current_close = _finite_float(row.get("current_close"))
        bull_level = _finite_float(row.get("bull_level"))
        bear_level = _finite_float(row.get("bear_level"))
        if not np.isfinite(current_close) or current_close <= 0:
            continue
        if not np.isfinite(bull_level) or not np.isfinite(bear_level):
            continue

        bull_distance_pct = (bull_level - current_close) / current_close
        bear_distance_pct = (current_close - bear_level) / current_close
        abs_bull_distance = abs(bull_distance_pct)
        abs_bear_distance = abs(bear_distance_pct)
        nearest_side = "bull" if abs_bull_distance <= abs_bear_distance else "bear"
        nearest_distance_pct = min(abs_bull_distance, abs_bear_distance)

        bull_wick = int(arrays["bull_wick_count"][idx]) > 0
        bear_wick = int(arrays["bear_wick_count"][idx]) > 0
        if bull_wick and bear_wick:
            wick_breach_side = "both"
        elif bull_wick:
            wick_breach_side = "bull"
        elif bear_wick:
            wick_breach_side = "bear"
        else:
            wick_breach_side = "none"

        target_label = {
            "nearest_level_side": nearest_side,
            "nearest_distance_bucket": _distance_bucket(nearest_distance_pct),
            "wick_breach_side": wick_breach_side,
        }[target]

        rows.append(
            {
                "index": idx,
                "symbol": str(row.get("symbol", "")),
                "timestamp": int(row.get("timestamp", 0)),
                "current_label": current_label,
                "current_close": current_close,
                "bull_level": bull_level,
                "bear_level": bear_level,
                "bull_distance_pct": float(bull_distance_pct),
                "bear_distance_pct": float(bear_distance_pct),
                "nearest_level_side": nearest_side,
                "nearest_distance_pct": float(nearest_distance_pct),
                "nearest_distance_bucket": _distance_bucket(nearest_distance_pct),
                "bull_wick_count": int(arrays["bull_wick_count"][idx]),
                "bear_wick_count": int(arrays["bear_wick_count"][idx]),
                "wick_breach_side": wick_breach_side,
                "_target_label": target_label,
            }
        )
    return rows


def _load_split(
    latent_path: Path,
    label_path: Path,
    embedding_key: str,
    target: str,
    intact_only: bool,
    drop_labels: set[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    arrays, metadata = load_latent_export(latent_path)
    if embedding_key not in arrays:
        raise ValueError(f"Embedding key not found in latent export: {embedding_key}")
    if target not in TARGET_LABEL_ORDER:
        raise ValueError(f"Unknown target: {target}. Options: {sorted(TARGET_LABEL_ORDER)}")

    rows = _build_rows(arrays, label_path, target=target, intact_only=intact_only)
    rows = [row for row in rows if str(row["_target_label"]) not in drop_labels]
    indices = np.array([int(row["index"]) for row in rows], dtype=np.int64)
    x = arrays[embedding_key][indices].astype(np.float32)
    y = np.array([str(row["_target_label"]) for row in rows], dtype=str)
    return x, y, rows, metadata


def _encode_labels(
    target: str,
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, str], dict[str, int]]:
    present = set(train_labels.tolist()) | set(eval_labels.tolist())
    ordered = [label for label in TARGET_LABEL_ORDER[target] if label in present]
    missing_train = sorted(present - set(train_labels.tolist()))
    if missing_train:
        raise ValueError(f"Eval contains labels absent from train split: {missing_train}")
    if len(ordered) < 2:
        raise ValueError(f"Probe target needs at least two classes, found: {ordered}")
    label_to_id = {label: idx for idx, label in enumerate(ordered)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    y_train = np.array([label_to_id[str(label)] for label in train_labels], dtype=np.int64)
    y_eval = np.array([label_to_id[str(label)] for label in eval_labels], dtype=np.int64)
    return y_train, y_eval, id_to_label, label_to_id


def _fit_probes(x_train: np.ndarray, y_train: np.ndarray, random_state: int) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Proximity probe requires scikit-learn. Run `uv pip install -e \".[viz]\"`.") from exc

    linear = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
    )
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=random_state,
        ),
    )
    linear.fit(x_train, y_train)
    mlp.fit(x_train, y_train)
    return {"linear": linear, "mlp": mlp}


def _evaluate(model: Any, x: np.ndarray, y: np.ndarray, id_to_label: dict[int, str]) -> dict[str, Any]:
    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    except ImportError as exc:
        raise RuntimeError("Proximity probe requires scikit-learn. Run `uv pip install -e \".[viz]\"`.") from exc

    label_ids = sorted(id_to_label)
    label_names = [id_to_label[idx] for idx in label_ids]
    preds = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, preds, labels=label_ids, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y,
            preds,
            labels=label_ids,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": label_names,
            "matrix": confusion_matrix(y, preds, labels=label_ids).astype(int).tolist(),
        },
        "preds": np.array([id_to_label[int(idx)] for idx in preds], dtype=str),
    }


def _majority_baseline(y_train: np.ndarray, y_eval: np.ndarray, id_to_label: dict[int, str]) -> dict[str, Any]:
    try:
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError as exc:
        raise RuntimeError("Proximity probe requires scikit-learn. Run `uv pip install -e \".[viz]\"`.") from exc

    values, counts = np.unique(y_train, return_counts=True)
    majority_id = int(values[np.argmax(counts)])
    preds = np.full(len(y_eval), majority_id, dtype=np.int64)
    return {
        "majority_label": id_to_label[majority_id],
        "accuracy": float(accuracy_score(y_eval, preds)),
        "macro_f1": float(f1_score(y_eval, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_eval, preds, average="weighted", zero_division=0)),
    }


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    return {label: int(count) for label, count in sorted(Counter(labels.tolist()).items())}


def _prediction_rows(rows: list[dict[str, Any]], y_pred: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row, pred in zip(rows, y_pred, strict=True):
        target_label = str(row["_target_label"])
        out.append({**row, "probe_pred": str(pred), "probe_correct": target_label == str(pred)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-latent-path", type=Path, required=True)
    parser.add_argument("--train-label-path", type=Path, required=True)
    parser.add_argument("--eval-latent-path", type=Path, required=True)
    parser.add_argument("--eval-label-path", type=Path, required=True)
    parser.add_argument("--embedding-key", type=str, default="z_fused")
    parser.add_argument("--target", type=str, default="nearest_level_side", choices=sorted(TARGET_LABEL_ORDER))
    parser.add_argument("--drop-labels", nargs="*", default=[])
    parser.add_argument("--include-non-intact", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    x_train, y_train_raw, train_rows, train_metadata = _load_split(
        args.train_latent_path,
        args.train_label_path,
        args.embedding_key,
        args.target,
        intact_only=not args.include_non_intact,
        drop_labels=set(args.drop_labels),
    )
    x_eval, y_eval_raw, eval_rows, eval_metadata = _load_split(
        args.eval_latent_path,
        args.eval_label_path,
        args.embedding_key,
        args.target,
        intact_only=not args.include_non_intact,
        drop_labels=set(args.drop_labels),
    )
    y_train, y_eval, id_to_label, label_to_id = _encode_labels(args.target, y_train_raw, y_eval_raw)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    probes = _fit_probes(x_train, y_train, args.random_state)
    majority = _majority_baseline(y_train, y_eval, id_to_label)

    summary: dict[str, Any] = {
        "target": args.target,
        "embedding_key": args.embedding_key,
        "intact_only": not args.include_non_intact,
        "drop_labels": list(args.drop_labels),
        "train_latent_path": str(args.train_latent_path),
        "eval_latent_path": str(args.eval_latent_path),
        "train_label_path": str(args.train_label_path),
        "eval_label_path": str(args.eval_label_path),
        "train_samples": int(len(y_train)),
        "eval_samples": int(len(y_eval)),
        "train_label_counts": _label_counts(y_train_raw),
        "eval_label_counts": _label_counts(y_eval_raw),
        "label_to_id": label_to_id,
        "majority_baseline": majority,
        "results": {},
        "train_export_metadata": train_metadata,
        "eval_export_metadata": eval_metadata,
    }

    for name, probe in probes.items():
        result = _evaluate(probe, x_eval, y_eval, id_to_label)
        summary["results"][name] = {key: value for key, value in result.items() if key != "preds"}
        pred_rows = _prediction_rows(eval_rows, result["preds"])
        write_rows_csv(
            args.output_dir / f"{name}_predictions.csv",
            list(pred_rows[0].keys()) if pred_rows else [],
            pred_rows,
        )

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "Proximity probe complete: output_dir={} target={} embedding={} linear_macro_f1={:.4f} mlp_macro_f1={:.4f}",
        args.output_dir,
        args.target,
        args.embedding_key,
        summary["results"]["linear"]["macro_f1"],
        summary["results"]["mlp"]["macro_f1"],
    )


if __name__ == "__main__":
    main()
