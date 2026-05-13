"""Probe factorized Stage 1A structural-state labels from frozen latent exports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from msb_repr.stage1a.analysis import LABEL_NAMES, load_latent_export, write_rows_csv

try:
    from scripts.analyze_stage1a_latents import _build_bucket_thresholds, _structural_state_fields
except ModuleNotFoundError:  # Direct execution via `python scripts/...`.
    from analyze_stage1a_latents import _build_bucket_thresholds, _structural_state_fields


TARGET_ORDERS = {
    "maturity_binary": ["borderline_intact", "confirmed_break"],
    "break_maturity": [
        "clean",
        "wick_sweep",
        "borderline",
        "confirmed",
        "strong_confirmed",
        "weak_or_choppy_confirmed",
        "high_vol_intact",
        "high_vol_intact_wick_sweep",
    ],
    "pressure_state": ["neutral", "up_pressure", "down_pressure", "mixed_pressure", "non_intact"],
    "holding_status": ["none", "holding", "reverted", "mixed", "unknown"],
}


def _target_label(fields: dict[str, str], target: str) -> str | None:
    if target == "maturity_binary":
        if fields["confirmed_state"] == "intact" and fields["break_maturity"] == "borderline":
            return "borderline_intact"
        if fields["confirmed_state"] in {"bullish", "bearish"} and fields["break_maturity"] in {
            "confirmed",
            "strong_confirmed",
        }:
            return "confirmed_break"
        return None
    if target == "break_maturity":
        return fields["break_maturity"]
    if target == "pressure_state":
        return fields["pressure_state"]
    if target == "holding_status":
        return fields["holding_status"]
    raise ValueError(f"Unknown probe target: {target}")


def _build_structural_rows(
    arrays: dict[str, np.ndarray],
    min_recent_break_bars: int,
    target: str,
) -> list[dict[str, Any]]:
    thresholds = _build_bucket_thresholds(arrays)
    rows: list[dict[str, Any]] = []
    for idx in range(len(arrays["labels"])):
        fields = _structural_state_fields(arrays, idx, min_recent_break_bars, thresholds)
        target_label = _target_label(fields, target)
        if target_label is None or target_label == "unknown":
            continue
        rows.append(
            {
                "index": idx,
                "target_label": target_label,
                "symbol": str(arrays["symbols"][idx]),
                "timestamp": int(arrays["timestamps"][idx]),
                "label_name": LABEL_NAMES[int(arrays["labels"][idx])],
                "pred_name": LABEL_NAMES[int(arrays["preds"][idx])],
                "model_correct": bool(arrays["labels"][idx] == arrays["preds"][idx]),
                "confidence": float(arrays["probs"][idx].max()),
                **fields,
            }
        )
    return rows


def _load_probe_split(
    latent_path: Path,
    embedding_key: str,
    target: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    arrays, metadata = load_latent_export(latent_path)
    if embedding_key not in arrays:
        raise ValueError(f"Embedding key not found in latent export: {embedding_key}")

    label_config = metadata.get("label_config", {})
    min_recent_break_bars = int(label_config.get("min_recent_break_bars", 2))
    rows = _build_structural_rows(arrays, min_recent_break_bars, target)
    indices = np.array([int(row["index"]) for row in rows], dtype=np.int64)
    features = arrays[embedding_key][indices]
    labels = np.array([str(row["target_label"]) for row in rows], dtype=str)
    return features.astype(np.float32), labels, rows, metadata


def _encode_labels(
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
    target: str,
) -> tuple[np.ndarray, np.ndarray, dict[int, str], dict[str, int]]:
    present = set(train_labels.tolist()) | set(eval_labels.tolist())
    ordered = [label for label in TARGET_ORDERS[target] if label in present]
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
        raise RuntimeError("Structural probe için scikit-learn gerekli. `uv pip install -e \".[viz]\"`.") from exc

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


def _evaluate_probe(model: Any, x: np.ndarray, y: np.ndarray, id_to_label: dict[int, str]) -> dict[str, Any]:
    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    except ImportError as exc:
        raise RuntimeError("Structural probe için scikit-learn gerekli. `uv pip install -e \".[viz]\"`.") from exc

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
        raise RuntimeError("Structural probe için scikit-learn gerekli. `uv pip install -e \".[viz]\"`.") from exc

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
        out.append({**row, "probe_pred": str(pred), "probe_correct": row["target_label"] == str(pred)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-latent-path", type=Path, required=True)
    parser.add_argument("--eval-latent-path", type=Path, required=True)
    parser.add_argument("--embedding-key", type=str, default="z_fused")
    parser.add_argument("--target", choices=sorted(TARGET_ORDERS), default="maturity_binary")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    x_train, y_train_raw, train_rows, train_metadata = _load_probe_split(
        args.train_latent_path,
        args.embedding_key,
        args.target,
    )
    x_eval, y_eval_raw, eval_rows, eval_metadata = _load_probe_split(
        args.eval_latent_path,
        args.embedding_key,
        args.target,
    )
    y_train, y_eval, id_to_label, label_to_id = _encode_labels(y_train_raw, y_eval_raw, args.target)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    probes = _fit_probes(x_train, y_train, random_state=args.random_state)
    majority = _majority_baseline(y_train, y_eval, id_to_label)

    summary: dict[str, Any] = {
        "target": args.target,
        "embedding_key": args.embedding_key,
        "train_latent_path": str(args.train_latent_path),
        "eval_latent_path": str(args.eval_latent_path),
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
        result = _evaluate_probe(probe, x_eval, y_eval, id_to_label)
        summary["results"][name] = {key: value for key, value in result.items() if key != "preds"}
        pred_rows = _prediction_rows(eval_rows, result["preds"])
        write_rows_csv(
            args.output_dir / f"{name}_predictions.csv",
            list(pred_rows[0].keys()) if pred_rows else [],
            pred_rows,
        )

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "Structural probe complete: output_dir={} target={} embedding={} linear_macro_f1={:.4f} mlp_macro_f1={:.4f}",
        args.output_dir,
        args.target,
        args.embedding_key,
        summary["results"]["linear"]["macro_f1"],
        summary["results"]["mlp"]["macro_f1"],
    )


if __name__ == "__main__":
    main()
