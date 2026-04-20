"""Train linear and MLP probes on Stage 1A latent exports for intact pressure labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from msb_repr.stage1a.analysis import load_latent_export, write_rows_csv

PRESSURE_LABEL_ORDER = ["neutral", "up_pressure", "down_pressure"]


def _load_probe_split(latent_path: Path, pressure_path: Path, embedding_key: str) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    arrays, _ = load_latent_export(latent_path)
    if embedding_key not in arrays:
        raise ValueError(f"Embedding key not found in latent export: {embedding_key}")

    pressure_df = pl.read_parquet(pressure_path)
    intact_df = pressure_df.filter(pl.col("is_intact_base_label") & (pl.col("pressure_label") != "non_intact"))
    indices = intact_df["index"].to_numpy()
    features = arrays[embedding_key][indices]
    labels = intact_df["pressure_label"].to_numpy()
    return features.astype(np.float32), labels.astype(str), intact_df


def _encode_labels(labels: np.ndarray) -> tuple[np.ndarray, dict[int, str], dict[str, int]]:
    ordered = [label for label in PRESSURE_LABEL_ORDER if label in set(labels.tolist())]
    label_to_id = {label: idx for idx, label in enumerate(ordered)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    encoded = np.array([label_to_id[str(label)] for label in labels], dtype=np.int64)
    return encoded, id_to_label, label_to_id


def _fit_probes(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Probe için scikit-learn gerekli. `uv pip install -e \".[viz]\"` çalıştırın.") from exc

    linear = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
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


def _evaluate_probe(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    except ImportError as exc:
        raise RuntimeError("Probe için scikit-learn gerekli. `uv pip install -e \".[viz]\"` çalıştırın.") from exc

    eval_label_order = [idx for idx, label in sorted(id_to_label.items())]
    preds = model.predict(x)
    report = classification_report(y, preds, labels=eval_label_order, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y, preds, labels=eval_label_order)
    pred_names = np.array([id_to_label[int(idx)] for idx in preds], dtype="<U16")
    label_names = [id_to_label[int(idx)] for idx in eval_label_order]
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, labels=eval_label_order, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, preds, labels=eval_label_order, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": {
            "labels": label_names,
            "matrix": matrix.astype(int).tolist(),
        },
        "preds": pred_names,
    }


def _majority_baseline(y_train: np.ndarray, y_eval: np.ndarray, id_to_label: dict[int, str]) -> dict[str, Any]:
    values, counts = np.unique(y_train, return_counts=True)
    majority_label = int(values[np.argmax(counts)])
    preds = np.full(len(y_eval), majority_label, dtype=np.int64)

    try:
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError as exc:
        raise RuntimeError("Probe için scikit-learn gerekli. `uv pip install -e \".[viz]\"` çalıştırın.") from exc

    return {
        "majority_label": id_to_label[majority_label],
        "accuracy": float(accuracy_score(y_eval, preds)),
        "macro_f1": float(f1_score(y_eval, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_eval, preds, average="weighted", zero_division=0)),
    }


def _prediction_rows(split_df: pl.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    indices = split_df["index"].to_numpy()
    symbols = split_df["symbol"].to_numpy()
    timestamps = split_df["timestamp"].to_numpy()
    up_scores = split_df["up_pressure_score"].to_numpy()
    down_scores = split_df["down_pressure_score"].to_numpy()

    for idx, sample_idx in enumerate(indices):
        rows.append(
            {
                "index": int(sample_idx),
                "symbol": str(symbols[idx]),
                "timestamp": int(timestamps[idx]),
                "y_true": str(y_true[idx]),
                "y_pred": str(y_pred[idx]),
                "correct": bool(y_true[idx] == y_pred[idx]),
                "up_pressure_score": float(up_scores[idx]),
                "down_pressure_score": float(down_scores[idx]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe intact pressure labels from Stage 1A latent exports")
    parser.add_argument("--train-latent-path", type=Path, required=True)
    parser.add_argument("--train-pressure-path", type=Path, required=True)
    parser.add_argument("--eval-latent-path", type=Path, required=True)
    parser.add_argument("--eval-pressure-path", type=Path, required=True)
    parser.add_argument("--embedding-key", type=str, default="z_fused")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    x_train, y_train_raw, train_df = _load_probe_split(
        args.train_latent_path,
        args.train_pressure_path,
        args.embedding_key,
    )
    x_eval, y_eval_raw, eval_df = _load_probe_split(
        args.eval_latent_path,
        args.eval_pressure_path,
        args.embedding_key,
    )
    y_train, id_to_label, label_to_id = _encode_labels(y_train_raw)
    y_eval = np.array([label_to_id[str(label)] for label in y_eval_raw], dtype=np.int64)

    output_dir = args.output_dir or (args.eval_latent_path.with_suffix("") / f"pressure_probe_{args.embedding_key}")
    output_dir.mkdir(parents=True, exist_ok=True)

    probes = _fit_probes(x_train, y_train, random_state=args.random_state)
    majority = _majority_baseline(y_train, y_eval, id_to_label)

    summary = {
        "embedding_key": args.embedding_key,
        "train_latent_path": str(args.train_latent_path),
        "eval_latent_path": str(args.eval_latent_path),
        "train_pressure_path": str(args.train_pressure_path),
        "eval_pressure_path": str(args.eval_pressure_path),
        "train_samples": int(len(y_train)),
        "eval_samples": int(len(y_eval)),
        "train_label_counts": train_df.group_by("pressure_label").len().sort("pressure_label").to_dicts(),
        "eval_label_counts": eval_df.group_by("pressure_label").len().sort("pressure_label").to_dicts(),
        "majority_baseline": majority,
        "results": {},
    }

    for probe_name, probe in probes.items():
        result = _evaluate_probe(probe, x_eval, y_eval, id_to_label=id_to_label)
        summary["results"][probe_name] = {
            key: value for key, value in result.items() if key != "preds"
        }
        eval_true_names = np.array([id_to_label[int(idx)] for idx in y_eval], dtype="<U16")
        prediction_rows = _prediction_rows(eval_df, eval_true_names, result["preds"])
        write_rows_csv(
            output_dir / f"{probe_name}_predictions.csv",
            list(prediction_rows[0].keys()) if prediction_rows else [],
            prediction_rows,
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "Pressure probe complete: output_dir={} linear_macro_f1={:.4f} mlp_macro_f1={:.4f}",
        output_dir,
        summary["results"]["linear"]["macro_f1"],
        summary["results"]["mlp"]["macro_f1"],
    )


if __name__ == "__main__":
    main()
