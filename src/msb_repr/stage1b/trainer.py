"""Training loop for Stage 1B forward-structure predictor."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from msb_repr.stage1b.model import Stage1BForwardPredictor


@dataclass
class Stage1BTrainerConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    break_loss_weight: float = 1.0
    direction_loss_weight: float = 1.0
    break_threshold: float = 0.5
    primary_horizon_index: int = 0
    checkpoint_dir: Path = Path("data/checkpoints/stage1b")


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> tuple[float, dict[str, float]]:
    per_label: dict[str, float] = {}
    scores = []
    for label in labels:
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        per_label[f"class_{label}_f1"] = f1
        scores.append(f1)
    return float(np.mean(scores)), per_label


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> list[list[int]]:
    return [
        [int(np.sum((y_true == true_label) & (y_pred == pred_label))) for pred_label in labels]
        for true_label in labels
    ]


class Stage1BTrainer:
    def __init__(
        self,
        model: Stage1BForwardPredictor,
        device: torch.device,
        cfg: Stage1BTrainerConfig,
    ) -> None:
        self.model = model
        self.device = device
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, list[float]]:
        best_score = -float("inf")
        patience_counter = 0
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_break_loss": [],
            "train_direction_loss": [],
            "train_recon_macro_f1": [],
            "train_break_macro_f1": [],
            "train_direction_macro_f1": [],
            "val_loss": [],
            "val_break_loss": [],
            "val_direction_loss": [],
            "val_recon_macro_f1": [],
            "val_break_macro_f1": [],
            "val_direction_macro_f1": [],
        }
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)
            for key in [
                "loss",
                "break_loss",
                "direction_loss",
                "recon_macro_f1",
                "break_macro_f1",
                "direction_macro_f1",
            ]:
                history[f"train_{key}"].append(float(train_metrics[key]))
                history[f"val_{key}"].append(float(val_metrics[key]))

            logger.info(
                "Epoch {}/{} | train_loss={:.5f} val_loss={:.5f} "
                "val_recon_f1={:.4f} val_break_f1={:.4f} val_dir_f1={:.4f}",
                epoch,
                self.cfg.epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["recon_macro_f1"],
                val_metrics["break_macro_f1"],
                val_metrics["direction_macro_f1"],
            )

            score = float(val_metrics["recon_macro_f1"])
            if score > best_score:
                best_score = score
                patience_counter = 0
                self._save_checkpoint("best.pt", epoch=epoch, metrics=val_metrics)
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.patience:
                    logger.info("Early stopping at epoch {}", epoch)
                    break

        final_metrics = self._run_epoch(val_loader, train=False)
        self._save_checkpoint("final.pt", epoch=len(history["val_loss"]), metrics=final_metrics)
        self._save_history(history)
        return history

    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        return self._run_epoch(loader, train=False, include_confusion=True)

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        include_confusion: bool = False,
    ) -> dict[str, Any]:
        self.model.train(train)
        total_loss = 0.0
        total_break_loss = 0.0
        total_direction_loss = 0.0
        total_samples = 0
        break_true = []
        break_pred = []
        direction_true = []
        direction_pred = []
        recon_true = []
        recon_pred = []
        all_break_true = []
        all_break_pred = []
        all_direction_true = []
        all_direction_pred = []
        all_recon_true = []
        all_recon_pred = []

        with torch.set_grad_enabled(train):
            for features, break_targets, direction_targets, recon_targets, _metas in loader:
                features = features.to(self.device)
                break_targets = break_targets.to(self.device)
                direction_targets = direction_targets.to(self.device)
                recon_targets = recon_targets.to(self.device)
                outputs = self.model(features)
                break_logits = outputs["break_logits"]
                direction_logits = outputs["direction_logits"]
                if break_targets.ndim == 1:
                    break_targets = break_targets[:, None]
                    direction_targets = direction_targets[:, None]
                    recon_targets = recon_targets[:, None]
                break_loss = self.criterion(
                    break_logits.reshape(-1, 2),
                    break_targets.reshape(-1),
                )

                direction_mask = direction_targets >= 0
                if bool(direction_mask.any()):
                    direction_loss = self.criterion(
                        direction_logits[direction_mask],
                        direction_targets[direction_mask],
                    )
                else:
                    direction_loss = break_loss.new_zeros(())

                loss = (
                    self.cfg.break_loss_weight * break_loss
                    + self.cfg.direction_loss_weight * direction_loss
                )
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                batch_size = int(features.size(0))
                total_samples += batch_size
                total_loss += float(loss.item()) * batch_size
                total_break_loss += float(break_loss.item()) * batch_size
                total_direction_loss += float(direction_loss.item()) * int(direction_mask.sum().item())

                break_probs = torch.softmax(break_logits, dim=2)[:, :, 1]
                batch_break_pred = (break_probs >= self.cfg.break_threshold).long()
                batch_direction_pred = direction_logits.argmax(dim=2)
                batch_recon_pred = torch.zeros_like(recon_targets)
                batch_recon_pred[batch_break_pred == 1] = batch_direction_pred[batch_break_pred == 1] + 1

                h = self.cfg.primary_horizon_index
                all_break_true.append(break_targets.detach().cpu().numpy())
                all_break_pred.append(batch_break_pred.detach().cpu().numpy())
                all_recon_true.append(recon_targets.detach().cpu().numpy())
                all_recon_pred.append(batch_recon_pred.detach().cpu().numpy())
                all_direction_true.append(direction_targets.detach().cpu().numpy())
                all_direction_pred.append(batch_direction_pred.detach().cpu().numpy())
                break_true.append(break_targets[:, h].detach().cpu().numpy())
                break_pred.append(batch_break_pred[:, h].detach().cpu().numpy())
                recon_true.append(recon_targets[:, h].detach().cpu().numpy())
                recon_pred.append(batch_recon_pred[:, h].detach().cpu().numpy())
                if bool(direction_mask.any()):
                    primary_direction_mask = direction_mask[:, h]
                    if bool(primary_direction_mask.any()):
                        direction_true.append(direction_targets[:, h][primary_direction_mask].detach().cpu().numpy())
                        direction_pred.append(batch_direction_pred[:, h][primary_direction_mask].detach().cpu().numpy())

        y_break_true = np.concatenate(break_true)
        y_break_pred = np.concatenate(break_pred)
        y_recon_true = np.concatenate(recon_true)
        y_recon_pred = np.concatenate(recon_pred)
        if direction_true:
            y_direction_true = np.concatenate(direction_true)
            y_direction_pred = np.concatenate(direction_pred)
            direction_macro_f1, direction_per_class = _macro_f1(y_direction_true, y_direction_pred, labels=[0, 1])
        else:
            y_direction_true = np.array([], dtype=np.int64)
            y_direction_pred = np.array([], dtype=np.int64)
            direction_macro_f1, direction_per_class = 0.0, {"class_0_f1": 0.0, "class_1_f1": 0.0}

        break_macro_f1, break_per_class = _macro_f1(y_break_true, y_break_pred, labels=[0, 1])
        recon_macro_f1, recon_per_class = _macro_f1(y_recon_true, y_recon_pred, labels=[0, 1, 2])
        direction_count = max(1, int(sum(len(values) for values in direction_true)))
        metrics: dict[str, Any] = {
            "loss": total_loss / max(total_samples, 1),
            "break_loss": total_break_loss / max(total_samples, 1),
            "direction_loss": total_direction_loss / direction_count,
            "break_macro_f1": break_macro_f1,
            "break_per_class_f1": break_per_class,
            "direction_macro_f1": direction_macro_f1,
            "direction_per_class_f1": direction_per_class,
            "recon_macro_f1": recon_macro_f1,
            "recon_per_class_f1": recon_per_class,
            "break_accuracy": float(np.mean(y_break_true == y_break_pred)),
            "direction_accuracy": float(np.mean(y_direction_true == y_direction_pred)) if len(y_direction_true) else 0.0,
            "recon_accuracy": float(np.mean(y_recon_true == y_recon_pred)),
        }
        all_y_break_true = np.concatenate(all_break_true, axis=0)
        all_y_break_pred = np.concatenate(all_break_pred, axis=0)
        all_y_recon_true = np.concatenate(all_recon_true, axis=0)
        all_y_recon_pred = np.concatenate(all_recon_pred, axis=0)
        all_y_direction_true = np.concatenate(all_direction_true, axis=0)
        all_y_direction_pred = np.concatenate(all_direction_pred, axis=0)
        per_horizon = []
        for horizon_idx in range(all_y_break_true.shape[1]):
            h_break_f1, h_break_per_class = _macro_f1(
                all_y_break_true[:, horizon_idx],
                all_y_break_pred[:, horizon_idx],
                labels=[0, 1],
            )
            h_recon_f1, h_recon_per_class = _macro_f1(
                all_y_recon_true[:, horizon_idx],
                all_y_recon_pred[:, horizon_idx],
                labels=[0, 1, 2],
            )
            h_direction_mask = all_y_direction_true[:, horizon_idx] >= 0
            h_direction_f1, h_direction_per_class = _macro_f1(
                all_y_direction_true[:, horizon_idx][h_direction_mask],
                all_y_direction_pred[:, horizon_idx][h_direction_mask],
                labels=[0, 1],
            )
            per_horizon.append(
                {
                    "horizon_index": horizon_idx,
                    "break_macro_f1": h_break_f1,
                    "break_per_class_f1": h_break_per_class,
                    "break_accuracy": float(np.mean(all_y_break_true[:, horizon_idx] == all_y_break_pred[:, horizon_idx])),
                    "direction_macro_f1": h_direction_f1,
                    "direction_per_class_f1": h_direction_per_class,
                    "direction_accuracy": float(
                        np.mean(
                            all_y_direction_true[:, horizon_idx][h_direction_mask]
                            == all_y_direction_pred[:, horizon_idx][h_direction_mask]
                        )
                    ),
                    "recon_macro_f1": h_recon_f1,
                    "recon_per_class_f1": h_recon_per_class,
                    "recon_accuracy": float(np.mean(all_y_recon_true[:, horizon_idx] == all_y_recon_pred[:, horizon_idx])),
                }
            )
        metrics["per_horizon"] = per_horizon
        if include_confusion:
            metrics["break_confusion_matrix"] = {
                "labels": ["none", "break"],
                "matrix": _confusion_matrix(y_break_true, y_break_pred, labels=[0, 1]),
            }
            metrics["direction_confusion_matrix"] = {
                "labels": ["bullish", "bearish"],
                "matrix": _confusion_matrix(y_direction_true, y_direction_pred, labels=[0, 1])
                if len(y_direction_true)
                else [[0, 0], [0, 0]],
            }
            metrics["recon_confusion_matrix"] = {
                "labels": ["none", "bullish", "bearish"],
                "matrix": _confusion_matrix(y_recon_true, y_recon_pred, labels=[0, 1, 2]),
            }
        return metrics

    def save_config(self) -> None:
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config = asdict(self.cfg)
        config["checkpoint_dir"] = str(self.cfg.checkpoint_dir)
        (self.cfg.checkpoint_dir / "train_config.json").write_text(json.dumps(config, indent=2))

    def _save_checkpoint(self, name: str, epoch: int, metrics: dict[str, Any]) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
            },
            self.cfg.checkpoint_dir / name,
        )

    def _save_history(self, history: dict[str, list[float]]) -> None:
        (self.cfg.checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))
