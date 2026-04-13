"""Training loop for the Stage 1A CE-only baseline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from msb_repr.stage1a.model import Stage1AModel


@dataclass
class Stage1ATrainerConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    checkpoint_dir: Path = Path("data/checkpoints/stage1a")


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> tuple[float, dict[str, float]]:
    per_class: dict[str, float] = {}
    scores = []
    for cls in range(num_classes):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        per_class[f"class_{cls}_f1"] = f1
        scores.append(f1)
    return float(np.mean(scores)), per_class


class Stage1ATrainer:
    def __init__(self, model: Stage1AModel, device: torch.device, cfg: Stage1ATrainerConfig) -> None:
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
        best_val_loss = float("inf")
        patience_counter = 0
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_macro_f1": [],
        }
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, _, _ = self._run_epoch(train_loader, train=True)
            val_loss, val_macro_f1, val_metrics = self._run_epoch(val_loader, train=False)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_macro_f1"].append(val_macro_f1)

            logger.info(
                "Epoch {}/{} | train_loss={:.5f} val_loss={:.5f} val_macro_f1={:.4f}",
                epoch,
                self.cfg.epochs,
                train_loss,
                val_loss,
                val_macro_f1,
            )
            logger.debug("Val metrics: {}", val_metrics)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint("best.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.patience:
                    logger.info("Early stopping at epoch {}", epoch)
                    break

        self._save_checkpoint("final.pt")
        self._save_history(history)
        return history

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float, dict[str, float]]:
        self.model.train(train)
        total_loss = 0.0
        preds = []
        labels = []

        with torch.set_grad_enabled(train):
            for short_x, long_x, y, _ in loader:
                short_x = short_x.to(self.device)
                long_x = long_x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(short_x, long_x)
                loss = self.criterion(outputs["logits"], y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * short_x.size(0)
                preds.append(outputs["logits"].argmax(dim=1).detach().cpu().numpy())
                labels.append(y.detach().cpu().numpy())

        y_true = np.concatenate(labels)
        y_pred = np.concatenate(preds)
        macro_f1, per_class = _macro_f1(y_true, y_pred)
        return total_loss / len(loader.dataset), macro_f1, per_class  # type: ignore[arg-type]

    def _save_checkpoint(self, name: str) -> None:
        path = self.cfg.checkpoint_dir / name
        torch.save({"model_state_dict": self.model.state_dict()}, path)

    def _save_history(self, history: dict[str, list[float]]) -> None:
        path = self.cfg.checkpoint_dir / "history.json"
        path.write_text(json.dumps(history, indent=2))

    def save_config(self) -> None:
        path = self.cfg.checkpoint_dir / "train_config.json"
        path.write_text(json.dumps(asdict(self.cfg), indent=2, default=str))
