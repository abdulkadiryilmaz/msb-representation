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

from msb_repr.stage1a.losses import SupConLoss
from msb_repr.stage1a.model import Stage1AModel


@dataclass
class Stage1ATrainerConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    use_supcon: bool = False
    ce_warmup_weight: float | None = None
    ce_warmup_epochs: int = 0
    supcon_weight: float = 0.05
    supcon_frontload_weight: float | None = None
    supcon_frontload_epochs: int = 0
    supcon_temperature: float = 0.1
    supcon_embedding_key: str = "z_proj"
    supcon_positive_mode: str = "label"
    pressure_loss_weight: float = 0.0
    maturity_loss_weight: float = 0.0
    long_aux_loss_weight: float = 0.0
    seed: int | None = None
    epoch_checkpoints: tuple[int, ...] = ()
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
        self.aux_criterion = nn.CrossEntropyLoss()
        self.supcon_criterion = SupConLoss(cfg.supcon_temperature) if cfg.use_supcon else None

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, list[float]]:
        best_val_loss = float("inf")
        patience_counter = 0
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_ce_loss": [],
            "train_ce_weight": [],
            "train_supcon_loss": [],
            "train_supcon_weight": [],
            "train_pressure_loss": [],
            "train_maturity_loss": [],
            "train_long_aux_loss": [],
            "train_supcon_anchor_rate": [],
            "train_supcon_avg_positive_count": [],
            "val_loss": [],
            "val_ce_loss": [],
            "val_ce_weight": [],
            "val_supcon_loss": [],
            "val_supcon_weight": [],
            "val_pressure_loss": [],
            "val_maturity_loss": [],
            "val_long_aux_loss": [],
            "val_supcon_anchor_rate": [],
            "val_supcon_avg_positive_count": [],
            "val_macro_f1": [],
        }
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            ce_weight = self._ce_weight_for_epoch(epoch)
            supcon_weight = self._supcon_weight_for_epoch(epoch)
            (
                train_loss,
                train_ce_loss,
                train_supcon_loss,
                train_pressure_loss,
                train_maturity_loss,
                train_long_aux_loss,
                train_anchor_rate,
                train_avg_pos,
                _,
                _,
            ) = self._run_epoch(train_loader, train=True, ce_weight=ce_weight, supcon_weight=supcon_weight)
            (
                val_loss,
                val_ce_loss,
                val_supcon_loss,
                val_pressure_loss,
                val_maturity_loss,
                val_long_aux_loss,
                val_anchor_rate,
                val_avg_pos,
                val_macro_f1,
                val_metrics,
            ) = self._run_epoch(val_loader, train=False, ce_weight=ce_weight, supcon_weight=supcon_weight)

            history["train_loss"].append(train_loss)
            history["train_ce_loss"].append(train_ce_loss)
            history["train_ce_weight"].append(ce_weight)
            history["train_supcon_loss"].append(train_supcon_loss)
            history["train_supcon_weight"].append(supcon_weight)
            history["train_pressure_loss"].append(train_pressure_loss)
            history["train_maturity_loss"].append(train_maturity_loss)
            history["train_long_aux_loss"].append(train_long_aux_loss)
            history["train_supcon_anchor_rate"].append(train_anchor_rate)
            history["train_supcon_avg_positive_count"].append(train_avg_pos)
            history["val_loss"].append(val_loss)
            history["val_ce_loss"].append(val_ce_loss)
            history["val_ce_weight"].append(ce_weight)
            history["val_supcon_loss"].append(val_supcon_loss)
            history["val_supcon_weight"].append(supcon_weight)
            history["val_pressure_loss"].append(val_pressure_loss)
            history["val_maturity_loss"].append(val_maturity_loss)
            history["val_long_aux_loss"].append(val_long_aux_loss)
            history["val_supcon_anchor_rate"].append(val_anchor_rate)
            history["val_supcon_avg_positive_count"].append(val_avg_pos)
            history["val_macro_f1"].append(val_macro_f1)

            logger.info(
                "Epoch {}/{} | train_loss={:.5f} val_loss={:.5f} val_macro_f1={:.4f}",
                epoch,
                self.cfg.epochs,
                train_loss,
                val_loss,
                val_macro_f1,
            )
            if self.cfg.use_supcon:
                logger.debug(
                    "Aux losses | train_ce={:.5f} train_supcon={:.5f} train_pressure={:.5f} "
                    "train_maturity={:.5f} train_long_aux={:.5f} "
                    "val_ce={:.5f} val_supcon={:.5f} val_pressure={:.5f} "
                    "val_maturity={:.5f} val_long_aux={:.5f} "
                    "train_anchor_rate={:.4f} val_anchor_rate={:.4f} ce_weight={:.5f} supcon_weight={:.5f}",
                    train_ce_loss,
                    train_supcon_loss,
                    train_pressure_loss,
                    train_maturity_loss,
                    train_long_aux_loss,
                    val_ce_loss,
                    val_supcon_loss,
                    val_pressure_loss,
                    val_maturity_loss,
                    val_long_aux_loss,
                    train_anchor_rate,
                    val_anchor_rate,
                    ce_weight,
                    supcon_weight,
                )
            logger.debug("Val metrics: {}", val_metrics)

            if epoch in self.cfg.epoch_checkpoints:
                self._save_checkpoint(f"epoch_{epoch:03d}.pt")

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

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        ce_weight: float,
        supcon_weight: float,
    ) -> tuple[float, float, float, float, float, float, float, float, float, dict[str, float]]:
        self.model.train(train)
        total_loss = 0.0
        total_ce_loss = 0.0
        total_supcon_loss = 0.0
        total_pressure_loss = 0.0
        total_maturity_loss = 0.0
        total_long_aux_loss = 0.0
        total_supcon_valid_anchors = 0.0
        total_supcon_positive_counts = 0.0
        total_supcon_anchors = 0.0
        preds = []
        labels = []

        with torch.set_grad_enabled(train):
            for short_x, long_x, y, metas in loader:
                short_x = short_x.to(self.device)
                long_x = long_x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(short_x, long_x)
                ce_loss = self.criterion(outputs["logits"], y)
                supcon_loss = ce_loss.new_zeros(())
                pressure_loss = ce_loss.new_zeros(())
                maturity_loss = ce_loss.new_zeros(())
                long_aux_loss = ce_loss.new_zeros(())
                if self.supcon_criterion is not None:
                    if self.cfg.supcon_embedding_key not in outputs:
                        raise KeyError(
                            f"SupCon embedding key not produced by model: {self.cfg.supcon_embedding_key}"
                        )
                    symbol_ids = None
                    supcon_labels = y
                    supcon_positive_mode = self.cfg.supcon_positive_mode
                    if self.cfg.supcon_positive_mode == "factor":
                        if any(meta.factor_target is None for meta in metas):
                            raise ValueError("factor_target metadata is required for supcon_positive_mode='factor'")
                        supcon_labels = torch.tensor(
                            [int(meta.factor_target) for meta in metas],
                            dtype=torch.long,
                            device=self.device,
                        )
                        supcon_positive_mode = "label"
                    if supcon_positive_mode in {"label_diff_symbol", "label_diff_symbol_neutral_same_symbol"}:
                        symbol_to_id = {symbol: idx for idx, symbol in enumerate(dict.fromkeys(meta.symbol for meta in metas))}
                        symbol_ids = torch.tensor(
                            [symbol_to_id[meta.symbol] for meta in metas],
                            dtype=torch.long,
                            device=self.device,
                        )
                    supcon_loss = self.supcon_criterion(
                        outputs[self.cfg.supcon_embedding_key],
                        supcon_labels,
                        symbol_ids=symbol_ids,
                        positive_mode=supcon_positive_mode,
                    )
                    positive_counts = self._supcon_positive_counts(
                        supcon_labels,
                        symbol_ids=symbol_ids,
                        positive_mode=supcon_positive_mode,
                    )
                    total_supcon_valid_anchors += float((positive_counts > 0).sum().item())
                    total_supcon_positive_counts += float(positive_counts.sum().item())
                    total_supcon_anchors += float(len(positive_counts))
                if self.cfg.pressure_loss_weight > 0.0:
                    if "pressure_logits" not in outputs:
                        raise KeyError("pressure_logits not produced by model")
                    if any(meta.pressure_target is None for meta in metas):
                        raise ValueError("pressure_target metadata is required when pressure_loss_weight > 0")
                    pressure_targets = torch.tensor(
                        [int(meta.pressure_target) for meta in metas],
                        dtype=torch.long,
                        device=self.device,
                    )
                    pressure_loss = self.aux_criterion(outputs["pressure_logits"], pressure_targets)
                if self.cfg.maturity_loss_weight > 0.0:
                    if "maturity_logits" not in outputs:
                        raise KeyError("maturity_logits not produced by model")
                    if any(meta.maturity_target is None for meta in metas):
                        raise ValueError("maturity_target metadata is required when maturity_loss_weight > 0")
                    maturity_targets = torch.tensor(
                        [int(meta.maturity_target) for meta in metas],
                        dtype=torch.long,
                        device=self.device,
                    )
                    maturity_loss = self.aux_criterion(outputs["maturity_logits"], maturity_targets)
                if self.cfg.long_aux_loss_weight > 0.0:
                    if "long_aux_logits" not in outputs:
                        raise KeyError("long_aux_logits not produced by model")
                    long_aux_loss = self.aux_criterion(outputs["long_aux_logits"], y)
                loss = (
                    ce_weight * ce_loss
                    + supcon_weight * supcon_loss
                    + self.cfg.pressure_loss_weight * pressure_loss
                    + self.cfg.maturity_loss_weight * maturity_loss
                    + self.cfg.long_aux_loss_weight * long_aux_loss
                )

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * short_x.size(0)
                total_ce_loss += ce_loss.item() * short_x.size(0)
                total_supcon_loss += supcon_loss.item() * short_x.size(0)
                total_pressure_loss += pressure_loss.item() * short_x.size(0)
                total_maturity_loss += maturity_loss.item() * short_x.size(0)
                total_long_aux_loss += long_aux_loss.item() * short_x.size(0)
                preds.append(outputs["logits"].argmax(dim=1).detach().cpu().numpy())
                labels.append(y.detach().cpu().numpy())

        y_true = np.concatenate(labels)
        y_pred = np.concatenate(preds)
        macro_f1, per_class = _macro_f1(y_true, y_pred)
        dataset_len = len(loader.dataset)  # type: ignore[arg-type]
        return (
            total_loss / dataset_len,
            total_ce_loss / dataset_len,
            total_supcon_loss / dataset_len,
            total_pressure_loss / dataset_len,
            total_maturity_loss / dataset_len,
            total_long_aux_loss / dataset_len,
            total_supcon_valid_anchors / total_supcon_anchors if total_supcon_anchors else 0.0,
            total_supcon_positive_counts / total_supcon_anchors if total_supcon_anchors else 0.0,
            macro_f1,
            per_class,
        )

    def _ce_weight_for_epoch(self, epoch: int) -> float:
        if (
            self.cfg.ce_warmup_weight is not None
            and self.cfg.ce_warmup_epochs > 0
            and epoch <= self.cfg.ce_warmup_epochs
        ):
            return float(self.cfg.ce_warmup_weight)
        return 1.0

    def _supcon_weight_for_epoch(self, epoch: int) -> float:
        if (
            self.cfg.use_supcon
            and self.cfg.supcon_frontload_weight is not None
            and self.cfg.supcon_frontload_epochs > 0
            and epoch <= self.cfg.supcon_frontload_epochs
        ):
            return float(self.cfg.supcon_frontload_weight)
        return float(self.cfg.supcon_weight)

    def _supcon_positive_counts(
        self,
        labels: torch.Tensor,
        symbol_ids: torch.Tensor | None,
        positive_mode: str,
    ) -> torch.Tensor:
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T)
        logits_mask = ~torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
        if positive_mode in {"label_diff_symbol", "label_diff_symbol_neutral_same_symbol"}:
            if symbol_ids is None:
                raise ValueError("symbol_ids required for symbol-aware positive counts")
            symbol_ids = symbol_ids.view(-1, 1)
            same_symbol_mask = torch.eq(symbol_ids, symbol_ids.T)
            positive_mask = positive_mask & (~same_symbol_mask)
            if positive_mode == "label_diff_symbol_neutral_same_symbol":
                neutral_mask = torch.eq(labels, labels.T) & same_symbol_mask
                logits_mask = logits_mask & (~neutral_mask)
        return (positive_mask & logits_mask).sum(dim=1).to(torch.float32)

    def _save_checkpoint(self, name: str) -> None:
        path = self.cfg.checkpoint_dir / name
        torch.save({"model_state_dict": self.model.state_dict()}, path)

    def _save_history(self, history: dict[str, list[float]]) -> None:
        path = self.cfg.checkpoint_dir / "history.json"
        path.write_text(json.dumps(history, indent=2))

    def save_config(self) -> None:
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.checkpoint_dir / "train_config.json"
        path.write_text(json.dumps(asdict(self.cfg), indent=2, default=str))
