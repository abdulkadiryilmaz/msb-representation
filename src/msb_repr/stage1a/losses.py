"""Loss helpers for Stage 1A objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss over a batch of projected embeddings."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        symbol_ids: torch.Tensor | None = None,
        positive_mode: str = "label",
    ) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape (batch, dim)")
        if labels.ndim != 1:
            raise ValueError("labels must have shape (batch,)")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels batch size must match")
        if positive_mode not in {"label", "label_diff_symbol"}:
            raise ValueError(f"Unknown positive_mode: {positive_mode}")
        if positive_mode == "label_diff_symbol":
            if symbol_ids is None:
                raise ValueError("symbol_ids required when positive_mode='label_diff_symbol'")
            if symbol_ids.ndim != 1 or symbol_ids.shape[0] != labels.shape[0]:
                raise ValueError("symbol_ids must have shape (batch,)")

        batch_size = features.shape[0]
        if batch_size < 2:
            return features.new_zeros(())

        features = F.normalize(features, dim=1)
        logits = features @ features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).to(features.dtype)
        if positive_mode == "label_diff_symbol":
            symbol_ids = symbol_ids.view(-1, 1)
            same_symbol_mask = torch.eq(symbol_ids, symbol_ids.T).to(features.dtype)
            positive_mask = positive_mask * (1.0 - same_symbol_mask)
        logits_mask = torch.ones_like(positive_mask) - torch.eye(batch_size, device=features.device, dtype=features.dtype)
        positive_mask = positive_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        positive_counts = positive_mask.sum(dim=1)
        valid = positive_counts > 0
        if not torch.any(valid):
            return features.new_zeros(())

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1.0)
        loss = -mean_log_prob_pos[valid].mean()
        return loss
