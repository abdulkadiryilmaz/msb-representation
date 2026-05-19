"""Small MLP predictor for Stage 1B forward-structure forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn


class Stage1BForwardPredictor(nn.Module):
    """Two-head predictor: break occurrence + direction given break."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        num_horizons: int = 1,
    ) -> None:
        super().__init__()
        self.num_horizons = num_horizons
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.break_heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(num_horizons)])
        self.direction_heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(num_horizons)])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        return {
            "break_logits": torch.stack([head(h) for head in self.break_heads], dim=1),
            "direction_logits": torch.stack([head(h) for head in self.direction_heads], dim=1),
        }
