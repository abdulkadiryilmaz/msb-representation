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


class Stage1BEventSequencePredictor(nn.Module):
    """Predict event type and direction over a forward horizon."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        num_event_types: int = 4,
    ) -> None:
        super().__init__()
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
        self.event_type_head = nn.Linear(hidden_dim, num_event_types)
        self.event_direction_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        return {
            "event_type_logits": self.event_type_head(h),
            "event_direction_logits": self.event_direction_head(h),
        }


class Stage1BJointEventSequencePredictor(nn.Module):
    """Predict a joint event-direction class over a forward horizon."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        num_classes: int = 7,
    ) -> None:
        super().__init__()
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
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.classifier(self.encoder(x))}


class Stage1BMultiHeadEventSequencePredictor(nn.Module):
    """Predict first event, outcome event, and dominant forward direction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        num_joint_classes: int = 7,
        num_direction_classes: int = 3,
    ) -> None:
        super().__init__()
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
        self.first_event_head = nn.Linear(hidden_dim, num_joint_classes)
        self.outcome_head = nn.Linear(hidden_dim, num_joint_classes)
        self.dominant_direction_head = nn.Linear(hidden_dim, num_direction_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        return {
            "first_event_logits": self.first_event_head(h),
            "outcome_logits": self.outcome_head(h),
            "dominant_direction_logits": self.dominant_direction_head(h),
        }
