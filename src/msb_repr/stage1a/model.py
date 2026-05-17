"""Temporal CNN baseline model for Stage 1A."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalResidualBlock(nn.Module):
    """Residual 1D conv block with optional dilation."""

    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x + self.block(x))


class TemporalConvEncoder(nn.Module):
    """Global-vector encoder for fixed-length temporal windows."""

    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )
        dilations = [1, 2, 4, 8][:num_blocks]
        self.blocks = nn.Sequential(
            *[TemporalResidualBlock(hidden_channels, dilation=d, dropout=dropout) for d in dilations]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_channels, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stage1AModel(nn.Module):
    """Short encoder + long encoder + concat fusion baseline."""

    def __init__(
        self,
        short_input_channels: int,
        long_input_channels: int,
        z_short: int = 64,
        z_long: int = 32,
        num_classes: int = 3,
        num_pressure_classes: int | None = None,
        num_maturity_classes: int | None = None,
        num_long_aux_classes: int | None = None,
        pressure_head_input: str = "z_fused",
        projection_dim: int = 64,
        long_projection_dim: int | None = None,
        hidden_channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if pressure_head_input not in {"z_fused", "z_short"}:
            raise ValueError(f"Unsupported pressure_head_input: {pressure_head_input}")
        self.pressure_head_input = pressure_head_input
        self.short_encoder = TemporalConvEncoder(
            input_channels=short_input_channels,
            latent_dim=z_short,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
        self.long_encoder = TemporalConvEncoder(
            input_channels=long_input_channels,
            latent_dim=z_long,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
        fused_dim = z_short + z_long
        pressure_dim = z_short if pressure_head_input == "z_short" else fused_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_classes),
        )
        self.pressure_classifier = (
            nn.Sequential(
                nn.Linear(pressure_dim, pressure_dim),
                nn.LayerNorm(pressure_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(pressure_dim, num_pressure_classes),
            )
            if num_pressure_classes is not None
            else None
        )
        self.maturity_classifier = (
            nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim, num_maturity_classes),
            )
            if num_maturity_classes is not None
            else None
        )
        self.long_aux_classifier = (
            nn.Sequential(
                nn.Linear(z_long, z_long),
                nn.LayerNorm(z_long),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(z_long, num_long_aux_classes),
            )
            if num_long_aux_classes is not None
            else None
        )
        self.projection_head = ProjectionHead(fused_dim, projection_dim=projection_dim, dropout=dropout)
        self.long_projection_head = (
            ProjectionHead(z_long, projection_dim=long_projection_dim, dropout=dropout)
            if long_projection_dim is not None
            else None
        )

    def forward(self, short_x: torch.Tensor, long_x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_short = self.short_encoder(short_x)
        z_long = self.long_encoder(long_x)
        z_fused = torch.cat([z_short, z_long], dim=1)
        logits = self.classifier(z_fused)
        z_proj = self.projection_head(z_fused)
        outputs = {
            "z_short": z_short,
            "z_long": z_long,
            "z_fused": z_fused,
            "z_proj": z_proj,
            "logits": logits,
        }
        if self.long_projection_head is not None:
            outputs["z_long_proj"] = self.long_projection_head(z_long)
        if self.pressure_classifier is not None:
            pressure_input = z_short if self.pressure_head_input == "z_short" else z_fused
            outputs["pressure_logits"] = self.pressure_classifier(pressure_input)
        if self.maturity_classifier is not None:
            outputs["maturity_logits"] = self.maturity_classifier(z_fused)
        if self.long_aux_classifier is not None:
            outputs["long_aux_logits"] = self.long_aux_classifier(z_long)
        return outputs


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
