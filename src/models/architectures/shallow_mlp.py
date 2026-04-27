"""Shallow regression MLP — Tier-1/Tier-2 MVP architecture.

Stub — implement during Tier 1.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ShallowMLP(nn.Module):
    """Single-hidden-layer MLP for fitness regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
