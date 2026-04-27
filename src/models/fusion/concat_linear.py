"""Concat-then-linear fusion baseline (Tier-2 MVP).

Stub — implement during Tier 2.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ConcatLinear(nn.Module):
    """Fuse gene embedding and condition vector by concatenation → linear head."""

    def __init__(self, gene_dim: int, cond_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(gene_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gene_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([gene_emb, cond_emb], dim=-1)
        return self.head(x).squeeze(-1)
