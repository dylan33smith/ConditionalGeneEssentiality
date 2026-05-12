"""Fixed shallow concat-linear MLP for S5 policy comparisons."""
from __future__ import annotations

import torch
from torch import nn


class ConcatLinearMLP(nn.Module):
    """Predict fit from concatenated gene embedding + chemistry multihot."""

    def __init__(
        self,
        *,
        gene_dim: int,
        chemistry_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = int(gene_dim) + int(chemistry_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, gene_emb: torch.Tensor, chem_multihot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([gene_emb, chem_multihot], dim=1)
        return self.net(x).squeeze(1)

