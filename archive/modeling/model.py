"""Baseline MLP: frozen gene vector + trainable categorical + numeric condition features."""

from __future__ import annotations

import torch
import torch.nn as nn


class GeneConditionMLP(nn.Module):
    def __init__(
        self,
        gene_dim: int,
        cat_field_max_ids: dict[str, int],
        cat_field_order: tuple[str, ...],
        cat_emb_dim: int,
        n_cont: int,
        hidden_dim: int,
        num_hidden: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cat_field_order = cat_field_order
        self.cat_embs = nn.ModuleDict()
        for name in cat_field_order:
            m = int(cat_field_max_ids[name])
            # Parquet stores 1..m; index 0 reserved (unused).
            self.cat_embs[name] = nn.Embedding(m + 1, cat_emb_dim, padding_idx=0)
        in_dim = gene_dim + len(cat_field_order) * cat_emb_dim + n_cont
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x_gene: torch.Tensor,
        cat_ids: torch.Tensor,
        x_cont: torch.Tensor,
    ) -> torch.Tensor:
        """cat_ids: [B, K] with K = len(cat_field_order), columns aligned to cat_field_order."""
        parts: list[torch.Tensor] = [x_gene]
        for j, name in enumerate(self.cat_field_order):
            parts.append(self.cat_embs[name](cat_ids[:, j]))
        parts.append(x_cont)
        x = torch.cat(parts, dim=-1)
        return self.mlp(x).squeeze(-1)
