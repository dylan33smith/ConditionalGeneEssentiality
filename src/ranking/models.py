"""Ranking model architectures.

Extracted verbatim (behavior-preserving) from the legacy T-tier so the ranking
package is self-contained:
  * ResidualBlock     — was src/experiments/tier3/_t3_common.py
  * AdapterResidualMLP — was src/experiments/tier5/_t5_common.py (T5-A locked head)

The locked ranking model is AdapterResidualMLP: a learnable adapter over the
frozen ProteomeLM gene embedding, concatenated with the condition chemistry
vector, then a residual-MLP head. New architectures register here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Linear→ReLU→Dropout block with a residual skip connection."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AdapterResidualMLP(nn.Module):
    """T5-A locked head, with a configurable learnable gene-side adapter.

    The adapter is an MLP that processes the frozen gene embedding before
    concatenation with the chemistry vector. It lets the model learn a
    task-specific transformation of the embedding without modifying
    ProteomeLM itself.

    Args:
        adapter_hidden: hidden dim of the adapter, or None to disable
            (passes gene_emb through unchanged for the baseline arm).
        adapter_out: final adapter output dim. Defaults to gene_dim.
        adapter_n_hidden_layers: number of hidden Linear→ReLU→Dropout
            blocks in the adapter. 1 = single hidden layer (T5-A default).
        adapter_layernorm: if True, prepend LayerNorm to the adapter
            (helps with per-organism distribution drift in frozen embeddings).
    """

    def __init__(self, *, gene_dim: int, chem_dim: int,
                 hidden_dim: int = 512, n_blocks: int = 1,
                 dropout: float = 0.1,
                 adapter_hidden: int | None = None,
                 adapter_out: int | None = None,
                 adapter_n_hidden_layers: int = 1,
                 adapter_layernorm: bool = False) -> None:
        super().__init__()
        if adapter_hidden is None:
            self.adapter = nn.Identity()
            effective_gene_dim = gene_dim
        else:
            out_dim = adapter_out if adapter_out is not None else gene_dim
            layers: list[nn.Module] = []
            if adapter_layernorm:
                layers.append(nn.LayerNorm(gene_dim))
            in_dim = gene_dim
            for _ in range(adapter_n_hidden_layers):
                layers.extend([
                    nn.Linear(in_dim, adapter_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                in_dim = adapter_hidden
            layers.append(nn.Linear(in_dim, out_dim))
            self.adapter = nn.Sequential(*layers)
            effective_gene_dim = out_dim

        self.proj = nn.Sequential(
            nn.Linear(effective_gene_dim + chem_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, gene_emb: torch.Tensor, chem: torch.Tensor) -> torch.Tensor:
        g = self.adapter(gene_emb)
        x = torch.cat([g, chem.float()], dim=1)
        x = self.proj(x)
        x = self.blocks(x)
        return self.out(x).squeeze(1)
