"""Condition encoder implementations.

Stub — implement during Tier 1 representation experiments.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class MultihotConditionEncoder(nn.Module):
    """Learnable embedding over multihot chemistry presence vectors.

    Experiment 1A baseline: component-level presence encoding.
    """

    def __init__(self, vocab_size: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, embed_dim, mode="sum", padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, component_ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embed(component_ids))


class MediaIDEncoder(nn.Module):
    """Categorical media-name encoding baseline.

    Experiment 1A baseline to compare against component-level encoding.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, media_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(media_ids)
