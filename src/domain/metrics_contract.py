"""Metric contracts and typed result containers.

Stub — fill in during Stage 0.5 evaluation trustworthiness lock.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalResult:
    """Metrics for a single scored row-set."""
    split_protocol_id: str
    scored_rowset_hash: str
    n_rows: int
    rmse: float
    mae: float
    spearman_mean: Optional[float] = None
    spearman_n_genes: Optional[int] = None
    null_rmse_global_train_mean: Optional[float] = None
    unknown_category_rate: Optional[float] = None
    notes: str = ""
