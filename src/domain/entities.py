"""Core domain entities for conditional gene essentiality prediction.

Stub — fill in during Stage 0 / Stage 1.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GeneKey:
    """Unique gene identifier: (org_id, locus_id)."""
    org_id: str
    locus_id: str


@dataclass(frozen=True)
class ConditionKey:
    """Unique condition identifier: experiment name within an organism context."""
    exp_name: str
    org_id: str


@dataclass(frozen=True)
class FitnessMeasurement:
    """A single (gene, condition) fitness observation."""
    gene_key: GeneKey
    condition_key: ConditionKey
    fit: float
    t_stat: Optional[float] = None
    cor12: Optional[float] = None
