"""Split protocol contract.

Stub — fill in during Stage 2 split protocol lock.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SplitProtocol:
    """Immutable descriptor for a data split."""
    protocol_id: str
    split_type: str          # e.g. "organism_holdout"
    train_org_ids: tuple[str, ...]
    val_org_ids: tuple[str, ...]
    test_org_ids: tuple[str, ...]
    seed: int
    notes: str = ""
    chemistry_seen_rate_val: Optional[float] = None
    chemistry_seen_rate_test: Optional[float] = None
