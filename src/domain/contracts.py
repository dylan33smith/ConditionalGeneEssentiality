"""Data and preprocessing contracts (interfaces).

Stub — fill in during Stage 0 / Stage 1.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class ConditionEncoder(Protocol):
    """Interface for condition feature encoders."""

    def fit(self, media_names: list[str]) -> "ConditionEncoder":
        """Fit encoder on train-only media names; return self."""
        ...

    def transform(self, media_names: list[str]) -> np.ndarray:
        """Transform media names → feature matrix (n_samples, n_features)."""
        ...

    def unknown_rate(self, media_names: list[str]) -> float:
        """Fraction of media_names that map to UNK."""
        ...
