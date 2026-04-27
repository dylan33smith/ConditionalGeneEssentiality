"""Fit concentration scalers on train-only data and persist artifact.

Stub — implement during Stage 1.
"""
from __future__ import annotations
from pathlib import Path


def fit_concentration_scaler(
    train_amounts,          # array-like
    transform: str,         # "raw" | "log1p" | "bounded"
    output_path: Path,
):
    """Fit and save scaler artifact. Must not touch val/test amounts."""
    raise NotImplementedError("Implement during Stage 1")
