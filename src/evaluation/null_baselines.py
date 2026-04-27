"""Null baseline suite (re-established from scratch per REFACTORPLAN).

Required baselines per split protocol:
  1. global train mean
  2. per-condition/per-experiment mean with safe fallback to global
  3. per-organism mean with safe fallback to global
  4. (future) embedding nearest-neighbour baseline

Stub — implement during Stage 0.5.
"""
from __future__ import annotations
import numpy as np


def global_train_mean_baseline(
    train_fit: np.ndarray,
    val_fit: np.ndarray,
) -> dict:
    """Predict global train mean for every val row."""
    pred = np.full(len(val_fit), train_fit.mean())
    rmse = float(np.sqrt(np.mean((val_fit - pred) ** 2)))
    mae = float(np.mean(np.abs(val_fit - pred)))
    return {"rmse": rmse, "mae": mae, "n_rows": len(val_fit)}
