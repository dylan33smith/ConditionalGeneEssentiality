"""Train-only numeric scaler fitting for condition features."""
from __future__ import annotations

from typing import Any

import numpy as np


def fit_log1p_scaler(train_amounts) -> dict[str, Any]:
    """Fit default log1p scaler on train-only Amount values."""
    arr = np.asarray(train_amounts, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "transform": "log1p",
            "mean": 0.0,
            "std": 1.0,
            "n_train": int(arr.size),
            "n_finite": 0,
        }
    # Amount should be non-negative; clip negatives to 0 for numerical safety.
    clipped = np.clip(finite, 0.0, None)
    transformed = np.log1p(clipped)
    mean = float(np.mean(transformed))
    std = float(np.std(transformed))
    if std == 0.0:
        std = 1.0
    return {
        "transform": "log1p",
        "mean": mean,
        "std": std,
        "n_train": int(arr.size),
        "n_finite": int(finite.size),
    }


def record_bounded_reference_stats(train_amounts) -> dict[str, Any]:
    """Store p1/p99 reference stats for bounded transform declaration."""
    arr = np.asarray(train_amounts, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"transform": "bounded", "p1": 0.0, "p99": 1.0, "n_train": int(arr.size)}
    p1 = float(np.percentile(finite, 1))
    p99 = float(np.percentile(finite, 99))
    if p99 <= p1:
        p99 = p1 + 1.0
    return {"transform": "bounded", "p1": p1, "p99": p99, "n_train": int(arr.size)}


def fit_standard_scaler(train_values) -> dict[str, Any]:
    """Fit mean/std on finite train values only (z-score). Degenerate std -> 1.0."""
    arr = np.asarray(train_values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"transform": "standard", "mean": 0.0, "std": 1.0, "n_train": int(arr.size), "n_finite": 0}
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    if std == 0.0:
        std = 1.0
    return {
        "transform": "standard",
        "mean": mean,
        "std": std,
        "n_train": int(arr.size),
        "n_finite": int(finite.size),
    }
