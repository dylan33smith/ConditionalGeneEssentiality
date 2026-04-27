"""Stub: verify null baselines reproduce to within tolerance across reruns."""
import pytest
import numpy as np
from src.evaluation.null_baselines import global_train_mean_baseline


def test_global_mean_baseline_deterministic():
    rng = np.random.default_rng(42)
    train_fit = rng.normal(0, 1, 1000).astype(np.float32)
    val_fit   = rng.normal(0, 1, 200).astype(np.float32)
    r1 = global_train_mean_baseline(train_fit, val_fit)
    r2 = global_train_mean_baseline(train_fit, val_fit)
    assert r1["rmse"] == r2["rmse"]
    assert r1["mae"]  == r2["mae"]
