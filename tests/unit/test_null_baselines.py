"""Unit tests for null baselines (S2 locked baselines).

These tests guard code correctness, not numerical results. The actual S2
baseline numbers are stored in artifacts/baselines/baselines_per_protocol.json.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.null_baselines import (
    global_train_mean_baseline,
    per_condition_mean_baseline,
    per_organism_mean_baseline,
)


# -----------------------------
# global_train_mean
# -----------------------------

def test_global_train_mean_returns_train_mean_for_every_val_row():
    train = np.array([0.0, 1.0, 2.0, 3.0])      # mean = 1.5
    val = np.array([1.5, 1.5, 1.5])
    res = global_train_mean_baseline(train, val)
    assert res["train_mean"] == pytest.approx(1.5)
    assert np.allclose(res["predictions"], 1.5)
    assert res["rmse"] == pytest.approx(0.0)    # val happens to equal train mean
    assert res["mae"] == pytest.approx(0.0)
    assert res["fallback_rate"] == 0.0
    assert res["fallback_count"] == 0
    assert res["n_rows"] == 3


def test_global_train_mean_rmse_known_case():
    train = np.array([1.0, 1.0, 1.0])           # mean = 1.0
    val = np.array([0.0, 2.0])                  # |val - 1| = [1, 1]
    res = global_train_mean_baseline(train, val)
    assert res["rmse"] == pytest.approx(1.0)
    assert res["mae"] == pytest.approx(1.0)


# -----------------------------
# per_condition_mean
# -----------------------------

def test_per_condition_mean_uses_per_condition_average_when_seen():
    train = pd.DataFrame({
        "expName": ["A", "A", "B", "B"],
        "fit":     [1.0, 3.0, 5.0, 7.0],
    })
    # cond A mean = 2.0; cond B mean = 6.0; global mean = 4.0
    val = pd.DataFrame({
        "expName": ["A", "B"],
        "fit":     [2.0, 6.0],
    })
    res = per_condition_mean_baseline(train, val)
    assert np.allclose(res["predictions"], [2.0, 6.0])
    assert res["rmse"] == pytest.approx(0.0)
    assert res["fallback_rate"] == 0.0


def test_per_condition_mean_falls_back_to_global_for_unseen_condition():
    train = pd.DataFrame({"expName": ["A", "A"], "fit": [1.0, 3.0]})
    val = pd.DataFrame({"expName": ["A", "C"], "fit": [2.0, 0.0]})
    res = per_condition_mean_baseline(train, val)
    # Cond A mean = 2.0 (seen). Cond C unseen → global mean = 2.0.
    assert np.allclose(res["predictions"], [2.0, 2.0])
    assert res["fallback_count"] == 1
    assert res["fallback_rate"] == pytest.approx(0.5)


def test_per_condition_mean_organism_holdout_collapses_to_global():
    """The S2 finding: under organism-holdout, all val expNames are unseen,
    so the baseline reduces to the global train mean."""
    train = pd.DataFrame({"expName": ["A", "B", "C"], "fit": [1.0, 2.0, 3.0]})
    # val expNames are all unique (= held-out organism's conditions)
    val = pd.DataFrame({"expName": ["X", "Y", "Z"], "fit": [10.0, 10.0, 10.0]})
    res = per_condition_mean_baseline(train, val)
    assert res["fallback_rate"] == pytest.approx(1.0)
    # All predictions should equal the global train mean (= 2.0)
    assert np.allclose(res["predictions"], 2.0)


# -----------------------------
# per_organism_mean
# -----------------------------

def test_per_organism_mean_uses_per_org_average_when_seen():
    train = pd.DataFrame({
        "orgId": ["o1", "o1", "o2", "o2"],
        "fit":   [0.0, 2.0, 10.0, 12.0],
    })
    # o1 mean = 1.0; o2 mean = 11.0
    val = pd.DataFrame({"orgId": ["o1", "o2"], "fit": [1.0, 11.0]})
    res = per_organism_mean_baseline(train, val)
    assert np.allclose(res["predictions"], [1.0, 11.0])
    assert res["rmse"] == pytest.approx(0.0)


def test_per_organism_mean_organism_holdout_falls_back_to_global():
    """By construction, val orgs are NOT in train under organism-holdout.
    Per-organism mean must fall back to global for every val row."""
    train = pd.DataFrame({"orgId": ["o1", "o2"], "fit": [0.0, 2.0]})
    val = pd.DataFrame({"orgId": ["o3", "o3", "o3"], "fit": [5.0, 5.0, 5.0]})
    res = per_organism_mean_baseline(train, val)
    assert res["fallback_rate"] == pytest.approx(1.0)
    # All predictions = global train mean = 1.0
    assert np.allclose(res["predictions"], 1.0)
