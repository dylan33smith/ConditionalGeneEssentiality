"""Null baseline suite (REFACTORPLAN §7 S2).

All five baselines per split protocol:
  1. global_train_mean
  2. per_condition_mean (key = expName, fallback = global)
  3. per_organism_mean (key = orgId, fallback = global)
  4. additive_baseline (in src/evaluation/additive_baseline.py)
  5. embedding_nn_baseline (in src/evaluation/nn_baseline.py)

Each baseline returns a dict with at minimum:
  rmse, mae, n_rows, predictions (np.ndarray of length n_val)

Predictions are returned so downstream code can compute Spearman + bootstrap
on the same row set without recomputing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"rmse": rmse, "mae": mae}


def global_train_mean_baseline(
    train_fit: np.ndarray,
    val_fit: np.ndarray,
) -> dict:
    """Predict global train mean for every val row."""
    train_mean = float(np.mean(train_fit))
    pred = np.full(len(val_fit), train_mean, dtype=np.float64)
    return {
        **_metrics(val_fit, pred),
        "n_rows": int(len(val_fit)),
        "predictions": pred,
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "train_mean": train_mean,
    }


def per_condition_mean_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    condition_col: str = "expName",
    fit_col: str = "fit",
) -> dict:
    """Predict the train mean of each val row's condition.

    Cold-start fallback: val rows whose condition is not in train get the
    global train mean. fallback_rate is logged.
    """
    cond_mean = train_df.groupby(condition_col)[fit_col].mean()
    global_mean = float(train_df[fit_col].mean())
    val_cond = val_df[condition_col].to_numpy()
    pred = pd.Series(val_cond).map(cond_mean).fillna(global_mean).to_numpy()
    fallback_mask = ~pd.Series(val_cond).isin(cond_mean.index).to_numpy()
    return {
        **_metrics(val_df[fit_col].to_numpy(), pred),
        "n_rows": int(len(val_df)),
        "predictions": pred,
        "fallback_count": int(fallback_mask.sum()),
        "fallback_rate": float(fallback_mask.mean()),
    }


def per_organism_mean_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    org_col: str = "orgId",
    fit_col: str = "fit",
) -> dict:
    """Predict the train mean of each val row's organism.

    Critical caveat: under organism-holdout, val orgs are by definition NOT
    in train. So fallback_rate will be ~100%; this baseline collapses to
    global_train_mean. Documented per REFACTORPLAN §7 S2.
    """
    org_mean = train_df.groupby(org_col)[fit_col].mean()
    global_mean = float(train_df[fit_col].mean())
    val_org = val_df[org_col].to_numpy()
    pred = pd.Series(val_org).map(org_mean).fillna(global_mean).to_numpy()
    fallback_mask = ~pd.Series(val_org).isin(org_mean.index).to_numpy()
    return {
        **_metrics(val_df[fit_col].to_numpy(), pred),
        "n_rows": int(len(val_df)),
        "predictions": pred,
        "fallback_count": int(fallback_mask.sum()),
        "fallback_rate": float(fallback_mask.mean()),
    }
