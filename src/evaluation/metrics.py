"""Evaluation metrics (RMSE, MAE, within-gene Spearman).

All functions require denominator parity: model and baseline must be
evaluated on the exact same scored row set.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def within_gene_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gene_keys: np.ndarray,
    min_conditions: int = 5,
    min_variability: float | None = None,
) -> dict:
    """Compute mean within-gene Spearman across eligible genes.

    Args:
        min_conditions: minimum number of conditions for a gene to be eligible
        min_variability: minimum IQR of y_true to include gene (None = no filter)

    Returns dict with keys: mean_spearman, n_genes_eligible, n_genes_used.
    """
    unique_genes = np.unique(gene_keys)
    spearmans = []
    n_eligible = 0
    for g in unique_genes:
        mask = gene_keys == g
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) < min_conditions:
            continue
        if min_variability is not None:
            iqr = float(np.percentile(yt, 75) - np.percentile(yt, 25))
            if iqr < min_variability:
                continue
        n_eligible += 1
        r, _ = spearmanr(yt, yp)
        if not np.isnan(r):
            spearmans.append(r)
    return {
        "mean_spearman": float(np.mean(spearmans)) if spearmans else float("nan"),
        "n_genes_eligible": n_eligible,
        "n_genes_used": len(spearmans),
    }
