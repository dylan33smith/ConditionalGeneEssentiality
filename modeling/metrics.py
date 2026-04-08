"""Val RMSE and mean within-gene Spearman (plan §7.5)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


def rmse_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    d = y_true - y_pred
    if not np.all(np.isfinite(d)):
        m = np.isfinite(d)
        if not np.any(m):
            return float("nan")
        d = d[m]
    return float(np.sqrt(np.mean(d * d)))


def mean_within_gene_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gene_keys: np.ndarray,
    min_conditions: int = 2,
) -> tuple[float, int]:
    rho, n, _diag = mean_within_gene_spearman_with_diagnostics(
        y_true, y_pred, gene_keys, min_conditions=min_conditions
    )
    return rho, n


def mean_within_gene_spearman_with_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gene_keys: np.ndarray,
    min_conditions: int = 2,
) -> tuple[float, int, dict[str, int]]:
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for i in range(len(y_true)):
        buckets[str(gene_keys[i])].append((float(y_true[i]), float(y_pred[i])))

    rhos: list[float] = []
    n_lt = 0
    n_const = 0
    n_nan = 0
    for pairs in buckets.values():
        if len(pairs) < min_conditions:
            n_lt += 1
            continue
        a = np.array([p[0] for p in pairs], dtype=np.float64)
        b = np.array([p[1] for p in pairs], dtype=np.float64)
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):
            n_const += 1
            continue
        r, _ = spearmanr(a, b)
        if np.isnan(r):
            n_nan += 1
            continue
        rhos.append(float(r))
    n_genes_total = len(buckets)
    n_used = len(rhos)
    diag = {
        "n_genes_total_in_val": n_genes_total,
        "n_genes_lt_min_conditions": n_lt,
        "n_genes_constant_true_or_pred": n_const,
        "n_genes_nan_spearman": n_nan,
        "n_genes_used_for_spearman": n_used,
    }
    if not rhos:
        return float("nan"), 0, diag
    return float(np.mean(rhos)), n_used, diag
