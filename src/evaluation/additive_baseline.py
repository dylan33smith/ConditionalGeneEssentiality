"""Additive baseline: fit_hat = a + alpha[gene] + beta[condition].

Required by H-BASE-01. A model that does not beat this baseline is not learning
gene×condition interactions, so it is ineligible for tier promotion.

The fit is a simple no-interaction additive model. Solving by least squares with
a stable iterative approach (alternating means with shrinkage to handle sparsity
and prevent ill-conditioning when a gene or condition has only one observation).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class AdditiveBaselineFit:
    """Fitted additive baseline parameters."""
    intercept: float
    gene_effect: dict[str, float]
    condition_effect: dict[str, float]
    n_iters: int
    converged: bool

    def predict(
        self,
        gene_keys: np.ndarray,
        condition_keys: np.ndarray,
    ) -> np.ndarray:
        """Predict fit for arrays of gene_keys and condition_keys."""
        n = len(gene_keys)
        out = np.full(n, self.intercept, dtype=np.float64)
        for i in range(n):
            out[i] += self.gene_effect.get(gene_keys[i], 0.0)
            out[i] += self.condition_effect.get(condition_keys[i], 0.0)
        return out


def fit_additive_baseline(
    fit: np.ndarray,
    gene_keys: np.ndarray,
    condition_keys: np.ndarray,
    *,
    max_iters: int = 100,
    tol: float = 1e-6,
    shrinkage: float = 0.0,
) -> AdditiveBaselineFit:
    """Fit additive baseline by alternating means.

    Args:
        fit: target values (n_rows,)
        gene_keys: gene identifier per row (n_rows,)
        condition_keys: condition identifier per row (n_rows,)
        max_iters: max alternating-means iterations
        tol: convergence tolerance on mean absolute change in residuals
        shrinkage: optional ridge-like shrinkage toward 0 for under-supported groups

    Returns:
        AdditiveBaselineFit with intercept, gene_effect, condition_effect.

    Notes:
        - Uses train rows only. Caller is responsible for not leaking val/test.
        - Predictions for unseen gene_keys or condition_keys default to 0 (no effect),
          which is the correct cold-start fallback for an additive baseline.
    """
    if not (len(fit) == len(gene_keys) == len(condition_keys)):
        raise ValueError("fit, gene_keys, condition_keys must have equal length")

    intercept = float(np.mean(fit))
    residuals = fit - intercept

    unique_genes = np.unique(gene_keys)
    unique_conds = np.unique(condition_keys)
    gene_eff = {g: 0.0 for g in unique_genes}
    cond_eff = {c: 0.0 for c in unique_conds}

    converged = False
    for it in range(max_iters):
        # Update gene effects: alpha[g] = mean(fit - intercept - beta[c]) over rows for g
        gene_residuals: dict = {}
        for i, g in enumerate(gene_keys):
            r = fit[i] - intercept - cond_eff[condition_keys[i]]
            gene_residuals.setdefault(g, []).append(r)
        new_gene_eff = {
            g: float(np.mean(rs)) * (1.0 - shrinkage) for g, rs in gene_residuals.items()
        }

        # Update condition effects: beta[c] = mean(fit - intercept - alpha[g]) over rows for c
        cond_residuals: dict = {}
        for i, c in enumerate(condition_keys):
            r = fit[i] - intercept - new_gene_eff[gene_keys[i]]
            cond_residuals.setdefault(c, []).append(r)
        new_cond_eff = {
            c: float(np.mean(rs)) * (1.0 - shrinkage) for c, rs in cond_residuals.items()
        }

        # Check convergence
        max_change = max(
            max(abs(new_gene_eff[g] - gene_eff[g]) for g in gene_eff),
            max(abs(new_cond_eff[c] - cond_eff[c]) for c in cond_eff),
        )
        gene_eff, cond_eff = new_gene_eff, new_cond_eff
        if max_change < tol:
            converged = True
            break

    return AdditiveBaselineFit(
        intercept=intercept,
        gene_effect=gene_eff,
        condition_effect=cond_eff,
        n_iters=it + 1,
        converged=converged,
    )


def additive_baseline_metrics(
    fit_pred: np.ndarray,
    fit_true: np.ndarray,
) -> dict:
    """RMSE / MAE for the additive baseline."""
    rmse = float(np.sqrt(np.mean((fit_true - fit_pred) ** 2)))
    mae = float(np.mean(np.abs(fit_true - fit_pred)))
    return {"rmse": rmse, "mae": mae, "n_rows": len(fit_true)}
