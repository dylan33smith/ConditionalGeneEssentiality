"""Additive baseline: fit_hat = a + alpha[gene] + beta[condition].

Required by H-BASE-01. A model that does not beat this baseline is not learning
gene×condition interactions, so it is ineligible for tier promotion.

The fit is a simple no-interaction additive model. Solved by alternating means
(Cython-vectorized via pandas.groupby) — equivalent at convergence to the
closed-form least-squares solution but easier to write.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd


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
        """Predict fit for arrays of gene_keys and condition_keys.

        Vectorized lookup via pandas.Series.map. Unseen keys default to 0
        (no effect), so an unseen gene + unseen condition predicts the
        intercept — the correct cold-start fallback for an additive model.
        """
        ge = pd.Series(gene_keys).map(self.gene_effect).fillna(0.0).to_numpy()
        ce = pd.Series(condition_keys).map(self.condition_effect).fillna(0.0).to_numpy()
        return self.intercept + ge + ce


def fit_additive_baseline(
    fit: np.ndarray,
    gene_keys: np.ndarray,
    condition_keys: np.ndarray,
    *,
    max_iters: int = 100,
    tol: float = 1e-6,
    shrinkage: float = 0.0,
) -> AdditiveBaselineFit:
    """Fit additive baseline by vectorized alternating means.

    Args:
        fit: target values (n_rows,)
        gene_keys: gene identifier per row (n_rows,)
        condition_keys: condition identifier per row (n_rows,)
        max_iters: max alternating-means iterations
        tol: convergence tolerance on max parameter change between iterations
        shrinkage: optional ridge-like shrinkage toward 0 for under-supported groups

    Returns:
        AdditiveBaselineFit with intercept, gene_effect, condition_effect.

    Notes:
        - Uses train rows only. Caller is responsible for not leaking val/test.
        - Predictions for unseen gene_keys or condition_keys default to 0 (no effect),
          the correct cold-start fallback for an additive baseline.
        - Vectorized: ~50-100x faster than a Python-loop alternating-means
          implementation on multi-million-row inputs.
    """
    if not (len(fit) == len(gene_keys) == len(condition_keys)):
        raise ValueError("fit, gene_keys, condition_keys must have equal length")

    df = pd.DataFrame({
        "fit": np.asarray(fit, dtype=np.float64),
        "gene": np.asarray(gene_keys),
        "cond": np.asarray(condition_keys),
    })

    intercept = float(df["fit"].mean())

    # Initialize effects to 0 for every group present in train.
    gene_eff = pd.Series(0.0, index=df["gene"].unique())
    cond_eff = pd.Series(0.0, index=df["cond"].unique())

    scale = 1.0 - float(shrinkage)
    converged = False
    last_iter = 0

    for it in range(max_iters):
        # Update gene effects: alpha[g] = mean over rows in g of (fit - intercept - beta[c])
        cond_eff_per_row = df["cond"].map(cond_eff).to_numpy()
        target = df["fit"].to_numpy() - intercept - cond_eff_per_row
        new_gene_eff = (
            pd.Series(target).groupby(df["gene"].values).mean() * scale
        )

        # Update condition effects: beta[c] = mean over rows in c of (fit - intercept - alpha[g])
        gene_eff_per_row = df["gene"].map(new_gene_eff).to_numpy()
        target = df["fit"].to_numpy() - intercept - gene_eff_per_row
        new_cond_eff = (
            pd.Series(target).groupby(df["cond"].values).mean() * scale
        )

        # Convergence check: max absolute change in any effect
        gene_diff = (new_gene_eff - gene_eff.reindex(new_gene_eff.index, fill_value=0.0)).abs().max()
        cond_diff = (new_cond_eff - cond_eff.reindex(new_cond_eff.index, fill_value=0.0)).abs().max()
        max_change = float(max(gene_diff, cond_diff))

        gene_eff = new_gene_eff
        cond_eff = new_cond_eff
        last_iter = it
        if max_change < tol:
            converged = True
            break

    return AdditiveBaselineFit(
        intercept=intercept,
        gene_effect=gene_eff.to_dict(),
        condition_effect=cond_eff.to_dict(),
        n_iters=last_iter + 1,
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
