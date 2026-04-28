"""Unit tests for the additive baseline (H-BASE-01)."""
import numpy as np
import pytest

from src.evaluation.additive_baseline import (
    AdditiveBaselineFit,
    additive_baseline_metrics,
    fit_additive_baseline,
)


def test_perfectly_additive_data_is_recovered():
    """If fit = a + alpha[g] + beta[c] exactly, baseline should recover it (RMSE ~ 0)."""
    rng = np.random.default_rng(0)
    genes = ["g1", "g2", "g3"]
    conds = ["c1", "c2", "c3", "c4"]
    a = 1.5
    alpha = {"g1": 0.2, "g2": -0.4, "g3": 0.1}
    beta  = {"c1": 0.0, "c2": 0.5, "c3": -0.3, "c4": 0.7}

    rows = []
    for g in genes:
        for c in conds:
            rows.append((g, c, a + alpha[g] + beta[c]))
    g_arr = np.array([r[0] for r in rows])
    c_arr = np.array([r[1] for r in rows])
    f_arr = np.array([r[2] for r in rows])

    fit = fit_additive_baseline(f_arr, g_arr, c_arr, max_iters=50, tol=1e-10)
    pred = fit.predict(g_arr, c_arr)
    rmse = np.sqrt(np.mean((f_arr - pred) ** 2))
    assert rmse == pytest.approx(0.0, abs=1e-6)


def test_unseen_gene_falls_back_to_intercept_only():
    rng = np.random.default_rng(1)
    f_arr = rng.normal(0, 1, 30)
    g_arr = np.array(["g1"] * 15 + ["g2"] * 15)
    c_arr = np.array(["c1"] * 10 + ["c2"] * 10 + ["c3"] * 10)
    fit = fit_additive_baseline(f_arr, g_arr, c_arr, max_iters=50)

    # Unseen gene + unseen condition → just the intercept
    pred = fit.predict(np.array(["g_unseen"]), np.array(["c_unseen"]))
    assert pred[0] == pytest.approx(fit.intercept)


def test_metrics_shape():
    f_true = np.array([1.0, 2.0, 3.0])
    f_pred = np.array([1.0, 2.0, 3.0])
    m = additive_baseline_metrics(f_pred, f_true)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["n_rows"] == 3


def test_vectorized_recovers_additive_at_moderate_scale():
    """Regression test: vectorized fit recovers truly-additive synthetic data
    at 100k rows / 500 genes / 200 conds with low noise."""
    rng = np.random.default_rng(42)
    n_genes, n_conds, n_rows = 500, 200, 100_000
    gene_idx = rng.integers(0, n_genes, n_rows)
    cond_idx = rng.integers(0, n_conds, n_rows)
    alpha = rng.normal(0, 0.5, n_genes)
    beta = rng.normal(0, 0.3, n_conds)
    fit = 0.5 + alpha[gene_idx] + beta[cond_idx] + rng.normal(0, 0.05, n_rows)
    g_keys = np.array([f"g{i}" for i in gene_idx])
    c_keys = np.array([f"c{i}" for i in cond_idx])

    res = fit_additive_baseline(fit, g_keys, c_keys, max_iters=20, tol=1e-4)
    assert res.converged, f"failed to converge in 20 iters (got {res.n_iters})"

    # Recovery vs noiseless ground truth on a held-out sample
    val_idx = rng.integers(0, n_genes, 5_000)
    val_cidx = rng.integers(0, n_conds, 5_000)
    truth = 0.5 + alpha[val_idx] + beta[val_cidx]
    pred = res.predict(np.array([f"g{i}" for i in val_idx]),
                       np.array([f"c{i}" for i in val_cidx]))
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    assert rmse < 0.05, f"recovery RMSE too high: {rmse}"


def test_vectorized_predict_handles_unseen_keys():
    """predict() must use 0-fill for unseen genes/conds (cold-start fallback)."""
    rng = np.random.default_rng(0)
    fit = rng.normal(0, 1, 200)
    g = np.array(["g1"] * 100 + ["g2"] * 100)
    c = np.array(["c1"] * 60 + ["c2"] * 80 + ["c3"] * 60)
    res = fit_additive_baseline(fit, g, c, max_iters=20)

    pred = res.predict(np.array(["g_unseen", "g1", "g2"]),
                       np.array(["c_unseen", "c1", "c_unseen"]))
    assert pred[0] == pytest.approx(res.intercept)                     # both unseen
    assert pred[1] == pytest.approx(
        res.intercept + res.gene_effect["g1"] + res.condition_effect["c1"]
    )
    assert pred[2] == pytest.approx(
        res.intercept + res.gene_effect["g2"]
    )
