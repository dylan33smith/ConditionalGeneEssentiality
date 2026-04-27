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
