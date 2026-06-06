"""Unit tests for R-CONF confidence-stratification helpers."""
import numpy as np
import pandas as pd
import pytest

from src.experiments.rconf._rconf_common import (
    confidence_factor, assign_gene_strata, cell_filter_analysis, T_CAP, CONF_FLOOR)


def test_confidence_factor_ramp_and_clip():
    # 0 -> floor, cap -> 1, midpoint -> 0.5
    f = confidence_factor(np.array([0.0, T_CAP / 2, T_CAP, 2 * T_CAP]))
    assert f[0] == pytest.approx(CONF_FLOOR)      # clipped up from 0
    assert f[1] == pytest.approx(0.5)
    assert f[2] == pytest.approx(1.0)
    assert f[3] == pytest.approx(1.0)             # saturates at cap


def test_confidence_factor_nan_is_neutral():
    f = confidence_factor(np.array([np.nan]))
    assert f[0] == pytest.approx(0.5)


def test_confidence_factor_monotone_nondecreasing():
    a = np.linspace(0, 8, 50)
    f = confidence_factor(a)
    assert np.all(np.diff(f) >= -1e-9)


def _toy_eval_frame(n_genes=40, n_cond=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_genes):
        # gene confidence increases with g -> creates a clean quartile gradient
        base_t = 0.5 + g * 0.2
        for c in range(n_cond):
            fit = rng.normal(0, 1)
            rows.append({"gene_key": f"org:g{g}", "orgId": "org",
                         "condition_key": f"c{c}",
                         "fit": fit, "model_pred": fit + rng.normal(0, 1),
                         "knn_pred": fit + rng.normal(0, 0.5),
                         "null_pred": rng.normal(0, 1),
                         "abs_t": abs(rng.normal(base_t, 0.1))})
    return pd.DataFrame(rows)


def test_assign_gene_strata_quartiles():
    ev = _toy_eval_frame()
    strata = assign_gene_strata(ev, n_strata=4, min_n=5)
    cats = list(strata["stratum"].cat.categories)
    assert cats == ["Q1", "Q2", "Q3", "Q4"]
    # higher quartile => higher median confidence
    med = strata.groupby("stratum", observed=True)["gene_conf"].median()
    assert med["Q4"] > med["Q1"]


def test_assign_gene_strata_drops_low_count_genes():
    ev = _toy_eval_frame(n_genes=10, n_cond=8)
    # add a gene with too few cells
    extra = pd.DataFrame([{"gene_key": "org:tiny", "orgId": "org",
                           "condition_key": f"c{c}", "fit": 0.0,
                           "model_pred": 0.0, "knn_pred": 0.0, "null_pred": 0.0,
                           "abs_t": 5.0} for c in range(3)])
    strata = assign_gene_strata(pd.concat([ev, extra]), n_strata=4, min_n=5)
    assert "org:tiny" not in set(strata["gene_key"])


def test_cell_filter_attrition_monotone():
    ev = _toy_eval_frame(n_genes=30, n_cond=10)
    cf = cell_filter_analysis(ev, thresholds=(0.0, 2.0, 4.0), min_n=5)
    # raising the |t| threshold can only keep fewer-or-equal cells
    assert cf["n_cells"].is_monotonic_decreasing
    # threshold 0 keeps everything
    assert cf.loc[cf["abs_t_threshold"] == 0.0, "n_cells"].iloc[0] == len(ev)
