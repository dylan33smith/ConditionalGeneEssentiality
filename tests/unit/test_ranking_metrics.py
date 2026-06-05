"""Unit tests for src/evaluation/ranking_metrics.py (R-LOCK-4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ranking_metrics import (
    BootstrapMetric,
    cross_gene_within_condition_noise_proxy,
    model_beats_baseline,
    per_condition_train_mean_predictions,
    ranking_baseline_metrics,
    task_relevant_noise_floor,
    within_gene_rank_metric,
)


# ---------------------------------------------------------------------------
# within_gene_rank_metric
# ---------------------------------------------------------------------------

def _ranking_df(per_gene_data: dict[str, tuple[list[float], list[float]]]) -> pd.DataFrame:
    """Build a (gene_key, fit, pred) frame from {gene: (fit_list, pred_list)}."""
    rows = []
    for gene, (fits, preds) in per_gene_data.items():
        for f, p in zip(fits, preds):
            rows.append({"gene_key": gene, "fit": f, "pred": p})
    return pd.DataFrame(rows)


def test_within_gene_spearman_perfect_correlation():
    df = _ranking_df({"g1": ([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])})
    m = within_gene_rank_metric(df, metric="spearman", n_bootstrap=100)
    assert m.mean == pytest.approx(1.0)
    assert m.n_genes_used == 1


def test_within_gene_spearman_anti_correlation():
    df = _ranking_df({"g1": ([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])})
    m = within_gene_rank_metric(df, metric="spearman", n_bootstrap=100)
    assert m.mean == pytest.approx(-1.0)


def test_within_gene_kendall_runs():
    df = _ranking_df({"g1": ([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])})
    m = within_gene_rank_metric(df, metric="kendall", n_bootstrap=100)
    assert m.mean == pytest.approx(1.0)


def test_within_gene_metric_excludes_constant_predictions():
    """Spearman is undefined when one side is constant — gene should be skipped."""
    df = _ranking_df({"g1": ([1, 2, 3, 4, 5], [7, 7, 7, 7, 7])})
    m = within_gene_rank_metric(df, metric="spearman", n_bootstrap=100)
    assert m.n_genes_used == 0
    assert np.isnan(m.mean)


def test_within_gene_metric_min_n_filter():
    df = _ranking_df({
        "g_short": ([1, 2], [1, 2]),               # too short
        "g_full":  ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    })
    m = within_gene_rank_metric(df, metric="spearman", min_n=5, n_bootstrap=100)
    assert m.n_genes_used == 1


def test_within_gene_metric_mean_across_genes():
    df = _ranking_df({
        "g_perfect":  ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),    # r = +1
        "g_inverted": ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),    # r = -1
    })
    m = within_gene_rank_metric(df, metric="spearman", n_bootstrap=100)
    assert m.mean == pytest.approx(0.0)
    assert m.n_genes_used == 2


def test_within_gene_metric_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    rows = []
    for g in range(50):
        fits = rng.normal(size=10)
        preds = fits + rng.normal(size=10) * 0.5
        for f, p in zip(fits, preds):
            rows.append({"gene_key": f"g{g}", "fit": f, "pred": p})
    df = pd.DataFrame(rows)
    m = within_gene_rank_metric(df, metric="spearman", n_bootstrap=500)
    assert m.ci_low <= m.mean <= m.ci_high
    assert m.ci_high - m.ci_low > 0
    assert m.n_bootstrap == 500


def test_eligible_mask_filters_genes():
    df = _ranking_df({
        "g_keep": ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        "g_drop": ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),
    })
    mask = df["gene_key"] == "g_keep"
    m = within_gene_rank_metric(df, metric="spearman", eligible_mask=mask, n_bootstrap=100)
    assert m.mean == pytest.approx(1.0)
    assert m.n_genes_used == 1


# ---------------------------------------------------------------------------
# Ranking baseline (H-RANK-01)
# ---------------------------------------------------------------------------

def test_per_condition_train_mean_predictions():
    train = pd.DataFrame({
        "condition_key": ["A", "A", "B", "B"],
        "fit": [1.0, 3.0, 10.0, 20.0],
    })
    val = pd.DataFrame({
        "condition_key": ["A", "B"],
    })
    preds = per_condition_train_mean_predictions(train, val)
    assert preds.iloc[0] == 2.0
    assert preds.iloc[1] == 15.0


def test_per_condition_train_mean_unknown_condition_is_nan():
    train = pd.DataFrame({"condition_key": ["A"], "fit": [1.0]})
    val = pd.DataFrame({"condition_key": ["A", "Z"]})
    preds = per_condition_train_mean_predictions(train, val)
    assert preds.iloc[0] == 1.0
    assert np.isnan(preds.iloc[1])


def test_ranking_baseline_metrics_runs():
    """Baseline predicts the same value per condition across all genes,
    so the constant-prediction filter should drop genes whose val
    condition_key set is itself a singleton."""
    rng = np.random.default_rng(0)
    rows = []
    conditions = [f"c{i}" for i in range(8)]
    for g in range(10):
        for c in conditions:
            rows.append({"gene_key": f"g{g}", "condition_key": c,
                         "fit": float(rng.normal())})
    df = pd.DataFrame(rows)
    # Use the same df as train and val for this smoke test
    out = ranking_baseline_metrics(val_df=df, train_df=df, n_bootstrap=200)
    assert "spearman_mean" in out
    assert "kendall_mean" in out
    assert "spearman_bootstrap" in out


# ---------------------------------------------------------------------------
# H-RANK-01 gate
# ---------------------------------------------------------------------------

def test_model_beats_baseline_disjoint_ci():
    model = BootstrapMetric(mean=0.20, ci_low=0.15, ci_high=0.25, n_genes_used=100, n_bootstrap=500)
    baseline = BootstrapMetric(mean=0.05, ci_low=0.00, ci_high=0.10, n_genes_used=100, n_bootstrap=500)
    assert model_beats_baseline(model, baseline)


def test_model_beats_baseline_overlapping_ci():
    model = BootstrapMetric(mean=0.20, ci_low=0.10, ci_high=0.30, n_genes_used=100, n_bootstrap=500)
    baseline = BootstrapMetric(mean=0.15, ci_low=0.05, ci_high=0.25, n_genes_used=100, n_bootstrap=500)
    assert not model_beats_baseline(model, baseline)


def test_model_beats_baseline_handles_nan():
    nan_metric = BootstrapMetric(mean=float("nan"), ci_low=float("nan"),
                                  ci_high=float("nan"), n_genes_used=0, n_bootstrap=0)
    other = BootstrapMetric(mean=0.10, ci_low=0.0, ci_high=0.2, n_genes_used=100, n_bootstrap=500)
    assert not model_beats_baseline(nan_metric, other)
    assert not model_beats_baseline(other, nan_metric)


# ---------------------------------------------------------------------------
# Task-relevant noise floor
# ---------------------------------------------------------------------------

def test_task_relevant_noise_floor_perfect_replicates():
    """If rep_A and rep_B agree exactly for every gene, noise floor Spearman = 1.0."""
    rows = []
    conditions = [f"c{i}" for i in range(6)]
    for g in range(3):
        for c in conditions:
            fit_val = float(np.random.default_rng(g + hash(c) % 1000).normal())
            rows.append({"orgId": "O", "gene_key": f"g{g}", "condition_key": c,
                         "expName": "expA", "fit": fit_val})
            rows.append({"orgId": "O", "gene_key": f"g{g}", "condition_key": c,
                         "expName": "expB", "fit": fit_val})   # identical
    df = pd.DataFrame(rows)
    out = task_relevant_noise_floor(df, min_conditions=5)
    assert out["primary_value"] == pytest.approx(1.0)
    assert out["n_genes_used"] == 3


def test_task_relevant_noise_floor_skips_genes_with_too_few_pairs():
    rows = []
    for c in range(3):                                # only 3 conditions
        rows.append({"orgId": "O", "gene_key": "g1", "condition_key": f"c{c}",
                     "expName": "expA", "fit": float(c)})
        rows.append({"orgId": "O", "gene_key": "g1", "condition_key": f"c{c}",
                     "expName": "expB", "fit": float(c)})
    df = pd.DataFrame(rows)
    out = task_relevant_noise_floor(df, min_conditions=5)
    assert out["n_genes_used"] == 0
    assert out["n_skipped_too_few_pairs"] == 1


def test_cross_gene_within_condition_noise_proxy_runs():
    rng = np.random.default_rng(0)
    rows = []
    for c in range(3):
        for g in range(30):
            base = float(rng.normal())
            rows.append({"orgId": "O", "gene_key": f"g{g}", "condition_key": f"c{c}",
                         "expName": "expA", "fit": base})
            rows.append({"orgId": "O", "gene_key": f"g{g}", "condition_key": f"c{c}",
                         "expName": "expB", "fit": base + 0.01 * rng.normal()})
    df = pd.DataFrame(rows)
    out = cross_gene_within_condition_noise_proxy(df, min_genes_per_pair=20)
    assert 0.5 < out["proxy_value"] < 1.0
    assert out["n_pairs_used"] == 3
