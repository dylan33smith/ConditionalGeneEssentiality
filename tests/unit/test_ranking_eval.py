"""Unit tests for src/evaluation/ranking_eval.py (R-LOCK-4 v2 harness)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ranking_eval import (
    benjamini_hochberg,
    bootstrap_pvalue_delta,
    chemistry_knn_predict,
    chemistry_nearest_condition_profile,
    hierarchical_bootstrap_ci,
    ndcg_at_k,
    per_gene_correlations,
    per_organism_breakdown,
    precision_at_k,
    retrieval_noise_floor,
    within_gene_retrieval,
)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def test_ndcg_perfect_ranking():
    # most-negative fit = most stressful; perfect predicted order
    fit = np.array([-3.0, -2.0, -1.0, 0.0, 0.5])
    pred = np.array([-3.0, -2.0, -1.0, 0.0, 0.5])  # identical => perfect
    assert ndcg_at_k(fit, pred, k=3) == pytest.approx(1.0)


def test_ndcg_inverted_ranking_below_one():
    fit = np.array([-3.0, -2.0, -1.0, 0.0, 0.5])
    pred = -fit  # inverted: predicts the beneficial ones as most essential
    assert ndcg_at_k(fit, pred, k=3) < 1.0


def test_ndcg_nan_when_no_stressors():
    fit = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # all relevance 0
    assert np.isnan(ndcg_at_k(fit, fit, k=3))


def test_precision_at_k_perfect():
    fit = np.array([-3.0, -2.0, -1.0, 0.0, 0.5])
    pred = fit.copy()
    assert precision_at_k(fit, pred, k=2) == pytest.approx(1.0)


def test_precision_at_k_half():
    # true top-2 stressors are indices 0,1 (fit -3,-2). Predict top-2 as 0 and 4.
    fit = np.array([-3.0, -2.0, -1.0, 0.0, 0.5])
    pred = np.array([-3.0, 0.0, 0.0, 0.0, -2.9])  # predicts idx 0 and 4 as top
    # overlap with true top-2 {0,1} = {0} -> 1/2
    assert precision_at_k(fit, pred, k=2) == pytest.approx(0.5)


def test_within_gene_retrieval_shape():
    rng = np.random.default_rng(0)
    rows = []
    for g in range(5):
        for c in range(8):
            f = float(rng.normal())
            rows.append({"gene_key": f"g{g}", "orgId": "O", "condition_key": f"c{c}",
                         "fit": f, "pred": f + 0.1 * rng.normal()})
    df = pd.DataFrame(rows)
    out = within_gene_retrieval(df, k_values=(1, 3, 5))
    assert {"ndcg_at_5", "precision_at_3", "orgId"} <= set(out.columns)
    assert len(out) <= 5


# ---------------------------------------------------------------------------
# Hierarchical bootstrap
# ---------------------------------------------------------------------------

def test_hierarchical_bootstrap_brackets_mean():
    rng = np.random.default_rng(0)
    rows = []
    for org in range(6):
        for g in range(40):
            rows.append({"orgId": f"O{org}", "gene_key": f"O{org}_g{g}",
                         "value": float(rng.normal(0.2, 0.1))})
    pg = pd.DataFrame(rows)
    res = hierarchical_bootstrap_ci(pg, n_bootstrap=500)
    assert res["ci_low"] <= res["mean"] <= res["ci_high"]
    assert res["n_orgs"] == 6
    assert res["n_genes"] == 240


def test_hierarchical_ci_wider_than_flat_for_clustered_data():
    """Clustered data: org-level signal => hierarchical CI should be wider
    than a naive flat gene bootstrap (the whole point of the change)."""
    rng = np.random.default_rng(1)
    rows = []
    for org in range(5):
        org_mean = rng.normal(0.2, 0.15)  # strong between-org variation
        for g in range(50):
            rows.append({"orgId": f"O{org}", "gene_key": f"O{org}_g{g}",
                         "value": float(rng.normal(org_mean, 0.02))})
    pg = pd.DataFrame(rows)
    hier = hierarchical_bootstrap_ci(pg, n_bootstrap=800, seed=0)
    # flat bootstrap over genes ignoring org structure
    rng2 = np.random.default_rng(0)
    vals = pg["value"].to_numpy()
    flat = [vals[rng2.integers(0, len(vals), len(vals))].mean() for _ in range(800)]
    flat_w = np.quantile(flat, 0.975) - np.quantile(flat, 0.025)
    hier_w = hier["ci_high"] - hier["ci_low"]
    assert hier_w > flat_w


# ---------------------------------------------------------------------------
# FDR
# ---------------------------------------------------------------------------

def test_bh_all_significant():
    rej, q = benjamini_hochberg([0.001, 0.002, 0.003], alpha=0.05)
    assert rej.all()


def test_bh_none_significant():
    rej, q = benjamini_hochberg([0.4, 0.6, 0.8], alpha=0.05)
    assert not rej.any()


def test_bh_partial_and_order_preserved():
    # classic BH example
    p = [0.01, 0.04, 0.03, 0.20, 0.50]
    rej, q = benjamini_hochberg(p, alpha=0.05)
    # the small ones get rejected, the big ones don't; output aligns to input order
    assert rej[0] and not rej[3] and not rej[4]
    assert len(rej) == 5 and len(q) == 5
    assert (q >= 0).all() and (q <= 1).all()


def test_bh_empty():
    rej, q = benjamini_hochberg([], alpha=0.05)
    assert len(rej) == 0 and len(q) == 0


# ---------------------------------------------------------------------------
# Bootstrap p-value for a delta
# ---------------------------------------------------------------------------

def test_bootstrap_pvalue_model_clearly_better():
    rng = np.random.default_rng(0)
    model = pd.DataFrame({"gene_key": [f"g{i}" for i in range(100)],
                          "orgId": ["O"] * 100,
                          "value": rng.normal(0.30, 0.05, 100)})
    base = pd.DataFrame({"gene_key": [f"g{i}" for i in range(100)],
                         "orgId": ["O"] * 100,
                         "value": rng.normal(0.10, 0.05, 100)})
    p = bootstrap_pvalue_delta(model, base, n_bootstrap=500)
    assert p < 0.05


def test_bootstrap_pvalue_no_difference():
    rng = np.random.default_rng(0)
    v = rng.normal(0.2, 0.05, 100)
    model = pd.DataFrame({"gene_key": [f"g{i}" for i in range(100)],
                          "orgId": ["O"] * 100, "value": v})
    base = pd.DataFrame({"gene_key": [f"g{i}" for i in range(100)],
                         "orgId": ["O"] * 100, "value": v.copy()})
    p = bootstrap_pvalue_delta(model, base, n_bootstrap=500)
    assert p > 0.1   # no real difference => not significant


# ---------------------------------------------------------------------------
# Chemistry baselines (cold-condition aware)
# ---------------------------------------------------------------------------

@pytest.fixture
def chem_split():
    """4 train conditions, 2 val conditions; val conditions are chemically
    close to specific train conditions. 5 genes."""
    # feature vectors: val c_v0 near c0, c_v1 near c3
    cond_features = {
        "c0": np.array([1.0, 0.0, 0.0]),
        "c1": np.array([0.0, 1.0, 0.0]),
        "c2": np.array([0.0, 0.0, 1.0]),
        "c3": np.array([1.0, 1.0, 0.0]),
        "cv0": np.array([0.95, 0.05, 0.0]),   # near c0
        "cv1": np.array([0.9, 0.9, 0.0]),     # near c3
    }
    rng = np.random.default_rng(0)
    tr, va = [], []
    for g in range(5):
        gfit = {c: float(rng.normal()) for c in ["c0", "c1", "c2", "c3"]}
        for c, f in gfit.items():
            tr.append({"gene_key": f"g{g}", "condition_key": c, "fit": f})
        for c in ["cv0", "cv1"]:
            va.append({"gene_key": f"g{g}", "condition_key": c,
                       "fit": float(rng.normal())})
    return pd.DataFrame(tr), pd.DataFrame(va), cond_features


def test_chemistry_nearest_profile_uses_nearest_train_condition(chem_split):
    train, val, feats = chem_split
    pred = chemistry_nearest_condition_profile(train, val, feats)
    assert pred.notna().all()
    # cv0 is nearest c0 => its prediction = train-gene-mean fit at c0 (same for all genes)
    cv0_preds = pred[val["condition_key"].values == "cv0"]
    assert cv0_preds.nunique() == 1   # population profile: identical across genes


def test_chemistry_knn_is_gene_specific(chem_split):
    train, val, feats = chem_split
    pred = chemistry_knn_predict(train, val, feats, k=2)
    assert pred.notna().all()
    # gene-specific (unlike the population profile): predictions vary across genes
    cv0_preds = pred[val["condition_key"].values == "cv0"]
    assert cv0_preds.nunique() > 1


def test_chemistry_baselines_empty_features_return_nan():
    train = pd.DataFrame({"gene_key": ["g0"], "condition_key": ["c0"], "fit": [1.0]})
    val = pd.DataFrame({"gene_key": ["g0"], "condition_key": ["cv0"], "fit": [0.0]})
    pred = chemistry_knn_predict(train, val, {}, k=2)
    assert pred.isna().all()


def test_chemistry_knn_vectorized_matches_bruteforce(chem_split):
    """Vectorized kNN must equal a transparent per-row reference implementation."""
    train, val, feats = chem_split
    k = 2
    fast = chemistry_knn_predict(train, val, feats, k=k)

    # brute-force reference
    train_conds = [c for c in train["condition_key"].unique() if c in feats]
    val_conds = [c for c in val["condition_key"].unique() if c in feats]
    tfeat = np.vstack([feats[c] for c in train_conds])
    vfeat = np.vstack([feats[c] for c in val_conds])
    tn = tfeat / np.maximum(np.linalg.norm(tfeat, axis=1, keepdims=True), 1e-9)
    vn = vfeat / np.maximum(np.linalg.norm(vfeat, axis=1, keepdims=True), 1e-9)
    dist = 1.0 - vn @ tn.T
    knn = {vc: [train_conds[j] for j in np.argsort(dist[i])[:k]]
           for i, vc in enumerate(val_conds)}
    lookup = train.groupby(["gene_key", "condition_key"])["fit"].mean().unstack()
    ref = []
    for _, row in val.iterrows():
        g, vc = row["gene_key"], row["condition_key"]
        neigh = [c for c in knn.get(vc, []) if c in lookup.columns]
        if g in lookup.index and neigh:
            ref.append(float(np.nanmean(lookup.loc[g, neigh].to_numpy())))
        else:
            ref.append(np.nan)
    np.testing.assert_allclose(fast.to_numpy(), np.array(ref), rtol=1e-9, equal_nan=True)


# ---------------------------------------------------------------------------
# Per-org breakdown
# ---------------------------------------------------------------------------

def test_retrieval_noise_floor_perfect_replicates():
    """Identical replicates, all conditions stressors (distinct negative fit) =>
    NDCG@k and precision@k ceiling = 1.0 (no zero-relevance ties)."""
    rows = []
    for g in range(3):
        for c in range(6):
            f = -float(c + 1)        # all negative & distinct => all stressors, strict order
            for en in ("A", "B"):
                rows.append({"orgId": "O", "gene_key": f"g{g}", "condition_key": f"c{c}",
                             "expName": en, "fit": f})   # A == B
    df = pd.DataFrame(rows)
    out = retrieval_noise_floor(df, k_values=(1, 3, 5), min_conditions=5)
    assert out["ndcg_at_5"] == pytest.approx(1.0)
    assert out["precision_at_5"] == pytest.approx(1.0)
    assert out["n_genes_used"] == 3


def test_retrieval_noise_floor_skips_too_few_conditions():
    rows = []
    for c in range(3):
        for en in ("A", "B"):
            rows.append({"orgId": "O", "gene_key": "g1", "condition_key": f"c{c}",
                         "expName": en, "fit": float(c)})
    df = pd.DataFrame(rows)
    out = retrieval_noise_floor(df, min_conditions=5)
    assert out["n_genes_used"] == 0


def test_per_organism_breakdown(chem_split):
    pg = pd.DataFrame({
        "gene_key": [f"g{i}" for i in range(6)],
        "orgId": ["A", "A", "A", "B", "B", "B"],
        "value": [0.3, 0.4, 0.2, 0.1, 0.0, 0.2],
    })
    out = per_organism_breakdown(pg)
    assert set(out["orgId"]) == {"A", "B"}
    assert (out["n_eligible_genes"] == 3).all()
    a_row = out[out["orgId"] == "A"].iloc[0]
    assert a_row["model_spearman"] == pytest.approx(0.3)
