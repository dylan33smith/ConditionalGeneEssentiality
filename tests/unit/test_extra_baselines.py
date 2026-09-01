"""E -- the tree and learned-local baselines."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
import pytest

from src.ranking.eval.extra_baselines import gbdt_predict, resmem_predict


def _toy(n_genes=40, n_exps=12, d_emb=16, d_chem=6, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n_genes, d_emb))
    chem = rng.normal(size=(n_exps, d_chem))
    gene_to_row = {f"g{i}": i for i in range(n_genes)}
    exp_to_row = {f"e{j}": j for j in range(n_exps)}
    rows = []
    for i in range(n_genes):
        for j in range(n_exps):
            fit = float(emb[i, 0] * chem[j, 0] + 0.1 * rng.normal())
            rows.append(dict(gene_key=f"g{i}", experiment_id=f"e{j}",
                             condition_key=f"c{j}", orgId="o", fit=fit))
    df = pd.DataFrame(rows)
    return df, emb, gene_to_row, chem, exp_to_row


def test_gbdt_learns_a_signal_it_should_be_able_to_learn():
    df, emb, g2r, chem, e2r = _toy()
    train = df[df.experiment_id != "e0"].reset_index(drop=True)
    val = df[df.experiment_id == "e0"].reset_index(drop=True)
    pred = gbdt_predict(train, val, gene_emb=emb, gene_to_row=g2r,
                        chem=chem, exp_to_row=e2r, emb_components=8, max_iter=30)
    assert pred.shape == (len(val),)
    assert np.isfinite(pred).all()
    r = np.corrcoef(pred, val.fit.to_numpy())[0, 1]
    assert r > 0.3, f"GBDT should recover a multiplicative signal, got r={r:.3f}"


def test_gbdt_returns_nan_for_rows_it_cannot_feature():
    df, emb, g2r, chem, e2r = _toy()
    train = df[df.experiment_id != "e0"].reset_index(drop=True)
    val = df[df.experiment_id == "e0"].copy()
    val.loc[val.index[:3], "gene_key"] = "UNKNOWN_GENE"
    val = val.reset_index(drop=True)
    pred = gbdt_predict(train, val, gene_emb=emb, gene_to_row=g2r,
                        chem=chem, exp_to_row=e2r, emb_components=8, max_iter=30)
    assert np.isnan(pred[:3]).all(), "unfeaturizable rows must be NaN, not 0"


def test_resmem_falls_back_to_the_model_on_a_cold_gene():
    """The property the earlier hybrids lacked: no own-history -> pure model."""
    rng = np.random.default_rng(0)
    feats = {f"c{j}": rng.normal(size=5) for j in range(4)}
    train = pd.DataFrame([dict(gene_key="warm", condition_key=f"c{j}", fit=1.0)
                          for j in range(3)])
    val = pd.DataFrame([dict(gene_key="cold", condition_key="c3", fit=0.0)])
    model_tr = np.zeros(len(train))
    model_va = np.array([0.42])
    out = resmem_predict(model_tr, model_va, train, val, feats)
    assert out[0] == pytest.approx(0.42), (
        "a gene with no train residuals must receive zero correction")


def test_resmem_corrects_a_systematic_model_bias_on_a_warm_gene():
    rng = np.random.default_rng(1)
    base = rng.normal(size=5)
    feats = {"c0": base, "c1": base + 1e-6, "c2": base + 2e-6, "c3": base + 3e-6}
    # model underpredicts this gene by exactly 1.0 everywhere
    train = pd.DataFrame([dict(gene_key="g", condition_key=f"c{j}", fit=1.0)
                          for j in range(3)])
    val = pd.DataFrame([dict(gene_key="g", condition_key="c3", fit=1.0)])
    out = resmem_predict(np.zeros(len(train)), np.array([0.0]), train, val, feats)
    assert out[0] == pytest.approx(1.0, abs=1e-3), (
        "ResMem should recover the gene's systematic residual")


def test_resmem_leaves_prediction_unchanged_when_residuals_are_zero():
    rng = np.random.default_rng(2)
    feats = {f"c{j}": rng.normal(size=4) for j in range(4)}
    train = pd.DataFrame([dict(gene_key="g", condition_key=f"c{j}", fit=0.5)
                          for j in range(3)])
    val = pd.DataFrame([dict(gene_key="g", condition_key="c3", fit=0.5)])
    out = resmem_predict(np.full(len(train), 0.5), np.array([0.5]), train, val, feats)
    assert out[0] == pytest.approx(0.5)
