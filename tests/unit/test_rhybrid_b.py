"""Unit tests for R-HYBRID-B: exclude_self kNN, retrieval features, and that
each of the three learned hybrids runs end-to-end on tiny synthetic data.

The end-to-end tests build a minimal R1Data-like object (only the fields the
R-HYBRID-B runners touch) so they run on CPU in seconds without the real
fitness DB / embeddings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ranking.eval import (
    chemistry_knn_predict, chemistry_retrieval_features)


# ---------------------------------------------------------------------------
# exclude_self correctness
# ---------------------------------------------------------------------------

def _toy_cond_features():
    # 4 conditions on a line in 2-D so nearest-neighbor order is unambiguous
    return {
        "c0": np.array([1.0, 0.0], np.float32),
        "c1": np.array([0.9, 0.1], np.float32),   # very close to c0
        "c2": np.array([0.0, 1.0], np.float32),
        "c3": np.array([0.1, 0.9], np.float32),   # very close to c2
    }


def test_exclude_self_changes_neighbor_for_in_vocab_condition():
    cf = _toy_cond_features()
    # gene g has a fit at every train condition; predict its own c0.
    train = pd.DataFrame({
        "gene_key": ["g"] * 4,
        "condition_key": ["c0", "c1", "c2", "c3"],
        "fit": [10.0, 1.0, 2.0, 3.0],   # c0 is an outlier so self vs non-self differ
    })
    query = pd.DataFrame({"gene_key": ["g"], "condition_key": ["c0"]})

    incl = chemistry_knn_predict(train, query, cf, k=1, exclude_self=False).iloc[0]
    excl = chemistry_knn_predict(train, query, cf, k=1, exclude_self=True).iloc[0]
    # k=1 incl self: nearest to c0 is c0 itself -> its own fit 10.0
    assert incl == pytest.approx(10.0)
    # exclude self: nearest becomes c1 -> fit 1.0
    assert excl == pytest.approx(1.0)


def test_exclude_self_noop_when_condition_not_in_train():
    cf = _toy_cond_features()
    train = pd.DataFrame({
        "gene_key": ["g"] * 3,
        "condition_key": ["c1", "c2", "c3"],
        "fit": [1.0, 2.0, 3.0],
    })
    # query a cold condition c0 (not in train) -> exclude_self has nothing to drop
    query = pd.DataFrame({"gene_key": ["g"], "condition_key": ["c0"]})
    incl = chemistry_knn_predict(train, query, cf, k=2, exclude_self=False).iloc[0]
    excl = chemistry_knn_predict(train, query, cf, k=2, exclude_self=True).iloc[0]
    assert incl == pytest.approx(excl)


def test_exclude_self_leave_one_out_on_train():
    # val_df == train_df with exclude_self=True is honest leave-one-out:
    # no train (g,c) should ever predict itself.
    cf = _toy_cond_features()
    train = pd.DataFrame({
        "gene_key": ["g"] * 4,
        "condition_key": ["c0", "c1", "c2", "c3"],
        "fit": [10.0, 1.0, 2.0, 3.0],
    })
    loo = chemistry_knn_predict(train, train, cf, k=1, exclude_self=True)
    # c0's LOO nearest is c1 (fit 1.0), not itself (10.0)
    pred_c0 = loo[train["condition_key"] == "c0"].iloc[0]
    assert pred_c0 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# retrieval features
# ---------------------------------------------------------------------------

def test_retrieval_features_shape_and_columns():
    cf = _toy_cond_features()
    train = pd.DataFrame({
        "gene_key": ["g"] * 4,
        "condition_key": ["c0", "c1", "c2", "c3"],
        "fit": [10.0, 1.0, 2.0, 3.0],
    })
    query = pd.DataFrame({"gene_key": ["g"], "condition_key": ["c0"]})
    k = 2
    feats = chemistry_retrieval_features(train, query, cf, k=k, exclude_self=True)
    assert feats.shape == (1, 2 * k + 2)
    # exclude_self: c0=[1,0]'s nearest train conds by cosine are c1 (sim~1.0),
    # then c3=[0.1,0.9] (sim~0.11) ahead of c2=[0,1] (sim 0).
    assert feats[0, 0] == pytest.approx(1.0)   # c1 fit
    assert feats[0, 1] == pytest.approx(3.0)   # c3 fit
    # similarities are descending
    assert feats[0, k] >= feats[0, k + 1]
    # weighted mean fit lies between the neighbor fits (1.0 .. 3.0)
    assert 1.0 <= feats[0, 2 * k] <= 3.0
    # coverage = both neighbors present -> 1.0
    assert feats[0, 2 * k + 1] == pytest.approx(1.0)


def test_retrieval_features_missing_fit_imputed_and_excluded():
    cf = _toy_cond_features()
    # gene g is missing a fit at c2 -> coverage should drop, slot imputed to 0
    train = pd.DataFrame({
        "gene_key": ["g", "g", "g"],
        "condition_key": ["c0", "c1", "c3"],
        "fit": [10.0, 1.0, 3.0],
    })
    query = pd.DataFrame({"gene_key": ["g"], "condition_key": ["c0"]})
    k = 2
    feats = chemistry_retrieval_features(train, query, cf, k=k, exclude_self=True)
    # c0's nearest train conds: c1 (present), then c2 (ABSENT for g) or c3.
    # The neighbor set is by chemistry over TRAIN conds {c0,c1,c3} (c2 not in
    # train), so neighbors are c1 then c3 — both present -> coverage 1.0.
    assert feats[0, 2 * k + 1] == pytest.approx(1.0)


def test_retrieval_features_val_uses_train_only_no_leakage():
    # A cold val condition must retrieve from train conditions only.
    cf = _toy_cond_features()
    train = pd.DataFrame({
        "gene_key": ["g", "g"],
        "condition_key": ["c1", "c3"],
        "fit": [1.0, 3.0],
    })
    val = pd.DataFrame({"gene_key": ["g"], "condition_key": ["c0"]})  # cold
    feats = chemistry_retrieval_features(train, val, cf, k=2, exclude_self=False)
    # only 2 train conds -> both used; fits are 1.0 and 3.0 in some order
    fit_slots = set(np.round(feats[0, :2], 3))
    assert fit_slots == {1.0, 3.0}


# ---------------------------------------------------------------------------
# End-to-end: each hybrid runs on a tiny synthetic R1Data
# ---------------------------------------------------------------------------

class _ToyData:
    """Minimal stand-in for R1Data exposing only the fields the runners use."""
    def __init__(self):
        rng = np.random.default_rng(0)
        n_genes, n_train_cond, n_val_cond = 12, 8, 4
        genes = [f"g{i}" for i in range(n_genes)]
        train_conds = [f"tc{j}" for j in range(n_train_cond)]
        val_conds = [f"vc{j}" for j in range(n_val_cond)]
        dim = 6
        # chemistry feature per condition
        self.cond_features = {}
        exp_to_chem = {}
        for c in train_conds + val_conds:
            v = rng.normal(size=dim).astype(np.float32)
            self.cond_features[c] = v
            exp_to_chem[c] = v
        # one experiment per condition (experiment_id == condition_key for the toy)
        def make_rows(conds):
            rows = []
            for g in genes:
                gi = int(g[1:])
                for c in conds:
                    base = rng.normal()
                    rows.append({"orgId": "TOY", "gene_key": g,
                                 "condition_key": c, "experiment_id": c,
                                 "fit": float(base - 0.3 * gi)})
            return pd.DataFrame(rows)
        self.train = make_rows(train_conds)
        self.train["w_g"] = 1.0
        val = make_rows(val_conds)
        val = (val.groupby(["orgId", "gene_key", "condition_key"])
               .agg(fit=("fit", "mean"),
                    experiment_id=("experiment_id", "first"))
               .reset_index())
        val["eligible"] = True
        self.val = val
        self.val_raw = val
        # embeddings: one row per gene
        self.emb = rng.normal(size=(n_genes, 16)).astype(np.float32)
        self.gene_to_row = {g: i for i, g in enumerate(genes)}
        # multihot / exp matrices keyed by experiment_id (== condition_key)
        all_exps = train_conds + val_conds
        self.exp_to_row = {e: i for i, e in enumerate(all_exps)}
        import scipy.sparse as sp
        mh = np.vstack([exp_to_chem[e] for e in all_exps])
        mh = (mh > 0).astype(np.float32)   # binary multihot
        self.multihot = sp.csr_matrix(mh)
        self.eligible_val_genes = set(genes)
        self.fp_bundle = {}
        self.mf_val_pred = None


def test_residual_runs_end_to_end():
    from src.experiments.rhybrid import _rhybrid_b as rb
    data = _ToyData()
    block = rb.run_residual(data, seed=0, epochs=2, k=3, n_bootstrap=20)
    assert "hybrid" in block and "chem_knn" in block
    assert np.isfinite(block["hybrid"]["ndcg_at_5"]) or \
        np.isnan(block["hybrid"]["ndcg_at_5"])
    assert "honest" in block


def test_retrieval_runs_end_to_end():
    from src.experiments.rhybrid import _rhybrid_b as rb
    data = _ToyData()
    block = rb.run_retrieval(data, seed=0, epochs=2, k=3, n_bootstrap=20)
    assert block["model"] == "retrieval"
    assert "honest" in block


def test_gating_runs_end_to_end():
    from src.experiments.rhybrid import _rhybrid_b as rb
    data = _ToyData()
    block = rb.run_gating(data, seed=0, epochs_model=2, k=3, n_bootstrap=20)
    assert block["model"] == "gating"
    assert 0.0 <= block["mean_alpha"] <= 1.0
    assert "honest" in block
