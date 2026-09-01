"""D -- the similarity-stratified diagnostic."""
import numpy as np
import pandas as pd
import pytest

from src.ranking.eval.harness import (
    nearest_train_condition_distance,
    similarity_stratified_report,
)


def _features(names, rng):
    return {n: rng.normal(size=8) for n in names}


def test_distance_is_small_when_a_gene_has_a_near_train_condition():
    rng = np.random.default_rng(0)
    base = rng.normal(size=8)
    feats = {"c_train": base,
             "c_near": base + 1e-6,          # essentially the same chemistry
             "c_far": -base}                  # opposite direction
    train = pd.DataFrame([dict(gene_key="g1", condition_key="c_train", orgId="o")])
    val_near = pd.DataFrame([dict(gene_key="g1", condition_key="c_near", orgId="o")])
    val_far = pd.DataFrame([dict(gene_key="g1", condition_key="c_far", orgId="o")])

    d_near = nearest_train_condition_distance(train, val_near, feats)
    d_far = nearest_train_condition_distance(train, val_far, feats)
    assert d_near["nearest_train_distance"].iloc[0] < 0.01
    assert d_far["nearest_train_distance"].iloc[0] > 1.0
    assert d_near["n_train_conditions"].iloc[0] == 1


def test_genes_without_train_or_val_conditions_are_dropped():
    rng = np.random.default_rng(1)
    feats = _features(["a", "b"], rng)
    train = pd.DataFrame([dict(gene_key="g1", condition_key="a", orgId="o")])
    val = pd.DataFrame([dict(gene_key="g2", condition_key="b", orgId="o")])  # cold gene
    out = nearest_train_condition_distance(train, val, feats)
    assert out.empty, "a gene with no train conditions has no defined distance"


def test_stratified_report_buckets_and_reports_every_method():
    rng = np.random.default_rng(2)
    n = 200
    dist = pd.DataFrame({
        "gene_key": [f"g{i}" for i in range(n)],
        "orgId": ["o"] * n,
        "nearest_train_distance": np.linspace(0.0, 1.0, n),
        "n_val_conditions": 5, "n_train_conditions": 5,
    })
    # lookup decays with distance; model is flat -- the predicted signature
    lookup = pd.DataFrame({
        "gene_key": dist.gene_key,
        "ndcg_at_5": 0.7 - 0.4 * dist.nearest_train_distance})
    model = pd.DataFrame({
        "gene_key": dist.gene_key,
        "ndcg_at_5": np.full(n, 0.45)})

    rep = similarity_stratified_report(
        {"chem_knn": lookup, "model": model}, dist, n_buckets=4)

    assert set(rep["method"]) == {"chem_knn", "model"}
    assert rep["bucket"].nunique() == 4
    # every method reported in every bucket -- no omitted rows
    assert len(rep) == 8

    near = rep[(rep.bucket == 0)].set_index("method")["value"]
    far = rep[(rep.bucket == 3)].set_index("method")["value"]
    assert near["chem_knn"] > near["model"], "lookup should dominate the near bucket"
    assert far["chem_knn"] < far["model"], "and lose the far bucket"


def test_stratified_report_is_empty_safe():
    empty = pd.DataFrame(columns=["gene_key", "orgId", "nearest_train_distance"])
    out = similarity_stratified_report({"m": pd.DataFrame()}, empty)
    assert out.empty


def test_bad_aggregate_is_rejected():
    rng = np.random.default_rng(3)
    feats = _features(["a"], rng)
    df = pd.DataFrame([dict(gene_key="g", condition_key="a", orgId="o")])
    with pytest.raises(ValueError, match="median"):
        nearest_train_condition_distance(df, df, feats, aggregate="bogus")
