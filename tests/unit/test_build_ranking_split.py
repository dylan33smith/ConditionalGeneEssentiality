"""Unit tests for src/data/datasets/build_ranking_split.py (R-LOCK-2 materializer)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.datasets.build_ranking_split import (
    assert_no_replicate_leakage,
    materialize_cell_holdout,
    materialize_cold_gene,
    materialize_condition_holdout,
)


@pytest.fixture
def synthetic_fit():
    """2 orgs × 20 conditions × 30 genes, each condition with 2 replicate expNames.

    Columns match what `_condition_key` needs: expDesc, media, temperature.
    """
    rows = []
    rng = np.random.default_rng(0)
    for org in ["OrgA", "OrgB"]:
        for c in range(20):
            expdesc = f"stressor_{c}"
            media = "LB"
            temp = 37
            expgroup = "stress" if c % 2 == 0 else "carbon source"
            # two replicate expNames sharing the same (expDesc, media, temperature)
            for rep in range(2):
                expname = f"{org}_set{c}_rep{rep}"
                for g in range(30):
                    rows.append({
                        "orgId": org, "gene_key": f"{org}:g{g}",
                        "expName": expname, "expDesc": expdesc, "media": media,
                        "temperature": temp, "expGroup": expgroup,
                        "fit": float(rng.normal()),
                    })
    return pd.DataFrame(rows)


def test_condition_holdout_partitions_are_disjoint(synthetic_fit):
    split = materialize_condition_holdout(synthetic_fit, fraction=0.20,
                                          min_holdout=3, seed=0)
    parts = set(split.partition.unique())
    assert parts <= {"train", "val"}
    assert split.stats["n_val_rows"] > 0
    assert split.stats["n_train_rows"] > 0


def test_condition_holdout_no_replicate_leakage(synthetic_fit):
    """The core guarantee: no condition_key spans both partitions."""
    df = synthetic_fit.dropna(subset=["orgId", "gene_key", "expDesc", "media"]).copy()
    split = materialize_condition_holdout(synthetic_fit, fraction=0.20,
                                          min_holdout=3, seed=0)
    # Should not raise
    assert_no_replicate_leakage(df, split)


def test_condition_holdout_both_replicates_same_partition(synthetic_fit):
    """Both replicate expNames of a held-out condition must be in val together."""
    split = materialize_condition_holdout(synthetic_fit, fraction=0.20,
                                          min_holdout=3, seed=0)
    df = synthetic_fit.loc[split.partition.index].copy()
    df["partition"] = split.partition.values
    df["condition_key"] = split.condition_key.values
    # For every condition_key, all its expNames share one partition
    for ck, sub in df.groupby("condition_key"):
        assert sub["partition"].nunique() == 1, f"{ck} leaked across partitions"


def test_condition_holdout_fraction_respected(synthetic_fit):
    """~20% of 20 conditions = ~4 held out per org (clipped to [3, 30])."""
    split = materialize_condition_holdout(synthetic_fit, fraction=0.20,
                                          min_holdout=3, max_holdout=30, seed=0)
    # 2 orgs × ~4 conditions = ~8 val conditions
    assert 6 <= split.stats["n_val_conditions"] <= 10


def test_condition_holdout_deterministic_hash(synthetic_fit):
    s1 = materialize_condition_holdout(synthetic_fit, seed=0)
    s2 = materialize_condition_holdout(synthetic_fit, seed=0)
    assert s1.split_hash == s2.split_hash
    s3 = materialize_condition_holdout(synthetic_fit, seed=1)
    assert s1.split_hash != s3.split_hash


def test_cold_gene_holds_out_whole_genes(synthetic_fit):
    """Cold-gene: a held-out gene appears in NO train row."""
    split = materialize_cold_gene(synthetic_fit, fraction=0.20, seed=0)
    df = synthetic_fit.loc[split.partition.index].copy()
    df["partition"] = split.partition.values
    val_genes = set(df.loc[df["partition"] == "val", "gene_key"])
    train_genes = set(df.loc[df["partition"] == "train", "gene_key"])
    assert val_genes.isdisjoint(train_genes), "cold-gene split leaked a gene into both"


def test_cell_holdout_runs(synthetic_fit):
    split = materialize_cell_holdout(synthetic_fit, fraction=0.20, seed=0)
    assert split.stats["n_val_cells"] > 0
    assert set(split.partition.unique()) <= {"train", "val"}


def test_leakage_guard_catches_injected_leak(synthetic_fit):
    """Sanity: the guard actually fires when a condition spans partitions."""
    df = synthetic_fit.dropna(subset=["orgId", "gene_key", "expDesc", "media"]).copy()
    split = materialize_condition_holdout(synthetic_fit, seed=0)
    # Corrupt: flip one row's partition to create a spanning condition_key
    bad_partition = split.partition.copy()
    # find a val row and flip it to train (its condition_key now spans both)
    val_idx = bad_partition[bad_partition == "val"].index[0]
    bad_partition.loc[val_idx] = "train"
    split.partition = bad_partition
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_replicate_leakage(df, split)
