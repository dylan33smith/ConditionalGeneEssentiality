"""Unit tests for src/data/datasets/ranking_batch.py (R-LOCK-3)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.datasets.ranking_batch import (
    PairwiseSampler, PointwiseSampler, ListwiseSampler,
    RankingBatch, build_sampler,
    collate_pairwise, collate_pointwise, collate_listwise,
    policy_hash,
)


# ---------------------------------------------------------------------------
# Fixture: a tiny dataset of 3 genes × 4 conditions each (12 rows)
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_dataset():
    n_genes = 3
    n_conds = 4
    n_rows = n_genes * n_conds
    gene_idx = np.repeat(np.arange(n_genes), n_conds)
    cond_idx = np.tile(np.arange(n_conds), n_genes)
    rng = np.random.default_rng(0)
    fit = rng.normal(size=n_rows).astype(np.float32)
    weight = np.ones(n_rows, dtype=np.float32)
    return dict(gene_idx=gene_idx, cond_idx=cond_idx, fit=fit, weight=weight)


# ---------------------------------------------------------------------------
# Dataclass validation
# ---------------------------------------------------------------------------

def test_rankingbatch_pairwise_requires_sign():
    with pytest.raises(ValueError, match="sign"):
        RankingBatch(
            mode="pairwise",
            gene_idx=torch.zeros(2, 2, dtype=torch.long),
            cond_idx=torch.zeros(2, 2, dtype=torch.long),
            fit=torch.zeros(2, 2),
            weight=torch.zeros(2),
            sign=None,
        )


def test_rankingbatch_listwise_requires_mask():
    with pytest.raises(ValueError, match="mask"):
        RankingBatch(
            mode="listwise",
            gene_idx=torch.zeros(2, 4, dtype=torch.long),
            cond_idx=torch.zeros(2, 4, dtype=torch.long),
            fit=torch.zeros(2, 4),
            weight=torch.zeros(2),
            mask=None,
        )


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def test_pointwise_sampler_yields_all_rows(tiny_dataset):
    s = PointwiseSampler(n_rows=len(tiny_dataset["fit"]), seed=0, shuffle=True)
    rows = list(iter(s))
    assert sorted(rows) == list(range(len(tiny_dataset["fit"])))


def test_pointwise_sampler_shuffle_changes_order_across_epochs(tiny_dataset):
    s = PointwiseSampler(n_rows=len(tiny_dataset["fit"]), seed=0, shuffle=True)
    epoch0 = list(iter(s))
    epoch1 = list(iter(s))
    assert epoch0 != epoch1  # shuffled differently each epoch


def test_pairwise_sampler_count(tiny_dataset):
    s = PairwiseSampler(gene_idx=tiny_dataset["gene_idx"],
                        pairs_per_gene=5, seed=0)
    pairs = list(iter(s))
    # 3 eligible genes × 5 pairs each = 15
    assert len(pairs) == 15


def test_pairwise_sampler_pairs_within_same_gene(tiny_dataset):
    s = PairwiseSampler(gene_idx=tiny_dataset["gene_idx"],
                        pairs_per_gene=10, seed=0)
    for (i, j) in s:
        assert tiny_dataset["gene_idx"][i] == tiny_dataset["gene_idx"][j], \
            "pairwise sampler must pair within the same gene"
        assert i != j


def test_pairwise_sampler_drops_singleton_genes():
    """A gene with only 1 row should be excluded."""
    gene_idx = np.array([0, 0, 0, 1])   # gene 1 has only 1 row
    s = PairwiseSampler(gene_idx=gene_idx, pairs_per_gene=3, seed=0,
                        min_gene_size=2)
    pairs = list(iter(s))
    assert len(pairs) == 3  # only gene 0


def test_listwise_sampler_yields_one_list_per_gene(tiny_dataset):
    s = ListwiseSampler(gene_idx=tiny_dataset["gene_idx"], seed=0)
    lists = list(iter(s))
    assert len(lists) == 3
    for grp in lists:
        # Each yielded list should be all-from-one-gene
        genes_in_grp = tiny_dataset["gene_idx"][grp]
        assert len(set(genes_in_grp.tolist())) == 1


def test_build_sampler_factory(tiny_dataset):
    assert isinstance(build_sampler("pointwise",
                      n_rows=len(tiny_dataset["fit"])), PointwiseSampler)
    assert isinstance(build_sampler("pairwise",
                      gene_idx=tiny_dataset["gene_idx"]), PairwiseSampler)
    assert isinstance(build_sampler("listwise",
                      gene_idx=tiny_dataset["gene_idx"]), ListwiseSampler)


def test_build_sampler_unknown_mode():
    with pytest.raises(ValueError, match="Unknown sampler mode"):
        build_sampler("unknown")


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def test_collate_pointwise_shapes(tiny_dataset):
    rows = [0, 3, 7]
    b = collate_pointwise(rows, **tiny_dataset)
    assert b.mode == "pointwise"
    assert b.gene_idx.shape == (3,)
    assert b.cond_idx.shape == (3,)
    assert b.fit.shape == (3,)
    assert b.weight.shape == (3,)
    assert b.sign is None and b.mask is None


def test_collate_pairwise_shapes_and_sign(tiny_dataset):
    pairs = [(0, 1), (2, 3)]   # both from gene 0
    b = collate_pairwise(pairs, **tiny_dataset)
    assert b.mode == "pairwise"
    assert b.gene_idx.shape == (2, 2)
    assert b.cond_idx.shape == (2, 2)
    assert b.fit.shape == (2, 2)
    assert b.sign.shape == (2,)
    # Sign convention: +1 if fit_i > fit_j; -1 if fit_i < fit_j
    expected_sign = np.sign(tiny_dataset["fit"][[0, 2]] - tiny_dataset["fit"][[1, 3]])
    assert torch.equal(b.sign, torch.from_numpy(expected_sign.astype(np.int64)))


def test_collate_listwise_shapes_and_mask(tiny_dataset):
    # Gene 0 has 4 rows, gene 1 has 4 rows (variable for testing → also truncate)
    lists = [np.array([0, 1, 2, 3]), np.array([4, 5])]
    b = collate_listwise(lists, **tiny_dataset)
    assert b.mode == "listwise"
    assert b.gene_idx.shape == (2, 4)   # padded to max list length
    assert b.fit.shape == (2, 4)
    assert b.mask.shape == (2, 4)
    assert b.mask[0].all().item() is True       # gene 0 fully valid
    assert b.mask[1, :2].all().item() is True   # gene 1 first 2 valid
    assert b.mask[1, 2:].any().item() is False  # gene 1 padding


# ---------------------------------------------------------------------------
# Policy hash
# ---------------------------------------------------------------------------

def test_policy_hash_deterministic(tmp_path: Path):
    p = tmp_path / "policy.yaml"
    p.write_text("policy_id: test\nm_min: 10\n")
    h1 = policy_hash(p)
    h2 = policy_hash(p)
    assert h1 == h2
    assert h1 == hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(h1) == 64


def test_policy_hash_changes_with_content(tmp_path: Path):
    p1 = tmp_path / "a.yaml"; p1.write_text("a: 1")
    p2 = tmp_path / "b.yaml"; p2.write_text("a: 2")
    assert policy_hash(p1) != policy_hash(p2)
