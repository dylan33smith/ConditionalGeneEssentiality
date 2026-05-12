"""Unit tests for the embedding NN baseline.

Uses synthetic embedding bundles written to a temp directory so the test is
fully hermetic — no dependence on real ProteomeLM data.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.nn_baseline import embedding_nn_baseline


def _write_bundle(path: Path, gene_keys: list[str], embeddings: np.ndarray) -> None:
    """Write a synthetic *_proteomelm.pt bundle in the same format as the real ones."""
    bundle = {
        "embeddings": torch.tensor(embeddings, dtype=torch.bfloat16),
        "group_labels": list(gene_keys),
    }
    torch.save(bundle, path)


@pytest.fixture
def synthetic_embeddings(tmp_path: Path) -> Path:
    """3 organisms with 2 genes each, 8-dim embeddings."""
    np.random.seed(0)
    # train: orgA, orgB
    _write_bundle(
        tmp_path / "orgA_proteomelm.pt",
        ["orgA:g1", "orgA:g2"],
        np.array([
            [1.0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1.0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.float32),
    )
    _write_bundle(
        tmp_path / "orgB_proteomelm.pt",
        ["orgB:g1", "orgB:g2"],
        np.array([
            [0, 0, 1.0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1.0, 0, 0, 0, 0],
        ], dtype=np.float32),
    )
    # val: orgC. orgC:g1 is closer to orgA:g1 (axis 0); orgC:g2 closer to orgB:g2 (axis 3)
    _write_bundle(
        tmp_path / "orgC_proteomelm.pt",
        ["orgC:g1", "orgC:g2"],
        np.array([
            [0.95, 0.05, 0, 0, 0, 0, 0, 0],   # closest to orgA:g1
            [0, 0, 0.05, 0.95, 0, 0, 0, 0],   # closest to orgB:g2
        ], dtype=np.float32),
    )
    return tmp_path


def test_nn_baseline_in_medium_path(synthetic_embeddings: Path):
    """Val medium 'M1' is in train. NN should pick the most-similar train gene
    that was seen in M1, and predict its (in-medium) mean fit."""
    train_df = pd.DataFrame({
        "orgId":    ["orgA", "orgA", "orgB", "orgB"],
        "gene_key": ["orgA:g1", "orgA:g2", "orgB:g1", "orgB:g2"],
        "media":    ["M1", "M1", "M1", "M1"],
        "fit":      [1.0,    2.0,    10.0,   20.0],
    })
    # val_org orgC not in train. Both val rows are in M1 (which IS in train).
    val_df = pd.DataFrame({
        "orgId":    ["orgC", "orgC"],
        "gene_key": ["orgC:g1", "orgC:g2"],
        "media":    ["M1",     "M1"],
        "fit":      [99.0,     99.0],   # pretend truth (irrelevant for predict)
    })
    res = embedding_nn_baseline(
        train_df, val_df,
        train_orgs=["orgA", "orgB"], val_orgs=["orgC"],
        embedding_dir=synthetic_embeddings,
    )
    # orgC:g1 → nearest train gene in M1 is orgA:g1 (fit 1.0)
    # orgC:g2 → nearest train gene in M1 is orgB:g2 (fit 20.0)
    assert np.allclose(res["predictions"], [1.0, 20.0])
    assert res["fallback_count"] == 0   # both val rows used the in-medium path
    assert res["fallback_rate"] == 0.0
    assert res["no_embedding_count"] == 0


def test_nn_baseline_fallback_path(synthetic_embeddings: Path):
    """Val medium 'M_unique' is NOT in train. NN must fall back to nearest
    train gene globally, predicting that gene's global mean fit."""
    train_df = pd.DataFrame({
        "orgId":    ["orgA", "orgA", "orgB", "orgB"],
        "gene_key": ["orgA:g1", "orgA:g2", "orgB:g1", "orgB:g2"],
        "media":    ["M1", "M1", "M1", "M1"],
        "fit":      [1.0,    2.0,    10.0,   20.0],
    })
    val_df = pd.DataFrame({
        "orgId":    ["orgC", "orgC"],
        "gene_key": ["orgC:g1", "orgC:g2"],
        "media":    ["M_unique", "M_unique"],   # NOT in train
        "fit":      [99.0,       99.0],
    })
    res = embedding_nn_baseline(
        train_df, val_df,
        train_orgs=["orgA", "orgB"], val_orgs=["orgC"],
        embedding_dir=synthetic_embeddings,
    )
    # Fallback path: nearest train gene globally → its GLOBAL mean fit
    # orgC:g1 → orgA:g1 (global mean = 1.0); orgC:g2 → orgB:g2 (global mean = 20.0)
    assert np.allclose(res["predictions"], [1.0, 20.0])
    assert res["fallback_count"] == 2
    assert res["fallback_rate"] == pytest.approx(1.0)


def test_nn_baseline_handles_val_gene_without_embedding(synthetic_embeddings: Path):
    """If a val gene_key has no embedding, the row must still get a prediction
    (defaults to global train mean) and is counted in no_embedding_count."""
    train_df = pd.DataFrame({
        "orgId":    ["orgA", "orgA"],
        "gene_key": ["orgA:g1", "orgA:g2"],
        "media":    ["M1", "M1"],
        "fit":      [1.0,    3.0],
    })
    val_df = pd.DataFrame({
        "orgId":    ["orgC", "orgC"],
        "gene_key": ["orgC:g1", "orgC:gMISSING"],  # second gene has no embedding
        "media":    ["M1", "M1"],
        "fit":      [99.0, 99.0],
    })
    res = embedding_nn_baseline(
        train_df, val_df,
        train_orgs=["orgA"], val_orgs=["orgC"],
        embedding_dir=synthetic_embeddings,
    )
    assert res["no_embedding_count"] == 1
    assert res["no_embedding_rate"] == pytest.approx(0.5)
    # First val row should predict orgA:g1's fit (1.0)
    # Second val row falls back to global train mean (= 2.0)
    assert res["predictions"][1] == pytest.approx(2.0)


def test_nn_baseline_n_rows_and_metrics_shape(synthetic_embeddings: Path):
    train_df = pd.DataFrame({
        "orgId":    ["orgA", "orgA", "orgB", "orgB"],
        "gene_key": ["orgA:g1", "orgA:g2", "orgB:g1", "orgB:g2"],
        "media":    ["M1", "M1", "M1", "M1"],
        "fit":      [1.0,    2.0,    10.0,   20.0],
    })
    val_df = pd.DataFrame({
        "orgId":    ["orgC", "orgC"],
        "gene_key": ["orgC:g1", "orgC:g2"],
        "media":    ["M1", "M1"],
        "fit":      [1.0, 20.0],
    })
    res = embedding_nn_baseline(
        train_df, val_df,
        train_orgs=["orgA", "orgB"], val_orgs=["orgC"],
        embedding_dir=synthetic_embeddings,
    )
    assert res["n_rows"] == 2
    assert res["rmse"] == pytest.approx(0.0)
    assert res["mae"] == pytest.approx(0.0)
