"""Integration-style unit tests for S2 `evaluate_candidate` orchestrator.

Guards wiring: train/val split, all five baselines present, power + noise blocks,
and internal consistency (e.g. global RMSE matches a direct recompute).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.null_baselines import global_train_mean_baseline
from src.experiments.stage2.baselines import evaluate_candidate


def _write_embedding_bundle(path: Path, gene_keys: list[str], vecs: np.ndarray) -> None:
    bundle = {
        "embeddings": torch.tensor(vecs, dtype=torch.bfloat16),
        "group_labels": list(gene_keys),
    }
    torch.save(bundle, path)


@pytest.fixture
def tiny_embedding_dir(tmp_path: Path) -> Path:
    """Train orgA/orgB; val orgC — same layout as test_nn_baseline."""
    _write_embedding_bundle(
        tmp_path / "orgA_proteomelm.pt",
        ["orgA:g1", "orgA:g2"],
        np.array([[1.0, 0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0, 0]], dtype=np.float32),
    )
    _write_embedding_bundle(
        tmp_path / "orgB_proteomelm.pt",
        ["orgB:g1", "orgB:g2"],
        np.array([[0, 0, 1.0, 0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0, 0, 0]], dtype=np.float32),
    )
    _write_embedding_bundle(
        tmp_path / "orgC_proteomelm.pt",
        ["orgC:g1", "orgC:g2"],
        np.array([[0.95, 0.05, 0, 0, 0, 0, 0, 0], [0, 0, 0.05, 0.95, 0, 0, 0, 0]], dtype=np.float32),
    )
    return tmp_path


def _build_fit_df() -> pd.DataFrame:
    """Train on orgA+orgB; val on orgC; test_o held out (excluded from train/val).

    Val genes need ≥5 conditions each for default power_report m=5.
    Shared expName keys c0..c5 between train and val so additive has signal.
    Train gene_keys must match synthetic embedding bundles (orgA:g1, …).
    """
    rows: list[dict] = []
    # Train: both orgs cover conditions c0..c5 with M1 (gene keys = embedding labels)
    for org, base, g1, g2 in [
        ("orgA", 0.0, "orgA:g1", "orgA:g2"),
        ("orgB", 10.0, "orgB:g1", "orgB:g2"),
    ]:
        for i in range(6):
            rows.append({
                "orgId": org,
                "gene_key": g1,
                "expName": f"c{i}",
                "media": "M1",
                "fit": base + float(i),
            })
        for i in range(6, 10):
            rows.append({
                "orgId": org,
                "gene_key": g2,
                "expName": f"c{i % 6}",
                "media": "M1",
                "fit": base + float(i) * 0.5,
            })
    # Val orgC: two genes × six conditions (>= m=5)
    for g_suffix, offset in [("g1", 0.0), ("g2", 1.0)]:
        for i in range(6):
            rows.append({
                "orgId": "orgC",
                "gene_key": f"orgC:{g_suffix}",
                "expName": f"c{i}",
                "media": "M1",
                "fit": 50.0 + offset + float(i) * 0.1,
            })
    # Test organism (held out from train; not used in val metrics)
    rows.append({
        "orgId": "test_o",
        "gene_key": "test_o:g1",
        "expName": "c0",
        "media": "M1",
        "fit": 0.0,
    })
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_candidate() -> dict:
    return {
        "protocol_id": "proto_tiny",
        "val_org_ids": ["orgC"],
        "test_org_ids": ["test_o"],
    }


def test_evaluate_candidate_split_and_structure(
    tiny_embedding_dir: Path, tiny_candidate: dict
):
    fit_df = _build_fit_df()
    out = evaluate_candidate(
        fit_df,
        tiny_candidate,
        embedding_dir=tiny_embedding_dir,
        n_bootstrap=30,
        n_permutations=8,
        additive_max_iters=50,
        additive_tol=1e-4,
    )

    assert out["protocol_id"] == "proto_tiny"
    assert out["val_org_ids"] == ["orgC"]
    assert out["test_org_ids"] == ["test_o"]

    held = {"orgC", "test_o"}
    n_train = fit_df[~fit_df["orgId"].isin(held)].shape[0]
    n_val = fit_df[fit_df["orgId"] == "orgC"].shape[0]
    assert out["n_train_rows"] == n_train
    assert out["n_val_rows"] == n_val

    for key in (
        "global_train_mean",
        "per_condition_mean",
        "per_organism_mean",
        "additive_baseline",
        "embedding_nn",
    ):
        assert key in out["baselines"]
        b = out["baselines"][key]
        assert "rmse" in b and "mae" in b
        assert b["n_rows"] == n_val
        assert np.isfinite(b["rmse"]) and np.isfinite(b["mae"])

    assert "power" in out
    assert out["power"]["m"] == 5
    assert out["power"]["spearman_role"] in ("primary", "diagnostic_only")

    nd = out["noise_diagnostics"]
    assert "additive_residual_quantiles" in nd
    assert "p50" in nd["additive_residual_quantiles"]
    por = nd["per_organism_residual"]
    assert isinstance(por, list) and len(por) == 1
    assert por[0]["orgId"] == "orgC"
    assert "std" in por[0] and "iqr" in por[0]

    assert "meaningful_gain_thresholds" in out
    assert "_predictions" in out and "_per_gene_spearman" in out


def test_evaluate_candidate_global_matches_null_module(
    tiny_embedding_dir: Path, tiny_candidate: dict
):
    fit_df = _build_fit_df()
    held = set(tiny_candidate["val_org_ids"]) | set(tiny_candidate["test_org_ids"])
    train_df = fit_df[~fit_df["orgId"].isin(held)]
    val_df = fit_df[fit_df["orgId"].isin(tiny_candidate["val_org_ids"])]
    train_fit = train_df["fit"].to_numpy()
    val_fit = val_df["fit"].to_numpy()
    direct = global_train_mean_baseline(train_fit, val_fit)

    out = evaluate_candidate(
        fit_df,
        tiny_candidate,
        embedding_dir=tiny_embedding_dir,
        n_bootstrap=20,
        n_permutations=5,
    )
    g = out["baselines"]["global_train_mean"]
    assert g["rmse"] == pytest.approx(direct["rmse"])
    assert g["mae"] == pytest.approx(direct["mae"])
    assert g["train_mean"] == pytest.approx(direct["train_mean"])


def test_evaluate_candidate_nn_matches_in_medium_path(
    tiny_embedding_dir: Path, tiny_candidate: dict
):
    """Same synthetic geometry as test_nn_baseline: NN predicts 1.0 and 20.0 on M1."""
    train_df = pd.DataFrame({
        "orgId": ["orgA", "orgA", "orgB", "orgB"],
        "gene_key": ["orgA:g1", "orgA:g2", "orgB:g1", "orgB:g2"],
        "media": ["M1", "M1", "M1", "M1"],
        "expName": ["e0", "e1", "e0", "e1"],
        "fit": [1.0, 2.0, 10.0, 20.0],
    })
    val_df = pd.DataFrame({
        "orgId": ["orgC", "orgC"],
        "gene_key": ["orgC:g1", "orgC:g2"],
        "media": ["M1", "M1"],
        "expName": ["ev0", "ev1"],
        "fit": [1.0, 20.0],
    })
    # Five extra val rows per gene so m=5 eligibility (duplicate expName pattern)
    extra_val = []
    for gene, fit0 in [("orgC:g1", 1.0), ("orgC:g2", 20.0)]:
        for j in range(5):
            extra_val.append({
                "orgId": "orgC",
                "gene_key": gene,
                "media": "M1",
                "expName": f"ex_{gene}_{j}",
                # Small spread so within-gene Spearman is defined (not constant y_true).
                "fit": fit0 + 0.01 * j + 0.001 * (j % 3),
            })
    val_df = pd.concat([val_df, pd.DataFrame(extra_val)], ignore_index=True)

    fit_all = pd.concat(
        [
            train_df,
            val_df,
            pd.DataFrame([{
                "orgId": "test_o",
                "gene_key": "x:g",
                "media": "M1",
                "expName": "e0",
                "fit": 0.0,
            }]),
        ],
        ignore_index=True,
    )
    cand = {"protocol_id": "nn_check", "val_org_ids": ["orgC"], "test_org_ids": ["test_o"]}
    out = evaluate_candidate(
        fit_all, cand, embedding_dir=tiny_embedding_dir, n_bootstrap=15, n_permutations=5
    )
    preds = out["_predictions"]["embedding_nn"]
    # First two rows of val_df were the canonical NN targets; extra rows same NN picks
    assert np.allclose(preds[:2], [1.0, 20.0])
    assert out["baselines"]["embedding_nn"]["fallback_rate"] == 0.0
