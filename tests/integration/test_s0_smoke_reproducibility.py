"""S0 acceptance gate: a fixed-seed rerun reproduces metrics within tolerance.

This test calls the S0 internals directly (not via the Hydra CLI) so we don't
depend on subprocesses or worry about Hydra working dirs. The contract being
tested: for the same seed and inputs, the smoke pipeline produces the same
metrics, the same scored_rowset_hash, and the same split_manifest_sha256.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

# Skip if any authoritative input is missing (e.g., on CI without data)
DATA_PATHS = [
    Path("data/raw/feba.db"),
    Path("data/media_composition_v4.xlsx"),
    Path("data/derived/canonical/v0/fitness_experiment_long.parquet"),
]
HAS_DATA = all(p.exists() for p in DATA_PATHS)


@pytest.mark.skipif(not HAS_DATA, reason="authoritative data inputs not present")
def test_smoke_split_is_deterministic():
    """build_smoke_dataset returns identical (train, val, manifest) for the same seed."""
    from src.experiments.stage0.run import build_smoke_dataset

    train1, val1, sha1 = build_smoke_dataset(seed=0, n_rows=8000)
    train2, val2, sha2 = build_smoke_dataset(seed=0, n_rows=8000)

    assert sha1 == sha2, "split_manifest_sha256 mismatch across reruns"
    assert len(train1) == len(train2)
    assert len(val1) == len(val2)
    # Row-for-row equality
    assert (train1.values == train2.values).all()
    assert (val1.values == val2.values).all()


@pytest.mark.skipif(not HAS_DATA, reason="authoritative data inputs not present")
def test_smoke_split_changes_with_seed():
    """Different seed → different split."""
    from src.experiments.stage0.run import build_smoke_dataset

    _, _, sha_a = build_smoke_dataset(seed=0, n_rows=8000)
    _, _, sha_b = build_smoke_dataset(seed=1, n_rows=8000)
    assert sha_a != sha_b


@pytest.mark.skipif(not HAS_DATA, reason="authoritative data inputs not present")
def test_smoke_pipeline_metrics_reproduce_within_tolerance():
    """Two runs with the same seed produce identical RMSE / MAE.

    This is the S0 acceptance gate: smoke run reproduces metrics across
    fixed-seed reruns within 1e-6 tolerance.
    """
    from src.evaluation.metrics import mae, rmse
    from src.experiments.stage0.run import build_smoke_dataset, hash_scored_rowset

    seed = 0
    n = 8000

    def run_smoke():
        train, val, split_sha = build_smoke_dataset(seed, n)
        train_mean = float(train["fit"].mean())
        y_val = val["fit"].to_numpy()
        pred = np.full(len(val), train_mean)
        return {
            "rmse": rmse(y_val, pred),
            "mae": mae(y_val, pred),
            "scored_rowset_hash": hash_scored_rowset(val),
            "split_manifest_sha": split_sha,
        }

    a = run_smoke()
    b = run_smoke()
    assert a["rmse"] == pytest.approx(b["rmse"], abs=1e-9)
    assert a["mae"] == pytest.approx(b["mae"], abs=1e-9)
    assert a["scored_rowset_hash"] == b["scored_rowset_hash"]
    assert a["split_manifest_sha"] == b["split_manifest_sha"]


@pytest.mark.skipif(not HAS_DATA, reason="authoritative data inputs not present")
def test_v4_verification_artifact_committed_or_emittable():
    """The S0-emitted v4 verification artifact, if present, has the verified status."""
    art = Path("data_contract/v4_schema_verification.json")
    if not art.exists():
        pytest.skip("v4_schema_verification.json not yet emitted; run S0 first")
    d = json.loads(art.read_text())
    assert d["status"] == "verified"
    assert d["sheet_name"] == "Media_Components_ML"
    assert d["verified_row_count"] >= 4000
