"""Integration test for condition chemistry feature loading (R-LOCK-4 baselines).

Skips when the canonical fitness table or S4 artifact is absent (CI without data).
Validated end-to-end on real data 2026-05-25 (1,058 conditions for high-rep orgs).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CANONICAL = Path("data/derived/canonical/v0/fitness_experiment_long.parquet")
S4 = Path("data_contract/preprocessing/de21504134c84a6c/experiment_chemistry.parquet")

pytestmark = pytest.mark.skipif(
    not (CANONICAL.exists() and S4.exists()),
    reason="requires canonical fitness table + S4 chemistry artifact",
)


def test_condition_chemistry_features_small_org():
    from src.data.datasets.condition_chemistry import load_condition_chemistry_features
    # A small organism keeps the test fast.
    feats = load_condition_chemistry_features(orgs=["Caulo"])
    assert len(feats) > 0
    # Each feature is a 425-dim (S4 vocab) non-negative vector with some mass.
    v = next(iter(feats.values()))
    assert v.ndim == 1 and v.shape[0] == 425
    assert (v >= 0).all() and v.sum() > 0


def test_chemistry_null_baseline_runs_on_real_split():
    """Smoke: materialize split + chemistry-null baseline produces finite preds."""
    import warnings; warnings.filterwarnings("ignore")
    from src.data.datasets.conditions import load_fitness
    from src.data.datasets.build_ranking_split import materialize_condition_holdout
    from src.data.datasets.condition_chemistry import load_condition_chemistry_features
    from src.ranking.eval import chemistry_nearest_condition_profile

    fit = load_fitness()
    fit = fit[fit["orgId"] == "Caulo"].copy()
    split = materialize_condition_holdout(fit, seed=0)
    df = fit.dropna(subset=["orgId", "gene_key", "expDesc", "media"]).copy()
    df = df.loc[split.partition.index].copy()
    df["partition"] = split.partition.values
    df["condition_key"] = split.condition_key.values
    train = df[df.partition == "train"]
    val = df[df.partition == "val"].copy()
    feats = load_condition_chemistry_features(orgs=["Caulo"])
    pred = chemistry_nearest_condition_profile(train, val, feats)
    assert pred.notna().any()                      # at least some scored
    assert np.isfinite(pred.dropna().to_numpy()).all()
