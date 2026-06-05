"""Unit tests for src/data/datasets/ranking_eligibility.py (R-LOCK-1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.datasets.ranking_eligibility import (
    compute_train_weights, per_gene_spread, tail_min_for_org, val_eligible_genes,
)

POLICY = {
    "m_min": 10, "m_min_val": 5,
    "tail_min_floor": 0.20, "tail_min_noise_coef": 0.80,
    "r_replicate_org": {"Clean": 0.90, "Noisy": 0.10},
}


def _gene_df(org, gene, fits):
    return pd.DataFrame({
        "orgId": org, "gene_key": gene,
        "condition_key": [f"c{i}" for i in range(len(fits))],
        "fit": fits,
    })


def test_tail_min_clean_vs_noisy_org():
    # Clean (r=0.90): 0.80*(1-0.90)=0.08 -> floored to 0.20
    assert tail_min_for_org("Clean", POLICY) == pytest.approx(0.20)
    # Noisy (r=0.10): 0.80*(1-0.10)=0.72
    assert tail_min_for_org("Noisy", POLICY) == pytest.approx(0.72)
    # Unknown org -> floor
    assert tail_min_for_org("Unknown", POLICY) == pytest.approx(0.20)


def test_per_gene_spread_tail_and_m():
    df = _gene_df("Clean", "g1", [-3.0, -1.0, 0.0, 0.0, 1.0, 2.0])
    out = per_gene_spread(df)
    row = out.iloc[0]
    assert row["m_g"] == 6
    # tail = p95 - p5
    expected = float(np.percentile(df["fit"], 95) - np.percentile(df["fit"], 5))
    assert row["tail_g"] == pytest.approx(expected)


def test_high_tail_gene_gets_full_weight():
    # A broadly-conditional gene with many conditions and large tail -> w_g ~ 1
    fits = list(np.linspace(-4, 4, 20))
    df = _gene_df("Clean", "g_big", fits)
    # add a second gene so tail_ref_org (p75) is defined sensibly
    df2 = _gene_df("Clean", "g_small", [0.0] * 20)
    train = pd.concat([df, df2], ignore_index=True)
    spread, gw = compute_train_weights(train, policy=POLICY)
    assert gw["g_big"] > gw["g_small"]
    assert gw["g_small"] == pytest.approx(0.0, abs=1e-6)  # flat gene -> zero weight


def test_low_condition_gene_downweighted_by_m_factor():
    # Same tail but few conditions -> m-factor < 1 reduces weight
    fits_many = list(np.linspace(-3, 3, 20))
    fits_few = list(np.linspace(-3, 3, 4))   # m=4 < m_min=10
    train = pd.concat([_gene_df("Clean", "many", fits_many),
                       _gene_df("Clean", "few", fits_few)], ignore_index=True)
    _, gw = compute_train_weights(train, policy=POLICY)
    assert gw["few"] < gw["many"]


def test_val_hard_filter_excludes_flat_and_short_genes():
    val = pd.concat([
        _gene_df("Clean", "good", list(np.linspace(-3, 3, 8))),   # big tail, m=8>=5
        _gene_df("Clean", "flat", [0.0] * 8),                      # tail 0 < 0.20
        _gene_df("Clean", "short", [-3.0, 3.0, 0.0]),              # m=3 < 5
    ], ignore_index=True)
    keep = val_eligible_genes(val, policy=POLICY)
    assert "good" in keep
    assert "flat" not in keep
    assert "short" not in keep
