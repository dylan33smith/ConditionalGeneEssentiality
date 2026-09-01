"""B (paired bootstrap) and K (ceiling denominator parity)."""
import numpy as np
import pandas as pd
import pytest

from src.ranking.eval.harness import (
    paired_hierarchical_bootstrap_ci,
    hierarchical_bootstrap_ci,
    retrieval_noise_floor,
)


def _paired_frames(n_org=6, n_gene=40, delta=0.03, gene_sd=0.25, seed=0):
    """Two methods on the SAME genes: a big shared per-gene effect + a small delta."""
    rng = np.random.default_rng(seed)
    rows_a, rows_b = [], []
    for o in range(n_org):
        for g in range(n_gene):
            shared = rng.normal(0.3, gene_sd)          # gene difficulty, shared
            rows_a.append(dict(gene_key=f"o{o}_g{g}", orgId=f"org{o}",
                               value=shared + delta + rng.normal(0, 0.01)))
            rows_b.append(dict(gene_key=f"o{o}_g{g}", orgId=f"org{o}",
                               value=shared + rng.normal(0, 0.01)))
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


def test_paired_bootstrap_detects_a_delta_that_marginal_cis_miss():
    """The whole point of B: shared per-gene variance swamps two marginal CIs."""
    a, b = _paired_frames()
    ci_a = hierarchical_bootstrap_ci(a, seed=0)
    ci_b = hierarchical_bootstrap_ci(b, seed=0)
    marginal_disjoint = ci_a["ci_low"] > ci_b["ci_high"]

    paired = paired_hierarchical_bootstrap_ci(a, b, seed=0)

    assert not marginal_disjoint, "fixture should have overlapping marginal CIs"
    assert paired["excludes_zero"], (
        "the paired delta CI must exclude zero where the per-gene difference is "
        "consistent, even though the marginal CIs overlap")
    assert paired["mean_delta"] == pytest.approx(0.03, abs=0.01)
    assert paired["p_two_sided"] < 0.05


def test_paired_bootstrap_reports_no_effect_when_there_is_none():
    a, b = _paired_frames(delta=0.0)
    paired = paired_hierarchical_bootstrap_ci(a, b, seed=0)
    assert not paired["excludes_zero"]
    assert paired["ci_low"] < 0.0 < paired["ci_high"]


def test_paired_bootstrap_uses_only_the_common_gene_set():
    a, b = _paired_frames()
    b_short = b.iloc[: len(b) // 2]
    paired = paired_hierarchical_bootstrap_ci(a, b_short, seed=0)
    assert paired["n_genes"] == len(b_short)


def test_paired_bootstrap_is_empty_safe():
    a, b = _paired_frames()
    empty = b.iloc[:0]
    out = paired_hierarchical_bootstrap_ci(a, empty, seed=0)
    assert out["n_genes"] == 0 and not out["excludes_zero"]


# --------------------------------------------------------------------------
# K -- the ceiling must honour the same denominator as the methods
# --------------------------------------------------------------------------

def _replicate_rows(n_gene=30, n_cond=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_gene):
        # half the genes are high-spread (the kind eligibility keeps)
        scale = 1.0 if g % 2 == 0 else 0.05
        for c in range(n_cond):
            true = rng.normal(0, scale)
            for rep, exp in enumerate(["A", "B"]):
                rows.append(dict(orgId="org0", gene_key=f"g{g}",
                                 condition_key=f"c{c}",
                                 expName=f"{exp}{c}",
                                 fit=true + rng.normal(0, 0.05)))
    return pd.DataFrame(rows)


def test_ceiling_respects_the_eligible_gene_set():
    df = _replicate_rows()
    eligible = [f"g{g}" for g in range(30) if g % 2 == 0]

    unfiltered = retrieval_noise_floor(df, k_values=(5,))
    filtered = retrieval_noise_floor(df, k_values=(5,), eligible_genes=eligible)

    assert unfiltered["n_genes_used"] > filtered["n_genes_used"], (
        "the parity-filtered ceiling must be computed on fewer genes")
    assert filtered["n_genes_used"] == len(eligible)
    assert unfiltered["parity_filtered"] is False
    assert filtered["parity_filtered"] is True
    # high-spread genes replicate better, so the parity ceiling is the higher one --
    # this is exactly why quoting the unfiltered value against a model is wrong
    assert filtered["ndcg_at_5"] > unfiltered["ndcg_at_5"]


def test_ceiling_can_return_per_gene_values_for_a_bootstrap():
    df = _replicate_rows()
    out, per_gene = retrieval_noise_floor(df, k_values=(5,), return_per_gene=True)
    assert len(per_gene) == out["n_genes_used"]
    assert {"gene_key", "orgId", "ndcg_at_5"} <= set(per_gene.columns)
    assert per_gene["ndcg_at_5"].mean() == pytest.approx(out["ndcg_at_5"], abs=1e-9)
