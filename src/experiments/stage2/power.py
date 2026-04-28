"""Power analysis for within-gene Spearman (REFACTORPLAN §7 S2).

Three components:
  1. Eligibility curve: n_genes_eligible vs (m, v_min) thresholds.
  2. Bootstrap CI: resample genes-with-replacement, recompute mean Spearman,
     report 95% CI.
  3. Permutation null: permute predictions, recompute Spearman, report 95th
     percentile of null.

The Spearman role per protocol is decided in S2 by:
  - if n_genes_eligible at locked m,v_min is small (<200) → diagnostic_only
  - if bootstrap CI half-width > 0.05 → diagnostic_only
  - else → primary
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


log = logging.getLogger(__name__)


def _per_gene_spearman_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gene_keys: np.ndarray,
    *,
    min_conditions: int,
    min_iqr: float | None,
) -> pd.DataFrame:
    """For each eligible gene: (gene_key, n_conditions, iqr_y_true, spearman)."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "gene": gene_keys})
    rows = []
    for gene, sub in df.groupby("gene"):
        n = len(sub)
        if n < min_conditions:
            continue
        iqr = float(np.percentile(sub["y_true"], 75) - np.percentile(sub["y_true"], 25))
        if min_iqr is not None and iqr < min_iqr:
            continue
        r, _ = spearmanr(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        if not np.isnan(r):
            rows.append({"gene": gene, "n_conditions": int(n),
                         "iqr_y_true": iqr, "spearman": float(r)})
    return pd.DataFrame(rows)


def eligibility_curve(
    y_true: np.ndarray,
    gene_keys: np.ndarray,
    *,
    m_values: Sequence[int] = (3, 5, 8, 12, 20),
) -> pd.DataFrame:
    """For each candidate m: how many genes are eligible (≥ m conditions in val)?"""
    df = pd.DataFrame({"y_true": y_true, "gene": gene_keys})
    counts = df.groupby("gene").size()
    rows = [{"m": m, "n_genes_eligible": int((counts >= m).sum())} for m in m_values]
    return pd.DataFrame(rows)


def cross_gene_iqr_p25(y_true: np.ndarray, gene_keys: np.ndarray, min_conditions: int) -> float:
    """Compute v_min = 25th percentile of cross-gene IQR on the val set.

    Pre-registered policy from REFACTORPLAN §7 S2.
    """
    df = pd.DataFrame({"y_true": y_true, "gene": gene_keys})
    iqrs = []
    for gene, sub in df.groupby("gene"):
        if len(sub) < min_conditions:
            continue
        q1, q3 = np.percentile(sub["y_true"], [25, 75])
        iqrs.append(q3 - q1)
    if not iqrs:
        return 0.0
    return float(np.percentile(iqrs, 25))


def bootstrap_spearman_ci(
    per_gene_spearman: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    ci_alpha: float = 0.05,
) -> dict:
    """Bootstrap-by-gene 95% CI for the mean within-gene Spearman."""
    if len(per_gene_spearman) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_genes_used": 0, "n_bootstrap": int(n_bootstrap)}
    vals = per_gene_spearman["spearman"].to_numpy()
    rng = np.random.default_rng(seed)
    n = len(vals)
    means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(vals, size=n, replace=True)
        means[b] = sample.mean()
    return {
        "mean": float(np.mean(vals)),
        "ci_low": float(np.percentile(means, 100 * ci_alpha / 2)),
        "ci_high": float(np.percentile(means, 100 * (1 - ci_alpha / 2))),
        "n_genes_used": int(n),
        "n_bootstrap": int(n_bootstrap),
    }


def permutation_null_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gene_keys: np.ndarray,
    *,
    min_conditions: int,
    min_iqr: float | None,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict:
    """Within each gene, permute predictions; recompute mean within-gene Spearman.

    Returns the distribution stats. The 95th percentile of the null is the
    threshold a real model must exceed to be declared above-chance.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "gene": gene_keys})
    eligible_genes = []
    eligible_subs = []
    for gene, sub in df.groupby("gene"):
        if len(sub) < min_conditions:
            continue
        if min_iqr is not None:
            q1, q3 = np.percentile(sub["y_true"], [25, 75])
            if q3 - q1 < min_iqr:
                continue
        eligible_genes.append(gene)
        eligible_subs.append(sub)
    if not eligible_subs:
        return {"null_p95": float("nan"), "null_p99": float("nan"),
                "null_mean": float("nan"), "n_genes_used": 0,
                "n_permutations": int(n_permutations)}
    null_means = np.empty(n_permutations)
    for p in range(n_permutations):
        per_gene_r = []
        for sub in eligible_subs:
            permuted = rng.permutation(sub["y_pred"].to_numpy())
            r, _ = spearmanr(sub["y_true"].to_numpy(), permuted)
            if not np.isnan(r):
                per_gene_r.append(r)
        null_means[p] = float(np.mean(per_gene_r)) if per_gene_r else float("nan")
    return {
        "null_p95": float(np.percentile(null_means, 95)),
        "null_p99": float(np.percentile(null_means, 99)),
        "null_mean": float(np.mean(null_means)),
        "null_std": float(np.std(null_means)),
        "n_genes_used": int(len(eligible_subs)),
        "n_permutations": int(n_permutations),
    }


def power_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gene_keys: np.ndarray,
    *,
    m: int = 5,
    n_bootstrap: int = 1000,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict:
    """Full power report: eligibility, v_min, bootstrap CI, permutation null."""
    elig = eligibility_curve(y_true, gene_keys)
    v_min = cross_gene_iqr_p25(y_true, gene_keys, min_conditions=m)
    per_gene = _per_gene_spearman_table(y_true, y_pred, gene_keys,
                                         min_conditions=m, min_iqr=v_min)
    boot = bootstrap_spearman_ci(per_gene, n_bootstrap=n_bootstrap, seed=seed)
    perm = permutation_null_spearman(y_true, y_pred, gene_keys,
                                      min_conditions=m, min_iqr=v_min,
                                      n_permutations=n_permutations, seed=seed)
    return {
        "m": int(m),
        "v_min": float(v_min),
        "v_min_method": "cross_gene_iqr_p25",
        "eligibility_curve": elig.to_dict(orient="records"),
        "per_gene_spearman": per_gene,                # full DataFrame
        "bootstrap_ci": boot,
        "permutation_null": perm,
    }


def decide_spearman_role(power: dict, *,
                         min_n_genes: int = 200,
                         max_ci_halfwidth: float = 0.05) -> str:
    """Decide whether Spearman is primary or diagnostic-only for this protocol."""
    n = power["bootstrap_ci"]["n_genes_used"]
    ci_low = power["bootstrap_ci"]["ci_low"]
    ci_high = power["bootstrap_ci"]["ci_high"]
    half = (ci_high - ci_low) / 2 if not (np.isnan(ci_low) or np.isnan(ci_high)) else float("inf")
    if n < min_n_genes or half > max_ci_halfwidth:
        return "diagnostic_only"
    return "primary"
