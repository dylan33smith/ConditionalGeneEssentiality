"""R-LOCK-4 evaluation harness for the ranking regime (v2 metric contract).

Implements `data_contract/ranking/metric_contract.yaml` (contract_id r_lock_4_v2):

  RETRIEVAL (valued above full-list correlation — "find the top stressors"):
    - ndcg_at_k, precision_at_k    (relevance = max(0, -fit))
  FULL-LIST (completeness / anti-gaming):
    - within-gene Spearman / Kendall   (from ranking_metrics.py)
  STATISTICS:
    - hierarchical_bootstrap_ci    (org -> gene; not flat)
    - benjamini_hochberg           (FDR across a tier's arms)
    - bootstrap_pvalue_delta       (one-sided p for model - baseline > 0)
  BASELINES (split-specific; primary split has COLD columns — see §2.2.1):
    - chemistry_nearest_condition_profile   (cold-condition null gate)
    - chemistry_knn_predict                 (primary competitive baseline)
  REPORTING:
    - per_organism_breakdown

Design note: chemistry baselines take a `cond_features` mapping
(condition_key -> feature vector) so they are decoupled from where the features
come from (multihot / fingerprints) and are unit-testable with synthetic data.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

log = logging.getLogger(__name__)


# ===========================================================================
# RETRIEVAL METRICS  (relevance = max(0, -fit): a stressor reduces fitness)
# ===========================================================================

def _relevance(fit: np.ndarray) -> np.ndarray:
    """Graded relevance for 'top stressors': more negative fit -> higher gain."""
    return np.maximum(0.0, -np.asarray(fit, dtype=float))


def ndcg_at_k(fit_true: np.ndarray, fit_pred: np.ndarray, k: int) -> float:
    """NDCG@k for one gene. Ranks conditions by predicted-most-essential first.

    Returns nan if the gene has no stressors (all relevance 0) or < 2 conditions.
    """
    rel = _relevance(fit_true)
    n = len(rel)
    if n < 2 or not np.any(rel > 0):
        return float("nan")
    kk = min(k, n)
    # order by predicted most-essential first = ascending predicted fit
    pred_order = np.argsort(np.asarray(fit_pred, dtype=float), kind="stable")
    dcg = float(np.sum(rel[pred_order[:kk]] / np.log2(np.arange(2, kk + 2))))
    ideal_order = np.argsort(-rel, kind="stable")
    idcg = float(np.sum(rel[ideal_order[:kk]] / np.log2(np.arange(2, kk + 2))))
    return dcg / idcg if idcg > 0 else float("nan")


def precision_at_k(fit_true: np.ndarray, fit_pred: np.ndarray, k: int) -> float:
    """Fraction of the predicted top-k that are in the TRUE top-k stressors.

    True top-k = the k conditions with highest relevance (most negative fit).
    Returns nan if no stressors or < 2 conditions.
    """
    rel = _relevance(fit_true)
    n = len(rel)
    if n < 2 or not np.any(rel > 0):
        return float("nan")
    kk = min(k, n)
    pred_top = set(np.argsort(np.asarray(fit_pred, dtype=float), kind="stable")[:kk].tolist())
    true_top = set(np.argsort(-rel, kind="stable")[:kk].tolist())
    return len(pred_top & true_top) / kk


def within_gene_retrieval(
    df: pd.DataFrame, *, k_values=(1, 3, 5),
    fit_col="fit", pred_col="pred", gene_col="gene_key",
    eligible_mask: pd.Series | None = None, min_n: int = 5,
) -> pd.DataFrame:
    """Per-gene NDCG@k and precision@k. Returns one row per gene with its org.

    Columns: gene_key, orgId (if present), ndcg_at_{k}, precision_at_{k}.
    """
    work = df if eligible_mask is None else df[eligible_mask]
    has_org = "orgId" in work.columns
    rows = []
    for gene, sub in work.groupby(gene_col, sort=False):
        if len(sub) < min_n:
            continue
        ft = sub[fit_col].to_numpy()
        fp = sub[pred_col].to_numpy()
        rec = {"gene_key": gene}
        if has_org:
            rec["orgId"] = sub["orgId"].iloc[0]
        for k in k_values:
            rec[f"ndcg_at_{k}"] = ndcg_at_k(ft, fp, k)
            rec[f"precision_at_{k}"] = precision_at_k(ft, fp, k)
        rows.append(rec)
    return pd.DataFrame(rows)


# ===========================================================================
# PER-GENE FULL-LIST CORRELATIONS (with org tag for hierarchical bootstrap)
# ===========================================================================

def per_gene_correlations(
    df: pd.DataFrame, *, metric="spearman",
    fit_col="fit", pred_col="pred", gene_col="gene_key",
    eligible_mask: pd.Series | None = None, min_n: int = 5,
) -> pd.DataFrame:
    """Per-gene Spearman/Kendall with the gene's org. One row per usable gene."""
    if metric == "spearman":
        fn = lambda a, b: spearmanr(a, b)[0]
    elif metric == "kendall":
        fn = lambda a, b: kendalltau(a, b)[0]
    else:
        raise ValueError(metric)
    work = df if eligible_mask is None else df[eligible_mask]
    has_org = "orgId" in work.columns
    rows = []
    for gene, sub in work.groupby(gene_col, sort=False):
        if len(sub) < min_n:
            continue
        yt, yp = sub[fit_col].to_numpy(), sub[pred_col].to_numpy()
        if np.all(yt == yt[0]) or np.all(yp == yp[0]):
            continue
        r = fn(yt, yp)
        if not np.isnan(r):
            rows.append({"gene_key": gene,
                         "orgId": sub["orgId"].iloc[0] if has_org else "ALL",
                         "value": float(r)})
    return pd.DataFrame(rows)


# ===========================================================================
# HIERARCHICAL (org -> gene) BOOTSTRAP
# ===========================================================================

def hierarchical_bootstrap_ci(
    per_gene: pd.DataFrame, *, value_col="value", org_col="orgId",
    n_bootstrap=1000, ci_level=0.95, seed=0,
) -> dict:
    """Two-level bootstrap CI: resample orgs w/ replacement, then genes within.

    Honest CI for clustered data (genes within an org are correlated). Returns
    {mean, ci_low, ci_high, n_genes, n_orgs, n_bootstrap}.
    """
    if per_gene.empty:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_genes": 0, "n_orgs": 0,
                "n_bootstrap": n_bootstrap}
    rng = np.random.default_rng(seed)
    orgs = per_gene[org_col].unique()
    by_org = {o: per_gene.loc[per_gene[org_col] == o, value_col].to_numpy()
              for o in orgs}
    point = float(per_gene[value_col].mean())
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        chosen = rng.choice(orgs, size=len(orgs), replace=True)
        vals = []
        for o in chosen:
            arr = by_org[o]
            if len(arr):
                vals.append(arr[rng.integers(0, len(arr), size=len(arr))])
        allv = np.concatenate(vals) if vals else np.array([np.nan])
        boot[b] = np.nanmean(allv)
    a = (1 - ci_level) / 2
    return {"mean": point,
            "ci_low": float(np.quantile(boot, a)),
            "ci_high": float(np.quantile(boot, 1 - a)),
            "n_genes": int(len(per_gene)), "n_orgs": int(len(orgs)),
            "n_bootstrap": n_bootstrap}


# ===========================================================================
# FDR (Benjamini-Hochberg) + bootstrap p-value for a delta
# ===========================================================================

def benjamini_hochberg(pvalues, alpha: float = 0.05):
    """BH-FDR. Returns (rejected: np.ndarray[bool], qvalues: np.ndarray).

    Controls the false-discovery rate at `alpha` across a family of tests
    (e.g., a tier's arms). Order-preserving: output aligns with input order.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([])
    order = np.argsort(p)
    ranked = p[order]
    # BH critical values and step-up q-values
    q_sorted = ranked * m / (np.arange(1, m + 1))
    # enforce monotonicity of q-values (from the largest down)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    # rejection: largest k with ranked[k] <= (k+1)/m * alpha
    thresh = (np.arange(1, m + 1) / m) * alpha
    below = ranked <= thresh
    if below.any():
        kmax = np.max(np.where(below)[0])
        rej_sorted = np.arange(m) <= kmax
    else:
        rej_sorted = np.zeros(m, dtype=bool)
    rejected = np.empty(m, dtype=bool)
    qvalues = np.empty(m, dtype=float)
    rejected[order] = rej_sorted
    qvalues[order] = q_sorted
    return rejected, qvalues


def bootstrap_pvalue_delta(
    model_per_gene: pd.DataFrame, baseline_per_gene: pd.DataFrame, *,
    value_col="value", org_col="orgId", n_bootstrap=1000, seed=0,
) -> float:
    """One-sided bootstrap p-value for H0: mean(model) - mean(baseline) <= 0.

    Uses the hierarchical (org->gene) scheme on the per-gene delta. Requires the
    two frames to be on the SAME genes (denominator parity); we inner-join on
    gene_key so the delta is paired per gene.
    """
    merged = model_per_gene.merge(
        baseline_per_gene[["gene_key", value_col]], on="gene_key",
        suffixes=("_m", "_b"))
    if merged.empty:
        return float("nan")
    merged["delta"] = merged[f"{value_col}_m"] - merged[f"{value_col}_b"]
    if org_col not in merged.columns:
        merged[org_col] = "ALL"
    rng = np.random.default_rng(seed)
    orgs = merged[org_col].unique()
    by_org = {o: merged.loc[merged[org_col] == o, "delta"].to_numpy() for o in orgs}
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        chosen = rng.choice(orgs, size=len(orgs), replace=True)
        vals = [by_org[o][rng.integers(0, len(by_org[o]), size=len(by_org[o]))]
                for o in chosen if len(by_org[o])]
        allv = np.concatenate(vals) if vals else np.array([0.0])
        boot[b] = np.nanmean(allv)
    # one-sided: how often is the resampled mean delta <= 0
    return float(np.mean(boot <= 0))


# ===========================================================================
# SPLIT-SPECIFIC CHEMISTRY BASELINES (primary split = cold columns)
# ===========================================================================

def _cosine_dist_matrix(val_feats: np.ndarray, train_feats: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance, rows=val conditions, cols=train conditions."""
    vn = val_feats / np.maximum(np.linalg.norm(val_feats, axis=1, keepdims=True), 1e-9)
    tn = train_feats / np.maximum(np.linalg.norm(train_feats, axis=1, keepdims=True), 1e-9)
    return 1.0 - vn @ tn.T


def chemistry_nearest_condition_profile(
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    cond_features: dict[str, np.ndarray], *,
    condition_col="condition_key", fit_col="fit",
) -> pd.Series:
    """Cold-condition NULL gate: predict a val gene's fit at held-out condition c
    as the TRAIN-GENE MEAN fit at the chemically-nearest TRAIN condition c'.

    Ignores gene identity (population profile transferred via chemistry).
    Returns predictions index-aligned to val_df.
    """
    train_conds = [c for c in train_df[condition_col].unique() if c in cond_features]
    val_conds = [c for c in val_df[condition_col].unique() if c in cond_features]
    if not train_conds or not val_conds:
        return pd.Series(np.nan, index=val_df.index)
    tfeat = np.vstack([cond_features[c] for c in train_conds])
    vfeat = np.vstack([cond_features[c] for c in val_conds])
    dist = _cosine_dist_matrix(vfeat, tfeat)
    nearest = {vc: train_conds[int(np.argmin(dist[i]))] for i, vc in enumerate(val_conds)}
    train_cond_mean = train_df.groupby(condition_col)[fit_col].mean()
    pred_for_val_cond = {vc: train_cond_mean.get(nearest[vc], np.nan)
                         for vc in val_conds}
    return val_df[condition_col].map(pred_for_val_cond)


def chemistry_knn_predict(
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    cond_features: dict[str, np.ndarray], *, k: int = 5,
    gene_col="gene_key", condition_col="condition_key", fit_col="fit",
) -> pd.Series:
    """Primary competitive baseline: predict a val gene g's fit at held-out
    condition c from g's OWN train fit at its k chemically-nearest TRAIN
    conditions. Uses warm rows bridged by chemistry. Index-aligned to val_df.

    Vectorized: loops only over val CONDITIONS (few hundred), not val rows
    (millions). Builds a gene×val_condition prediction matrix, then fancy-indexes.
    """
    train_conds = [c for c in train_df[condition_col].unique() if c in cond_features]
    val_conds = [c for c in val_df[condition_col].unique() if c in cond_features]
    if not train_conds or not val_conds:
        return pd.Series(np.nan, index=val_df.index)
    tfeat = np.vstack([cond_features[c] for c in train_conds])
    vfeat = np.vstack([cond_features[c] for c in val_conds])
    dist = _cosine_dist_matrix(vfeat, tfeat)                 # [n_val_cond, n_train_cond]
    knn_idx = np.argsort(dist, axis=1)[:, :k]                # [n_val_cond, k]

    # gene × train_condition mean-fit matrix (NaN where gene lacks a condition)
    train_lookup = (train_df.groupby([gene_col, condition_col])[fit_col]
                    .mean().unstack().reindex(columns=train_conds))
    train_mat = train_lookup.to_numpy(dtype=float)           # [n_genes, n_train_cond]
    gene_to_row = {g: i for i, g in enumerate(train_lookup.index)}

    # P[gene, val_cond_i] = nanmean over the k nearest train cols of vc_i
    n_genes = train_mat.shape[0]
    P = np.full((n_genes, len(val_conds)), np.nan)
    with np.errstate(invalid="ignore"):
        for i in range(len(val_conds)):
            sub = train_mat[:, knn_idx[i]]                   # [n_genes, k]
            allnan = np.all(np.isnan(sub), axis=1)
            col = np.nanmean(np.where(np.isnan(sub), np.nan, sub), axis=1)
            col[allnan] = np.nan
            P[:, i] = col

    val_cond_to_i = {vc: i for i, vc in enumerate(val_conds)}
    g_rows = val_df[gene_col].map(gene_to_row).to_numpy()
    c_cols = val_df[condition_col].map(val_cond_to_i).to_numpy()
    preds = np.full(len(val_df), np.nan)
    valid = ~pd.isna(g_rows) & ~pd.isna(c_cols)
    preds[valid] = P[g_rows[valid].astype(int), c_cols[valid].astype(int)]
    return pd.Series(preds, index=val_df.index)


# ===========================================================================
# PER-ORGANISM BREAKDOWN
# ===========================================================================

def per_organism_breakdown(
    per_gene_spearman: pd.DataFrame, *,
    retrieval_per_gene: pd.DataFrame | None = None,
    value_col="value", org_col="orgId",
) -> pd.DataFrame:
    """Aggregate per-gene scores to a per-org table (cheap — just a groupby).

    Exposes whether a global-mean win is driven by one large org.
    """
    g = (per_gene_spearman.groupby(org_col)[value_col]
         .agg(n_eligible_genes="count", model_spearman="mean").reset_index())
    if retrieval_per_gene is not None and "ndcg_at_5" in retrieval_per_gene.columns:
        nd = (retrieval_per_gene.groupby(org_col)["ndcg_at_5"]
              .mean().reset_index().rename(columns={"ndcg_at_5": "model_ndcg_at_5"}))
        g = g.merge(nd, on=org_col, how="left")
    return g.sort_values("n_eligible_genes", ascending=False)
