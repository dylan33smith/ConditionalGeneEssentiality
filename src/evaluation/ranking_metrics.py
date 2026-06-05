"""Ranking metrics for the R-regime (R-LOCK-4).

Implements the metric contract `data_contract/ranking/metric_contract.yaml`:

  - within_gene_spearman_mean: PRIMARY. Mean Spearman across eligible val
    genes, with bootstrap CI over genes (n=1000, 95% percentile).
  - within_gene_kendall_mean: secondary co-primary. Same shape.
  - per_condition_train_mean_baseline: H-RANK-01 baseline. Any model that
    loses to this on within_gene_spearman_mean is ineligible for promotion.
  - task_relevant_noise_floor: per-gene cross-condition Spearman across
    replicate pairs of fit (median over eligible val genes). The PRIMARY
    noise floor — the actual upper bound on within_gene_spearman.
  - cross_gene_within_condition_noise_proxy: SECONDARY noise diagnostic.

Replicate handling on val: rows are expected to already be mean-pooled per
(orgId, gene_key, condition_key) before being passed in. The noise-floor
computation reads pre-pooled replicate rows separately.

See R-LOCK-4-DEC-001.md for the full decision rationale.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core helper: per-gene rank metric with bootstrap CI
# ---------------------------------------------------------------------------

@dataclass
class BootstrapMetric:
    """Result of a per-gene metric aggregated with a bootstrap CI."""
    mean: float
    ci_low: float
    ci_high: float
    n_genes_used: int           # genes that contributed a non-nan value
    n_bootstrap: int

    def to_dict(self) -> dict:
        return {
            "mean": float(self.mean),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "n_genes_used": int(self.n_genes_used),
            "n_bootstrap": int(self.n_bootstrap),
        }


def _per_gene_correlations(
    df: pd.DataFrame, *,
    fit_col: str, pred_col: str, gene_col: str,
    eligible_mask: pd.Series | None,
    corr_fn: Callable[[np.ndarray, np.ndarray], float],
    min_n: int,
) -> dict[str, float]:
    """Compute per-gene correlation (Spearman or Kendall) on eligible genes."""
    work = df if eligible_mask is None else df[eligible_mask]
    out: dict[str, float] = {}
    for gene, sub in work.groupby(gene_col, sort=False):
        if len(sub) < min_n:
            continue
        yt = sub[fit_col].to_numpy()
        yp = sub[pred_col].to_numpy()
        # Skip constants — correlation undefined
        if np.all(yt == yt[0]) or np.all(yp == yp[0]):
            continue
        r = corr_fn(yt, yp)
        if not np.isnan(r):
            out[str(gene)] = float(r)
    return out


def _bootstrap_ci(values: list[float], *, n_boot: int, ci_level: float,
                  seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = arr[idx].mean()
    alpha = (1 - ci_level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def within_gene_rank_metric(
    df: pd.DataFrame, *,
    metric: str = "spearman",                 # "spearman" or "kendall"
    eligible_mask: pd.Series | None = None,   # boolean; True = include in metric
    fit_col: str = "fit",
    pred_col: str = "pred",
    gene_col: str = "gene_key",
    min_n: int = 5,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    bootstrap_seed: int = 0,
) -> BootstrapMetric:
    """PRIMARY metric helper.

    Per-gene correlation, then mean across genes, with bootstrap CI over genes.
    Requires denominator parity: model & baseline must be scored on the same
    `eligible_mask` so the per-gene sets are identical.
    """
    if metric == "spearman":
        def fn(a, b):
            r, _ = spearmanr(a, b)
            return r
    elif metric == "kendall":
        def fn(a, b):
            r, _ = kendalltau(a, b)
            return r
    else:
        raise ValueError(f"metric must be 'spearman' or 'kendall', got {metric!r}")

    per_gene = _per_gene_correlations(
        df, fit_col=fit_col, pred_col=pred_col, gene_col=gene_col,
        eligible_mask=eligible_mask, corr_fn=fn, min_n=min_n,
    )
    values = list(per_gene.values())
    if not values:
        return BootstrapMetric(float("nan"), float("nan"), float("nan"), 0, n_bootstrap)
    mean_val = float(np.mean(values))
    ci_lo, ci_hi = _bootstrap_ci(values, n_boot=n_bootstrap, ci_level=ci_level,
                                  seed=bootstrap_seed)
    return BootstrapMetric(mean_val, ci_lo, ci_hi, len(values), n_bootstrap)


# ---------------------------------------------------------------------------
# H-RANK-01 baseline: per-condition train mean
# ---------------------------------------------------------------------------

def per_condition_train_mean_predictions(
    train_df: pd.DataFrame, val_df: pd.DataFrame, *,
    condition_col: str = "condition_key", fit_col: str = "fit",
) -> pd.Series:
    """For each val row, return the train-mean of `fit` at that condition_key.

    Conditions absent from train (shouldn't happen under condition-holdout
    by construction, but defensive) get `nan` predictions.
    """
    means = train_df.groupby(condition_col)[fit_col].mean()
    return val_df[condition_col].map(means)


def ranking_baseline_metrics(
    val_df: pd.DataFrame, train_df: pd.DataFrame, *,
    eligible_mask: pd.Series | None = None,
    n_bootstrap: int = 1000, ci_level: float = 0.95, bootstrap_seed: int = 0,
) -> dict:
    """Compute H-RANK-01 baseline Spearman + Kendall under metric contract."""
    work = val_df.copy()
    work["_baseline_pred"] = per_condition_train_mean_predictions(
        train_df, val_df, condition_col="condition_key", fit_col="fit",
    )
    sp = within_gene_rank_metric(
        work, metric="spearman", eligible_mask=eligible_mask,
        pred_col="_baseline_pred",
        n_bootstrap=n_bootstrap, ci_level=ci_level, bootstrap_seed=bootstrap_seed,
    )
    kd = within_gene_rank_metric(
        work, metric="kendall", eligible_mask=eligible_mask,
        pred_col="_baseline_pred",
        n_bootstrap=n_bootstrap, ci_level=ci_level, bootstrap_seed=bootstrap_seed,
    )
    return {
        "baseline_id": "per_condition_train_mean",
        "spearman_mean": sp.mean,
        "kendall_mean":  kd.mean,
        "spearman_bootstrap": sp.to_dict(),
        "kendall_bootstrap":  kd.to_dict(),
    }


def model_beats_baseline(model_metric: BootstrapMetric,
                         baseline_metric: BootstrapMetric) -> bool:
    """H-RANK-01 promotion gate.

    True iff model mean > baseline mean AND the 95% CIs are disjoint
    (model.ci_low > baseline.ci_high).
    """
    if np.isnan(model_metric.mean) or np.isnan(baseline_metric.mean):
        return False
    return (model_metric.mean > baseline_metric.mean
            and model_metric.ci_low > baseline_metric.ci_high)


# ---------------------------------------------------------------------------
# Noise floor — PRIMARY (task-relevant)
# ---------------------------------------------------------------------------

def task_relevant_noise_floor(
    val_rows_pre_pool: pd.DataFrame, *,
    orgId_col: str = "orgId",
    gene_col: str = "gene_key",
    condition_col: str = "condition_key",
    expName_col: str = "expName",
    fit_col: str = "fit",
    min_conditions: int = 5,
    eligible_genes: Iterable[str] | None = None,
) -> dict:
    """Per-gene cross-condition Spearman across replicate pairs.

    For each eligible val gene with ≥ 2 replicate `expName`s at ≥ `min_conditions`
    distinct `condition_key`s:
      - Pair up the first two replicates per condition
      - rep_A[c] = fit from expName_a at condition c
      - rep_B[c] = fit from expName_b at condition c
      - Spearman across conditions for this gene
    Median across genes is the noise floor.
    """
    df = val_rows_pre_pool.dropna(subset=[orgId_col, gene_col, condition_col,
                                          expName_col, fit_col]).copy()
    # Median over any (gene, condition, expName) duplicates (rare).
    df = (df.groupby([orgId_col, gene_col, condition_col, expName_col])[fit_col]
          .median().reset_index())
    if eligible_genes is not None:
        df = df[df[gene_col].isin(set(eligible_genes))]

    per_gene_r: list[float] = []
    n_skipped_too_few_pairs = 0
    for (org, gene), g in df.groupby([orgId_col, gene_col], sort=False):
        # For each condition, take the first two expNames in stable order.
        pairs_a: list[float] = []
        pairs_b: list[float] = []
        for cond, gc in g.groupby(condition_col):
            expnames = sorted(gc[expName_col].unique())
            if len(expnames) < 2:
                continue
            a = float(gc.loc[gc[expName_col] == expnames[0], fit_col].iloc[0])
            b = float(gc.loc[gc[expName_col] == expnames[1], fit_col].iloc[0])
            pairs_a.append(a); pairs_b.append(b)
        if len(pairs_a) < min_conditions:
            n_skipped_too_few_pairs += 1
            continue
        if np.all(np.asarray(pairs_a) == pairs_a[0]) or np.all(np.asarray(pairs_b) == pairs_b[0]):
            continue
        r, _ = spearmanr(np.asarray(pairs_a), np.asarray(pairs_b))
        if not np.isnan(r):
            per_gene_r.append(float(r))

    if not per_gene_r:
        return {
            "primary_value": float("nan"),
            "n_genes_used": 0,
            "n_skipped_too_few_pairs": int(n_skipped_too_few_pairs),
        }
    return {
        "primary_value": float(np.median(per_gene_r)),
        "n_genes_used": int(len(per_gene_r)),
        "n_skipped_too_few_pairs": int(n_skipped_too_few_pairs),
        "per_gene_mean": float(np.mean(per_gene_r)),
    }


def cross_gene_within_condition_noise_proxy(
    val_rows_pre_pool: pd.DataFrame, *,
    orgId_col: str = "orgId",
    gene_col: str = "gene_key",
    condition_col: str = "condition_key",
    expName_col: str = "expName",
    fit_col: str = "fit",
    min_genes_per_pair: int = 20,
) -> dict:
    """SECONDARY: cross-gene Spearman within a single condition between
    two replicate expNames (matches R0 fig 04 definition).

    For each (org, condition_key) with ≥ 2 expNames:
      - rep_a[g] = fit values of expName_a across genes
      - rep_b[g] = fit values of expName_b across genes
      - Spearman across genes for this condition
    Median across (org, condition) pairs is the proxy.
    """
    df = val_rows_pre_pool.dropna(subset=[gene_col, condition_col, expName_col, fit_col])
    rs: list[float] = []
    for (_org, cond), g in df.groupby([orgId_col, condition_col]):
        expnames = sorted(g[expName_col].unique())
        if len(expnames) < 2:
            continue
        a, b = expnames[0], expnames[1]
        wide = (g[g[expName_col].isin([a, b])]
                .pivot_table(index=gene_col, columns=expName_col,
                              values=fit_col, aggfunc="median"))
        wide = wide.dropna()
        if len(wide) < min_genes_per_pair:
            continue
        r, _ = spearmanr(wide[a].to_numpy(), wide[b].to_numpy())
        if not np.isnan(r):
            rs.append(float(r))
    if not rs:
        return {"proxy_value": float("nan"), "n_pairs_used": 0}
    return {"proxy_value": float(np.median(rs)), "n_pairs_used": int(len(rs))}
