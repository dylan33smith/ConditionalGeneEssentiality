"""Pure analysis functions for R0 (no plotting, no figure I/O).

Each function returns a DataFrame / dict / numpy array that figures.py and
candidates.py consume. Mirrors the layout of src/experiments/stage1/analyses.py.

All metrics are computed at the (gene_key, condition) granularity where a
"condition" is an experiment (`expName`). Replicate handling is deferred to
R-LOCK-3; for R0 each row is treated as an independent observation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Condition identity + fitness loading now live in the data layer (single source
# of truth, used by the data modules and the ranking pipeline alike).
from src.data.datasets.conditions import (
    CANONICAL_DIR, _FITNESS_COLS, _STRING_KEY_COLS,
    _normalize_string_keys, _condition_key, load_fitness)


# ---------------------------------------------------------------------------
# A: Conditions per gene
# ---------------------------------------------------------------------------

def conditions_per_gene(fit_df: pd.DataFrame) -> pd.DataFrame:
    """For each (orgId, gene_key): count distinct conditions.

    A "condition" = `(expDesc, media, temperature)` replicate group (see
    `_condition_key` for rationale). Counting distinct `expName` would
    overstate by replicate factor; counting `(expDesc, media)` alone would
    miss the temperature-variant case (R0 audit 2.4% of groups).
    """
    df = fit_df.dropna(subset=["gene_key", "expDesc", "media"]).copy()
    df["condition_key"] = _condition_key(df)
    out = (df.groupby(["orgId", "gene_key"])["condition_key"]
           .nunique()
           .reset_index(name="m_g"))
    return out


# ---------------------------------------------------------------------------
# B: IQR per gene
# ---------------------------------------------------------------------------

def iqr_per_gene(fit_df: pd.DataFrame) -> pd.DataFrame:
    """For each (orgId, gene_key): IQR and MAD of fit across conditions.

    A "condition" here is the `_condition_key` = `(expDesc, media, temperature)`
    replicate group, so multiple replicate `expName`s collapse to one
    observation via median. This avoids
    inflating IQR by counting replicate variance as cross-condition variance.
    """
    needed = {"orgId", "gene_key", "expDesc", "media", "fit", "temperature"}
    missing = needed - set(fit_df.columns)
    if missing:
        raise KeyError(f"iqr_per_gene needs columns {missing}")
    sub = fit_df.dropna(subset=["gene_key", "expDesc", "media", "fit"]).copy()
    sub["condition_key"] = _condition_key(sub)
    df = (sub.groupby(["orgId", "gene_key", "condition_key"])["fit"]
          .median()
          .reset_index())
    rows = []
    for (org, gene), g in df.groupby(["orgId", "gene_key"]):
        vals = g["fit"].to_numpy()
        p5, q25, q75, p95 = np.percentile(vals, [5, 25, 75, 95])
        rows.append({
            "orgId": org, "gene_key": gene, "m_g": int(len(vals)),
            "median_fit": float(np.median(vals)),
            "q25": float(q25), "q75": float(q75),
            "p5": float(p5), "p95": float(p95),
            "iqr_g": float(q75 - q25),
            # PRIMARY eligibility metric per R-LOCK-1 (p95−p5 instead of IQR):
            # catches sparsely-conditional genes (a gene essential in 3-5 of
            # 100 conditions has IQR ≈ 0 but tail_g captures it). IQR retained
            # as a diagnostic — high IQR_g + high tail_g → broadly conditional;
            # low IQR_g + high tail_g → sparsely conditional.
            "tail_g": float(p95 - p5),
            "mad": float(np.median(np.abs(vals - np.median(vals)))),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# D: Replicate noise floor
# ---------------------------------------------------------------------------

def replicate_noise_floor(fit_df: pd.DataFrame,
                          max_orgs: int | None = None) -> pd.DataFrame:
    """Per-org cross-replicate Spearman.

    Biological replicates in FEBA are NOT rows-within-an-expName (each expName
    is a single assay). Replicates share the `_condition_key`
    `(orgId, expDesc, media, temperature)` across different `expName`s. We group
    by that key, pick the first two distinct
    expNames in each replicate group, then for that pair compute the
    cross-gene Spearman of fit (one Spearman per replicate pair, not per
    gene).

    Returns long-form: (orgId, expDesc, media, expName_a, expName_b,
    n_genes, spearman).
    """
    needed = {"orgId", "expDesc", "media", "expName", "gene_key", "fit", "temperature"}
    missing = needed - set(fit_df.columns)
    if missing:
        raise KeyError(f"replicate_noise_floor needs columns {missing} that are absent")

    df = fit_df.dropna(subset=["gene_key", "expName", "expDesc", "media", "fit"]).copy()
    df["condition_key"] = _condition_key(df)
    # Median over duplicate (org, exp, gene) rows (rare but safe).
    df = (df.groupby(["orgId", "condition_key", "expName", "gene_key"])["fit"]
          .median().reset_index())

    orgs = sorted(df["orgId"].unique())
    if max_orgs is not None:
        orgs = orgs[:max_orgs]

    out = []
    for org in orgs:
        sub = df[df["orgId"] == org]
        # Replicate group = (orgId, condition_key) with ≥2 expNames
        for cond, g in sub.groupby("condition_key"):
            expnames = sorted(g["expName"].unique())
            if len(expnames) < 2:
                continue
            a, b = expnames[0], expnames[1]
            wide = (g[g["expName"].isin([a, b])]
                    .pivot_table(index="gene_key", columns="expName",
                                  values="fit", aggfunc="median"))
            wide = wide.dropna()
            if len(wide) < 20:  # need enough genes for a meaningful Spearman
                continue
            r, _ = spearmanr(wide[a].to_numpy(), wide[b].to_numpy())
            if np.isnan(r):
                continue
            out.append({
                "orgId": org, "condition_key": cond,
                "expName_a": a, "expName_b": b,
                "n_genes": int(len(wide)), "spearman": float(r),
            })
    return pd.DataFrame(out)


def per_org_noise_summary(noise_df: pd.DataFrame) -> pd.DataFrame:
    """Per-org summary statistics of cross-replicate Spearman.

    Note: cross-replicate Spearman in this module is the EXPERIMENT-level
    correlation (how well two replicate experiments agree on ranking
    *genes*). The within-gene-across-conditions ceiling (the actual upper
    bound on our R-regime primary metric) requires the split-locked val
    rows and is computed in R-LOCK-4, not R0.
    """
    if noise_df.empty:
        return pd.DataFrame()
    return (noise_df.groupby("orgId")["spearman"]
            .agg(count="count", median="median", mean="mean", std="std")
            .reset_index())


# ---------------------------------------------------------------------------
# E: Signal-to-noise ratio
# ---------------------------------------------------------------------------

def signal_to_noise(iqr_df: pd.DataFrame, noise_summary: pd.DataFrame) -> pd.DataFrame:
    """Per-(org, gene) ratio of spread metric to that org's median replicate noise.

    Returns the input table with TWO new columns (the DENOMINATOR
    `(1 − r_replicate_org_median)` is floored at 0.05 to avoid blow-up for
    very-clean orgs; the SNR value itself is not clipped):
      - `snr_tail`: tail_g / max(1 − r_replicate_org_median, 0.05)
        (PRIMARY — matches R-LOCK-1's tail-based eligibility)
      - `snr_iqr`: iqr_g / max(1 − r_replicate_org_median, 0.05)
        (DIAGNOSTIC — for comparison with legacy decisions)

    Caveat: `r_replicate_org_median` here measures cross-gene Spearman
    within-condition between replicate assays. That's a PROXY for the
    task-relevant noise floor (per-gene cross-condition Spearman across
    replicate pairs); R-LOCK-4 computes the task-relevant version.
    """
    if noise_summary.empty:
        return iqr_df.assign(snr_tail=np.nan, snr_iqr=np.nan)
    noise_lookup = dict(zip(noise_summary["orgId"], noise_summary["median"]))
    snr_tail, snr_iqr = [], []
    for _, row in iqr_df.iterrows():
        org_noise_r = noise_lookup.get(row["orgId"])
        if org_noise_r is None or np.isnan(org_noise_r):
            snr_tail.append(np.nan); snr_iqr.append(np.nan)
            continue
        denom = max(1.0 - float(org_noise_r), 0.05)
        snr_tail.append(float(row["tail_g"]) / denom)
        snr_iqr.append(float(row["iqr_g"]) / denom)
    out = iqr_df.copy()
    out["snr_tail"] = snr_tail
    out["snr_iqr"] = snr_iqr
    return out


# ---------------------------------------------------------------------------
# F: Eligibility frontier
# ---------------------------------------------------------------------------

def eligibility_frontier(iqr_df: pd.DataFrame,
                         m_grid: list[int],
                         tail_grid: list[float],
                         metric: str = "tail_g") -> pd.DataFrame:
    """For each (orgId, m_min, threshold): count of eligible genes and fraction.

    `metric` selects which per-gene spread statistic to gate on:
      - "tail_g" (PRIMARY, locked in R-LOCK-1): p95 − p5 of fit across
        conditions; catches sparsely-conditional genes that IQR misses.
      - "iqr_g" (DIAGNOSTIC only): for legacy / comparison purposes.

    Long-form: (orgId, m_min, threshold, metric, n_eligible, n_total, frac_eligible).
    """
    if metric not in {"tail_g", "iqr_g"}:
        raise ValueError(f"metric must be tail_g or iqr_g, got {metric!r}")
    rows = []
    for org, sub in iqr_df.groupby("orgId"):
        n_total = len(sub)
        for m_min in m_grid:
            for thr in tail_grid:
                mask = (sub["m_g"] >= m_min) & (sub[metric] >= thr)
                n_elig = int(mask.sum())
                rows.append({"orgId": org, "m_min": m_min, "threshold": thr,
                             "metric": metric,
                             "n_eligible": n_elig, "n_total": n_total,
                             "frac_eligible": n_elig / n_total if n_total else 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# G: Experiment structure
# ---------------------------------------------------------------------------

def experiment_structure(fit_df: pd.DataFrame) -> pd.DataFrame:
    """Per-org: experiments, distinct conditions, replicate depth.

    WHY THIS ANALYSIS MATTERS
    -------------------------
    The ranking task ranks **distinct conditions** within a gene. But the raw
    data is at the **`expName`** (assay) level, and many assays are replicates
    of the same condition. Three downstream decisions depend on understanding
    this structure:

      (a) **Split protocol (R-LOCK-2):** if we hold out random `expName`s,
          ~half the held-out assays will have a replicate sibling in train
          (silently leaking the val condition into training). The split MUST
          hold out at the condition level — i.e. (expDesc, media, temperature) — and pull
          all replicate assays of a held-out condition into val together.
          This analysis quantifies the replicate factor (`exp_per_cond_p50`)
          and the fraction of conditions where this matters
          (`frac_cond_with_replicates`).

      (b) **Replicate handling (R-LOCK-3):** training can treat each replicate
          as an independent noise sample of the same target; val must
          mean-pool replicates before computing per-gene Spearman, or the
          metric is inflated by intra-condition noise. The choice depends on
          how prevalent replicates are.

      (c) **Eligibility (R-LOCK-1):** the per-gene "conditions per gene"
          metric `m_g` is at the condition level (post-replicate-collapse),
          NOT the expName level. Without this analysis we'd inadvertently
          treat replicate count as condition count, inflating m_g and
          under-filtering.

    WHAT IS COMPUTED (per org)
    --------------------------
    - `n_experiments`: distinct `expName`s. The raw assay count; bigger = more
      observations, but doesn't tell you how many distinct conditions were tested.
    - `n_conditions`: distinct `(expDesc, media, temperature)` keys. **The
      granularity the ranking task cares about.** The ratio `n_experiments / n_conditions`
      is the replicate factor (e.g. DvH: 757/248 ≈ 3.0 — every condition was
      run ~3 times on average).
    - `n_media`: distinct media base names. A coarser scale than `condition`
      because one `media` can pair with multiple `expDesc` (stressor) values.
      Reported for sanity-checking against the workbook's media inventory.
    - `exp_per_cond_p50`, `p90`, `max`: distribution of replicate depth across
      conditions. p50=1 means most conditions are run once (no replicates);
      p50≥2 means replication is the norm. Drives the urgency of point (a).
    - `frac_cond_with_replicates`: fraction of conditions where
      `exp_per_cond ≥ 2`. If this is high (e.g. DvH 0.99, Caulo 0.98),
      replicate-group-together holdout is **mandatory** to avoid leakage.
      If low (e.g. Miya 0.22), most conditions are singletons and the
      constraint is mostly a no-op for those orgs.
    """
    df = fit_df.dropna(subset=["orgId", "expName", "expDesc", "media"]).copy()
    df["condition_key"] = _condition_key(df)
    summary = []
    for org, sub in df.groupby("orgId"):
        n_exp = sub["expName"].nunique()
        n_media = sub["media"].nunique()
        # Conditions = distinct (expDesc, media, temperature) — see _condition_key
        n_cond = sub["condition_key"].nunique()
        # Replicate depth: assays per condition
        rep_per_cond = (sub.drop_duplicates(subset=["condition_key", "expName"])
                        .groupby("condition_key").size())
        summary.append({
            "orgId": org, "n_experiments": int(n_exp), "n_conditions": int(n_cond),
            "n_media": int(n_media),
            "exp_per_cond_p50": float(np.percentile(rep_per_cond, 50)) if len(rep_per_cond) else np.nan,
            "exp_per_cond_p90": float(np.percentile(rep_per_cond, 90)) if len(rep_per_cond) else np.nan,
            "exp_per_cond_max": int(rep_per_cond.max()) if len(rep_per_cond) else 0,
            "frac_cond_with_replicates": float((rep_per_cond >= 2).mean()) if len(rep_per_cond) else 0.0,
        })
    return pd.DataFrame(summary).sort_values("n_experiments", ascending=False)


# ---------------------------------------------------------------------------
# K: expGroup coverage (NEW — for R-LOCK-2 stratification)
# ---------------------------------------------------------------------------

def expgroup_coverage(fit_df: pd.DataFrame) -> pd.DataFrame:
    """Per-org expGroup distribution.

    Returns long-form (orgId, expGroup, n_experiments, frac_of_org_experiments).
    Tells us whether stratifying val by expGroup is feasible:
      - if an org has only 1 distinct expGroup, stratification is a no-op
      - if it has many small expGroups, stratification may force val to oversample
        rare groups (or undersample, depending on scheme)
    """
    df = fit_df.dropna(subset=["orgId", "expName"]).copy()
    # Replace null expGroup with sentinel so we can still count
    if "expGroup" not in df.columns:
        raise KeyError("fit_df is missing expGroup")
    df["expGroup"] = df["expGroup"].fillna("<unknown>")
    rows = []
    for org, sub in df.groupby("orgId"):
        per_exp = sub.drop_duplicates(subset=["expName"])
        total = len(per_exp)
        for g, gg in per_exp.groupby("expGroup"):
            rows.append({
                "orgId": org, "expGroup": str(g),
                "n_experiments": int(len(gg)),
                "frac_of_org_experiments": float(len(gg) / total) if total else 0.0,
            })
    return pd.DataFrame(rows)


def expgroup_summary(coverage: pd.DataFrame, *,
                     min_holdout_per_group: int = 2) -> pd.DataFrame:
    """Per-org summary: number of distinct groups, smallest group size,
    and whether stratification at the 20% holdout level is *feasible*
    (each group must have ≥ min_holdout_per_group experiments to be held out).
    """
    rows = []
    for org, sub in coverage.groupby("orgId"):
        n_groups = sub["expGroup"].nunique()
        smallest_group = int(sub["n_experiments"].min())
        # Feasible if every non-trivial group can supply at least
        # min_holdout_per_group val items at the 20% rate
        # i.e. group_size * 0.2 >= min_holdout_per_group
        # rearranged: group_size >= 5 * min_holdout_per_group
        threshold = int(5 * min_holdout_per_group)
        n_groups_passing = int((sub["n_experiments"] >= threshold).sum())
        rows.append({
            "orgId": org, "n_expgroups": int(n_groups),
            "smallest_group_size": smallest_group,
            "n_groups_passing_strat_threshold": n_groups_passing,
            "stratification_feasible": bool(n_groups >= 2 and n_groups_passing >= 2),
        })
    return pd.DataFrame(rows).sort_values("n_expgroups", ascending=False)


# ---------------------------------------------------------------------------
# H: Universal vs org-specific low-IQR genes
# ---------------------------------------------------------------------------

def low_iqr_overlap(iqr_df: pd.DataFrame, decile: float = 0.1) -> tuple[pd.DataFrame, dict]:
    """Jaccard similarity of bottom-decile-IQR gene sets across organisms.

    Returns (jaccard_matrix, per_org_low_set).
    """
    per_org_low = {}
    for org, sub in iqr_df.groupby("orgId"):
        if len(sub) < 50:  # too few genes to identify a meaningful tail
            continue
        thr = float(np.quantile(sub["iqr_g"], decile))
        per_org_low[org] = set(sub[sub["iqr_g"] <= thr]["gene_key"].tolist())
    orgs = sorted(per_org_low.keys())
    n = len(orgs)
    mat = np.zeros((n, n))
    for i, a in enumerate(orgs):
        for j, b in enumerate(orgs):
            if i == j:
                mat[i, j] = 1.0
                continue
            sa, sb = per_org_low[a], per_org_low[b]
            inter = len(sa & sb)
            union = len(sa | sb)
            mat[i, j] = inter / union if union else 0.0
    jaccard = pd.DataFrame(mat, index=orgs, columns=orgs)
    return jaccard, per_org_low


# ---------------------------------------------------------------------------
# I: Condition discriminability
# ---------------------------------------------------------------------------

def condition_discriminability(fit_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(org, condition): IQR of fit across genes, where condition =
    `_condition_key` = (expDesc, media, temperature).

    High IQR = condition spreads genes apart = useful for ranking signal.
    Low IQR = condition produces ~uniform fit = uninformative.
    """
    df = fit_df.dropna(subset=["gene_key", "expDesc", "media", "fit"]).copy()
    df["condition_key"] = _condition_key(df)
    rows = []
    for (org, cond), g in df.groupby(["orgId", "condition_key"]):
        vals = g["fit"].to_numpy()
        rows.append({
            "orgId": org, "condition_key": cond, "n_genes": int(len(vals)),
            "cond_iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "cond_mad": float(np.median(np.abs(vals - np.median(vals)))),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# J: Split feasibility preview
# ---------------------------------------------------------------------------

def split_feasibility(fit_df: pd.DataFrame,
                      holdout_fractions: list[float],
                      *,
                      seed: int = 0,
                      min_holdout: int = 3,
                      max_holdout: int = 30,
                      replicate_group_together: bool = True) -> pd.DataFrame:
    """Simulate fraction-based per-org holdouts; report per-gene val-condition stats.

    Per R-LOCK-2 proposal:
      - Holdout unit = condition key `(expDesc, media, temperature)`. All
        `expName`s sharing a condition key are held together (no replicate leakage).
      - Holdout size = `holdout_fractions × n_conditions`, clipped to
        `[min_holdout, max_holdout]`.

    Returns long-form: (orgId, holdout_fraction, n_train_conditions,
    n_val_conditions, n_val_exps, n_genes_with_any_val, n_genes_val_m_ge_5,
    median_val_m_per_gene).
    """
    rng = np.random.default_rng(seed)
    df = fit_df.dropna(subset=["gene_key", "expName", "expDesc", "media"]).copy()
    df["condition_key"] = _condition_key(df)

    rows = []
    for org, sub in df.groupby("orgId"):
        org_conds = sorted(sub["condition_key"].unique())
        n_cond = len(org_conds)
        if n_cond < min_holdout + 1:
            continue
        for frac in holdout_fractions:
            k = max(min_holdout, min(max_holdout, int(round(frac * n_cond))))
            if k >= n_cond:
                continue
            if replicate_group_together:
                holdout_conds = set(rng.choice(org_conds, size=k, replace=False).tolist())
            else:
                # fall back: hold out k expNames at random (leaky)
                org_exps = sorted(sub["expName"].unique())
                holdout_exps = set(rng.choice(org_exps, size=k, replace=False).tolist())
                holdout_conds = set(sub[sub["expName"].isin(holdout_exps)]["condition_key"].unique())
            val_rows = sub[sub["condition_key"].isin(holdout_conds)]
            per_gene_val_m = val_rows.groupby("gene_key")["condition_key"].nunique()
            rows.append({
                "orgId": org,
                "holdout_fraction": float(frac),
                "n_train_conditions": int(n_cond - len(holdout_conds)),
                "n_val_conditions": int(len(holdout_conds)),
                "n_val_exps": int(val_rows["expName"].nunique()),
                "n_genes_with_any_val": int((per_gene_val_m >= 1).sum()),
                "n_genes_val_m_ge_5": int((per_gene_val_m >= 5).sum()),
                "n_genes_val_m_ge_10": int((per_gene_val_m >= 10).sum()),
                "median_val_m_per_gene": float(per_gene_val_m.median()) if len(per_gene_val_m) else 0.0,
                "p25_val_m_per_gene": float(per_gene_val_m.quantile(0.25)) if len(per_gene_val_m) else 0.0,
            })
    return pd.DataFrame(rows)
