"""Target transforms: de-meaning and denoising (item F).

The decomposition that governs this project:

    fit(g, c) = mu + a_g + b_c + I(g, c) + noise

`a_g` is the gene main effect (how important g is on average), `b_c` the condition
main effect (how harsh c is on average), and `I(g, c)` the interaction -- the only
part that is gene-by-condition specific, and the only part the ranking task is
actually about. Two consequences the project asserted for months but never tested
directly:

  * **De-meaning.** Within-gene ranking already removes `a_g` implicitly (it is
    constant within a gene) and the eligibility filter suppresses `b_c`. Modelling
    `I` explicitly, rather than modelling `fit` and hoping the metric cancels the
    main effects, is a different and better-posed regression problem.
  * **Denoising.** The replicate ceiling is the ceiling of the SINGLE NOISY
    MEASUREMENT task. Averaging replicates reduces the noise term, which raises the
    ceiling and changes what "fraction of achievable" means.

A CONSTRAINT THAT CANNOT BE ENGINEERED AWAY. Under `condition_holdout` the val
conditions are 100% disjoint from train, so `b_c` is NOT ESTIMABLE for any val
condition -- there are no training rows at that condition to average. Full
`a_g + b_c` de-meaning is therefore only available on splits with warm columns.
On the primary split you may de-mean by `a_g` only, or predict `b_c` from chemistry
(which is exactly the chem_null baseline, so the "de-meaned" target then embeds a
baseline and the comparison stops being clean). `fit_additive_effects` records which
components are estimable so this cannot be forgotten.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class AdditiveEffects:
    """Train-only estimates of mu, a_g and b_c."""
    mu: float
    gene_effect: dict
    condition_effect: dict
    n_train_rows: int
    estimable: tuple = ("gene", "condition")
    stats: dict = field(default_factory=dict)


def fit_additive_effects(
    train_df: pd.DataFrame, *, gene_col="gene_key", condition_col="condition_key",
    fit_col="fit", n_iter: int = 3,
) -> AdditiveEffects:
    """Estimate mu, a_g, b_c on TRAIN ROWS ONLY, by alternating means.

    Alternating (rather than one-shot) means because the design is unbalanced: genes
    are not measured at identical condition sets, so a single pass leaves main-effect
    mass in the residual. Three iterations is ample for a two-factor additive fit.
    """
    df = train_df[[gene_col, condition_col, fit_col]].dropna()
    if df.empty:
        return AdditiveEffects(0.0, {}, {}, 0, stats={"empty": True})

    y = df[fit_col].to_numpy(dtype=float)
    mu = float(np.mean(y))
    resid = y - mu
    a = pd.Series(0.0, index=pd.Index(df[gene_col].unique(), name=gene_col))
    b = pd.Series(0.0, index=pd.Index(df[condition_col].unique(), name=condition_col))

    g_idx = df[gene_col].to_numpy()
    c_idx = df[condition_col].to_numpy()
    for _ in range(n_iter):
        a_new = pd.Series(resid + a.reindex(g_idx).to_numpy(), index=g_idx).groupby(level=0).mean()
        a = a_new
        resid = y - mu - a.reindex(g_idx).to_numpy() - b.reindex(c_idx).to_numpy()
        b_new = pd.Series(resid + b.reindex(c_idx).to_numpy(), index=c_idx).groupby(level=0).mean()
        b = b_new
        resid = y - mu - a.reindex(g_idx).to_numpy() - b.reindex(c_idx).to_numpy()

    # IDENTIFIABILITY. mu, a_g and b_c are only identified up to a constant that can
    # be shuffled between them: (mu+k, a-k, b) fits identically. Impose the standard
    # sum-to-zero constraint so `mu` is the grand mean and the effects are deviations
    # from it. Without this, `mu` silently absorbs mean(a)+mean(b) and any downstream
    # comparison of effect magnitudes across fits is meaningless.
    a_bar, b_bar = float(a.mean()), float(b.mean())
    a = a - a_bar
    b = b - b_bar
    mu = mu + a_bar + b_bar
    resid = y - mu - a.reindex(g_idx).to_numpy() - b.reindex(c_idx).to_numpy()

    return AdditiveEffects(
        mu=mu, gene_effect=a.to_dict(), condition_effect=b.to_dict(),
        n_train_rows=int(len(df)),
        stats={"resid_sd": float(np.std(resid)), "raw_sd": float(np.std(y)),
               "var_explained_by_main_effects": float(1 - np.var(resid) / max(np.var(y), 1e-12))},
    )


def apply_demeaning(
    df: pd.DataFrame, effects: AdditiveEffects, *,
    components: tuple = ("gene",), gene_col="gene_key", condition_col="condition_key",
    fit_col="fit", out_col="fit_demeaned",
) -> pd.DataFrame:
    """Subtract the requested main effects, returning a copy with `out_col`.

    Unknown genes/conditions contribute 0 (i.e. fall back to `mu`) and the unknown
    rate is logged -- the train-only-preprocessing rule applied to the target.
    Requesting `condition` on a cold-column split is a loud error, not a silent 0:
    silently treating an unestimable `b_c` as 0 would understate the target and make
    the resulting numbers quietly incomparable.
    """
    for comp in components:
        if comp not in ("gene", "condition"):
            raise ValueError(f"unknown component {comp!r}")
    out = df.copy()
    val = out[fit_col].to_numpy(dtype=float) - effects.mu

    if "gene" in components:
        keys = out[gene_col]
        unknown = float((~keys.isin(effects.gene_effect)).mean())
        log.info("apply_demeaning: unknown gene rate %.4f", unknown)
        val = val - keys.map(effects.gene_effect).fillna(0.0).to_numpy()

    if "condition" in components:
        keys = out[condition_col]
        unknown = float((~keys.isin(effects.condition_effect)).mean())
        if unknown > 0.5:
            raise ValueError(
                f"{unknown:.1%} of conditions have no train estimate for b_c. On a "
                "cold-column split b_c is not estimable; de-mean by gene only, or "
                "predict b_c from chemistry and accept that the target then embeds "
                "the chem_null baseline. See src/ranking/targets.py.")
        log.info("apply_demeaning: unknown condition rate %.4f", unknown)
        val = val - keys.map(effects.condition_effect).fillna(0.0).to_numpy()

    out[out_col] = val
    return out


def replicate_average_target(
    df: pd.DataFrame, *, orgId_col="orgId", gene_col="gene_key",
    condition_col="condition_key", expName_col="expName", fit_col="fit",
    out_col="fit_denoised", min_replicates: int = 1,
) -> pd.DataFrame:
    """Average `fit` across replicate experiments of the same (gene, condition).

    Returns one row per (org, gene, condition) with `out_col` and `n_replicates`.
    The measurement-noise variance falls by ~1/n_replicates, so the replicate ceiling
    computed against this target is HIGHER than the single-measurement ceiling. A
    "fraction of achievable" figure must therefore say which target it refers to;
    the two are not interchangeable.
    """
    g = (df.groupby([orgId_col, gene_col, condition_col])
           .agg(**{out_col: (fit_col, "mean"),
                   "n_replicates": (expName_col, "nunique")})
           .reset_index())
    if min_replicates > 1:
        before = len(g)
        g = g[g["n_replicates"] >= min_replicates].reset_index(drop=True)
        log.info("replicate_average_target: kept %d/%d cells with >=%d replicates",
                 len(g), before, min_replicates)
    return g
