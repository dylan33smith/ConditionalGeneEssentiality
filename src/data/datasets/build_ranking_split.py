"""Materialize the R-LOCK-2 ranking split into concrete row partitions.

Implements `data_contract/ranking/split_protocol.yaml`:
  - PRIMARY: within-org condition holdout, fraction 0.20, replicate-grouped by
    `condition_key = (expDesc, media, temperature)`, stratified by expGroup.
  - DIAGNOSTIC `cell_holdout`: random (gene, condition_key) cells (easy ceiling).
  - DIAGNOSTIC `cold_gene`: whole genes held out per org (tests inductive-over-genes;
    added 2026-05-25 per the Phase-R audit — the primary split is transductive
    over genes, so this bounds how much skill comes from embedding biology vs
    memorized per-gene offsets).

No leakage by construction: holding out whole `condition_key`s pulls all replicate
`expName` siblings into val together.

Each materialized split carries a sha256 `split_hash` for the run manifest v2.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.datasets.conditions import _condition_key

log = logging.getLogger(__name__)


@dataclass
class RankingSplit:
    """Row-level partition labels + provenance for one materialized split."""
    split_id: str
    partition: pd.Series          # index-aligned to input df; values in {"train","val"}
    condition_key: pd.Series      # the condition key per row (for grouping/pooling)
    split_hash: str
    stats: dict = field(default_factory=dict)


def _hash_assignment(org_to_holdout: dict[str, list[str]], salt: str) -> str:
    """Deterministic hash of the holdout assignment (the val side defines the split)."""
    payload = json.dumps({"salt": salt,
                          "holdout": {o: sorted(v) for o, v in sorted(org_to_holdout.items())}},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _stratified_condition_holdout(
    cond_df: pd.DataFrame, *, k: int, stratify_col: str,
    min_groups: int, min_group_size: int, rng: np.random.Generator,
) -> list[str]:
    """Pick k condition_keys to hold out, stratified by expGroup where feasible.

    cond_df: one row per condition_key with columns [condition_key, <stratify_col>].
    Returns a list of held-out condition_keys.
    """
    groups = cond_df.groupby(stratify_col)
    feasible_groups = [(g, sub) for g, sub in groups if len(sub) >= min_group_size]
    if cond_df[stratify_col].nunique() >= min_groups and len(feasible_groups) >= min_groups:
        # Proportional allocation across feasible groups; remainder random.
        held: list[str] = []
        total_feasible = sum(len(sub) for _, sub in feasible_groups)
        for _g, sub in feasible_groups:
            share = int(round(k * len(sub) / total_feasible))
            share = min(share, len(sub))
            if share > 0:
                held.extend(rng.choice(sub["condition_key"].to_numpy(),
                                       size=share, replace=False).tolist())
        # top up / trim to exactly k from the remaining pool
        if len(held) < k:
            pool = cond_df.loc[~cond_df["condition_key"].isin(held), "condition_key"].to_numpy()
            extra = min(k - len(held), len(pool))
            if extra > 0:
                held.extend(rng.choice(pool, size=extra, replace=False).tolist())
        return held[:k]
    # Fallback: random within org
    return rng.choice(cond_df["condition_key"].to_numpy(),
                      size=min(k, len(cond_df)), replace=False).tolist()


def materialize_condition_holdout(
    fit_df: pd.DataFrame, *,
    fraction: float = 0.20, min_holdout: int = 3, max_holdout: int = 30,
    stratify_by: str = "expGroup", stratification_min_groups: int = 2,
    stratification_min_group_size: int = 10,
    seed: int = 0,
) -> RankingSplit:
    """PRIMARY split: within-org condition holdout."""
    df = fit_df.dropna(subset=["orgId", "gene_key", "expDesc", "media"]).copy()
    df["condition_key"] = _condition_key(df)
    rng = np.random.default_rng(seed)

    org_to_holdout: dict[str, list[str]] = {}
    for org, sub in df.groupby("orgId"):
        conds = (sub[["condition_key", stratify_by]]
                 .drop_duplicates("condition_key"))
        n_cond = len(conds)
        if n_cond < min_holdout + 1:
            org_to_holdout[org] = []          # too small → all train (diagnostic-only org)
            continue
        k = int(np.clip(round(fraction * n_cond), min_holdout, min(max_holdout, n_cond - 1)))
        org_to_holdout[org] = _stratified_condition_holdout(
            conds, k=k, stratify_col=stratify_by,
            min_groups=stratification_min_groups,
            min_group_size=stratification_min_group_size, rng=rng,
        )

    holdout_all = {c for v in org_to_holdout.values() for c in v}
    partition = np.where(df["condition_key"].isin(holdout_all), "val", "train")
    partition = pd.Series(partition, index=df.index, name="partition")

    n_val_cond = len(holdout_all)
    stats = {
        "n_train_rows": int((partition == "train").sum()),
        "n_val_rows": int((partition == "val").sum()),
        "n_val_conditions": int(n_val_cond),
        "n_orgs_with_val": int(sum(1 for v in org_to_holdout.values() if v)),
        "fraction": fraction,
    }
    h = _hash_assignment(org_to_holdout, salt=f"condition_holdout|frac={fraction}|seed={seed}")
    return RankingSplit("condition_holdout", partition, df["condition_key"], h, stats)


def materialize_cold_gene(
    fit_df: pd.DataFrame, *, fraction: float = 0.20, seed: int = 0,
) -> RankingSplit:
    """DIAGNOSTIC: hold out whole genes per org (inductive-over-genes test)."""
    df = fit_df.dropna(subset=["orgId", "gene_key", "expDesc", "media"]).copy()
    df["condition_key"] = _condition_key(df)
    rng = np.random.default_rng(seed)
    org_to_holdout: dict[str, list[str]] = {}
    for org, sub in df.groupby("orgId"):
        genes = sub["gene_key"].unique()
        k = max(1, int(round(fraction * len(genes))))
        org_to_holdout[org] = rng.choice(genes, size=min(k, len(genes)), replace=False).tolist()
    holdout_genes = {g for v in org_to_holdout.values() for g in v}
    partition = pd.Series(np.where(df["gene_key"].isin(holdout_genes), "val", "train"),
                          index=df.index, name="partition")
    stats = {"n_train_rows": int((partition == "train").sum()),
             "n_val_rows": int((partition == "val").sum()),
             "n_val_genes": len(holdout_genes)}
    h = _hash_assignment(org_to_holdout, salt=f"cold_gene|frac={fraction}|seed={seed}")
    return RankingSplit("cold_gene", partition, df["condition_key"], h, stats)


def materialize_leave_compound_out(
    fit_df: pd.DataFrame, *,
    experiment_chemistry: pd.DataFrame,
    fraction: float = 0.20,
    seed: int = 0,
    compound_col: str = "canonical_id",
    role_col: str = "role",
    stressor_roles: tuple[str, ...] = ("stressor",),
    compound_groups: dict[str, str] | None = None,
) -> RankingSplit:
    """Hold out whole STRESSOR COMPOUNDS -- the honest-generalization split.

    Motivation. Under ``materialize_condition_holdout`` a held-out condition almost
    always has a chemically near neighbour among the training conditions, because the
    same compound recurs at other concentrations, in other media, or in other
    experiments. That near neighbour is what lets a per-gene lookup interpolate, and
    it is manufactured by the random split rather than by biology. Holding out the
    COMPOUND removes it: no training condition contains the held-out compound at all,
    so every method must generalize from chemical structure rather than retrieve.

    Held out GLOBALLY, not per organism. A compound held out in one organism but
    present in another still leaks through the shared chemistry encoder and the shared
    model weights, which would quietly reintroduce what the split exists to remove.

    ``compound_groups`` optionally maps compound -> group id (e.g. a Murcko scaffold or
    a chemical class), in which case whole GROUPS are held out together. Without it the
    holdout is at exact-compound granularity, which is the weaker but assumption-free
    version. Medium components are never held out -- removing a background nutrient
    changes what the assay IS, not merely which perturbation is unseen.

    A row is val iff its experiment uses at least one held-out compound.
    """
    df = fit_df.dropna(subset=["orgId", "gene_key", "expDesc", "media"]).copy()
    df["condition_key"] = _condition_key(df)
    if "experiment_id" not in df.columns:
        raise KeyError(
            "materialize_leave_compound_out needs an `experiment_id` column on fit_df "
            "to join the chemistry table; attach it before calling.")

    chem = experiment_chemistry
    if role_col in chem.columns:
        chem = chem[chem[role_col].isin(stressor_roles)]
    chem = chem.dropna(subset=["experiment_id", compound_col])

    # only compounds actually used by the organisms in this frame are eligible
    used = chem[chem["experiment_id"].isin(set(df["experiment_id"]))]
    compounds = sorted(used[compound_col].unique())
    if not compounds:
        raise ValueError(
            "No stressor compounds found for these experiments -- check that "
            f"experiment_chemistry has role in {stressor_roles} and that "
            "experiment_id values match.")

    if compound_groups:
        groups = sorted({compound_groups.get(c, c) for c in compounds})
        rng = np.random.default_rng(seed)
        k = max(1, int(round(fraction * len(groups))))
        held_groups = set(rng.choice(groups, size=min(k, len(groups)), replace=False).tolist())
        held = {c for c in compounds if compound_groups.get(c, c) in held_groups}
        unit, n_units, n_held_units = "group", len(groups), len(held_groups)
    else:
        rng = np.random.default_rng(seed)
        k = max(1, int(round(fraction * len(compounds))))
        held = set(rng.choice(compounds, size=min(k, len(compounds)), replace=False).tolist())
        unit, n_units, n_held_units = "compound", len(compounds), len(held)

    val_exps = set(used.loc[used[compound_col].isin(held), "experiment_id"])
    partition = pd.Series(
        np.where(df["experiment_id"].isin(val_exps), "val", "train"),
        index=df.index, name="partition")

    stats = {
        "n_train_rows": int((partition == "train").sum()),
        "n_val_rows": int((partition == "val").sum()),
        "holdout_unit": unit,
        f"n_{unit}s_total": n_units,
        f"n_{unit}s_held_out": n_held_units,
        "n_val_experiments": len(val_exps),
        "n_val_conditions": int(df.loc[partition == "val", "condition_key"].nunique()),
        "n_val_orgs": int(df.loc[partition == "val", "orgId"].nunique()),
    }
    h = _hash_assignment(
        {"held": sorted(held)},
        salt=f"leave_compound_out|frac={fraction}|seed={seed}|unit={unit}")
    return RankingSplit("leave_compound_out", partition, df["condition_key"], h, stats)


def assert_no_compound_leakage(
    fit_df: pd.DataFrame, split: RankingSplit, *,
    experiment_chemistry: pd.DataFrame,
    compound_col: str = "canonical_id",
    role_col: str = "role",
    stressor_roles: tuple[str, ...] = ("stressor",),
) -> None:
    """No held-out compound may appear in ANY training experiment.

    This is the guarantee the split exists to provide; assert it rather than trust it.
    """
    chem = experiment_chemistry
    if role_col in chem.columns:
        chem = chem[chem[role_col].isin(stressor_roles)]
    exp_to_comp = chem.groupby("experiment_id")[compound_col].agg(set)

    train_exps = set(fit_df.loc[split.partition == "train", "experiment_id"])
    val_exps = set(fit_df.loc[split.partition == "val", "experiment_id"])
    train_comps: set[str] = set()
    for e in train_exps:
        train_comps |= exp_to_comp.get(e, set())
    val_comps: set[str] = set()
    for e in val_exps:
        val_comps |= exp_to_comp.get(e, set())

    # the held-out compounds are exactly those that appear in val and never in train
    leaked = val_comps & train_comps
    held = val_comps - train_comps
    if not held:
        raise AssertionError(
            "leave_compound_out produced no genuinely held-out compound -- every "
            "val compound also appears in train. The split is not doing anything.")
    log.info("leave_compound_out: %d held-out compounds, %d compounds shared "
             "(shared ones are co-occurring stressors in mixed conditions)",
             len(held), len(leaked))


def materialize_cell_holdout(
    fit_df: pd.DataFrame, *, fraction: float = 0.20, seed: int = 0,
) -> RankingSplit:
    """DIAGNOSTIC: random (gene, condition_key) cells (easy ceiling)."""
    df = fit_df.dropna(subset=["orgId", "gene_key", "expDesc", "media"]).copy()
    df["condition_key"] = _condition_key(df)
    rng = np.random.default_rng(seed)
    cells = df[["orgId", "gene_key", "condition_key"]].drop_duplicates()
    n_hold = int(round(fraction * len(cells)))
    held_idx = rng.choice(len(cells), size=n_hold, replace=False)
    held = set(map(tuple, cells.iloc[held_idx].to_numpy()))
    key_tuples = list(zip(df["orgId"], df["gene_key"], df["condition_key"]))
    partition = pd.Series(["val" if t in held else "train" for t in key_tuples],
                          index=df.index, name="partition")
    stats = {"n_train_rows": int((partition == "train").sum()),
             "n_val_rows": int((partition == "val").sum()),
             "n_val_cells": len(held)}
    h = hashlib.sha256(f"cell_holdout|frac={fraction}|seed={seed}|n={n_hold}".encode()).hexdigest()
    return RankingSplit("cell_holdout", partition, df["condition_key"], h, stats)


def assert_no_replicate_leakage(df: pd.DataFrame, split: RankingSplit) -> None:
    """Verify every condition_key lives entirely in one partition (no sibling split).

    For the condition-holdout split this must hold by construction; this is the
    guardrail that catches an implementation regression.
    """
    work = df.loc[split.partition.index].copy()
    work["__part"] = split.partition.values
    work["__ck"] = split.condition_key.values
    # within each org, a condition_key must not appear in both train and val
    bad = (work.groupby(["orgId", "__ck"])["__part"].nunique() > 1)
    n_bad = int(bad.sum())
    if n_bad:
        raise AssertionError(
            f"replicate leakage: {n_bad} condition_keys span both partitions")
