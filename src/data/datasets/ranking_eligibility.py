"""R-LOCK-1 eligibility filter + weighting (implementation of eligibility_policy.yaml).

Per R-LOCK-1-DEC-001:
  - spread metric = tail_g = p95 - p5 of fit across a gene's conditions (NOT IQR)
  - per-org threshold tail_min_org = max(floor, coef * (1 - r_replicate_org))
  - train: weighted_all, w_g = clip((tail_g - tail_min_org)/(tail_ref_org - tail_min_org), 0, 1)
           * min(m_g/m_min, 1), tail_ref_org = per-org p75 of tail_g
  - val:   hard filter (tail_g_val >= tail_min_org AND m_val >= m_min_val)
  - all thresholds computed on TRAIN rows only (val uses its own rows for the
    val hard filter; no train/val leakage).

⚠ NOTE (audit 2026-05-25): the per-org r_replicate values in eligibility_policy.yaml
were computed full-data, not train-only (documented waiver). This module recomputes
tail_g/m_g per fold but uses the locked r_replicate table as-is.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

POLICY_PATH = Path("data_contract/ranking/eligibility_policy.yaml")


def load_policy(path: Path = POLICY_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def _tail(s: np.ndarray) -> float:
    p5, p95 = np.percentile(s, [5, 95])
    return float(p95 - p5)


def per_gene_spread(df: pd.DataFrame, *, gene_col="gene_key",
                    condition_col="condition_key", fit_col="fit") -> pd.DataFrame:
    """tail_g (p95-p5) and m_g (distinct conditions) per (orgId, gene).

    Conditions are median-pooled over replicates first.
    """
    pooled = (df.groupby(["orgId", gene_col, condition_col])[fit_col]
              .median().reset_index())
    rows = []
    for (org, gene), g in pooled.groupby(["orgId", gene_col]):
        vals = g[fit_col].to_numpy()
        rows.append({"orgId": org, gene_col: gene,
                     "tail_g": _tail(vals), "m_g": int(len(vals))})
    return pd.DataFrame(rows)


def tail_min_for_org(org: str, policy: dict) -> float:
    floor = float(policy["tail_min_floor"])
    coef = float(policy["tail_min_noise_coef"])
    r = policy.get("r_replicate_org", {}).get(org)
    if r is None:
        return floor
    return max(floor, coef * (1.0 - float(r)))


def compute_train_weights(
    train_df: pd.DataFrame, *, policy: dict | None = None,
    gene_col="gene_key", condition_col="condition_key", fit_col="fit",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-gene train weight w_g (R-LOCK-1). Returns (spread_df_with_w_g, gene->w_g).

    spread_df columns: orgId, gene_key, tail_g, m_g, tail_min_org, tail_ref_org, w_g.
    """
    policy = policy or load_policy()
    m_min = int(policy["m_min"])
    spread = per_gene_spread(train_df, gene_col=gene_col,
                             condition_col=condition_col, fit_col=fit_col)
    spread["tail_min_org"] = spread["orgId"].map(lambda o: tail_min_for_org(o, policy))
    # tail_ref_org = per-org p75 of tail_g (on train)
    ref = spread.groupby("orgId")["tail_g"].quantile(0.75).rename("tail_ref_org")
    spread = spread.merge(ref, on="orgId", how="left")
    denom = (spread["tail_ref_org"] - spread["tail_min_org"]).clip(lower=1e-6)
    w_spread = ((spread["tail_g"] - spread["tail_min_org"]) / denom).clip(0.0, 1.0)
    w_m = (spread["m_g"] / m_min).clip(upper=1.0)
    spread["w_g"] = (w_spread * w_m).astype(float)
    gene_to_w = dict(zip(spread[gene_col], spread["w_g"]))
    return spread, gene_to_w


def val_eligible_genes(
    val_df: pd.DataFrame, *, policy: dict | None = None,
    gene_col="gene_key", condition_col="condition_key", fit_col="fit",
) -> set:
    """Hard val filter: genes with tail_g_val >= tail_min_org AND m_val >= m_min_val.

    Computed on VAL rows only (val uses its own spread; no leakage from train).
    """
    policy = policy or load_policy()
    m_min_val = int(policy["m_min_val"])
    spread = per_gene_spread(val_df, gene_col=gene_col,
                             condition_col=condition_col, fit_col=fit_col)
    spread["tail_min_org"] = spread["orgId"].map(lambda o: tail_min_for_org(o, policy))
    keep = spread[(spread["m_g"] >= m_min_val)
                  & (spread["tail_g"] >= spread["tail_min_org"])]
    return set(keep[gene_col].tolist())
