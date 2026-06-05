"""Emit candidate eligibility & split protocols for R-LOCK-1 and R-LOCK-2.

This is the R0 analog of S1's candidates.py: data exploration in R0 emits
candidates; the lock-tier decisions choose among them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CANDIDATE_OUT = Path("data_contract/ranking/r0_candidates.yaml")


def build_eligibility_candidates(frontier: pd.DataFrame,
                                  noise_summary: pd.DataFrame,
                                  *, min_org_coverage: float = 0.5,
                                  min_eligible_genes_per_org: int = 100) -> list[dict]:
    """Surface (m_min, IQR_min) pairs that pass per-org coverage thresholds.

    A candidate must:
      - leave ≥ min_eligible_genes_per_org eligible in at least min_org_coverage
        fraction of orgs (so val Spearman is computable everywhere).
      - have IQR_min ≥ median replicate noise floor across orgs (so the filter
        is at least as strict as the noise).
    """
    if noise_summary.empty:
        median_noise_r = np.nan
    else:
        median_noise_r = float(noise_summary["median"].median())
    # Convert "noise as Spearman" → "noise as IQR-equivalent" is non-trivial;
    # for R0 we just report it as informational.
    out: list[dict] = []
    metric = frontier["metric"].iloc[0] if "metric" in frontier.columns else "iqr_g"
    grouped = frontier.groupby(["m_min", "threshold"])
    for (m_min, thr), g in grouped:
        n_orgs_pass = int((g["n_eligible"] >= min_eligible_genes_per_org).sum())
        n_orgs_total = len(g)
        coverage = n_orgs_pass / n_orgs_total if n_orgs_total else 0.0
        if coverage < min_org_coverage:
            continue
        out.append({
            "metric": metric,
            "m_min": int(m_min),
            "threshold": float(thr),
            "n_orgs_pass": n_orgs_pass,
            "n_orgs_total": n_orgs_total,
            "coverage": float(coverage),
            "median_eligible_genes_per_org": float(g["n_eligible"].median()),
            "min_eligible_genes_per_org": int(g["n_eligible"].min()),
            "informational_median_replicate_spearman": median_noise_r,
        })
    out.sort(key=lambda x: (-x["coverage"], -x["median_eligible_genes_per_org"]))
    return out


def build_split_candidates(split_df: pd.DataFrame,
                            *, min_genes_with_val_m_ge_5: int = 100,
                            min_org_coverage: float = 0.6) -> list[dict]:
    """Surface holdout-fraction candidates where most orgs retain enough val signal."""
    out: list[dict] = []
    for frac, g in split_df.groupby("holdout_fraction"):
        n_pass = int((g["n_genes_val_m_ge_5"] >= min_genes_with_val_m_ge_5).sum())
        n_total = len(g)
        coverage = n_pass / n_total if n_total else 0.0
        if coverage < min_org_coverage:
            continue
        out.append({
            "split_id": f"within_org_holdout_frac{frac:.2f}",
            "holdout_fraction": float(frac),
            "n_orgs_pass": n_pass,
            "n_orgs_total": n_total,
            "coverage": float(coverage),
            "median_val_genes_per_org": float(g["n_genes_val_m_ge_5"].median()),
            "median_val_conditions_per_org": float(g["n_val_conditions"].median()),
        })
    out.sort(key=lambda x: -x["coverage"])
    return out


def write_candidates(eligibility: list[dict], splits: list[dict]) -> None:
    CANDIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "emitted_by": "R0-A_data_characterization",
        "eligibility_candidates": eligibility,
        "split_candidates": splits,
    }
    with CANDIDATE_OUT.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
