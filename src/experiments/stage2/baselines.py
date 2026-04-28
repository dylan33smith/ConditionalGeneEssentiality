"""Per-candidate baseline evaluation orchestrator (S2).

For each candidate protocol from S1:
  - Build train/val splits from the canonical fitness table.
  - Compute the 5 required baselines (global, per-condition, per-organism,
    additive, embedding-NN).
  - Run the power report (eligibility, bootstrap CI, permutation null).
  - Compute heteroscedastic-noise diagnostics on the additive baseline residuals.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.additive_baseline import (
    additive_baseline_metrics,
    fit_additive_baseline,
)
from src.evaluation.nn_baseline import embedding_nn_baseline
from src.evaluation.null_baselines import (
    global_train_mean_baseline,
    per_condition_mean_baseline,
    per_organism_mean_baseline,
)
from src.experiments.stage2 import power as P


log = logging.getLogger(__name__)


def evaluate_candidate(
    fit_df: pd.DataFrame,
    candidate: dict,
    *,
    fit_col: str = "fit",
    gene_col: str = "gene_key",
    cond_col: str = "expName",
    media_col: str = "media",
    org_col: str = "orgId",
    embedding_dir: Path = Path("data/processed/ProtLM_embeddings_layer8"),
    n_bootstrap: int = 1000,
    n_permutations: int = 200,
    additive_max_iters: int = 30,
    additive_tol: float = 1e-3,
) -> dict:
    """Evaluate one candidate protocol; return baseline metrics + power report."""
    pid = candidate["protocol_id"]
    val_orgs = candidate["val_org_ids"]
    test_orgs = candidate["test_org_ids"]
    log.info("──── candidate %s ────", pid)
    log.info("    val_orgs=%s, test_orgs=%s", val_orgs, test_orgs)

    held_out = set(val_orgs) | set(test_orgs)
    train_df = fit_df[~fit_df[org_col].isin(held_out)].dropna(subset=[fit_col])
    val_df = fit_df[fit_df[org_col].isin(set(val_orgs))].dropna(subset=[fit_col])
    log.info("    train rows=%d, val rows=%d", len(train_df), len(val_df))

    train_orgs = sorted(train_df[org_col].unique().tolist())

    train_fit = train_df[fit_col].to_numpy()
    val_fit = val_df[fit_col].to_numpy()

    # ── 1. global train mean ─────────────────────────────────────────────
    log.info("    [1/5] global_train_mean")
    res_global = global_train_mean_baseline(train_fit, val_fit)

    # ── 2. per-condition mean ────────────────────────────────────────────
    log.info("    [2/5] per_condition_mean")
    res_cond = per_condition_mean_baseline(train_df, val_df,
                                            condition_col=cond_col, fit_col=fit_col)

    # ── 3. per-organism mean ─────────────────────────────────────────────
    log.info("    [3/5] per_organism_mean")
    res_org = per_organism_mean_baseline(train_df, val_df,
                                          org_col=org_col, fit_col=fit_col)

    # ── 4. additive baseline ─────────────────────────────────────────────
    log.info("    [4/5] additive_baseline (vectorized)")
    add_fit = fit_additive_baseline(
        train_df[fit_col].to_numpy(),
        train_df[gene_col].to_numpy(),
        train_df[cond_col].to_numpy(),
        max_iters=additive_max_iters,
        tol=additive_tol,
    )
    pred_add = add_fit.predict(val_df[gene_col].to_numpy(), val_df[cond_col].to_numpy())
    metrics_add = additive_baseline_metrics(pred_add, val_fit)
    res_additive = {
        **metrics_add,
        "predictions": pred_add,
        "n_iters": int(add_fit.n_iters),
        "converged": bool(add_fit.converged),
    }
    log.info("        rmse=%.4f mae=%.4f converged=%s in %d iters",
             res_additive["rmse"], res_additive["mae"],
             res_additive["converged"], res_additive["n_iters"])

    # ── 5. embedding NN ──────────────────────────────────────────────────
    log.info("    [5/5] embedding_nn")
    res_nn = embedding_nn_baseline(
        train_df, val_df, train_orgs=train_orgs, val_orgs=val_orgs,
        embedding_dir=embedding_dir,
        media_col=media_col, gene_col=gene_col, fit_col=fit_col,
    )
    log.info("        rmse=%.4f mae=%.4f fallback=%.1f%% no_emb=%.1f%%",
             res_nn["rmse"], res_nn["mae"],
             100 * res_nn["fallback_rate"], 100 * res_nn["no_embedding_rate"])

    # ── Power report (use additive predictions as the strongest non-trivial null) ──
    log.info("    power: eligibility + bootstrap CI + permutation null")
    val_genes_arr = val_df[gene_col].to_numpy()
    power = P.power_report(
        y_true=val_fit, y_pred=pred_add, gene_keys=val_genes_arr,
        n_bootstrap=n_bootstrap, n_permutations=n_permutations,
    )
    spearman_role = P.decide_spearman_role(power)
    log.info("        m=%d v_min=%.4f n_genes_eligible=%d boot_CI=[%.4f,%.4f] role=%s",
             power["m"], power["v_min"], power["bootstrap_ci"]["n_genes_used"],
             power["bootstrap_ci"]["ci_low"], power["bootstrap_ci"]["ci_high"],
             spearman_role)

    # ── Heteroscedastic-noise diagnostics on additive residuals ──────────
    add_resid = val_fit - pred_add
    quantiles = {
        "p1":  float(np.percentile(add_resid, 1)),
        "p5":  float(np.percentile(add_resid, 5)),
        "p25": float(np.percentile(add_resid, 25)),
        "p50": float(np.percentile(add_resid, 50)),
        "p75": float(np.percentile(add_resid, 75)),
        "p95": float(np.percentile(add_resid, 95)),
        "p99": float(np.percentile(add_resid, 99)),
    }
    per_org_residual = (val_df.assign(_resid=add_resid)
                        .groupby(org_col)["_resid"]
                        .agg(["std", lambda s: float(np.percentile(s, 75) - np.percentile(s, 25))])
                        .rename(columns={"<lambda_0>": "iqr"})
                        .reset_index())

    # ── Meaningful-gain thresholds: 50% of (additive − global) gap ───────
    gain_rmse = (res_global["rmse"] - res_additive["rmse"]) * 0.5
    gain_mae = (res_global["mae"] - res_additive["mae"]) * 0.5

    # ── Strip predictions before serialization-safe export ───────────────
    def _strip(d: dict) -> dict:
        return {k: v for k, v in d.items()
                if k != "predictions" and not isinstance(v, np.ndarray)}

    return {
        "protocol_id": pid,
        "val_org_ids": list(val_orgs),
        "test_org_ids": list(test_orgs),
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "baselines": {
            "global_train_mean": _strip(res_global),
            "per_condition_mean": _strip(res_cond),
            "per_organism_mean": _strip(res_org),
            "additive_baseline": _strip(res_additive),
            "embedding_nn": _strip(res_nn),
        },
        "power": {
            "m": power["m"],
            "v_min": power["v_min"],
            "v_min_method": power["v_min_method"],
            "eligibility_curve": power["eligibility_curve"],
            "bootstrap_ci": power["bootstrap_ci"],
            "permutation_null": power["permutation_null"],
            "spearman_role": spearman_role,
        },
        "noise_diagnostics": {
            "additive_residual_quantiles": quantiles,
            "per_organism_residual": per_org_residual.to_dict(orient="records"),
        },
        "meaningful_gain_thresholds": {
            "rmse": float(gain_rmse),
            "mae": float(gain_mae),
            "rule": "0.5 * (global_train_mean - additive)",
        },
        "_predictions": {
            "global_train_mean": res_global["predictions"],
            "additive": pred_add,
            "embedding_nn": res_nn["predictions"],
        },
        "_per_gene_spearman": power["per_gene_spearman"],
    }
