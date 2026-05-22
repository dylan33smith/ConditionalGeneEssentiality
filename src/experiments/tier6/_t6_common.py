"""Shared building blocks for Tier 6 experiments (Chemistry Representation).

T6 questions the assumption that the 425-dim binary multihot is the right
chemistry encoding. It tests:
  - T6-A: chemical fingerprints (Morgan / RDKit / MACCS) vs multihot

T6 keeps the locked T5-A gene adapter architecture and only varies the
chemistry input vector.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.experiments.tier2._t2_common import (
    bootstrap_ci,
    get_thresholds,
    load_t2_inputs,
    run_one_arm,
    to_jsonable,
)
from src.experiments.tier3._t3_common import (
    ResidualBlock,
    plot_multi_arm_figures,
)
from src.experiments.tier5._t5_common import AdapterResidualMLP

log = logging.getLogger(__name__)


def load_experiment_fingerprints(
    path: Path = Path("data_contract/chemistry/experiment_fingerprints.npz"),
) -> dict:
    """Load per-experiment fingerprint bundle produced by
    build_fingerprint_experiment_chemistry.py.

    Returns dict with keys: experiment_ids, morgan_mean, rdkit_mean,
    maccs_mean, presence_multihot, coverage, and exp_to_row.
    """
    bundle = np.load(path)
    exp_ids = bundle["experiment_ids"].tolist()
    return {
        "experiment_ids": exp_ids,
        "morgan_mean": bundle["morgan_mean"],
        "rdkit_mean": bundle["rdkit_mean"],
        "maccs_mean": bundle["maccs_mean"],
        "presence_multihot": bundle["presence_multihot"],
        "coverage": bundle["coverage"],
        "exp_to_row": {eid: i for i, eid in enumerate(exp_ids)},
    }


def reorder_to_inputs(
    chem_matrix: np.ndarray,
    fp_exp_to_row: dict[str, int],
    inputs_exp_to_row: dict[str, int],
) -> np.ndarray:
    """Reorder a fingerprint matrix so that row i corresponds to the same
    experiment_id as row i of inputs.chemistry_dense.

    Returns a (len(inputs_exp_to_row), chem_matrix.shape[1]) array.
    Missing experiments get a zero row.
    """
    out = np.zeros((len(inputs_exp_to_row), chem_matrix.shape[1]), dtype=np.float32)
    for eid, target_row in inputs_exp_to_row.items():
        src_row = fp_exp_to_row.get(eid)
        if src_row is not None:
            out[target_row] = chem_matrix[src_row]
    return out


def build_chemistry_for_arm(
    arm_name: str,
    inputs,
    fp_bundle: dict,
) -> np.ndarray:
    """Construct the chemistry matrix for a given arm name.

    Supported arm encodings:
      - multihot:        425-d binary (current locked T5-A baseline)
      - morgan_only:     2048-d mean Morgan fingerprint
      - rdkit_only:      2048-d mean RDKit fingerprint
      - maccs_only:      167-d  mean MACCS keys
      - morgan_plus_multihot: 2048+425 concat
      - maccs_plus_multihot:  167+425 concat
    """
    # Multihot from existing pipeline (binary, no amount info)
    multihot = inputs.chemistry_dense.copy()
    multihot[multihot != 0] = 1.0
    multihot = multihot.astype(np.float32)

    if arm_name == "multihot":
        return multihot

    fp_exp_to_row = fp_bundle["exp_to_row"]
    if arm_name == "morgan_only":
        return reorder_to_inputs(
            fp_bundle["morgan_mean"], fp_exp_to_row, inputs.exp_to_row,
        )
    if arm_name == "rdkit_only":
        return reorder_to_inputs(
            fp_bundle["rdkit_mean"], fp_exp_to_row, inputs.exp_to_row,
        )
    if arm_name == "maccs_only":
        return reorder_to_inputs(
            fp_bundle["maccs_mean"], fp_exp_to_row, inputs.exp_to_row,
        )
    if arm_name == "morgan_plus_multihot":
        morgan = reorder_to_inputs(
            fp_bundle["morgan_mean"], fp_exp_to_row, inputs.exp_to_row,
        )
        return np.concatenate([morgan, multihot], axis=1)
    if arm_name == "maccs_plus_multihot":
        maccs = reorder_to_inputs(
            fp_bundle["maccs_mean"], fp_exp_to_row, inputs.exp_to_row,
        )
        return np.concatenate([maccs, multihot], axis=1)
    raise ValueError(f"unknown chemistry arm: {arm_name}")


def make_t5a_locked_model(gene_dim: int, chem_dim: int) -> nn.Module:
    """Build the T5-A locked architecture: adapter_1024_proj + T3 head."""
    return AdapterResidualMLP(
        gene_dim=gene_dim, chem_dim=chem_dim,
        hidden_dim=512, n_blocks=1, dropout=0.1,
        adapter_hidden=1024, adapter_out=512,
        adapter_n_hidden_layers=1, adapter_layernorm=False,
    )


def run_t6_experiment(
    *,
    experiment_id: str,
    hypothesis: str,
    title: str,
    arm_names: list[str],
    output_root: str,
    figures_dirname: str,
) -> dict:
    output_dir = Path(f"artifacts/runs/{output_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(f"research_log/figures/{figures_dirname}")

    log.info("=" * 60)
    log.info(title)
    log.info("=" * 60)

    log.info("[1/4] loading T6 inputs (T1+T2+T3+T5A locked)")
    inputs = load_t2_inputs(cache_subdir=output_root)
    log.info("    train=%d val=%d artifact=%s",
             len(inputs.train_df), len(inputs.val_df), inputs.artifact_id)
    fp_bundle = load_experiment_fingerprints()
    log.info("    fingerprint bundle: %d experiments, coverage mean=%.3f",
             len(fp_bundle["experiment_ids"]), fp_bundle["coverage"].mean())

    gene_dim = int(inputs.embedding_matrix.shape[1])
    seeds = [0, 1, 2]
    model_cfg = {
        "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 8192,
        "epochs": 8, "device": "auto",
    }

    all_metrics: list[pd.DataFrame] = []
    all_summaries: list[dict] = []
    per_arm_preds: dict[str, dict[int, np.ndarray]] = {a: {} for a in arm_names}
    per_arm_true: dict[str, dict[int, np.ndarray]] = {a: {} for a in arm_names}

    log.info("[2/4] training %d arms × %d seeds", len(arm_names), len(seeds))
    for arm in arm_names:
        log.info("──── arm = %s ────", arm)
        chem = build_chemistry_for_arm(arm, inputs, fp_bundle)
        chem_dim = int(chem.shape[1])
        log.info("    chem_dim=%d (gene_dim=%d)", chem_dim, gene_dim)

        for seed in seeds:
            model = make_t5a_locked_model(gene_dim, chem_dim)
            metrics_df, summary = run_one_arm(
                arm_name=arm, seed=seed, inputs=inputs, model=model,
                chemistry_matrix=chem, weights=inputs.weighted_weights,
                cfg_model=model_cfg,
            )
            all_metrics.append(metrics_df)
            all_summaries.append(
                {k: v for k, v in summary.items() if not k.startswith("_")}
                | {"arm": arm}
            )
            per_arm_preds[arm][seed] = summary["_best_val_pred"]
            per_arm_true[arm][seed] = summary["_best_val_true"]
            log.info("    arm=%s seed=%d best_val_rmse=%.4f mae=%.4f",
                     arm, seed, summary["best_val_rmse"], summary["best_val_mae"])

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summaries_df = pd.DataFrame(all_summaries)

    log.info("[3/4] aggregating + bootstrap")
    agg = summaries_df.groupby("arm")[["best_val_rmse", "best_val_mae"]].agg(
        ["mean", "std"]
    )
    agg.columns = ["rmse_mean", "rmse_std", "mae_mean", "mae_std"]
    arm_metrics = agg.reset_index()
    arm_metrics["arm"] = pd.Categorical(
        arm_metrics["arm"], categories=arm_names, ordered=True
    )
    arm_metrics = arm_metrics.sort_values("arm").reset_index(drop=True)
    log.info("per-arm summary:\n%s", arm_metrics.to_string(index=False))

    boot = {}
    for arm in arm_names:
        boot[arm] = bootstrap_ci(per_arm_true[arm][0], per_arm_preds[arm][0])
        b = boot[arm]
        log.info("    arm=%s bootstrap: RMSE [%.4f, %.4f] MAE [%.4f, %.4f]",
                 arm, b["rmse_ci_low"], b["rmse_ci_high"],
                 b["mae_ci_low"], b["mae_ci_high"])

    threshold_rmse, threshold_mae = get_thresholds()

    arm_rmse = {row["arm"]: float(row["rmse_mean"]) for _, row in arm_metrics.iterrows()}
    arm_mae = {row["arm"]: float(row["mae_mean"]) for _, row in arm_metrics.iterrows()}
    best_arm = min(arm_names, key=lambda a: arm_rmse[a])

    pairwise = {}
    for a in arm_names:
        for b in arm_names:
            if a == b:
                continue
            pairwise[f"{a}_vs_{b}"] = {
                "rmse_gap": float(arm_rmse[a] - arm_rmse[b]),
                "mae_gap": float(arm_mae[a] - arm_mae[b]),
            }

    comparison = {
        "best_arm": best_arm,
        "best_rmse": arm_rmse[best_arm],
        "best_mae": arm_mae[best_arm],
        "pairwise_gaps": pairwise,
        "rmse_threshold": threshold_rmse,
        "mae_threshold": threshold_mae,
    }
    log.info("Best arm: %s", best_arm)

    log.info("[4/4] writing artifacts")
    plot_multi_arm_figures(figures_dir, metrics_df, summaries_df, arm_names)
    metrics_df.to_parquet(output_dir / f"{output_root}_metrics.parquet", index=False)
    summaries_df.to_parquet(output_dir / f"{output_root}_summaries.parquet", index=False)
    payload = {
        "experiment_id": experiment_id,
        "stage_or_tier": "T6",
        "hypothesis": hypothesis,
        "locked_protocol_id": inputs.locked_protocol_id,
        "feature_contract_artifact_id": inputs.artifact_id,
        "seeds": seeds,
        "arm_metrics": to_jsonable(arm_metrics.to_dict(orient="records")),
        "bootstrap_ci_per_arm": to_jsonable(boot),
        "comparison": to_jsonable(comparison),
        "n_train_rows": int(len(inputs.train_df)),
        "n_val_rows": int(len(inputs.val_df)),
        "fingerprint_coverage_mean": float(fp_bundle["coverage"].mean()),
    }
    (output_dir / f"{output_root}_summary.json").write_text(
        json.dumps(payload, indent=2)
    )
    log.info("%s complete. outputs at %s", experiment_id, output_dir)
    return payload
