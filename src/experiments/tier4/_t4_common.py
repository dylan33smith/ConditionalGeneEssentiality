"""Shared building blocks for Tier 4 experiments.

T4 inherits the locked T1 representation, T2 fusion, and T3 architecture.
It varies only the optimization recipe: loss function, target normalization,
learning rate schedule, and other training hyperparameters.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.experiments.tier2._t2_common import (
    bootstrap_ci,
    get_binary_chemistry,
    get_thresholds,
    load_t2_inputs,
    to_jsonable,
)
from src.experiments.tier3._t3_common import ResidualMLP, plot_multi_arm_figures
from src.experiments.tier1._t1_common import T1Inputs
from src.data.datasets.build_s5_dataset import S5TorchDataset
from src.training.train_loop import TrainLoopConfig

log = logging.getLogger(__name__)


def load_t4_inputs(*, cache_subdir: str = "t4") -> T1Inputs:
    return load_t2_inputs(cache_subdir=cache_subdir)


def make_locked_model(gene_dim: int, chem_dim: int) -> ResidualMLP:
    return ResidualMLP(
        gene_dim=gene_dim, chem_dim=chem_dim,
        hidden_dim=512, n_blocks=1, dropout=0.1,
    )


def run_t4_experiment(
    *,
    experiment_id: str,
    hypothesis: str,
    title: str,
    arm_names: list[str],
    make_model_fn,
    train_arm_fn,
    output_root: str,
    figures_dirname: str,
) -> dict:
    """Generic T4 experiment runner.

    Unlike T3, T4 may vary the training loop itself (loss, LR schedule, etc),
    so it takes a `train_arm_fn` callback instead of using the standard
    `run_one_arm`.

    Args:
        make_model_fn: callable(arm_name, gene_dim, chem_dim) -> nn.Module
        train_arm_fn: callable(arm_name, seed, inputs, model, chem, weights)
                      -> (metrics_df, summary_dict)
    """
    output_dir = Path(f"artifacts/runs/{output_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(f"research_log/figures/{figures_dirname}")

    log.info("=" * 60)
    log.info(title)
    log.info("=" * 60)

    log.info("[1/4] loading T4 inputs (T1+T2+T3 locked)")
    inputs = load_t4_inputs(cache_subdir=output_root)
    log.info("    train=%d val=%d artifact=%s",
             len(inputs.train_df), len(inputs.val_df), inputs.artifact_id)

    chem = get_binary_chemistry(inputs)
    gene_dim = int(inputs.embedding_matrix.shape[1])
    chem_dim = int(chem.shape[1])
    log.info("    gene_dim=%d chem_dim=%d", gene_dim, chem_dim)

    seeds = [0, 1, 2]

    all_metrics: list[pd.DataFrame] = []
    all_summaries: list[dict] = []
    per_arm_preds: dict[str, dict[int, np.ndarray]] = {a: {} for a in arm_names}
    per_arm_true: dict[str, dict[int, np.ndarray]] = {a: {} for a in arm_names}

    log.info("[2/4] training %d arms × %d seeds", len(arm_names), len(seeds))
    for arm in arm_names:
        log.info("──── arm = %s ────", arm)
        for seed in seeds:
            model = make_model_fn(arm, gene_dim, chem_dim)
            metrics_df, summary = train_arm_fn(
                arm_name=arm, seed=seed, inputs=inputs, model=model,
                chemistry_matrix=chem, weights=inputs.weighted_weights,
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

    arm_rmse = {
        row["arm"]: float(row["rmse_mean"])
        for _, row in arm_metrics.iterrows()
    }
    arm_mae = {
        row["arm"]: float(row["mae_mean"])
        for _, row in arm_metrics.iterrows()
    }
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
        "stage_or_tier": "T4",
        "hypothesis": hypothesis,
        "locked_protocol_id": inputs.locked_protocol_id,
        "feature_contract_artifact_id": inputs.artifact_id,
        "seeds": seeds,
        "arm_metrics": to_jsonable(arm_metrics.to_dict(orient="records")),
        "bootstrap_ci_per_arm": to_jsonable(boot),
        "comparison": to_jsonable(comparison),
        "n_train_rows": int(len(inputs.train_df)),
        "n_val_rows": int(len(inputs.val_df)),
    }
    (output_dir / f"{output_root}_summary.json").write_text(
        json.dumps(payload, indent=2)
    )
    log.info("%s complete. outputs at %s", experiment_id, output_dir)
    return payload
