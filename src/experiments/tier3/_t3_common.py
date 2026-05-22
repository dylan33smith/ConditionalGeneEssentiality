"""Shared building blocks for Tier 3 experiments.

T3 inherits locked T1 representation (425-dim binary multihot) and locked T2
fusion (early-concat shallow MLP). T3 varies capacity: depth, residuals,
width, efficiency frontier.

Model architectures defined here:
  - ShallowMLP: 1-layer head (T2 winner, used as baseline)
  - ResidualMLP: N-layer head with residual connections
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
    ShallowMLP,
    bootstrap_ci,
    get_binary_chemistry,
    get_thresholds,
    load_t2_inputs,
    run_one_arm,
    to_jsonable,
)
from src.experiments.tier1._t1_common import T1Inputs

log = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualMLP(nn.Module):
    def __init__(self, *, gene_dim: int, chem_dim: int,
                 hidden_dim: int = 256, n_blocks: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(gene_dim + chem_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, gene_emb: torch.Tensor, chem: torch.Tensor) -> torch.Tensor:
        x = torch.cat([gene_emb, chem.float()], dim=1)
        x = self.proj(x)
        x = self.blocks(x)
        return self.out(x).squeeze(1)


def load_t3_inputs(*, cache_subdir: str = "t3") -> T1Inputs:
    return load_t2_inputs(cache_subdir=cache_subdir)


def plot_multi_arm_figures(
    figures_dir: Path,
    metrics_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    arm_names: list[str],
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    n_arms = len(arm_names)
    colors = [f"C{i}" for i in range(n_arms)]
    arm_color = dict(zip(arm_names, colors))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for col, ax, title in [
        ("best_val_rmse", axs[0], "Best val RMSE per arm"),
        ("best_val_mae", axs[1], "Best val MAE per arm"),
    ]:
        agg = summaries_df.groupby("arm")[col].agg(["mean", "std"]).reset_index()
        agg["arm"] = pd.Categorical(agg["arm"], categories=arm_names, ordered=True)
        agg = agg.sort_values("arm")
        x = np.arange(len(agg))
        ax.bar(x, agg["mean"], yerr=agg["std"], capsize=6,
               color=[arm_color[a] for a in agg["arm"]])
        ax.set_xticks(x)
        ax.set_xticklabels(agg["arm"].astype(str), rotation=15, fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    p1 = figures_dir / "01_val_metrics_per_arm.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    summaries_df.to_csv(p1.with_suffix(".csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for (arm, seed), sub in metrics_df.groupby(["arm", "seed"]):
        c = arm_color.get(arm, "k")
        axs[0].plot(sub["epoch"], sub["train_rmse"], color=c, alpha=0.3)
        axs[0].plot(sub["epoch"], sub["val_rmse"], color=c, lw=2, alpha=0.9,
                    label=f"{arm}" if seed == 0 else None)
        axs[1].plot(sub["epoch"], sub["train_mae"], color=c, alpha=0.3)
        axs[1].plot(sub["epoch"], sub["val_mae"], color=c, lw=2, alpha=0.9)
    axs[0].set_title("RMSE per epoch")
    axs[1].set_title("MAE per epoch")
    for ax in axs:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, ncol=1, loc="upper right")
    p2 = figures_dir / "02_train_val_curves_per_arm.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    metrics_df.to_csv(p2.with_suffix(".csv"), index=False)


def run_t3_experiment(
    *,
    experiment_id: str,
    hypothesis: str,
    title: str,
    arm_names: list[str],
    make_model_fn,
    output_root: str,
    figures_dirname: str,
) -> dict:
    output_dir = Path(f"artifacts/runs/{output_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(f"research_log/figures/{figures_dirname}")

    log.info("=" * 60)
    log.info(title)
    log.info("=" * 60)

    log.info("[1/4] loading T3 inputs (T1+T2 locked)")
    inputs = load_t3_inputs(cache_subdir=output_root)
    log.info("    train=%d val=%d artifact=%s",
             len(inputs.train_df), len(inputs.val_df), inputs.artifact_id)

    chem = get_binary_chemistry(inputs)
    gene_dim = int(inputs.embedding_matrix.shape[1])
    chem_dim = int(chem.shape[1])
    log.info("    gene_dim=%d chem_dim=%d", gene_dim, chem_dim)

    model_cfg = {
        "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 8192,
        "epochs": 8, "device": "auto",
    }
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

    log.info("[3/4] aggregating + bootstrap + parsimony selection")
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
    best_rmse = arm_rmse[best_arm]
    best_mae = arm_mae[best_arm]

    pairwise = {}
    for a in arm_names:
        for b in arm_names:
            if a == b:
                continue
            key = f"{a}_vs_{b}"
            pairwise[key] = {
                "rmse_gap": float(arm_rmse[a] - arm_rmse[b]),
                "mae_gap": float(arm_mae[a] - arm_mae[b]),
            }

    # Parsimony rule: pick the smallest (earliest in arm_names, which is
    # ordered from simplest to most complex) arm that is within threshold
    # of the best on both metrics.
    winner = None
    for arm in arm_names:
        rmse_deficit = arm_rmse[arm] - best_rmse
        mae_deficit = arm_mae[arm] - best_mae
        within_tol = (rmse_deficit <= threshold_rmse) and (mae_deficit <= threshold_mae)
        log.info("    parsimony check arm=%s deficit RMSE=%.4f MAE=%.4f within_tol=%s",
                 arm, rmse_deficit, mae_deficit, within_tol)
        if within_tol:
            winner = arm
            break

    if winner is None:
        winner = best_arm

    comparison = {
        "best_arm_raw": best_arm,
        "best_rmse": best_rmse,
        "best_mae": best_mae,
        "pairwise_gaps": pairwise,
        "rmse_threshold": threshold_rmse,
        "mae_threshold": threshold_mae,
        "parsimony_winner": winner,
    }
    log.info("Parsimony winner: %s (best_raw=%s)", winner, best_arm)

    log.info("[4/4] writing artifacts")
    plot_multi_arm_figures(figures_dir, metrics_df, summaries_df, arm_names)
    metrics_df.to_parquet(output_dir / f"{output_root}_metrics.parquet", index=False)
    summaries_df.to_parquet(output_dir / f"{output_root}_summaries.parquet", index=False)
    payload = {
        "experiment_id": experiment_id,
        "stage_or_tier": "T3",
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
