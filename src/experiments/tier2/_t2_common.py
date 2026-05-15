"""Shared building blocks for Tier 2 experiments.

T2 inherits the T1 data-loading pipeline (same inputs, same chemistry matrix)
and varies only the fusion topology. All T2 experiments use the locked T1
representation: 425-dim binary multihot, chemistry only.

Model architectures defined here:
  - LinearHead: cat(gene, chem) → Linear → 1  (T2-A baseline)
  - ShallowMLP: cat(gene, chem) → Linear → ReLU → Dropout → Linear → 1  (T1 winner / T2-A comparison)
  - TwoTower: separate gene MLP + chem MLP → cat → Linear → 1  (T2-B)
  - FiLMFusion: chem → (gamma, beta); gene * gamma + beta → MLP → 1  (T2-C)
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

from src.data.datasets.build_s5_dataset import S5TorchDataset
from src.training.train_loop import TrainLoopConfig, train_one_arm
from src.experiments.tier1._t1_common import (
    T1Inputs,
    load_t1_inputs,
)


log = logging.getLogger(__name__)


class LinearHead(nn.Module):
    def __init__(self, *, gene_dim: int, chem_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(gene_dim + chem_dim, 1)

    def forward(self, gene_emb: torch.Tensor, chem: torch.Tensor) -> torch.Tensor:
        x = torch.cat([gene_emb, chem.float()], dim=1)
        return self.linear(x).squeeze(1)


class ShallowMLP(nn.Module):
    def __init__(self, *, gene_dim: int, chem_dim: int,
                 hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(gene_dim + chem_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gene_emb: torch.Tensor, chem: torch.Tensor) -> torch.Tensor:
        x = torch.cat([gene_emb, chem.float()], dim=1)
        return self.head(x).squeeze(1)


class TwoTower(nn.Module):
    def __init__(self, *, gene_dim: int, chem_dim: int,
                 tower_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.gene_tower = nn.Sequential(
            nn.Linear(gene_dim, tower_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.chem_tower = nn.Sequential(
            nn.Linear(chem_dim, tower_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.merge = nn.Linear(tower_dim * 2, 1)

    def forward(self, gene_emb: torch.Tensor, chem: torch.Tensor) -> torch.Tensor:
        g = self.gene_tower(gene_emb)
        c = self.chem_tower(chem.float())
        return self.merge(torch.cat([g, c], dim=1)).squeeze(1)


class FiLMFusion(nn.Module):
    def __init__(self, *, gene_dim: int, chem_dim: int,
                 hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.film_gen = nn.Linear(chem_dim, gene_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gene_emb: torch.Tensor, chem: torch.Tensor) -> torch.Tensor:
        film_params = self.film_gen(chem.float())
        gamma, beta = film_params.chunk(2, dim=1)
        modulated = gene_emb * (1.0 + gamma) + beta
        return self.head(modulated).squeeze(1)


def run_one_arm(
    *,
    arm_name: str,
    seed: int,
    inputs: T1Inputs,
    model: nn.Module,
    chemistry_matrix: np.ndarray,
    weights: np.ndarray,
    cfg_model: dict,
) -> tuple[pd.DataFrame, dict]:
    train_ds = S5TorchDataset(
        inputs.train_batch,
        embedding_matrix=inputs.embedding_matrix,
        chemistry_matrix=chemistry_matrix,
        weights=weights,
    )
    val_ds = S5TorchDataset(
        inputs.val_batch,
        embedding_matrix=inputs.embedding_matrix,
        chemistry_matrix=chemistry_matrix,
        weights=np.ones(len(inputs.val_batch.y), dtype=np.float32),
    )
    loop_cfg = TrainLoopConfig(
        lr=float(cfg_model["lr"]),
        weight_decay=float(cfg_model["weight_decay"]),
        batch_size=int(cfg_model["batch_size"]),
        epochs=int(cfg_model["epochs"]),
        device=str(cfg_model["device"]),
    )
    return train_one_arm(
        arm_name=arm_name,
        seed=seed,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        val_gene_keys=inputs.val_batch.gene_key,
        val_org_ids=inputs.val_batch.org_id,
        spearman_min_conditions=inputs.spearman_m,
        spearman_min_iqr=inputs.spearman_vmin,
        config=loop_cfg,
    )


def bootstrap_ci(val_true, val_pred, *, n_boot=1000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(val_true)
    rmse_samples = np.empty(n_boot)
    mae_samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diff = val_true[idx] - val_pred[idx]
        rmse_samples[b] = float(np.sqrt(np.mean(diff ** 2)))
        mae_samples[b] = float(np.mean(np.abs(diff)))
    return {
        "rmse_ci_low": float(np.percentile(rmse_samples, 2.5)),
        "rmse_ci_high": float(np.percentile(rmse_samples, 97.5)),
        "mae_ci_low": float(np.percentile(mae_samples, 2.5)),
        "mae_ci_high": float(np.percentile(mae_samples, 97.5)),
        "n_boot": int(n_boot),
    }


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items() if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_t2_inputs(*, cache_subdir: str = "t2") -> T1Inputs:
    return load_t1_inputs(
        fitness_path=Path("data/derived/canonical/v0/fitness_experiment_long.parquet"),
        feature_contract_path=Path("data_contract/feature_contract.yaml"),
        locked_protocol_path=Path("data_contract/splits/locked_protocol.yaml"),
        eval_policy_path=Path("data_contract/policy/eval_policy.yaml"),
        embedding_dir=Path("data/processed/ProtLM_embeddings_layer8"),
        cache_dir=Path(f"artifacts/cache/{cache_subdir}"),
    )


def get_binary_chemistry(inputs: T1Inputs) -> np.ndarray:
    chem = inputs.chemistry_dense.copy()
    chem[chem != 0] = 1.0
    return chem


def get_thresholds() -> tuple[float, float]:
    eval_policy = yaml.safe_load(Path("data_contract/policy/eval_policy.yaml").read_text())
    locked = yaml.safe_load(Path("data_contract/splits/locked_protocol.yaml").read_text())
    pid = str(locked["protocol_id"])
    th = eval_policy["gain_thresholds_per_protocol"][pid]
    return float(th["rmse"]), float(th["mae"])


def plot_two_arm_figures(
    figures_dir: Path,
    metrics_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    arm_names: list[str],
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for col, ax, title in [
        ("best_val_rmse", axs[0], "Best val RMSE per arm"),
        ("best_val_mae", axs[1], "Best val MAE per arm"),
    ]:
        agg = summaries_df.groupby("arm")[col].agg(["mean", "std"]).reset_index()
        agg["arm"] = pd.Categorical(agg["arm"], categories=arm_names, ordered=True)
        agg = agg.sort_values("arm")
        x = np.arange(len(agg))
        ax.bar(x, agg["mean"], yerr=agg["std"], capsize=6, color=["C0", "C1"])
        ax.set_xticks(x); ax.set_xticklabels(agg["arm"].astype(str), rotation=15, fontsize=8)
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    p1 = figures_dir / "01_val_metrics_per_arm.png"
    fig.tight_layout(); fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)
    summaries_df.to_csv(p1.with_suffix(".csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    colors = {arm_names[0]: "C0", arm_names[1]: "C1"}
    for (arm, seed), sub in metrics_df.groupby(["arm", "seed"]):
        c = colors.get(arm, "k")
        axs[0].plot(sub["epoch"], sub["train_rmse"], color=c, alpha=0.4)
        axs[0].plot(sub["epoch"], sub["val_rmse"], color=c, lw=2, alpha=0.9,
                    label=f"{arm}-s{seed}" if seed == 0 else None)
        axs[1].plot(sub["epoch"], sub["train_mae"], color=c, alpha=0.4)
        axs[1].plot(sub["epoch"], sub["val_mae"], color=c, lw=2, alpha=0.9)
    axs[0].set_title("RMSE per epoch"); axs[1].set_title("MAE per epoch")
    for ax in axs:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, ncol=1, loc="upper right")
    p2 = figures_dir / "02_train_val_curves_per_arm.png"
    fig.tight_layout(); fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)
    metrics_df.to_csv(p2.with_suffix(".csv"), index=False)


def run_t2_experiment(
    *,
    experiment_id: str,
    hypothesis: str,
    title: str,
    arm_names: list[str],
    make_model_fn,
    output_root: str,
    figures_dirname: str,
) -> dict:
    """Generic T2 experiment runner.

    Args:
        make_model_fn: callable(arm_name, gene_dim, chem_dim) -> nn.Module
    """
    output_dir = Path(f"artifacts/runs/{output_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(f"research_log/figures/{figures_dirname}")

    log.info("=" * 60); log.info(title); log.info("=" * 60)

    log.info("[1/4] loading T2 inputs (T1 representation locked)")
    inputs = load_t2_inputs(cache_subdir=output_root)
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
                {k: v for k, v in summary.items() if not k.startswith("_")} | {"arm": arm}
            )
            per_arm_preds[arm][seed] = summary["_best_val_pred"]
            per_arm_true[arm][seed] = summary["_best_val_true"]
            log.info("    arm=%s seed=%d best_val_rmse=%.4f mae=%.4f",
                     arm, seed, summary["best_val_rmse"], summary["best_val_mae"])

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summaries_df = pd.DataFrame(all_summaries)

    log.info("[3/4] aggregating + bootstrap")
    agg = summaries_df.groupby("arm")[["best_val_rmse", "best_val_mae"]].agg(["mean", "std"])
    agg.columns = ["rmse_mean", "rmse_std", "mae_mean", "mae_std"]
    arm_metrics = agg.reset_index()
    arm_metrics["arm"] = pd.Categorical(arm_metrics["arm"], categories=arm_names, ordered=True)
    arm_metrics = arm_metrics.sort_values("arm").reset_index(drop=True)
    log.info("per-arm summary:\n%s", arm_metrics.to_string(index=False))

    boot = {}
    for arm in arm_names:
        boot[arm] = bootstrap_ci(per_arm_true[arm][0], per_arm_preds[arm][0])
        b = boot[arm]
        log.info("    arm=%s bootstrap: RMSE [%.4f, %.4f] MAE [%.4f, %.4f]",
                 arm, b["rmse_ci_low"], b["rmse_ci_high"], b["mae_ci_low"], b["mae_ci_high"])

    threshold_rmse, threshold_mae = get_thresholds()
    a0 = arm_names[0]; a1 = arm_names[1]
    rmse_0 = float(arm_metrics.set_index("arm").loc[a0, "rmse_mean"])
    rmse_1 = float(arm_metrics.set_index("arm").loc[a1, "rmse_mean"])
    mae_0 = float(arm_metrics.set_index("arm").loc[a0, "mae_mean"])
    mae_1 = float(arm_metrics.set_index("arm").loc[a1, "mae_mean"])

    gap_rmse = rmse_0 - rmse_1
    gap_mae = mae_0 - mae_1
    a1_wins = (gap_rmse > threshold_rmse) and (gap_mae > threshold_mae)
    a0_wins = (-gap_rmse > threshold_rmse) and (-gap_mae > threshold_mae)
    if a1_wins:
        decision = f"promote_{a1}"
    elif a0_wins:
        decision = f"promote_{a0}"
    else:
        decision = "no_winner"

    comparison = {
        f"rmse_gap_{a0}_minus_{a1}": float(gap_rmse),
        f"mae_gap_{a0}_minus_{a1}": float(gap_mae),
        "rmse_threshold": threshold_rmse,
        "mae_threshold": threshold_mae,
        "decision": decision,
    }
    log.info("Decision: %s | RMSE gap = %.4f (thr=%.4f) | MAE gap = %.4f (thr=%.4f)",
             decision, gap_rmse, threshold_rmse, gap_mae, threshold_mae)

    log.info("[4/4] writing artifacts")
    plot_two_arm_figures(figures_dir, metrics_df, summaries_df, arm_names)
    metrics_df.to_parquet(output_dir / f"{output_root}_metrics.parquet", index=False)
    summaries_df.to_parquet(output_dir / f"{output_root}_summaries.parquet", index=False)
    payload = {
        "experiment_id": experiment_id,
        "stage_or_tier": "T2",
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
    (output_dir / f"{output_root}_summary.json").write_text(json.dumps(payload, indent=2))
    log.info("%s complete. outputs at %s", experiment_id, output_dir)
    return payload
