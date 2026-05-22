"""Shared building blocks for Tier 5 experiments (Embedding).

T5 questions the assumption that the frozen ProteomeLM-L layer 8 embeddings
are the right gene representation. It tests:
  - T5-A: Adding a learnable gene-side adapter on top of frozen embeddings
  - T5-B: Using different ProteomeLM-L layers (after re-encoding)
  - T5-C: Fine-tuning top-N ProteomeLM layers (only if T5-A/B suggest signal)
  - T5-D: Bypassing ProteomeLM with raw ESM-C embeddings

T5 keeps the locked T2 fusion and T3 capacity from prior tiers; only the
gene representation pathway varies.
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
    run_one_arm,
    to_jsonable,
)
from src.experiments.tier3._t3_common import (
    ResidualMLP,
    ResidualBlock,
    plot_multi_arm_figures,
)
from src.experiments.tier1._t1_common import T1Inputs, load_t1_inputs

log = logging.getLogger(__name__)


def load_t5_inputs(
    *, cache_subdir: str = "t5",
    embedding_dir: Path | None = None,
    embedding_filename_suffix: str = "_proteomelm.pt",
) -> T1Inputs:
    """Load T5 inputs, optionally with a different embedding directory.

    Default uses the same ProtLM_embeddings_layer8 directory as T2-T4.
    Override embedding_dir + suffix for T5-B (other layers) and T5-C (ESM-C).
    """
    if embedding_dir is None:
        return load_t2_inputs(cache_subdir=cache_subdir)
    return load_t1_inputs(
        fitness_path=Path("data/derived/canonical/v0/fitness_experiment_long.parquet"),
        feature_contract_path=Path("data_contract/feature_contract.yaml"),
        locked_protocol_path=Path("data_contract/splits/locked_protocol.yaml"),
        eval_policy_path=Path("data_contract/policy/eval_policy.yaml"),
        embedding_dir=embedding_dir,
        cache_dir=Path(f"artifacts/cache/{cache_subdir}"),
        embedding_filename_suffix=embedding_filename_suffix,
    )


class AdapterResidualMLP(nn.Module):
    """T3-locked head, with a configurable learnable gene-side adapter.

    The adapter is an MLP that processes the frozen gene embedding before
    concatenation with the chemistry vector. It lets the model learn a
    task-specific transformation of the embedding without modifying
    ProteomeLM itself.

    Args:
        adapter_hidden: hidden dim of the adapter, or None to disable
            (passes gene_emb through unchanged for the T3 baseline arm).
        adapter_out: final adapter output dim. Defaults to gene_dim.
        adapter_n_hidden_layers: number of hidden Linear→ReLU→Dropout
            blocks in the adapter. 1 = single hidden layer (T5-A default).
        adapter_layernorm: if True, prepend LayerNorm to the adapter
            (helps with per-organism distribution drift in frozen embeddings).
    """

    def __init__(self, *, gene_dim: int, chem_dim: int,
                 hidden_dim: int = 512, n_blocks: int = 1,
                 dropout: float = 0.1,
                 adapter_hidden: int | None = None,
                 adapter_out: int | None = None,
                 adapter_n_hidden_layers: int = 1,
                 adapter_layernorm: bool = False) -> None:
        super().__init__()
        if adapter_hidden is None:
            self.adapter = nn.Identity()
            effective_gene_dim = gene_dim
        else:
            out_dim = adapter_out if adapter_out is not None else gene_dim
            layers: list[nn.Module] = []
            if adapter_layernorm:
                layers.append(nn.LayerNorm(gene_dim))
            in_dim = gene_dim
            for _ in range(adapter_n_hidden_layers):
                layers.extend([
                    nn.Linear(in_dim, adapter_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                in_dim = adapter_hidden
            layers.append(nn.Linear(in_dim, out_dim))
            self.adapter = nn.Sequential(*layers)
            effective_gene_dim = out_dim

        self.proj = nn.Sequential(
            nn.Linear(effective_gene_dim + chem_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, gene_emb: torch.Tensor, chem: torch.Tensor) -> torch.Tensor:
        g = self.adapter(gene_emb)
        x = torch.cat([g, chem.float()], dim=1)
        x = self.proj(x)
        x = self.blocks(x)
        return self.out(x).squeeze(1)


def run_t5_experiment(
    *,
    experiment_id: str,
    hypothesis: str,
    title: str,
    arm_names: list[str],
    make_model_fn,
    output_root: str,
    figures_dirname: str,
    per_arm_embedding_fn=None,
) -> dict:
    """Run a T5 multi-arm experiment.

    Args:
        per_arm_embedding_fn: optional callable(arm_name) -> (embedding_dir, suffix)
            If provided, each arm reloads inputs with its own embedding source.
            If None, all arms share the locked ProtLM_embeddings_layer8 inputs.
    """
    output_dir = Path(f"artifacts/runs/{output_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(f"research_log/figures/{figures_dirname}")

    log.info("=" * 60)
    log.info(title)
    log.info("=" * 60)

    # Default inputs (used when no per-arm embedding override)
    default_inputs = None
    default_chem = None

    if per_arm_embedding_fn is None:
        log.info("[1/4] loading T5 inputs (T1+T2+T3 locked)")
        default_inputs = load_t5_inputs(cache_subdir=output_root)
        default_chem = get_binary_chemistry(default_inputs)
        log.info("    train=%d val=%d artifact=%s",
                 len(default_inputs.train_df), len(default_inputs.val_df),
                 default_inputs.artifact_id)
        log.info("    gene_dim=%d chem_dim=%d",
                 default_inputs.embedding_matrix.shape[1], default_chem.shape[1])
    else:
        log.info("[1/4] per-arm embedding override — inputs loaded per arm")

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
    arm_inputs_cache: dict = {}
    for arm in arm_names:
        log.info("──── arm = %s ────", arm)
        if per_arm_embedding_fn is not None:
            emb_dir, suffix = per_arm_embedding_fn(arm)
            cache_key = (str(emb_dir), suffix)
            if cache_key not in arm_inputs_cache:
                log.info("    loading inputs from %s (suffix=%s)", emb_dir, suffix)
                arm_inputs_cache[cache_key] = load_t5_inputs(
                    cache_subdir=f"{output_root}_{arm}",
                    embedding_dir=emb_dir,
                    embedding_filename_suffix=suffix,
                )
            inputs = arm_inputs_cache[cache_key]
            chem = get_binary_chemistry(inputs)
        else:
            inputs = default_inputs
            chem = default_chem
        gene_dim = int(inputs.embedding_matrix.shape[1])
        chem_dim = int(chem.shape[1])

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
    # Use last loaded inputs (all arms share split/contract metadata regardless
    # of which embedding source they used).
    ref_inputs = inputs
    payload = {
        "experiment_id": experiment_id,
        "stage_or_tier": "T5",
        "hypothesis": hypothesis,
        "locked_protocol_id": ref_inputs.locked_protocol_id,
        "feature_contract_artifact_id": ref_inputs.artifact_id,
        "seeds": seeds,
        "arm_metrics": to_jsonable(arm_metrics.to_dict(orient="records")),
        "bootstrap_ci_per_arm": to_jsonable(boot),
        "comparison": to_jsonable(comparison),
        "n_train_rows": int(len(ref_inputs.train_df)),
        "n_val_rows": int(len(ref_inputs.val_df)),
    }
    (output_dir / f"{output_root}_summary.json").write_text(
        json.dumps(payload, indent=2)
    )
    log.info("%s complete. outputs at %s", experiment_id, output_dir)
    return payload
