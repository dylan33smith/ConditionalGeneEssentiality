"""T4-C — Hyperparameter Sweep (LR schedule, epochs, early stopping).

Tests whether cosine LR annealing with longer training improves over the
fixed 8-epoch constant-LR baseline. All prior tiers used lr=1e-3 constant
for 8 epochs. The T3 architecture (2-layer ResidualMLP, 512-wide) may
benefit from longer training with a decaying learning rate.

Arms:
  - baseline:       8 epochs, constant lr=1e-3 (current default)
  - cosine_16ep:    16 epochs, cosine annealing lr → 0
  - cosine_32ep:    32 epochs, cosine annealing lr → 0, patience=5
  - cosine_32ep_wd: 32 epochs, cosine, weight_decay=1e-3, patience=5
"""
from __future__ import annotations

import numpy as np

from src.data.datasets.build_s5_dataset import S5TorchDataset
from src.training.train_loop import TrainLoopConfig, train_one_arm
from src.experiments.tier4._t4_common import (
    make_locked_model,
    run_t4_experiment,
)

ARM_BASELINE = "baseline"
ARM_COS16 = "cosine_16ep"
ARM_COS32 = "cosine_32ep"
ARM_COS32_WD = "cosine_32ep_wd"

ARM_CONFIGS = {
    ARM_BASELINE: dict(
        lr=1e-3, weight_decay=1e-4, epochs=8,
        lr_schedule="constant", early_stopping_patience=0,
    ),
    ARM_COS16: dict(
        lr=1e-3, weight_decay=1e-4, epochs=16,
        lr_schedule="cosine", early_stopping_patience=0,
    ),
    ARM_COS32: dict(
        lr=1e-3, weight_decay=1e-4, epochs=32,
        lr_schedule="cosine", early_stopping_patience=5,
    ),
    ARM_COS32_WD: dict(
        lr=1e-3, weight_decay=1e-3, epochs=32,
        lr_schedule="cosine", early_stopping_patience=5,
    ),
}


def _make_model(arm_name, gene_dim, chem_dim):
    return make_locked_model(gene_dim, chem_dim)


def _train_arm(*, arm_name, seed, inputs, model, chemistry_matrix, weights):
    cfg = ARM_CONFIGS[arm_name]
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
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        batch_size=8192,
        epochs=cfg["epochs"],
        device="auto",
        lr_schedule=cfg["lr_schedule"],
        early_stopping_patience=cfg["early_stopping_patience"],
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


def run_t4c(cfg) -> dict:
    return run_t4_experiment(
        experiment_id="T4-C_hparams",
        hypothesis="H-OPT-01",
        title="T4-C Hyperparameter Sweep: LR Schedule + Epochs (H-OPT-01)",
        arm_names=[ARM_BASELINE, ARM_COS16, ARM_COS32, ARM_COS32_WD],
        make_model_fn=_make_model,
        train_arm_fn=_train_arm,
        output_root="t4c",
        figures_dirname="tier4_c",
    )
