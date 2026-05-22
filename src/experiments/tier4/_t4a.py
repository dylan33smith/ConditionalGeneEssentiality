"""T4-A — Loss Family (Hypothesis H-LOSS-01).

Tests whether Huber loss improves robustness to extreme fitness scores
vs the current MSE loss.
  - mse:        standard MSE (current default)
  - huber_0.5:  Huber with delta=0.5 (more aggressive outlier clipping)
  - huber_1.0:  Huber with delta=1.0 (moderate)
  - huber_1.5:  Huber with delta=1.5 (mild)

Huber loss is quadratic for errors < delta and linear beyond, reducing the
influence of outlier fitness scores on gradient updates.
"""
from __future__ import annotations

from src.data.datasets.build_s5_dataset import S5TorchDataset
from src.training.train_loop import TrainLoopConfig, train_one_arm
from src.experiments.tier4._t4_common import (
    load_t4_inputs,
    make_locked_model,
    run_t4_experiment,
)

ARM_MSE = "mse"
ARM_HUBER_05 = "huber_0.5"
ARM_HUBER_10 = "huber_1.0"
ARM_HUBER_15 = "huber_1.5"

LOSS_CFG = {
    ARM_MSE: {"loss_fn_name": "mse", "huber_delta": 1.0},
    ARM_HUBER_05: {"loss_fn_name": "huber", "huber_delta": 0.5},
    ARM_HUBER_10: {"loss_fn_name": "huber", "huber_delta": 1.0},
    ARM_HUBER_15: {"loss_fn_name": "huber", "huber_delta": 1.5},
}


def _make_model(arm_name, gene_dim, chem_dim):
    return make_locked_model(gene_dim, chem_dim)


def _train_arm(*, arm_name, seed, inputs, model, chemistry_matrix, weights):
    import numpy as np
    loss_cfg = LOSS_CFG[arm_name]
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
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=8192,
        epochs=8,
        device="auto",
        loss_fn_name=loss_cfg["loss_fn_name"],
        huber_delta=loss_cfg["huber_delta"],
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


def run_t4a(cfg) -> dict:
    return run_t4_experiment(
        experiment_id="T4-A_loss_family",
        hypothesis="H-LOSS-01",
        title="T4-A Loss Family: MSE vs Huber (H-LOSS-01)",
        arm_names=[ARM_MSE, ARM_HUBER_05, ARM_HUBER_10, ARM_HUBER_15],
        make_model_fn=_make_model,
        train_arm_fn=_train_arm,
        output_root="t4a",
        figures_dirname="tier4_a",
    )
