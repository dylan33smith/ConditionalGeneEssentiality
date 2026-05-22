"""T4-B — Target Normalization (Hypothesis H-TARGET-01).

Tests whether per-experiment z-score normalization of fitness targets
improves optimization vs raw targets. Evaluation is always on raw-scale
metrics — z-scoring must translate to real-world improvement to promote.

  - raw:    train on raw `fit` scores (current default)
  - zscore: train on per-experiment z-scored `fit`; predictions inverse-
            transformed to raw scale before evaluation

Z-score stats (mean, std per experiment) are computed on train rows only.
Val/test experiments use their own stats for inverse transform (no leakage).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.datasets.build_s5_dataset import S5TorchDataset, S5RowBatch
from src.training.train_loop import TrainLoopConfig, train_one_arm
from src.experiments.tier4._t4_common import (
    load_t4_inputs,
    make_locked_model,
    run_t4_experiment,
)

log = logging.getLogger(__name__)

ARM_RAW = "raw"
ARM_ZSCORE = "zscore"


def _compute_experiment_stats(train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Compute per-experiment (mean, std) from train rows only."""
    stats = {}
    for exp_id, group in train_df.groupby("experiment_id"):
        mu = float(group["fit"].mean())
        sigma = float(group["fit"].std())
        if sigma < 1e-9:
            sigma = 1.0
        stats[str(exp_id)] = (mu, sigma)
    return stats


def _zscore_row_batch(batch: S5RowBatch, exp_to_row: dict[str, int],
                      stats: dict[str, tuple[float, float]],
                      default_mu: float, default_sigma: float) -> S5RowBatch:
    """Z-score the y values in a row batch using per-experiment stats."""
    row_to_exp = {v: k for k, v in exp_to_row.items()}
    y_new = batch.y.copy()
    for i in range(len(y_new)):
        exp_id = row_to_exp.get(batch.exp_idx[i])
        if exp_id and exp_id in stats:
            mu, sigma = stats[exp_id]
        else:
            mu, sigma = default_mu, default_sigma
        y_new[i] = (y_new[i] - mu) / sigma
    return S5RowBatch(
        gene_idx=batch.gene_idx,
        exp_idx=batch.exp_idx,
        y=y_new,
        gene_key=batch.gene_key,
        org_id=batch.org_id,
        cor12=batch.cor12,
        abs_t=batch.abs_t,
        row_index=batch.row_index,
    )


def _inverse_zscore_predictions(preds: np.ndarray, batch: S5RowBatch,
                                exp_to_row: dict[str, int],
                                stats: dict[str, tuple[float, float]],
                                default_mu: float, default_sigma: float) -> np.ndarray:
    """Inverse z-score predictions back to raw scale."""
    row_to_exp = {v: k for k, v in exp_to_row.items()}
    out = preds.copy()
    for i in range(len(out)):
        exp_id = row_to_exp.get(batch.exp_idx[i])
        if exp_id and exp_id in stats:
            mu, sigma = stats[exp_id]
        else:
            mu, sigma = default_mu, default_sigma
        out[i] = out[i] * sigma + mu
    return out


def _make_model(arm_name, gene_dim, chem_dim):
    return make_locked_model(gene_dim, chem_dim)


def _train_arm(*, arm_name, seed, inputs, model, chemistry_matrix, weights):
    if arm_name == ARM_RAW:
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
            lr=1e-3, weight_decay=1e-4, batch_size=8192,
            epochs=8, device="auto",
        )
        return train_one_arm(
            arm_name=arm_name, seed=seed, model=model,
            train_dataset=train_ds, val_dataset=val_ds,
            val_gene_keys=inputs.val_batch.gene_key,
            val_org_ids=inputs.val_batch.org_id,
            spearman_min_conditions=inputs.spearman_m,
            spearman_min_iqr=inputs.spearman_vmin,
            config=loop_cfg,
        )

    # Z-score arm: transform targets, train, then inverse-transform predictions
    stats = _compute_experiment_stats(inputs.train_df)
    all_mu = float(inputs.train_df["fit"].mean())
    all_sigma = float(inputs.train_df["fit"].std())

    train_batch_z = _zscore_row_batch(
        inputs.train_batch, inputs.exp_to_row, stats, all_mu, all_sigma
    )
    val_batch_z = _zscore_row_batch(
        inputs.val_batch, inputs.exp_to_row, stats, all_mu, all_sigma
    )

    train_ds = S5TorchDataset(
        train_batch_z,
        embedding_matrix=inputs.embedding_matrix,
        chemistry_matrix=chemistry_matrix,
        weights=weights,
    )
    val_ds = S5TorchDataset(
        val_batch_z,
        embedding_matrix=inputs.embedding_matrix,
        chemistry_matrix=chemistry_matrix,
        weights=np.ones(len(val_batch_z.y), dtype=np.float32),
    )
    loop_cfg = TrainLoopConfig(
        lr=1e-3, weight_decay=1e-4, batch_size=8192,
        epochs=8, device="auto",
    )
    metrics_df, summary = train_one_arm(
        arm_name=arm_name, seed=seed, model=model,
        train_dataset=train_ds, val_dataset=val_ds,
        val_gene_keys=inputs.val_batch.gene_key,
        val_org_ids=inputs.val_batch.org_id,
        spearman_min_conditions=inputs.spearman_m,
        spearman_min_iqr=inputs.spearman_vmin,
        config=loop_cfg,
    )

    # Inverse-transform predictions and recompute metrics on raw scale
    raw_true = inputs.val_batch.y.astype(np.float64)
    z_pred = summary["_best_val_pred"]
    raw_pred = _inverse_zscore_predictions(
        z_pred, inputs.val_batch, inputs.exp_to_row,
        stats, all_mu, all_sigma,
    )

    raw_rmse = float(np.sqrt(np.mean((raw_true - raw_pred) ** 2)))
    raw_mae = float(np.mean(np.abs(raw_true - raw_pred)))

    summary["best_val_rmse"] = raw_rmse
    summary["best_val_mae"] = raw_mae
    summary["_best_val_pred"] = raw_pred
    summary["_best_val_true"] = raw_true
    summary["zscore_rmse_before_inverse"] = float(
        np.sqrt(np.mean((val_batch_z.y.astype(np.float64) - z_pred) ** 2))
    )

    log.info("    zscore arm: raw-scale RMSE=%.4f MAE=%.4f", raw_rmse, raw_mae)

    return metrics_df, summary


def run_t4b(cfg) -> dict:
    return run_t4_experiment(
        experiment_id="T4-B_target_norm",
        hypothesis="H-TARGET-01",
        title="T4-B Target Normalization: Raw vs Z-Score (H-TARGET-01)",
        arm_names=[ARM_RAW, ARM_ZSCORE],
        make_model_fn=_make_model,
        train_arm_fn=_train_arm,
        output_root="t4b",
        figures_dirname="tier4_b",
    )
