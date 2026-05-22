"""T1-D — Metadata Bundle Test (Hypothesis H-ENC-04).

Compares:
  A1 (chemistry_only):         425-dim binary multihot (T1-DEC-004 winner)
  A2 (chemistry_plus_metadata): 425-dim multihot + 9 metadata features

Metadata features (from experiment_metadata.parquet, S4 artifact):
  Categorical (one-hot encoded, train-only vocab):
    - oxygen_idx:           4 levels → 4-dim one-hot
    - experiment_group_idx: 42 levels → 42-dim one-hot
    - liquid_state_idx:     3 levels → 3-dim one-hot
  Numeric (z-scored, train-only stats, with finite indicators):
    - temperature_c_z + temperature_c_is_finite
    - pH_z + pH_is_finite
    - shaking_rpm_z + shaking_rpm_is_finite

Total condition dim:
  chemistry_only:         425
  chemistry_plus_metadata: 425 + 49 (one-hot) + 6 (numeric) = 480

Controls held fixed:
  - Locked split (multi_org_balanced), S4 artifact (de21504134c84a6c)
  - S5 weighted_full quality weighting, full org pool
  - Binary multihot chemistry (T1-DEC-004)
  - Same shallow concat-linear MLP, same seeds, same epochs
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
import yaml

from src.data.datasets.build_s5_dataset import S5TorchDataset
from src.training.train_loop import TrainLoopConfig, train_one_arm
from src.experiments.tier1._t1_common import (
    T1Inputs,
    load_t1_inputs,
    make_multihot_model,
)


log = logging.getLogger(__name__)

ARM_CHEM_ONLY = "chemistry_only"
ARM_CHEM_META = "chemistry_plus_metadata"


def _build_metadata_matrix(
    *,
    metadata_path: Path,
    exp_to_row: dict[str, int],
    train_exp_ids: set[str],
) -> tuple[np.ndarray, dict]:
    """Build a dense (n_experiments, 55) metadata feature matrix.

    Categorical columns are one-hot encoded using train-only vocab.
    Numeric columns are already z-scored in the S4 artifact (train-only stats).
    Non-finite numeric values are zeroed; finite-indicator flags are included.

    Returns the matrix and a summary dict for logging/auditing.
    """
    meta_df = pd.read_parquet(metadata_path)
    meta_df["experiment_id"] = meta_df["experiment_id"].astype(str)

    cat_cols = ["oxygen_idx", "experiment_group_idx", "liquid_state_idx"]
    num_cols = ["temperature_c_z", "pH_z", "shaking_rpm_z"]
    finite_cols = ["temperature_c_is_finite", "pH_is_finite", "shaking_rpm_is_finite"]

    # Build train-only vocab for each categorical
    train_meta = meta_df[meta_df["experiment_id"].isin(train_exp_ids)]
    cat_vocabs: dict[str, dict[int, int]] = {}
    onehot_dims: dict[str, int] = {}
    for col in cat_cols:
        unique_vals = sorted(train_meta[col].dropna().unique().tolist())
        cat_vocabs[col] = {int(v): i for i, v in enumerate(unique_vals)}
        onehot_dims[col] = len(unique_vals)

    total_onehot = sum(onehot_dims.values())
    total_numeric = len(num_cols) + len(finite_cols)
    total_meta_dim = total_onehot + total_numeric

    n_exp = max(exp_to_row.values()) + 1
    mat = np.zeros((n_exp, total_meta_dim), dtype=np.float32)

    # Track unknown rates
    unk_counts = {col: 0 for col in cat_cols}
    mapped_counts = {col: 0 for col in cat_cols}

    for _, row in meta_df.iterrows():
        eid = str(row["experiment_id"])
        if eid not in exp_to_row:
            continue
        r = exp_to_row[eid]
        offset = 0

        # One-hot categoricals
        for col in cat_cols:
            val = int(row[col])
            vocab = cat_vocabs[col]
            dim = onehot_dims[col]
            if val in vocab:
                mat[r, offset + vocab[val]] = 1.0
                mapped_counts[col] += 1
            else:
                unk_counts[col] += 1
            offset += dim

        # Numeric + finite indicators
        for num_col, fin_col in zip(num_cols, finite_cols):
            is_fin = float(row[fin_col])
            mat[r, offset] = float(row[num_col]) if is_fin else 0.0
            offset += 1
            mat[r, offset] = is_fin
            offset += 1

    summary = {
        "total_meta_dim": total_meta_dim,
        "onehot_dims": onehot_dims,
        "total_onehot": total_onehot,
        "total_numeric": total_numeric,
        "cat_vocabs_sizes": {c: len(v) for c, v in cat_vocabs.items()},
        "unk_counts": unk_counts,
        "mapped_counts": mapped_counts,
    }
    return mat, summary


def _run_one(
    *,
    arm_name: str,
    seed: int,
    inputs: T1Inputs,
    condition_matrix: np.ndarray,
    weights: np.ndarray,
    cfg_model: dict,
) -> tuple[pd.DataFrame, dict]:
    gene_dim = int(inputs.embedding_matrix.shape[1])
    cond_dim = int(condition_matrix.shape[1])
    model = make_multihot_model(
        gene_dim=gene_dim, chem_dim=cond_dim,
        hidden_dim=int(cfg_model["hidden_dim"]),
        dropout=float(cfg_model["dropout"]),
    )
    train_ds = S5TorchDataset(
        inputs.train_batch,
        embedding_matrix=inputs.embedding_matrix,
        chemistry_matrix=condition_matrix,
        weights=weights,
    )
    val_ds = S5TorchDataset(
        inputs.val_batch,
        embedding_matrix=inputs.embedding_matrix,
        chemistry_matrix=condition_matrix,
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


def _bootstrap_ci(val_true, val_pred, *, n_boot=1000, seed=0) -> dict:
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


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items() if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _plot_figures(figures_dir: Path, metrics_df: pd.DataFrame, summaries_df: pd.DataFrame) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    arm_names = [ARM_CHEM_ONLY, ARM_CHEM_META]

    # 01: best val RMSE+MAE per arm
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

    # 02: train/val curves
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    colors = {ARM_CHEM_ONLY: "C0", ARM_CHEM_META: "C1"}
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


def run_t1d(cfg, *,
            protocol_path: Path | None = None,
            output_root: str = "t1d",
            figures_dirname: str = "tier1_d",
            experiment_id_label: str = "T1-D_metadata_bundle",
            title: str = "T1-D Metadata Bundle Test (H-ENC-04)") -> dict:
    """Run T1-D end-to-end. Returns the comparison payload."""
    fitness_path = Path("data/derived/canonical/v0/fitness_experiment_long.parquet")
    feature_contract_path = Path("data_contract/feature_contract.yaml")
    if protocol_path is None:
        protocol_path = Path("data_contract/splits/locked_protocol.yaml")
    eval_policy_path = Path("data_contract/policy/eval_policy.yaml")
    embedding_dir = Path("data/processed/ProtLM_embeddings_layer8")
    cache_dir = Path(f"artifacts/cache/{output_root}")
    figures_dir = Path(f"research_log/figures/{figures_dirname}")
    output_dir = Path(f"artifacts/runs/{output_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60); log.info(title); log.info("=" * 60)

    log.info("[1/5] loading shared T1 inputs")
    inputs = load_t1_inputs(
        fitness_path=fitness_path,
        feature_contract_path=feature_contract_path,
        locked_protocol_path=protocol_path,
        eval_policy_path=eval_policy_path,
        embedding_dir=embedding_dir,
        cache_dir=cache_dir,
    )
    log.info("    train rows=%d val rows=%d artifact_id=%s",
             len(inputs.train_df), len(inputs.val_df), inputs.artifact_id)

    # Load metadata
    feature_contract = yaml.safe_load(feature_contract_path.read_text())
    artifact_root = Path("data_contract/preprocessing") / inputs.artifact_id
    metadata_path = artifact_root / feature_contract["experiment_metadata_table"]["path"]

    train_exp_ids = set(
        inputs.train_df["experiment_id"].astype(str).unique().tolist()
    )

    log.info("[2/5] building condition matrices")
    # Arm 1: chemistry only (binary multihot, locked T1-DEC-004)
    chem_only = inputs.chemistry_dense.copy()
    chem_only[chem_only != 0] = 1.0
    log.info("    chemistry_only shape=%s", chem_only.shape)

    # Arm 2: chemistry + metadata
    meta_mat, meta_summary = _build_metadata_matrix(
        metadata_path=metadata_path,
        exp_to_row=inputs.exp_to_row,
        train_exp_ids=train_exp_ids,
    )
    log.info("    metadata shape=%s  dims: onehot=%d numeric=%d total=%d",
             meta_mat.shape, meta_summary["total_onehot"],
             meta_summary["total_numeric"], meta_summary["total_meta_dim"])
    log.info("    cat vocab sizes: %s", meta_summary["cat_vocabs_sizes"])
    log.info("    unk counts: %s", meta_summary["unk_counts"])

    chem_plus_meta = np.concatenate([chem_only, meta_mat], axis=1)
    log.info("    chemistry_plus_metadata shape=%s", chem_plus_meta.shape)

    condition_matrices = {
        ARM_CHEM_ONLY: chem_only,
        ARM_CHEM_META: chem_plus_meta,
    }

    model_cfg = {
        "hidden_dim": 256, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4,
        "batch_size": 8192, "epochs": 8, "device": "auto",
    }
    seeds = [0, 1, 2]

    all_metrics: list[pd.DataFrame] = []
    all_summaries: list[dict] = []
    per_arm_predictions: dict[str, dict[int, np.ndarray]] = {a: {} for a in condition_matrices}
    per_arm_true: dict[str, dict[int, np.ndarray]] = {a: {} for a in condition_matrices}

    log.info("[3/5] training %d arms × %d seeds", len(condition_matrices), len(seeds))
    for arm in [ARM_CHEM_ONLY, ARM_CHEM_META]:
        log.info("──── arm = %s (dim=%d) ────", arm, condition_matrices[arm].shape[1])
        for seed in seeds:
            metrics_df, summary = _run_one(
                arm_name=arm, seed=seed, inputs=inputs,
                condition_matrix=condition_matrices[arm],
                weights=inputs.weighted_weights, cfg_model=model_cfg,
            )
            all_metrics.append(metrics_df)
            all_summaries.append(
                {k: v for k, v in summary.items() if not k.startswith("_")} | {"arm": arm}
            )
            per_arm_predictions[arm][seed] = summary["_best_val_pred"]
            per_arm_true[arm][seed] = summary["_best_val_true"]
            log.info("    arm=%s seed=%d best_val_rmse=%.4f mae=%.4f",
                     arm, seed, summary["best_val_rmse"], summary["best_val_mae"])

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summaries_df = pd.DataFrame(all_summaries)

    log.info("[4/5] aggregating + bootstrap")
    agg = summaries_df.groupby("arm")[["best_val_rmse", "best_val_mae"]].agg(["mean", "std"])
    agg.columns = ["rmse_mean", "rmse_std", "mae_mean", "mae_std"]
    arm_metrics = agg.reset_index()
    arm_metrics["arm"] = pd.Categorical(
        arm_metrics["arm"], categories=[ARM_CHEM_ONLY, ARM_CHEM_META], ordered=True
    )
    arm_metrics = arm_metrics.sort_values("arm").reset_index(drop=True)
    log.info("per-arm summary:\n%s", arm_metrics.to_string(index=False))

    bootstrap_per_arm = {}
    for arm in [ARM_CHEM_ONLY, ARM_CHEM_META]:
        bootstrap_per_arm[arm] = _bootstrap_ci(
            per_arm_true[arm][0], per_arm_predictions[arm][0], n_boot=1000, seed=0
        )
        b = bootstrap_per_arm[arm]
        log.info("    arm=%s bootstrap (seed 0): RMSE [%.4f, %.4f] MAE [%.4f, %.4f]",
                 arm, b["rmse_ci_low"], b["rmse_ci_high"], b["mae_ci_low"], b["mae_ci_high"])

    # Decision logic
    eval_policy = yaml.safe_load(eval_policy_path.read_text())
    th = eval_policy["gain_thresholds_per_protocol"][inputs.locked_protocol_id]
    threshold_rmse = float(th["rmse"])
    threshold_mae = float(th["mae"])

    rmse_chem = float(arm_metrics.set_index("arm").loc[ARM_CHEM_ONLY, "rmse_mean"])
    rmse_meta = float(arm_metrics.set_index("arm").loc[ARM_CHEM_META, "rmse_mean"])
    mae_chem = float(arm_metrics.set_index("arm").loc[ARM_CHEM_ONLY, "mae_mean"])
    mae_meta = float(arm_metrics.set_index("arm").loc[ARM_CHEM_META, "mae_mean"])

    # Positive gap = metadata is better (lower error)
    rmse_gap = rmse_chem - rmse_meta
    mae_gap = mae_chem - mae_meta

    meta_clearly_better = (rmse_gap > threshold_rmse) and (mae_gap > threshold_mae)
    chem_clearly_better = (-rmse_gap > threshold_rmse) and (-mae_gap > threshold_mae)

    if meta_clearly_better:
        decision = "promote_chemistry_plus_metadata"
    elif chem_clearly_better:
        decision = "promote_chemistry_only"
    else:
        decision = "no_winner_default_chemistry_only"

    comparison = {
        "rmse_gap_chem_minus_meta": float(rmse_gap),
        "mae_gap_chem_minus_meta": float(mae_gap),
        "rmse_threshold_locked_s2": threshold_rmse,
        "mae_threshold_locked_s2": threshold_mae,
        "meta_clearly_better": bool(meta_clearly_better),
        "chem_clearly_better": bool(chem_clearly_better),
        "decision": decision,
    }
    log.info("Decision: %s | RMSE gap (chem - meta) = %.4f (thr=%.4f) | MAE gap = %.4f (thr=%.4f)",
             decision, rmse_gap, threshold_rmse, mae_gap, threshold_mae)

    log.info("[5/5] writing artifacts")
    _plot_figures(figures_dir, metrics_df, summaries_df)
    metrics_df.to_parquet(output_dir / "t1d_metrics.parquet", index=False)
    summaries_df.to_parquet(output_dir / "t1d_summaries.parquet", index=False)
    payload = {
        "experiment_id": experiment_id_label,
        "stage_or_tier": "T1",
        "hypothesis": "H-ENC-04",
        "locked_protocol_id": inputs.locked_protocol_id,
        "feature_contract_artifact_id": inputs.artifact_id,
        "seeds": seeds,
        "arm_metrics": _to_jsonable(arm_metrics.to_dict(orient="records")),
        "bootstrap_ci_per_arm": _to_jsonable(bootstrap_per_arm),
        "comparison": _to_jsonable(comparison),
        "metadata_summary": _to_jsonable(meta_summary),
        "condition_dims": {
            ARM_CHEM_ONLY: int(chem_only.shape[1]),
            ARM_CHEM_META: int(chem_plus_meta.shape[1]),
        },
        "n_train_rows": int(len(inputs.train_df)),
        "n_val_rows": int(len(inputs.val_df)),
    }
    (output_dir / "t1d_summary.json").write_text(json.dumps(payload, indent=2))
    log.info("T1-D complete. outputs at %s", output_dir)
    return payload
