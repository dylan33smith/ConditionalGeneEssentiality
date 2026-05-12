"""S5 — Training-recipe lock runner."""
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
from omegaconf import DictConfig

from src.data.datasets.build_s5_dataset import (
    S5TorchDataset,
    add_experiment_id,
    build_or_load_experiment_multihot,
    dense_chemistry_from_csr,
    make_row_batch,
    _load_concatenated_embeddings,
)
from src.evaluation.additive_baseline import additive_baseline_metrics, fit_additive_baseline
from src.models.concat_linear import ConcatLinearMLP
from src.training.train_loop import TrainLoopConfig, train_one_arm


log = logging.getLogger(__name__)


def compute_train_thresholds(
    train_df: pd.DataFrame, *, cor12_quantile: float, abs_t_quantile: float
) -> dict[str, float]:
    cor12 = pd.to_numeric(train_df["cor12"], errors="coerce")
    abs_t = np.abs(pd.to_numeric(train_df["t"], errors="coerce"))
    return {
        "cor12_median_train": float(np.nanmedian(cor12)),
        "abs_t_median_train": float(np.nanmedian(abs_t)),
        "cor12_q_train": float(np.nanquantile(cor12, cor12_quantile)),
        "abs_t_q_train": float(np.nanquantile(abs_t, abs_t_quantile)),
        "cor12_quantile": float(cor12_quantile),
        "abs_t_quantile": float(abs_t_quantile),
    }


def compute_weighted_full_weights(
    cor12: np.ndarray, abs_t: np.ndarray, thresholds: dict[str, float]
) -> np.ndarray:
    cor = np.clip(cor12 / max(thresholds["cor12_median_train"], 1e-9), 0.0, 1.0)
    tt = np.clip(abs_t / max(thresholds["abs_t_median_train"], 1e-9), 0.0, 1.0)
    w = (cor * tt).astype(np.float32)
    return np.where(np.isfinite(w), w, 0.0).astype(np.float32)


def compute_strict_slice_mask(
    cor12: np.ndarray, abs_t: np.ndarray, thresholds: dict[str, float]
) -> np.ndarray:
    return (cor12 >= thresholds["cor12_q_train"]) & (abs_t >= thresholds["abs_t_q_train"])


def summarize_drop_causes(
    cor12: np.ndarray, abs_t: np.ndarray, thresholds: dict[str, float]
) -> dict[str, int]:
    cor_bad = cor12 < thresholds["cor12_q_train"]
    t_bad = abs_t < thresholds["abs_t_q_train"]
    return {
        "dropped_only_cor12": int(np.sum(cor_bad & ~t_bad)),
        "dropped_only_abs_t": int(np.sum(~cor_bad & t_bad)),
        "dropped_both": int(np.sum(cor_bad & t_bad)),
    }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_split_masks(fitness_df: pd.DataFrame, locked_protocol_path: Path) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    locked = yaml.safe_load(locked_protocol_path.read_text())
    val_orgs = set(locked["val_org_ids"])
    test_orgs = set(locked["test_org_ids"])
    train_mask = ~(fitness_df["orgId"].isin(val_orgs | test_orgs))
    val_mask = fitness_df["orgId"].isin(val_orgs)
    test_mask = fitness_df["orgId"].isin(test_orgs)
    return train_mask, val_mask, test_mask, locked


def _add_sidecar_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path.with_suffix(".csv"), index=False)


def _plot_figures(figures_dir: Path, metrics_df: pd.DataFrame, summaries_df: pd.DataFrame, sensitivity_df: pd.DataFrame) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 01: train/val curves
    fig1, axs = plt.subplots(1, 2, figsize=(12, 4))
    for (arm, seed), sub in metrics_df.groupby(["arm", "seed"]):
        axs[0].plot(sub["epoch"], sub["train_rmse"], alpha=0.6, label=f"{arm}-s{seed}-train")
        axs[0].plot(sub["epoch"], sub["val_rmse"], alpha=0.6, ls="--", label=f"{arm}-s{seed}-val")
        axs[1].plot(sub["epoch"], sub["train_mae"], alpha=0.6, label=f"{arm}-s{seed}-train")
        axs[1].plot(sub["epoch"], sub["val_mae"], alpha=0.6, ls="--", label=f"{arm}-s{seed}-val")
    axs[0].set_title("RMSE by epoch")
    axs[1].set_title("MAE by epoch")
    for ax in axs:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=6, ncol=2)
    axs[1].legend(fontsize=6, ncol=2)
    p1 = figures_dir / "01_arm_train_val_curves.png"
    fig1.tight_layout()
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    _add_sidecar_csv(p1, metrics_df)

    # 02: arm summary
    agg = (
        summaries_df.groupby("arm")[["best_val_rmse", "best_val_mae"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["arm", "rmse_mean", "rmse_std", "mae_mean", "mae_std"]
    x = np.arange(len(agg))
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.15, agg["rmse_mean"], 0.3, yerr=agg["rmse_std"], label="RMSE")
    ax.bar(x + 0.15, agg["mae_mean"], 0.3, yerr=agg["mae_std"], label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["arm"])
    ax.set_title("Final val metrics by arm")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    p2 = figures_dir / "02_arm_summary_bar.png"
    fig2.tight_layout()
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    _add_sidecar_csv(p2, agg)

    # 03: per-org val RMSE
    per_org_rows = []
    for _, row in metrics_df.sort_values(["arm", "seed", "epoch"]).groupby(["arm", "seed"]).tail(1).iterrows():
        vals = json.loads(row["per_org_val_rmse_json"])
        for org, rmse in vals.items():
            per_org_rows.append({"arm": row["arm"], "seed": int(row["seed"]), "orgId": org, "rmse": float(rmse)})
    per_org_df = pd.DataFrame(per_org_rows)
    fig3, ax = plt.subplots(figsize=(12, 4))
    if len(per_org_df) > 0:
        pivot = per_org_df.groupby(["orgId", "arm"])["rmse"].mean().unstack("arm")
        pivot.plot(kind="bar", ax=ax)
    ax.set_title("Per-org val RMSE (final epoch mean over seeds)")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.3)
    p3 = figures_dir / "03_per_org_val_rmse.png"
    fig3.tight_layout()
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    _add_sidecar_csv(p3, per_org_df)

    # 04: residual quantiles
    q_rows = []
    for _, row in summaries_df.iterrows():
        q = json.loads(row["val_residual_quantiles_json"])
        q_rows.append({"arm": row["arm"], "seed": int(row["seed"]), **q})
    q_df = pd.DataFrame(q_rows)
    q_long = q_df.melt(id_vars=["arm", "seed"], var_name="quantile", value_name="residual")
    fig4, ax = plt.subplots(figsize=(8, 4))
    for arm, sub in q_long.groupby("arm"):
        z = sub.groupby("quantile")["residual"].mean()
        ax.plot(z.index, z.values, marker="o", label=arm)
    ax.set_title("Residual quantiles by arm")
    ax.grid(alpha=0.3)
    ax.legend()
    p4 = figures_dir / "04_residual_quantiles.png"
    fig4.tight_layout()
    fig4.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    _add_sidecar_csv(p4, q_long)

    # 05: train dynamics
    fig5, axs = plt.subplots(1, 2, figsize=(12, 4))
    for (arm, seed), sub in metrics_df.groupby(["arm", "seed"]):
        axs[0].plot(sub["epoch"], sub["grad_norm_mean"], alpha=0.7, label=f"{arm}-s{seed}")
        axs[1].plot(sub["epoch"], sub["param_norm"], alpha=0.7, label=f"{arm}-s{seed}")
    axs[0].set_title("Gradient norm by epoch")
    axs[1].set_title("Parameter norm by epoch")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=6)
    p5 = figures_dir / "05_train_dynamics.png"
    fig5.tight_layout()
    fig5.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close(fig5)
    _add_sidecar_csv(p5, metrics_df[["arm", "seed", "epoch", "grad_norm_mean", "param_norm", "wallclock_seconds"]])

    # 06: threshold sensitivity
    fig6, ax = plt.subplots(figsize=(7, 4))
    if len(sensitivity_df) > 0:
        ax.plot(sensitivity_df["setting"], sensitivity_df["val_rmse"], marker="o", label="RMSE")
        ax.plot(sensitivity_df["setting"], sensitivity_df["val_mae"], marker="o", label="MAE")
    ax.set_title("Threshold sensitivity (single seed)")
    ax.grid(alpha=0.3)
    ax.legend()
    p6 = figures_dir / "06_threshold_sensitivity.png"
    fig6.tight_layout()
    fig6.savefig(p6, dpi=150, bbox_inches="tight")
    plt.close(fig6)
    _add_sidecar_csv(p6, sensitivity_df)


def _write_report(path: Path, payload: dict) -> None:
    _ensure_parent(path)
    lines = [
        "# Stage 5 Report — Training-Recipe Lock",
        "",
        "## TL;DR",
        f"- Chosen policy: **{payload['chosen_arm']}**",
        f"- Best arm mean val RMSE: **{payload['chosen_arm_rmse_mean']:.6f}**",
        f"- Best arm mean val MAE: **{payload['chosen_arm_mae_mean']:.6f}**",
        f"- Additive gate pass: **{payload['additive_gate_pass']}**",
        "",
        "## Frozen thresholds",
        f"- cor12 median (train): `{payload['thresholds']['cor12_median_train']:.6f}`",
        f"- |t| median (train): `{payload['thresholds']['abs_t_median_train']:.6f}`",
        f"- cor12 q{int(payload['thresholds']['cor12_quantile']*100)}: `{payload['thresholds']['cor12_q_train']:.6f}`",
        f"- |t| q{int(payload['thresholds']['abs_t_quantile']*100)}: `{payload['thresholds']['abs_t_q_train']:.6f}`",
        "",
        "## Outputs",
        "- `data_contract/policy/quality_policy.yaml`",
        "- `research_log/figures/stage5/*.png` + sibling CSV",
        "- `artifacts/runs/.../s5_metrics.parquet`",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_quality_policy(path: Path, payload: dict) -> None:
    _ensure_parent(path)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _run_single_training(
    *,
    arm_name: str,
    seed: int,
    cfg_s5: DictConfig,
    embedding_matrix: np.ndarray,
    chemistry_dense: np.ndarray,
    train_batch,
    val_batch,
    weights: np.ndarray,
    train_selector: np.ndarray,
    spearman_m: int,
    spearman_vmin: float,
) -> tuple[pd.DataFrame, dict]:
    train_ds = S5TorchDataset(
        train_batch,
        embedding_matrix=embedding_matrix,
        chemistry_matrix=chemistry_dense,
        weights=weights,
    )
    # keep selected rows (strict slice)
    if not np.all(train_selector):
        selected = np.where(train_selector)[0]
        train_ds = S5TorchDataset(
            type(train_batch)(
                gene_idx=train_batch.gene_idx[selected],
                exp_idx=train_batch.exp_idx[selected],
                y=train_batch.y[selected],
                abs_t=train_batch.abs_t[selected],
                cor12=train_batch.cor12[selected],
                org_id=train_batch.org_id[selected],
                gene_key=train_batch.gene_key[selected],
                row_index=train_batch.row_index[selected],
            ),
            embedding_matrix=embedding_matrix,
            chemistry_matrix=chemistry_dense,
            weights=weights[selected],
        )
    val_ds = S5TorchDataset(
        val_batch,
        embedding_matrix=embedding_matrix,
        chemistry_matrix=chemistry_dense,
        weights=np.ones(len(val_batch.y), dtype=np.float32),
    )
    model = ConcatLinearMLP(
        gene_dim=int(embedding_matrix.shape[1]),
        chemistry_dim=int(chemistry_dense.shape[1]),
        hidden_dim=int(cfg_s5.model.hidden_dim),
        dropout=float(cfg_s5.model.dropout),
    )
    loop_cfg = TrainLoopConfig(
        lr=float(cfg_s5.model.lr),
        weight_decay=float(cfg_s5.model.weight_decay),
        batch_size=int(cfg_s5.model.batch_size),
        epochs=int(cfg_s5.model.epochs),
        device=str(cfg_s5.model.device),
    )
    return train_one_arm(
        arm_name=arm_name,
        seed=seed,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        val_gene_keys=val_batch.gene_key,
        val_org_ids=val_batch.org_id,
        spearman_min_conditions=spearman_m,
        spearman_min_iqr=spearman_vmin,
        config=loop_cfg,
    )


def main(cfg: DictConfig) -> None:
    s5 = cfg.stage.s5
    fitness_path = Path(str(s5.fitness_path))
    locked_protocol_path = Path(str(s5.locked_protocol_path))
    feature_contract_path = Path(str(s5.feature_contract_path))
    eval_policy_path = Path(str(s5.eval_policy_path))
    embedding_dir = Path(str(s5.embedding_dir))
    cache_dir = Path(str(s5.cache_dir))
    figures_dir = Path(str(s5.figures_dir))
    output_quality_policy = Path(str(s5.output_quality_policy))
    report_path = Path(str(s5.report_path))

    log.info("=" * 60)
    log.info("S5 Training-Recipe Lock")
    log.info("=" * 60)

    feature_contract = yaml.safe_load(feature_contract_path.read_text())
    artifact_id = str(feature_contract["artifact_id"])
    artifact_root = Path("data_contract/preprocessing") / artifact_id
    chemistry_parquet = artifact_root / feature_contract["experiment_chemistry_table"]["path"]
    canonical_vocab_path = artifact_root / "canonical_id_vocab.json"
    expected_chem_len = int(s5.chemistry_multihot.length)

    eval_policy = yaml.safe_load(eval_policy_path.read_text())
    protocol_id = "multi_org_balanced"
    spearman_m = int(eval_policy["spearman_eligibility"]["m"])
    spearman_vmin = float(eval_policy["spearman_eligibility"]["v_min_value_per_protocol"][protocol_id])

    fitness_df = pd.read_parquet(fitness_path)
    fitness_df = add_experiment_id(fitness_df)
    train_mask, val_mask, test_mask, locked = _load_split_masks(fitness_df, locked_protocol_path)
    train_df = fitness_df.loc[train_mask].copy()
    val_df = fitness_df.loc[val_mask].copy()
    test_df = fitness_df.loc[test_mask].copy()
    log.info("Rows: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    thresholds = compute_train_thresholds(
        train_df,
        cor12_quantile=float(s5.thresholds.cor12_quantile),
        abs_t_quantile=float(s5.thresholds.abs_t_quantile),
    )

    all_orgs = fitness_df["orgId"].astype(str).unique().tolist()
    embedding_matrix, gene_key_to_idx = _load_concatenated_embeddings(all_orgs, embedding_dir)
    exp_ids = fitness_df["experiment_id"].astype(str).unique().tolist()
    chem_csr, exp_to_row, chem_len = build_or_load_experiment_multihot(
        chemistry_parquet_path=chemistry_parquet,
        canonical_vocab_json_path=canonical_vocab_path,
        target_experiment_ids=exp_ids,
        cache_dir=cache_dir / artifact_id,
        sparse_cache_filename=str(s5.chemistry_multihot.sparse_cache_filename),
    )
    if expected_chem_len != int(chem_len):
        raise ValueError(
            f"Configured chemistry_multihot.length={expected_chem_len} but built {chem_len}"
        )
    chemistry_dense = dense_chemistry_from_csr(chem_csr)

    train_batch = make_row_batch(train_df, gene_key_to_idx=gene_key_to_idx, experiment_id_to_row=exp_to_row)
    val_batch = make_row_batch(val_df, gene_key_to_idx=gene_key_to_idx, experiment_id_to_row=exp_to_row)

    weighted_weights = compute_weighted_full_weights(train_batch.cor12, train_batch.abs_t, thresholds)
    strict_mask = compute_strict_slice_mask(train_batch.cor12, train_batch.abs_t, thresholds)
    drop_causes = summarize_drop_causes(train_batch.cor12, train_batch.abs_t, thresholds)

    all_metrics: list[pd.DataFrame] = []
    all_summaries: list[dict] = []
    arm_effective_n: dict[str, float] = {
        "weighted_full": float(np.sum(weighted_weights)),
        "strict_slice": float(np.sum(strict_mask)),
    }
    arms = list(s5.arms)
    for arm in arms:
        arm_name = str(arm.row_quality_policy)
        if arm_name not in {"weighted_full", "strict_slice"}:
            raise ValueError(f"Unknown S5 arm row_quality_policy={arm_name}")
        for seed in list(s5.seeds):
            if arm_name == "weighted_full":
                selector = np.ones(len(train_batch.y), dtype=bool)
                weights = weighted_weights
            else:
                selector = strict_mask
                weights = np.ones(len(train_batch.y), dtype=np.float32)
            metrics_df, summary = _run_single_training(
                arm_name=arm_name,
                seed=int(seed),
                cfg_s5=s5,
                embedding_matrix=embedding_matrix,
                chemistry_dense=chemistry_dense,
                train_batch=train_batch,
                val_batch=val_batch,
                weights=weights,
                train_selector=selector,
                spearman_m=spearman_m,
                spearman_vmin=spearman_vmin,
            )
            all_metrics.append(metrics_df)
            all_summaries.append(summary)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summaries_df = pd.DataFrame(all_summaries)

    # Additive baseline gate per arm on matched train/val rowsets.
    gate_rows = []
    for arm in ["weighted_full", "strict_slice"]:
        selector = np.ones(len(train_batch.y), dtype=bool) if arm == "weighted_full" else strict_mask
        train_rows = train_df.loc[train_batch.row_index[selector]]
        val_rows = val_df.loc[val_batch.row_index]
        fit = fit_additive_baseline(
            train_rows["fit"].to_numpy(),
            train_rows["gene_key"].to_numpy(),
            train_rows["expName"].to_numpy(),
            max_iters=30,
            tol=1e-3,
        )
        pred = fit.predict(val_rows["gene_key"].to_numpy(), val_rows["expName"].to_numpy())
        add_m = additive_baseline_metrics(pred, val_rows["fit"].to_numpy())
        arm_best = summaries_df[summaries_df["arm"] == arm]["best_val_rmse"].mean()
        gate_rows.append(
            {
                "arm": arm,
                "additive_rmse": float(add_m["rmse"]),
                "additive_mae": float(add_m["mae"]),
                "model_best_rmse_mean": float(arm_best),
                "passes_h_base_01_rmse": bool(arm_best < float(add_m["rmse"])),
            }
        )
    gate_df = pd.DataFrame(gate_rows)

    agg = summaries_df.groupby("arm")[["best_val_rmse", "best_val_mae"]].agg(["mean", "std"])
    agg.columns = ["rmse_mean", "rmse_std", "mae_mean", "mae_std"]
    agg = agg.reset_index().sort_values(["rmse_mean", "mae_mean"])
    chosen_arm = str(agg.iloc[0]["arm"])

    # Threshold sensitivity (single-seed check on chosen arm).
    sens_rows: list[dict] = []
    if bool(s5.threshold_sensitivity.enabled):
        sens_seed = int(s5.threshold_sensitivity.seed)
        if chosen_arm == "strict_slice":
            for q in list(s5.threshold_sensitivity.quantiles):
                th = compute_train_thresholds(train_df, cor12_quantile=float(q), abs_t_quantile=float(q))
                sm = compute_strict_slice_mask(train_batch.cor12, train_batch.abs_t, th)
                met, summ = _run_single_training(
                    arm_name=f"strict_slice_q{int(float(q)*100)}",
                    seed=sens_seed,
                    cfg_s5=s5,
                    embedding_matrix=embedding_matrix,
                    chemistry_dense=chemistry_dense,
                    train_batch=train_batch,
                    val_batch=val_batch,
                    weights=np.ones(len(train_batch.y), dtype=np.float32),
                    train_selector=sm,
                    spearman_m=spearman_m,
                    spearman_vmin=spearman_vmin,
                )
                sens_rows.append(
                    {
                        "setting": f"q{int(float(q)*100)}",
                        "val_rmse": float(summ["best_val_rmse"]),
                        "val_mae": float(summ["best_val_mae"]),
                    }
                )
            # Include base q25 for reference.
            sens_rows.append(
                {
                    "setting": f"q{int(100*float(s5.thresholds.cor12_quantile))}",
                    "val_rmse": float(agg.loc[agg["arm"] == "strict_slice", "rmse_mean"].iloc[0]),
                    "val_mae": float(agg.loc[agg["arm"] == "strict_slice", "mae_mean"].iloc[0]),
                }
            )
        else:
            base = compute_weighted_full_weights(train_batch.cor12, train_batch.abs_t, thresholds)
            for pct in [5, 15]:
                floor = float(np.percentile(base, pct))
                w = base.copy()
                w[w < floor] = 0.0
                _met, summ = _run_single_training(
                    arm_name=f"weighted_full_floor{pct}",
                    seed=sens_seed,
                    cfg_s5=s5,
                    embedding_matrix=embedding_matrix,
                    chemistry_dense=chemistry_dense,
                    train_batch=train_batch,
                    val_batch=val_batch,
                    weights=w,
                    train_selector=np.ones(len(train_batch.y), dtype=bool),
                    spearman_m=spearman_m,
                    spearman_vmin=spearman_vmin,
                )
                sens_rows.append(
                    {"setting": f"weight_floor_p{pct}", "val_rmse": float(summ["best_val_rmse"]), "val_mae": float(summ["best_val_mae"])}
                )
            sens_rows.append(
                {
                    "setting": "base",
                    "val_rmse": float(agg.loc[agg["arm"] == "weighted_full", "rmse_mean"].iloc[0]),
                    "val_mae": float(agg.loc[agg["arm"] == "weighted_full", "mae_mean"].iloc[0]),
                }
            )
    sensitivity_df = pd.DataFrame(sens_rows)

    # Persist run-level metrics.
    run_dir = Path.cwd()
    metrics_path = run_dir / str(s5.metrics_filename)
    metrics_df.to_parquet(metrics_path, index=False)
    summaries_df.to_parquet(run_dir / "s5_summary.parquet", index=False)
    gate_df.to_parquet(run_dir / "s5_additive_gate.parquet", index=False)

    _plot_figures(figures_dir, metrics_df, summaries_df, sensitivity_df)

    chosen_row = agg.iloc[0].to_dict()
    additive_gate_pass = bool(gate_df.loc[gate_df["arm"] == chosen_arm, "passes_h_base_01_rmse"].iloc[0])
    policy_payload = {
        "status": "locked",
        "emitted_by": "stage5",
        "row_quality_policy": chosen_arm,
        "organism_pool": "full",
        "weighting_thresholds": thresholds,
        "effective_sample_size_train": arm_effective_n,
        "chemistry_multihot_length": int(chem_len),
        "chemistry_multihot_role_distinguished": False,
        "additive_gate": gate_df.to_dict(orient="records"),
        "drop_cause_attribution_strict_slice": drop_causes,
        "threshold_sensitivity": sensitivity_df.to_dict(orient="records"),
        "split_protocol_id": locked["protocol_id"],
        "feature_contract_artifact_id": artifact_id,
    }
    _write_quality_policy(output_quality_policy, policy_payload)

    report_payload = {
        "chosen_arm": chosen_arm,
        "chosen_arm_rmse_mean": float(chosen_row["rmse_mean"]),
        "chosen_arm_mae_mean": float(chosen_row["mae_mean"]),
        "additive_gate_pass": additive_gate_pass,
        "thresholds": thresholds,
    }
    _write_report(report_path, report_payload)
    _ensure_parent(Path(str(s5.decision_path)))

    log.info("S5 complete.")
    log.info("  chosen_arm=%s", chosen_arm)
    log.info("  wrote %s", output_quality_policy)
