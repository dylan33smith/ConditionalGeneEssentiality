"""T1-A — Granularity Test (Hypothesis H-ENC-01).

Compares:
  A1 (media_id):  per-media learnable embedding indexed by `media` string.
                  Val medias not in train → UNK (idx 0).
  A2 (multihot):  the locked S4 425-dim canonical multihot (medium + stressor).

Controls held fixed (S5 locks):
  - locked split (multi_org_balanced)
  - weighted_full quality weighting
  - shallow concat-linear MLP head
  - same seeds, epochs, optimizer
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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
    build_media_id_chemistry_matrix,
    load_t1_inputs,
    make_media_id_model,
    make_multihot_model,
)


log = logging.getLogger(__name__)

ARM_MULTIHOT = "multihot_canonical_id"
ARM_MEDIA_ID = "media_id"


# ---------------------------------------------------------------------------
# Per-arm training
# ---------------------------------------------------------------------------

def _run_one(
    *,
    arm_name: str,
    seed: int,
    inputs: T1Inputs,
    chemistry_matrix: np.ndarray,
    model,
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


# ---------------------------------------------------------------------------
# Post-training analyses
# ---------------------------------------------------------------------------

def _bootstrap_rmse_ci(val_true: np.ndarray, val_pred: np.ndarray, *, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap CI of RMSE/MAE by resampling rows with replacement."""
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


def _homology_similarity_per_val_gene(
    val_gene_keys: np.ndarray,
    val_org_ids: np.ndarray,
    train_orgs: list[str],
    embedding_dir: Path,
) -> dict[str, float]:
    """For each unique val gene_key: max cosine to any train gene's embedding.

    This is the same shape as S1's cross_org_max_cosine but reused here so
    T1-A can stratify metrics by similarity bin.
    """
    from src.experiments.stage1.analyses import (
        cross_org_max_cosine,
        load_org_embeddings,
    )
    val_orgs = sorted(set(val_org_ids.tolist()))
    homology_map: dict[str, float] = {}
    for val_org in val_orgs:
        df = cross_org_max_cosine(
            val_org=val_org,
            train_orgs=train_orgs,
            embedding_dir=embedding_dir,
            max_train_genes=2000,
        )
        for _, r in df.iterrows():
            homology_map[str(r["gene_key"])] = float(r["max_cosine"])
    return homology_map


def _bin_metrics_by_similarity(
    val_true: np.ndarray,
    val_pred: np.ndarray,
    val_gene_keys: np.ndarray,
    homology_map: dict[str, float],
    *,
    bins: tuple[float, ...] = (0.0, 0.5, 0.7, 0.85, 1.01),
) -> pd.DataFrame:
    sims = np.array([homology_map.get(g, np.nan) for g in val_gene_keys])
    df = pd.DataFrame({"y_true": val_true, "y_pred": val_pred, "sim": sims}).dropna()
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (df["sim"] >= lo) & (df["sim"] < hi)
        sub = df[m]
        if len(sub) == 0:
            out.append({"bin_lo": lo, "bin_hi": hi, "n_rows": 0,
                        "rmse": float("nan"), "mae": float("nan")})
            continue
        rmse = float(np.sqrt(np.mean((sub["y_true"] - sub["y_pred"]) ** 2)))
        mae = float(np.mean(np.abs(sub["y_true"] - sub["y_pred"])))
        out.append({"bin_lo": lo, "bin_hi": hi, "n_rows": int(len(sub)),
                    "rmse": rmse, "mae": mae})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Promotion-rule evaluation
# ---------------------------------------------------------------------------

def _evaluate_promotion(
    multihot_metrics: dict,
    media_id_metrics: dict,
    eval_policy_path: Path,
    locked_protocol_id: str,
) -> dict:
    """Apply the S2-locked promotion rule for multi_org_balanced.

    The rule (S2-DEC-001): a model must beat the best non-global baseline
    on both RMSE and MAE by the per-protocol thresholds.
    """
    eval_policy = yaml.safe_load(eval_policy_path.read_text())
    gain_thresholds = eval_policy["gain_thresholds_per_protocol"][locked_protocol_id]
    rmse_threshold = float(gain_thresholds["rmse"])
    mae_threshold = float(gain_thresholds["mae"])

    rmse_gap = float(media_id_metrics["rmse_mean"] - multihot_metrics["rmse_mean"])
    mae_gap = float(media_id_metrics["mae_mean"] - multihot_metrics["mae_mean"])
    multihot_beats_media_id_rmse = rmse_gap > rmse_threshold
    multihot_beats_media_id_mae = mae_gap > mae_threshold

    # Also check vs the locked S2 best_non_global baseline (embedding_nn for multi_org_balanced).
    best_non_global = float(gain_thresholds["best_baseline_rmse"])
    multihot_beats_best_baseline = float(multihot_metrics["rmse_mean"]) < (best_non_global - rmse_threshold)

    decision = "no_winner"
    if multihot_beats_media_id_rmse and multihot_beats_media_id_mae:
        decision = "promote_multihot_canonical_id"
    elif (rmse_gap < -rmse_threshold) and (mae_gap < -mae_threshold):
        decision = "promote_media_id"

    return {
        "rmse_gap_media_id_minus_multihot": rmse_gap,
        "mae_gap_media_id_minus_multihot": mae_gap,
        "rmse_gap_meets_threshold": bool(abs(rmse_gap) > rmse_threshold),
        "mae_gap_meets_threshold": bool(abs(mae_gap) > mae_threshold),
        "rmse_threshold_locked_s2": rmse_threshold,
        "mae_threshold_locked_s2": mae_threshold,
        "best_non_global_baseline_rmse_s2": best_non_global,
        "multihot_beats_best_non_global_baseline": multihot_beats_best_baseline,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_figures(figures_dir: Path, metrics_df: pd.DataFrame, summaries_df: pd.DataFrame,
                  per_org: pd.DataFrame, homology_bins: pd.DataFrame) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 01: best val RMSE+MAE per arm, error bars across seeds
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for col, ax, title in [
        ("best_val_rmse", axs[0], "Best val RMSE per arm"),
        ("best_val_mae", axs[1], "Best val MAE per arm"),
    ]:
        agg = summaries_df.groupby("arm")[col].agg(["mean", "std"]).reset_index()
        x = np.arange(len(agg))
        ax.bar(x, agg["mean"], yerr=agg["std"], capsize=6, color=["C0", "C1"])
        ax.set_xticks(x)
        ax.set_xticklabels(agg["arm"], rotation=15)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    p1 = figures_dir / "01_val_metrics_per_arm.png"
    fig.tight_layout(); fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)
    summaries_df.to_csv(p1.with_suffix(".csv"), index=False)

    # 02: per-organism val RMSE bars per arm
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = per_org.pivot_table(index="orgId", columns="arm", values="val_rmse", aggfunc="mean")
    x = np.arange(len(pivot))
    w = 0.4
    ax.bar(x - w/2, pivot.iloc[:, 0].to_numpy(), w, label=str(pivot.columns[0]), color="C0")
    ax.bar(x + w/2, pivot.iloc[:, 1].to_numpy(), w, label=str(pivot.columns[1]), color="C1")
    ax.set_xticks(x); ax.set_xticklabels(pivot.index, rotation=20)
    ax.set_title("Per-organism val RMSE (best epoch, averaged across seeds)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    p2 = figures_dir / "02_per_org_val_rmse_per_arm.png"
    fig.tight_layout(); fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)
    per_org.to_csv(p2.with_suffix(".csv"), index=False)

    # 03: homology-bin RMSE per arm
    fig, ax = plt.subplots(figsize=(8, 4))
    bin_labels = homology_bins[homology_bins["arm"] == ARM_MULTIHOT].apply(
        lambda r: f"[{r['bin_lo']:.2f},{r['bin_hi']:.2f})", axis=1
    ).tolist()
    multihot_rmse = homology_bins[homology_bins["arm"] == ARM_MULTIHOT]["rmse"].to_numpy()
    media_rmse = homology_bins[homology_bins["arm"] == ARM_MEDIA_ID]["rmse"].to_numpy()
    multihot_n = homology_bins[homology_bins["arm"] == ARM_MULTIHOT]["n_rows"].to_numpy()
    x = np.arange(len(bin_labels)); w = 0.4
    ax.bar(x - w/2, multihot_rmse, w, label=ARM_MULTIHOT, color="C0")
    ax.bar(x + w/2, media_rmse, w, label=ARM_MEDIA_ID, color="C1")
    ax.set_xticks(x); ax.set_xticklabels(bin_labels)
    for xi, n in zip(x, multihot_n):
        ax.text(xi, ax.get_ylim()[1] * 0.95, f"n={n}", ha="center", fontsize=7)
    ax.set_xlabel("Cosine similarity to nearest train gene (bins)")
    ax.set_ylabel("val RMSE")
    ax.set_title("Per-arm val RMSE by homology bin")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    p3 = figures_dir / "03_homology_bin_metrics_per_arm.png"
    fig.tight_layout(); fig.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(fig)
    homology_bins.to_csv(p3.with_suffix(".csv"), index=False)

    # 04: train/val curves
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for (arm, seed), sub in metrics_df.groupby(["arm", "seed"]):
        ls = "-" if arm == ARM_MULTIHOT else "--"
        c = "C0" if arm == ARM_MULTIHOT else "C1"
        axs[0].plot(sub["epoch"], sub["train_rmse"], ls=ls, alpha=0.5, color=c,
                    label=f"{arm}-s{seed}-train")
        axs[0].plot(sub["epoch"], sub["val_rmse"], ls=ls, alpha=0.9, color=c, lw=2,
                    label=f"{arm}-s{seed}-val")
        axs[1].plot(sub["epoch"], sub["train_mae"], ls=ls, alpha=0.5, color=c)
        axs[1].plot(sub["epoch"], sub["val_mae"], ls=ls, alpha=0.9, color=c, lw=2)
    axs[0].set_title("RMSE by epoch (per arm × seed)"); axs[1].set_title("MAE by epoch")
    for ax in axs:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=6, ncol=2, loc="upper right")
    p4 = figures_dir / "04_train_val_curves_per_arm.png"
    fig.tight_layout(); fig.savefig(p4, dpi=150, bbox_inches="tight"); plt.close(fig)
    metrics_df.to_csv(p4.with_suffix(".csv"), index=False)


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------

def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_"))}
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


def run_t1a(cfg, *,
            protocol_path: Path | None = None,
            output_root: str = "t1a",
            figures_dirname: str = "tier1_a",
            experiment_id_label: str = "T1-A_granularity",
            title: str = "T1-A Granularity Test (H-ENC-01)") -> dict:
    """Run T1-A end-to-end. Returns the comparison payload.

    Args:
        cfg: Hydra config (unused at present; reserved for future overrides).
        protocol_path: Override the locked split protocol. T1-A uses the locked
            `multi_org_balanced`; T1-A.2 passes the largest_by_rows diagnostic
            protocol to stress-test H-ENC-01 on unseen val media.
        output_root: Folder name under artifacts/runs/ and artifacts/cache/.
        figures_dirname: Folder name under research_log/figures/.
        experiment_id_label: Label used in the summary JSON.
        title: Banner string in logs.
    """
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
    log.info("[1/6] loading shared T1 inputs")
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

    log.info("[2/6] building media-id chemistry matrix")
    media_id_chem, media_vocab = build_media_id_chemistry_matrix(inputs)
    log.info("    media vocab size=%d (including <UNK>)", len(media_vocab) + 1)

    # Model dims
    gene_dim = int(inputs.embedding_matrix.shape[1])
    chem_dim = int(inputs.chemistry_dense.shape[1])
    hidden_dim = 256
    dropout = 0.1
    media_embed_dim = chem_dim  # match arms' condition dim for fair comparison

    model_cfg = {"lr": 1e-3, "weight_decay": 1e-4, "batch_size": 8192,
                 "epochs": 8, "device": "auto"}
    seeds = [0, 1, 2]

    all_metrics: list[pd.DataFrame] = []
    all_summaries: list[dict] = []
    per_arm_predictions: dict[str, dict[int, np.ndarray]] = {ARM_MULTIHOT: {}, ARM_MEDIA_ID: {}}
    per_arm_true: dict[str, dict[int, np.ndarray]] = {ARM_MULTIHOT: {}, ARM_MEDIA_ID: {}}

    log.info("[3/6] training arm A2 = %s × %d seeds", ARM_MULTIHOT, len(seeds))
    for seed in seeds:
        model = make_multihot_model(
            gene_dim=gene_dim, chem_dim=chem_dim, hidden_dim=hidden_dim, dropout=dropout
        )
        metrics_df, summary = _run_one(
            arm_name=ARM_MULTIHOT, seed=seed, inputs=inputs,
            chemistry_matrix=inputs.chemistry_dense, model=model,
            weights=inputs.weighted_weights, cfg_model=model_cfg,
        )
        all_metrics.append(metrics_df)
        all_summaries.append({k: v for k, v in summary.items() if not k.startswith("_")} | {"arm": ARM_MULTIHOT})
        per_arm_predictions[ARM_MULTIHOT][seed] = summary["_best_val_pred"]
        per_arm_true[ARM_MULTIHOT][seed] = summary["_best_val_true"]
        log.info("    seed=%d best_val_rmse=%.4f mae=%.4f", seed,
                 summary["best_val_rmse"], summary["best_val_mae"])

    log.info("[4/6] training arm A1 = %s × %d seeds", ARM_MEDIA_ID, len(seeds))
    for seed in seeds:
        model = make_media_id_model(
            gene_dim=gene_dim,
            n_media_vocab=len(media_vocab) + 1,   # +1 for UNK at idx 0
            embed_dim=media_embed_dim,
            hidden_dim=hidden_dim, dropout=dropout,
        )
        metrics_df, summary = _run_one(
            arm_name=ARM_MEDIA_ID, seed=seed, inputs=inputs,
            chemistry_matrix=media_id_chem, model=model,
            weights=inputs.weighted_weights, cfg_model=model_cfg,
        )
        all_metrics.append(metrics_df)
        all_summaries.append({k: v for k, v in summary.items() if not k.startswith("_")} | {"arm": ARM_MEDIA_ID})
        per_arm_predictions[ARM_MEDIA_ID][seed] = summary["_best_val_pred"]
        per_arm_true[ARM_MEDIA_ID][seed] = summary["_best_val_true"]
        log.info("    seed=%d best_val_rmse=%.4f mae=%.4f", seed,
                 summary["best_val_rmse"], summary["best_val_mae"])

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summaries_df = pd.DataFrame(all_summaries)

    # Aggregate metrics per arm (mean ± std across seeds)
    agg = summaries_df.groupby("arm")[["best_val_rmse", "best_val_mae"]].agg(["mean", "std"])
    agg.columns = ["rmse_mean", "rmse_std", "mae_mean", "mae_std"]
    arm_metrics = agg.reset_index()
    log.info("[5/6] per-arm summary:\n%s", arm_metrics.to_string(index=False))

    # Bootstrap CI on best-seed predictions per arm (using seed=0 as representative)
    bootstrap_per_arm = {}
    for arm in [ARM_MULTIHOT, ARM_MEDIA_ID]:
        boot = _bootstrap_rmse_ci(
            per_arm_true[arm][0], per_arm_predictions[arm][0], n_boot=1000, seed=0
        )
        bootstrap_per_arm[arm] = boot
        log.info("    %s bootstrap (seed 0): RMSE [%.4f, %.4f]  MAE [%.4f, %.4f]",
                 arm, boot["rmse_ci_low"], boot["rmse_ci_high"],
                 boot["mae_ci_low"], boot["mae_ci_high"])

    # Per-organism breakdown averaged across seeds
    per_org_rows = []
    for arm in [ARM_MULTIHOT, ARM_MEDIA_ID]:
        for seed in seeds:
            sub = summaries_df[(summaries_df["arm"] == arm) & (summaries_df["seed"] == seed)]
            if sub.empty:
                continue
            # per_org_val_rmse_json is in best_summary via train_one_arm but only on metrics_df rows;
            # use the last-epoch row (best epoch is usually near the end)
            best_epoch_row = metrics_df[
                (metrics_df["arm"] == arm) & (metrics_df["seed"] == seed)
            ].sort_values("val_rmse").iloc[0]
            per_org_dict = json.loads(best_epoch_row["per_org_val_rmse_json"])
            for org, rmse in per_org_dict.items():
                per_org_rows.append({"arm": arm, "seed": seed, "orgId": org, "val_rmse": float(rmse)})
    per_org_df = pd.DataFrame(per_org_rows)
    per_org_summary = per_org_df.groupby(["orgId", "arm"])["val_rmse"].mean().reset_index()

    # Homology bins (use seed 0 predictions)
    log.info("[6/6] homology-bin breakdown")
    train_orgs = sorted(set(inputs.train_batch.org_id.tolist()))
    val_orgs = sorted(set(inputs.val_batch.org_id.tolist()))
    log.info("    computing per-val-gene homology (train_orgs=%d, val_orgs=%d)",
             len(train_orgs), len(val_orgs))
    homology_map = _homology_similarity_per_val_gene(
        val_gene_keys=inputs.val_batch.gene_key,
        val_org_ids=inputs.val_batch.org_id,
        train_orgs=train_orgs,
        embedding_dir=embedding_dir,
    )
    homology_bin_rows = []
    for arm in [ARM_MULTIHOT, ARM_MEDIA_ID]:
        bins_df = _bin_metrics_by_similarity(
            per_arm_true[arm][0], per_arm_predictions[arm][0],
            inputs.val_batch.gene_key, homology_map,
        )
        bins_df["arm"] = arm
        homology_bin_rows.append(bins_df)
    homology_bins_df = pd.concat(homology_bin_rows, ignore_index=True)
    log.info("    homology bin RMSEs:\n%s",
             homology_bins_df.pivot_table(index=["bin_lo", "bin_hi"], columns="arm",
                                          values="rmse").to_string())

    # Promotion-rule evaluation
    arm_dict = arm_metrics.set_index("arm").to_dict(orient="index")
    promotion = _evaluate_promotion(
        multihot_metrics=arm_dict[ARM_MULTIHOT],
        media_id_metrics=arm_dict[ARM_MEDIA_ID],
        eval_policy_path=eval_policy_path,
        locked_protocol_id=inputs.locked_protocol_id,
    )
    log.info("Promotion outcome: %s (RMSE gap = %.4f, threshold = %.4f)",
             promotion["decision"], promotion["rmse_gap_media_id_minus_multihot"],
             promotion["rmse_threshold_locked_s2"])

    # Save artifacts
    _plot_figures(figures_dir, metrics_df, summaries_df, per_org_summary, homology_bins_df)
    metrics_df.to_parquet(output_dir / "t1a_metrics.parquet", index=False)
    summaries_df.to_parquet(output_dir / "t1a_summaries.parquet", index=False)
    payload = {
        "experiment_id": experiment_id_label,
        "stage_or_tier": "T1",
        "hypothesis": "H-ENC-01",
        "locked_protocol_id": inputs.locked_protocol_id,
        "feature_contract_artifact_id": inputs.artifact_id,
        "seeds": seeds,
        "arm_metrics": _to_jsonable(arm_metrics.to_dict(orient="records")),
        "bootstrap_ci_per_arm": _to_jsonable(bootstrap_per_arm),
        "per_org_summary": _to_jsonable(per_org_summary.to_dict(orient="records")),
        "homology_bins": _to_jsonable(homology_bins_df.to_dict(orient="records")),
        "promotion": _to_jsonable(promotion),
        "n_train_rows": int(len(inputs.train_df)),
        "n_val_rows": int(len(inputs.val_df)),
        "media_vocab_size_including_unk": int(len(media_vocab) + 1),
    }
    (output_dir / "t1a_summary.json").write_text(json.dumps(payload, indent=2))
    log.info("T1-A complete. outputs at %s", output_dir)
    return payload
