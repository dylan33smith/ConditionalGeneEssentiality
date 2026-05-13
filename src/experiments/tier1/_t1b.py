"""T1-B — Numeric Transform Test (Hypothesis H-ENC-02).

Compares three numeric encodings of `amount` per (experiment, canonical_id) cell:

  A1 (raw):     amount as-is (range 0 to 2000+)
  A2 (log1p):   log1p(amount) — the locked S5 default
  A3 (bounded): clip(amount, p1, p99) then min-max to [0,1] using train-only stats

For (experiment, canonical_id) cells where amount is NaN (~98% of medium-chemistry
rows that don't have explicit concentrations), all three arms encode presence as 1.0.
So the per-arm difference is in how the ~2% of non-null-amount entries are scaled
(mostly stressor concentrations).

Controls held fixed:
  - Locked split (multi_org_balanced) and S4 artifact (de21504134c84a6c)
  - S5 weighted_full quality weighting, full org pool
  - Locked T1 encoder = multihot_canonical_id (per T1-DEC-002)
  - Same shallow concat-linear MLP, same seeds, same epochs

Mandatory diagnostic: after picking a T1-B winner, also run that arm on
`largest_by_rows` per T1-DEC-002 policy.
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

from src.data.datasets.build_s5_dataset import (
    S5TorchDataset,
    _load_vocab_map,
)
from src.training.train_loop import TrainLoopConfig, train_one_arm
from src.experiments.tier1._t1_common import (
    T1Inputs,
    load_t1_inputs,
    make_multihot_model,
)


log = logging.getLogger(__name__)

ARM_NAMES = ("raw", "log1p", "bounded")
ALL_TRANSFORMS = ("raw", "log1p", "bounded", "binary")
DEFAULT_BOUNDED_STATS_PATH = Path("artifacts/cache/t1b/bounded_reference_stats.json")


# ---------------------------------------------------------------------------
# Build per-arm chemistry matrices
# ---------------------------------------------------------------------------

def build_chemistry_matrix(
    *,
    experiment_chemistry_df: pd.DataFrame,
    vocab: dict[str, int],
    exp_to_row: dict[str, int],
    transform: str,
    bounded_stats: dict | None = None,
    vocab_size: int = 425,
) -> np.ndarray:
    """Build a dense (n_experiments, vocab_size) chemistry matrix.

    Args:
        transform: one of {"raw", "log1p", "bounded"}.
        bounded_stats: required if transform == "bounded".
            Must contain "p1" and "p99" fitted on train.

    Encoding policy for each (experiment_id, canonical_id) cell:
      - If amount is NaN: use 1.0 (presence indicator — same for all 3 arms)
      - If amount is finite:
        - raw:     amount
        - log1p:   log1p(amount)  (precomputed in `log1p_amount` column)
        - bounded: clip(amount, p1, p99), then (val - p1) / (p99 - p1)
    """
    if transform not in ALL_TRANSFORMS:
        raise ValueError(f"Unknown transform {transform!r}; must be one of {ALL_TRANSFORMS}")
    if transform == "bounded" and bounded_stats is None:
        raise ValueError("bounded transform requires bounded_stats")

    n_exp = max(exp_to_row.values()) + 1
    mat = np.zeros((n_exp, vocab_size), dtype=np.float32)

    df = experiment_chemistry_df
    exp_idx = df["experiment_id"].map(exp_to_row).to_numpy()
    canon_idx = df["canonical_id"].map(vocab).fillna(vocab.get("<UNK>", 0)).astype(np.int64).to_numpy()
    amt = df["amount"].to_numpy(dtype=np.float64)
    log1p_amt = df["log1p_amount"].to_numpy(dtype=np.float64)
    has_amt = np.isfinite(amt)

    # Default cell value: 1.0 (presence indicator)
    values = np.ones(len(df), dtype=np.float32)
    if transform == "raw":
        values[has_amt] = amt[has_amt].astype(np.float32)
    elif transform == "log1p":
        values[has_amt] = log1p_amt[has_amt].astype(np.float32)
    elif transform == "bounded":
        p1 = float(bounded_stats["p1"]); p99 = float(bounded_stats["p99"])
        clipped = np.clip(amt[has_amt], p1, p99)
        scale = max(p99 - p1, 1e-12)
        values[has_amt] = ((clipped - p1) / scale).astype(np.float32)
    elif transform == "binary":
        # Keep 1.0 everywhere chemistry is present, regardless of recorded amount.
        # This is the T1-A multihot encoding — no concentration information.
        pass
    # else: keep 1.0 (presence only)

    # Drop rows that didn't map to a valid experiment or canonical_id
    keep = (~pd.isna(exp_idx)) & (canon_idx < vocab_size)
    if not keep.all():
        log.warning("dropping %d chemistry rows with unmappable exp_id/canonical_id",
                    int((~keep).sum()))
    valid_exp_idx = exp_idx[keep].astype(np.int64)
    valid_canon_idx = canon_idx[keep]
    valid_values = values[keep]

    # Multiple chemistry rows per (experiment, canonical_id) shouldn't happen, but if
    # they do (e.g. via stressor + medium duplication), max-aggregate so we preserve
    # the strongest signal rather than zeroing it out via overwrite ordering.
    np.maximum.at(mat, (valid_exp_idx, valid_canon_idx), valid_values)
    return mat


# ---------------------------------------------------------------------------
# Training one arm
# ---------------------------------------------------------------------------

def _run_one(
    *,
    arm_name: str,
    seed: int,
    inputs: T1Inputs,
    chemistry_matrix: np.ndarray,
    weights: np.ndarray,
    cfg_model: dict,
) -> tuple[pd.DataFrame, dict]:
    gene_dim = int(inputs.embedding_matrix.shape[1])
    chem_dim = int(chemistry_matrix.shape[1])
    model = make_multihot_model(
        gene_dim=gene_dim, chem_dim=chem_dim,
        hidden_dim=int(cfg_model["hidden_dim"]),
        dropout=float(cfg_model["dropout"]),
    )
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
# Post-training utilities (shared with T1-A pattern)
# ---------------------------------------------------------------------------

def _bootstrap_rmse_ci(val_true, val_pred, *, n_boot=1000, seed=0) -> dict:
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


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_figures(figures_dir: Path, metrics_df: pd.DataFrame, summaries_df: pd.DataFrame) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 01: best val RMSE+MAE per arm, error bars across seeds
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for col, ax, title in [
        ("best_val_rmse", axs[0], "Best val RMSE per arm"),
        ("best_val_mae", axs[1], "Best val MAE per arm"),
    ]:
        agg = summaries_df.groupby("arm")[col].agg(["mean", "std"]).reset_index()
        # consistent order
        agg["arm"] = pd.Categorical(agg["arm"], categories=list(ARM_NAMES), ordered=True)
        agg = agg.sort_values("arm")
        x = np.arange(len(agg))
        ax.bar(x, agg["mean"], yerr=agg["std"], capsize=6, color=["C0", "C1", "C2"])
        ax.set_xticks(x); ax.set_xticklabels(agg["arm"].astype(str))
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    p1 = figures_dir / "01_val_metrics_per_arm.png"
    fig.tight_layout(); fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)
    summaries_df.to_csv(p1.with_suffix(".csv"), index=False)

    # 02: train/val curves
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"raw": "C0", "log1p": "C1", "bounded": "C2"}
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
    axs[0].legend(fontsize=7, ncol=2, loc="upper right")
    p2 = figures_dir / "02_train_val_curves_per_arm.png"
    fig.tight_layout(); fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)
    metrics_df.to_csv(p2.with_suffix(".csv"), index=False)


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------

def run_t1b(cfg, *,
            protocol_path: Path | None = None,
            output_root: str = "t1b",
            figures_dirname: str = "tier1_b",
            experiment_id_label: str = "T1-B_numeric_transform",
            title: str = "T1-B Numeric Transform Test (H-ENC-02)") -> dict:
    """Run T1-B end-to-end. Returns the comparison payload.

    Args mirror T1-A's run_t1a for protocol/path overrides.
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

    # Load chemistry rows, vocab, bounded stats
    feature_contract = yaml.safe_load(feature_contract_path.read_text())
    artifact_root = Path("data_contract/preprocessing") / inputs.artifact_id
    chem_df = pd.read_parquet(
        artifact_root / feature_contract["experiment_chemistry_table"]["path"]
    )
    vocab = _load_vocab_map(artifact_root / "canonical_id_vocab.json")
    bounded_stats = json.loads(DEFAULT_BOUNDED_STATS_PATH.read_text())
    log.info("    chem rows=%d vocab=%d bounded_p99=%.2f",
             len(chem_df), len(vocab), bounded_stats["p99"])

    # Build per-arm chemistry matrices
    log.info("[2/5] building chemistry matrices for 3 arms")
    chem_matrices = {}
    for arm in ARM_NAMES:
        m = build_chemistry_matrix(
            experiment_chemistry_df=chem_df,
            vocab=vocab,
            exp_to_row=inputs.exp_to_row,
            transform=arm,
            bounded_stats=bounded_stats,
            vocab_size=int(inputs.chemistry_dense.shape[1]),
        )
        log.info("    arm=%s shape=%s nnz=%d mean(nonzero)=%.4f max=%.4f",
                 arm, m.shape, int(np.count_nonzero(m)),
                 float(m[m != 0].mean()) if (m != 0).any() else 0.0,
                 float(m.max()))
        chem_matrices[arm] = m

    # Model config
    model_cfg = {
        "hidden_dim": 256, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4,
        "batch_size": 8192, "epochs": 8, "device": "auto",
    }
    seeds = [0, 1, 2]

    all_metrics: list[pd.DataFrame] = []
    all_summaries: list[dict] = []
    per_arm_predictions: dict[str, dict[int, np.ndarray]] = {a: {} for a in ARM_NAMES}
    per_arm_true: dict[str, dict[int, np.ndarray]] = {a: {} for a in ARM_NAMES}

    log.info("[3/5] training %d arms × %d seeds", len(ARM_NAMES), len(seeds))
    for arm in ARM_NAMES:
        log.info("──── arm = %s ────", arm)
        for seed in seeds:
            metrics_df, summary = _run_one(
                arm_name=arm, seed=seed, inputs=inputs,
                chemistry_matrix=chem_matrices[arm],
                weights=inputs.weighted_weights, cfg_model=model_cfg,
            )
            all_metrics.append(metrics_df)
            all_summaries.append({k: v for k, v in summary.items() if not k.startswith("_")} | {"arm": arm})
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
    arm_metrics["arm"] = pd.Categorical(arm_metrics["arm"], categories=list(ARM_NAMES), ordered=True)
    arm_metrics = arm_metrics.sort_values("arm").reset_index(drop=True)
    log.info("per-arm summary:\n%s", arm_metrics.to_string(index=False))

    bootstrap_per_arm = {}
    for arm in ARM_NAMES:
        bootstrap_per_arm[arm] = _bootstrap_rmse_ci(
            per_arm_true[arm][0], per_arm_predictions[arm][0], n_boot=1000, seed=0
        )
        b = bootstrap_per_arm[arm]
        log.info("    arm=%s bootstrap (seed 0): RMSE [%.4f, %.4f] MAE [%.4f, %.4f]",
                 arm, b["rmse_ci_low"], b["rmse_ci_high"], b["mae_ci_low"], b["mae_ci_high"])

    # Pick winner: lowest mean RMSE
    winner = str(arm_metrics.sort_values("rmse_mean").iloc[0]["arm"])
    runner_up = str(arm_metrics.sort_values("rmse_mean").iloc[1]["arm"])
    gap_rmse = float(arm_metrics.set_index("arm").loc[runner_up, "rmse_mean"]
                     - arm_metrics.set_index("arm").loc[winner, "rmse_mean"])
    gap_mae = float(arm_metrics.set_index("arm").loc[runner_up, "mae_mean"]
                    - arm_metrics.set_index("arm").loc[winner, "mae_mean"])

    eval_policy = yaml.safe_load(eval_policy_path.read_text())
    threshold_rmse = float(
        eval_policy["gain_thresholds_per_protocol"][inputs.locked_protocol_id]["rmse"]
    )
    threshold_mae = float(
        eval_policy["gain_thresholds_per_protocol"][inputs.locked_protocol_id]["mae"]
    )
    promotion = {
        "winner": winner,
        "runner_up": runner_up,
        "rmse_gap_runner_minus_winner": gap_rmse,
        "mae_gap_runner_minus_winner": gap_mae,
        "rmse_threshold_locked_s2": threshold_rmse,
        "mae_threshold_locked_s2": threshold_mae,
        "rmse_gap_meets_threshold": bool(gap_rmse > threshold_rmse),
        "mae_gap_meets_threshold": bool(gap_mae > threshold_mae),
        "decision": (
            f"promote_{winner}"
            if (gap_rmse > threshold_rmse and gap_mae > threshold_mae)
            else "no_winner_tiebreaker_required"
        ),
    }
    log.info("Promotion winner=%s vs runner_up=%s | RMSE gap=%.4f (thr=%.4f) MAE gap=%.4f (thr=%.4f) → %s",
             winner, runner_up, gap_rmse, threshold_rmse, gap_mae, threshold_mae,
             promotion["decision"])

    log.info("[5/5] writing artifacts")
    _plot_figures(figures_dir, metrics_df, summaries_df)
    metrics_df.to_parquet(output_dir / "t1b_metrics.parquet", index=False)
    summaries_df.to_parquet(output_dir / "t1b_summaries.parquet", index=False)
    payload = {
        "experiment_id": experiment_id_label,
        "stage_or_tier": "T1",
        "hypothesis": "H-ENC-02",
        "locked_protocol_id": inputs.locked_protocol_id,
        "feature_contract_artifact_id": inputs.artifact_id,
        "seeds": seeds,
        "arm_metrics": _to_jsonable(arm_metrics.to_dict(orient="records")),
        "bootstrap_ci_per_arm": _to_jsonable(bootstrap_per_arm),
        "promotion": _to_jsonable(promotion),
        "n_train_rows": int(len(inputs.train_df)),
        "n_val_rows": int(len(inputs.val_df)),
        "bounded_reference_stats": bounded_stats,
    }
    (output_dir / "t1b_summary.json").write_text(json.dumps(payload, indent=2))
    log.info("T1-B complete. outputs at %s", output_dir)
    return payload


# ---------------------------------------------------------------------------
# T1-B.3 — Binary vs log1p (controlled head-to-head)
# ---------------------------------------------------------------------------

def run_t1b3(cfg, *,
             output_root: str = "t1b3",
             figures_dirname: str = "tier1_b3",
             experiment_id_label: str = "T1-B.3_binary_vs_log1p",
             title: str = "T1-B.3 Concentration-Inclusion Test (binary vs log1p)") -> dict:
    """Controlled 2-arm comparison: pure binary presence vs log1p concentration.

    This is the side-by-side test that T1-B did not include. T1-B compared
    *transforms of* concentration (raw vs log1p vs bounded) but never tested
    "with vs without concentration info at all." T1-B.3 closes that gap.

    Arm A1 (binary): same encoding as T1-A's multihot — every present chemistry
        cell is exactly 1.0, regardless of whether `amount` is recorded.
    Arm A2 (log1p):  the currently locked default — presence (1.0) for cells
        with NaN amount, log1p(amount) for cells with recorded concentrations.
    """
    fitness_path = Path("data/derived/canonical/v0/fitness_experiment_long.parquet")
    feature_contract_path = Path("data_contract/feature_contract.yaml")
    protocol_path = Path("data_contract/splits/locked_protocol.yaml")
    eval_policy_path = Path("data_contract/policy/eval_policy.yaml")
    embedding_dir = Path("data/processed/ProtLM_embeddings_layer8")
    cache_dir = Path(f"artifacts/cache/{output_root}")
    figures_dir = Path(f"research_log/figures/{figures_dirname}")
    output_dir = Path(f"artifacts/runs/{output_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60); log.info(title); log.info("=" * 60)

    log.info("[1/4] loading shared T1 inputs")
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

    feature_contract = yaml.safe_load(feature_contract_path.read_text())
    artifact_root = Path("data_contract/preprocessing") / inputs.artifact_id
    chem_df = pd.read_parquet(
        artifact_root / feature_contract["experiment_chemistry_table"]["path"]
    )
    vocab = _load_vocab_map(artifact_root / "canonical_id_vocab.json")
    log.info("    chem rows=%d vocab=%d", len(chem_df), len(vocab))

    log.info("[2/4] building chemistry matrices (binary, log1p)")
    arms_to_run = ["binary", "log1p"]
    chem_matrices = {}
    for arm in arms_to_run:
        m = build_chemistry_matrix(
            experiment_chemistry_df=chem_df,
            vocab=vocab,
            exp_to_row=inputs.exp_to_row,
            transform=arm,
            bounded_stats=None,
            vocab_size=int(inputs.chemistry_dense.shape[1]),
        )
        log.info("    arm=%s shape=%s nnz=%d mean(nonzero)=%.4f max=%.4f",
                 arm, m.shape, int(np.count_nonzero(m)),
                 float(m[m != 0].mean()) if (m != 0).any() else 0.0,
                 float(m.max()))
        chem_matrices[arm] = m

    model_cfg = {
        "hidden_dim": 256, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4,
        "batch_size": 8192, "epochs": 8, "device": "auto",
    }
    seeds = [0, 1, 2]

    all_metrics: list[pd.DataFrame] = []
    all_summaries: list[dict] = []
    per_arm_predictions: dict[str, dict[int, np.ndarray]] = {a: {} for a in arms_to_run}
    per_arm_true: dict[str, dict[int, np.ndarray]] = {a: {} for a in arms_to_run}

    log.info("[3/4] training %d arms × %d seeds", len(arms_to_run), len(seeds))
    for arm in arms_to_run:
        log.info("──── arm = %s ────", arm)
        for seed in seeds:
            metrics_df, summary = _run_one(
                arm_name=arm, seed=seed, inputs=inputs,
                chemistry_matrix=chem_matrices[arm],
                weights=inputs.weighted_weights, cfg_model=model_cfg,
            )
            all_metrics.append(metrics_df)
            all_summaries.append({k: v for k, v in summary.items() if not k.startswith("_")} | {"arm": arm})
            per_arm_predictions[arm][seed] = summary["_best_val_pred"]
            per_arm_true[arm][seed] = summary["_best_val_true"]
            log.info("    arm=%s seed=%d best_val_rmse=%.4f mae=%.4f",
                     arm, seed, summary["best_val_rmse"], summary["best_val_mae"])

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summaries_df = pd.DataFrame(all_summaries)

    log.info("[4/4] aggregating + bootstrap")
    agg = summaries_df.groupby("arm")[["best_val_rmse", "best_val_mae"]].agg(["mean", "std"])
    agg.columns = ["rmse_mean", "rmse_std", "mae_mean", "mae_std"]
    arm_metrics = agg.reset_index()
    log.info("per-arm summary:\n%s", arm_metrics.to_string(index=False))

    bootstrap_per_arm = {}
    for arm in arms_to_run:
        bootstrap_per_arm[arm] = _bootstrap_rmse_ci(
            per_arm_true[arm][0], per_arm_predictions[arm][0], n_boot=1000, seed=0
        )
        b = bootstrap_per_arm[arm]
        log.info("    arm=%s bootstrap (seed 0): RMSE [%.4f, %.4f] MAE [%.4f, %.4f]",
                 arm, b["rmse_ci_low"], b["rmse_ci_high"], b["mae_ci_low"], b["mae_ci_high"])

    # Decision logic: binary is the "simpler / no-concentration" arm; log1p is "includes
    # unit-mixed concentrations". Tie → prefer binary on epistemic grounds (no
    # unit-mixing concern). Log1p must beat binary by threshold to justify
    # including concentrations.
    rmse_log1p = float(arm_metrics.set_index("arm").loc["log1p", "rmse_mean"])
    rmse_binary = float(arm_metrics.set_index("arm").loc["binary", "rmse_mean"])
    mae_log1p = float(arm_metrics.set_index("arm").loc["log1p", "mae_mean"])
    mae_binary = float(arm_metrics.set_index("arm").loc["binary", "mae_mean"])
    eval_policy = yaml.safe_load(eval_policy_path.read_text())
    th = eval_policy["gain_thresholds_per_protocol"][inputs.locked_protocol_id]
    threshold_rmse = float(th["rmse"]); threshold_mae = float(th["mae"])

    rmse_gap_binary_minus_log1p = rmse_binary - rmse_log1p
    mae_gap_binary_minus_log1p = mae_binary - mae_log1p
    log1p_clearly_better = (
        rmse_gap_binary_minus_log1p > threshold_rmse
        and mae_gap_binary_minus_log1p > threshold_mae
    )
    decision = "promote_log1p_keep_concentrations" if log1p_clearly_better else "promote_binary_drop_concentrations"

    # CI overlap check (binary "ties" if log1p doesn't statistically beat it)
    binary_ci_hi = bootstrap_per_arm["binary"]["rmse_ci_high"]
    log1p_ci_lo = bootstrap_per_arm["log1p"]["rmse_ci_low"]
    ci_overlap_rmse = log1p_ci_lo < binary_ci_hi  # overlap if log1p lower bound below binary upper bound
    log.info("Decision: %s | RMSE gap (binary - log1p) = %.4f (thr=%.4f) | "
             "CI overlap on RMSE: %s",
             decision, rmse_gap_binary_minus_log1p, threshold_rmse, ci_overlap_rmse)

    _plot_figures(figures_dir, metrics_df, summaries_df)
    metrics_df.to_parquet(output_dir / "t1b3_metrics.parquet", index=False)
    summaries_df.to_parquet(output_dir / "t1b3_summaries.parquet", index=False)
    payload = {
        "experiment_id": experiment_id_label,
        "stage_or_tier": "T1",
        "hypothesis": "H-ENC-02 (concentration-inclusion controlled test)",
        "locked_protocol_id": inputs.locked_protocol_id,
        "feature_contract_artifact_id": inputs.artifact_id,
        "seeds": seeds,
        "arm_metrics": _to_jsonable(arm_metrics.to_dict(orient="records")),
        "bootstrap_ci_per_arm": _to_jsonable(bootstrap_per_arm),
        "comparison": {
            "rmse_gap_binary_minus_log1p": rmse_gap_binary_minus_log1p,
            "mae_gap_binary_minus_log1p": mae_gap_binary_minus_log1p,
            "rmse_threshold_locked_s2": threshold_rmse,
            "mae_threshold_locked_s2": threshold_mae,
            "log1p_clearly_better": bool(log1p_clearly_better),
            "ci_overlap_rmse": bool(ci_overlap_rmse),
            "decision": decision,
        },
        "n_train_rows": int(len(inputs.train_df)),
        "n_val_rows": int(len(inputs.val_df)),
    }
    (output_dir / "t1b3_summary.json").write_text(json.dumps(payload, indent=2))
    log.info("T1-B.3 complete. outputs at %s", output_dir)
    return payload
