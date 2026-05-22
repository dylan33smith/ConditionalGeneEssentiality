"""T7-prep — Diagnostics before launching ranking-objective experiments.

Runs four diagnostic analyses (some require GPU) and writes outputs to
artifacts/runs/t7_prep/ for the final report:

  D1: Null-baseline comparison
      Compute within-gene Spearman for: random, global mean, per-org mean,
      per-cond mean (train), per-cond mean (val, leaky), additive baseline.
      Also computes for our best model (loaded predictions or freshly
      trained T5-A locked architecture).

  D2: Within-org cross-condition diagnostic
      Train the locked T5-A architecture on a within-organism cross-
      condition split (hold out 20% of experiments per org).
      Compare within-gene Spearman to the cross-organism baseline from T5-A.

  D3: Noise floor from biological replicates
      For (org, expDesc, media) combos with replicates, compute the
      cross-replicate Spearman per gene. Upper bound on what any model
      can achieve.

  D4: IQR distribution analysis
      Per-organism, per-gene IQR histograms + tail behavior. Informs which
      genes carry meaningful conditional signal.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

from src.data.datasets.build_s5_dataset import S5TorchDataset
from src.training.train_loop import TrainLoopConfig, train_one_arm
from src.experiments.tier2._t2_common import (
    bootstrap_ci, get_binary_chemistry, load_t2_inputs,
)
from src.experiments.tier5._t5_common import AdapterResidualMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("artifacts/runs/t7_prep")
FIGURES_DIR = Path("research_log/figures/t7_prep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Eligibility filter from eval policy (multi_org_balanced)
V_MIN = 0.1992
M_MIN = 5


def within_gene_spearman(
    df: pd.DataFrame,
    pred_col: str,
    *,
    min_iqr: float = V_MIN,
    min_n: int = M_MIN,
    fit_col: str = "fit",
    gene_col: str = "gene_key",
) -> tuple[float, int, int]:
    """Compute mean within-gene Spearman.

    Returns: (mean_spearman, n_eligible, n_computed).
    """
    vals = []
    eligible = 0
    for _gene, sub in df.groupby(gene_col):
        if len(sub) < min_n:
            continue
        iqr = float(np.percentile(sub[fit_col], 75) - np.percentile(sub[fit_col], 25))
        if iqr < min_iqr:
            continue
        eligible += 1
        if sub[pred_col].nunique() <= 1:
            continue
        r, _ = spearmanr(sub[fit_col].to_numpy(), sub[pred_col].to_numpy())
        if not np.isnan(r):
            vals.append(float(r))
    return float(np.mean(vals)) if vals else float("nan"), eligible, len(vals)


def bootstrap_within_gene_spearman(
    df: pd.DataFrame, pred_col: str, n_boot: int = 200, seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI over genes."""
    eligible_genes = []
    per_gene_sp = {}
    for gene, sub in df.groupby("gene_key"):
        if len(sub) < M_MIN:
            continue
        iqr = float(np.percentile(sub["fit"], 75) - np.percentile(sub["fit"], 25))
        if iqr < V_MIN:
            continue
        if sub[pred_col].nunique() <= 1:
            continue
        r, _ = spearmanr(sub["fit"].to_numpy(), sub[pred_col].to_numpy())
        if not np.isnan(r):
            per_gene_sp[gene] = r
            eligible_genes.append(gene)
    if not eligible_genes:
        return float("nan"), float("nan")
    arr = np.array([per_gene_sp[g] for g in eligible_genes])
    rng = np.random.default_rng(seed)
    n = len(arr)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        samples[b] = arr[idx].mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


# ============================================================
# D1: Null baseline + model comparison
# ============================================================

def diagnostic_1_baselines() -> dict:
    log.info("=" * 60)
    log.info("D1: Null baseline + model comparison")
    log.info("=" * 60)

    locked = yaml.safe_load(Path("data_contract/splits/locked_protocol.yaml").read_text())
    val_orgs = set(locked["val_org_ids"])

    # Use the pipeline-loaded inputs so we have the exact rows that model
    # predictions correspond to (some rows are filtered by make_row_batch).
    log.info("  loading inputs (this ensures alignment with model predictions)...")
    inputs = load_t2_inputs(cache_subdir="t7_prep")
    val_batch = inputs.val_batch
    val_df = pd.DataFrame({
        "row_index": val_batch.row_index,
        "gene_key": val_batch.gene_key,
        "orgId": val_batch.org_id,
        "exp_idx": val_batch.exp_idx,
        "fit": val_batch.y.astype(np.float64),
    })
    # Map exp_idx back to experiment_id for per-condition baselines
    row_to_exp_id = {v: k for k, v in inputs.exp_to_row.items()}
    val_df["exp_key"] = val_df["exp_idx"].map(row_to_exp_id)

    full_df = pd.read_parquet("data/derived/canonical/v0/fitness_experiment_long.parquet")
    train_df = full_df[~full_df["orgId"].isin(val_orgs)].copy()
    train_df["gene_key"] = train_df["orgId"].astype(str) + ":" + train_df["locusId"].astype(str)
    # Add the hashed experiment_id (same as val side) so per-condition maps line up
    from src.data.datasets.build_s5_dataset import add_experiment_id
    train_df = add_experiment_id(train_df)
    train_df["exp_key"] = train_df["experiment_id"].astype(str)

    log.info("  val rows=%d, val genes=%d, val orgs=%d, val experiments=%d",
             len(val_df), val_df["gene_key"].nunique(),
             val_df["orgId"].nunique(), val_df["exp_key"].nunique())

    # Build all baseline predictions
    rng = np.random.default_rng(0)
    val_df["pred_random"] = rng.normal(size=len(val_df))
    val_df["pred_global"] = train_df["fit"].mean()
    val_df["pred_per_org"] = val_df["orgId"].map(train_df.groupby("orgId")["fit"].mean()).fillna(0.0)
    train_per_exp_mean = train_df.groupby("exp_key")["fit"].mean().to_dict()
    val_df["pred_per_cond_train"] = val_df["exp_key"].map(train_per_exp_mean).fillna(0.0)
    val_per_exp_mean = val_df.groupby("exp_key")["fit"].mean().to_dict()
    val_df["pred_per_cond_val"] = val_df["exp_key"].map(val_per_exp_mean).fillna(0.0)

    # Additive baseline from train: fit ~ per_org + per_cond
    org_mean = train_df.groupby("orgId")["fit"].mean()
    cond_mean = train_df.groupby("exp_key")["fit"].mean()
    grand = train_df["fit"].mean()
    org_eff = (org_mean - grand).to_dict()
    cond_eff = (cond_mean - grand).to_dict()
    val_df["pred_additive"] = (
        grand
        + val_df["orgId"].map(org_eff).fillna(0.0)
        + val_df["exp_key"].map(cond_eff).fillna(0.0)
    )

    # Per-gene mean from VAL (leaky oracle — uses val info, illustrative ceiling)
    val_gene_mean = val_df.groupby("gene_key")["fit"].mean().to_dict()
    val_df["pred_per_gene_val"] = val_df["gene_key"].map(val_gene_mean)

    # Try to load best model predictions (will train fresh if missing)
    pred_path = OUTPUT_DIR / "t5a_locked_val_predictions.parquet"
    if pred_path.exists():
        log.info("  loading T5-A locked model predictions from %s", pred_path)
        model_preds = pd.read_parquet(pred_path)
        # model_preds[row_index] is the positional index within val_batch
        # val_df is already in val_batch order; just attach
        assert len(model_preds) == len(val_df), (
            f"prediction count {len(model_preds)} != val_df {len(val_df)}"
        )
        val_df["pred_model"] = model_preds["pred_model"].to_numpy()
    else:
        log.info("  no cached model predictions; will train T5-A locked + save")
        model_preds = _train_t5a_and_save_predictions()
        val_df["pred_model"] = model_preds

    # Compute Spearman for each baseline
    baselines = [
        "pred_random", "pred_global", "pred_per_org",
        "pred_per_cond_train", "pred_per_cond_val", "pred_additive",
        "pred_per_gene_val", "pred_model",
    ]
    results = []
    for b in baselines:
        sp, eligible, computed = within_gene_spearman(val_df, b)
        ci_low, ci_high = bootstrap_within_gene_spearman(val_df, b, n_boot=200)
        results.append({
            "baseline": b,
            "within_gene_spearman_mean": sp,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_eligible": eligible,
            "n_computed": computed,
        })
        log.info("    %-22s sp=%.4f [%.4f, %.4f] (n=%d)",
                 b, sp, ci_low, ci_high, computed)

    # Per-org breakdown for the model
    per_org = []
    for org in sorted(val_orgs):
        sub = val_df[val_df["orgId"] == org].copy()
        sp_model, _, _ = within_gene_spearman(sub, "pred_model")
        sp_baseline, _, _ = within_gene_spearman(sub, "pred_per_cond_train")
        per_org.append({
            "org": org, "n_rows": len(sub),
            "model_sp": sp_model, "per_cond_baseline_sp": sp_baseline,
            "gap": sp_model - sp_baseline,
        })
        log.info("    [%s] model=%.4f, per-cond-mean=%.4f, gap=%+.4f",
                 org, sp_model, sp_baseline, sp_model - sp_baseline)

    out = {
        "baseline_results": results,
        "per_org_breakdown": per_org,
        "n_val_rows": len(val_df),
        "n_val_genes": int(val_df["gene_key"].nunique()),
    }
    (OUTPUT_DIR / "d1_baselines.json").write_text(json.dumps(out, indent=2))
    return out


def _train_t5a_and_save_predictions() -> np.ndarray:
    """Train the T5-A locked architecture once and save val predictions."""
    log.info("  training T5-A locked architecture (seed=0)...")
    inputs = load_t2_inputs(cache_subdir="t7_prep")
    chem = get_binary_chemistry(inputs)
    gene_dim = int(inputs.embedding_matrix.shape[1])
    chem_dim = int(chem.shape[1])

    model = AdapterResidualMLP(
        gene_dim=gene_dim, chem_dim=chem_dim,
        hidden_dim=512, n_blocks=1, dropout=0.1,
        adapter_hidden=1024, adapter_out=512,
        adapter_n_hidden_layers=1, adapter_layernorm=False,
    )

    train_ds = S5TorchDataset(
        inputs.train_batch, embedding_matrix=inputs.embedding_matrix,
        chemistry_matrix=chem, weights=inputs.weighted_weights,
    )
    val_ds = S5TorchDataset(
        inputs.val_batch, embedding_matrix=inputs.embedding_matrix,
        chemistry_matrix=chem,
        weights=np.ones(len(inputs.val_batch.y), dtype=np.float32),
    )
    cfg = TrainLoopConfig(
        lr=1e-3, weight_decay=1e-4, batch_size=8192,
        epochs=8, device="auto",
    )
    _, summary = train_one_arm(
        arm_name="t5a_locked", seed=0, model=model,
        train_dataset=train_ds, val_dataset=val_ds,
        val_gene_keys=inputs.val_batch.gene_key,
        val_org_ids=inputs.val_batch.org_id,
        spearman_min_conditions=inputs.spearman_m,
        spearman_min_iqr=inputs.spearman_vmin,
        config=cfg,
    )
    pred = summary["_best_val_pred"]
    # Save predictions
    pd.DataFrame({
        "row_index": np.arange(len(pred)),
        "pred_model": pred,
    }).to_parquet(OUTPUT_DIR / "t5a_locked_val_predictions.parquet", index=False)
    log.info("  T5-A locked val RMSE=%.4f, MAE=%.4f",
             summary["best_val_rmse"], summary["best_val_mae"])
    return pred


# ============================================================
# D2: Within-org cross-condition diagnostic
# ============================================================

def diagnostic_2_cross_condition() -> dict:
    log.info("=" * 60)
    log.info("D2: Within-org cross-condition split — train + eval")
    log.info("=" * 60)

    locked = yaml.safe_load(Path("data_contract/splits/locked_protocol.yaml").read_text())
    val_orgs_locked = set(locked["val_org_ids"])
    test_orgs_locked = set(locked["test_org_ids"])

    df = pd.read_parquet("data/derived/canonical/v0/fitness_experiment_long.parquet")
    # Only use training orgs (no val/test orgs)
    train_orgs_df = df[~df["orgId"].isin(val_orgs_locked | test_orgs_locked)].copy()
    train_orgs_df["exp_key"] = train_orgs_df["expName"].astype(str)
    train_orgs_df["gene_key"] = (
        train_orgs_df["orgId"].astype(str) + ":" + train_orgs_df["locusId"].astype(str)
    )

    log.info("  training-org rows: %d, orgs: %d, experiments: %d",
             len(train_orgs_df), train_orgs_df["orgId"].nunique(),
             train_orgs_df["exp_key"].nunique())

    # Hold out 20% of unique experiments per organism (seed=0)
    rng = np.random.default_rng(0)
    val_exp_keys = set()
    for org, sub in train_orgs_df.groupby("orgId"):
        uniq = sub["exp_key"].unique()
        n_val = max(1, int(0.2 * len(uniq)))
        val_choice = rng.choice(uniq, size=n_val, replace=False)
        val_exp_keys.update(val_choice.tolist())

    is_val = train_orgs_df["exp_key"].isin(val_exp_keys)
    train_split = train_orgs_df[~is_val].copy()
    val_split = train_orgs_df[is_val].copy()
    log.info("  D2 train rows=%d, val rows=%d (%d val experiments)",
             len(train_split), len(val_split), len(val_exp_keys))

    # We need the same machinery as T5-A but with this custom split. Easiest:
    # rebuild minimal pipeline using existing helpers but with custom row batches.
    from src.data.datasets.build_s5_dataset import (
        make_row_batch, build_or_load_experiment_multihot,
        dense_chemistry_from_csr, _load_concatenated_embeddings, add_experiment_id,
    )
    from src.experiments.stage5.run import (
        compute_train_thresholds, compute_weighted_full_weights,
    )

    # Reuse the locked feature contract artifact for chemistry
    feature_contract = yaml.safe_load(
        Path("data_contract/feature_contract.yaml").read_text()
    )
    artifact_id = str(feature_contract["artifact_id"])
    artifact_root = Path("data_contract/preprocessing") / artifact_id
    chemistry_parquet = artifact_root / feature_contract["experiment_chemistry_table"]["path"]
    canonical_vocab_path = artifact_root / "canonical_id_vocab.json"

    train_split = add_experiment_id(train_split)
    val_split = add_experiment_id(val_split)

    all_orgs = sorted(set(train_split["orgId"]).union(val_split["orgId"]))
    embedding_matrix, gene_key_to_idx = _load_concatenated_embeddings(
        all_orgs, Path("data/processed/ProtLM_embeddings_layer8"),
        filename_suffix="_proteomelm.pt",
    )

    exp_ids = list(
        set(train_split["experiment_id"]).union(set(val_split["experiment_id"]))
    )
    chem_csr, exp_to_row, _ = build_or_load_experiment_multihot(
        chemistry_parquet_path=chemistry_parquet,
        canonical_vocab_json_path=canonical_vocab_path,
        target_experiment_ids=exp_ids,
        cache_dir=Path("artifacts/cache/t7_prep_d2") / artifact_id,
        sparse_cache_filename="multihot.npz",
    )
    chemistry_dense = dense_chemistry_from_csr(chem_csr)

    train_batch = make_row_batch(
        train_split, gene_key_to_idx=gene_key_to_idx,
        experiment_id_to_row=exp_to_row,
    )
    val_batch = make_row_batch(
        val_split, gene_key_to_idx=gene_key_to_idx,
        experiment_id_to_row=exp_to_row,
    )
    thresholds = compute_train_thresholds(
        train_split, cor12_quantile=0.25, abs_t_quantile=0.25
    )
    weights = compute_weighted_full_weights(
        train_batch.cor12, train_batch.abs_t, thresholds
    )

    chem = chemistry_dense.copy()
    chem[chem != 0] = 1.0

    # Train T5-A locked model on cross-condition split
    gene_dim = int(embedding_matrix.shape[1])
    chem_dim = int(chem.shape[1])
    model = AdapterResidualMLP(
        gene_dim=gene_dim, chem_dim=chem_dim,
        hidden_dim=512, n_blocks=1, dropout=0.1,
        adapter_hidden=1024, adapter_out=512,
        adapter_n_hidden_layers=1, adapter_layernorm=False,
    )
    train_ds = S5TorchDataset(
        train_batch, embedding_matrix=embedding_matrix,
        chemistry_matrix=chem, weights=weights,
    )
    val_ds = S5TorchDataset(
        val_batch, embedding_matrix=embedding_matrix,
        chemistry_matrix=chem,
        weights=np.ones(len(val_batch.y), dtype=np.float32),
    )
    cfg = TrainLoopConfig(
        lr=1e-3, weight_decay=1e-4, batch_size=8192,
        epochs=8, device="auto",
    )

    # Eligibility from eval policy (use multi_org_balanced bar even though split differs)
    log.info("  training D2 model...")
    _, summary = train_one_arm(
        arm_name="d2_cross_condition", seed=0, model=model,
        train_dataset=train_ds, val_dataset=val_ds,
        val_gene_keys=val_batch.gene_key,
        val_org_ids=val_batch.org_id,
        spearman_min_conditions=M_MIN,
        spearman_min_iqr=V_MIN,
        config=cfg,
    )
    log.info("  D2 RMSE=%.4f MAE=%.4f val_within_gene_spearman=%.4f",
             summary["best_val_rmse"], summary["best_val_mae"],
             summary.get("val_within_gene_spearman", float("nan")))

    # Compute within-gene Spearman from val_batch (matches prediction count)
    val_pred = summary["_best_val_pred"]
    val_eval = pd.DataFrame({
        "gene_key": val_batch.gene_key,
        "fit": val_batch.y.astype(np.float64),
        "pred_model": val_pred,
    })
    # Cross-condition split has fewer conditions per gene than cross-org, so
    # per-gene IQR is systematically smaller. Use a more permissive v_min
    # computed from this split's own cross-gene IQR p25.
    iqrs = val_eval.groupby("gene_key")["fit"].apply(
        lambda s: np.percentile(s, 75) - np.percentile(s, 25)
    )
    v_min_d2 = float(iqrs.quantile(0.25))
    log.info("  D2-specific v_min (cross-gene IQR p25) = %.4f (vs %.4f for cross-org)",
             v_min_d2, V_MIN)

    sp_d2, eligible_d2, computed_d2 = within_gene_spearman(
        val_eval, "pred_model", min_iqr=v_min_d2,
    )
    # Also compute at the cross-org V_MIN for direct comparison (likely few eligible)
    sp_xorg, eligible_xorg, computed_xorg = within_gene_spearman(
        val_eval, "pred_model", min_iqr=V_MIN,
    )
    log.info("  D2 within-gene Spearman @ v_min=%.4f: %.4f (eligible=%d, computed=%d)",
             v_min_d2, sp_d2, eligible_d2, computed_d2)
    log.info("  D2 within-gene Spearman @ v_min=%.4f: %.4f (eligible=%d, computed=%d)",
             V_MIN, sp_xorg, eligible_xorg, computed_xorg)
    sp = sp_d2
    eligible = eligible_d2
    computed = computed_d2

    # Compare to D1 cross-org Spearman (it'll be in d1_baselines.json)
    out = {
        "split_type": "within-org cross-condition (20% hold out per org)",
        "n_train_rows": int(len(train_split)),
        "n_val_rows": int(len(val_split)),
        "n_val_experiments": len(val_exp_keys),
        "best_val_rmse": float(summary["best_val_rmse"]),
        "best_val_mae": float(summary["best_val_mae"]),
        "v_min_d2_local": float(v_min_d2),
        "v_min_xorg": float(V_MIN),
        "within_gene_spearman_local_v_min": float(sp_d2),
        "within_gene_spearman_xorg_v_min": float(sp_xorg),
        "n_eligible_local_v_min": int(eligible_d2),
        "n_eligible_xorg_v_min": int(eligible_xorg),
        "n_computed_local_v_min": int(computed_d2),
        "n_computed_xorg_v_min": int(computed_xorg),
    }
    (OUTPUT_DIR / "d2_cross_condition.json").write_text(json.dumps(out, indent=2))
    return out


# ============================================================
# D3: Noise floor from biological replicates
# ============================================================

def diagnostic_3_noise_floor() -> dict:
    log.info("=" * 60)
    log.info("D3: Noise floor from biological replicates")
    log.info("=" * 60)

    exp = pd.read_parquet("data/derived/canonical/v0/experiments.parquet")
    df = pd.read_parquet("data/derived/canonical/v0/fitness_experiment_long.parquet")

    # Identify replicate groups: same (orgId, expDesc, media)
    exp["rep_key"] = (
        exp["orgId"].astype(str) + "|" + exp["expDesc"].astype(str) + "|" + exp["media"].astype(str)
    )
    rep_groups = exp.groupby("rep_key")["expName"].apply(list)
    rep_groups = rep_groups[rep_groups.apply(len) >= 2]
    log.info("  %d replicate groups (≥2 reps each)", len(rep_groups))

    # For each replicate group, compute within-gene Spearman between any 2 replicates
    # This estimates the noise floor for the per-gene cross-condition ranking
    # Caveat: each replicate group is a SINGLE condition, so we can't do
    # within-gene Spearman directly. Instead, compute per-gene fitness in each
    # replicate, then look at how consistent fit values are across replicates.

    # We'll use a different proxy: for each replicate group, compute
    # cor(fit_rep1, fit_rep2) over the shared genes. This is the
    # *experiment-level* reproducibility, which bounds the achievable Spearman.
    df["gene_key"] = df["orgId"].astype(str) + ":" + df["locusId"].astype(str)

    correlations = []
    n_used = 0
    for rep_key, expnames in rep_groups.items():
        if n_used >= 500:  # cap for speed
            break
        org = rep_key.split("|")[0]
        rep_dfs = []
        for en in expnames[:2]:  # first two replicates
            sub = df[(df["orgId"] == org) & (df["expName"] == en)][["gene_key", "fit"]]
            rep_dfs.append(sub)
        if len(rep_dfs) < 2 or rep_dfs[0].empty or rep_dfs[1].empty:
            continue
        merged = rep_dfs[0].merge(rep_dfs[1], on="gene_key", suffixes=("_a", "_b"))
        if len(merged) < 50:
            continue
        if merged["fit_a"].nunique() <= 1 or merged["fit_b"].nunique() <= 1:
            continue
        r, _ = spearmanr(merged["fit_a"], merged["fit_b"])
        if not np.isnan(r):
            correlations.append(r)
            n_used += 1

    correlations = np.array(correlations)
    log.info("  cross-replicate Spearman over %d pairs:", len(correlations))
    log.info("    mean=%.3f median=%.3f", correlations.mean(), np.median(correlations))
    log.info("    p25=%.3f p75=%.3f", np.percentile(correlations, 25), np.percentile(correlations, 75))

    # Plot distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(correlations, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(correlations.mean(), color="red", ls="--",
               label=f"mean={correlations.mean():.3f}")
    ax.axvline(np.median(correlations), color="green", ls="--",
               label=f"median={np.median(correlations):.3f}")
    ax.set_xlabel("Spearman correlation between replicate experiments")
    ax.set_ylabel("count")
    ax.set_title(f"D3: Noise floor — cross-replicate Spearman (n={len(correlations)} pairs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "d3_noise_floor.png", dpi=150)
    plt.close(fig)

    out = {
        "n_replicate_pairs_analyzed": int(len(correlations)),
        "cross_replicate_spearman_mean": float(correlations.mean()),
        "cross_replicate_spearman_median": float(np.median(correlations)),
        "cross_replicate_spearman_p25": float(np.percentile(correlations, 25)),
        "cross_replicate_spearman_p75": float(np.percentile(correlations, 75)),
        "interpretation": (
            "This is per-experiment cross-replicate Spearman over GENES "
            "(many genes in one condition), not within-gene cross-condition Spearman. "
            "It's a proxy upper bound: if a single experiment's gene-ranking can only "
            "be reproduced at Spearman=X, then any model that predicts the 'true' "
            "ranking is bounded by approximately X."
        ),
    }
    (OUTPUT_DIR / "d3_noise_floor.json").write_text(json.dumps(out, indent=2))
    return out


# ============================================================
# D4: IQR distribution analysis
# ============================================================

def diagnostic_4_iqr_distribution() -> dict:
    log.info("=" * 60)
    log.info("D4: IQR distribution analysis")
    log.info("=" * 60)

    locked = yaml.safe_load(Path("data_contract/splits/locked_protocol.yaml").read_text())
    val_orgs = sorted(locked["val_org_ids"])
    train_orgs = [
        o for o in pd.read_parquet(
            "data/derived/canonical/v0/fitness_experiment_long.parquet",
            columns=["orgId"]
        )["orgId"].unique()
        if o not in set(locked["val_org_ids"]) and o not in set(locked["test_org_ids"])
    ]

    df = pd.read_parquet("data/derived/canonical/v0/fitness_experiment_long.parquet")
    df["gene_key"] = df["orgId"].astype(str) + ":" + df["locusId"].astype(str)

    # Per-gene IQR for val orgs (per-org breakdown)
    fig, axes = plt.subplots(1, len(val_orgs), figsize=(5 * len(val_orgs), 4))
    if len(val_orgs) == 1:
        axes = [axes]
    summary = []
    for ax, org in zip(axes, val_orgs):
        sub = df[df["orgId"] == org]
        iqrs = sub.groupby("gene_key")["fit"].apply(
            lambda s: np.percentile(s, 75) - np.percentile(s, 25)
        )
        ax.hist(iqrs, bins=60, edgecolor="black", alpha=0.7)
        ax.axvline(V_MIN, color="red", ls="--", label=f"V_MIN={V_MIN:.3f}")
        ax.set_title(f"{org}\n n={len(iqrs):,}, eligible={(iqrs >= V_MIN).sum()}")
        ax.set_xlabel("Per-gene IQR")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        summary.append({
            "org": str(org),
            "n_genes": int(len(iqrs)),
            "iqr_mean": float(iqrs.mean()),
            "iqr_median": float(iqrs.median()),
            "iqr_p25": float(iqrs.quantile(0.25)),
            "iqr_p75": float(iqrs.quantile(0.75)),
            "n_eligible_v_min": int((iqrs >= V_MIN).sum()),
            "pct_eligible_v_min": float((iqrs >= V_MIN).mean() * 100),
        })
        log.info("  [%s] genes=%d median_iqr=%.3f eligible=%d (%.1f%%)",
                 org, len(iqrs), iqrs.median(),
                 (iqrs >= V_MIN).sum(), (iqrs >= V_MIN).mean() * 100)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "d4_iqr_per_val_org.png", dpi=150)
    plt.close(fig)

    out = {
        "per_org_iqr_summary": summary,
        "v_min_threshold": V_MIN,
    }
    (OUTPUT_DIR / "d4_iqr_distribution.json").write_text(json.dumps(out, indent=2))
    return out


# ============================================================
# Main
# ============================================================

def main():
    log.info("Starting T7-prep diagnostics")
    log.info("Outputs: %s", OUTPUT_DIR)

    # Skip phases that already have saved outputs (idempotent on rerun)
    def cached_or_run(json_name, fn):
        p = OUTPUT_DIR / json_name
        if p.exists():
            log.info("  [SKIP] %s already exists; loading cached", p.name)
            return json.loads(p.read_text())
        return fn()

    d4 = cached_or_run("d4_iqr_distribution.json", diagnostic_4_iqr_distribution)
    d3 = cached_or_run("d3_noise_floor.json", diagnostic_3_noise_floor)
    d1 = cached_or_run("d1_baselines.json", diagnostic_1_baselines)
    d2 = cached_or_run("d2_cross_condition.json", diagnostic_2_cross_condition)

    # Write combined report
    report = {
        "d1_baselines": d1,
        "d2_cross_condition": d2,
        "d3_noise_floor": d3,
        "d4_iqr_distribution": d4,
    }
    (OUTPUT_DIR / "combined_report.json").write_text(json.dumps(report, indent=2))
    log.info("All diagnostics complete. Report: %s", OUTPUT_DIR / "combined_report.json")


if __name__ == "__main__":
    main()
