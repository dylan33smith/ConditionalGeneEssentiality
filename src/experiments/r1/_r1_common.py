"""R1 data-prep + training + evaluation under the ranking regime.

Pipeline (per organism set):
  1. load fitness + frozen ProteomeLM-L8 embeddings
  2. materialize R-LOCK-2 condition-holdout split
  3. compute R-LOCK-1 eligibility: per-gene w_g (train), val hard-filter mask
  4. build per-arm chemistry features (multihot / fingerprints)
  5. train AdapterResidualMLP (T5-A locked) with weighted pointwise MSE,
     early-stop on within-gene Spearman
  6. evaluate with the R-LOCK-4 harness (Spearman/Kendall/NDCG + chem baselines)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.datasets.build_s5_dataset import (
    _load_concatenated_embeddings, build_or_load_experiment_multihot)
from src.data.preprocessing.build_experiment_chemistry import experiment_uid
from src.data.datasets.build_ranking_split import materialize_condition_holdout
from src.data.datasets.ranking_eligibility import (
    compute_train_weights, val_eligible_genes, load_policy)
from src.data.datasets.condition_chemistry import load_condition_chemistry_features
from src.experiments.r0.analyses import load_fitness, _condition_key
from src.experiments.tier5._t5_common import AdapterResidualMLP
from src.evaluation.ranking_eval import (
    per_gene_correlations, hierarchical_bootstrap_ci, within_gene_retrieval,
    per_organism_breakdown, chemistry_nearest_condition_profile, chemistry_knn_predict)

log = logging.getLogger(__name__)

EMB_DIR = Path("data/processed/ProtLM_embeddings_layer8")
S4_DIR = Path("data_contract/preprocessing/de21504134c84a6c")
_FIT_COLS = ["orgId", "setName", "seqindex", "media", "expName", "expDesc",
             "temperature", "expGroup", "gene_key", "fit"]


@dataclass
class R1Data:
    train: pd.DataFrame            # rows: gene_key, exp_id, condition_key, fit, w_g, gene_row
    val: pd.DataFrame             # pooled per (gene, condition); + eligible flag
    val_raw: pd.DataFrame         # pre-pool val (for noise floor)
    emb: np.ndarray               # [n_genes, 1152]
    gene_to_row: dict
    multihot: "object"            # csr [n_exp, 425]
    exp_to_row: dict
    cond_features: dict           # condition_key -> multihot vec (for baselines)
    eligible_val_genes: set
    fp_bundle: dict               # experiment fingerprint bundle (morgan/rdkit/maccs)


def prepare_r1_data(orgs: list[str] | None, *, seed: int = 0) -> R1Data:
    raw = pd.read_parquet("data/derived/canonical/v0/fitness_experiment_long.parquet",
                          columns=_FIT_COLS)
    if orgs is not None:
        raw = raw[raw["orgId"].isin(orgs)].copy()
    raw = raw.dropna(subset=["orgId", "gene_key", "expDesc", "media", "fit",
                             "setName", "seqindex"]).copy()

    # experiment_id on RAW media (S4-compatible)
    key_df = raw[["orgId", "setName", "seqindex", "media"]].drop_duplicates().copy()
    key_df["experiment_id"] = key_df.apply(experiment_uid, axis=1)
    raw = raw.merge(key_df, on=["orgId", "setName", "seqindex", "media"], how="left")

    # normalized condition_key (lowercase media/expDesc + temperature)
    from src.experiments.r0.analyses import _normalize_string_keys
    norm = _normalize_string_keys(raw.copy())
    raw["condition_key"] = _condition_key(norm).values

    # R-LOCK-2 split
    split = materialize_condition_holdout(raw, seed=seed)
    raw = raw.loc[split.partition.index]
    raw["partition"] = split.partition.values
    train = raw[raw["partition"] == "train"].copy()
    val_raw = raw[raw["partition"] == "val"].copy()

    # R-LOCK-1 eligibility
    policy = load_policy()
    _, gene_to_w = compute_train_weights(train, policy=policy)
    train["w_g"] = train["gene_key"].map(gene_to_w).fillna(0.0)
    elig_val = val_eligible_genes(val_raw, policy=policy)

    # val pooled per (gene, condition)
    val = (val_raw.groupby(["orgId", "gene_key", "condition_key"])
           .agg(fit=("fit", "mean"), experiment_id=("experiment_id", "first"))
           .reset_index())
    val["eligible"] = val["gene_key"].isin(elig_val)

    # embeddings
    emb, gene_to_row = _load_concatenated_embeddings(
        sorted(raw["orgId"].unique()), EMB_DIR)

    # multihot chemistry (keyed by experiment_id)
    exp_ids = sorted(raw["experiment_id"].unique())
    multihot, exp_to_row, _ = build_or_load_experiment_multihot(
        chemistry_parquet_path=S4_DIR / "experiment_chemistry.parquet",
        canonical_vocab_json_path=S4_DIR / "canonical_id_vocab.json",
        target_experiment_ids=exp_ids,
        cache_dir=Path("artifacts/cache/r1") / ("_".join(orgs) if orgs else "full"),
    )
    cond_features = load_condition_chemistry_features(orgs=orgs)
    from src.experiments.tier6._t6_common import load_experiment_fingerprints
    fp_bundle = load_experiment_fingerprints()

    return R1Data(train, val, val_raw, emb, gene_to_row, multihot, exp_to_row,
                  cond_features, elig_val, fp_bundle)


# ---------------------------------------------------------------------------
# Chemistry per arm
# ---------------------------------------------------------------------------

def _multihot_rows(exp_ids: np.ndarray, data: R1Data) -> np.ndarray:
    rows = np.array([data.exp_to_row.get(e, -1) for e in exp_ids])
    out = np.zeros((len(exp_ids), data.multihot.shape[1]), dtype=np.float32)
    valid = rows >= 0
    out[valid] = data.multihot[rows[valid]].toarray()
    out[out != 0] = 1.0
    return out


def _fp_rows(exp_ids: np.ndarray, data: R1Data, key: str) -> np.ndarray:
    """Per-row fingerprint matrix for a bundle key (morgan/rdkit/maccs_mean).

    Missing experiments (not in the fingerprint bundle) get a zero row.
    """
    mat = data.fp_bundle[key]
    e2r = data.fp_bundle["exp_to_row"]
    out = np.zeros((len(exp_ids), mat.shape[1]), dtype=np.float32)
    for i, e in enumerate(exp_ids):
        r = e2r.get(e)
        if r is not None:
            out[i] = mat[r]
    return out


def chem_matrix_for_rows(arm: str, exp_ids: np.ndarray, data: R1Data) -> np.ndarray:
    """Dense chemistry features for a list of experiment_ids, per arm.

    Arms (match R1-A_chemistry.yaml / T6-A): multihot_425, morgan_2048,
    rdkit_2048, maccs_167, morgan_plus_multihot, maccs_plus_multihot.
    """
    if arm == "multihot_425":
        return _multihot_rows(exp_ids, data)
    if arm == "morgan_2048":
        return _fp_rows(exp_ids, data, "morgan_mean")
    if arm == "rdkit_2048":
        return _fp_rows(exp_ids, data, "rdkit_mean")
    if arm == "maccs_167":
        return _fp_rows(exp_ids, data, "maccs_mean")
    if arm == "morgan_plus_multihot":
        return np.concatenate([_fp_rows(exp_ids, data, "morgan_mean"),
                               _multihot_rows(exp_ids, data)], axis=1)
    if arm == "maccs_plus_multihot":
        return np.concatenate([_fp_rows(exp_ids, data, "maccs_mean"),
                               _multihot_rows(exp_ids, data)], axis=1)
    raise ValueError(f"unknown chemistry arm: {arm!r}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_r1_arm(arm: str, data: R1Data, *, seed: int = 0, epochs: int = 8,
                 lr: float = 1e-3, batch_size: int = 8192) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    dev = _device()

    # Build train arrays (drop genes missing an embedding)
    tr = data.train[data.train["gene_key"].isin(data.gene_to_row)].copy()
    g_row = tr["gene_key"].map(data.gene_to_row).to_numpy()
    y = tr["fit"].to_numpy(np.float32)
    w = tr["w_g"].to_numpy(np.float32)

    # MEMORY-SAFE chemistry: build a small per-experiment matrix and gather per
    # batch (materializing per-row chem for ~11M rows × 2048 dims would be ~90GB).
    uexp = pd.unique(tr["experiment_id"])
    exp_chem = chem_matrix_for_rows(arm, uexp, data)          # [n_uexp, chem_dim]
    exp_to_i = {e: i for i, e in enumerate(uexp)}
    row_exp = tr["experiment_id"].map(exp_to_i).to_numpy()

    emb_t = torch.tensor(data.emb, dtype=torch.float32, device=dev)
    exp_chem_t = torch.tensor(exp_chem, dtype=torch.float32, device=dev)
    g_row_t = torch.tensor(g_row, dtype=torch.long, device=dev)
    row_exp_t = torch.tensor(row_exp, dtype=torch.long, device=dev)
    y_t = torch.tensor(y, device=dev)
    w_t = torch.tensor(w, device=dev)

    model = AdapterResidualMLP(
        gene_dim=data.emb.shape[1], chem_dim=exp_chem.shape[1], hidden_dim=512,
        n_blocks=1, dropout=0.1, adapter_hidden=1024, adapter_out=512,
        adapter_n_hidden_layers=1, adapter_layernorm=False).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(y)
    best_spear, best_state = -2.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            chem_b = exp_chem_t[row_exp_t[idx]]
            pred = model(emb_t[g_row_t[idx]], chem_b).squeeze(-1)
            wse = (w_t[idx] * (pred - y_t[idx]) ** 2).sum() / w_t[idx].sum().clamp_min(1e-6)
            opt.zero_grad(); wse.backward(); opt.step()
        sp = _val_spearman(model, data, arm, dev)
        log.info("    [%s seed=%d] epoch %d  val within-gene Spearman=%.4f",
                 arm, seed, ep, sp)
        if sp > best_spear:
            best_spear = sp
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return _full_eval(model, data, arm, dev, seed=seed, best_spear=best_spear)


def _predict_val(model, data: R1Data, arm: str, dev) -> pd.DataFrame:
    v = data.val[data.val["gene_key"].isin(data.gene_to_row)].copy()
    g_row = torch.tensor(v["gene_key"].map(data.gene_to_row).to_numpy(),
                         dtype=torch.long, device=dev)
    # gather chem per batch from a small per-experiment matrix (memory-safe)
    uexp = pd.unique(v["experiment_id"])
    exp_chem = chem_matrix_for_rows(arm, uexp, data)
    exp_to_i = {e: i for i, e in enumerate(uexp)}
    row_exp = torch.tensor(v["experiment_id"].map(exp_to_i).to_numpy(),
                           dtype=torch.long, device=dev)
    exp_chem_t = torch.tensor(exp_chem, dtype=torch.float32, device=dev)
    emb_t = torch.tensor(data.emb, dtype=torch.float32, device=dev)
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(v), 16384):
            sl = slice(i, i + 16384)
            chem_b = exp_chem_t[row_exp[sl]]
            preds.append(model(emb_t[g_row[sl]], chem_b).squeeze(-1).cpu().numpy())
    v["pred"] = np.concatenate(preds)
    return v


def _val_spearman(model, data: R1Data, arm: str, dev) -> float:
    v = _predict_val(model, data, arm, dev)
    v = v[v["eligible"]]
    pg = per_gene_correlations(v, metric="spearman", pred_col="pred",
                               fit_col="fit", gene_col="gene_key")
    return float(pg["value"].mean()) if len(pg) else float("nan")


def _metrics_for_pred(df: pd.DataFrame, pred_col: str) -> dict:
    """Spearman/Kendall + NDCG@k + precision@k for one prediction column.

    All on the SAME rows passed in (caller restricts to the common gene set
    for denominator parity).
    """
    d = df.rename(columns={pred_col: "pred"})
    pg_sp = per_gene_correlations(d, metric="spearman", pred_col="pred")
    pg_kd = per_gene_correlations(d, metric="kendall", pred_col="pred")
    sp_ci = hierarchical_bootstrap_ci(pg_sp, n_bootstrap=300)
    ret = within_gene_retrieval(d, k_values=(1, 3, 5), pred_col="pred")
    out = {
        "spearman": sp_ci["mean"],
        "spearman_ci_low": sp_ci["ci_low"], "spearman_ci_high": sp_ci["ci_high"],
        "kendall": float(pg_kd["value"].mean()) if len(pg_kd) else float("nan"),
        "n_genes": int(sp_ci["n_genes"]),
    }
    for k in (1, 3, 5):
        out[f"ndcg_at_{k}"] = float(ret[f"ndcg_at_{k}"].mean()) if len(ret) else float("nan")
        out[f"precision_at_{k}"] = float(ret[f"precision_at_{k}"].mean()) if len(ret) else float("nan")
    return out


def _full_eval(model, data: R1Data, arm: str, dev, *, seed: int, best_spear: float) -> dict:
    v = _predict_val(model, data, arm, dev)
    elig = v[v["eligible"]].rename(columns={"pred": "model_pred"}).copy()

    # Baseline predictions on the eligible val rows
    tr = data.train
    elig["null_pred"] = chemistry_nearest_condition_profile(tr, elig, data.cond_features).values
    elig["knn_pred"] = chemistry_knn_predict(tr, elig, data.cond_features, k=5).values

    # COMMON gene set: rows where ALL THREE methods produce a prediction
    # (denominator parity — model, chem-null, chem-kNN scored on identical genes).
    common = elig.dropna(subset=["model_pred", "null_pred", "knn_pred"]).copy()

    model_m = _metrics_for_pred(common, "model_pred")
    null_m = _metrics_for_pred(common, "null_pred")
    knn_m = _metrics_for_pred(common, "knn_pred")

    # per-org breakdown on the model (side metric)
    pg_model = per_gene_correlations(common.rename(columns={"model_pred": "pred"}),
                                     metric="spearman", pred_col="pred")
    ret_model = within_gene_retrieval(common.rename(columns={"model_pred": "pred"}),
                                      k_values=(1, 3, 5), pred_col="pred")
    per_org = per_organism_breakdown(pg_model, retrieval_per_gene=ret_model)

    # Flat row for the results CSV (model headline + key baseline comparators)
    flat = {
        "arm": arm, "seed": seed,
        "n_common_genes": model_m["n_genes"],
        "model_spearman": model_m["spearman"],
        "model_ndcg_at_1": model_m["ndcg_at_1"], "model_ndcg_at_3": model_m["ndcg_at_3"],
        "model_ndcg_at_5": model_m["ndcg_at_5"],
        "model_precision_at_1": model_m["precision_at_1"],
        "model_precision_at_5": model_m["precision_at_5"],
        "knn_spearman": knn_m["spearman"], "knn_ndcg_at_5": knn_m["ndcg_at_5"],
        "knn_precision_at_1": knn_m["precision_at_1"],
        "null_spearman": null_m["spearman"], "null_ndcg_at_5": null_m["ndcg_at_5"],
        "beats_knn_spearman": bool(model_m["spearman"] > knn_m["spearman"]),
        "beats_knn_ndcg5": bool(model_m["ndcg_at_5"] > knn_m["ndcg_at_5"]),
    }
    return {"flat": flat, "comparison": {"model": model_m, "chem_knn": knn_m,
                                         "chem_null": null_m}, "per_org": per_org}
