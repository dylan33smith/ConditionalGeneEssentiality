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

    return R1Data(train, val, val_raw, emb, gene_to_row, multihot, exp_to_row,
                  cond_features, elig_val)


# ---------------------------------------------------------------------------
# Chemistry per arm
# ---------------------------------------------------------------------------

def chem_matrix_for_rows(arm: str, exp_ids: np.ndarray, data: R1Data) -> np.ndarray:
    """Dense chemistry features for a list of experiment_ids, per arm.

    R1-A arm 'multihot_425' is implemented here; fingerprint arms reuse the
    T6 fingerprint bundle (left as a follow-up — multihot is the control and
    suffices for the integration smoke + the headline comparison)."""
    if arm == "multihot_425":
        rows = np.array([data.exp_to_row.get(e, -1) for e in exp_ids])
        out = np.zeros((len(exp_ids), data.multihot.shape[1]), dtype=np.float32)
        valid = rows >= 0
        out[valid] = data.multihot[rows[valid]].toarray()
        return out
    raise NotImplementedError(f"arm {arm!r} not wired yet (multihot_425 available)")


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
    chem = chem_matrix_for_rows(arm, tr["experiment_id"].to_numpy(), data)
    y = tr["fit"].to_numpy(np.float32)
    w = tr["w_g"].to_numpy(np.float32)

    emb_t = torch.tensor(data.emb, dtype=torch.float32, device=dev)
    g_row_t = torch.tensor(g_row, dtype=torch.long, device=dev)
    chem_t = torch.tensor(chem, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y, device=dev)
    w_t = torch.tensor(w, device=dev)

    model = AdapterResidualMLP(
        gene_dim=data.emb.shape[1], chem_dim=chem.shape[1], hidden_dim=512,
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
            pred = model(emb_t[g_row_t[idx]], chem_t[idx]).squeeze(-1)
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
    chem = torch.tensor(chem_matrix_for_rows(arm, v["experiment_id"].to_numpy(), data),
                        dtype=torch.float32, device=dev)
    emb_t = torch.tensor(data.emb, dtype=torch.float32, device=dev)
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(v), 16384):
            sl = slice(i, i + 16384)
            preds.append(model(emb_t[g_row[sl]], chem[sl]).squeeze(-1).cpu().numpy())
    v["pred"] = np.concatenate(preds)
    return v


def _val_spearman(model, data: R1Data, arm: str, dev) -> float:
    v = _predict_val(model, data, arm, dev)
    v = v[v["eligible"]]
    pg = per_gene_correlations(v, metric="spearman", pred_col="pred",
                               fit_col="fit", gene_col="gene_key")
    return float(pg["value"].mean()) if len(pg) else float("nan")


def _full_eval(model, data: R1Data, arm: str, dev, *, seed: int, best_spear: float) -> dict:
    v = _predict_val(model, data, arm, dev)
    elig = v[v["eligible"]].copy()
    pg_sp = per_gene_correlations(elig, metric="spearman", pred_col="pred")
    pg_kd = per_gene_correlations(elig, metric="kendall", pred_col="pred")
    sp_ci = hierarchical_bootstrap_ci(pg_sp, n_bootstrap=300)
    kd_ci = hierarchical_bootstrap_ci(pg_kd, n_bootstrap=300)
    ret = within_gene_retrieval(elig, k_values=(1, 3, 5), pred_col="pred")
    ndcg5 = float(ret["ndcg_at_5"].mean()) if "ndcg_at_5" in ret else float("nan")
    per_org = per_organism_breakdown(pg_sp, retrieval_per_gene=ret)

    # baselines on the SAME eligible val gene set (denominator parity)
    tr = data.train
    base_null = chemistry_nearest_condition_profile(tr, elig, data.cond_features)
    elig_n = elig.assign(pred=base_null.values).dropna(subset=["pred"])
    null_sp = per_gene_correlations(elig_n, metric="spearman", pred_col="pred")
    base_knn = chemistry_knn_predict(tr, elig, data.cond_features, k=5)
    elig_k = elig.assign(pred=base_knn.values).dropna(subset=["pred"])
    knn_sp = per_gene_correlations(elig_k, metric="spearman", pred_col="pred")

    return {
        "arm": arm, "seed": seed,
        "model_spearman": sp_ci["mean"], "model_spearman_ci": [sp_ci["ci_low"], sp_ci["ci_high"]],
        "model_kendall": kd_ci["mean"], "model_ndcg_at_5": ndcg5,
        "n_eligible_val_genes": int(sp_ci["n_genes"]), "n_orgs": int(sp_ci["n_orgs"]),
        "baseline_chem_null_spearman": float(null_sp["value"].mean()) if len(null_sp) else float("nan"),
        "baseline_chem_knn_spearman": float(knn_sp["value"].mean()) if len(knn_sp) else float("nan"),
        "per_org": per_org,
    }
