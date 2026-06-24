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
from src.data.datasets.build_ranking_split import (
    materialize_condition_holdout, materialize_cold_gene)
from src.data.datasets.ranking_eligibility import (
    compute_train_weights, val_eligible_genes, load_policy)
from src.data.datasets.condition_chemistry import load_condition_chemistry_features
from src.data.datasets.conditions import _condition_key
from src.ranking.models import AdapterResidualMLP
from src.ranking.eval import (
    per_gene_correlations, hierarchical_bootstrap_ci, within_gene_retrieval,
    per_organism_breakdown, chemistry_nearest_condition_profile, chemistry_knn_predict)

log = logging.getLogger(__name__)

EMB_DIR = Path("data/processed/ProtLM_embeddings_layer8")
S4_DIR = Path("data_contract/preprocessing/de21504134c84a6c")
_FIT_COLS = ["orgId", "setName", "seqindex", "media", "expName", "expDesc",
             "temperature", "expGroup", "gene_key", "fit", "abs_t"]


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
    mf_val_pred: "object" = None  # cached linear inductive-MF val predictions (Series)
    baseline_train: "object" = None  # train rows the BASELINES score on. None → use
                                  # .train. R-AUG sets this to the locked eval-org
                                  # train so the chem-kNN gate stays bit-exact while
                                  # .train (the MODEL's input) is augmented with more
                                  # organisms — see prepare_r_aug_data.
    parity_pred_cols: "object" = None  # prediction columns that must ALL be non-NaN to
                                  # enter the denominator-parity common set. None → the
                                  # default 4 (model+knn+null+mf), preserving the primary
                                  # split's behavior bit-exactly. The cold_gene diagnostic
                                  # sets ["model_pred","null_pred"] because chem-kNN and
                                  # inductive-MF are STRUCTURALLY inapplicable to a held-out
                                  # whole gene (no own-gene train history / no learned
                                  # per-gene latent) — see prepare_cold_gene_data.


def _load_canonical(orgs: list[str] | None) -> pd.DataFrame:
    """Load + filter canonical fitness and attach experiment_id + condition_key.

    The shared front-half of prepare_r1_data, factored out so R-AUG can process
    extra training organisms through the EXACT same path (same experiment_id /
    condition_key derivation) before the split.
    """
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
    from src.data.datasets.conditions import _normalize_string_keys
    norm = _normalize_string_keys(raw.copy())
    raw["condition_key"] = _condition_key(norm).values
    return raw


def prepare_r1_data(orgs: list[str] | None, *, seed: int = 0,
                    split_fn=materialize_condition_holdout,
                    compute_mf: bool = True,
                    parity_pred_cols: list[str] | None = None) -> R1Data:
    """Prepare the ranking dataset (split + eligibility + features + baselines).

    Defaults reproduce the PRIMARY within-org condition-holdout split bit-for-bit.
    `split_fn` swaps in a diagnostic split (e.g. materialize_cold_gene);
    `compute_mf=False` skips the linear inductive-MF baseline (inapplicable when a
    baseline can't be defined for the split — e.g. cold genes have no per-gene
    latent); `parity_pred_cols` overrides the denominator-parity method set.
    """
    raw = _load_canonical(orgs)

    # R-LOCK-2 split (or a diagnostic split via split_fn)
    split = split_fn(raw, seed=seed)
    raw = raw.loc[split.partition.index]
    raw["partition"] = split.partition.values
    train = raw[raw["partition"] == "train"].copy()
    val_raw = raw[raw["partition"] == "val"].copy()

    # R-LOCK-1 eligibility
    policy = load_policy()
    _, gene_to_w = compute_train_weights(train, policy=policy)
    train["w_g"] = train["gene_key"].map(gene_to_w).fillna(0.0)
    elig_val = val_eligible_genes(val_raw, policy=policy)

    # val pooled per (gene, condition). abs_t (Wetmore moderated-t) is carried as
    # a per-cell measurement-confidence signal for R-CONF confidence stratification
    # (mean over replicate measurements); n_rep = #replicate rows behind the cell.
    val = (val_raw.groupby(["orgId", "gene_key", "condition_key"])
           .agg(fit=("fit", "mean"), experiment_id=("experiment_id", "first"),
                abs_t=("abs_t", "mean"), n_rep=("fit", "size"))
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
    from src.ranking.data.fingerprints import load_experiment_fingerprints
    fp_bundle = load_experiment_fingerprints()

    # Linear inductive-MF baseline — depends only on (train, val, features), so
    # compute ONCE here and reuse across all arms/seeds (it's model-independent).
    # Skipped when inapplicable (compute_mf=False): a held-out whole gene has no
    # free per-gene latent U[g], so MF cannot predict for it.
    if compute_mf:
        from src.ranking.eval import inductive_mf_predict
        mf_val_pred = inductive_mf_predict(
            train, val, cond_features, rank=32, epochs=20, lr=0.05, weight_col="w_g")
    else:
        mf_val_pred = None

    data = R1Data(train, val, val_raw, emb, gene_to_row, multihot, exp_to_row,
                  cond_features, elig_val, fp_bundle, mf_val_pred)
    data.parity_pred_cols = parity_pred_cols
    return data


def prepare_cold_gene_data(orgs: list[str] | None, *, seed: int = 0) -> R1Data:
    """DIAGNOSTIC: hold out WHOLE genes per org (inductive-over-genes test).

    The one regime where a global embedding model could beat the per-gene
    retrieval baselines: chem-kNN predicts a gene's held-out conditions from that
    gene's OWN train conditions, and inductive-MF needs a learned per-gene latent —
    both are STRUCTURALLY blind to a gene with zero training rows (coverage → 0).
    chem-NULL (the population condition-profile: train-gene-mean fit at the nearest
    train condition, gene-identity-free) is the only non-model baseline that still
    applies, so it becomes the gate. The scientific question: does the frozen
    ProteomeLM embedding carry gene-specific conditional-response signal beyond the
    population average, on genes the model never saw?
    """
    return prepare_r1_data(orgs, seed=seed,
                           split_fn=materialize_cold_gene,
                           compute_mf=False,
                           parity_pred_cols=["model_pred", "null_pred"])


def prepare_r_aug_data(eval_orgs: list[str], extra_orgs: list[str], *,
                       seed: int = 0) -> R1Data:
    """R-AUG: train the MODEL on (eval_orgs ∪ extra_orgs) while keeping the
    evaluation locked to `eval_orgs`.

    The val set, eligibility, and EVERY baseline (chem-kNN gate, chem-NULL,
    linear-MF) are produced exactly as `prepare_r1_data(eval_orgs, seed)` would —
    so the gate and the eligible-val denominator are bit-identical to the locked
    23-org run. Only `.train` (the model's input) and the feature stores
    (`.emb`/`.gene_to_row`, `.multihot`/`.exp_to_row`) are augmented with the
    extra organisms, which contribute ENTIRELY to training (no val rows, no
    influence on the baselines). `.baseline_train` carries the locked eval-org
    train that the baselines score on.

    This makes R-AUG a clean controlled A/B: identical denominator + identical
    gate, the single manipulated variable being how many organisms the global
    model trains on. chem-kNN is per-organism-local, so its number is unchanged;
    only the global model can use the extra data.

    NOTE: extra orgs are processed in isolation (their own split-free load +
    per-org weights) precisely because `materialize_condition_holdout` draws from
    one shared RNG in sorted-org order — folding extra orgs into a single split
    would shift the eval orgs' holdout. Building the eval object first guarantees
    the locked split is untouched.
    """
    import hashlib

    data = prepare_r1_data(eval_orgs, seed=seed)
    data.baseline_train = data.train               # baselines score on locked train

    # extra orgs → all rows train; same experiment_id/condition_key derivation,
    # same R-LOCK-1 per-org weighting policy (computed on extra rows alone == the
    # per-org result, since the policy is per-organism).
    extra = _load_canonical(extra_orgs)
    policy = load_policy()
    _, gene_to_w = compute_train_weights(extra, policy=policy)
    extra["w_g"] = extra["gene_key"].map(gene_to_w).fillna(0.0)
    extra["partition"] = "train"
    aug_train = pd.concat(
        [data.baseline_train, extra[data.baseline_train.columns]], ignore_index=True)

    # embeddings + multihot extended to the union (model needs extra-org features;
    # eval-org val genes/experiments remain present and correctly keyed)
    union = sorted(set(eval_orgs) | set(extra_orgs))
    emb, gene_to_row = _load_concatenated_embeddings(union, EMB_DIR)
    exp_ids = sorted(set(aug_train["experiment_id"]) | set(data.val["experiment_id"]))
    cache_key = "aug_" + hashlib.md5("|".join(union).encode()).hexdigest()[:12]
    multihot, exp_to_row, _ = build_or_load_experiment_multihot(
        chemistry_parquet_path=S4_DIR / "experiment_chemistry.parquet",
        canonical_vocab_json_path=S4_DIR / "canonical_id_vocab.json",
        target_experiment_ids=exp_ids,
        cache_dir=Path("artifacts/cache/r1") / cache_key,
    )

    data.train = aug_train
    data.emb, data.gene_to_row = emb, gene_to_row
    data.multihot, data.exp_to_row = multihot, exp_to_row
    return data


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
    # A baseline with zero coverage on these rows (every pred NaN) is INAPPLICABLE
    # — report NaN, not a number. Otherwise within_gene_retrieval would "rank" the
    # all-NaN column by arbitrary row order and emit a meaningless non-NaN NDCG
    # (this is exactly what chem-kNN/MF look like on the cold_gene split).
    if not d["pred"].notna().any():
        nan_out = {"spearman": float("nan"), "spearman_ci_low": float("nan"),
                   "spearman_ci_high": float("nan"), "kendall": float("nan"),
                   "n_genes": 0}
        for k in (1, 3, 5):
            nan_out[f"ndcg_at_{k}"] = float("nan")
            nan_out[f"precision_at_{k}"] = float("nan")
        return nan_out
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

    # Baseline predictions on the eligible val rows. Scored on baseline_train (the
    # locked eval-org train) when set, so the chem-kNN gate is bit-exact even when
    # data.train is augmented with extra organisms for the MODEL only (R-AUG).
    tr = data.baseline_train if data.baseline_train is not None else data.train
    elig["null_pred"] = chemistry_nearest_condition_profile(tr, elig, data.cond_features).values
    elig["knn_pred"] = chemistry_knn_predict(tr, elig, data.cond_features, k=5).values
    # linear inductive-MF: precomputed once in prepare_r1_data (model-independent)
    if data.mf_val_pred is not None:
        elig["mf_pred"] = data.mf_val_pred.reindex(elig.index).values
    else:
        elig["mf_pred"] = np.nan

    # COMMON gene set: rows where every PARITY method produces a prediction
    # (denominator parity — scored on identical genes). The default is all four;
    # the cold_gene diagnostic restricts parity to {model, chem-NULL} because
    # chem-kNN and inductive-MF are structurally inapplicable to held-out whole
    # genes (their coverage is reported below to make that explicit).
    parity_cols = data.parity_pred_cols or ["model_pred", "null_pred", "knn_pred", "mf_pred"]
    common = elig.dropna(subset=parity_cols).copy()

    model_m = _metrics_for_pred(common, "model_pred")
    null_m = _metrics_for_pred(common, "null_pred")
    knn_m = _metrics_for_pred(common, "knn_pred")
    mf_m = _metrics_for_pred(common, "mf_pred")

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
        # baseline coverage on the eligible val genes (fraction with a non-NaN
        # prediction). On the cold_gene split chem-kNN/MF coverage → ~0, which is
        # the whole point: only model + chem-NULL can score a held-out gene.
        "knn_coverage": float(elig["knn_pred"].notna().mean()) if len(elig) else float("nan"),
        "mf_coverage": float(elig["mf_pred"].notna().mean()) if len(elig) else float("nan"),
        "null_coverage": float(elig["null_pred"].notna().mean()) if len(elig) else float("nan"),
    }
    return {"flat": flat, "comparison": {"model": model_m, "chem_knn": knn_m,
                                         "linear_mf": mf_m, "chem_null": null_m},
            "per_org": per_org}
