"""R-HYBRID-B: three LEARNED hybrids of the global model + local chem-kNN.

R-HYBRID-A showed a static per-gene z-score ensemble beats chem-kNN by only
+0.008 NDCG@5 (honest held-out) — below the ~0.023/~0.026 promotion delta.
The complementary signal exists but a static ensemble is the weakest fusion.
This module builds three learned hybrids to try to extract more:

  (a) residual model — train AdapterResidualMLP on the kNN leave-one-out
      RESIDUAL r = fit - knn_loo; val pred = knn_val + model_residual_val.
  (b) retrieval-augmented model — concat(gene_emb, chem_c, retrieval_features)
      where retrieval_features are gene g's k nearest train-condition fits +
      similarities; train end-to-end to predict fit.
  (c) learned gating — α(g,c) = sigmoid(MLP(gate_features)); per-gene z-scored
      final = α·z(knn) + (1-α)·z(model). α's MLP trained on TRAIN (LOO kNN).

All three share the R-HYBRID-A evaluation protocol for comparability:
  - eval on the eligible val gene set, denominator parity (SAME genes as kNN),
  - within-gene Spearman (hierarchical org->gene bootstrap CI) + NDCG@1/3/5 +
    precision@5,
  - HONEST held-out selection for any tunable mixing (split val genes 50/50 by
    hash, select on tune half, report on test half).

Leakage discipline:
  - all TRAIN-side kNN / retrieval uses leave-one-out (exclude_self=True),
  - VAL-side kNN / retrieval draws neighbors from TRAIN conditions only,
  - chemistry is gathered per-experiment (never materialized per-row).
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.experiments.r1._r1_common import R1Data, chem_matrix_for_rows, _predict_val
from src.experiments.rhybrid._rhybrid_common import (
    train_standalone_model, _zscore_per_gene)
from src.ranking.models import AdapterResidualMLP
from src.ranking.eval import (
    chemistry_knn_predict, chemistry_retrieval_features,
    per_gene_correlations, within_gene_retrieval, hierarchical_bootstrap_ci)

log = logging.getLogger(__name__)
ARM = "multihot_425"


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _hash_half(gk) -> int:
    return int(hashlib.md5(str(gk).encode()).hexdigest(), 16) % 2


# ===========================================================================
# Shared evaluation: metrics for one prediction column on a common gene set
# ===========================================================================

def _metrics(df: pd.DataFrame, pred_col: str, *, n_bootstrap: int = 300) -> dict:
    d = df.rename(columns={pred_col: "pred"})
    pg = per_gene_correlations(d, metric="spearman", pred_col="pred")
    ci = hierarchical_bootstrap_ci(pg, n_bootstrap=n_bootstrap)
    ret = within_gene_retrieval(d, k_values=(1, 3, 5), pred_col="pred")
    out = {
        "spearman": ci["mean"],
        "spearman_ci_low": ci["ci_low"], "spearman_ci_high": ci["ci_high"],
        "n_genes": int(ci["n_genes"]),
    }
    for k in (1, 3, 5):
        out[f"ndcg_at_{k}"] = float(ret[f"ndcg_at_{k}"].mean()) if len(ret) else float("nan")
        out[f"precision_at_{k}"] = float(ret[f"precision_at_{k}"].mean()) if len(ret) else float("nan")
    return out


def _ndcg5_on(sub: pd.DataFrame, col: str) -> float:
    r = within_gene_retrieval(sub, k_values=(5,), pred_col=col)
    return float(r["ndcg_at_5"].mean()) if len(r) else float("nan")


def _eval_block(common: pd.DataFrame, model_col: str, *, n_bootstrap: int = 300) -> dict:
    """Side-by-side metrics for the hybrid `model_col` vs chem-kNN, on the SAME
    rows (denominator parity). `common` must already be restricted to rows where
    both predictions are present.
    """
    return {
        "hybrid": _metrics(common, model_col, n_bootstrap=n_bootstrap),
        "chem_knn": _metrics(common, "knn_pred", n_bootstrap=n_bootstrap),
        "n_common_genes": int(common["gene_key"].nunique()),
    }


# ===========================================================================
# (a) RESIDUAL MODEL
# ===========================================================================

def train_residual_model(data: R1Data, *, seed: int = 0, epochs: int = 8,
                         lr: float = 1e-3, batch_size: int = 8192,
                         huber_delta: float = 1.0, k: int = 5):
    """Train AdapterResidualMLP to predict r = fit - knn_loo on TRAIN rows.

    knn_loo = leave-one-out chem-kNN on train (each train (g,c) predicted from
    gene g's OTHER train conditions). Rows where knn_loo is NaN are dropped.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    dev = _device()

    tr = data.train[data.train["gene_key"].isin(data.gene_to_row)].copy()
    # leave-one-out kNN target on train rows (per (gene, condition))
    tr_cond = (tr.groupby(["gene_key", "condition_key"])
               .agg(fit=("fit", "mean"),
                    experiment_id=("experiment_id", "first"),
                    w_g=("w_g", "first"))
               .reset_index())
    knn_loo = chemistry_knn_predict(
        data.train, tr_cond, data.cond_features, k=k, exclude_self=True)
    tr_cond["knn_loo"] = knn_loo.values
    tr_cond = tr_cond.dropna(subset=["knn_loo"]).copy()
    tr_cond["residual"] = tr_cond["fit"] - tr_cond["knn_loo"]
    log.info("    [residual] %d train (g,c) rows with a LOO-kNN target", len(tr_cond))

    g_row = tr_cond["gene_key"].map(data.gene_to_row).to_numpy()
    y = tr_cond["residual"].to_numpy(np.float32)
    w = tr_cond["w_g"].to_numpy(np.float32)
    uexp = pd.unique(tr_cond["experiment_id"])
    exp_chem = chem_matrix_for_rows(ARM, uexp, data)
    exp_to_i = {e: i for i, e in enumerate(uexp)}
    row_exp = tr_cond["experiment_id"].map(exp_to_i).to_numpy()

    emb_t = torch.tensor(data.emb, dtype=torch.float32, device=dev)
    exp_chem_t = torch.tensor(exp_chem, dtype=torch.float32, device=dev)
    g_row_t = torch.tensor(g_row, dtype=torch.long, device=dev)
    row_exp_t = torch.tensor(row_exp, dtype=torch.long, device=dev)
    y_t = torch.tensor(y, device=dev); w_t = torch.tensor(w, device=dev)

    model = AdapterResidualMLP(
        gene_dim=data.emb.shape[1], chem_dim=exp_chem.shape[1], hidden_dim=512,
        n_blocks=1, dropout=0.1, adapter_hidden=1024, adapter_out=512,
        adapter_n_hidden_layers=1, adapter_layernorm=False).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(y)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(emb_t[g_row_t[idx]], exp_chem_t[row_exp_t[idx]]).squeeze(-1)
            err = pred - y_t[idx]
            a = err.abs()
            pl = torch.where(a <= huber_delta, 0.5 * a ** 2,
                             huber_delta * (a - 0.5 * huber_delta))
            l = (w_t[idx] * pl).sum() / w_t[idx].sum().clamp_min(1e-6)
            opt.zero_grad(); l.backward(); opt.step()
        log.info("    [residual seed=%d] epoch %d done", seed, ep)
    return model


def run_residual(data: R1Data, *, seed: int = 0, epochs: int = 8, k: int = 5,
                 n_bootstrap: int = 300) -> dict:
    dev = _device()
    model = train_residual_model(data, seed=seed, epochs=epochs, k=k)

    v = _predict_val(model, data, ARM, dev)          # model predicts the RESIDUAL
    elig = v[v["eligible"]].rename(columns={"pred": "model_residual"}).copy()
    elig["knn_pred"] = chemistry_knn_predict(
        data.train, elig, data.cond_features, k=k).values
    common = elig.dropna(subset=["model_residual", "knn_pred"]).copy()
    common["hybrid_pred"] = common["knn_pred"] + common["model_residual"]

    block = _eval_block(common, "hybrid_pred", n_bootstrap=n_bootstrap)
    block = add_honest_split(block, common, "hybrid_pred")
    block["model"] = "residual"; block["seed"] = seed
    log.info("    [residual] hybrid NDCG@5=%.4f Spearman=%.4f | kNN NDCG@5=%.4f Spearman=%.4f",
             block["hybrid"]["ndcg_at_5"], block["hybrid"]["spearman"],
             block["chem_knn"]["ndcg_at_5"], block["chem_knn"]["spearman"])
    return block


# ===========================================================================
# (b) RETRIEVAL-AUGMENTED MODEL
# ===========================================================================

class RetrievalAugmentedModel(nn.Module):
    """AdapterResidualMLP head fed concat(adapter(gene_emb), chem_c, retrieval).

    Reuses the T5-A adapter + T3 head, but the proj layer now also ingests the
    retrieval-feature vector (gene g's k nearest train-condition fits + sims).
    """

    def __init__(self, *, gene_dim: int, chem_dim: int, retr_dim: int,
                 hidden_dim: int = 512, dropout: float = 0.1,
                 adapter_hidden: int = 1024, adapter_out: int = 512):
        super().__init__()
        self.base = AdapterResidualMLP(
            gene_dim=gene_dim, chem_dim=chem_dim + retr_dim, hidden_dim=hidden_dim,
            n_blocks=1, dropout=dropout, adapter_hidden=adapter_hidden,
            adapter_out=adapter_out, adapter_n_hidden_layers=1,
            adapter_layernorm=False)
        self.retr_dim = retr_dim

    def forward(self, gene_emb, chem, retr):
        return self.base(gene_emb, torch.cat([chem.float(), retr.float()], dim=1))


def _retrieval_features_train(data: R1Data, k: int) -> pd.DataFrame:
    """Per (gene, condition) train rows with LOO retrieval features + target fit."""
    tr = data.train[data.train["gene_key"].isin(data.gene_to_row)].copy()
    tr_cond = (tr.groupby(["gene_key", "condition_key"])
               .agg(fit=("fit", "mean"),
                    experiment_id=("experiment_id", "first"),
                    w_g=("w_g", "first"))
               .reset_index())
    feats = chemistry_retrieval_features(
        data.train, tr_cond, data.cond_features, k=k, exclude_self=True)
    tr_cond["_retr"] = list(feats)
    return tr_cond


def train_retrieval_model(data: R1Data, *, seed: int = 0, epochs: int = 8,
                          lr: float = 1e-3, batch_size: int = 8192,
                          huber_delta: float = 1.0, k: int = 5):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = _device()

    tr_cond = _retrieval_features_train(data, k)
    retr = np.vstack(tr_cond["_retr"].to_numpy()).astype(np.float32)
    g_row = tr_cond["gene_key"].map(data.gene_to_row).to_numpy()
    y = tr_cond["fit"].to_numpy(np.float32)
    w = tr_cond["w_g"].to_numpy(np.float32)
    uexp = pd.unique(tr_cond["experiment_id"])
    exp_chem = chem_matrix_for_rows(ARM, uexp, data)
    exp_to_i = {e: i for i, e in enumerate(uexp)}
    row_exp = tr_cond["experiment_id"].map(exp_to_i).to_numpy()

    emb_t = torch.tensor(data.emb, dtype=torch.float32, device=dev)
    exp_chem_t = torch.tensor(exp_chem, dtype=torch.float32, device=dev)
    retr_t = torch.tensor(retr, dtype=torch.float32, device=dev)
    g_row_t = torch.tensor(g_row, dtype=torch.long, device=dev)
    row_exp_t = torch.tensor(row_exp, dtype=torch.long, device=dev)
    y_t = torch.tensor(y, device=dev); w_t = torch.tensor(w, device=dev)

    model = RetrievalAugmentedModel(
        gene_dim=data.emb.shape[1], chem_dim=exp_chem.shape[1],
        retr_dim=retr.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(y)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(emb_t[g_row_t[idx]], exp_chem_t[row_exp_t[idx]],
                         retr_t[idx]).squeeze(-1)
            err = pred - y_t[idx]
            a = err.abs()
            pl = torch.where(a <= huber_delta, 0.5 * a ** 2,
                             huber_delta * (a - 0.5 * huber_delta))
            l = (w_t[idx] * pl).sum() / w_t[idx].sum().clamp_min(1e-6)
            opt.zero_grad(); l.backward(); opt.step()
        log.info("    [retrieval seed=%d] epoch %d done", seed, ep)
    return model


def _predict_val_retrieval(model, data: R1Data, k: int, dev) -> pd.DataFrame:
    """Val prediction with retrieval features drawn from TRAIN conditions only."""
    v = data.val[data.val["gene_key"].isin(data.gene_to_row)].copy()
    # VAL retrieval: neighbors from TRAIN (no exclude_self — val conds are cold)
    retr = chemistry_retrieval_features(
        data.train, v, data.cond_features, k=k, exclude_self=False)
    g_row = torch.tensor(v["gene_key"].map(data.gene_to_row).to_numpy(),
                         dtype=torch.long, device=dev)
    uexp = pd.unique(v["experiment_id"])
    exp_chem = chem_matrix_for_rows(ARM, uexp, data)
    exp_to_i = {e: i for i, e in enumerate(uexp)}
    row_exp = torch.tensor(v["experiment_id"].map(exp_to_i).to_numpy(),
                           dtype=torch.long, device=dev)
    exp_chem_t = torch.tensor(exp_chem, dtype=torch.float32, device=dev)
    emb_t = torch.tensor(data.emb, dtype=torch.float32, device=dev)
    retr_t = torch.tensor(retr.astype(np.float32), device=dev)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(v), 16384):
            sl = slice(i, i + 16384)
            preds.append(model(emb_t[g_row[sl]], exp_chem_t[row_exp[sl]],
                               retr_t[sl]).squeeze(-1).cpu().numpy())
    v["pred"] = np.concatenate(preds)
    return v


def run_retrieval(data: R1Data, *, seed: int = 0, epochs: int = 8, k: int = 5,
                  n_bootstrap: int = 300) -> dict:
    dev = _device()
    model = train_retrieval_model(data, seed=seed, epochs=epochs, k=k)
    v = _predict_val_retrieval(model, data, k, dev)
    elig = v[v["eligible"]].rename(columns={"pred": "hybrid_pred"}).copy()
    elig["knn_pred"] = chemistry_knn_predict(
        data.train, elig, data.cond_features, k=k).values
    common = elig.dropna(subset=["hybrid_pred", "knn_pred"]).copy()

    block = _eval_block(common, "hybrid_pred", n_bootstrap=n_bootstrap)
    block = add_honest_split(block, common, "hybrid_pred")
    block["model"] = "retrieval"; block["seed"] = seed
    log.info("    [retrieval] hybrid NDCG@5=%.4f Spearman=%.4f | kNN NDCG@5=%.4f Spearman=%.4f",
             block["hybrid"]["ndcg_at_5"], block["hybrid"]["spearman"],
             block["chem_knn"]["ndcg_at_5"], block["chem_knn"]["spearman"])
    return block


# ===========================================================================
# (c) LEARNED GATING
# ===========================================================================

class GateMLP(nn.Module):
    """Small MLP mapping per-(g,c) gate features -> α ∈ (0,1) via sigmoid."""

    def __init__(self, in_dim: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def _gate_features(query_df: pd.DataFrame, data: R1Data, *, k: int = 5,
                   sim_threshold: float = 0.5, exclude_self: bool = False
                   ) -> np.ndarray:
    """Per (gene, condition) gate features describing the LOCAL neighborhood:
      [ dist_to_nearest_train_condition,            # 1 - max cosine sim
        neighborhood_density,                       # # train conds within sim_threshold
        n_valid_knn_neighbors / k ]                 # how many of g's k nearest g actually has
    Computed from chemistry over TRAIN conditions only.
    """
    train_conds = [c for c in data.train["condition_key"].unique()
                   if c in data.cond_features]
    out = np.zeros((len(query_df), 3), dtype=np.float32)
    if not train_conds:
        return out
    q_conds = [c for c in query_df["condition_key"].unique()
               if c in data.cond_features]
    from src.ranking.eval.harness import _cosine_dist_matrix
    tfeat = np.vstack([data.cond_features[c] for c in train_conds])
    train_cond_to_j = {c: j for j, c in enumerate(train_conds)}

    feats = chemistry_retrieval_features(
        data.train, query_df, data.cond_features, k=k, exclude_self=exclude_self)
    coverage = feats[:, 2 * k + 1]                    # n_valid / k

    cond_meta: dict[str, tuple[float, float]] = {}
    if q_conds:
        qfeat = np.vstack([data.cond_features[c] for c in q_conds])
        dist = _cosine_dist_matrix(qfeat, tfeat)
        sim = 1.0 - dist
        for i, qc in enumerate(q_conds):
            s = sim[i].copy()
            if exclude_self:
                j = train_cond_to_j.get(qc)
                if j is not None:
                    s[j] = -np.inf
            nearest_dist = 1.0 - float(np.max(s))
            density = float(np.sum(s >= sim_threshold))
            cond_meta[qc] = (nearest_dist, density)

    nearest = query_df["condition_key"].map(
        {c: m[0] for c, m in cond_meta.items()}).to_numpy(dtype=float)
    density = query_df["condition_key"].map(
        {c: m[1] for c, m in cond_meta.items()}).to_numpy(dtype=float)
    out[:, 0] = np.nan_to_num(nearest, nan=0.0)
    out[:, 1] = np.nan_to_num(density, nan=0.0)
    out[:, 2] = coverage
    return out


def train_gate(data: R1Data, model, *, seed: int = 0, k: int = 5,
               epochs: int = 200, lr: float = 1e-2, sim_threshold: float = 0.5):
    """Train α(g,c)'s MLP on TRAIN rows to make α·z(knn_loo)+(1-α)·z(model) match
    the true fit ranking, per gene. Uses a per-gene-z-scored MSE-to-true surrogate
    (lower z = more essential), which is monotone-aligned with within-gene rank.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    dev = _device()

    tr = data.train[data.train["gene_key"].isin(data.gene_to_row)].copy()
    tr_cond = (tr.groupby(["gene_key", "condition_key"])
               .agg(fit=("fit", "mean"),
                    experiment_id=("experiment_id", "first"),
                    w_g=("w_g", "first"))
               .reset_index())
    # LOO kNN target on train
    tr_cond["knn_pred"] = chemistry_knn_predict(
        data.train, tr_cond, data.cond_features, k=k, exclude_self=True).values
    # standalone model prediction on the same train (g,c) rows
    tr_cond["model_pred"] = _model_predict_rows(model, tr_cond, data, dev)
    tr_cond = tr_cond.dropna(subset=["knn_pred", "model_pred"]).copy()
    # need >= 2 conditions per gene to z-score / rank
    tr_cond = tr_cond.groupby("gene_key").filter(lambda g: len(g) >= 2).copy()

    gate_feats = _gate_features(tr_cond, data, k=k, sim_threshold=sim_threshold,
                                exclude_self=True)
    tr_cond["z_knn"] = _zscore_per_gene(tr_cond, "knn_pred")
    tr_cond["z_model"] = _zscore_per_gene(tr_cond, "model_pred")
    tr_cond["z_true"] = _zscore_per_gene(tr_cond, "fit")

    X = torch.tensor(gate_feats, dtype=torch.float32, device=dev)
    z_knn = torch.tensor(tr_cond["z_knn"].to_numpy(np.float32), device=dev)
    z_model = torch.tensor(tr_cond["z_model"].to_numpy(np.float32), device=dev)
    z_true = torch.tensor(tr_cond["z_true"].to_numpy(np.float32), device=dev)
    w = torch.tensor(tr_cond["w_g"].to_numpy(np.float32), device=dev)
    # standardize gate features (train-only stats)
    mu = X.mean(0, keepdim=True); sd = X.std(0, keepdim=True).clamp_min(1e-6)
    Xn = (X - mu) / sd

    gate = GateMLP(in_dim=X.shape[1]).to(dev)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    for ep in range(epochs):
        gate.train()
        alpha = gate(Xn)
        h = alpha * z_knn + (1 - alpha) * z_model
        loss = (w * (h - z_true) ** 2).sum() / w.sum().clamp_min(1e-6)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0:
            log.info("    [gate] epoch %d loss=%.4f mean_alpha=%.3f",
                     ep, float(loss.detach()), float(alpha.mean().detach()))
    return gate, (mu.cpu().numpy(), sd.cpu().numpy())


def _model_predict_rows(model, rows_df: pd.DataFrame, data: R1Data, dev) -> np.ndarray:
    """Standalone-model prediction for a (gene_key, experiment_id) row frame."""
    g_row = torch.tensor(rows_df["gene_key"].map(data.gene_to_row).to_numpy(),
                         dtype=torch.long, device=dev)
    uexp = pd.unique(rows_df["experiment_id"])
    exp_chem = chem_matrix_for_rows(ARM, uexp, data)
    exp_to_i = {e: i for i, e in enumerate(uexp)}
    row_exp = torch.tensor(rows_df["experiment_id"].map(exp_to_i).to_numpy(),
                           dtype=torch.long, device=dev)
    exp_chem_t = torch.tensor(exp_chem, dtype=torch.float32, device=dev)
    emb_t = torch.tensor(data.emb, dtype=torch.float32, device=dev)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows_df), 16384):
            sl = slice(i, i + 16384)
            out.append(model(emb_t[g_row[sl]], exp_chem_t[row_exp[sl]]
                             ).squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def run_gating(data: R1Data, *, seed: int = 0, epochs_model: int = 8, k: int = 5,
               sim_threshold: float = 0.5, n_bootstrap: int = 300) -> dict:
    dev = _device()
    model = train_standalone_model(data, seed=seed, epochs=epochs_model)
    gate, (mu, sd) = train_gate(data, model, seed=seed, k=k,
                                sim_threshold=sim_threshold)

    # Val predictions
    v = _predict_val(model, data, ARM, dev)
    elig = v[v["eligible"]].rename(columns={"pred": "model_pred"}).copy()
    elig["knn_pred"] = chemistry_knn_predict(
        data.train, elig, data.cond_features, k=k).values
    common = elig.dropna(subset=["model_pred", "knn_pred"]).copy()
    # per-gene z requires >= 2 conditions (per_gene_correlations enforces min anyway)
    common = common.groupby("gene_key").filter(lambda g: len(g) >= 2).copy()
    common["z_model"] = _zscore_per_gene(common, "model_pred")
    common["z_knn"] = _zscore_per_gene(common, "knn_pred")

    gate_feats = _gate_features(common, data, k=k, sim_threshold=sim_threshold,
                               exclude_self=False)
    gate_feats = (gate_feats - mu) / sd
    gate.eval()
    with torch.no_grad():
        alpha = gate(torch.tensor(gate_feats, dtype=torch.float32, device=dev)
                     ).cpu().numpy()
    common["alpha"] = alpha
    common["hybrid_pred"] = common["alpha"] * common["z_knn"] + \
        (1 - common["alpha"]) * common["z_model"]
    common["z_knn_only"] = common["z_knn"]

    block = _eval_block(common, "hybrid_pred", n_bootstrap=n_bootstrap)
    # gating's gate baseline is z(kNN) (same per-gene z-space as the hybrid)
    block["chem_knn"] = _metrics(common, "z_knn_only", n_bootstrap=n_bootstrap)
    block["model"] = "gating"; block["seed"] = seed
    block["mean_alpha"] = float(np.mean(alpha))

    # HONEST held-out: gating has no scalar hyperparam selected on val, but we
    # still report a 50/50 gene split (test half) for comparability with the
    # other methods' honest numbers.
    common["_half"] = common["gene_key"].map(_hash_half)
    test = common[common._half == 1]
    block["honest"] = {
        "hybrid_ndcg5": _ndcg5_on(test, "hybrid_pred"),
        "knn_ndcg5": _ndcg5_on(test, "z_knn_only"),
        "n_test_genes": int(test["gene_key"].nunique()),
    }
    log.info("    [gating] mean_alpha=%.3f hybrid NDCG@5=%.4f Spearman=%.4f | "
             "z(kNN) NDCG@5=%.4f Spearman=%.4f",
             block["mean_alpha"], block["hybrid"]["ndcg_at_5"],
             block["hybrid"]["spearman"], block["chem_knn"]["ndcg_at_5"],
             block["chem_knn"]["spearman"])
    return block


# ===========================================================================
# HONEST held-out wrapper for residual / retrieval (50/50 gene split)
# ===========================================================================

def add_honest_split(block: dict, common: pd.DataFrame, hybrid_col: str) -> dict:
    """Report hybrid vs kNN NDCG@5 on a hash-held-out test half of val genes.

    Residual & retrieval models have NO val-selected hyperparameter (the hybrid
    is parameter-free given the trained model), so the honest number is just the
    same hybrid evaluated on the test half — included for comparability with
    R-HYBRID-A's protocol.
    """
    common = common.copy()
    common["_half"] = common["gene_key"].map(_hash_half)
    test = common[common._half == 1]
    block["honest"] = {
        "hybrid_ndcg5": _ndcg5_on(test, hybrid_col),
        "knn_ndcg5": _ndcg5_on(test, "knn_pred"),
        "n_test_genes": int(test["gene_key"].nunique()),
    }
    return block
