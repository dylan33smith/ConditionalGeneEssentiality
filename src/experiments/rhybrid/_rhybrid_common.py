"""R-HYBRID-A: ensemble α-curve between the standalone model and chem-kNN.

Per-gene z-score each predictor (so α is a scale-balanced mixing weight, and the
combination is appropriate for a within-gene RANKING metric), then sweep
  hybrid = α · z(kNN) + (1-α) · z(model)
and measure within-gene Spearman / NDCG@5 at each α. α=1 is pure kNN (the gate),
α=0 is the model alone.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from src.experiments.r1._r1_common import R1Data, chem_matrix_for_rows, _predict_val
from src.experiments.tier5._t5_common import AdapterResidualMLP
from src.evaluation.ranking_eval import (
    chemistry_knn_predict, per_gene_correlations, within_gene_retrieval,
    hierarchical_bootstrap_ci)

log = logging.getLogger(__name__)
ARM = "multihot_425"


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_standalone_model(data: R1Data, *, loss="pointwise_huber", seed=0,
                           epochs=8, lr=1e-3, batch_size=8192, huber_delta=1.0):
    """Row-batched pointwise model (Huber default — the R-LOSS winner)."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = _device()
    tr = data.train[data.train["gene_key"].isin(data.gene_to_row)].copy()
    g_row = tr["gene_key"].map(data.gene_to_row).to_numpy()
    y = tr["fit"].to_numpy(np.float32); w = tr["w_g"].to_numpy(np.float32)
    uexp = pd.unique(tr["experiment_id"])
    exp_chem = chem_matrix_for_rows(ARM, uexp, data)
    exp_to_i = {e: i for i, e in enumerate(uexp)}
    row_exp = tr["experiment_id"].map(exp_to_i).to_numpy()

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
            if loss == "pointwise_huber":
                a = err.abs()
                pl = torch.where(a <= huber_delta, 0.5 * a ** 2,
                                 huber_delta * (a - 0.5 * huber_delta))
            else:
                pl = err ** 2
            l = (w_t[idx] * pl).sum() / w_t[idx].sum().clamp_min(1e-6)
            opt.zero_grad(); l.backward(); opt.step()
        log.info("    [standalone %s] epoch %d done", loss, ep)
    return model


def _zscore_per_gene(df, col, gene_col="gene_key"):
    g = df.groupby(gene_col)[col]
    mean = g.transform("mean"); std = g.transform("std").replace(0, 1.0)
    return (df[col] - mean) / std


def alpha_curve(data: R1Data, *, loss="pointwise_huber", seed=0, epochs=8,
                alphas=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                n_bootstrap=300) -> dict:
    """Train standalone model, build per-gene z-scored hybrid with chem-kNN +
    linear-MF, sweep α, return NDCG@5/Spearman at each α (α=1 pure kNN)."""
    dev = _device()
    model = train_standalone_model(data, loss=loss, seed=seed, epochs=epochs)

    # Predictions on eligible val
    v = _predict_val(model, data, ARM, dev)
    elig = v[v["eligible"]].rename(columns={"pred": "model_pred"}).copy()
    elig["knn_pred"] = chemistry_knn_predict(
        data.train, elig, data.cond_features, k=5).values
    if data.mf_val_pred is not None:
        elig["mf_pred"] = data.mf_val_pred.reindex(elig.index).values
    else:
        elig["mf_pred"] = np.nan

    common = elig.dropna(subset=["model_pred", "knn_pred", "mf_pred"]).copy()
    # per-gene z-scores (need >=2 conditions; per_gene_correlations enforces min_n)
    common["z_model"] = _zscore_per_gene(common, "model_pred")
    common["z_knn"] = _zscore_per_gene(common, "knn_pred")
    common["z_mf"] = _zscore_per_gene(common, "mf_pred")

    def score(pred_col):
        pg = per_gene_correlations(common, metric="spearman", pred_col=pred_col)
        ret = within_gene_retrieval(common, k_values=(1, 3, 5), pred_col=pred_col)
        ci = hierarchical_bootstrap_ci(pg, n_bootstrap=n_bootstrap)
        return {"spearman": ci["mean"], "spearman_ci": [ci["ci_low"], ci["ci_high"]],
                "ndcg1": float(ret["ndcg_at_1"].mean()),
                "ndcg5": float(ret["ndcg_at_5"].mean()),
                "n_genes": ci["n_genes"]}

    rows = []
    for a in alphas:
        # NOTE: lower z-score = more essential (since model/kNN predict fit; low
        # fit = essential). The hybrid score is a fit-like quantity; eval ranks
        # ascending. Combine in z-space then it's still ascending-essential.
        common["_h_knn_model"] = a * common["z_knn"] + (1 - a) * common["z_model"]
        common["_h_knn_mf"] = a * common["z_knn"] + (1 - a) * common["z_mf"]
        sm = score("_h_knn_model"); sf = score("_h_knn_mf")
        rows.append({"alpha": a,
                     "knn+model_spearman": sm["spearman"], "knn+model_ndcg5": sm["ndcg5"],
                     "knn+model_ndcg1": sm["ndcg1"],
                     "knn+mf_spearman": sf["spearman"], "knn+mf_ndcg5": sf["ndcg5"]})
        log.info("    α=%.1f  knn+model NDCG@5=%.4f Spearman=%.4f | knn+mf NDCG@5=%.4f",
                 a, sm["ndcg5"], sm["spearman"], sf["ndcg5"])

    # HONEST held-out α-selection: split val GENES 50/50 (by hash), pick α* that
    # maximizes NDCG@5 on the tune half, report NDCG@5(α*) on the test half.
    # This removes the val-tuning leakage of reading the curve's peak directly.
    import hashlib
    def _half(gk):
        return int(hashlib.md5(str(gk).encode()).hexdigest(), 16) % 2
    common["_half"] = common["gene_key"].map(_half)
    tune, test = common[common._half == 0], common[common._half == 1]
    def ndcg5_on(sub, col):
        r = within_gene_retrieval(sub, k_values=(5,), pred_col=col)
        return float(r["ndcg_at_5"].mean()) if len(r) else float("nan")
    best_a, best_tune = 1.0, -1.0
    for a in alphas:
        tune = tune.copy()
        tune["_h"] = a * tune["z_knn"] + (1 - a) * tune["z_model"]
        s = ndcg5_on(tune, "_h")
        if s > best_tune:
            best_tune, best_a = s, a
    test = test.copy()
    test["_h_sel"] = best_a * test["z_knn"] + (1 - best_a) * test["z_model"]
    honest_hybrid = ndcg5_on(test, "_h_sel")
    honest_knn = ndcg5_on(test, "z_knn")
    log.info("    HONEST (α*=%.1f selected on tune): hybrid NDCG@5=%.4f vs kNN %.4f on test half",
             best_a, honest_hybrid, honest_knn)
    return {"curve": pd.DataFrame(rows),
            "n_genes": int(common["gene_key"].nunique()),
            "honest_alpha": best_a, "honest_hybrid_ndcg5": honest_hybrid,
            "honest_knn_ndcg5": honest_knn}
