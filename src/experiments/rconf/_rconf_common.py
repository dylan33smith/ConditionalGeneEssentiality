"""R-CONF internals: confidence-weighted training + confidence-stratified eval.

Confidence signal = `abs_t` (Wetmore et al. 2015 moderated t for a gene's fitness
in an experiment), already in the canonical fitness table. Higher |t| = the
fitness effect is large relative to that measurement's noise => a more reliable
label. |t| > 4 is the field-standard "this effect is real" threshold.

Two analyses:
  1. CONFIDENCE-STRATIFIED EVAL (primary). Stratify eligible val GENES by their
     median per-cell abs_t into quartiles. Per stratum, score model / chem-kNN /
     chem-NULL (within-gene Spearman + NDCG@5 + precision@5, denominator parity)
     AND the biological-replicate noise-floor CEILING restricted to that stratum.
     Reading: if the model still loses to kNN in the high-confidence stratum, the
     negative is not a noise artifact; if it catches kNN only there, the signal
     was noise-masked.
  2. T-WEIGHTED TRAINING (secondary). Retrain the locked base with per-row loss
     weight w_g * conf_factor(abs_t) and compare to the w_g-only baseline.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from src.experiments.r1._r1_common import (
    R1Data, chem_matrix_for_rows, _predict_val, _metrics_for_pred, _device)
from src.experiments.tier5._t5_common import AdapterResidualMLP
from src.evaluation.ranking_eval import (
    chemistry_knn_predict, chemistry_nearest_condition_profile,
    retrieval_noise_floor)

log = logging.getLogger(__name__)
ARM = "multihot_425"
T_CAP = 4.0          # |t| significance threshold => full confidence
CONF_FLOOR = 0.1     # never fully zero a row out on confidence alone


# ---------------------------------------------------------------------------
# Confidence weighting
# ---------------------------------------------------------------------------

def confidence_factor(abs_t, *, cap: float = T_CAP, floor: float = CONF_FLOOR):
    """Map a per-row |t| to a multiplicative confidence weight in [floor, 1].

    Saturating ramp: 0 at |t|=0 rising linearly to 1 at |t|=cap, clipped to
    [floor, 1]. NaN |t| (no t reported) -> neutral 0.5. This down-weights noisy,
    low-|t| measurements without discarding them.
    """
    a = np.asarray(abs_t, dtype=float)
    fac = np.clip(a / cap, floor, 1.0)
    fac = np.where(np.isnan(a), 0.5, fac)
    return fac


# ---------------------------------------------------------------------------
# Training (generic: takes an explicit per-row weight vector)
# ---------------------------------------------------------------------------

def train_weighted_model(data: R1Data, *, weight_col_extra: str | None = None,
                         seed: int = 0, epochs: int = 8, lr: float = 1e-3,
                         batch_size: int = 8192, huber_delta: float = 1.0):
    """Train the locked base (AdapterResidualMLP + weighted Huber, multihot).

    Row weight = w_g, optionally multiplied by a confidence factor derived from
    `abs_t` when weight_col_extra == "conf". Returns the trained model.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    dev = _device()
    tr = data.train[data.train["gene_key"].isin(data.gene_to_row)].copy()
    g_row = tr["gene_key"].map(data.gene_to_row).to_numpy()
    y = tr["fit"].to_numpy(np.float32)
    w = tr["w_g"].to_numpy(np.float32)
    if weight_col_extra == "conf":
        w = (w * confidence_factor(tr["abs_t"].to_numpy())).astype(np.float32)

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
            err = pred - y_t[idx]; a = err.abs()
            pl = torch.where(a <= huber_delta, 0.5 * a ** 2,
                             huber_delta * (a - 0.5 * huber_delta))
            l = (w_t[idx] * pl).sum() / w_t[idx].sum().clamp_min(1e-6)
            opt.zero_grad(); l.backward(); opt.step()
        log.info("    [%s seed=%d] epoch %d done",
                 weight_col_extra or "w_g", seed, ep)
    return model


# ---------------------------------------------------------------------------
# Build the common eval frame (predictions + baselines + per-cell confidence)
# ---------------------------------------------------------------------------

def build_eval_frame(model, data: R1Data, dev) -> pd.DataFrame:
    """Eligible val cells with model_pred + knn_pred + null_pred + abs_t, on the
    COMMON gene set where every predictor is defined (denominator parity)."""
    v = _predict_val(model, data, ARM, dev)
    elig = v[v["eligible"]].rename(columns={"pred": "model_pred"}).copy()
    elig["knn_pred"] = chemistry_knn_predict(
        data.train, elig, data.cond_features, k=5).values
    elig["null_pred"] = chemistry_nearest_condition_profile(
        data.train, elig, data.cond_features).values
    common = elig.dropna(subset=["model_pred", "knn_pred", "null_pred"]).copy()
    return common


# ---------------------------------------------------------------------------
# Gene-level confidence strata
# ---------------------------------------------------------------------------

def assign_gene_strata(eval_df: pd.DataFrame, *, n_strata: int = 4,
                       min_n: int = 5) -> pd.DataFrame:
    """Per-gene median abs_t -> quartile label Q1(low)..Q{n}(high).

    Only genes with >= min_n eligible val cells are eligible to be ranked, so we
    compute confidence over those genes and bin by quantile. Returns a frame
    gene_key, orgId, gene_conf, stratum.
    """
    counts = eval_df.groupby("gene_key").size()
    keep = counts[counts >= min_n].index
    sub = eval_df[eval_df["gene_key"].isin(keep)]
    g = (sub.groupby(["gene_key", "orgId"])["abs_t"].median()
         .reset_index().rename(columns={"abs_t": "gene_conf"}))
    # quantile bins on gene_conf; labels Q1..Qn from low to high confidence
    try:
        g["stratum"] = pd.qcut(g["gene_conf"], q=n_strata,
                               labels=[f"Q{i+1}" for i in range(n_strata)],
                               duplicates="drop")
    except ValueError:
        g["stratum"] = "Q1"
    return g


# ---------------------------------------------------------------------------
# Stratified metrics + per-stratum noise-floor ceiling
# ---------------------------------------------------------------------------

def stratified_metrics(eval_df: pd.DataFrame, val_raw: pd.DataFrame,
                       strata: pd.DataFrame) -> pd.DataFrame:
    """Per confidence stratum: model / chem-kNN / chem-NULL Spearman + NDCG@5 +
    precision@5 (denominator parity within the stratum) and the replicate
    noise-floor CEILING restricted to the stratum's genes."""
    gene_to_stratum = dict(zip(strata["gene_key"], strata["stratum"]))
    ev = eval_df[eval_df["gene_key"].isin(gene_to_stratum)].copy()
    ev["stratum"] = ev["gene_key"].map(gene_to_stratum)
    vr = val_raw.copy()
    vr["stratum"] = vr["gene_key"].map(gene_to_stratum)

    # ordered strata (Q1..Qn low->high confidence) that actually have genes
    if hasattr(strata["stratum"], "cat"):
        ordered = [s for s in strata["stratum"].cat.categories
                   if (strata["stratum"] == s).any()]
    else:
        ordered = sorted(ev["stratum"].dropna().unique())

    rows = []
    for stratum in ordered:
        cell = ev[ev["stratum"] == stratum]
        if cell.empty:
            continue
        rec = {"stratum": stratum,
               "n_genes": int(cell["gene_key"].nunique()),
               "n_cells": int(len(cell)),
               "gene_conf_median": float(strata.loc[strata["stratum"] == stratum,
                                                    "gene_conf"].median())}
        for method, col in (("model", "model_pred"), ("knn", "knn_pred"),
                            ("null", "null_pred")):
            m = _metrics_for_pred(cell, col)
            rec[f"{method}_spearman"] = m["spearman"]
            rec[f"{method}_ndcg5"] = m["ndcg_at_5"]
            rec[f"{method}_prec5"] = m["precision_at_5"]
        # replicate ceiling on this stratum's genes
        nf = retrieval_noise_floor(vr[vr["stratum"] == stratum], k_values=(5,))
        rec["ceiling_ndcg5"] = nf["ndcg_at_5"]
        rec["ceiling_n_genes"] = nf["n_genes_used"]
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cell-level high-|t| filter (selection-bias-aware companion analysis)
# ---------------------------------------------------------------------------

def cell_filter_analysis(eval_df: pd.DataFrame, *, thresholds=(0.0, 2.0, 4.0),
                         min_n: int = 5) -> pd.DataFrame:
    """Restrict each gene's ranked conditions to cells with abs_t >= threshold,
    recompute model/kNN/null metrics on the surviving (gene>=min_n) set, and log
    how many genes survive. NOTE: this introduces selection bias (well-measured
    cells are big-effect); reported as a companion, not the headline."""
    rows = []
    for thr in thresholds:
        sub = eval_df[eval_df["abs_t"].fillna(0.0) >= thr]
        counts = sub.groupby("gene_key").size()
        keep = counts[counts >= min_n].index
        sub = sub[sub["gene_key"].isin(keep)].copy()
        rec = {"abs_t_threshold": thr,
               "n_genes": int(sub["gene_key"].nunique()),
               "n_cells": int(len(sub))}
        if sub.empty:
            rows.append(rec); continue
        for method, col in (("model", "model_pred"), ("knn", "knn_pred"),
                            ("null", "null_pred")):
            m = _metrics_for_pred(sub, col)
            rec[f"{method}_spearman"] = m["spearman"]
            rec[f"{method}_ndcg5"] = m["ndcg_at_5"]
        rows.append(rec)
    return pd.DataFrame(rows)
