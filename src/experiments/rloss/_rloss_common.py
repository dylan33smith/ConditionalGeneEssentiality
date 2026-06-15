"""R-LOSS gene-batched training. Reuses R1 data-prep, model, and eval harness;
only the OBJECTIVE changes (multihot encoder + T5-A architecture held constant).

Pointwise losses could use row batches, but pairwise/listwise need per-gene
groups, so we batch by GENE for all losses uniformly: each batch is B genes,
each padded to L conditions (sampled up to L_cap), with a mask.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from src.experiments.r1._r1_common import (
    R1Data, chem_matrix_for_rows, _full_eval, _predict_val)
from src.ranking.models import AdapterResidualMLP
from src.experiments.rloss._losses import LOSSES
from src.evaluation.ranking_eval import within_gene_retrieval

log = logging.getLogger(__name__)

ARM = "multihot_425"          # encoder held constant in R-LOSS (R1-DEC-001)
POINTWISE = {"pointwise_mse", "pointwise_huber"}   # trained ROW-batched (natural)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_arm(loss_name: str, data: R1Data, *, seed: int = 0, epochs: int = 15) -> dict:
    """Dispatch: pointwise losses train ROW-batched (their natural/best form, =
    the R1 control); pairwise/listwise train GENE-batched (required). Each
    objective gets its best training rather than a handicapped matched regime."""
    if loss_name in POINTWISE:
        return _train_pointwise_rowbatched(loss_name, data, seed=seed, epochs=8)
    return _train_rloss_genebatched(loss_name, data, seed=seed, epochs=epochs)


def _train_pointwise_rowbatched(loss_name, data, *, seed=0, epochs=8,
                                lr=1e-3, batch_size=8192, huber_delta=1.0):
    """Row-batched pointwise training (MSE or Huber), matching R1's regime."""
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
    n = len(y); best, best_state = -2.0, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(emb_t[g_row_t[idx]], exp_chem_t[row_exp_t[idx]]).squeeze(-1)
            err = pred - y_t[idx]
            if loss_name == "pointwise_huber":
                a = err.abs()
                pl = torch.where(a <= huber_delta, 0.5 * a ** 2,
                                 huber_delta * (a - 0.5 * huber_delta))
            else:
                pl = err ** 2
            loss = (w_t[idx] * pl).sum() / w_t[idx].sum().clamp_min(1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
        m = _val_ndcg5(model, data, dev)
        log.info("    [%s seed=%d] epoch %d  val NDCG@5=%.4f", loss_name, seed, ep, m)
        if m > best:
            best = m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return _full_eval(model, data, ARM, dev, seed=seed, best_spear=best)


def _build_gene_groups(data: R1Data):
    """Return per-gene arrays needed for batched training.

    genes: list of (emb_row, w_g, exp_idx_array, fit_array) for each train gene
    with an embedding. exp_idx indexes into a per-experiment chemistry matrix.
    """
    tr = data.train[data.train["gene_key"].isin(data.gene_to_row)].copy()
    uexp = pd.unique(tr["experiment_id"])
    exp_chem = chem_matrix_for_rows(ARM, uexp, data)              # [n_uexp, 425]
    exp_to_i = {e: i for i, e in enumerate(uexp)}
    tr["_exp_i"] = tr["experiment_id"].map(exp_to_i).to_numpy()
    tr["_grow"] = tr["gene_key"].map(data.gene_to_row).to_numpy()

    genes = []
    for _gk, g in tr.groupby("gene_key", sort=False):
        genes.append((
            int(g["_grow"].iloc[0]),
            float(g["w_g"].iloc[0]),
            g["_exp_i"].to_numpy(np.int64),
            g["fit"].to_numpy(np.float32),
        ))
    return genes, exp_chem


def _make_batch(genes, idxs, L_cap, rng):
    """Pad a set of genes to [B, L] tensors (numpy)."""
    rows = [genes[i] for i in idxs]
    lengths = [min(len(r[2]), L_cap) for r in rows]
    L = max(lengths)
    B = len(rows)
    exp_idx = np.zeros((B, L), np.int64)
    fit = np.zeros((B, L), np.float32)
    mask = np.zeros((B, L), bool)
    grow = np.zeros(B, np.int64)
    w = np.zeros(B, np.float32)
    for b, (gr, wg, ei, ff) in enumerate(rows):
        n = len(ei)
        if n > L_cap:
            sel = rng.choice(n, size=L_cap, replace=False)
            ei, ff, n = ei[sel], ff[sel], L_cap
        exp_idx[b, :n] = ei
        fit[b, :n] = ff
        mask[b, :n] = True
        grow[b] = gr
        w[b] = wg
    return grow, exp_idx, fit, mask, w


def _train_rloss_genebatched(loss_name: str, data: R1Data, *, seed: int = 0,
                    epochs: int = 15, lr: float = 2e-3, batch_genes: int = 128,
                    L_cap: int = 64) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    dev = _device(); rng = np.random.default_rng(seed)
    loss_fn = LOSSES[loss_name]

    genes, exp_chem = _build_gene_groups(data)
    emb_t = torch.tensor(data.emb, dtype=torch.float32, device=dev)
    exp_chem_t = torch.tensor(exp_chem, dtype=torch.float32, device=dev)
    chem_dim = exp_chem.shape[1]

    model = AdapterResidualMLP(
        gene_dim=data.emb.shape[1], chem_dim=chem_dim, hidden_dim=512,
        n_blocks=1, dropout=0.1, adapter_hidden=1024, adapter_out=512,
        adapter_n_hidden_layers=1, adapter_layernorm=False).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_genes = len(genes)
    best_metric, best_state = -2.0, None
    for ep in range(epochs):
        model.train()
        order = rng.permutation(n_genes)
        for i in range(0, n_genes, batch_genes):
            idxs = order[i:i + batch_genes]
            grow, exp_idx, fit, mask, w = _make_batch(genes, idxs, L_cap, rng)
            B, L = exp_idx.shape
            grow_t = torch.tensor(grow, device=dev)
            exp_t = torch.tensor(exp_idx, device=dev)
            fit_t = torch.tensor(fit, device=dev)
            mask_t = torch.tensor(mask, device=dev)
            w_t = torch.tensor(w, device=dev)
            ge = emb_t[grow_t].unsqueeze(1).expand(B, L, data.emb.shape[1]).reshape(B * L, -1)
            ch = exp_chem_t[exp_t].reshape(B * L, chem_dim)
            scores = model(ge, ch).squeeze(-1).reshape(B, L)
            loss = loss_fn(scores, fit_t, w_t, mask_t)
            opt.zero_grad(); loss.backward(); opt.step()
        metric = _val_ndcg5(model, data, dev)
        log.info("    [%s seed=%d] epoch %d  val NDCG@5=%.4f", loss_name, seed, ep, metric)
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return _full_eval(model, data, ARM, dev, seed=seed, best_spear=best_metric)


def _val_ndcg5(model, data: R1Data, dev) -> float:
    """Early-stopping metric: mean within-gene NDCG@5 on eligible val (we value
    top-k retrieval; aligns the stopping criterion with the objective)."""
    v = _predict_val(model, data, ARM, dev)
    v = v[v["eligible"]]
    ret = within_gene_retrieval(v, k_values=(5,), pred_col="pred")
    return float(ret["ndcg_at_5"].mean()) if len(ret) else float("nan")
