"""Ranking trainers built on the RankingBatch contract (R-LOCK-3).

Batching is dispatched by loss family through the tested samplers in
`src/data/datasets/ranking_batch.py` (single source of truth, no hand-rolled
batching):

  pointwise (mse/huber) -> PointwiseSampler  : one (gene, condition) per item.
  ranking (ranknet/lambdarank/listmle/approxndcg) -> ListwiseSampler : a gene's
      whole condition set, padded to [B, L] with a mask (what the ranking losses
      consume). PairwiseSampler is available for pairwise-native losses (not yet
      wired — the current pairwise losses derive pairs from the listwise [B, L]).

The model forward is `(gene_emb, cond_feat) -> scalar`. Each RankingBatch carries
`gene_idx` (→ embedding rows) and `cond_idx` (→ per-experiment chemistry rows);
the trainer gathers the features and runs the loss. Multihot encoder + T5-A
architecture held constant (R1-DEC-001).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from src.ranking.pipeline import (
    R1Data, chem_matrix_for_rows, _full_eval, _predict_val)
from src.ranking.models import AdapterResidualMLP
from src.ranking.losses import LOSSES
from src.ranking.eval import within_gene_retrieval
from src.data.datasets.ranking_batch import (
    PointwiseSampler, ListwiseSampler, collate_pointwise, collate_listwise)

log = logging.getLogger(__name__)

ARM = "multihot_425"          # encoder held constant in R-LOSS (R1-DEC-001)
POINTWISE = {"pointwise_mse", "pointwise_huber"}   # row-batched (their natural form)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_arrays(data: R1Data):
    """Per-train-row arrays for the RankingBatch collates + a per-experiment
    chemistry matrix. Returns (gene_row, cond_idx, fit, weight, exp_chem):
      gene_row  row -> embedding row (data.gene_to_row)
      cond_idx  row -> per-experiment chemistry row (index into exp_chem)
      fit       row -> raw fitness target
      weight    row -> R-LOCK-1 per-gene weight w_g
    """
    tr = data.train[data.train["gene_key"].isin(data.gene_to_row)].copy()
    gene_row = tr["gene_key"].map(data.gene_to_row).to_numpy(np.int64)
    uexp = pd.unique(tr["experiment_id"])
    exp_chem = chem_matrix_for_rows(ARM, uexp, data)              # [n_uexp, chem_dim]
    exp_to_i = {e: i for i, e in enumerate(uexp)}
    cond_idx = tr["experiment_id"].map(exp_to_i).to_numpy(np.int64)
    fit = tr["fit"].to_numpy(np.float32)
    weight = tr["w_g"].to_numpy(np.float32)
    return gene_row, cond_idx, fit, weight, exp_chem


def _build_model(data: R1Data, chem_dim: int, dev, lr: float):
    model = AdapterResidualMLP(
        gene_dim=data.emb.shape[1], chem_dim=chem_dim, hidden_dim=512,
        n_blocks=1, dropout=0.1, adapter_hidden=1024, adapter_out=512,
        adapter_n_hidden_layers=1, adapter_layernorm=False).to(dev)
    return model, torch.optim.Adam(model.parameters(), lr=lr)


def train_arm(loss_name: str, data: R1Data, *, seed: int = 0, epochs: int = 15) -> dict:
    """Dispatch: pointwise losses train ROW-batched (PointwiseSampler);
    pairwise/listwise train GENE/LIST-batched (ListwiseSampler). Each objective
    gets its natural batch structure."""
    if loss_name in POINTWISE:
        return _train_pointwise(loss_name, data, seed=seed, epochs=8)
    return _train_listwise(loss_name, data, seed=seed, epochs=epochs)


def _train_pointwise(loss_name, data, *, seed=0, epochs=8,
                     lr=1e-3, batch_size=8192, huber_delta=1.0):
    """Row-batched pointwise training (MSE or Huber) via PointwiseSampler."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = _device()
    gene_row, cond_idx, fit, weight, exp_chem = _train_arrays(data)

    emb_t = torch.tensor(data.emb, dtype=torch.float32, device=dev)
    exp_chem_t = torch.tensor(exp_chem, dtype=torch.float32, device=dev)

    model, opt = _build_model(data, exp_chem.shape[1], dev, lr)
    n = len(fit)
    sampler = PointwiseSampler(n_rows=n, seed=seed, shuffle=True)
    best, best_state = -2.0, None
    for ep in range(epochs):
        model.train()
        order = list(iter(sampler))                       # epoch-advancing shuffle
        for i in range(0, n, batch_size):
            b = collate_pointwise(order[i:i + batch_size], gene_idx=gene_row,
                                  cond_idx=cond_idx, fit=fit, weight=weight)
            pred = model(emb_t[b.gene_idx.to(dev)],
                         exp_chem_t[b.cond_idx.to(dev)]).squeeze(-1)
            y = b.fit.to(dev); w = b.weight.to(dev)
            err = pred - y
            if loss_name == "pointwise_huber":
                a = err.abs()
                pl = torch.where(a <= huber_delta, 0.5 * a ** 2,
                                 huber_delta * (a - 0.5 * huber_delta))
            else:
                pl = err ** 2
            loss = (w * pl).sum() / w.sum().clamp_min(1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
        m = _val_ndcg5(model, data, dev)
        log.info("    [%s seed=%d] epoch %d  val NDCG@5=%.4f", loss_name, seed, ep, m)
        if m > best:
            best = m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return _full_eval(model, data, ARM, dev, seed=seed, best_spear=best)


def _train_listwise(loss_name: str, data: R1Data, *, seed: int = 0,
                    epochs: int = 15, lr: float = 2e-3, batch_genes: int = 128,
                    max_list_len: int = 64) -> dict:
    """Gene/list-batched training via ListwiseSampler: each gene's condition set
    padded to [B, L] with a mask; the ranking loss consumes (scores, fit, w, mask)."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = _device()
    loss_fn = LOSSES[loss_name]
    gene_row, cond_idx, fit, weight, exp_chem = _train_arrays(data)
    chem_dim = exp_chem.shape[1]

    emb_t = torch.tensor(data.emb, dtype=torch.float32, device=dev)
    exp_chem_t = torch.tensor(exp_chem, dtype=torch.float32, device=dev)

    model, opt = _build_model(data, chem_dim, dev, lr)
    sampler = ListwiseSampler(gene_idx=gene_row, seed=seed, min_gene_size=2,
                              max_list_len=max_list_len)
    best_metric, best_state = -2.0, None
    for ep in range(epochs):
        model.train()
        lists = list(iter(sampler))                       # one row-array per gene
        for i in range(0, len(lists), batch_genes):
            b = collate_listwise(lists[i:i + batch_genes], gene_idx=gene_row,
                                 cond_idx=cond_idx, fit=fit, weight=weight)
            B, L = b.gene_idx.shape
            # padded slots carry idx -1 → clamp to a real row (row 0) and rely on
            # the mask to zero their loss contribution (never use negative indexing)
            gi = b.gene_idx.clamp_min(0).to(dev)
            ci = b.cond_idx.clamp_min(0).to(dev)
            ge = emb_t[gi].reshape(B * L, -1)
            ch = exp_chem_t[ci].reshape(B * L, chem_dim)
            scores = model(ge, ch).squeeze(-1).reshape(B, L)
            loss = loss_fn(scores, b.fit.to(dev), b.weight.to(dev), b.mask.to(dev))
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
