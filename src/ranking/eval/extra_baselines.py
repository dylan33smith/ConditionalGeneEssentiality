"""Baselines the project never ran, added 2026-08-25 for the paper (item E).

Every model this project has beaten was a GLOBAL PARAMETRIC neural one (MLP,
bilinear MF, retrieval-concat, learned gating). Two families were never tested,
and a reviewer will ask about both:

  * **Gradient-boosted trees.** The tabular-learning literature (Grinsztajn 2022,
    McElfresh 2023) reports trees beating deep nets on tabular features far more
    often than not. Our inputs ARE tabular (a frozen embedding concatenated with a
    sparse chemistry vector). Not running a tree baseline leaves the obvious
    alternative untested.
  * **Learned non-parametric / local models.** chem-kNN is an UNLEARNED local
    method. A LEARNED local method (ResMem, EASE, TabR) is the family actually
    designed for this regime.

Note the interpretive trap, and state it in the paper: if a learned-LOCAL method
wins, that CONFIRMS the memorization finding rather than refuting it. These arms
exist to make the negative airtight, not to rescue a modelling win.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_MAX_GBDT_ROWS = 400_000


def gbdt_predict(
    train_df: pd.DataFrame, val_df: pd.DataFrame, *,
    gene_emb: np.ndarray, gene_to_row: dict,
    chem: np.ndarray, exp_to_row: dict,
    gene_col: str = "gene_key", exp_col: str = "experiment_id",
    fit_col: str = "fit", emb_components: int | None = 64,
    max_rows: int = _MAX_GBDT_ROWS, seed: int = 0,
    max_iter: int = 300,
) -> np.ndarray:
    """Gradient-boosted trees on (gene embedding + condition chemistry).

    The tabular rival the deep model was never compared against.

    Two documented concessions, both needed to keep this tractable at genome scale,
    and both stated in the paper rather than hidden:
      * the gene embedding is reduced with PCA to `emb_components` dims (fit on TRAIN
        rows only). Trees split on individual features and scale poorly to 1152
        near-collinear dense dims.
      * training rows are subsampled to `max_rows` when larger.
    Both concessions can only HURT the tree arm, so a tree win despite them would be
    a strong result and a tree loss remains suggestive rather than conclusive.
    """
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor

    def _rows(df):
        g = df[gene_col].map(gene_to_row)
        e = df[exp_col].map(exp_to_row)
        ok = g.notna() & e.notna()
        return ok.to_numpy(), g[ok].astype(int).to_numpy(), e[ok].astype(int).to_numpy()

    tr_ok, tr_g, tr_e = _rows(train_df)
    va_ok, va_g, va_e = _rows(val_df)

    rng = np.random.default_rng(seed)
    if len(tr_g) > max_rows:
        keep = rng.choice(len(tr_g), size=max_rows, replace=False)
        tr_g, tr_e = tr_g[keep], tr_e[keep]
        y = train_df.loc[tr_ok, fit_col].to_numpy()[keep]
        log.info("gbdt: subsampled train %d -> %d rows", int(tr_ok.sum()), max_rows)
    else:
        y = train_df.loc[tr_ok, fit_col].to_numpy()

    emb = gene_emb
    if emb_components and emb.shape[1] > emb_components:
        # TRAIN-ONLY fit -- val genes must not influence the projection
        pca = PCA(n_components=emb_components, random_state=seed)
        pca.fit(emb[np.unique(tr_g)])
        emb = pca.transform(emb)
        log.info("gbdt: PCA %d -> %d dims (train-only fit, %.1f%% var)",
                 gene_emb.shape[1], emb_components,
                 100 * float(pca.explained_variance_ratio_.sum()))

    X_tr = np.hstack([emb[tr_g], chem[tr_e]])
    X_va = np.hstack([emb[va_g], chem[va_e]])

    model = HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=0.1, max_depth=None,
        early_stopping=True, validation_fraction=0.1, random_state=seed)
    model.fit(X_tr, y)

    out = np.full(len(val_df), np.nan)
    out[va_ok] = model.predict(X_va)
    return out


def resmem_predict(
    model_pred_train: np.ndarray, model_pred_val: np.ndarray,
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    cond_features: dict, *, k: int = 5,
    gene_col: str = "gene_key", condition_col: str = "condition_key",
    fit_col: str = "fit",
) -> np.ndarray:
    """ResMem: the lookup memorizes the MODEL'S RESIDUAL, not a rival prediction.

    prediction = model(g, c) + kNN over gene g's own TRAIN residuals, weighted by
    chemical similarity between c and g's train conditions.

    Why this and not the naive model+kNN ensemble the project already rejected: a
    naive ensemble makes the two methods compete for the same signal, and the earlier
    learned fusions all settled on "trust the lookup". ResMem instead lets the global
    model take whatever it can explain and gives the local component only what is
    LEFT OVER. It also degrades gracefully on cold genes -- a gene with no train rows
    has no residuals, so the correction is zero and the prediction falls back exactly
    to the model. That is the property the earlier hybrids lacked.
    """
    from src.ranking.eval.harness import _cosine_dist_matrix

    tr = train_df.copy()
    tr["_resid"] = tr[fit_col].to_numpy() - np.asarray(model_pred_train)

    resid_by_gene: dict = {}
    for gene, sub in tr.groupby(gene_col, sort=False):
        conds = [c for c in sub[condition_col] if c in cond_features]
        if not conds:
            continue
        sub2 = sub[sub[condition_col].isin(conds)]
        resid_by_gene[gene] = (
            list(sub2[condition_col]),
            sub2["_resid"].to_numpy(),
        )

    out = np.asarray(model_pred_val, dtype=float).copy()
    for gene, sub in val_df.groupby(gene_col, sort=False):
        entry = resid_by_gene.get(gene)
        if entry is None:
            continue                     # cold gene -> pure model, by construction
        tconds, tres = entry
        tf = np.vstack([cond_features[c] for c in tconds])
        idx, vconds = [], []
        for i, c in zip(sub.index, sub[condition_col]):
            if c in cond_features:
                idx.append(i); vconds.append(c)
        if not idx:
            continue
        vf = np.vstack([cond_features[c] for c in vconds])
        d = _cosine_dist_matrix(vf, tf)
        kk = min(k, d.shape[1])
        nn = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        w = 1.0 / (1e-6 + np.take_along_axis(d, nn, axis=1))
        corr = (w * tres[nn]).sum(axis=1) / w.sum(axis=1)
        pos = val_df.index.get_indexer(idx)
        out[pos] = out[pos] + corr
    return out
