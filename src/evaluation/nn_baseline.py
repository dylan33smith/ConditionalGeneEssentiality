"""Embedding nearest-neighbour baseline (5th of the required null suite).

For each val row (val_gene, val_media):
  1. If val_media is also in train: candidate train rows = those with same media.
     Find nearest train gene by ProteomeLM cosine; predict that gene's
     in-medium mean fit on train.
  2. Else: candidate train rows = all train rows. Find nearest train gene
     globally; predict that gene's global mean fit on train (fallback).

Per-medium grouping reduces cosine-similarity work to per-medium matmuls.

Memory: loads ALL train-organism embeddings into a dense fp32 matrix
(~250k genes × 1152 dims × 4 bytes ≈ 1.2 GB peak). The full-fallback
path uses batched matmuls (default batch=512) to bound memory.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch


log = logging.getLogger(__name__)


def _load_concatenated_embeddings(
    orgs: list[str], embedding_dir: Path
) -> tuple[np.ndarray, dict[str, int], list[str]]:
    """Concatenate embeddings into one L2-normalized fp32 matrix.

    Returns (matrix (n, 1152), key_to_index, idx_to_key list).
    """
    matrices: list[np.ndarray] = []
    idx_to_key: list[str] = []
    for org in orgs:
        pt = embedding_dir / f"{org}_proteomelm.pt"
        if not pt.exists():
            log.warning("missing embedding bundle for %s; skipping", org)
            continue
        bundle = torch.load(pt, map_location="cpu", weights_only=False)
        emb = bundle["embeddings"].to(torch.float32).numpy()
        matrices.append(emb)
        idx_to_key.extend(bundle["group_labels"])
    if not matrices:
        return np.zeros((0, 1152), dtype=np.float32), {}, []
    big = np.concatenate(matrices, axis=0)
    norms = np.maximum(np.linalg.norm(big, axis=1, keepdims=True), 1e-9)
    big = (big / norms).astype(np.float32)
    key_to_idx = {k: i for i, k in enumerate(idx_to_key)}
    return big, key_to_idx, idx_to_key


def embedding_nn_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_orgs: list[str],
    val_orgs: list[str],
    *,
    embedding_dir: Path = Path("data/processed/ProtLM_embeddings_layer8"),
    media_col: str = "media",
    gene_col: str = "gene_key",
    fit_col: str = "fit",
    batch_size: int = 512,
) -> dict:
    """Compute the embedding-NN baseline for one (train, val) split.

    Returns dict with rmse / mae / n_rows / predictions / fallback fields.
    """
    log.info("    NN: loading train embeddings (%d orgs)", len(train_orgs))
    train_emb, train_k2i, _train_i2k = _load_concatenated_embeddings(
        train_orgs, embedding_dir
    )
    log.info("    NN: loading val embeddings (%d orgs)", len(val_orgs))
    val_emb, val_k2i, _val_i2k = _load_concatenated_embeddings(val_orgs, embedding_dir)
    log.info("    NN: train=%d genes, val=%d genes",
             train_emb.shape[0], val_emb.shape[0])

    # Per-gene mean fit (global)
    log.info("    NN: per-gene mean fit on train (global + per-media)")
    global_gene_mean = train_df.groupby(gene_col)[fit_col].mean()
    # Sparse map keyed by train embedding index → fit value
    global_mean_fit_by_idx = np.full(train_emb.shape[0], np.nan, dtype=np.float64)
    for g, mean_fit in global_gene_mean.items():
        if g in train_k2i:
            global_mean_fit_by_idx[train_k2i[g]] = float(mean_fit)
    global_mean_fit_fallback = float(train_df[fit_col].mean())

    # Per-medium index of train rows
    media_mean_fit = (train_df.groupby([media_col, gene_col])[fit_col].mean()
                      .reset_index())
    train_media_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    # For each medium: (sub-embedding-matrix, mean-fit-vector) aligned by row
    for media, sub in media_mean_fit.groupby(media_col):
        # Filter to genes that have an embedding
        mask = sub[gene_col].isin(train_k2i)
        sub_keep = sub[mask]
        if len(sub_keep) == 0:
            continue
        idxs = np.fromiter((train_k2i[g] for g in sub_keep[gene_col]), dtype=np.int64)
        train_media_index[media] = (
            train_emb[idxs],
            sub_keep[fit_col].to_numpy(dtype=np.float64),
        )
    log.info("    NN: indexed %d media (with embeddings)", len(train_media_index))

    # ---- Score val rows ----
    n_val = len(val_df)
    predictions = np.full(n_val, np.nan, dtype=np.float64)
    val_genes = val_df[gene_col].to_numpy()
    val_media = val_df[media_col].to_numpy()
    val_fit = val_df[fit_col].to_numpy()

    fallback_count = 0
    no_emb_count = 0

    # Group val row indices by media
    val_idx_by_media: dict[str, list[int]] = {}
    for i, m in enumerate(val_media):
        val_idx_by_media.setdefault(m, []).append(i)

    log.info("    NN: scoring %d val rows across %d distinct media",
             n_val, len(val_idx_by_media))

    fallback_val_indices: list[int] = []

    # First pass: in-medium path
    for media, idxs in val_idx_by_media.items():
        if media not in train_media_index:
            fallback_val_indices.extend(idxs)
            fallback_count += len(idxs)
            continue
        tmat, tfits = train_media_index[media]   # (n_t, 1152), (n_t,)
        idxs_np = np.array(idxs, dtype=np.int64)
        v_keys = val_genes[idxs_np]
        v_resolved = np.array([val_k2i.get(k, -1) for k in v_keys], dtype=np.int64)
        for batch_start in range(0, len(idxs_np), batch_size):
            be = min(batch_start + batch_size, len(idxs_np))
            batch_rows = idxs_np[batch_start:be]
            batch_vidx = v_resolved[batch_start:be]
            has_emb = batch_vidx >= 0
            # Rows without embeddings → global mean fallback
            for i_local, hasv in enumerate(has_emb):
                if not hasv:
                    predictions[batch_rows[i_local]] = global_mean_fit_fallback
                    no_emb_count += 1
            if not np.any(has_emb):
                continue
            v_emb_batch = val_emb[batch_vidx[has_emb]]   # (b, 1152)
            sims = v_emb_batch @ tmat.T                  # (b, n_t)
            nearest = np.argmax(sims, axis=1)
            preds_batch = tfits[nearest]
            rows_with_emb = batch_rows[has_emb]
            predictions[rows_with_emb] = preds_batch

    # Second pass: full-fallback (val media not in train)
    if fallback_val_indices:
        log.info("    NN: %d val rows in fallback path (val media unseen in train)",
                 len(fallback_val_indices))
        fb = np.array(fallback_val_indices, dtype=np.int64)
        v_keys = val_genes[fb]
        v_resolved = np.array([val_k2i.get(k, -1) for k in v_keys], dtype=np.int64)
        # Train rows that have a valid global mean fit
        valid_mask = ~np.isnan(global_mean_fit_by_idx)
        valid_train_idxs = np.where(valid_mask)[0]
        valid_train_emb = train_emb[valid_train_idxs]      # (n_valid, 1152)
        valid_train_fits = global_mean_fit_by_idx[valid_train_idxs]
        for batch_start in range(0, len(fb), batch_size):
            be = min(batch_start + batch_size, len(fb))
            batch_rows = fb[batch_start:be]
            batch_vidx = v_resolved[batch_start:be]
            has_emb = batch_vidx >= 0
            for i_local, hasv in enumerate(has_emb):
                if not hasv:
                    predictions[batch_rows[i_local]] = global_mean_fit_fallback
                    no_emb_count += 1
            if not np.any(has_emb):
                continue
            v_emb_batch = val_emb[batch_vidx[has_emb]]            # (b, 1152)
            sims = v_emb_batch @ valid_train_emb.T                # (b, n_valid)
            nearest = np.argmax(sims, axis=1)
            preds_batch = valid_train_fits[nearest]
            rows_with_emb = batch_rows[has_emb]
            predictions[rows_with_emb] = preds_batch

    pred = predictions
    if np.isnan(pred).any():
        n_unscored = int(np.isnan(pred).sum())
        log.warning("    NN: %d unscored rows; filling with global mean", n_unscored)
        pred = np.where(np.isnan(pred), global_mean_fit_fallback, pred)

    rmse = float(np.sqrt(np.mean((val_fit - pred) ** 2)))
    mae = float(np.mean(np.abs(val_fit - pred)))
    return {
        "rmse": rmse,
        "mae": mae,
        "n_rows": int(n_val),
        "predictions": pred,
        "fallback_count": int(fallback_count),
        "fallback_rate": float(fallback_count / max(n_val, 1)),
        "no_embedding_count": int(no_emb_count),
        "no_embedding_rate": float(no_emb_count / max(n_val, 1)),
    }
