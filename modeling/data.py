"""Stream canonical Parquet rows; join condition encoding; shuffle buffer; batch tensors."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

import pyarrow.parquet as pq
import torch

from embedding_store import EmbeddingStore

if TYPE_CHECKING:
    from condition_store import ExperimentConditionEncoding

ArmName = Literal["weighted_full", "strict_slice"]

PARQUET_COLUMNS = (
    "orgId",
    "gene_key",
    "expName",
    "media",
    "fit",
    "cor12",
    "abs_t",
)


def _float_cell(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x):
            return None
        return float(x)
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def row_weight(cor12: float | None, abs_t: float | None, cor12_floor: float, t_scale: float) -> float:
    c = _float_cell(cor12)
    if c is None:
        c = cor12_floor
    c = max(cor12_floor, min(1.0, c))
    t = _float_cell(abs_t)
    if t is None:
        t = 0.0
    wt = min(1.0, t / t_scale) if t_scale > 0 else 0.0
    return max(1e-6, c * wt)


def strict_row_ok(
    cor12: float | None,
    abs_t: float | None,
    min_cor12: float,
    min_abs_t: float,
) -> bool:
    c = _float_cell(cor12)
    t = _float_cell(abs_t)
    if c is None or t is None:
        return False
    return c >= min_cor12 and t >= min_abs_t


def _row_used_by_model(
    arm: ArmName,
    cor12: Any,
    abs_t: Any,
    strict_min_cor12: float,
    strict_min_abs_t: float,
) -> bool:
    if arm == "strict_slice":
        return strict_row_ok(cor12, abs_t, strict_min_cor12, strict_min_abs_t)
    return True


def _nonempty_text(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def count_split_row_stats(
    parquet_path,
    train_orgs: set[str],
    val_orgs: set[str],
    embed_store: EmbeddingStore,
    arm: ArmName,
    condition_keys: frozenset[tuple[str, str]],
    *,
    strict_min_cor12: float,
    strict_min_abs_t: float,
) -> dict[str, int | float]:
    """Full-file counts for metrics (matches iter_filtered_row_dicts gating, excluding max_rows)."""
    n_tr_fit = n_tr_emb = n_tr_used = 0
    n_va_fit = n_va_emb = n_va_used = 0
    pf = pq.ParquetFile(parquet_path)
    for col in PARQUET_COLUMNS:
        if col not in pf.schema_arrow.names:
            raise RuntimeError(f"Parquet missing column {col!r}")
    for batch in pf.iter_batches(batch_size=500_000, columns=list(PARQUET_COLUMNS)):
        cols = {PARQUET_COLUMNS[i]: batch.column(i).to_pylist() for i in range(len(PARQUET_COLUMNS))}
        n = len(cols["orgId"])
        for i in range(n):
            o = cols["orgId"][i]
            if o is None:
                continue
            org = str(o)
            in_tr = org in train_orgs
            in_va = org in val_orgs
            if not in_tr and not in_va:
                continue
            fit = _float_cell(cols["fit"][i])
            if fit is None:
                continue
            if in_tr:
                n_tr_fit += 1
            else:
                n_va_fit += 1
            gk = cols["gene_key"][i]
            if gk is None:
                continue
            gene_key = str(gk)
            if not embed_store.has_embedding(org, gene_key):
                continue
            if in_tr:
                n_tr_emb += 1
            else:
                n_va_emb += 1
            media = _nonempty_text(cols["media"][i])
            if media is None:
                continue
            ex = cols["expName"][i]
            if ex is None:
                continue
            exp_name = str(ex)
            if (org, exp_name) not in condition_keys:
                continue
            cor12 = cols["cor12"][i]
            abs_t = cols["abs_t"][i]
            if not _row_used_by_model(arm, cor12, abs_t, strict_min_cor12, strict_min_abs_t):
                continue
            if in_tr:
                n_tr_used += 1
            else:
                n_va_used += 1

    frac_no_emb = ((n_va_fit - n_va_emb) / n_va_fit) if n_va_fit > 0 else 0.0
    frac_strict = ((n_va_emb - n_va_used) / n_va_emb) if n_va_emb > 0 else 0.0
    return {
        "n_train_rows_with_fit": n_tr_fit,
        "n_train_rows_with_fit_and_embedding": n_tr_emb,
        "n_train_rows_used_by_model_under_arm": n_tr_used,
        "n_val_rows_with_fit": n_va_fit,
        "n_val_rows_with_fit_and_embedding": n_va_emb,
        "n_val_rows_used_by_model_under_arm": n_va_used,
        "frac_val_rows_dropped_no_embedding": float(frac_no_emb),
        "frac_val_rows_dropped_strict_after_embedding": float(frac_strict),
    }


def iter_filtered_row_dicts(
    parquet_path,
    org_ids: set[str],
    arm: ArmName,
    embed_store: EmbeddingStore,
    condition_store: ExperimentConditionEncoding,
    *,
    strict_min_cor12: float,
    strict_min_abs_t: float,
    cor12_floor: float,
    weight_t_scale: float,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    pf = pq.ParquetFile(parquet_path)
    for col in PARQUET_COLUMNS:
        if col not in pf.schema_arrow.names:
            raise RuntimeError(f"Parquet missing column {col!r}")
    n_out = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=list(PARQUET_COLUMNS)):
        cols = {PARQUET_COLUMNS[i]: batch.column(i).to_pylist() for i in range(len(PARQUET_COLUMNS))}
        n = len(cols["orgId"])
        for i in range(n):
            o = cols["orgId"][i]
            if o is None:
                continue
            org = str(o)
            if org not in org_ids:
                continue
            gk = cols["gene_key"][i]
            if gk is None:
                continue
            gene_key = str(gk)
            fit = _float_cell(cols["fit"][i])
            if fit is None:
                continue
            if not embed_store.has_embedding(org, gene_key):
                continue
            media = _nonempty_text(cols["media"][i])
            if media is None:
                continue
            ex = cols["expName"][i]
            if ex is None:
                continue
            exp_name = str(ex)
            enc = condition_store.encode(org, exp_name)
            if enc is None:
                continue
            cats, conts = enc
            cor12 = cols["cor12"][i]
            abs_t = cols["abs_t"][i]
            if arm == "strict_slice":
                if not strict_row_ok(cor12, abs_t, strict_min_cor12, strict_min_abs_t):
                    continue
            w = (
                1.0
                if arm == "strict_slice"
                else row_weight(cor12, abs_t, cor12_floor=cor12_floor, t_scale=weight_t_scale)
            )
            yield {
                "orgId": org,
                "gene_key": gene_key,
                "expName": exp_name,
                "media": media,
                "fit": fit,
                "weight": w,
                "ce_cat": cats,
                "ce_cont": conts,
            }
            n_out += 1
            if max_rows is not None and n_out >= max_rows:
                return


def _collate(
    rows: list[dict[str, Any]],
    embed_store: EmbeddingStore,
    device: torch.device,
    cat_field_order: tuple[str, ...],
    n_cont: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    org_ids = [r["orgId"] for r in rows]
    gene_keys = [r["gene_key"] for r in rows]
    x_gene = embed_store.vectors_for_rows(org_ids, gene_keys)
    k = len(cat_field_order)
    cat_mat = torch.zeros(len(rows), k, dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        cats: tuple[int, ...] = r["ce_cat"]
        for j in range(k):
            cat_mat[i, j] = int(cats[j])
    cont = torch.zeros(len(rows), n_cont, dtype=torch.float32, device=device)
    for i, r in enumerate(rows):
        ct: tuple[float, ...] = r["ce_cont"]
        for j in range(n_cont):
            cont[i, j] = float(ct[j])
    y = torch.tensor([r["fit"] for r in rows], dtype=torch.float32, device=device)
    w = torch.tensor([r["weight"] for r in rows], dtype=torch.float32, device=device)
    return x_gene, cat_mat, cont, y, w


def shuffled_training_batches(
    parquet_path,
    org_ids: set[str],
    arm: ArmName,
    embed_store: EmbeddingStore,
    condition_store: ExperimentConditionEncoding,
    device: torch.device,
    *,
    batch_size: int,
    shuffle_buffer: int,
    seed: int,
    cat_field_order: tuple[str, ...],
    n_cont: int,
    strict_min_cor12: float,
    strict_min_abs_t: float,
    cor12_floor: float,
    weight_t_scale: float,
    max_rows: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    rng = random.Random(seed)
    buf: list[dict[str, Any]] = []

    def flush_shuffled() -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        nonlocal buf
        if not buf:
            return
        rng.shuffle(buf)
        pos = 0
        while pos + batch_size <= len(buf):
            chunk = buf[pos : pos + batch_size]
            pos += batch_size
            yield _collate(chunk, embed_store, device, cat_field_order, n_cont)
        buf = buf[pos:]

    for row in iter_filtered_row_dicts(
        parquet_path,
        org_ids,
        arm,
        embed_store,
        condition_store,
        strict_min_cor12=strict_min_cor12,
        strict_min_abs_t=strict_min_abs_t,
        cor12_floor=cor12_floor,
        weight_t_scale=weight_t_scale,
        max_rows=max_rows,
    ):
        buf.append(row)
        if len(buf) >= shuffle_buffer:
            yield from flush_shuffled()
    rng.shuffle(buf)
    pos = 0
    while pos + batch_size <= len(buf):
        chunk = buf[pos : pos + batch_size]
        pos += batch_size
        yield _collate(chunk, embed_store, device, cat_field_order, n_cont)


def iter_val_batches(
    parquet_path,
    org_ids: set[str],
    arm: ArmName,
    embed_store: EmbeddingStore,
    condition_store: ExperimentConditionEncoding,
    device: torch.device,
    *,
    batch_size: int,
    cat_field_order: tuple[str, ...],
    n_cont: int,
    strict_min_cor12: float,
    strict_min_abs_t: float,
    cor12_floor: float,
    weight_t_scale: float,
    max_rows: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]:
    buf: list[dict[str, Any]] = []
    for row in iter_filtered_row_dicts(
        parquet_path,
        org_ids,
        arm,
        embed_store,
        condition_store,
        strict_min_cor12=strict_min_cor12,
        strict_min_abs_t=strict_min_abs_t,
        cor12_floor=cor12_floor,
        weight_t_scale=weight_t_scale,
        max_rows=max_rows,
    ):
        buf.append(row)
        if len(buf) >= batch_size:
            gks = [r["gene_key"] for r in buf[:batch_size]]
            x, c, xc, y, w = _collate(buf[:batch_size], embed_store, device, cat_field_order, n_cont)
            buf = buf[batch_size:]
            yield x, c, xc, y, w, gks
    if buf:
        gks = [r["gene_key"] for r in buf]
        x, c, xc, y, w = _collate(buf, embed_store, device, cat_field_order, n_cont)
        yield x, c, xc, y, w, gks
