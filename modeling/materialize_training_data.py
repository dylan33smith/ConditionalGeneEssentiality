#!/usr/bin/env python3
"""Pre-materialize filtered train/val rows to numpy arrays for fast epoch iteration.

Streams the canonical Parquet once, applies all joins (embedding + condition encoding)
and arm-specific filters, then writes compact numpy arrays to disk.  Training can then
skip all per-row Python filtering every epoch and use fast vectorized tensor ops instead.

Expected speedup: 10–50× per epoch over the streaming pipeline.

Usage
-----
# Arm A (weighted_full):
  python modeling/materialize_training_data.py \\
    --protocol splits/organism_multi_holdout_overlap_v0/protocol.json \\
    --arm weighted_full \\
    --output-dir data/materialized/multi_overlap_armA

# Arm B (strict_slice, relaxed thresholds):
  python modeling/materialize_training_data.py \\
    --protocol splits/organism_multi_holdout_overlap_v0/protocol.json \\
    --arm strict_slice --strict-min-cor12 0.2 --strict-min-abs-t 1.0 \\
    --output-dir data/materialized/multi_overlap_armB

Then pass --materialized-dir <same path> to train.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from condition_store import ExperimentConditionEncoding
from data import ArmName, iter_filtered_row_dicts
from embedding_store import EmbeddingStore
from paths import EMBEDDING_LAYER8_DIR, REPO_ROOT, resolve_parquet_path
from split_protocol import load_split_protocol

DEFAULT_CONDITION_MANIFEST = REPO_ROOT / "docs" / "condition_encoding_manifest_v0.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize filtered rows to fast numpy arrays.")
    p.add_argument("--protocol", type=str,
                   default="splits/organism_multi_holdout_overlap_v0/protocol.json")
    p.add_argument("--arm", type=str, choices=("weighted_full", "strict_slice"),
                   default="weighted_full")
    p.add_argument("--condition-manifest", type=str, default=str(DEFAULT_CONDITION_MANIFEST))
    p.add_argument("--embed-dir", type=str, default=str(EMBEDDING_LAYER8_DIR))
    p.add_argument("--parquet", type=str, default="",
                   help="Override canonical Parquet path (default from protocol).")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory to write materialized arrays (train/ and val/ subdirs).")
    p.add_argument("--strict-min-cor12", type=float, default=0.4)
    p.add_argument("--strict-min-abs-t", type=float, default=2.0)
    p.add_argument("--cor12-floor", type=float, default=0.05)
    p.add_argument("--weight-t-scale", type=float, default=4.0)
    p.add_argument("--splits", type=str, default="both",
                   choices=("train", "val", "both"),
                   help="Which splits to materialize.")
    return p.parse_args()


def _build_emb_matrix(
    embed_store: EmbeddingStore,
    org_ids: set[str],
) -> tuple[np.ndarray, dict[tuple[str, str], int]]:
    """Stack all gene embeddings for the given orgs into one matrix.

    Returns:
        emb_matrix: float32 ndarray of shape (N_unique_genes, gene_dim)
        slot_map:   dict mapping (orgId, gene_key) -> row index in emb_matrix
    """
    rows: list[np.ndarray] = []
    slot_map: dict[tuple[str, str], int] = {}
    for org in sorted(org_ids):
        tab = embed_store._by_org[org]
        emb_np = tab.embeddings.numpy()  # already float32, CPU
        for gene_key, local_idx in tab.gene_to_row.items():
            key = (org, gene_key)
            if key not in slot_map:
                slot_map[key] = len(rows)
                rows.append(emb_np[local_idx])
    emb_matrix = np.stack(rows, axis=0)  # (N, gene_dim)
    return emb_matrix, slot_map


def _materialize_split(
    split_name: str,
    org_ids: set[str],
    parquet_path: Path,
    arm: ArmName,
    embed_store: EmbeddingStore,
    condition_store: ExperimentConditionEncoding,
    out_dir: Path,
    *,
    strict_min_cor12: float,
    strict_min_abs_t: float,
    cor12_floor: float,
    weight_t_scale: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{split_name}] building embedding matrix for {len(org_ids)} orgs…", flush=True)
    t0 = time.perf_counter()
    emb_matrix, slot_map = _build_emb_matrix(embed_store, org_ids)
    gene_dim = emb_matrix.shape[1]
    print(f"  emb_matrix shape={emb_matrix.shape}  ({time.perf_counter()-t0:.1f}s)", flush=True)

    print(f"[{split_name}] streaming Parquet and collecting filtered rows…", flush=True)
    t1 = time.perf_counter()

    gene_emb_idx_list: list[int] = []
    fit_list: list[float] = []
    weight_list: list[float] = []
    ce_cat_list: list[tuple[int, ...]] = []
    ce_cont_list: list[tuple[float, ...]] = []
    gene_key_list: list[str] = []  # kept for val (Spearman grouping)

    n_rows = 0
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
    ):
        org = row["orgId"]
        gk = row["gene_key"]
        slot = slot_map[(org, gk)]
        gene_emb_idx_list.append(slot)
        fit_list.append(float(row["fit"]))
        weight_list.append(float(row["weight"]))
        ce_cat_list.append(row["ce_cat"])
        ce_cont_list.append(row["ce_cont"])
        gene_key_list.append(gk)
        n_rows += 1
        if n_rows % 1_000_000 == 0:
            print(f"  {n_rows:,} rows  ({time.perf_counter()-t1:.0f}s)", flush=True)

    elapsed = time.perf_counter() - t1
    print(f"  done: {n_rows:,} rows  ({elapsed:.1f}s)", flush=True)

    if n_rows == 0:
        raise RuntimeError(f"No rows passed filters for {split_name} split — check arm/thresholds.")

    k_cat = len(ce_cat_list[0])
    n_cont = len(ce_cont_list[0])

    print(f"[{split_name}] converting to numpy arrays…", flush=True)
    t2 = time.perf_counter()

    gene_emb_idx = np.array(gene_emb_idx_list, dtype=np.int32)
    fit_arr = np.array(fit_list, dtype=np.float32)
    weight_arr = np.array(weight_list, dtype=np.float32)
    ce_cat_arr = np.array(ce_cat_list, dtype=np.int32).reshape(n_rows, k_cat)
    ce_cont_arr = np.array(ce_cont_list, dtype=np.float32).reshape(n_rows, n_cont)
    gene_keys_arr = np.array(gene_key_list, dtype=object)

    print(f"  conversion done ({time.perf_counter()-t2:.1f}s)", flush=True)

    print(f"[{split_name}] saving to {out_dir}…", flush=True)
    np.save(out_dir / "emb_matrix.npy", emb_matrix)
    np.save(out_dir / "gene_emb_idx.npy", gene_emb_idx)
    np.save(out_dir / "fit.npy", fit_arr)
    np.save(out_dir / "weight.npy", weight_arr)
    np.save(out_dir / "ce_cat.npy", ce_cat_arr)
    np.save(out_dir / "ce_cont.npy", ce_cont_arr)
    np.save(out_dir / "gene_keys.npy", gene_keys_arr)

    meta = {
        "split": split_name,
        "n_rows": n_rows,
        "gene_dim": int(gene_dim),
        "k_cat": int(k_cat),
        "n_cont": int(n_cont),
        "n_unique_gene_slots": int(emb_matrix.shape[0]),
        "arm": arm,
        "strict_min_cor12": strict_min_cor12,
        "strict_min_abs_t": strict_min_abs_t,
        "cor12_floor": cor12_floor,
        "weight_t_scale": weight_t_scale,
        "org_ids": sorted(org_ids),
        "files": {
            "emb_matrix": "emb_matrix.npy",
            "gene_emb_idx": "gene_emb_idx.npy",
            "fit": "fit.npy",
            "weight": "weight.npy",
            "ce_cat": "ce_cat.npy",
            "ce_cont": "ce_cont.npy",
            "gene_keys": "gene_keys.npy",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    total_bytes = sum(
        (out_dir / f).stat().st_size
        for f in ["emb_matrix.npy", "gene_emb_idx.npy", "fit.npy",
                  "weight.npy", "ce_cat.npy", "ce_cont.npy", "gene_keys.npy"]
    )
    print(
        f"[{split_name}] done. {n_rows:,} rows, "
        f"{total_bytes / 1e9:.2f} GB on disk  ({time.perf_counter()-t0:.1f}s total)",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    arm: ArmName = args.arm  # type: ignore[assignment]

    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = REPO_ROOT / protocol_path
    protocol = load_split_protocol(protocol_path)

    rel_pq = args.parquet.strip() or protocol.canonical_fitness_parquet
    parquet_path = Path(rel_pq) if Path(rel_pq).is_absolute() else resolve_parquet_path(rel_pq)
    if not parquet_path.is_file():
        print(f"Missing Parquet: {parquet_path}", file=sys.stderr)
        return 1

    cond_manifest = Path(args.condition_manifest)
    if not cond_manifest.is_absolute():
        cond_manifest = REPO_ROOT / cond_manifest

    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root

    print("Loading condition encoding…", flush=True)
    condition_store = ExperimentConditionEncoding(cond_manifest, repo_root=REPO_ROOT)
    summ = condition_store.manifest_summary()
    print(f"  {summ['encoding_id']}  n_experiments={summ['n_experiments_indexed']}", flush=True)

    train_orgs = set(protocol.train_org_ids)
    val_orgs = set(protocol.val_org_ids)
    all_orgs = train_orgs | val_orgs

    embed_dir = Path(args.embed_dir)
    if not embed_dir.is_absolute():
        embed_dir = REPO_ROOT / embed_dir
    print(f"Loading embeddings ({len(all_orgs)} orgs)…", flush=True)
    embed_store = EmbeddingStore(embed_dir, all_orgs, device=torch.device("cpu"))
    print("  embeddings loaded.", flush=True)

    kwargs = dict(
        parquet_path=parquet_path,
        arm=arm,
        embed_store=embed_store,
        condition_store=condition_store,
        strict_min_cor12=args.strict_min_cor12,
        strict_min_abs_t=args.strict_min_abs_t,
        cor12_floor=args.cor12_floor,
        weight_t_scale=args.weight_t_scale,
    )

    t_total = time.perf_counter()
    if args.splits in ("train", "both"):
        _materialize_split("train", train_orgs, out_dir=out_root / "train", **kwargs)
    if args.splits in ("val", "both"):
        _materialize_split("val", val_orgs, out_dir=out_root / "val", **kwargs)

    print(f"\nAll done. Total wall time: {time.perf_counter()-t_total:.1f}s", flush=True)
    print(f"Pass --materialized-dir {out_root.relative_to(REPO_ROOT)} to train.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
