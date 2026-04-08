#!/usr/bin/env python3
"""M4: embedding bundle manifest + coverage vs canonical long table gene keys.

Reads:
  - data/derived/canonical/v0/fitness_experiment_long.parquet (unique gene_key per orgId)
  - data/processed/ProtLM_embeddings_layer8/*.pt

Writes:
  - docs/embedding_manifest_m4.json

Usage:
  python embeddings/build_embedding_manifest_m4.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from paths import (
    CANONICAL_FITNESS_LONG,
    CANONICAL_MANIFEST,
    EMBEDDING_LAYER8_DIR,
    REPO_ROOT,
)

OUT_MANIFEST = REPO_ROOT / "docs" / "embedding_manifest_m4.json"
SUFFIX = "_proteomelm.pt"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_gene_keys_by_org(parquet_path: Path) -> dict[str, set[str]]:
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    pf = pq.ParquetFile(parquet_path)
    for col in ("orgId", "locusId"):
        if col not in pf.schema_arrow.names:
            raise RuntimeError(f"Canonical parquet missing {col}")
    by_org: dict[str, set[str]] = defaultdict(set)
    for batch in pf.iter_batches(batch_size=500_000, columns=["orgId", "locusId"]):
        orgs = batch.column(0).to_pylist()
        locs = batch.column(1).to_pylist()
        for o, loc in zip(orgs, locs, strict=False):
            if o is None or loc is None:
                continue
            o = str(o)
            loc = str(loc)
            by_org[o].add(f"{o}:{loc}")
    return dict(by_org)


def org_id_from_filename(name: str) -> str:
    if not name.endswith(SUFFIX):
        raise ValueError(f"Unexpected embedding filename: {name}")
    return name[: -len(SUFFIX)]


def load_expected_canonical_sha256() -> str | None:
    if not CANONICAL_MANIFEST.is_file():
        return None
    data = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    for o in data.get("outputs", []):
        if o.get("path", "").endswith("fitness_experiment_long.parquet"):
            return o.get("sha256")
    return None


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch not installed; install requirements.txt for tensor inspection.", file=sys.stderr)
        torch = None  # type: ignore

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = {
        "manifest_version": 1,
        "milestone": "M4",
        "generated_utc": generated,
        "bundle_id": "processed_ProtLM_layer8",
        "relative_directory": str(EMBEDDING_LAYER8_DIR.relative_to(REPO_ROOT)),
        "model_note": "Bitbol-Lab/ProteomeLM-L layer 8, D=1152 (plan §2.4)",
        "canonical_fitness_parquet": str(CANONICAL_FITNESS_LONG.relative_to(REPO_ROOT)),
        "canonical_fitness_sha256_expected": load_expected_canonical_sha256(),
        "files": [],
        "coverage_summary": {},
    }

    if not EMBEDDING_LAYER8_DIR.is_dir():
        base["status"] = "missing_directory"
        base["error"] = f"Create or link embeddings at {EMBEDDING_LAYER8_DIR}"
        OUT_MANIFEST.write_text(json.dumps(base, indent=2), encoding="utf-8")
        print(json.dumps(base, indent=2))
        return 0

    pt_files = sorted(EMBEDDING_LAYER8_DIR.glob("*_proteomelm.pt"))
    if not pt_files:
        base["status"] = "no_pt_files"
        OUT_MANIFEST.write_text(json.dumps(base, indent=2), encoding="utf-8")
        print(json.dumps(base, indent=2))
        return 0

    by_org = canonical_gene_keys_by_org(CANONICAL_FITNESS_LONG)

    total_missing = 0
    total_extra = 0
    total_canon = sum(len(s) for s in by_org.values())
    orgs_with_embed = set()

    for p in pt_files:
        name = p.name
        try:
            org = org_id_from_filename(name)
        except ValueError:
            org = p.stem
        orgs_with_embed.add(org)
        entry: dict = {
            "filename": name,
            "orgId_inferred": org,
            "bytes": p.stat().st_size,
            "sha256": file_sha256(p),
        }
        canon_set = by_org.get(org, set())
        entry["n_gene_keys_in_canonical"] = len(canon_set)

        if torch is None:
            entry["torch_load"] = "skipped"
            base["files"].append(entry)
            continue

        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            d = torch.load(p, map_location="cpu")
        emb = d.get("embeddings")
        labels = d.get("group_labels")
        if emb is not None:
            entry["embeddings_shape"] = list(emb.shape)
            entry["embedding_dim"] = int(emb.shape[1]) if len(emb.shape) > 1 else None
        if labels is None:
            entry["error"] = "missing group_labels"
            base["files"].append(entry)
            continue
        embed_set = {str(x) for x in labels}
        entry["n_group_labels"] = len(embed_set)
        miss = canon_set - embed_set
        extra = embed_set - canon_set
        entry["n_missing_in_embedding_vs_canonical"] = len(miss)
        entry["n_extra_in_embedding_vs_canonical"] = len(extra)
        total_missing += len(miss)
        total_extra += len(extra)
        if len(miss) <= 20:
            entry["missing_gene_keys_sample"] = sorted(miss)
        if len(extra) <= 20:
            entry["extra_gene_keys_sample"] = sorted(extra)
        base["files"].append(entry)

    canon_orgs = set(by_org.keys())
    base["coverage_summary"] = {
        "n_organisms_in_canonical": len(canon_orgs),
        "n_embedding_files_matched_pattern": len(pt_files),
        "canonical_orgs_without_embedding_file": sorted(canon_orgs - orgs_with_embed),
        "embedding_orgs_without_canonical_rows": sorted(orgs_with_embed - canon_orgs),
        "total_gene_keys_canonical": total_canon,
        "total_missing_gene_keys_in_embeddings": total_missing,
        "total_extra_gene_keys_in_embeddings": total_extra,
    }
    base["status"] = "ok"

    OUT_MANIFEST.write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_MANIFEST.relative_to(REPO_ROOT)}")
    print(json.dumps(base["coverage_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
