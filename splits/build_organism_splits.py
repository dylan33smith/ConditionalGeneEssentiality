#!/usr/bin/env python3
"""Build organism-level split protocols (M3) from canonical long Parquet.

Row assignment rule: a fitness row belongs to **train** / **val** / **test** according to
its ``orgId`` membership in the protocol's org lists (disjoint partition).

Usage (from repo root, after canonical_v0 exists):
  python splits/build_organism_splits.py

Writes under ``splits/``:
  - organism_single_holdout_largest_v0/protocol.json
  - organism_looo_v0/protocol.json

Also writes ``docs/splits_build_manifest_m3.json`` (build metadata + input checksum).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from paths import CANONICAL_FITNESS_LONG, CANONICAL_MANIFEST, REPO_ROOT, SPLITS_ROOT


def load_expected_fitness_sha256() -> str | None:
    if not CANONICAL_MANIFEST.is_file():
        return None
    data = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    for o in data.get("outputs", []):
        if o.get("path", "").endswith("fitness_experiment_long.parquet"):
            return o.get("sha256")
    return None


def count_rows_per_org(parquet_path: Path) -> tuple[dict[str, int], int]:
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Missing {parquet_path}. Run data_processing/build_canonical_v0.py first."
        )
    pf = pq.ParquetFile(parquet_path)
    if "orgId" not in pf.schema_arrow.names:
        raise RuntimeError("Parquet schema missing orgId column")
    counts: Counter[str] = Counter()
    total = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=["orgId"]):
        col = batch.column(0)
        for v in col.to_pylist():
            if v is None:
                continue
            counts[str(v)] += 1
            total += 1
    return dict(counts), total


def write_single_holdout_largest(
    counts: dict[str, int],
    *,
    n_val: int,
    n_test: int,
    expected_sha256: str | None,
) -> dict:
    """First largest -> val, next -> test (if n_test>0), rest train."""
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    val_orgs = [ranked[i][0] for i in range(min(n_val, len(ranked)))]
    rest = ranked[len(val_orgs) :]
    test_orgs = [rest[i][0] for i in range(min(n_test, len(rest)))]
    val_set = set(val_orgs)
    test_set = set(test_orgs)
    train_orgs = [o for o, _ in ranked if o not in val_set and o not in test_set]

    out_dir = SPLITS_ROOT / "organism_single_holdout_largest_v0"
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "protocol_id": "organism_single_holdout_largest_v0",
        "description": "Single-axis organism split: val = org with most rows; test = second-most; train = remaining.",
        "split_axis": "orgId",
        "canonical_fitness_parquet": str(CANONICAL_FITNESS_LONG.relative_to(REPO_ROOT)),
        "canonical_fitness_sha256_expected": expected_sha256,
        "train_org_ids": sorted(train_orgs),
        "val_org_ids": val_orgs,
        "test_org_ids": test_orgs,
        "row_counts_in_canonical_by_org": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "assignment_rule": "Row is val iff orgId in val_org_ids; test iff orgId in test_org_ids; else train.",
    }
    proto_path = out_dir / "protocol.json"
    proto_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return {"protocol_path": str(proto_path.relative_to(REPO_ROOT)), "n_train_orgs": len(train_orgs)}


def write_looo(
    counts: dict[str, int],
    *,
    expected_sha256: str | None,
) -> dict:
    all_orgs = sorted(counts.keys())
    out_dir = SPLITS_ROOT / "organism_looo_v0"
    folds_dir = out_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    folds_meta = []
    for i, val_org in enumerate(all_orgs):
        train_orgs = sorted(o for o in all_orgs if o != val_org)
        fold = {
            "fold_index": i,
            "val_org_id": val_org,
            "train_org_ids": train_orgs,
            "n_val_rows": counts[val_org],
            "n_train_rows": sum(counts[o] for o in train_orgs),
        }
        safe = val_org.replace("/", "_").replace("\\", "_")
        fp = folds_dir / f"fold_{i:03d}_val_{safe}.json"
        fp.write_text(json.dumps(fold, indent=2), encoding="utf-8")
        folds_meta.append(
            {
                "fold_index": i,
                "val_org_id": val_org,
                "file": str(fp.relative_to(REPO_ROOT)),
            }
        )

    protocol = {
        "protocol_id": "organism_looo_v0",
        "description": "Leave-one-organism-out: each fold holds out one orgId for val; all others train. No test split.",
        "split_axis": "orgId",
        "canonical_fitness_parquet": str(CANONICAL_FITNESS_LONG.relative_to(REPO_ROOT)),
        "canonical_fitness_sha256_expected": expected_sha256,
        "n_folds": len(all_orgs),
        "n_organisms": len(all_orgs),
        "folds": folds_meta,
        "row_counts_in_canonical_by_org": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "assignment_rule": "For fold i, row is val iff orgId == val_org_id; else train.",
    }
    proto_path = out_dir / "protocol.json"
    proto_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return {
        "protocol_path": str(proto_path.relative_to(REPO_ROOT)),
        "n_folds": len(all_orgs),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-looo",
        action="store_true",
        help="Only write single-holdout protocol (faster manifest for huge org counts).",
    )
    args = parser.parse_args()

    expected_sha = load_expected_fitness_sha256()
    counts, total = count_rows_per_org(CANONICAL_FITNESS_LONG)
    summed = sum(counts.values())
    if summed != total:
        raise RuntimeError(f"Internal count mismatch: {summed} vs {total}")

    sh = write_single_holdout_largest(counts, n_val=1, n_test=1, expected_sha256=expected_sha)
    looo_info = None
    if not args.skip_looo:
        looo_info = write_looo(counts, expected_sha256=expected_sha)

    build_manifest = {
        "manifest_version": 1,
        "milestone": "M3",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "canonical_fitness_parquet": str(CANONICAL_FITNESS_LONG.relative_to(REPO_ROOT)),
        "canonical_total_rows_read": total,
        "n_distinct_orgId": len(counts),
        "organism_single_holdout_largest_v0": sh,
        "organism_looo_v0": looo_info,
    }
    out_doc = REPO_ROOT / "docs" / "splits_build_manifest_m3.json"
    out_doc.write_text(json.dumps(build_manifest, indent=2), encoding="utf-8")
    print(json.dumps(build_manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
