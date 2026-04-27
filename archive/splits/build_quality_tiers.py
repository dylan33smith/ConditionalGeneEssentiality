#!/usr/bin/env python3
"""Build organism-quality-tiered split protocols from Phase 0 diagnostics.

Reads condition_diversity_by_org.csv and embedding_coverage_by_org.csv (produced
by run_phase0.py) and the baseline single-holdout protocol.  Generates two
additional protocols that restrict the **train** set while keeping val and test
organisms unchanged.

Criteria are defined *a priori* from data-quality metrics, not from model
performance.

Tier definitions
----------------
  curated (moderate gate):
    DROP any train organism meeting ANY of:
      • n_unique_condition_signatures ≤ 5  (near-zero condition diversity)
      • pct_unique_missing_embedding  > 10  (systematic embedding dropout)

  curated_strict (aggressive gate):
    DROP everything in curated, PLUS any organism meeting ANY of:
      • n_unique_condition_signatures < 15  (low condition diversity)

Val (Btheta) and test (DvH) are never dropped.

Usage (from repo root):
  python splits/build_quality_tiers.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PHASE0_OUTPUTS = REPO_ROOT / "data_analysis" / "outputs"
COND_DIV_CSV = PHASE0_OUTPUTS / "condition_diversity_by_org.csv"
EMBED_COV_CSV = PHASE0_OUTPUTS / "embedding_coverage_by_org.csv"
BASELINE_PROTOCOL = (
    REPO_ROOT / "splits" / "organism_single_holdout_largest_v0" / "protocol.json"
)


def load_condition_diversity() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with open(COND_DIV_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            org = row["orgId_n"]
            out[org] = {
                "n_experiments": int(row["n_experiments"]),
                "n_unique_condition_signatures": int(row["n_unique_condition_signatures"]),
                "n_unique_media_codes": int(row["n_unique_media_codes"]),
            }
    return out


def load_embedding_coverage() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with open(EMBED_COV_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            org = row["orgId"]
            out[org] = {
                "n_unique_gene_keys": int(row["n_unique_gene_keys"]),
                "n_unique_missing_embedding": int(row["n_unique_missing_embedding"]),
                "pct_unique_missing_embedding": float(row["pct_unique_missing_embedding"]),
            }
    return out


CURATED_CRITERIA = {
    "gate_name": "curated",
    "description": (
        "Drop train organisms with near-zero condition diversity "
        "(n_unique_condition_signatures <= 5) OR systematic embedding dropout "
        "(pct_unique_missing_embedding > 10%)."
    ),
    "max_condition_sigs_lte": 5,
    "max_pct_missing_embedding_gt": 10.0,
}

CURATED_STRICT_CRITERIA = {
    "gate_name": "curated_strict",
    "description": (
        "Drop all organisms from 'curated' gate, PLUS organisms with low condition "
        "diversity (n_unique_condition_signatures < 15)."
    ),
    "max_condition_sigs_lt": 15,
}


def apply_curated_gate(
    org: str,
    cond: dict[str, float],
    embed: dict[str, float],
) -> str | None:
    """Return a short reason string if organism should be dropped, else None."""
    n_sigs = cond.get("n_unique_condition_signatures", 0)
    pct_miss = embed.get("pct_unique_missing_embedding", 0.0)

    if n_sigs <= 5:
        return f"n_unique_condition_signatures={n_sigs} <= 5"
    if pct_miss > 10.0:
        return f"pct_unique_missing_embedding={pct_miss:.1f}% > 10%"
    return None


def apply_strict_gate(
    org: str,
    cond: dict[str, float],
    embed: dict[str, float],
) -> str | None:
    """Additional gate on top of curated."""
    n_sigs = cond.get("n_unique_condition_signatures", 0)
    if n_sigs < 15:
        return f"n_unique_condition_signatures={n_sigs} < 15"
    return None


def build_protocol(
    *,
    baseline: dict,
    tier_name: str,
    tier_description: str,
    excluded: dict[str, str],
    criteria: dict,
) -> dict:
    base_train = set(baseline["train_org_ids"])
    new_train = sorted(base_train - set(excluded.keys()))

    protocol_id = f"organism_single_holdout_largest_{tier_name}_v0"
    row_counts = baseline.get("row_counts_in_canonical_by_org", {})
    train_rows = sum(row_counts.get(o, 0) for o in new_train)
    excluded_rows = sum(row_counts.get(o, 0) for o in excluded)

    return {
        "protocol_id": protocol_id,
        "description": (
            f"{baseline['description']}  Quality tier '{tier_name}': {tier_description}"
        ),
        "split_axis": "orgId",
        "canonical_fitness_parquet": baseline["canonical_fitness_parquet"],
        "canonical_fitness_sha256_expected": baseline.get("canonical_fitness_sha256_expected"),
        "train_org_ids": new_train,
        "val_org_ids": baseline["val_org_ids"],
        "test_org_ids": baseline["test_org_ids"],
        "row_counts_in_canonical_by_org": dict(
            sorted(row_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "assignment_rule": baseline["assignment_rule"],
        "quality_tier": {
            "tier_name": tier_name,
            "criteria": criteria,
            "baseline_protocol_id": baseline["protocol_id"],
            "n_train_orgs_baseline": len(base_train),
            "n_train_orgs_after_gate": len(new_train),
            "n_excluded": len(excluded),
            "train_rows_after_gate": train_rows,
            "excluded_rows": excluded_rows,
            "excluded_organisms": {
                org: reason for org, reason in sorted(excluded.items())
            },
        },
    }


def main() -> int:
    for p in (COND_DIV_CSV, EMBED_COV_CSV, BASELINE_PROTOCOL):
        if not p.is_file():
            print(f"Missing prerequisite: {p}", file=sys.stderr)
            return 1

    cond = load_condition_diversity()
    embed = load_embedding_coverage()
    baseline = json.loads(BASELINE_PROTOCOL.read_text(encoding="utf-8"))

    train_orgs = set(baseline["train_org_ids"])

    curated_excluded: dict[str, str] = {}
    for org in sorted(train_orgs):
        c = cond.get(org, {"n_unique_condition_signatures": 0})
        e = embed.get(org, {"pct_unique_missing_embedding": 0.0})
        reason = apply_curated_gate(org, c, e)
        if reason:
            curated_excluded[org] = reason

    strict_excluded: dict[str, str] = dict(curated_excluded)
    for org in sorted(train_orgs):
        if org in strict_excluded:
            continue
        c = cond.get(org, {"n_unique_condition_signatures": 0})
        e = embed.get(org, {"pct_unique_missing_embedding": 0.0})
        reason = apply_strict_gate(org, c, e)
        if reason:
            strict_excluded[org] = reason

    print(f"Curated gate drops {len(curated_excluded)} organisms:", flush=True)
    for org, reason in sorted(curated_excluded.items()):
        print(f"  {org}: {reason}")
    print(f"Curated-strict gate drops {len(strict_excluded)} organisms:", flush=True)
    for org, reason in sorted(strict_excluded.items()):
        print(f"  {org}: {reason}")

    curated_proto = build_protocol(
        baseline=baseline,
        tier_name="curated",
        tier_description=CURATED_CRITERIA["description"],
        excluded=curated_excluded,
        criteria=CURATED_CRITERIA,
    )
    strict_proto = build_protocol(
        baseline=baseline,
        tier_name="curated_strict",
        tier_description=CURATED_STRICT_CRITERIA["description"],
        excluded=strict_excluded,
        criteria={**CURATED_CRITERIA, **CURATED_STRICT_CRITERIA},
    )

    for proto in (curated_proto, strict_proto):
        tier = proto["quality_tier"]["tier_name"]
        out_dir = REPO_ROOT / "splits" / f"organism_single_holdout_largest_{tier}_v0"
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / "protocol.json"
        fp.write_text(json.dumps(proto, indent=2), encoding="utf-8")
        print(f"Wrote {fp.relative_to(REPO_ROOT)}")

    manifest = {
        "manifest_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase0_condition_diversity_csv": str(COND_DIV_CSV.relative_to(REPO_ROOT)),
        "phase0_embedding_coverage_csv": str(EMBED_COV_CSV.relative_to(REPO_ROOT)),
        "baseline_protocol": str(BASELINE_PROTOCOL.relative_to(REPO_ROOT)),
        "curated": {
            "protocol_path": str(
                (REPO_ROOT / "splits" / "organism_single_holdout_largest_curated_v0" / "protocol.json")
                .relative_to(REPO_ROOT)
            ),
            "n_excluded": len(curated_excluded),
            "excluded": curated_excluded,
        },
        "curated_strict": {
            "protocol_path": str(
                (REPO_ROOT / "splits" / "organism_single_holdout_largest_curated_strict_v0" / "protocol.json")
                .relative_to(REPO_ROOT)
            ),
            "n_excluded": len(strict_excluded),
            "excluded": strict_excluded,
        },
    }
    manifest_path = REPO_ROOT / "docs" / "quality_tier_manifest_v0.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
