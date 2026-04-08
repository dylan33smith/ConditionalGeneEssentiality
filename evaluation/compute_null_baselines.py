#!/usr/bin/env python3
"""Compute null baselines (M3.5 / plan §7.4) for organism-level split protocols.

Baselines (train statistics applied to val/test rows):
  - **global_train_mean:** constant = mean(fit) on train rows.
  - **per_experiment_train_mean:** mean(fit) on train rows with same (orgId, expName);
    if no train rows exist for that key (typical when val organism is fully held out),
    fall back to global_train_mean.
  - **per_organism_train_mean:** mean(fit) on train rows with same orgId;
    if no train rows for that org (organism holdout), fall back to global_train_mean.

Usage:
  python evaluation/compute_null_baselines.py

Reads:
  - data/derived/canonical/v0/fitness_experiment_long.parquet
  - splits/organism_single_holdout_largest_v0/protocol.json
  - splits/organism_looo_v0/protocol.json

Writes:
  - evaluation/outputs/null_baselines_m35.json
  - evaluation/outputs/null_baselines_m35.csv  (one row per slice × baseline)
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from paths import (
    CANONICAL_FITNESS_LONG,
    CANONICAL_MANIFEST,
    OUTPUT_DIR,
    REPO_ROOT,
    SPLITS_ROOT,
)


def load_fitness_sha256() -> str | None:
    if not CANONICAL_MANIFEST.is_file():
        return None
    data = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    for o in data.get("outputs", []):
        if o.get("path", "").endswith("fitness_experiment_long.parquet"):
            return o.get("sha256")
    return None


def train_aggregates_single(
    parquet_path: Path,
    train_orgs: set[str],
) -> tuple[float, int, dict[tuple[str, str], tuple[float, int]], dict[str, tuple[float, int]]]:
    """Pass 1: sums/counts on train rows only."""
    pf = pq.ParquetFile(parquet_path)
    g_sum, g_n = 0.0, 0
    exp_sums: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    org_sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])

    for batch in pf.iter_batches(batch_size=500_000, columns=["orgId", "expName", "fit"]):
        orgs = batch.column(0).to_pylist()
        exps = batch.column(1).to_pylist()
        fits = batch.column(2).to_pylist()
        for o, e, f in zip(orgs, exps, fits, strict=False):
            if o is None or f is None or (isinstance(f, float) and math.isnan(f)):
                continue
            o = str(o)
            if o not in train_orgs:
                continue
            fv = float(f)
            g_sum += fv
            g_n += 1
            key = (o, str(e) if e is not None else "")
            exp_sums[key][0] += fv
            exp_sums[key][1] += 1
            org_sums[o][0] += fv
            org_sums[o][1] += 1

    exp_means = {k: (v[0], int(v[1])) for k, v in exp_sums.items()}
    org_means = {k: (v[0], int(v[1])) for k, v in org_sums.items()}
    return g_sum, g_n, exp_means, org_means


def eval_slice(
    parquet_path: Path,
    target_orgs: set[str],
    train_orgs: set[str],
    g_sum: float,
    g_n: int,
    exp_sums: dict[tuple[str, str], tuple[float, int]],
    org_sums: dict[str, tuple[float, int]],
    slice_name: str,
) -> dict:
    """Pass 2: RMSE on rows whose orgId is in target_orgs."""
    global_mean = g_sum / g_n if g_n else float("nan")

    def mean_exp(o: str, e: str) -> float:
        key = (o, e)
        if key in exp_sums:
            s, c = exp_sums[key]
            return s / c
        return global_mean

    def mean_org(o: str) -> float:
        if o in org_sums:
            s, c = org_sums[o]
            return s / c
        return global_mean

    pf = pq.ParquetFile(parquet_path)
    sse_g, sse_e, sse_o = 0.0, 0.0, 0.0
    n = 0
    n_fallback_exp = 0
    n_fallback_org = 0

    for batch in pf.iter_batches(batch_size=500_000, columns=["orgId", "expName", "fit"]):
        orgs = batch.column(0).to_pylist()
        exps = batch.column(1).to_pylist()
        fits = batch.column(2).to_pylist()
        for o, e, f in zip(orgs, exps, fits, strict=False):
            if o is None or f is None or (isinstance(f, float) and math.isnan(f)):
                continue
            o = str(o)
            if o not in target_orgs:
                continue
            fv = float(f)
            e_str = str(e) if e is not None else ""
            pg = global_mean
            sse_g += (fv - pg) ** 2

            key = (o, e_str)
            if key in exp_sums:
                pe = exp_sums[key][0] / exp_sums[key][1]
            else:
                pe = global_mean
                n_fallback_exp += 1
            sse_e += (fv - pe) ** 2

            if o in org_sums:
                po = org_sums[o][0] / org_sums[o][1]
            else:
                po = global_mean
                n_fallback_org += 1
            sse_o += (fv - po) ** 2
            n += 1

    def rmse(sse: float, count: int) -> float | None:
        if count == 0:
            return None
        return math.sqrt(sse / count)

    return {
        "slice": slice_name,
        "n_rows": n,
        "global_train_mean": global_mean,
        "n_train_rows_used": g_n,
        "rmse_global_train_mean": rmse(sse_g, n),
        "rmse_per_experiment_train_mean": rmse(sse_e, n),
        "rmse_per_organism_train_mean": rmse(sse_o, n),
        "n_val_rows_fallback_per_experiment_baseline": n_fallback_exp,
        "n_val_rows_fallback_per_organism_baseline": n_fallback_org,
        "note": "Fallback = global train mean when no train rows share (orgId, expName) or orgId.",
    }


def looo_global_baselines(parquet_path: Path) -> list[dict]:
    """Pass 1: org totals. Pass 2: single scan, accumulate SSE per val org."""
    pf = pq.ParquetFile(parquet_path)
    org_sum: dict[str, float] = defaultdict(float)
    org_cnt: dict[str, int] = defaultdict(int)
    total_sum, total_n = 0.0, 0

    for batch in pf.iter_batches(batch_size=500_000, columns=["orgId", "fit"]):
        orgs = batch.column(0).to_pylist()
        fits = batch.column(1).to_pylist()
        for o, f in zip(orgs, fits, strict=False):
            if o is None or f is None or (isinstance(f, float) and math.isnan(f)):
                continue
            o = str(o)
            fv = float(f)
            org_sum[o] += fv
            org_cnt[o] += 1
            total_sum += fv
            total_n += 1

    sse_by_org: dict[str, float] = defaultdict(float)
    for batch in pq.ParquetFile(parquet_path).iter_batches(
        batch_size=500_000, columns=["orgId", "fit"]
    ):
        orgs = batch.column(0).to_pylist()
        fits = batch.column(1).to_pylist()
        for o, f in zip(orgs, fits, strict=False):
            if o is None or f is None or (isinstance(f, float) and math.isnan(f)):
                continue
            o = str(o)
            fv = float(f)
            s_o, n_o = org_sum[o], org_cnt[o]
            tr_sum = total_sum - s_o
            tr_n = total_n - n_o
            pred = tr_sum / tr_n if tr_n else float("nan")
            sse_by_org[o] += (fv - pred) ** 2

    folds_out = []
    for o in sorted(org_sum.keys()):
        n_o = org_cnt[o]
        s_o = org_sum[o]
        tr_n = total_n - n_o
        tr_sum = total_sum - s_o
        pred = tr_sum / tr_n if tr_n else float("nan")
        sse = sse_by_org[o]
        rmse = math.sqrt(sse / n_o) if n_o else None
        folds_out.append(
            {
                "val_org_id": o,
                "n_val_rows": n_o,
                "train_rows_excluded_val_org": tr_n,
                "global_train_mean_excluding_val_org": pred,
                "rmse_global_train_mean": rmse,
            }
        )
    return folds_out


def main() -> int:
    if not CANONICAL_FITNESS_LONG.is_file():
        raise SystemExit(f"Missing canonical parquet: {CANONICAL_FITNESS_LONG}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sha = load_fitness_sha256()

    single_path = SPLITS_ROOT / "organism_single_holdout_largest_v0" / "protocol.json"
    if not single_path.is_file():
        raise SystemExit(f"Missing split protocol: {single_path}")

    single = json.loads(single_path.read_text(encoding="utf-8"))
    train_orgs = set(single["train_org_ids"])
    val_orgs = set(single["val_org_ids"])
    test_orgs = set(single["test_org_ids"])

    g_sum, g_n, exp_sums, org_sums = train_aggregates_single(CANONICAL_FITNESS_LONG, train_orgs)
    val_metrics = eval_slice(
        CANONICAL_FITNESS_LONG,
        val_orgs,
        train_orgs,
        g_sum,
        g_n,
        exp_sums,
        org_sums,
        "val",
    )
    test_metrics = eval_slice(
        CANONICAL_FITNESS_LONG,
        test_orgs,
        train_orgs,
        g_sum,
        g_n,
        exp_sums,
        org_sums,
        "test",
    )

    looo_folds = looo_global_baselines(CANONICAL_FITNESS_LONG)
    rmse_vals = [x["rmse_global_train_mean"] for x in looo_folds if x["rmse_global_train_mean"] is not None]
    looo_summary = {
        "n_folds": len(looo_folds),
        "mean_rmse_global_train_mean_across_folds": sum(rmse_vals) / len(rmse_vals) if rmse_vals else None,
        "min_rmse": min(rmse_vals) if rmse_vals else None,
        "max_rmse": max(rmse_vals) if rmse_vals else None,
    }

    report = {
        "manifest_version": 1,
        "milestone": "M3.5",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "canonical_fitness_parquet": str(CANONICAL_FITNESS_LONG.relative_to(REPO_ROOT)),
        "canonical_fitness_sha256_expected": sha,
        "baselines_documented": [
            "global_train_mean",
            "per_experiment_train_mean_with_fallback",
            "per_organism_train_mean_with_fallback",
        ],
        "organism_single_holdout_largest_v0": {
            "protocol_path": str(single_path.relative_to(REPO_ROOT)),
            "val": val_metrics,
            "test": test_metrics,
        },
        "organism_looo_v0": {
            "note": "LOOO table lists global train mean excluding val organism only. "
            "Per-experiment / per-organism baselines coincide with global for organism-cold val rows.",
            "summary": looo_summary,
            "folds": looo_folds,
        },
    }

    json_path = OUTPUT_DIR / "null_baselines_m35.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = OUTPUT_DIR / "null_baselines_m35.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "protocol",
                "slice",
                "n_rows",
                "rmse_global_train_mean",
                "rmse_per_experiment_train_mean",
                "rmse_per_organism_train_mean",
            ]
        )
        w.writerow(
            [
                "organism_single_holdout_largest_v0",
                val_metrics["slice"],
                val_metrics["n_rows"],
                val_metrics["rmse_global_train_mean"],
                val_metrics["rmse_per_experiment_train_mean"],
                val_metrics["rmse_per_organism_train_mean"],
            ]
        )
        w.writerow(
            [
                "organism_single_holdout_largest_v0",
                test_metrics["slice"],
                test_metrics["n_rows"],
                test_metrics["rmse_global_train_mean"],
                test_metrics["rmse_per_experiment_train_mean"],
                test_metrics["rmse_per_organism_train_mean"],
            ]
        )
        for row in looo_folds:
            w.writerow(
                [
                    "organism_looo_v0",
                    f"val_{row['val_org_id']}",
                    row["n_val_rows"],
                    row["rmse_global_train_mean"],
                    "",
                    "",
                ]
            )

    print(json.dumps({k: v for k, v in report.items() if k != "organism_looo_v0"}, indent=2))
    print(f"... looo folds: {len(looo_folds)} (see {json_path.relative_to(REPO_ROOT)})")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
