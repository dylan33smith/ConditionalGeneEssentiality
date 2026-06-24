"""Shared train→eval→report runner for ranking experiments.

A new test is declarative: build one or more `ArmSpec`s and call
`run_experiment(...)`. The runner prepares the data ONCE, trains each arm across
seeds, and emits the STANDARDIZED comparison — the model plus the
split-specific baselines (chem-kNN gate, chem-NULL, linear-MF) on the identical
eligible val gene set (denominator parity), with one metric schema everywhere
(NDCG@5 PRIMARY + NDCG@1/3 + precision@5, then within-gene Spearman/Kendall —
NDCG@5 and Spearman each with a hierarchical-bootstrap CI), per-seed and
seed-averaged, written to a tidy CSV and logged side-by-side vs the gate.

Example (a 2-arm loss comparison):

    from src.ranking.runner import ArmSpec, run_experiment
    run_experiment(
        [ArmSpec("huber", loss="pointwise_huber"),
         ArmSpec("lambdarank", loss="lambdarank")],
        orgs=None, model_seeds=(0, 1, 2), out_dir="artifacts/runs/my_test", tag="my_test")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ranking.pipeline import prepare_r1_data, train_r1_arm
from src.ranking.train import train_arm

log = logging.getLogger(__name__)

# the single metric schema every ranking result reports. NDCG@5 is the PRIMARY
# metric project-wide (retrieval / "top stressors"); within-gene Spearman is the
# secondary completeness metric — the order here reflects that. Both headline
# metrics carry the harness's hierarchical (org->gene) bootstrap CI (computed per
# method in _metrics_for_pred); carrying the bounds here surfaces them in the CSV +
# report so a model-vs-gate disjoint-CI check is possible on EITHER metric
# (NDCG@5 first; e.g. the R-COLD confirmatory step).
METRIC_KEYS = ("ndcg_at_5", "ndcg_at_5_ci_low", "ndcg_at_5_ci_high",
               "ndcg_at_1", "ndcg_at_3", "precision_at_5",
               "spearman", "spearman_ci_low", "spearman_ci_high",
               "kendall", "n_genes")
# methods always scored side-by-side (denominator parity)
METHODS = ("model", "chem_knn", "linear_mf", "chem_null")
GATE = "chem_knn"   # the baseline to beat (R1-DEC-001)


@dataclass
class ArmSpec:
    """One declarative arm. `loss` applies to the multihot encoder; a fingerprint
    `encoder` (morgan/rdkit/maccs/…) trains under MSE (the R1 encoder axis)."""
    name: str
    loss: str = "pointwise_huber"
    encoder: str = "multihot_425"
    epochs: int = 8


def _train_one(spec: ArmSpec, data, seed: int) -> dict:
    """Dispatch to the right trainer and return its comparison dict."""
    if spec.encoder == "multihot_425":
        res = train_arm(spec.loss, data, seed=seed, epochs=spec.epochs)
    else:
        res = train_r1_arm(spec.encoder, data, seed=seed, epochs=spec.epochs)
    return res["comparison"]


def _agg(comps: list[dict], method: str) -> dict:
    return {k: float(np.mean([c[method][k] for c in comps])) for k in METRIC_KEYS}


def run_arm(spec: ArmSpec, data, *, model_seeds=(0,), gate: str = GATE) -> dict:
    """Train one arm across seeds on prepared data; return per-seed + seed-mean
    comparisons (model + baselines). `gate` selects the baseline-to-beat for the
    logged side-by-side (default chem-kNN; the cold_gene diagnostic uses chem-NULL,
    since chem-kNN can't score a held-out gene)."""
    comps = []
    for s in model_seeds:
        c = _train_one(spec, data, s)
        comps.append(c)
        log.info("    [%s seed=%d] model NDCG@5=%.4f Spear=%.4f | %s NDCG@5=%.4f",
                 spec.name, s, c["model"]["ndcg_at_5"], c["model"]["spearman"],
                 gate, c[gate]["ndcg_at_5"])
    return {"name": spec.name, "spec": spec, "per_seed": comps,
            "agg": {m: _agg(comps, m) for m in METHODS}, "n_seeds": len(model_seeds)}


def standardized_report(results: list[dict], *, out_dir: str | Path, tag: str,
                        gate: str = GATE) -> pd.DataFrame:
    """Tidy CSV (one row per arm×method, full metric schema) + a side-by-side log
    of each arm's model vs the gate baseline. Returns the long DataFrame. `gate`
    defaults to chem-kNN; the cold_gene diagnostic passes chem-NULL."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        for method in METHODS:
            rows.append({"arm": r["name"], "method": method, "n_seeds": r["n_seeds"],
                         **r["agg"][method]})
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{tag}_metrics.csv", index=False)

    # NDCG@5 is PRIMARY — it leads the table; Spearman is the secondary column.
    log.info("STANDARDIZED REPORT [%s] — model vs %s gate (seed-mean):", tag, gate)
    log.info("    %-16s %8s %9s %8s   %-14s", "arm", "NDCG@5", "Spearman", "prec@5", "vs gate NDCG@5")
    for r in results:
        m, g = r["agg"]["model"], r["agg"][gate]
        d = m["ndcg_at_5"] - g["ndcg_at_5"]
        verdict = "BEATS gate" if d > 0 else f"{d:+.4f}"
        log.info("    %-16s %8.4f %9.4f %8.4f   %-14s",
                 r["name"], m["ndcg_at_5"], m["spearman"], m["precision_at_5"], verdict)
    log.info("    gate(%s) NDCG@5=%.4f Spearman=%.4f", gate,
             results[0]["agg"][gate]["ndcg_at_5"], results[0]["agg"][gate]["spearman"])

    # Hierarchical-bootstrap CI: model vs gate, with a disjointness check, on BOTH
    # headline metrics — NDCG@5 PRIMARY (the promotion metric), Spearman secondary.
    # (For multi-seed runs these bounds are the mean of the per-seed CIs — a summary
    # band, not a pooled-across-seeds CI; a pooled-prediction bootstrap is stronger.)
    for metric, label in (("ndcg_at_5", "NDCG@5  "), ("spearman", "Spearman")):
        log.info("    %-16s  %s [95%% CI]      vs gate(%s) [95%% CI]   disjoint?",
                 "arm", label, gate)
        for r in results:
            m, g = r["agg"]["model"], r["agg"][gate]
            disjoint = _ci_disjoint(m, g, metric)
            flag = "—" if disjoint is None else ("YES" if disjoint else "no (overlap)")
            log.info("    %-16s  %.4f [%.4f, %.4f]   %.4f [%.4f, %.4f]   %s",
                     r["name"], m[metric], m[f"{metric}_ci_low"], m[f"{metric}_ci_high"],
                     g[metric], g[f"{metric}_ci_low"], g[f"{metric}_ci_high"], flag)
    return df


def _ci_disjoint(a: dict, b: dict, metric: str = "ndcg_at_5") -> bool | None:
    """True if a and b have non-overlapping 95% CIs for `metric` (either direction).
    `metric` defaults to NDCG@5 (the primary metric). None when a CI is unavailable
    (NaN bound, e.g. an inapplicable baseline)."""
    lo_a, hi_a = a.get(f"{metric}_ci_low"), a.get(f"{metric}_ci_high")
    lo_b, hi_b = b.get(f"{metric}_ci_low"), b.get(f"{metric}_ci_high")
    if any(x is None or x != x for x in (lo_a, hi_a, lo_b, hi_b)):  # NaN-safe
        return None
    return hi_a < lo_b or hi_b < lo_a


def run_experiment(specs: list[ArmSpec], *, orgs=None, split_seed: int = 0,
                   model_seeds=(0,), out_dir: str | Path, tag: str,
                   data=None) -> list[dict]:
    """Prepare data once, run every arm across seeds, emit the standardized report.
    Pass `data` to reuse an already-prepared dataset (e.g. from R-EVAL)."""
    if data is None:
        log.info("[runner] preparing data (split seed=%d, orgs=%s)",
                 split_seed, orgs if orgs else "ALL")
        data = prepare_r1_data(list(orgs) if orgs else None, seed=split_seed)
    results = [run_arm(s, data, model_seeds=model_seeds) for s in specs]
    standardized_report(results, out_dir=out_dir, tag=tag)
    return results
