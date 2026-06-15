"""Shared train→eval→report runner for ranking experiments.

A new test is declarative: build one or more `ArmSpec`s and call
`run_experiment(...)`. The runner prepares the data ONCE, trains each arm across
seeds, and emits the STANDARDIZED comparison — the model plus the
split-specific baselines (chem-kNN gate, chem-NULL, linear-MF) on the identical
eligible val gene set (denominator parity), with one metric schema everywhere
(within-gene Spearman/Kendall + NDCG@1/3/5 + precision@5), per-seed and
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

# the single metric schema every ranking result reports
METRIC_KEYS = ("spearman", "kendall", "ndcg_at_1", "ndcg_at_3", "ndcg_at_5",
               "precision_at_5", "n_genes")
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


def run_arm(spec: ArmSpec, data, *, model_seeds=(0,)) -> dict:
    """Train one arm across seeds on prepared data; return per-seed + seed-mean
    comparisons (model + baselines)."""
    comps = []
    for s in model_seeds:
        c = _train_one(spec, data, s)
        comps.append(c)
        log.info("    [%s seed=%d] model NDCG@5=%.4f Spear=%.4f | %s NDCG@5=%.4f",
                 spec.name, s, c["model"]["ndcg_at_5"], c["model"]["spearman"],
                 GATE, c[GATE]["ndcg_at_5"])
    return {"name": spec.name, "spec": spec, "per_seed": comps,
            "agg": {m: _agg(comps, m) for m in METHODS}, "n_seeds": len(model_seeds)}


def standardized_report(results: list[dict], *, out_dir: str | Path, tag: str) -> pd.DataFrame:
    """Tidy CSV (one row per arm×method, full metric schema) + a side-by-side log
    of each arm's model vs the chem-kNN gate. Returns the long DataFrame."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        for method in METHODS:
            rows.append({"arm": r["name"], "method": method, "n_seeds": r["n_seeds"],
                         **r["agg"][method]})
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{tag}_metrics.csv", index=False)

    log.info("STANDARDIZED REPORT [%s] — model vs chem-kNN gate (seed-mean):", tag)
    log.info("    %-16s %9s %8s %8s   %-14s", "arm", "Spearman", "NDCG@5", "prec@5", "vs gate NDCG@5")
    for r in results:
        m, g = r["agg"]["model"], r["agg"][GATE]
        d = m["ndcg_at_5"] - g["ndcg_at_5"]
        verdict = "BEATS gate" if d > 0 else f"{d:+.4f}"
        log.info("    %-16s %9.4f %8.4f %8.4f   %-14s",
                 r["name"], m["spearman"], m["ndcg_at_5"], m["precision_at_5"], verdict)
    log.info("    gate(chem-kNN) NDCG@5=%.4f Spearman=%.4f",
             results[0]["agg"][GATE]["ndcg_at_5"], results[0]["agg"][GATE]["spearman"])
    return df


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
