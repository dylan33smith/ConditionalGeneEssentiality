"""R-EVAL handler — locked-best ranking model vs chem-kNN, in one pinned command.

The regression check for the cleanup migration: trains the locked arm
(pointwise_huber + multihot_425) at a fixed split seed via the shared runner and
compares to the chem-kNN baseline on the identical eligible val gene set, then
gates the result against a stored baseline (PASS/FAIL within tolerance).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig

from src.ranking.pipeline import prepare_r1_data, prepare_leave_compound_out_data
from src.ranking.runner import ArmSpec, run_arm, METHODS

log = logging.getLogger(__name__)
OUT = Path("artifacts/runs/reval")
BASELINE = Path("data_contract/ranking/reval_baseline.json")
LOCKED_LOSS = "pointwise_huber"          # R-LOSS-DEC-001 carried-forward arm
GATE_METRICS = ("ndcg_at_5", "spearman")  # the metrics the regression gate checks
TOL = 0.003                               # abs tolerance (GPU jitter + seed noise)


def main(cfg: DictConfig) -> None:
    exp = cfg.get("experiment", {})
    orgs = exp.get("orgs", ["Keio", "Caulo", "MR1"])
    orgs = list(orgs) if orgs is not None else None
    split_seed = int(exp.get("split_seed", 0))
    model_seeds = list(exp.get("model_seeds", [0]))
    epochs = int(exp.get("epochs", 8))
    tag = str(exp.get("tag", "fast"))
    OUT.mkdir(parents=True, exist_ok=True)

    # determinism aids (computation-preserving refactors should reproduce closely)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log.info("=" * 64)
    log.info("R-EVAL [%s] — locked '%s' vs chem-kNN | orgs=%s split_seed=%d model_seeds=%s",
             tag, LOCKED_LOSS, orgs if orgs else "ALL", split_seed, model_seeds)
    log.info("=" * 64)

    # `experiment.split` selects the split protocol. Default reproduces the primary
    # condition-holdout bit-for-bit, so the regression gate is unaffected.
    split_name = str(exp.get("split", "condition_holdout"))
    if split_name == "leave_compound_out":
        frac = float(exp.get("compound_holdout_fraction", 0.20))
        log.info("    split=leave_compound_out (fraction=%.2f) — stressor compounds "
                 "held out GLOBALLY; no train row anywhere contains them", frac)
        data = prepare_leave_compound_out_data(orgs, seed=split_seed, fraction=frac)
    elif split_name == "condition_holdout":
        data = prepare_r1_data(orgs, seed=split_seed)
    else:
        raise ValueError(
            f"unknown experiment.split={split_name!r}; "
            "expected 'condition_holdout' or 'leave_compound_out'")
    log.info("    train=%d val=%d eligible val genes=%d",
             len(data.train), len(data.val), len(data.eligible_val_genes))

    res = run_arm(ArmSpec("locked", loss=LOCKED_LOSS, epochs=epochs),
                  data, model_seeds=model_seeds)
    agg = res["agg"]

    summary = {"tag": tag, "orgs": orgs, "split_seed": split_seed,
               "model_seeds": model_seeds, "epochs": epochs, "locked_loss": LOCKED_LOSS,
               **{m: agg[m] for m in METHODS}}
    (OUT / f"reval_{tag}.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([{"method": m, **agg[m]} for m in METHODS]).to_csv(
        OUT / f"reval_{tag}.csv", index=False)

    log.info("RESULT (mean over %d seed(s), %d common eligible val genes):",
             len(model_seeds), int(agg["model"]["n_genes"]))
    # NDCG@5 PRIMARY (leads), then NDCG@1, then Spearman (secondary completeness).
    log.info("    %-12s %8s %8s %9s", "method", "NDCG@5", "NDCG@1", "Spearman")
    for m in METHODS:
        log.info("    %-12s %8.4f %8.4f %9.4f",
                 m, agg[m]["ndcg_at_5"], agg[m]["ndcg_at_1"], agg[m]["spearman"])
    log.info("    gap (kNN - model) NDCG@5 = %.4f",
             agg["chem_knn"]["ndcg_at_5"] - agg["model"]["ndcg_at_5"])

    _gate(summary, tag)


def _gate(summary: dict, tag: str) -> None:
    if not BASELINE.exists():
        log.info("    [gate] no baseline at %s — writing this run AS the baseline.", BASELINE)
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps({tag: summary}, indent=2))
        return
    base = json.loads(BASELINE.read_text())
    if tag not in base:
        log.info("    [gate] baseline has no '%s' entry — adding it.", tag)
        base[tag] = summary; BASELINE.write_text(json.dumps(base, indent=2)); return

    b = base[tag]; ok = True
    log.info("    [gate] comparing vs baseline '%s' (tol=%.3f):", tag, TOL)
    for method in ("model", "chem_knn"):
        for metric in GATE_METRICS:
            cur = summary[method][metric]; ref = b[method][metric]
            d = abs(cur - ref); flag = "OK " if d <= TOL else "DRIFT"
            if d > TOL:
                ok = False
            log.info("      %-9s %-9s cur=%.4f base=%.4f Δ=%+.4f  %s",
                     method, metric, cur, ref, cur - ref, flag)
    if ok:
        log.info("    [gate] PASS — numbers reproduce within tolerance.")
    else:
        log.error("    [gate] FAIL — a number moved beyond tolerance. STOP and investigate.")
        raise SystemExit(2)
