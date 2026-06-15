"""R-EVAL handler — locked-best ranking model vs chem-kNN, in one pinned command.

This is the regression check for the cleanup migration. It trains the locked
arm (pointwise_huber + multihot_425) at a fixed split seed and compares it to the
chem-kNN baseline on the identical eligible val gene set (denominator parity),
reusing the existing R-LOSS training + R1 eval harness (no behavior change).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from src.experiments.r1._r1_common import prepare_r1_data
from src.experiments.rloss._rloss_common import train_arm

log = logging.getLogger(__name__)
OUT = Path("artifacts/runs/reval")
BASELINE = Path("data_contract/ranking/reval_baseline.json")
LOCKED_LOSS = "pointwise_huber"          # R-LOSS-DEC-001 carried-forward arm
GATE_METRICS = ("ndcg_at_5", "spearman")  # the metrics the regression gate checks
TOL = 0.003                               # abs tolerance (GPU jitter + seed noise)


def _agg(rows: list[dict], key: str) -> dict:
    """Mean over seeds for one method's metric dict."""
    keys = ("spearman", "kendall", "ndcg_at_1", "ndcg_at_3", "ndcg_at_5",
            "precision_at_5", "n_genes")
    return {k: float(np.mean([r[key][k] for r in rows])) for k in keys}


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

    log.info("[1/3] preparing data (split seed=%d)", split_seed)
    data = prepare_r1_data(orgs, seed=split_seed)
    log.info("    train=%d val=%d eligible val genes=%d",
             len(data.train), len(data.val), len(data.eligible_val_genes))

    log.info("[2/3] training '%s' x %d seed(s)", LOCKED_LOSS, len(model_seeds))
    comps = []
    for s in model_seeds:
        res = train_arm(LOCKED_LOSS, data, seed=s, epochs=epochs)
        comps.append(res["comparison"])
        c = res["comparison"]
        log.info("    seed=%d  model NDCG@5=%.4f Spearman=%.4f | kNN NDCG@5=%.4f Spearman=%.4f",
                 s, c["model"]["ndcg_at_5"], c["model"]["spearman"],
                 c["chem_knn"]["ndcg_at_5"], c["chem_knn"]["spearman"])

    model_m = _agg(comps, "model")
    knn_m = _agg(comps, "chem_knn")
    null_m = _agg(comps, "chem_null")
    mf_m = _agg(comps, "linear_mf")

    summary = {
        "tag": tag, "orgs": orgs, "split_seed": split_seed, "model_seeds": model_seeds,
        "epochs": epochs, "locked_loss": LOCKED_LOSS,
        "model": model_m, "chem_knn": knn_m, "chem_null": null_m, "linear_mf": mf_m,
    }
    (OUT / f"reval_{tag}.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([
        {"method": m, **summary[m]} for m in ("model", "chem_knn", "linear_mf", "chem_null")
    ]).to_csv(OUT / f"reval_{tag}.csv", index=False)

    # ---- side-by-side ----
    log.info("[3/3] RESULT (mean over %d seed(s), %d common eligible val genes):",
             len(model_seeds), int(model_m["n_genes"]))
    hdr = f"    {'method':<12} {'Spearman':>9} {'NDCG@1':>8} {'NDCG@5':>8} {'prec@5':>8}"
    log.info(hdr)
    for name, m in (("model", model_m), ("chem_knn", knn_m),
                    ("linear_mf", mf_m), ("chem_null", null_m)):
        log.info("    %-12s %9.4f %8.4f %8.4f %8.4f",
                 name, m["spearman"], m["ndcg_at_1"], m["ndcg_at_5"], m["precision_at_5"])
    log.info("    gap (kNN - model) NDCG@5 = %.4f", knn_m["ndcg_at_5"] - model_m["ndcg_at_5"])

    # ---- regression gate vs baseline ----
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
