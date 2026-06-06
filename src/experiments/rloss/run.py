"""R-LOSS handler — loss-family retest under ranking.

Trains the T5-A architecture (multihot encoder held constant) under each loss
and reports the R-LOCK-4 side-by-side vs chem-kNN gate + linear-MF + chem-NULL.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from src.experiments.r1._r1_common import prepare_r1_data
from src.experiments.rloss._rloss_common import train_arm

log = logging.getLogger(__name__)
OUT = Path("artifacts/runs/rloss")


def main(cfg: DictConfig) -> None:
    exp = cfg.get("experiment", {})
    losses = [str(a["loss"]) for a in exp.get("arms", [{"loss": "pointwise_mse"}])]
    seeds = list(exp.get("seeds", [0]))
    orgs = exp.get("orgs", None)
    orgs = list(orgs) if orgs is not None else None
    epochs = int(exp.get("epochs", 8))
    OUT.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("R-LOSS — loss family under ranking | losses=%s seeds=%s orgs=%s",
             losses, seeds, orgs if orgs else "ALL")
    log.info("=" * 60)

    log.info("[1/3] preparing data (split + eligibility + features + MF baseline)")
    data = prepare_r1_data(orgs, seed=seeds[0])
    log.info("    train rows=%d, val pooled=%d, eligible val genes=%d",
             len(data.train), len(data.val), len(data.eligible_val_genes))

    log.info("[2/3] training %d loss(es) × %d seed(s)", len(losses), len(seeds))
    results, comparisons = [], []
    for loss in losses:
        for seed in seeds:
            log.info("──── loss=%s seed=%d ────", loss, seed)
            res = train_arm(loss, data, seed=seed, epochs=epochs)
            res["flat"]["loss"] = loss
            res["per_org"].to_csv(OUT / f"per_org_{loss}_s{seed}.csv", index=False)
            results.append(res["flat"])
            comp = res["comparison"]
            for method in ("model", "chem_knn", "linear_mf", "chem_null"):
                m = comp[method]
                comparisons.append({"loss": loss, "seed": seed, "method": method,
                                    "spearman": m["spearman"], "kendall": m["kendall"],
                                    "ndcg_at_1": m["ndcg_at_1"], "ndcg_at_3": m["ndcg_at_3"],
                                    "ndcg_at_5": m["ndcg_at_5"],
                                    "precision_at_5": m["precision_at_5"],
                                    "n_genes": m["n_genes"]})
            cmp_df = pd.DataFrame([c for c in comparisons
                                   if c["loss"] == loss and c["seed"] == seed])
            log.info("    SIDE-BY-SIDE (loss=%s seed=%d, %d genes):\n%s",
                     loss, seed, comp["model"]["n_genes"],
                     cmp_df[["method", "spearman", "ndcg_at_1", "ndcg_at_5",
                             "precision_at_5"]].to_string(index=False))
            pd.DataFrame(results).to_csv(OUT / "rloss_results.csv", index=False)
            pd.DataFrame(comparisons).to_csv(OUT / "rloss_metric_comparison.csv", index=False)

    df = pd.DataFrame(results)
    log.info("[3/3] DONE. headline (model vs chem-kNN gate NDCG@5 0.485):\n%s",
             df[["loss", "seed", "model_spearman", "model_ndcg_at_5",
                 "beats_knn_ndcg5"]].to_string(index=False))
    log.info("artifacts in %s", OUT)
