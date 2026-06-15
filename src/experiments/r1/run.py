"""R1 handler — invoked by src/cli/run_experiment.py when stage_or_tier == 'R1'.

Trains the chemistry arms under the ranking regime and reports the R-LOCK-4
metric bundle (within-gene Spearman/Kendall/NDCG vs chem-null + chem-kNN
baselines), per organism.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.ranking.pipeline import prepare_r1_data, train_r1_arm

log = logging.getLogger(__name__)

OUT = Path("artifacts/runs/r1")


def main(cfg: DictConfig) -> None:
    exp = cfg.get("experiment", {})
    arms = [str(a["chemistry"]) for a in exp.get("arms", [{"chemistry": "multihot_425"}])]
    seeds = list(exp.get("seeds", [0]))
    orgs = exp.get("orgs", None)
    orgs = list(orgs) if orgs is not None else None
    epochs = int(exp.get("epochs", 8))
    OUT.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("R1 — chemistry retest under ranking | arms=%s seeds=%s orgs=%s",
             arms, seeds, orgs if orgs else "ALL")
    log.info("=" * 60)

    log.info("[1/3] preparing data (split + eligibility + features)")
    data = prepare_r1_data(orgs, seed=seeds[0])
    log.info("    train rows=%d, val pooled=%d, eligible val genes=%d",
             len(data.train), len(data.val), len(data.eligible_val_genes))

    log.info("[2/3] training %d arm(s) × %d seed(s)", len(arms), len(seeds))
    results = []
    comparisons = []
    for arm in arms:
        for seed in seeds:
            log.info("──── arm=%s seed=%d ────", arm, seed)
            res = train_r1_arm(arm, data, seed=seed, epochs=epochs)
            res["per_org"].to_csv(OUT / f"per_org_{arm}_s{seed}.csv", index=False)
            results.append(res["flat"])

            # side-by-side: model vs chem-kNN vs chem-NULL on the SAME genes
            comp = res["comparison"]
            for method in ("model", "chem_knn", "linear_mf", "chem_null"):
                m = comp[method]
                comparisons.append({"arm": arm, "seed": seed, "method": method,
                                    "spearman": m["spearman"], "kendall": m["kendall"],
                                    "ndcg_at_1": m["ndcg_at_1"], "ndcg_at_3": m["ndcg_at_3"],
                                    "ndcg_at_5": m["ndcg_at_5"],
                                    "precision_at_1": m["precision_at_1"],
                                    "precision_at_3": m["precision_at_3"],
                                    "precision_at_5": m["precision_at_5"],
                                    "n_genes": m["n_genes"]})
            cmp_df = pd.DataFrame([c for c in comparisons
                                   if c["arm"] == arm and c["seed"] == seed])
            log.info("    SIDE-BY-SIDE (arm=%s seed=%d, same %d genes):\n%s",
                     arm, seed, comp["model"]["n_genes"],
                     cmp_df[["method", "spearman", "ndcg_at_1", "ndcg_at_3",
                             "ndcg_at_5", "precision_at_5"]].to_string(index=False))
            # write incrementally so partial sweep results are available
            pd.DataFrame(results).to_csv(OUT / "r1_results.csv", index=False)
            pd.DataFrame(comparisons).to_csv(OUT / "r1_metric_comparison.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv(OUT / "r1_results.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "r1_metric_comparison.csv", index=False)
    log.info("[3/3] DONE. headline:\n%s",
             df[["arm", "seed", "model_spearman", "model_ndcg_at_5",
                 "knn_ndcg_at_5", "beats_knn_spearman", "beats_knn_ndcg5"]].to_string(index=False))
    log.info("artifacts in %s (r1_results.csv + r1_metric_comparison.csv)", OUT)
