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

from src.experiments.r1._r1_common import prepare_r1_data, train_r1_arm

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
    for arm in arms:
        for seed in seeds:
            log.info("──── arm=%s seed=%d ────", arm, seed)
            res = train_r1_arm(arm, data, seed=seed, epochs=epochs)
            per_org = res.pop("per_org")
            per_org.to_csv(OUT / f"per_org_{arm}_s{seed}.csv", index=False)
            results.append(res)
            log.info("    arm=%s seed=%d  model Spearman=%.4f [%.4f,%.4f]  "
                     "NDCG@5=%.4f  | chem-null=%.4f  chem-kNN=%.4f",
                     arm, seed, res["model_spearman"], res["model_spearman_ci"][0],
                     res["model_spearman_ci"][1], res["model_ndcg_at_5"],
                     res["baseline_chem_null_spearman"], res["baseline_chem_knn_spearman"])

    df = pd.DataFrame(results)
    df.to_csv(OUT / "r1_results.csv", index=False)
    log.info("[3/3] DONE. results:\n%s",
             df[["arm", "seed", "model_spearman", "model_ndcg_at_5",
                 "baseline_chem_knn_spearman"]].to_string(index=False))
    log.info("artifacts in %s", OUT)
