"""R-LOSS handler — loss-family retest under ranking (now a thin runner spec).

Declares one arm per loss (multihot encoder + T5-A architecture held constant)
and hands them to the shared runner, which trains each across seeds and emits the
standardized side-by-side vs the chem-kNN gate. See R-LOSS-DEC-001 for the result.
"""
from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.ranking.runner import ArmSpec, run_experiment

log = logging.getLogger(__name__)
DEFAULT_LOSSES = ["pointwise_mse", "pointwise_huber", "pairwise_ranknet",
                  "lambdarank", "listmle", "approxndcg"]


def main(cfg: DictConfig) -> None:
    exp = cfg.get("experiment", {})
    losses = [str(a["loss"]) for a in exp.get("arms", [])] or DEFAULT_LOSSES
    seeds = list(exp.get("seeds", [0]))
    orgs = exp.get("orgs", None)
    orgs = list(orgs) if orgs is not None else None
    epochs = int(exp.get("epochs", 15))

    log.info("R-LOSS — loss family under ranking | losses=%s seeds=%s orgs=%s",
             losses, seeds, orgs if orgs else "ALL")
    specs = [ArmSpec(name=ls, loss=ls, epochs=epochs) for ls in losses]
    run_experiment(specs, orgs=orgs, model_seeds=seeds,
                   out_dir="artifacts/runs/rloss", tag="rloss")
    log.info("R-LOSS done — see artifacts/runs/rloss/rloss_metrics.csv")
