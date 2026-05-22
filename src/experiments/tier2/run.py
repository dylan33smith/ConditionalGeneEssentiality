"""tier2 handler — invoked by src/cli/run_experiment.py.

Dispatches to the right T2 experiment runner based on `cfg.experiment_id`.
"""
from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.experiments.tier2._t2a import run_t2a
from src.experiments.tier2._t2b import run_t2b
from src.experiments.tier2._t2c import run_t2c

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    exp_id = str(cfg.get("experiment_id", ""))
    if exp_id == "T2-A_linear_vs_mlp":
        run_t2a(cfg)
        return
    if exp_id == "T2-B_early_vs_late":
        run_t2b(cfg)
        return
    if exp_id == "T2-C_gating":
        run_t2c(cfg)
        return
    raise NotImplementedError(
        f"tier2 experiment_id={exp_id!r} not yet implemented. "
        f"Available: T2-A_linear_vs_mlp, T2-B_early_vs_late, T2-C_gating"
    )
