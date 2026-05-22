"""tier4 handler — invoked by src/cli/run_experiment.py.

Dispatches to the right T4 experiment runner based on `cfg.experiment_id`.
"""
from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.experiments.tier4._t4a import run_t4a
from src.experiments.tier4._t4b import run_t4b
from src.experiments.tier4._t4c import run_t4c

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    exp_id = str(cfg.get("experiment_id", ""))
    if exp_id == "T4-A_loss_family":
        run_t4a(cfg)
        return
    if exp_id == "T4-B_target_norm":
        run_t4b(cfg)
        return
    if exp_id == "T4-C_hparams":
        run_t4c(cfg)
        return
    raise NotImplementedError(
        f"tier4 experiment_id={exp_id!r} not yet implemented. "
        f"Available: T4-A_loss_family, T4-B_target_norm, T4-C_hparams"
    )
