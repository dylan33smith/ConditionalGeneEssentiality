"""tier1 handler — invoked by src/cli/run_experiment.py.

Dispatches to the right T1 experiment runner based on `cfg.experiment_id`.
"""
from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.experiments.tier1._t1a import run_t1a

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    exp_id = str(cfg.get("experiment_id", ""))
    if exp_id == "T1-A_granularity":
        run_t1a(cfg)
        return
    raise NotImplementedError(
        f"tier1 experiment_id={exp_id!r} not yet implemented. "
        f"Available: T1-A_granularity"
    )
