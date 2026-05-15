"""tier3 handler — invoked by src/cli/run_experiment.py.

Dispatches to the right T3 experiment runner based on `cfg.experiment_id`.
"""
from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.experiments.tier3._t3a import run_t3a
from src.experiments.tier3._t3b import run_t3b
from src.experiments.tier3._t3d import run_t3d

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    exp_id = str(cfg.get("experiment_id", ""))
    if exp_id == "T3-A_depth":
        run_t3a(cfg)
        return
    if exp_id == "T3-B_width":
        run_t3b(cfg)
        return
    if exp_id == "T3-D_film_at_depth":
        run_t3d(cfg)
        return
    raise NotImplementedError(
        f"tier3 experiment_id={exp_id!r} not yet implemented. "
        f"Available: T3-A_depth, T3-B_width, T3-D_film_at_depth"
    )
