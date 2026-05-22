"""tier6 handler — invoked by src/cli/run_experiment.py.

Dispatches to the right T6 experiment runner based on `cfg.experiment_id`.
"""
from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.experiments.tier6._t6a import run_t6a

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    exp_id = str(cfg.get("experiment_id", ""))
    if exp_id == "T6-A_fingerprints":
        run_t6a(cfg)
        return
    raise NotImplementedError(
        f"tier6 experiment_id={exp_id!r} not yet implemented. "
        f"Available: T6-A_fingerprints"
    )
