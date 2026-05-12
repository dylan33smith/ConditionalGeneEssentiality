"""tier1 handler — invoked by src/cli/run_experiment.py.

Dispatches to the right T1 experiment runner based on `cfg.experiment_id`.
"""
from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.experiments.tier1._t1a import run_t1a

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    from pathlib import Path

    exp_id = str(cfg.get("experiment_id", ""))
    if exp_id == "T1-A_granularity":
        run_t1a(cfg)
        return
    if exp_id == "T1-A.2_stress_test_largest_by_rows":
        protocol_path = Path(str(cfg.experiment.protocol_path))
        run_t1a(
            cfg,
            protocol_path=protocol_path,
            output_root="t1a2",
            figures_dirname="tier1_a2",
            experiment_id_label=exp_id,
            title="T1-A.2 Stress Test (H-ENC-01 on largest_by_rows)",
        )
        return
    raise NotImplementedError(
        f"tier1 experiment_id={exp_id!r} not yet implemented. "
        f"Available: T1-A_granularity, T1-A.2_stress_test_largest_by_rows"
    )
