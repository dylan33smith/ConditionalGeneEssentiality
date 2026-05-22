"""tier5 handler — invoked by src/cli/run_experiment.py.

Dispatches to the right T5 experiment runner based on `cfg.experiment_id`.
"""
from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.experiments.tier5._t5a import run_t5a
from src.experiments.tier5._t5b import run_t5b
from src.experiments.tier5._t5c import run_t5c
from src.experiments.tier5._t5d import run_t5d

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    exp_id = str(cfg.get("experiment_id", ""))
    if exp_id == "T5-A_gene_adapter":
        run_t5a(cfg)
        return
    if exp_id == "T5-B_layer_ablation":
        run_t5b(cfg)
        return
    if exp_id == "T5-C_esmc_bypass":
        run_t5c(cfg)
        return
    if exp_id == "T5-D_adapter_variants":
        run_t5d(cfg)
        return
    raise NotImplementedError(
        f"tier5 experiment_id={exp_id!r} not yet implemented. "
        f"Available: T5-A_gene_adapter, T5-B_layer_ablation, T5-C_esmc_bypass, "
        f"T5-D_adapter_variants"
    )
