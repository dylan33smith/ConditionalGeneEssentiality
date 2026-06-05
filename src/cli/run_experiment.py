"""Hydra entrypoint for running stages and tier experiments.

Usage:
    # Run a stage:
    python -m src.cli.run_experiment +stage=s0_reproducibility

    # Run a tier experiment:
    python -m src.cli.run_experiment +experiment=T1-A_granularity

    # Override at the CLI:
    python -m src.cli.run_experiment +experiment=T1-A_granularity train.seed=1

    # Multirun (Hydra sweep) over seeds:
    python -m src.cli.run_experiment +experiment=T1-A_granularity train.seed=0,1,2 -m

The runner dispatches to the right stage or tier handler based on
`cfg.stage_or_tier`. Each handler lives in `src.experiments.<stage_or_tier>.run`.
"""
from __future__ import annotations
import logging

import hydra
from omegaconf import DictConfig, OmegaConf


log = logging.getLogger(__name__)


_HANDLERS = {
    "S0": "src.experiments.stage0.run:main",
    "S1": "src.experiments.stage1.run:main",
    "S2": "src.experiments.stage2.run:main",
    "S3": "src.experiments.stage3.run:main",
    "S4": "src.experiments.stage4.run:main",
    "S5": "src.experiments.stage5.run:main",
    "T1": "src.experiments.tier1.run:main",
    "T2": "src.experiments.tier2.run:main",
    "T3": "src.experiments.tier3.run:main",
    "T4": "src.experiments.tier4.run:main",
    "T5": "src.experiments.tier5.run:main",
    "T6": "src.experiments.tier6.run:main",
    "R0": "src.experiments.r0.run:main",
    "R1": "src.experiments.r1.run:main",
}


def _resolve_handler(spec: str):
    module_path, attr = spec.split(":")
    module = __import__(module_path, fromlist=[attr])
    return getattr(module, attr)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Top-level dispatch."""
    stage_or_tier = cfg.get("stage_or_tier", "exploratory")
    log.info("Running %s (experiment_id=%s)", stage_or_tier, cfg.get("experiment_id"))
    log.debug("Config:\n%s", OmegaConf.to_yaml(cfg))

    if stage_or_tier == "exploratory":
        log.warning(
            "stage_or_tier=exploratory; this run is ineligible for promotion. "
            "Add +stage=... or +experiment=... to declare a real run."
        )
        return

    if stage_or_tier not in _HANDLERS:
        raise ValueError(
            f"Unknown stage_or_tier={stage_or_tier!r}. "
            f"Valid values: {sorted(_HANDLERS)}"
        )

    try:
        handler = _resolve_handler(_HANDLERS[stage_or_tier])
    except (ImportError, AttributeError) as e:
        raise NotImplementedError(
            f"Handler for {stage_or_tier} not yet implemented: {e}. "
            f"Expected entrypoint: {_HANDLERS[stage_or_tier]}"
        )

    handler(cfg)


if __name__ == "__main__":
    main()
