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
    # Ranking regime (R) — the active objective. The T-regime (S0-S5, T1-T6) was
    # pruned in the ranking-branch cleanup; its results live in the decision
    # ledger + SCIENTIFIC_SYNTHESIS (see the pruned→learning index in the docs).
    "R0": "src.experiments.r0.run:main",
    "R1": "src.experiments.r1.run:main",
    "R-LOSS": "src.experiments.rloss.run:main",
    "R-CONF": "src.experiments.rconf.run:main",
    "R-EVAL": "src.experiments.reval.run:main",
    "R-AUG": "src.experiments.raug.run:main",
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
