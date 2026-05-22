"""T3-B — Width Sweep (Hypothesis H-CAP-02).

Tests whether wider hidden layers improve performance at the T3-A winning
depth. The 256-dim hidden layer compresses the 1577-dim input by ~84%. A
wider layer retains more capacity for gene×condition interaction patterns.

Arms (ordered narrow → wide):
  - width_128:  hidden_dim=128
  - width_256:  hidden_dim=256 (current default)
  - width_512:  hidden_dim=512
  - width_1024: hidden_dim=1024

Promotion rule: best performing arm on co-primary metrics (not parsimony).
The winning depth (n_blocks) is read from the T3-A summary at runtime.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.experiments.tier2._t2_common import ShallowMLP
from src.experiments.tier3._t3_common import ResidualMLP, run_t3_experiment

log = logging.getLogger(__name__)

ARM_128 = "width_128"
ARM_256 = "width_256"
ARM_512 = "width_512"
ARM_1024 = "width_1024"

WIDTH_MAP = {
    ARM_128: 128,
    ARM_256: 256,
    ARM_512: 512,
    ARM_1024: 1024,
}


def _read_t3a_winner() -> tuple[str, int]:
    summary_path = Path("artifacts/runs/t3a/t3a_summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(
            "T3-A summary not found. Run T3-A_depth first."
        )
    summary = json.loads(summary_path.read_text())
    winner = summary["comparison"]["parsimony_winner"]
    arm_to_blocks = {"1_layer": 0, "2_layer": 1, "4_layer": 3}
    n_blocks = arm_to_blocks[winner]
    return winner, n_blocks


def run_t3b(cfg) -> dict:
    winner_arm, n_blocks = _read_t3a_winner()
    log.info("T3-A winner: %s (n_blocks=%d). Sweeping width at this depth.",
             winner_arm, n_blocks)

    def _make_model(arm_name, gene_dim, chem_dim):
        hidden_dim = WIDTH_MAP[arm_name]
        if n_blocks == 0:
            return ShallowMLP(gene_dim=gene_dim, chem_dim=chem_dim,
                              hidden_dim=hidden_dim, dropout=0.1)
        return ResidualMLP(gene_dim=gene_dim, chem_dim=chem_dim,
                           hidden_dim=hidden_dim, n_blocks=n_blocks,
                           dropout=0.1)

    return run_t3_experiment(
        experiment_id="T3-B_width",
        hypothesis="H-CAP-02",
        title=f"T3-B Width Sweep at Winning Depth ({winner_arm}, n_blocks={n_blocks})",
        arm_names=[ARM_128, ARM_256, ARM_512, ARM_1024],
        make_model_fn=_make_model,
        output_root="t3b",
        figures_dirname="tier3_b",
    )
