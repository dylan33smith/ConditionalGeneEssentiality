"""T2-B — Early Concat vs Two-Tower Merge (Hypothesis H-FUSE-02).

Tests whether separate encoders for gene and condition with late merge
beats early concatenation.
  - early_concat: cat(gene, chem) → MLP → 1  (T1 vehicle)
  - two_tower:    gene → MLP_g(128) ; chem → MLP_c(128) ; cat → Linear → 1

Tower dim is 128 each (256 total) to match the T1 vehicle's hidden_dim=256
in total parameter budget.
"""
from __future__ import annotations

from src.experiments.tier2._t2_common import (
    ShallowMLP,
    TwoTower,
    run_t2_experiment,
)

ARM_CONCAT = "early_concat"
ARM_TOWER = "two_tower"


def _make_model(arm_name, gene_dim, chem_dim):
    if arm_name == ARM_CONCAT:
        return ShallowMLP(gene_dim=gene_dim, chem_dim=chem_dim,
                          hidden_dim=256, dropout=0.1)
    return TwoTower(gene_dim=gene_dim, chem_dim=chem_dim,
                    tower_dim=128, dropout=0.1)


def run_t2b(cfg) -> dict:
    return run_t2_experiment(
        experiment_id="T2-B_early_vs_late",
        hypothesis="H-FUSE-02",
        title="T2-B Early Concat vs Two-Tower (H-FUSE-02)",
        arm_names=[ARM_CONCAT, ARM_TOWER],
        make_model_fn=_make_model,
        output_root="t2b",
        figures_dirname="tier2_b",
    )
