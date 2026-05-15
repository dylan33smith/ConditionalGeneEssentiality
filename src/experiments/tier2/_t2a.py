"""T2-A — Linear vs Shallow MLP Head (Hypothesis H-FUSE-01).

Tests whether the nonlinearity in the fusion head matters.
  - linear_head: cat(gene, chem) → Linear(1577, 1)
  - shallow_mlp:  cat(gene, chem) → Linear(1577, 256) → ReLU → Dropout → Linear(256, 1)
    (identical to the T1 vehicle)
"""
from __future__ import annotations

from src.experiments.tier2._t2_common import (
    LinearHead,
    ShallowMLP,
    run_t2_experiment,
)

ARM_LINEAR = "linear_head"
ARM_MLP = "shallow_mlp"


def _make_model(arm_name, gene_dim, chem_dim):
    if arm_name == ARM_LINEAR:
        return LinearHead(gene_dim=gene_dim, chem_dim=chem_dim)
    return ShallowMLP(gene_dim=gene_dim, chem_dim=chem_dim,
                      hidden_dim=256, dropout=0.1)


def run_t2a(cfg) -> dict:
    return run_t2_experiment(
        experiment_id="T2-A_linear_vs_mlp",
        hypothesis="H-FUSE-01",
        title="T2-A Linear vs Shallow MLP Head (H-FUSE-01)",
        arm_names=[ARM_LINEAR, ARM_MLP],
        make_model_fn=_make_model,
        output_root="t2a",
        figures_dirname="tier2_a",
    )
