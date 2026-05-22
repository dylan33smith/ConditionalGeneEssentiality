"""T3-A — Depth Ablation (Hypothesis H-CAP-01).

Tests whether additional hidden layers with residual connections improve over
the locked 1-layer shallow MLP.
  - 1_layer:  cat(gene, chem) → Linear(1577, 256) → ReLU → Dropout → Linear(256, 1)
  - 2_layer:  cat → proj(256) → ResBlock(256)×1 → Linear(256, 1)
  - 4_layer:  cat → proj(256) → ResBlock(256)×3 → Linear(256, 1)

Arms are ordered simplest-first so the parsimony rule picks the smallest model
within tolerance of the best.
"""
from __future__ import annotations

from src.experiments.tier2._t2_common import ShallowMLP
from src.experiments.tier3._t3_common import ResidualMLP, run_t3_experiment

ARM_1L = "1_layer"
ARM_2L = "2_layer"
ARM_4L = "4_layer"


def _make_model(arm_name, gene_dim, chem_dim):
    if arm_name == ARM_1L:
        return ShallowMLP(gene_dim=gene_dim, chem_dim=chem_dim,
                          hidden_dim=256, dropout=0.1)
    if arm_name == ARM_2L:
        return ResidualMLP(gene_dim=gene_dim, chem_dim=chem_dim,
                           hidden_dim=256, n_blocks=1, dropout=0.1)
    return ResidualMLP(gene_dim=gene_dim, chem_dim=chem_dim,
                       hidden_dim=256, n_blocks=3, dropout=0.1)


def run_t3a(cfg) -> dict:
    return run_t3_experiment(
        experiment_id="T3-A_depth",
        hypothesis="H-CAP-01",
        title="T3-A Depth Ablation: 1-layer vs 2-layer vs 4-layer (H-CAP-01)",
        arm_names=[ARM_1L, ARM_2L, ARM_4L],
        make_model_fn=_make_model,
        output_root="t3a",
        figures_dirname="tier3_a",
    )
