"""T2-C — Condition-Gated Fusion / FiLM (Hypothesis H-FUSE-03).

Tests whether condition-gated gene features (FiLM) improve over un-gated
early concat.
  - ungated:    cat(gene, chem) → MLP → 1  (T1 vehicle)
  - film_gated: chem → (gamma, beta); gene * (1 + gamma) + beta → MLP → 1

FiLM allows the condition vector to modulate the gene embedding via
learned affine transform before the prediction head, enabling
multiplicative gene×condition interactions that additive concat cannot
express.
"""
from __future__ import annotations

from src.experiments.tier2._t2_common import (
    ShallowMLP,
    FiLMFusion,
    run_t2_experiment,
)

ARM_UNGATED = "ungated"
ARM_FILM = "film_gated"


def _make_model(arm_name, gene_dim, chem_dim):
    if arm_name == ARM_UNGATED:
        return ShallowMLP(gene_dim=gene_dim, chem_dim=chem_dim,
                          hidden_dim=256, dropout=0.1)
    return FiLMFusion(gene_dim=gene_dim, chem_dim=chem_dim,
                      hidden_dim=256, dropout=0.1)


def run_t2c(cfg) -> dict:
    return run_t2_experiment(
        experiment_id="T2-C_gating",
        hypothesis="H-FUSE-03",
        title="T2-C Un-gated vs FiLM Gating (H-FUSE-03)",
        arm_names=[ARM_UNGATED, ARM_FILM],
        make_model_fn=_make_model,
        output_root="t2c",
        figures_dirname="tier2_c",
    )
