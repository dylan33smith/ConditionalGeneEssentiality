"""T3-D — FiLM at Winning Depth (Hypothesis H-CAP-03).

Re-tests FiLM gating at the T3-A winning depth. T2-C showed FiLM was
sub-threshold at 1 layer (RMSE 0.003, MAE 0.004 — disjoint CIs but below
bar). This experiment tests whether FiLM's advantage amplifies with the
capacity found optimal in T3-A.

Arms:
  - concat:     early-concat MLP at T3-A winning depth (locked fusion)
  - film_gated: FiLM modulation + MLP at T3-A winning depth

The winning depth (n_blocks) is read from the T3-A summary at runtime.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch.nn as nn

from src.experiments.tier2._t2_common import ShallowMLP, FiLMFusion
from src.experiments.tier3._t3_common import ResidualMLP, run_t3_experiment

log = logging.getLogger(__name__)

ARM_CONCAT = "concat"
ARM_FILM = "film_gated"


class FiLMResidualMLP(nn.Module):
    """FiLM modulation followed by a residual MLP head."""

    def __init__(self, *, gene_dim: int, chem_dim: int,
                 hidden_dim: int = 256, n_blocks: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.film_gen = nn.Linear(chem_dim, gene_dim * 2)
        from src.experiments.tier3._t3_common import ResidualBlock
        self.proj = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, gene_emb, chem):
        import torch
        film_params = self.film_gen(chem.float())
        gamma, beta = film_params.chunk(2, dim=1)
        modulated = gene_emb * (1.0 + gamma) + beta
        x = self.proj(modulated)
        x = self.blocks(x)
        return self.out(x).squeeze(1)


def _read_t3a_winner() -> tuple[str, int]:
    """Read T3-A results and return (winner_arm, n_blocks)."""
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


def run_t3d(cfg) -> dict:
    winner_arm, n_blocks = _read_t3a_winner()
    log.info("T3-A winner: %s (n_blocks=%d). Testing FiLM at this depth.",
             winner_arm, n_blocks)

    def _make_model(arm_name, gene_dim, chem_dim):
        if arm_name == ARM_CONCAT:
            if n_blocks == 0:
                return ShallowMLP(gene_dim=gene_dim, chem_dim=chem_dim,
                                  hidden_dim=256, dropout=0.1)
            return ResidualMLP(gene_dim=gene_dim, chem_dim=chem_dim,
                               hidden_dim=256, n_blocks=n_blocks, dropout=0.1)
        if n_blocks == 0:
            return FiLMFusion(gene_dim=gene_dim, chem_dim=chem_dim,
                              hidden_dim=256, dropout=0.1)
        return FiLMResidualMLP(gene_dim=gene_dim, chem_dim=chem_dim,
                               hidden_dim=256, n_blocks=n_blocks, dropout=0.1)

    return run_t3_experiment(
        experiment_id="T3-D_film_at_depth",
        hypothesis="H-CAP-03",
        title=f"T3-D FiLM at Winning Depth ({winner_arm}, n_blocks={n_blocks})",
        arm_names=[ARM_CONCAT, ARM_FILM],
        make_model_fn=_make_model,
        output_root="t3d",
        figures_dirname="tier3_d",
    )
