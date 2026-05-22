"""T5-C — ProteomeLM Bypass: ESM-C Direct (Hypothesis H-EMB-04).

Tests whether ProteomeLM's proteome-context layer helps at all, by
comparing against raw ESM-C 600M embeddings (no ProteomeLM pass).

ESM-C 600M outputs 1152-dim embeddings (same dim as ProteomeLM-L layer 8),
so the downstream model architecture is identical — only the gene
embedding source changes.

Arms:
  - plm_layer8: current locked choice (ProteomeLM-L layer 8 via ESM-C)
  - esmc_only:  raw mean-pooled ESM-C 600M, bypassing ProteomeLM entirely

If `esmc_only` matches or beats `plm_layer8`, the proteome-context pass is
not helping for this task — we should consider dropping it.
"""
from __future__ import annotations

from pathlib import Path

from src.experiments.tier3._t3_common import ResidualMLP
from src.experiments.tier5._t5_common import run_t5_experiment

ARM_PLM = "plm_layer8"
ARM_ESMC = "esmc_only"

ARM_DIRS = {
    ARM_PLM: (Path("data/processed/ProtLM_embeddings_layer8"), "_proteomelm.pt"),
    ARM_ESMC: (Path("data/processed/ESMC_embeddings"), "_esmc.pt"),
}


def _make_model(arm_name, gene_dim, chem_dim):
    return ResidualMLP(
        gene_dim=gene_dim, chem_dim=chem_dim,
        hidden_dim=512, n_blocks=1, dropout=0.1,
    )


def _per_arm_embedding_fn(arm_name):
    return ARM_DIRS[arm_name]


def run_t5c(cfg) -> dict:
    return run_t5_experiment(
        experiment_id="T5-C_esmc_bypass",
        hypothesis="H-EMB-04",
        title="T5-C ProteomeLM Bypass: ESM-C Direct (H-EMB-04)",
        arm_names=[ARM_PLM, ARM_ESMC],
        make_model_fn=_make_model,
        output_root="t5c",
        figures_dirname="tier5_c",
        per_arm_embedding_fn=_per_arm_embedding_fn,
    )
