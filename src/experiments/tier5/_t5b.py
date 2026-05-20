"""T5-B — ProteomeLM-L Layer Ablation (Hypothesis H-EMB-03).

Tests whether the choice of ProteomeLM-L hidden layer (currently layer 8)
is optimal for downstream conditional essentiality prediction. ProteomeLM-L
has 18 transformer layers; different layers capture different abstractions.

Arms (each uses the locked T3 architecture; only gene-embedding source varies):
  - layer_0:  raw input embedding (ESM-C passed through projection only)
  - layer_4:  early ProteomeLM representation
  - layer_8:  current locked choice
  - layer_12: late-middle ProteomeLM representation
  - layer_18: final ProteomeLM layer

Promotion rule: best arm on co-primary metrics.
Encoded embeddings produced by src/data/encode_proteomelm_layers.py.
"""
from __future__ import annotations

from pathlib import Path

from src.experiments.tier3._t3_common import ResidualMLP
from src.experiments.tier5._t5_common import run_t5_experiment

LAYER_DIRS = {
    "layer_0": (Path("data/processed/PLM_embeddings_layer0"), "_proteomelm.pt"),
    "layer_4": (Path("data/processed/PLM_embeddings_layer4"), "_proteomelm.pt"),
    "layer_8": (Path("data/processed/ProtLM_embeddings_layer8"), "_proteomelm.pt"),
    "layer_12": (Path("data/processed/PLM_embeddings_layer12"), "_proteomelm.pt"),
    "layer_18": (Path("data/processed/PLM_embeddings_layer18"), "_proteomelm.pt"),
}


def _make_model(arm_name, gene_dim, chem_dim):
    return ResidualMLP(
        gene_dim=gene_dim, chem_dim=chem_dim,
        hidden_dim=512, n_blocks=1, dropout=0.1,
    )


def _per_arm_embedding_fn(arm_name):
    return LAYER_DIRS[arm_name]


def run_t5b(cfg) -> dict:
    return run_t5_experiment(
        experiment_id="T5-B_layer_ablation",
        hypothesis="H-EMB-03",
        title="T5-B ProteomeLM-L Layer Ablation (H-EMB-03)",
        arm_names=list(LAYER_DIRS.keys()),
        make_model_fn=_make_model,
        output_root="t5b",
        figures_dirname="tier5_b",
        per_arm_embedding_fn=_per_arm_embedding_fn,
    )
