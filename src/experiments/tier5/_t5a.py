"""T5-A — Learnable Gene-Side Adapter (Hypothesis H-EMB-02).

Tests whether adding a learnable MLP on top of the frozen gene embedding
unlocks signal that the current concat-then-project pathway is missing.

The current locked T3 architecture treats the frozen 1152-dim gene
embedding and 425-dim chemistry vector symmetrically — both get concatenated
and projected to 512 together. A gene-side adapter lets the model first
transform the gene embedding into a more task-specific space before mixing
with chemistry, akin to a learned re-encoding of the protein representation.

Arms:
  - no_adapter:       locked T3 (no adapter — identity passthrough on gene)
  - adapter_256:      adapter MLP 1152→256→1152, then concat as usual
  - adapter_512:      adapter MLP 1152→512→1152
  - adapter_1024_proj: adapter MLP 1152→1024→512 (also reduces gene dim
                      to 512 before concat — tests dimension reduction)
"""
from __future__ import annotations

from src.experiments.tier5._t5_common import AdapterResidualMLP, run_t5_experiment

ARM_NONE = "no_adapter"
ARM_256 = "adapter_256"
ARM_512 = "adapter_512"
ARM_1024_PROJ = "adapter_1024_proj"


def _make_model(arm_name, gene_dim, chem_dim):
    if arm_name == ARM_NONE:
        return AdapterResidualMLP(
            gene_dim=gene_dim, chem_dim=chem_dim,
            hidden_dim=512, n_blocks=1, dropout=0.1,
            adapter_hidden=None,
        )
    if arm_name == ARM_256:
        return AdapterResidualMLP(
            gene_dim=gene_dim, chem_dim=chem_dim,
            hidden_dim=512, n_blocks=1, dropout=0.1,
            adapter_hidden=256, adapter_out=gene_dim,
        )
    if arm_name == ARM_512:
        return AdapterResidualMLP(
            gene_dim=gene_dim, chem_dim=chem_dim,
            hidden_dim=512, n_blocks=1, dropout=0.1,
            adapter_hidden=512, adapter_out=gene_dim,
        )
    # adapter_1024_proj
    return AdapterResidualMLP(
        gene_dim=gene_dim, chem_dim=chem_dim,
        hidden_dim=512, n_blocks=1, dropout=0.1,
        adapter_hidden=1024, adapter_out=512,
    )


def run_t5a(cfg) -> dict:
    return run_t5_experiment(
        experiment_id="T5-A_gene_adapter",
        hypothesis="H-EMB-02",
        title="T5-A Learnable Gene-Side Adapter (H-EMB-02)",
        arm_names=[ARM_NONE, ARM_256, ARM_512, ARM_1024_PROJ],
        make_model_fn=_make_model,
        output_root="t5a",
        figures_dirname="tier5_a",
    )
