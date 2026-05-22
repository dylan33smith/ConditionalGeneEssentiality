"""T5-D — Adapter Variants (Hypothesis H-EMB-05).

Follow-up to T5-A. T5-A established that a learnable gene-side adapter
helps, with the dimension-reducing variant (1152→1024→512) being the only
arm to beat baseline. T5-D probes which adapter design choices matter:

  - baseline_1024_512:   T5-A winner (single hidden Linear(1152,1024),
                         out_dim 512). Control arm — should reproduce.
  - out_dim_256:         Same as baseline but out_dim 256 (more aggressive
                         dimension reduction).
  - hidden_2048:         Wider adapter hidden (Linear(1152,2048), out 512).
  - two_hidden_layers:   Deeper adapter (two hidden layers, hidden 1024).
  - layernorm:           Add LayerNorm before adapter (fixes per-organism
                         distribution drift in frozen embeddings).

Promotion rule: best arm on co-primary metrics. Note that all arms in T5-D
have changed architectures vs T5-A's baseline — the "baseline_1024_512" arm
here is the T5-A winner being re-tested as a control.

NOTE: T5-D originally planned as ProteomeLM fine-tuning was dropped in
favor of this adapter follow-up. Fine-tuning deferred indefinitely
(see T5-DEC-002 when written).
"""
from __future__ import annotations

from src.experiments.tier5._t5_common import AdapterResidualMLP, run_t5_experiment

ARM_BASELINE = "baseline_1024_512"
ARM_OUT256 = "out_dim_256"
ARM_HIDDEN2048 = "hidden_2048"
ARM_2LAYER = "two_hidden_layers"
ARM_LAYERNORM = "layernorm"


def _make_model(arm_name, gene_dim, chem_dim):
    common = dict(
        gene_dim=gene_dim, chem_dim=chem_dim,
        hidden_dim=512, n_blocks=1, dropout=0.1,
    )
    if arm_name == ARM_BASELINE:
        return AdapterResidualMLP(
            **common,
            adapter_hidden=1024, adapter_out=512,
            adapter_n_hidden_layers=1, adapter_layernorm=False,
        )
    if arm_name == ARM_OUT256:
        return AdapterResidualMLP(
            **common,
            adapter_hidden=1024, adapter_out=256,
            adapter_n_hidden_layers=1, adapter_layernorm=False,
        )
    if arm_name == ARM_HIDDEN2048:
        return AdapterResidualMLP(
            **common,
            adapter_hidden=2048, adapter_out=512,
            adapter_n_hidden_layers=1, adapter_layernorm=False,
        )
    if arm_name == ARM_2LAYER:
        return AdapterResidualMLP(
            **common,
            adapter_hidden=1024, adapter_out=512,
            adapter_n_hidden_layers=2, adapter_layernorm=False,
        )
    # layernorm
    return AdapterResidualMLP(
        **common,
        adapter_hidden=1024, adapter_out=512,
        adapter_n_hidden_layers=1, adapter_layernorm=True,
    )


def run_t5d(cfg) -> dict:
    return run_t5_experiment(
        experiment_id="T5-D_adapter_variants",
        hypothesis="H-EMB-05",
        title="T5-D Adapter Variants (H-EMB-05)",
        arm_names=[ARM_BASELINE, ARM_OUT256, ARM_HIDDEN2048, ARM_2LAYER, ARM_LAYERNORM],
        make_model_fn=_make_model,
        output_root="t5d",
        figures_dirname="tier5_d",
    )
