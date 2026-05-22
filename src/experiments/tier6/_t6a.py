"""T6-A — Chemical Fingerprints vs Multihot (Hypothesis H-CHEM-01).

Tests whether replacing (or augmenting) the 425-dim binary multihot
chemistry vector with chemical structural fingerprints unlocks signal.

Multihot encoding treats each canonical_id as an arbitrary token; nothing
about the chemical structure is shared between similar compounds. Morgan
and RDKit fingerprints encode substructural features — chemically similar
compounds end up with similar vectors, enabling the model to generalize
across structurally related conditions.

Arms (all use locked T5-A architecture; only chemistry input dim varies):
  - multihot:              425-d binary (current locked baseline)
  - morgan_only:           2048-d mean Morgan fingerprint (radius=2)
  - rdkit_only:            2048-d mean RDKit topological fingerprint
  - maccs_only:            167-d  mean MACCS keys
  - morgan_plus_multihot:  2048+425 concat (preserves both signals)
  - maccs_plus_multihot:   167+425 concat (smaller alternative)

Per-experiment fingerprint = mean over canonical_ids present in the
experiment that have a valid SMILES (mixtures/polymers skipped).
"""
from __future__ import annotations

from src.experiments.tier6._t6_common import run_t6_experiment


def run_t6a(cfg) -> dict:
    return run_t6_experiment(
        experiment_id="T6-A_fingerprints",
        hypothesis="H-CHEM-01",
        title="T6-A Chemical Fingerprints vs Multihot (H-CHEM-01)",
        arm_names=[
            "multihot",
            "morgan_only",
            "rdkit_only",
            "maccs_only",
            "morgan_plus_multihot",
            "maccs_plus_multihot",
        ],
        output_root="t6a",
        figures_dirname="tier6_a",
    )
