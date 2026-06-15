"""Per-experiment molecular fingerprint bundles (Morgan / RDKit / MACCS).

Extracted verbatim from the legacy tier6/_t6_common.py so the ranking package
carries no T-tier dependency. The bundle is built offline by
src/data/build_fingerprint_experiment_chemistry.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_experiment_fingerprints(
    path: Path = Path("data_contract/chemistry/experiment_fingerprints.npz"),
) -> dict:
    """Load the per-experiment fingerprint bundle.

    Returns dict with keys: experiment_ids, morgan_mean, rdkit_mean,
    maccs_mean, presence_multihot, coverage, and exp_to_row.
    """
    bundle = np.load(path)
    exp_ids = bundle["experiment_ids"].tolist()
    return {
        "experiment_ids": exp_ids,
        "morgan_mean": bundle["morgan_mean"],
        "rdkit_mean": bundle["rdkit_mean"],
        "maccs_mean": bundle["maccs_mean"],
        "presence_multihot": bundle["presence_multihot"],
        "coverage": bundle["coverage"],
        "exp_to_row": {eid: i for i, eid in enumerate(exp_ids)},
    }
