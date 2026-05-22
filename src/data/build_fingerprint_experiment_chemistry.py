"""Build per-experiment fingerprint vectors by aggregating canonical_id fingerprints.

For each experiment, compute the mean fingerprint over all canonical_ids
present in that experiment's chemistry. This is the chemistry analog of
mean-pooling per-protein ESM-C embeddings to get a proteome-level vector.

Input:
  - data_contract/preprocessing/<artifact>/experiment_chemistry.parquet
    (long table: experiment_id, canonical_id, role, amount, log1p_amount)
  - data_contract/chemistry/canonical_fingerprints.npz
    (produced by compute_chem_fingerprints.py)

Output:
  - data_contract/chemistry/experiment_fingerprints.npz with arrays:
    - experiment_ids:    str[n_exp]
    - morgan_mean:       float32[n_exp, 2048]   mean Morgan fp over present cids
    - rdkit_mean:        float32[n_exp, 2048]   mean RDKit fp
    - maccs_mean:        float32[n_exp, 167]    mean MACCS keys
    - presence_multihot: float32[n_exp, 425]    binary presence (current encoding)
    - coverage:          float32[n_exp]         fraction of present cids that
                                                have a fingerprint
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-id", default="de21504134c84a6c",
    )
    parser.add_argument(
        "--fingerprints", type=Path,
        default=Path("data_contract/chemistry/canonical_fingerprints.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data_contract/chemistry/experiment_fingerprints.npz"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    artifact_root = Path("data_contract/preprocessing") / args.artifact_id
    chem_long = pd.read_parquet(artifact_root / "experiment_chemistry.parquet")
    vocab = json.loads((artifact_root / "canonical_id_vocab.json").read_text())[
        "canonical_id_to_index"
    ]
    n_vocab = max(vocab.values()) + 1

    log.info("Loaded %d (experiment, canonical_id) rows across %d experiments",
             len(chem_long), chem_long["experiment_id"].nunique())

    fps_bundle = np.load(args.fingerprints)
    morgan_per_cid = fps_bundle["morgan_2048"].astype(np.float32)
    rdkit_per_cid = fps_bundle["rdkit_2048"].astype(np.float32)
    maccs_per_cid = fps_bundle["maccs_167"].astype(np.float32)
    has_fp = fps_bundle["has_fingerprint"]
    log.info("Per-canonical_id fingerprints loaded: %d/%d have valid fp",
             has_fp.sum(), len(has_fp))

    # Map canonical_id strings to vocab indices
    cid_to_idx = vocab
    chem_long = chem_long[chem_long["canonical_id"].isin(cid_to_idx)].copy()
    chem_long["cid_idx"] = chem_long["canonical_id"].map(cid_to_idx)

    # Group by experiment_id, collect present canonical_id indices
    sorted_experiments = sorted(chem_long["experiment_id"].unique())
    exp_to_row = {eid: i for i, eid in enumerate(sorted_experiments)}
    n_exp = len(sorted_experiments)

    morgan_mean = np.zeros((n_exp, 2048), dtype=np.float32)
    rdkit_mean = np.zeros((n_exp, 2048), dtype=np.float32)
    maccs_mean = np.zeros((n_exp, 167), dtype=np.float32)
    presence = np.zeros((n_exp, n_vocab), dtype=np.float32)
    coverage = np.zeros(n_exp, dtype=np.float32)

    grouped = chem_long.groupby("experiment_id")["cid_idx"].apply(list)
    for eid, cid_idxs in grouped.items():
        row = exp_to_row[eid]
        cid_idxs = np.unique(np.asarray(cid_idxs, dtype=np.int64))
        presence[row, cid_idxs] = 1.0
        valid = cid_idxs[has_fp[cid_idxs]]
        if len(valid) > 0:
            morgan_mean[row] = morgan_per_cid[valid].mean(axis=0)
            rdkit_mean[row] = rdkit_per_cid[valid].mean(axis=0)
            maccs_mean[row] = maccs_per_cid[valid].mean(axis=0)
            coverage[row] = len(valid) / len(cid_idxs)

    log.info("Coverage stats (fraction of present cids with a fingerprint):")
    log.info("  mean=%.3f  median=%.3f  min=%.3f  max=%.3f",
             coverage.mean(), np.median(coverage), coverage.min(), coverage.max())
    log.info("  experiments with 0%% coverage: %d", (coverage == 0).sum())
    log.info("  experiments with 100%% coverage: %d", (coverage == 1).sum())

    np.savez_compressed(
        args.output,
        experiment_ids=np.array(sorted_experiments, dtype="<U128"),
        morgan_mean=morgan_mean,
        rdkit_mean=rdkit_mean,
        maccs_mean=maccs_mean,
        presence_multihot=presence,
        coverage=coverage,
    )
    log.info("Saved experiment-level fingerprints to %s", args.output)
    log.info("Shapes: morgan_mean=%s presence=%s", morgan_mean.shape, presence.shape)


if __name__ == "__main__":
    main()
