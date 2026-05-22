"""Compute Morgan / RDKit fingerprints for canonical chemistry IDs.

Reads:
  data_contract/chemistry/canonical_id_smiles.json
    (produced by fetch_canonical_smiles.py)

Writes:
  data_contract/chemistry/canonical_fingerprints.npz with arrays:
    - canonical_ids:    str[n] (in vocab index order)
    - has_fingerprint:  bool[n]
    - morgan_2048:      uint8[n, 2048] (Morgan radius=2, 2048 bits)
    - maccs_167:        uint8[n, 167]  (MACCS keys)
    - rdkit_2048:       uint8[n, 2048] (RDKit topological fingerprint)

For canonical_ids without a valid SMILES (mixtures, polymers, lookup
failures), all fingerprint rows are zero and has_fingerprint=False.
Downstream code (e.g. per-experiment aggregation) can use this flag
to either skip them or fall back to a multihot bit.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def compute_fingerprints(smiles: str) -> dict[str, np.ndarray] | None:
    """Compute three fingerprint types for one SMILES.

    Returns None if SMILES cannot be parsed.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    morgan_bits = morgan_gen.GetFingerprintAsNumPy(mol).astype(np.uint8)

    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
    rdkit_bits = rdkit_gen.GetFingerprintAsNumPy(mol).astype(np.uint8)

    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_bits = np.zeros(167, dtype=np.uint8)
    on_bits = list(maccs_fp.GetOnBits())
    for b in on_bits:
        maccs_bits[b] = 1

    return {
        "morgan_2048": morgan_bits,
        "rdkit_2048": rdkit_bits,
        "maccs_167": maccs_bits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles-json", type=Path,
        default=Path("data_contract/chemistry/canonical_id_smiles.json"),
    )
    parser.add_argument(
        "--artifact-id", default="de21504134c84a6c",
        help="Feature contract artifact ID for vocab order",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data_contract/chemistry/canonical_fingerprints.npz"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    vocab_path = (
        Path("data_contract/preprocessing") / args.artifact_id / "canonical_id_vocab.json"
    )
    vocab = json.loads(vocab_path.read_text())["canonical_id_to_index"]
    n = max(vocab.values()) + 1
    # Ensure ordered by vocab index
    ordered_ids = [None] * n
    for cid, idx in vocab.items():
        ordered_ids[idx] = cid

    smiles_data = json.loads(args.smiles_json.read_text())

    morgan = np.zeros((n, 2048), dtype=np.uint8)
    rdkit = np.zeros((n, 2048), dtype=np.uint8)
    maccs = np.zeros((n, 167), dtype=np.uint8)
    has_fp = np.zeros(n, dtype=bool)
    parse_errors: list[str] = []

    n_ok, n_no_smiles, n_parse_fail = 0, 0, 0
    for i, cid in enumerate(ordered_ids):
        if cid is None:
            continue
        entry = smiles_data.get(cid)
        if entry is None or not entry.get("smiles"):
            n_no_smiles += 1
            continue
        fps = compute_fingerprints(entry["smiles"])
        if fps is None:
            n_parse_fail += 1
            parse_errors.append(cid)
            continue
        morgan[i] = fps["morgan_2048"]
        rdkit[i] = fps["rdkit_2048"]
        maccs[i] = fps["maccs_167"]
        has_fp[i] = True
        n_ok += 1

    log.info("Fingerprint summary:")
    log.info("  with valid fingerprint: %d / %d", n_ok, n)
    log.info("  no SMILES (skipped):    %d", n_no_smiles)
    log.info("  SMILES parse failure:   %d", n_parse_fail)
    if parse_errors:
        log.info("  parse-fail canonical_ids: %s", parse_errors[:10])

    np.savez_compressed(
        args.output,
        canonical_ids=np.array([cid or "" for cid in ordered_ids], dtype="<U64"),
        has_fingerprint=has_fp,
        morgan_2048=morgan,
        rdkit_2048=rdkit,
        maccs_167=maccs,
    )
    log.info("Saved fingerprints to %s", args.output)


if __name__ == "__main__":
    main()
