"""Look up SMILES strings for every canonical_id in the locked vocab.

Reads:
  - data_contract/preprocessing/<artifact>/canonical_id_vocab.json
  - data/media_composition_v4.xlsx (sheet Media_Components_ML) for workbook
    canonical_id → compound_name resolution
  - data_contract/preprocessing/<artifact>/stressor_to_canonical_id.yaml
    for stressor-name aliasing

For each canonical_id, picks the best human-readable chemical name then queries
PubChem (via pubchempy). Caches results to a JSON file so re-runs are cheap.

Writes:
  data_contract/chemistry/canonical_id_smiles.json

The output schema is:
  {
    "<canonical_id>": {
      "lookup_name": "<the name queried at PubChem>",
      "smiles": "<isomeric SMILES or null>",
      "pubchem_cid": <int or null>,
      "status": "ok" | "no_match" | "ambiguous_mixture" | "error: ..."
    },
    ...
  }
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Canonical_ids that are mixtures/polymers/undefined; skip PubChem lookup.
KNOWN_NON_MOLECULAR = {
    "yeast extract", "casamino acids", "casein digest peptone",
    "wolfe's mineral mix", "wolfe's vitamin mix", "polygalacturonic acid",
    "starch", "tryptone", "(+)-arabinogalactan", "agar", "lb broth",
    "peptone", "soytone", "alginic acid", "xylan", "cellulose",
    "casein", "beef extract", "malt extract", "polypeptone",
    "<unk>", "<unk_stressor>",
}


def build_id_to_name(
    vocab: dict[str, int],
    workbook_df: pd.DataFrame,
    stressor_map: dict[str, str] | None,
) -> dict[str, str]:
    """Resolve canonical_id → a human-readable compound name.

    Strategy:
      1. If the canonical_id matches a workbook short-id (e.g., 'glc-D'),
         use the workbook's Compound_name.
      2. Otherwise the canonical_id itself is the chemical name (stressors).
      3. If the stressor map has an alias (long name → short id), invert it.
    """
    workbook_name = (
        workbook_df.groupby("Canonical_ID")["Compound_name"]
        .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "")
        .to_dict()
    )
    inv_stressor: dict[str, str] = {}
    if stressor_map:
        for full_name, short_id in stressor_map.items():
            # Prefer the longest full_name per short_id (more descriptive)
            if short_id not in inv_stressor or len(full_name) > len(inv_stressor[short_id]):
                inv_stressor[short_id] = full_name

    out: dict[str, str] = {}
    for cid in vocab:
        if cid in workbook_name and workbook_name[cid]:
            out[cid] = workbook_name[cid]
        elif cid in inv_stressor:
            out[cid] = inv_stressor[cid]
        else:
            out[cid] = cid
    return out


def clean_name_for_pubchem(name: str) -> str:
    """Light normalization for PubChem name search."""
    # Strip common qualifiers
    for suffix in [
        " (pABA)", " (Vitamin B6)", " (Vitamin B12)", " (Vitamin B1)",
        " (Vitamin B2)", " (Vitamin B3)", " hydrochloride hydrate",
        " hydrochloride", " hydrate", " heptahydrate", " hexahydrate",
        " pentahydrate", " tetrahydrate", " dihydrate", " monohydrate",
        " anhydrous", " disodium salt", " monosodium salt",
        " monopotassium salt", " disodium", " sodium salt",
    ]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def lookup_smiles(name: str, sleep_after: float = 0.15) -> tuple[str | None, int | None, str]:
    """Query PubChem for a SMILES given a compound name.

    Returns (smiles, cid, status).
    """
    import pubchempy as pcp

    cleaned = clean_name_for_pubchem(name)
    if cleaned.lower() in KNOWN_NON_MOLECULAR:
        return None, None, "ambiguous_mixture"

    try:
        compounds = pcp.get_compounds(cleaned, "name")
    except Exception as e:
        return None, None, f"error: {type(e).__name__}: {str(e)[:80]}"
    finally:
        time.sleep(sleep_after)

    if not compounds:
        return None, None, "no_match"
    c = compounds[0]
    smiles = None
    for attr in ("isomeric_smiles", "smiles", "canonical_smiles"):
        smiles = getattr(c, attr, None)
        if smiles:
            break
    if not smiles:
        return None, c.cid, "no_smiles_attr"
    return smiles, c.cid, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-id", default="de21504134c84a6c",
        help="Feature contract artifact ID (default: locked S4 artifact)",
    )
    parser.add_argument(
        "--workbook", type=Path,
        default=Path("data/media_composition_v4.xlsx"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data_contract/chemistry/canonical_id_smiles.json"),
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Ignore cache and re-fetch all SMILES",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    artifact_root = Path("data_contract/preprocessing") / args.artifact_id
    vocab_path = artifact_root / "canonical_id_vocab.json"
    stressor_path = artifact_root / "stressor_to_canonical_id.yaml"

    log.info("Loading vocab from %s", vocab_path)
    vocab = json.loads(vocab_path.read_text())["canonical_id_to_index"]
    log.info("  %d canonical_ids in vocab", len(vocab))

    log.info("Loading workbook compound names from %s", args.workbook)
    workbook = pd.read_excel(args.workbook, sheet_name="Media_Components_ML")

    stressor_map = None
    if stressor_path.exists():
        stressor_map = yaml.safe_load(stressor_path.read_text()).get("map", {})

    id_to_name = build_id_to_name(vocab, workbook, stressor_map)

    cache: dict = {}
    if args.output.exists() and not args.force_refresh:
        cache = json.loads(args.output.read_text())
        log.info("Loaded %d cached entries from %s", len(cache), args.output)

    to_fetch = [cid for cid in vocab if cid not in cache]
    log.info("Need to fetch %d new SMILES (%d cached)",
             len(to_fetch), len(cache))

    fetched = 0
    for i, cid in enumerate(to_fetch):
        name = id_to_name.get(cid, cid)
        smiles, pubchem_cid, status = lookup_smiles(name)
        cache[cid] = {
            "lookup_name": name,
            "smiles": smiles,
            "pubchem_cid": pubchem_cid,
            "status": status,
        }
        fetched += 1
        if fetched % 25 == 0:
            log.info("[%d/%d] %s -> %s (cid=%s)",
                     i + 1, len(to_fetch), name[:40], status, pubchem_cid)
            args.output.write_text(json.dumps(cache, indent=2))
    args.output.write_text(json.dumps(cache, indent=2))

    # Summary
    status_counts: dict[str, int] = {}
    for v in cache.values():
        s = v["status"]
        if s.startswith("error"):
            s = "error"
        status_counts[s] = status_counts.get(s, 0) + 1
    log.info("=" * 60)
    log.info("SMILES lookup complete:")
    for s, n in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        log.info("  %s: %d", s, n)
    log.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
