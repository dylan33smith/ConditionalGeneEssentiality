#!/usr/bin/env python3
"""Build stressor ↔ v4 workbook match report + draft YAML (S4 Option D, Phase 0).

Writes:
  data_contract/preprocessing/_review/stressor_match_report.csv
  data_contract/preprocessing/_review/stressor_to_canonical_id.draft.yaml

Ratified file (human or CI): stressor_to_canonical_id.yaml — same keys as draft
but with ``<NEW>`` replaced by real Canonical_ID or keys removed to use
normalize-only fallback in the S4 runner.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.preprocessing.stressor_matcher import (
    build_candidate_report,
    build_draft_yaml_payload,
    build_ratified_mapping_from_report,
)


def _train_stressor_counts(
    experiments_path: Path,
    locked_protocol_path: Path,
) -> dict[str, int]:
    locked = yaml.safe_load(locked_protocol_path.read_text())
    held_out = set(locked["val_org_ids"]) | set(locked["test_org_ids"])
    exp = pd.read_parquet(experiments_path)
    train = exp[~exp["orgId"].isin(held_out)]
    ctr: Counter[str] = Counter()
    for col in ("condition_1", "condition_2", "condition_3", "condition_4"):
        if col not in train.columns:
            continue
        for v in train[col].dropna().astype(str):
            ctr[v.strip()] += 1
    return dict(ctr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiments",
        type=Path,
        default=Path("data/derived/canonical/v0/experiments.parquet"),
    )
    p.add_argument(
        "--workbook",
        type=Path,
        default=Path("data/media_composition_v4.xlsx"),
    )
    p.add_argument(
        "--sheet",
        default="Media_Components_ML",
    )
    p.add_argument(
        "--locked-protocol",
        type=Path,
        default=Path("data_contract/splits/locked_protocol.yaml"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data_contract/preprocessing/_review"),
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comps = pd.read_excel(args.workbook, sheet_name=args.sheet)
    if "Include_in_ml" in comps.columns:
        comps = comps[comps["Include_in_ml"] == True].copy()  # noqa: E712

    stressors = _train_stressor_counts(args.experiments, args.locked_protocol)
    report = build_candidate_report(stressors, comps)
    csv_path = args.out_dir / "stressor_match_report.csv"
    report.to_csv(csv_path, index=False)

    draft_payload = build_draft_yaml_payload(report)
    draft_path = args.out_dir / "stressor_to_canonical_id.draft.yaml"
    draft_path.write_text(yaml.safe_dump(draft_payload, sort_keys=False, default_flow_style=False))

    # Auto-ratified tier: exact + fuzzy >= 0.95 only (no <NEW> keys — cleaner for S4).
    ratified_map = build_ratified_mapping_from_report(report, fuzzy_auto_threshold=0.95)
    ratified_payload = {
        "version": 1,
        "description": "Auto ratified: exact_after_norm + fuzzy top1 >= 0.95. "
        "Omitted stressors fall back to normalize_chemical_name() as new canonical slots.",
        "map": ratified_map,
    }
    ratified_path = args.out_dir / "stressor_to_canonical_id.yaml"
    ratified_path.write_text(yaml.safe_dump(ratified_payload, sort_keys=False, default_flow_style=False))

    print(f"Wrote {csv_path} ({len(report)} rows)")
    print(f"Wrote {draft_path}")
    print(f"Wrote {ratified_path} ({len(ratified_map)} explicit mappings)")


if __name__ == "__main__":
    main()
