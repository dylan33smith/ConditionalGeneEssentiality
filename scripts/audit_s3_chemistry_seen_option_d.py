"""Audit val canonical seen-rates under Option D chemistry for each S3 candidate protocol."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.preprocessing.build_experiment_chemistry import (
    build_experiment_chemistry_long,
    experiment_uid,
    load_stressor_resolution_map,
)


def _load_components() -> pd.DataFrame:
    p = _ROOT / "data/media_composition_v4.xlsx"
    df = pd.read_excel(p, sheet_name="Media_Components_ML")
    return df[df["Include_in_ml"] == True].copy()  # noqa: E712


def seen_rate_for_protocol(
    experiments: pd.DataFrame,
    chemistry: pd.DataFrame,
    *,
    val_orgs: set[str],
) -> tuple[float, float]:
    """Fraction of val experiments where every non-special canonical_id appeared in train chem."""
    train_mask = ~experiments["orgId"].isin(val_orgs)
    val_mask = experiments["orgId"].isin(val_orgs)
    train_uids = {experiment_uid(r) for _, r in experiments.loc[train_mask].iterrows()}
    val_uids = {experiment_uid(r) for _, r in experiments.loc[val_mask].iterrows()}
    train_c = chemistry[chemistry["experiment_id"].isin(train_uids)]
    val_c = chemistry[chemistry["experiment_id"].isin(val_uids)]
    train_canon = set(train_c["canonical_id"].astype(str).unique())
    ok = 0
    n_val_exp = 0
    for uid in sorted(val_uids):
        sub = val_c[val_c["experiment_id"] == uid]
        if sub.empty:
            continue
        n_val_exp += 1
        cans = set(sub["canonical_id"].astype(str).unique())
        if all(c in train_canon for c in cans):
            ok += 1
    rate = ok / max(n_val_exp, 1)
    # Row-weighted: fraction of val chemistry rows whose canonical is in train_canon
    n_rows = len(val_c)
    row_rate = float((val_c["canonical_id"].astype(str).isin(train_canon)).sum() / max(n_rows, 1))
    return rate, row_rate


def main() -> None:
    exp_path = _ROOT / "data/derived/canonical/v0/experiments.parquet"
    cand_path = _ROOT / "data_contract/splits/candidate_protocols.yaml"
    res_path = _ROOT / "data_contract/preprocessing/_review/stressor_to_canonical_id.yaml"

    experiments = pd.read_parquet(exp_path)
    comps = _load_components()
    res = load_stressor_resolution_map(res_path)
    chem = build_experiment_chemistry_long(experiments, comps, res, include_in_ml_only=True)

    data = yaml.safe_load(cand_path.read_text())
    rows = []
    for c in data["candidates"]:
        pid = c["protocol_id"]
        val_orgs = set(c["val_org_ids"])
        test_orgs = set(c["test_org_ids"])
        held = val_orgs | test_orgs
        train_mask = ~experiments["orgId"].isin(held)
        train_uids = {experiment_uid(r) for _, r in experiments.loc[train_mask].iterrows()}
        val_uids = {experiment_uid(r) for _, r in experiments.loc[experiments["orgId"].isin(val_orgs)].iterrows()}
        train_c = chem[chem["experiment_id"].isin(train_uids)]
        val_c = chem[chem["experiment_id"].isin(val_uids)]
        train_canon = set(train_c["canonical_id"].astype(str).unique())
        n_val_exp = 0
        ok_exp = 0
        for uid in val_uids:
            sub = val_c[val_c["experiment_id"] == uid]
            if sub.empty:
                continue
            n_val_exp += 1
            cans = set(sub["canonical_id"].astype(str).unique())
            if all(x in train_canon for x in cans):
                ok_exp += 1
        exp_rate = ok_exp / max(n_val_exp, 1)
        row_rate = float((val_c["canonical_id"].astype(str).isin(train_canon)).sum() / max(len(val_c), 1))
        rows.append((pid, exp_rate, row_rate, n_val_exp, len(val_c)))

    print("protocol_id\tval_exp_all_canon_seen_rate\tval_row_canon_seen_rate\tval_n_exp\tval_n_chem_rows")
    for r in rows:
        print(f"{r[0]}\t{r[1]:.6f}\t{r[2]:.6f}\t{r[3]}\t{r[4]}")


if __name__ == "__main__":
    main()
