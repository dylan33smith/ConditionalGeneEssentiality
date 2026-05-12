"""Build per-experiment long chemistry table (media + stressors)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.preprocessing.stressor_matcher import normalize_chemical_name


def experiment_uid(row: pd.Series) -> str:
    """Stable 64-hex SHA256 id from ``(orgId, setName, seqindex, media)``."""
    media_val = row.get("media")
    if media_val is None or pd.isna(media_val):
        media_key = ""
    else:
        media_key = str(media_val).strip()
    key = "{}\x1f{}\x1f{}\x1f{}".format(
        str(row["orgId"]),
        str(row["setName"]),
        str(row["seqindex"]),
        media_key,
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def load_stressor_resolution_map(path: Path) -> dict[str, str]:
    """Load stressor string -> workbook Canonical_ID; skips ``<NEW>`` placeholders."""
    data = yaml.safe_load(path.read_text())
    raw = data.get("map", {}) if isinstance(data, dict) else {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if v is None or str(v).strip() in {"", "<NEW>"}:
            continue
        out[str(k).strip()] = str(v).strip()
    return out


def resolve_stressor_to_canonical(stressor: str, resolution_map: dict[str, str]) -> str:
    """Map stressor display string to workbook canonical or normalized new-slot id."""
    s = str(stressor).strip()
    if not s:
        return ""
    if s in resolution_map:
        return resolution_map[s]
    ns = normalize_chemical_name(s)
    for key, cid in resolution_map.items():
        if normalize_chemical_name(key) == ns:
            return cid
    return ns if ns else s


def build_experiment_chemistry_long(
    experiments_df: pd.DataFrame,
    components_df: pd.DataFrame,
    resolution_map: dict[str, str],
    *,
    include_in_ml_only: bool = True,
) -> pd.DataFrame:
    """Return long table: experiment_id, canonical_id, role, amount (raw scalar or NaN).

    ``amount`` for ``medium`` rows comes from workbook ``Amount`` (first row per
    (media, canonical_id)). For ``stressor`` rows it is ``concentration_i`` when
    present, else NaN.
    """
    comps = components_df.copy()
    if include_in_ml_only and "Include_in_ml" in comps.columns:
        comps = comps[comps["Include_in_ml"] == True].copy()  # noqa: E712
    comps["Media"] = comps["Media"].astype(str)
    comps["Canonical_ID"] = comps["Canonical_ID"].astype(str)
    if "Amount" in comps.columns:
        comps["_amt"] = pd.to_numeric(comps["Amount"], errors="coerce")
    else:
        comps["_amt"] = float("nan")
    # First non-NaN amount per (Media, Canonical_ID)
    comp_first = comps.sort_values(["Media", "Canonical_ID"]).groupby(["Media", "Canonical_ID"], as_index=False).first()

    rows: list[dict[str, Any]] = []
    for _, exp in experiments_df.iterrows():
        eid = experiment_uid(exp)
        media = str(exp["media"]) if pd.notna(exp.get("media")) else ""
        if media:
            sub = comp_first[comp_first["Media"] == media]
            for _, cr in sub.iterrows():
                rows.append(
                    {
                        "experiment_id": eid,
                        "canonical_id": str(cr["Canonical_ID"]),
                        "role": "medium",
                        "amount": float(cr["_amt"]) if pd.notna(cr["_amt"]) else float("nan"),
                    }
                )
        for i in (1, 2, 3, 4):
            ccol = f"condition_{i}"
            if ccol not in exp.index:
                continue
            raw_cond = exp.get(ccol)
            if pd.isna(raw_cond):
                continue
            canon = resolve_stressor_to_canonical(str(raw_cond), resolution_map)
            if not canon:
                continue
            conc_col = f"concentration_{i}"
            amt = float("nan")
            if conc_col in exp.index and pd.notna(exp.get(conc_col)):
                amt = float(pd.to_numeric(exp.get(conc_col), errors="coerce"))
                if not (amt == amt):  # NaN check
                    amt = float("nan")
            rows.append(
                {
                    "experiment_id": eid,
                    "canonical_id": canon,
                    "role": "stressor",
                    "amount": amt,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["experiment_id", "canonical_id", "role", "amount"])
    out = pd.DataFrame(rows)

    def _agg_amount_series(s: pd.Series) -> float:
        vals = pd.to_numeric(s, errors="coerce").dropna()
        if vals.empty:
            return float("nan")
        return float(vals.max())

    out = (
        out.groupby(["experiment_id", "canonical_id", "role"], as_index=False)
        .agg(amount=("amount", _agg_amount_series))
        .sort_values(["experiment_id", "role", "canonical_id"])
        .reset_index(drop=True)
    )
    return out


def apply_chemistry_vocab_trim(
    chemistry_df: pd.DataFrame,
    *,
    kept_canonicals: set[str],
    unk_token: str,
    unk_stressor_token: str,
) -> pd.DataFrame:
    """Map OOV canonical_id to ``unk_token`` (medium) or ``unk_stressor_token`` (stressor)."""
    out = chemistry_df.copy()

    def _map_row(r: pd.Series) -> str:
        c = str(r["canonical_id"])
        if c in kept_canonicals:
            return c
        if str(r["role"]) == "stressor":
            return unk_stressor_token
        return unk_token

    out["canonical_id"] = out.apply(_map_row, axis=1)
    return out
