"""Condition vocabulary fitting utilities (train-only)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _as_media_counter(train_rows_df: pd.DataFrame, media_col: str = "media") -> Counter:
    media = train_rows_df[media_col].dropna().astype(str)
    return Counter(media.tolist())


def fit_canonical_id_vocab(
    train_rows_df: pd.DataFrame,
    components_df: pd.DataFrame,
    *,
    prevalence_threshold: float,
    media_col: str = "media",
    component_media_col: str = "Media",
    canonical_col: str = "Canonical_ID",
    include_in_ml_col: str = "Include_in_ml",
    unk_token: str = "<UNK>",
    unk_index: int = 0,
) -> dict[str, Any]:
    """Fit canonical_id vocabulary on train rows only.

    A canonical id is included if its train-row prevalence is >= threshold, where
    prevalence is measured by summing train-row counts for all media that include
    the canonical id, divided by n_train_rows.
    """
    if len(train_rows_df) == 0:
        raise ValueError("train_rows_df is empty; cannot fit feature contract.")

    comps = components_df.copy()
    if include_in_ml_col in comps.columns:
        comps = comps[comps[include_in_ml_col] == True].copy()  # noqa: E712
    comps[component_media_col] = comps[component_media_col].astype(str)
    comps[canonical_col] = comps[canonical_col].astype(str)
    comps = comps.drop_duplicates(subset=[component_media_col, canonical_col])

    media_counts = _as_media_counter(train_rows_df, media_col=media_col)
    n_train_rows = int(sum(media_counts.values()))
    train_media = set(media_counts.keys())
    comps_train = comps[comps[component_media_col].isin(train_media)].copy()

    canonical_row_mass: Counter = Counter()
    for media, canon in comps_train[[component_media_col, canonical_col]].itertuples(index=False):
        canonical_row_mass[str(canon)] += int(media_counts.get(str(media), 0))

    min_rows = max(1, int(prevalence_threshold * n_train_rows))
    kept = sorted([c for c, n in canonical_row_mass.items() if n >= min_rows])
    canonical_id_vocab = [unk_token] + kept
    if canonical_id_vocab[unk_index] != unk_token:
        raise ValueError("UNK token must be placed at unk_index")
    canonical_id_to_index = {c: i for i, c in enumerate(canonical_id_vocab)}

    return {
        "canonical_id_vocab": canonical_id_vocab,
        "canonical_id_to_index": canonical_id_to_index,
        "n_train_rows": n_train_rows,
        "prevalence_threshold": float(prevalence_threshold),
        "min_rows_required": int(min_rows),
        "n_canonical_pretrim": int(len(canonical_row_mass)),
        "n_canonical_posttrim": int(len(kept)),
    }


def fit_canonical_id_vocab_from_chemistry_long(
    train_chemistry_df: pd.DataFrame,
    *,
    prevalence_threshold: float,
    unk_token: str = "<UNK>",
    unk_stressor_token: str = "<UNK_STRESSOR>",
    unk_index: int = 0,
) -> dict[str, Any]:
    """Fit canonical_id vocabulary from a train-only long chemistry table.

    Prevalence is measured as: for each ``canonical_id``, count distinct
    ``experiment_id`` in the train slice, divided by the number of distinct
    train ``experiment_id`` overall.
    """
    if train_chemistry_df.empty:
        raise ValueError("train_chemistry_df is empty; cannot fit feature contract.")
    n_train_exp = int(train_chemistry_df["experiment_id"].nunique())
    exp_per_canon = train_chemistry_df.groupby("canonical_id")["experiment_id"].nunique()
    min_exp = max(1, int(prevalence_threshold * n_train_exp))
    reserved = {unk_token, unk_stressor_token}
    kept = sorted(
        str(c)
        for c, n in exp_per_canon.items()
        if int(n) >= min_exp and str(c) not in reserved
    )
    canonical_id_vocab = [unk_token, unk_stressor_token] + kept
    if canonical_id_vocab[unk_index] != unk_token:
        raise ValueError("UNK token must be placed at unk_index")
    canonical_id_to_index = {c: i for i, c in enumerate(canonical_id_vocab)}
    return {
        "canonical_id_vocab": canonical_id_vocab,
        "canonical_id_to_index": canonical_id_to_index,
        "unk_stressor_token": unk_stressor_token,
        "n_train_experiments": n_train_exp,
        "prevalence_threshold": float(prevalence_threshold),
        "min_experiments_required": int(min_exp),
        "n_canonical_pretrim": int(len(exp_per_canon)),
        "n_canonical_posttrim": int(len(kept)),
    }


def build_media_to_multihot(
    media_names: list[str],
    components_df: pd.DataFrame,
    canonical_id_to_index: dict[str, int],
    *,
    component_media_col: str = "Media",
    canonical_col: str = "Canonical_ID",
    include_in_ml_col: str = "Include_in_ml",
    unk_index: int = 0,
) -> pd.DataFrame:
    """Build idempotent media × canonical multihot table.

    Returns a DataFrame indexed by media with one column per canonical index.
    """
    medias = sorted({str(m) for m in media_names if pd.notna(m)})
    n_vocab = len(canonical_id_to_index)
    out = pd.DataFrame(0, index=medias, columns=[str(i) for i in range(n_vocab)], dtype="int8")

    comps = components_df.copy()
    if include_in_ml_col in comps.columns:
        comps = comps[comps[include_in_ml_col] == True].copy()  # noqa: E712
    comps[component_media_col] = comps[component_media_col].astype(str)
    comps[canonical_col] = comps[canonical_col].astype(str)
    comps = comps.drop_duplicates(subset=[component_media_col, canonical_col])
    comps = comps[comps[component_media_col].isin(medias)]

    grouped = comps.groupby(component_media_col)[canonical_col].apply(list)
    for media, canonicals in grouped.items():
        indices = {canonical_id_to_index.get(c, unk_index) for c in canonicals}
        for idx in indices:
            out.at[media, str(idx)] = 1
        if not indices:
            out.at[media, str(unk_index)] = 1

    # Any media without component rows become all-UNK.
    for media in medias:
        if int(out.loc[media].sum()) == 0:
            out.at[media, str(unk_index)] = 1

    out.index.name = "media"
    return out.reset_index()


def build_representation_mode_per_medium(
    components_df: pd.DataFrame,
    mode_mapping_yaml: Path,
    *,
    component_media_col: str = "Media",
    decomposition_col: str = "Decomposition_type",
    include_in_ml_col: str = "Include_in_ml",
) -> pd.DataFrame:
    """Return per-medium dominant mode and mode proportions."""
    mapping_payload = yaml.safe_load(mode_mapping_yaml.read_text())
    mode_by_decomposition = {
        str(k): str(v["mode"]) for k, v in mapping_payload["mapping"].items()
    }
    modes = list(mapping_payload["modes"])

    comps = components_df.copy()
    if include_in_ml_col in comps.columns:
        comps = comps[comps[include_in_ml_col] == True].copy()  # noqa: E712
    comps[component_media_col] = comps[component_media_col].astype(str)
    comps["_mode"] = comps[decomposition_col].astype(str).map(mode_by_decomposition).fillna("in_silico")

    rows: list[dict[str, Any]] = []
    for media, sub in comps.groupby(component_media_col):
        counts = sub["_mode"].value_counts()
        total = float(max(int(counts.sum()), 1))
        dominant = str(counts.idxmax()) if len(counts) > 0 else "in_silico"
        row: dict[str, Any] = {"media": str(media), "dominant_mode": dominant}
        for mode in modes:
            row[f"prop_{mode}"] = float(counts.get(mode, 0) / total)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("media").reset_index(drop=True)


def build_representation_mode_per_canonical(
    chemistry_canonical_ids: list[str],
    components_df: pd.DataFrame,
    mode_mapping_yaml: Path,
    *,
    component_media_col: str = "Media",
    canonical_col: str = "Canonical_ID",
    decomposition_col: str = "Decomposition_type",
    include_in_ml_col: str = "Include_in_ml",
) -> pd.DataFrame:
    """One row per canonical_id in the chemistry union with dominant mode + props.

    Canonical IDs that never appear in the workbook ML sheet are tagged
    ``dominant_mode=stressor`` with ``prop_stressor=1.0`` so T1 can join a mode
    row for every chemistry-table token.
    """
    mapping_payload = yaml.safe_load(mode_mapping_yaml.read_text())
    mode_by_decomposition = {
        str(k): str(v["mode"]) for k, v in mapping_payload["mapping"].items()
    }
    modes = list(mapping_payload["modes"])

    comps = components_df.copy()
    if include_in_ml_col in comps.columns:
        comps = comps[comps[include_in_ml_col] == True].copy()  # noqa: E712
    comps[canonical_col] = comps[canonical_col].astype(str)
    comps["_mode"] = comps[decomposition_col].astype(str).map(mode_by_decomposition).fillna("in_silico")

    rows: list[dict[str, Any]] = []
    for cid in sorted({str(c) for c in chemistry_canonical_ids if str(c)}):
        sub = comps[comps[canonical_col] == cid]
        if len(sub) == 0:
            row: dict[str, Any] = {"canonical_id": cid, "dominant_mode": "stressor"}
            for m in modes:
                row[f"prop_{m}"] = 1.0 if m == "stressor" else 0.0
            rows.append(row)
            continue
        counts = sub["_mode"].value_counts()
        total = float(max(int(counts.sum()), 1))
        dominant = str(counts.idxmax()) if len(counts) > 0 else "in_silico"
        row = {"canonical_id": cid, "dominant_mode": dominant}
        for mode in modes:
            row[f"prop_{mode}"] = float(counts.get(mode, 0) / total)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("canonical_id").reset_index(drop=True)
