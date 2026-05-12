"""Build per-experiment wide metadata table with encoded categoricals + z-scored numerics."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocessing.build_experiment_chemistry import experiment_uid
from src.data.preprocessing.fit_condition_scalers import fit_standard_scaler
from src.data.preprocessing.fit_metadata_vocabs import fit_metadata_field_vocab
from src.data.preprocessing.parse_experiment_metadata import (
    parse_ph,
    parse_shaking_rpm,
    parse_temperature_celsius,
)


def _encode_categorical_series(
    series: pd.Series,
    vocab_path_payload: dict[str, Any],
) -> np.ndarray:
    """Map values to indices using frozen vocab_to_index (UNK/MISSING policy)."""
    v2i = vocab_path_payload["vocab_to_index"]
    unk = int(v2i[vocab_path_payload["unk_token"]])
    missing = int(v2i[vocab_path_payload["missing_token"]])
    out = np.full(len(series), unk, dtype=np.int32)
    for i, val in enumerate(series):
        if pd.isna(val):
            out[i] = missing
            continue
        key = str(val)
        out[i] = int(v2i[key]) if key in v2i else unk
    return out


def fit_and_build_experiment_metadata_table(
    experiments_df: pd.DataFrame,
    train_mask: pd.Series,
    *,
    categorical_specs: dict[str, str],
    numeric_source_cols: dict[str, str],
    metadata_prevalence_threshold: float,
    unk_token: str,
    missing_token: str,
    unk_index: int,
    missing_index: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Fit vocabs/scalers on train rows; return full encoded metadata + vocab payloads + numeric scalers."""
    train_df = experiments_df.loc[train_mask].copy()
    all_uids = [experiment_uid(r) for _, r in experiments_df.iterrows()]

    vocab_payloads: dict[str, dict[str, Any]] = {}
    for logical, src in categorical_specs.items():
        if src in train_df.columns:
            train_vals = train_df[src]
        else:
            train_vals = pd.Series([None] * len(train_df))
        vocab_payloads[logical] = fit_metadata_field_vocab(
            train_vals,
            field_name=logical,
            prevalence_threshold=metadata_prevalence_threshold,
            unk_token=unk_token,
            missing_token=missing_token,
            unk_index=unk_index,
            missing_index=missing_index,
        )

    # Numeric columns parsed
    parsed_train: dict[str, np.ndarray] = {}
    parsed_all: dict[str, np.ndarray] = {}
    for logical, src in numeric_source_cols.items():
        if src in experiments_df.columns:
            raw_all = experiments_df[src]
        else:
            raw_all = pd.Series([None] * len(experiments_df))
        if logical == "temperature_c":
            parsed_all[logical] = np.array([parse_temperature_celsius(x) for x in raw_all], dtype=np.float64)
            tr = train_df[src] if src in train_df.columns else pd.Series([None] * len(train_df))
            parsed_train[logical] = np.array([parse_temperature_celsius(x) for x in tr], dtype=np.float64)
        elif logical == "pH":
            parsed_all[logical] = np.array([parse_ph(x) for x in raw_all], dtype=np.float64)
            tr = train_df[src] if src in train_df.columns else pd.Series([None] * len(train_df))
            parsed_train[logical] = np.array([parse_ph(x) for x in tr], dtype=np.float64)
        elif logical == "shaking_rpm":
            parsed_all[logical] = np.array([parse_shaking_rpm(x) for x in raw_all], dtype=np.float64)
            tr = train_df[src] if src in train_df.columns else pd.Series([None] * len(train_df))
            parsed_train[logical] = np.array([parse_shaking_rpm(x) for x in tr], dtype=np.float64)
        else:
            raise ValueError(f"Unknown numeric metadata field: {logical}")

    numeric_scalers: dict[str, dict[str, Any]] = {}
    for logical in numeric_source_cols:
        arr = parsed_train[logical]
        numeric_scalers[logical] = fit_standard_scaler(arr)

    # Build encoded frame
    out_cols: dict[str, Any] = {"experiment_id": all_uids}
    for logical, src in categorical_specs.items():
        if src in experiments_df.columns:
            s = experiments_df[src]
        else:
            s = pd.Series([None] * len(experiments_df))
        out_cols[f"{logical}_idx"] = _encode_categorical_series(s, vocab_payloads[logical])

    for logical in numeric_source_cols:
        arr = parsed_all[logical]
        scaler = numeric_scalers[logical]
        mean, std = scaler["mean"], scaler["std"]
        z = np.zeros(len(arr), dtype=np.float32)
        m = np.isfinite(arr)
        z[m] = ((arr[m] - mean) / std).astype(np.float32)
        out_cols[f"{logical}_z"] = z
        out_cols[f"{logical}_is_finite"] = m.astype(np.int8)

    meta_df = pd.DataFrame(out_cols)
    return meta_df, vocab_payloads, numeric_scalers
