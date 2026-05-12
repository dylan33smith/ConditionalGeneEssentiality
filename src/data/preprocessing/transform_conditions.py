"""Apply frozen condition preprocessing artifacts (S4 Option D)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _resolve_artifact_dir(path: Path) -> Path:
    p = Path(path)
    if p.is_dir():
        return p
    return p.parent


def load_experiment_chemistry(artifact_dir: Path) -> pd.DataFrame:
    """Load locked ``experiment_chemistry.parquet``."""
    p = _resolve_artifact_dir(artifact_dir) / "experiment_chemistry.parquet"
    return pd.read_parquet(p)


def load_experiment_metadata(artifact_dir: Path) -> pd.DataFrame:
    """Load locked ``experiment_metadata.parquet``."""
    p = _resolve_artifact_dir(artifact_dir) / "experiment_metadata.parquet"
    return pd.read_parquet(p)


def chemistry_row_unknown_rates(
    chemistry_df: pd.DataFrame,
    *,
    unk_token: str,
    unk_stressor_token: str,
) -> dict[str, float]:
    """Fraction of rows labeled ``unk_token`` (medium OOV) vs ``unk_stressor_token``."""
    n = max(len(chemistry_df), 1)
    return {
        "unk_medium_fraction": float((chemistry_df["canonical_id"] == unk_token).sum() / n),
        "unk_stressor_fraction": float((chemistry_df["canonical_id"] == unk_stressor_token).sum() / n),
    }


def apply_amount_scaler(amounts, scaler_artifact_path: Path) -> np.ndarray:
    """Apply log1p standardization with missing values mapped to 0."""
    payload = json.loads(Path(scaler_artifact_path).read_text())
    if payload.get("transform") != "log1p":
        raise ValueError(f"Unsupported transform: {payload.get('transform')}")
    mean = float(payload["mean"])
    std = float(payload["std"])
    std = std if std > 0 else 1.0

    arr = np.asarray(amounts, dtype=float)
    out = np.zeros_like(arr, dtype=np.float32)
    finite_mask = np.isfinite(arr)
    if np.any(finite_mask):
        clipped = np.clip(arr[finite_mask], 0.0, None)
        out[finite_mask] = ((np.log1p(clipped) - mean) / std).astype(np.float32)
    return out


def apply_metadata_encoding(values, vocab_artifact_path: Path) -> tuple[np.ndarray, float]:
    """Map metadata values to frozen vocab indices and report unknown rate."""
    payload = json.loads(Path(vocab_artifact_path).read_text())
    vocab_to_index = payload["vocab_to_index"]
    unk_idx = int(vocab_to_index[payload["unk_token"]])
    missing_idx = int(vocab_to_index[payload["missing_token"]])

    series = pd.Series(values)
    out = np.full(len(series), unk_idx, dtype=np.int64)
    unknown = 0
    for i, val in enumerate(series):
        if pd.isna(val):
            out[i] = missing_idx
            continue
        key = str(val)
        if key in vocab_to_index:
            out[i] = int(vocab_to_index[key])
        else:
            out[i] = unk_idx
            unknown += 1
    unknown_rate = float(unknown / max(len(series), 1))
    return out, unknown_rate


def apply_numeric_metadata_z(values, scaler_payload: dict[str, Any]) -> np.ndarray:
    """Apply frozen mean/std z-score; non-finite inputs become 0.0."""
    mean = float(scaler_payload["mean"])
    std = float(scaler_payload["std"]) or 1.0
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr, dtype=np.float32)
    m = np.isfinite(arr)
    out[m] = ((arr[m] - mean) / std).astype(np.float32)
    return out
