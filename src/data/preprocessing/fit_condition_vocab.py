"""Fit condition vocabulary on train-only data and persist artifact.

Stub — implement during Stage 1.
Key requirement: vocab fitted ONLY on train rows; no val/test leakage.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable


def fit_canonical_id_vocab(
    train_media_names: Iterable[str],
    components_df,          # DataFrame from load_media_components_ml
    output_path: Path,
) -> dict:
    """Fit and save canonical_id_vocab artifact.

    Returns {'canonical_id_vocab': [...], 'canonical_id_to_index': {...}}.
    Only components present in train media are included (train-only).
    """
    raise NotImplementedError("Implement during Stage 1")
