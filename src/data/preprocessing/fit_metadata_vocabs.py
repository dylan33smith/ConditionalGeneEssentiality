"""Train-only metadata vocabulary fitters."""
from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


def fit_metadata_field_vocab(
    train_values,
    *,
    field_name: str,
    prevalence_threshold: float,
    unk_token: str = "<UNK>",
    missing_token: str = "<MISSING>",
    unk_index: int = 0,
    missing_index: int = 1,
) -> dict[str, Any]:
    """Fit one metadata field vocab with fixed UNK/MISSING indices."""
    series = pd.Series(train_values)
    non_missing = series.dropna().astype(str)
    n_train = int(len(series))
    min_count = max(1, int(prevalence_threshold * max(n_train, 1)))
    counts = Counter(non_missing.tolist())
    kept = sorted([k for k, c in counts.items() if c >= min_count])
    vocab = [unk_token, missing_token] + kept
    if vocab[unk_index] != unk_token or vocab[missing_index] != missing_token:
        raise ValueError("UNK/MISSING index policy violated.")
    vocab_to_index = {v: i for i, v in enumerate(vocab)}
    return {
        "field_name": field_name,
        "vocab": vocab,
        "vocab_to_index": vocab_to_index,
        "unk_token": unk_token,
        "missing_token": missing_token,
        "n_train": n_train,
        "n_non_missing": int(len(non_missing)),
        "n_unique_pretrim": int(len(counts)),
        "n_unique_posttrim": int(len(kept)),
        "prevalence_threshold": float(prevalence_threshold),
        "min_count_required": int(min_count),
    }
