"""Canonical condition identity for the ranking task (data layer).

Defines what a "condition" IS — the `(expDesc, media, temperature)` key — plus
the string normalization and fitness loader that go with it. Lives in the data
layer because both the data modules (condition chemistry, the ranking split) and
the ranking pipeline consume it; keeping it here avoids any dependency on the
experiment packages.

Extracted verbatim from src/experiments/r0/analyses.py during the ranking-branch
cleanup (single source of truth; r0 now imports these).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

CANONICAL_DIR = Path("data/derived/canonical/v0")

_FITNESS_COLS = [
    "orgId", "locusId", "expName", "expDesc", "fit",
    "media", "gene_key",
    "expGroup", "condition_1", "condition_2",
    "temperature",
]

# Columns where we collapse case/whitespace variants on load. These string
# fields are used as identifiers — e.g. `(expDesc, media, temperature)` is
# the condition key — so 'mg/ml' vs 'mg/mL' would split one condition in two.
# Audit on 2026-05-24 found 5 such collisions across the dataset (negligible
# impact but worth fixing once at the source).
_STRING_KEY_COLS = ("expDesc", "media", "expGroup")


def _normalize_string_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + collapse whitespace on identifier columns. NaN-safe."""
    for col in _STRING_KEY_COLS:
        if col in df.columns:
            df[col] = (df[col].astype("string")
                       .str.strip().str.replace(r"\s+", " ", regex=True)
                       .str.lower())
    return df


def _condition_key(df: pd.DataFrame) -> pd.Series:
    """Canonical condition key for the R-regime ranking task.

    Defined as `(expDesc, media, temperature)`. Temperature is included
    because R0 audit (2026-05-24) found 53 of 2246 (2.4%) `(expDesc, media)`
    groups actually contained assays at distinct temperatures — which are
    different conditions, not replicates. pH is excluded (53% null
    overall; never varies within a `(expDesc, media)` group); aerobic is
    excluded (97% populated but never varies within a group → no-op).
    """
    temp_str = df["temperature"].astype("string").fillna("NA")
    return (df["expDesc"].astype("string").fillna("NA") + "|"
            + df["media"].astype("string").fillna("NA") + "|" + temp_str)


def load_fitness(cols: Iterable[str] | None = None) -> pd.DataFrame:
    """Load canonical long-form fitness with one normalization step at load:
    case/whitespace folding on identifier columns (`expDesc`, `media`,
    `expGroup`). See `_normalize_string_keys` for rationale.
    """
    cols = list(cols) if cols is not None else _FITNESS_COLS
    df = pd.read_parquet(CANONICAL_DIR / "fitness_experiment_long.parquet",
                         columns=cols)
    df = _normalize_string_keys(df)
    return df
