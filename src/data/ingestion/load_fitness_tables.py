"""Load canonical fitness tables from Parquet.

Stub — implement during Stage 0.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd


CANONICAL_V0 = Path("data/derived/canonical/v0")


def load_fitness_experiment_long(path: Path | None = None) -> pd.DataFrame:
    """Load fitness_experiment_long.parquet."""
    path = path or CANONICAL_V0 / "fitness_experiment_long.parquet"
    return pd.read_parquet(path)


def load_experiments(path: Path | None = None) -> pd.DataFrame:
    path = path or CANONICAL_V0 / "experiments.parquet"
    return pd.read_parquet(path)


def load_media_master(path: Path | None = None) -> pd.DataFrame:
    path = path or CANONICAL_V0 / "media_master.parquet"
    return pd.read_parquet(path)


def load_media_components_long(path: Path | None = None) -> pd.DataFrame:
    path = path or CANONICAL_V0 / "media_components_long.parquet"
    return pd.read_parquet(path)
