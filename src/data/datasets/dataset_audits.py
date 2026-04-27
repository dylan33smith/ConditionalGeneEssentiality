"""Dataset integrity and leakage audits.

These checks are required before any training run.
"""
from __future__ import annotations
import pandas as pd


def check_no_organism_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    org_col: str = "orgId",
) -> None:
    """Raise if any org_id appears in more than one partition."""
    train_orgs = set(train_df[org_col].unique())
    val_orgs = set(val_df[org_col].unique())
    test_orgs = set(test_df[org_col].unique())
    overlap_tv = train_orgs & val_orgs
    overlap_tt = train_orgs & test_orgs
    overlap_vt = val_orgs & test_orgs
    if overlap_tv or overlap_tt or overlap_vt:
        raise AssertionError(
            f"Organism overlap detected: train∩val={overlap_tv}, "
            f"train∩test={overlap_tt}, val∩test={overlap_vt}"
        )


def check_no_vocab_leakage(
    train_media: list[str],
    val_media: list[str],
    vocab: set[str],
) -> float:
    """Return unknown_rate for val media against train vocab.

    Raises if vocab contains any val-only media (leakage).
    """
    val_only = set(val_media) - set(train_media)
    leaked = val_only & vocab
    if leaked:
        raise AssertionError(f"Vocab leakage: val-only media in vocab: {leaked}")
    unknown_rate = len(set(val_media) - vocab) / max(len(set(val_media)), 1)
    return unknown_rate
