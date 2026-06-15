"""Per-condition chemistry feature vectors for the R-LOCK-4 chemistry baselines.

Maps each ranking `condition_key = (expDesc, media, temperature)` to a role-blind
425-dim multihot chemistry vector (over the S4 Canonical_ID vocab), by:
  1. computing the S4 `experiment_id` per assay (SHA256 of raw orgId/setName/
     seqindex/media — RAW media, before normalization),
  2. building/loading the experiment×canonical_id multihot (S4 artifact),
  3. averaging the multihot rows of all experiments sharing a condition_key.

IMPORTANT: `experiment_id` must be hashed on RAW media (matching how the S4
artifact `de21504134c84a6c` was built). The condition_key, by contrast, uses
NORMALIZED (lowercased) media for grouping. This module keeps the two separate.

These vectors feed `chemistry_nearest_condition_profile` (cold-condition null)
and `chemistry_knn_predict` (competitive baseline) in `ranking_eval.py`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing.build_experiment_chemistry import experiment_uid
from src.data.datasets.build_s5_dataset import build_or_load_experiment_multihot
from src.data.datasets.conditions import _normalize_string_keys, _condition_key

log = logging.getLogger(__name__)

CANONICAL_DIR = Path("data/derived/canonical/v0")
S4_ARTIFACT_DIR = Path("data_contract/preprocessing/de21504134c84a6c")
_ID_COLS = ["orgId", "setName", "seqindex", "media"]
_NEEDED = _ID_COLS + ["expDesc", "temperature"]


def load_condition_chemistry_features(
    orgs: list[str] | None = None,
    *,
    artifact_dir: Path = S4_ARTIFACT_DIR,
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Return {condition_key -> mean multihot vector (float32)}.

    Conditions whose experiments are absent from the S4 chemistry artifact
    (no media composition) are omitted — the chemistry baselines treat those
    as unscorable (NaN), preserving denominator parity downstream.
    """
    # Cache must be namespaced by org set — build_or_load_experiment_multihot
    # caches by directory only, so a shared dir returns a STALE matrix when the
    # org set changes (bug found in R1 smoke 2026-05-25).
    if cache_dir is None:
        import hashlib
        tag = ("full" if orgs is None
               else hashlib.sha1("|".join(sorted(orgs)).encode()).hexdigest()[:12])
        cache_dir = Path("artifacts/cache/ranking_condition_chem") / tag

    df = pd.read_parquet(CANONICAL_DIR / "fitness_experiment_long.parquet",
                         columns=_NEEDED).drop_duplicates()
    df = df.dropna(subset=_ID_COLS)
    if orgs is not None:
        df = df[df["orgId"].isin(orgs)]

    # experiment_id on RAW media (S4-compatible)
    key_df = df[_ID_COLS].drop_duplicates().copy()
    key_df["experiment_id"] = key_df.apply(experiment_uid, axis=1)
    df = df.merge(key_df, on=_ID_COLS, how="left", validate="many_to_one")

    # condition_key on NORMALIZED fields
    norm = _normalize_string_keys(df.copy())
    df["condition_key"] = _condition_key(norm).values

    target_ids = sorted(df["experiment_id"].unique())
    mat, exp_to_row, vocab_size = build_or_load_experiment_multihot(
        chemistry_parquet_path=artifact_dir / "experiment_chemistry.parquet",
        canonical_vocab_json_path=artifact_dir / "canonical_id_vocab.json",
        target_experiment_ids=target_ids,
        cache_dir=cache_dir,
    )

    cond_feats: dict[str, np.ndarray] = {}
    for ck, sub in df.groupby("condition_key"):
        rows = [exp_to_row[e] for e in sub["experiment_id"].unique() if e in exp_to_row]
        if not rows:
            continue
        vec = np.asarray(mat[rows].mean(axis=0)).ravel().astype(np.float32)
        if vec.sum() > 0:                    # skip all-zero (no chemistry) conditions
            cond_feats[ck] = vec
    log.info("built chemistry features for %d/%d condition_keys",
             len(cond_feats), df["condition_key"].nunique())
    return cond_feats
