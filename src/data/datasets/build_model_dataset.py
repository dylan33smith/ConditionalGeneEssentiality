"""Build train/val/test datasets from canonical tables + preprocessing artifacts.

Stub — implement during Stage 1.
"""
from __future__ import annotations
from pathlib import Path


def build_dataset(split_protocol_id: str, artifact_dir: Path):
    """Build and cache a model-ready dataset for one split protocol.

    Must log:
      - scored_rowset_id / hash
      - inclusion/exclusion counters (pre-join, post-join, post-filter, post-split)
    """
    raise NotImplementedError("Implement during Stage 1")
