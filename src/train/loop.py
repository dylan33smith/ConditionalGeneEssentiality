"""Training loop skeleton.

Stub — implement during Stage 0 / Tier 1.
"""
from __future__ import annotations
from pathlib import Path


def train(config: dict, output_dir: Path) -> dict:
    """Run training according to config; write artifacts to output_dir.

    Must log: git SHA, split id, preprocessing artifact id, config snapshot,
    seed, metrics, null-baseline deltas, unknown-category rates.
    """
    raise NotImplementedError("Implement during Stage 0 / Tier 1")
