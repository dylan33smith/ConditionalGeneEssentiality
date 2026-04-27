"""Apply frozen condition preprocessing artifacts to produce feature arrays.

Stub — implement during Stage 1.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def build_multihot_matrix(
    media_names: list[str],
    vocab_artifact_path: Path,
) -> tuple[np.ndarray, float]:
    """Build (n_media, n_components) multihot matrix.

    Returns (matrix, unknown_rate).
    Unseen components map to UNK (index 0).
    """
    raise NotImplementedError("Implement during Stage 1")
