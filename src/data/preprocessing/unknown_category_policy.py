"""Unknown category handling policy.

All val/test categories not seen in train map to UNK index 0.
Unknown-category rate is logged per run.
"""
from __future__ import annotations
from typing import Sequence


def apply_unk_policy(
    categories: Sequence[str],
    known_vocab: set[str],
    unk_token: str = "<UNK>",
) -> tuple[list[str], float]:
    """Replace unseen categories with unk_token.

    Returns (mapped_categories, unknown_rate).
    """
    mapped = [c if c in known_vocab else unk_token for c in categories]
    unknown_rate = sum(1 for m in mapped if m == unk_token) / max(len(mapped), 1)
    return mapped, unknown_rate
