"""src.ranking.losses — the ranking loss family.

Operates on padded [B, L] score/relevance tensors (scores = predicted fit;
relevance = max(0, -fit)). Pointwise (mse, huber) and ranking (ranknet,
lambdarank, listmle, approxndcg) objectives share one signature
`loss(scores, fit, w, mask)` and are dispatched via the LOSSES registry — the
substrate for the top-k objective. Add a new loss here and register it in LOSSES.
"""
from src.ranking.losses.family import (
    LOSSES,
    pointwise_mse,
    pointwise_huber,
    pairwise_ranknet,
    lambdarank,
    listmle,
    approxndcg,
    lambdarank_top5,
    approxndcg_top5,
)

__all__ = ["LOSSES", "pointwise_mse", "pointwise_huber", "pairwise_ranknet",
           "lambdarank", "listmle", "approxndcg",
           "lambdarank_top5", "approxndcg_top5"]
