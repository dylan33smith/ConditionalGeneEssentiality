"""Ranking loss family for R-LOSS.

CONVENTIONS (anchored to the eval harness — do not change without updating eval):
  - The model outputs `score = predicted fit`. Eval ranks conditions by ASCENDING
    score (lowest fit = most essential = rank 1) and computes within-gene Spearman
    against true fit + NDCG with relevance = max(0, -fit).
  - So we want `score` to be MONOTONIC INCREASING in true fit: for two conditions
    with fit_i < fit_j, we want score_i < score_j.
  - Internally, ranking losses use the "ranking score" r = -score (higher r =
    more essential = better rank for the stressor-retrieval objective).
  - relevance(fit) = max(0, -fit): a stressor reduces fitness, so more-negative
    fit = higher gain.

All losses take padded batch tensors:
  scores [B, L]  — model output (predicted fit), padded
  fit    [B, L]  — true fit, padded
  w      [B]     — per-gene weight (R-LOCK-1 w_g)
  mask   [B, L]  — True where the slot is a real condition
Return a scalar loss (lower = better). Padded slots never contribute.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_NEG_INF = -1e9


def _relevance(fit: torch.Tensor) -> torch.Tensor:
    return torch.clamp(-fit, min=0.0)


# ---------------------------------------------------------------------------
# Pointwise
# ---------------------------------------------------------------------------

def pointwise_mse(scores, fit, w, mask):
    # PER-ROW w_g weighting (matches R-LOCK-1 / R1: each (gene,condition) row is
    # weighted by its gene's w_g; genes with more conditions contribute more).
    wexp = w.unsqueeze(1) * mask.float()
    se = (scores - fit) ** 2
    return (wexp * se).sum() / wexp.sum().clamp_min(1e-6)


def pointwise_huber(scores, fit, w, mask, delta: float = 1.0):
    wexp = w.unsqueeze(1) * mask.float()
    err = (scores - fit).abs()
    huber = torch.where(err <= delta, 0.5 * err ** 2, delta * (err - 0.5 * delta))
    return (wexp * huber).sum() / wexp.sum().clamp_min(1e-6)


# ---------------------------------------------------------------------------
# Pairwise helpers — build per-gene pair tensors
# ---------------------------------------------------------------------------

def _pair_terms(scores, fit, mask):
    """Return (s_diff, valid, rel) for all ordered pairs (i, j) per gene.

    s_diff[b,i,j] = score_i - score_j.  valid[b,i,j] True where i,j both real and
    fit_i < fit_j (i is strictly MORE essential than j — the pair has a defined
    target: we want score_i < score_j, i.e. s_diff < 0).
    """
    B, L = scores.shape
    si = scores.unsqueeze(2)            # [B, L, 1]
    sj = scores.unsqueeze(1)            # [B, 1, L]
    s_diff = si - sj                    # score_i - score_j
    fi = fit.unsqueeze(2)
    fj = fit.unsqueeze(1)
    mm = (mask.unsqueeze(2) & mask.unsqueeze(1))
    valid = mm & (fi < fj)              # i strictly more essential than j
    return s_diff, valid, (fi, fj)


def pairwise_ranknet(scores, fit, w, mask):
    """RankNet: for pairs with fit_i < fit_j, penalize score_i >= score_j.

    loss_pair = softplus(score_i - score_j); want score_i < score_j.
    """
    s_diff, valid, _ = _pair_terms(scores, fit, mask)
    loss_pair = F.softplus(s_diff) * valid.float()
    per_gene = loss_pair.sum((1, 2)) / valid.float().sum((1, 2)).clamp_min(1.0)
    return (w * per_gene).sum() / w.sum().clamp_min(1e-6)


def _dcg_discount(ranks):
    return 1.0 / torch.log2(ranks.float() + 2.0)   # ranks 0-indexed


def lambdarank(scores, fit, w, mask, sigma: float = 1.0):
    """LambdaRank: RankNet pairwise loss weighted by |ΔNDCG| from swapping i,j.

    Emphasizes pairs whose swap most changes NDCG (i.e. near the top). Ranks are
    derived from the CURRENT model ordering (ascending score = most essential
    first); gains from relevance = max(0,-fit).
    """
    B, L = scores.shape
    s_diff, valid, _ = _pair_terms(scores, fit, mask)
    rel = _relevance(fit)
    gain = (2.0 ** rel - 1.0)                                  # [B, L]

    # current ranks: ascending score = rank 0 first (most essential)
    masked_score = scores.masked_fill(~mask, _NEG_INF * -1)    # push padded to the end
    order = torch.argsort(masked_score, dim=1)                 # ascending
    ranks = torch.empty_like(order)
    ar = torch.arange(L, device=scores.device).expand(B, L)
    ranks.scatter_(1, order, ar)                               # rank of each item
    disc = _dcg_discount(ranks) * mask.float()                 # [B, L]

    # ideal DCG (sort by relevance desc)
    ideal_gain, _ = torch.sort(gain.masked_fill(~mask, 0.0), dim=1, descending=True)
    ideal_disc = _dcg_discount(torch.arange(L, device=scores.device).expand(B, L))
    idcg = (ideal_gain * ideal_disc).sum(1).clamp_min(1e-6)    # [B]

    gi = gain.unsqueeze(2); gj = gain.unsqueeze(1)
    di = disc.unsqueeze(2); dj = disc.unsqueeze(1)
    delta_ndcg = ((gi - gj) * (di - dj)).abs() / idcg.view(B, 1, 1)   # [B, L, L]

    loss_pair = F.softplus(s_diff * sigma) * delta_ndcg * valid.float()
    per_gene = loss_pair.sum((1, 2)) / valid.float().sum((1, 2)).clamp_min(1.0)
    return (w * per_gene).sum() / w.sum().clamp_min(1e-6)


# ---------------------------------------------------------------------------
# Listwise
# ---------------------------------------------------------------------------

def listmle(scores, fit, w, mask):
    """ListMLE: negative Plackett-Luce log-likelihood of the TRUE order.

    True order = ascending fit (most essential first). Ranking score r = -score
    (higher r = earlier in the order). We sort positions by true fit and compute
    the PL likelihood that this order is produced by r.
    """
    B, L = scores.shape
    r = -scores
    # order positions by ascending fit (most essential first); padded -> last
    fit_key = fit.masked_fill(~mask, 1e9)
    order = torch.argsort(fit_key, dim=1)                      # [B, L] indices in true order
    r_sorted = torch.gather(r, 1, order)
    mask_sorted = torch.gather(mask, 1, order)
    r_sorted = r_sorted.masked_fill(~mask_sorted, _NEG_INF)
    # PL: sum_k [ r_k - logsumexp(r_k..r_end) ] over valid positions
    # reverse-cumulative logsumexp
    rev = torch.flip(r_sorted, dims=[1])
    rev_lse = torch.logcumsumexp(rev, dim=1)
    lse_suffix = torch.flip(rev_lse, dims=[1])                 # logsumexp(r_k..end)
    log_probs = (r_sorted - lse_suffix) * mask_sorted.float()
    per_gene = -log_probs.sum(1) / mask_sorted.float().sum(1).clamp_min(1.0)
    return (w * per_gene).sum() / w.sum().clamp_min(1e-6)


def approxndcg(scores, fit, w, mask, temp: float = 0.5):
    """ApproxNDCG: maximize a smooth approximation of NDCG.

    Approximate rank of item i (ascending score = most essential first):
      rank_i ≈ sum_{j != i} sigmoid((score_i - score_j)/temp)   (# items ranked
      before i = items with smaller score). Then NDCG with smooth discounts.
    Loss = 1 - approxNDCG (so lower = better).
    """
    B, L = scores.shape
    rel = _relevance(fit)
    gain = (2.0 ** rel - 1.0) * mask.float()

    si = scores.unsqueeze(2); sj = scores.unsqueeze(1)
    mm = (mask.unsqueeze(2) & mask.unsqueeze(1)).float()
    # P(j before i) = sigmoid((score_i - score_j)/temp): j earlier if smaller score
    before = torch.sigmoid((si - sj) / temp) * mm
    eye = torch.eye(L, device=scores.device).unsqueeze(0)
    approx_rank = (before * (1 - eye)).sum(2)                  # [B, L], 0-indexed approx
    disc = 1.0 / torch.log2(approx_rank + 2.0)
    dcg = (gain * disc * mask.float()).sum(1)

    ideal_gain, _ = torch.sort(gain, dim=1, descending=True)
    ideal_disc = 1.0 / torch.log2(torch.arange(L, device=scores.device).float() + 2.0)
    idcg = (ideal_gain * ideal_disc).sum(1).clamp_min(1e-6)
    ndcg = dcg / idcg
    per_gene = 1.0 - ndcg
    return (w * per_gene).sum() / w.sum().clamp_min(1e-6)


LOSSES = {
    "pointwise_mse": pointwise_mse,
    "pointwise_huber": pointwise_huber,
    "pairwise_ranknet": pairwise_ranknet,
    "lambdarank": lambdarank,
    "listmle": listmle,
    "approxndcg": approxndcg,
}
