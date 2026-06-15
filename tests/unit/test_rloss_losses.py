"""Unit tests for the R-LOSS ranking losses — sign conventions + optimization."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from src.experiments.rloss._losses import LOSSES
from src.ranking.eval import ndcg_at_k

# Full-list losses optimize the whole ordering; NDCG losses are top-focused
# (they only rank the stressors, relevance = max(0,-fit) > 0).
FULL_LIST = {"pointwise_mse", "pointwise_huber", "pairwise_ranknet", "listmle"}
TOP_FOCUSED = {"lambdarank", "approxndcg"}


def _batch():
    # 3 genes x 5 conditions; gene 2 has a padded slot (mask False)
    fit = torch.tensor([[-3., -1., 0., 1., 2.],
                        [2., 1., 0., -1., -2.],
                        [-2., -1., 0., 1., 0.]])
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[2, 4] = False
    w = torch.ones(3)
    return fit, mask, w


@pytest.mark.parametrize("name", list(LOSSES.keys()))
def test_perfect_beats_inverted(name):
    """A score perfectly aligned with fit (score=fit) must have LOWER loss than
    the inverted score (score=-fit) — confirms the sign convention."""
    fit, mask, w = _batch()
    loss = LOSSES[name]
    perfect = loss(fit.clone(), fit, w, mask)            # score == fit
    inverted = loss(-fit.clone(), fit, w, mask)          # score anti-aligned
    assert perfect.item() < inverted.item(), f"{name}: perfect {perfect.item()} !< inverted {inverted.item()}"


@pytest.mark.parametrize("name", list(LOSSES.keys()))
def test_optimizing_recovers_ordering(name):
    """Gradient descent on free scores should recover the correct ranking:
    full-list Spearman for full-list losses; NDCG@k recovery for the top-focused
    NDCG losses (which correctly ignore the order among non-stressors)."""
    torch.manual_seed(0)
    fit, mask, w = _batch()
    scores = torch.randn(3, 5, requires_grad=True)
    opt = torch.optim.Adam([scores], lr=0.2)
    loss_fn = LOSSES[name]
    for _ in range(400):
        opt.zero_grad()
        l = loss_fn(scores, fit, w, mask)
        l.backward()
        opt.step()
    sc = scores.detach()
    if name in FULL_LIST:
        rs = [spearmanr(fit[b][mask[b]].numpy(), sc[b][mask[b]].numpy())[0] for b in range(3)]
        assert min(rs) > 0.8, f"{name}: per-gene Spearman {rs} not all > 0.8"
    else:  # TOP_FOCUSED: stressors ranked correctly => NDCG@3 near 1
        nd = []
        for b in range(3):
            m = mask[b].numpy()
            ft, fp = fit[b].numpy()[m], sc[b].numpy()[m]
            if np.any(np.maximum(0, -ft) > 0):
                nd.append(ndcg_at_k(ft, fp, k=3))
        assert min(nd) > 0.9, f"{name}: per-gene NDCG@3 {nd} not all > 0.9"


def test_mse_zero_at_perfect():
    fit, mask, w = _batch()
    assert LOSSES["pointwise_mse"](fit.clone(), fit, w, mask).item() == pytest.approx(0.0, abs=1e-6)


def test_mask_ignores_padding():
    """Changing a padded slot's score must not change any loss."""
    fit, mask, w = _batch()
    for name, loss in LOSSES.items():
        s1 = torch.randn(3, 5)
        s2 = s1.clone(); s2[2, 4] += 100.0      # padded slot
        l1 = loss(s1, fit, w, mask).item()
        l2 = loss(s2, fit, w, mask).item()
        assert l1 == pytest.approx(l2, abs=1e-5), f"{name} leaked padded slot"


def test_weight_scales_contribution():
    """A gene with weight 0 must not affect the loss."""
    fit, mask, _ = _batch()
    loss = LOSSES["pointwise_mse"]
    scores = torch.randn(3, 5)
    w_all = torch.ones(3)
    w_drop2 = torch.tensor([1.0, 1.0, 0.0])
    # loss over genes 0,1 only should equal the w_drop2 loss
    l_drop = loss(scores, fit, w_drop2, mask).item()
    l_first2 = loss(scores[:2], fit[:2], torch.ones(2), mask[:2]).item()
    assert l_drop == pytest.approx(l_first2, abs=1e-5)
