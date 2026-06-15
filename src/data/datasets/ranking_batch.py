"""RankingBatch — data contract for the R-regime ranking task (R-LOCK-3).

Three sampler modes share a single model forward signature
`(gene_emb, cond_feat) -> scalar`; only the loss function differs:

    pointwise: each item is one (gene, condition, fit, weight) — for MSE/Huber.
    pairwise:  each item is (gene, cond_i, cond_j, sign, weight) — for margin / RankNet.
    listwise:  each item is (gene, [cond_idx], [fit], mask, weight) — for ListMLE / SoftRank.

Pointwise is the locked default for R1, R2 (per ARCHITECTURE.md §2.3); pairwise and
listwise are tested explicitly in R-LOSS.

Replicate handling (per ARCHITECTURE.md §2.3):
  - train: per-replicate rows (each replicate is a noisy observation of the
    same target; don't collapse).
  - val:   mean-pool replicates within (orgId, gene_key, condition_key).

Sign convention (per ARCHITECTURE.md §2.3):
  - model predicts `fit` (low → essential)
  - pairwise margin sign = sign(fit_i − fit_j)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch
from torch.utils.data import Sampler

log = logging.getLogger(__name__)

SamplerMode = Literal["pointwise", "pairwise", "listwise"]


# ---------------------------------------------------------------------------
# Batch dataclass
# ---------------------------------------------------------------------------

@dataclass
class RankingBatch:
    """Container for one batch under any of the three sampler modes.

    All tensor fields live on CPU; the train loop is responsible for moving
    them to the device of choice.
    """
    mode: SamplerMode
    gene_idx: torch.LongTensor          # pointwise: (B,);   pairwise: (B, 2);  listwise: (B, L)
    cond_idx: torch.LongTensor          # same shape as gene_idx
    fit: torch.FloatTensor              # same shape; raw target values
    weight: torch.FloatTensor           # (B,) per-gene weight from R-LOCK-1 (w_g)
    sign: torch.LongTensor | None = None   # pairwise only; (B,) ∈ {-1, +1}
    mask: torch.BoolTensor | None = None   # listwise only; (B, L) True = valid slot
    # Optional bookkeeping (kept on CPU, used for evaluation / debugging)
    gene_key: list[str] | None = None
    condition_key: list[str] | None = None

    def __post_init__(self):
        if self.mode == "pairwise" and self.sign is None:
            raise ValueError("pairwise batch requires `sign` tensor")
        if self.mode == "listwise" and self.mask is None:
            raise ValueError("listwise batch requires `mask` tensor")


# ---------------------------------------------------------------------------
# Sampler factory + implementations
# ---------------------------------------------------------------------------

def build_sampler(mode: SamplerMode, **kwargs) -> Sampler:
    """Factory. Sampler kwargs vary by mode; see each class."""
    if mode == "pointwise":
        return PointwiseSampler(**kwargs)
    if mode == "pairwise":
        return PairwiseSampler(**kwargs)
    if mode == "listwise":
        return ListwiseSampler(**kwargs)
    raise ValueError(f"Unknown sampler mode {mode!r}")


class PointwiseSampler(Sampler[int]):
    """Yield row indices in (optionally shuffled) order.

    Compatible with the existing T-tier training loop — pointwise mode is
    deliberately a no-op transform of the row layout.
    """
    def __init__(self, n_rows: int, *, seed: int = 0, shuffle: bool = True):
        self.n_rows = int(n_rows)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self._epoch = 0

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            order = torch.randperm(self.n_rows, generator=g).tolist()
        else:
            order = range(self.n_rows)
        self._epoch += 1
        yield from order

    def __len__(self) -> int:
        return self.n_rows


class PairwiseSampler(Sampler[tuple[int, int]]):
    """Per gene, sample `pairs_per_gene` ordered condition pairs per epoch.

    Yields tuples (row_idx_i, row_idx_j) such that gene[i] == gene[j] and
    cond[i] != cond[j]. The downstream collate computes
    sign = sign(fit_i − fit_j).
    """
    def __init__(self, gene_idx: np.ndarray, *, pairs_per_gene: int = 8,
                 seed: int = 0, min_gene_size: int = 2):
        self.gene_idx = np.asarray(gene_idx)
        self.pairs_per_gene = int(pairs_per_gene)
        self.seed = int(seed)
        self.min_gene_size = int(min_gene_size)
        self._epoch = 0
        # Pre-group row indices by gene
        order = np.argsort(self.gene_idx, kind="stable")
        sorted_genes = self.gene_idx[order]
        cuts = np.searchsorted(sorted_genes, np.unique(sorted_genes), side="left")
        self._groups: list[np.ndarray] = []
        for i, start in enumerate(cuts):
            end = cuts[i + 1] if i + 1 < len(cuts) else len(order)
            g = order[start:end]
            if len(g) >= self.min_gene_size:
                self._groups.append(g)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        for grp in self._groups:
            n = len(grp)
            for _ in range(self.pairs_per_gene):
                i, j = rng.choice(n, size=2, replace=False)
                yield (int(grp[i]), int(grp[j]))
        self._epoch += 1

    def __len__(self) -> int:
        return len(self._groups) * self.pairs_per_gene


class ListwiseSampler(Sampler[np.ndarray]):
    """Per gene, yield its full set of row indices (padded to max length).

    Returned values are integer arrays of length L_max with -1 padding;
    the collate function turns them into a (B, L) batch with a mask.
    """
    def __init__(self, gene_idx: np.ndarray, *, seed: int = 0,
                 min_gene_size: int = 2, max_list_len: int | None = None):
        self.gene_idx = np.asarray(gene_idx)
        self.seed = int(seed)
        self.min_gene_size = int(min_gene_size)
        self.max_list_len = max_list_len
        self._epoch = 0
        order = np.argsort(self.gene_idx, kind="stable")
        sorted_genes = self.gene_idx[order]
        cuts = np.searchsorted(sorted_genes, np.unique(sorted_genes), side="left")
        self._groups: list[np.ndarray] = []
        for i, start in enumerate(cuts):
            end = cuts[i + 1] if i + 1 < len(cuts) else len(order)
            g = order[start:end]
            if len(g) >= self.min_gene_size:
                self._groups.append(g)
        L = max(len(g) for g in self._groups) if self._groups else 0
        self.L_max = min(L, self.max_list_len) if self.max_list_len else L

    def __iter__(self) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self.seed + self._epoch)
        gene_order = rng.permutation(len(self._groups))
        for gi in gene_order:
            grp = self._groups[gi]
            if self.max_list_len and len(grp) > self.max_list_len:
                grp = rng.choice(grp, size=self.max_list_len, replace=False)
            yield np.asarray(grp, dtype=np.int64)
        self._epoch += 1

    def __len__(self) -> int:
        return len(self._groups)


# ---------------------------------------------------------------------------
# Collate functions — turn sampled row-index iterables into RankingBatch
# ---------------------------------------------------------------------------

def collate_pointwise(rows: list[int],
                      *, gene_idx: np.ndarray, cond_idx: np.ndarray,
                      fit: np.ndarray, weight: np.ndarray,
                      gene_key: list[str] | None = None,
                      condition_key: list[str] | None = None) -> RankingBatch:
    idx = np.asarray(rows, dtype=np.int64)
    return RankingBatch(
        mode="pointwise",
        gene_idx=torch.from_numpy(gene_idx[idx].astype(np.int64)),
        cond_idx=torch.from_numpy(cond_idx[idx].astype(np.int64)),
        fit=torch.from_numpy(fit[idx].astype(np.float32)),
        weight=torch.from_numpy(weight[idx].astype(np.float32)),
        gene_key=[gene_key[i] for i in idx] if gene_key is not None else None,
        condition_key=[condition_key[i] for i in idx] if condition_key is not None else None,
    )


def collate_pairwise(pairs: list[tuple[int, int]],
                     *, gene_idx: np.ndarray, cond_idx: np.ndarray,
                     fit: np.ndarray, weight: np.ndarray) -> RankingBatch:
    pairs_arr = np.asarray(pairs, dtype=np.int64)            # (B, 2)
    i = pairs_arr[:, 0]
    j = pairs_arr[:, 1]
    gene_pair = np.stack([gene_idx[i], gene_idx[j]], axis=1).astype(np.int64)
    cond_pair = np.stack([cond_idx[i], cond_idx[j]], axis=1).astype(np.int64)
    fit_pair = np.stack([fit[i], fit[j]], axis=1).astype(np.float32)
    # Sign convention: low fit = essential; model is rewarded for matching
    # the observed ordering. sign = +1 if fit_i > fit_j (i.e. i less essential)
    sign = np.sign(fit_pair[:, 0] - fit_pair[:, 1]).astype(np.int64)
    # Use the i-side weight (both items belong to the same gene by construction)
    w = weight[i].astype(np.float32)
    return RankingBatch(
        mode="pairwise",
        gene_idx=torch.from_numpy(gene_pair),
        cond_idx=torch.from_numpy(cond_pair),
        fit=torch.from_numpy(fit_pair),
        weight=torch.from_numpy(w),
        sign=torch.from_numpy(sign),
    )


def collate_listwise(lists: list[np.ndarray],
                     *, gene_idx: np.ndarray, cond_idx: np.ndarray,
                     fit: np.ndarray, weight: np.ndarray) -> RankingBatch:
    L_max = max(len(g) for g in lists)
    B = len(lists)
    g_out = np.full((B, L_max), -1, dtype=np.int64)
    c_out = np.full((B, L_max), -1, dtype=np.int64)
    f_out = np.zeros((B, L_max), dtype=np.float32)
    m_out = np.zeros((B, L_max), dtype=bool)
    w_out = np.zeros((B,), dtype=np.float32)
    for k, idx in enumerate(lists):
        n = len(idx)
        g_out[k, :n] = gene_idx[idx]
        c_out[k, :n] = cond_idx[idx]
        f_out[k, :n] = fit[idx]
        m_out[k, :n] = True
        # All slots in a list belong to the same gene → take any slot's weight
        w_out[k] = weight[idx[0]]
    return RankingBatch(
        mode="listwise",
        gene_idx=torch.from_numpy(g_out),
        cond_idx=torch.from_numpy(c_out),
        fit=torch.from_numpy(f_out),
        weight=torch.from_numpy(w_out),
        mask=torch.from_numpy(m_out),
    )


# ---------------------------------------------------------------------------
# Eligibility-policy hashing (used in run manifest v2)
# ---------------------------------------------------------------------------

def policy_hash(path: Path | str) -> str:
    """sha256 of a policy YAML body (whole file bytes). Stable across formatting."""
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
