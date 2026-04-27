"""Fast epoch iteration from pre-materialized numpy arrays.

Use MaterializedSplit after running materialize_training_data.py.
Replaces the streaming Parquet pipeline with pure numpy slicing + tensor ops:

  - embedding lookup: emb_matrix[gene_emb_idx[batch]]  (single vectorized gather)
  - condition fields: numpy slice -> torch.as_tensor  (no Python loops)
  - shuffling: np.random.permutation on row indices (no buffer fill/drain)

Expected speedup over the streaming pipeline: 10–50×.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch


class MaterializedSplit:
    """Loads pre-materialized split arrays and yields fast training/val batches."""

    def __init__(self, split_dir: Path, device: torch.device) -> None:
        split_dir = Path(split_dir)
        meta_path = split_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"MaterializedSplit: missing {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        self.n_rows: int = meta["n_rows"]
        self.k_cat: int = meta["k_cat"]
        self.n_cont: int = meta["n_cont"]
        self.gene_dim: int = meta["gene_dim"]
        self.arm: str = meta["arm"]
        self.meta: dict = meta
        self.device = device

        # Embedding matrix on device for fast vectorized gather
        emb_np = np.load(split_dir / "emb_matrix.npy")  # (N_slots, gene_dim)
        self._emb_matrix = torch.as_tensor(emb_np, dtype=torch.float32).to(device)

        # Row arrays stay on CPU as numpy; converted to tensors per batch
        self._gene_emb_idx = np.load(split_dir / "gene_emb_idx.npy")   # (N,) int32
        self._fit = np.load(split_dir / "fit.npy")                      # (N,) float32
        self._weight = np.load(split_dir / "weight.npy")                # (N,) float32
        self._ce_cat = np.load(split_dir / "ce_cat.npy")               # (N, k_cat) int32
        self._ce_cont = np.load(split_dir / "ce_cont.npy")             # (N, n_cont) float32
        gk_path = split_dir / "gene_keys.npy"
        self._gene_keys: np.ndarray | None = (
            np.load(gk_path, allow_pickle=True) if gk_path.is_file() else None
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def iter_train_batches(
        self,
        *,
        batch_size: int,
        seed: int,
        epoch: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield (x_gene, cat, cont, y, w) batches in shuffled order.

        Drops the last partial batch (same behaviour as shuffled_training_batches).
        """
        rng = np.random.default_rng(seed + epoch * 10_000)
        idx = rng.permutation(self.n_rows)
        for start in range(0, self.n_rows - batch_size + 1, batch_size):
            batch_idx = idx[start : start + batch_size]
            yield self._make_batch(batch_idx)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def iter_val_batches(
        self,
        *,
        batch_size: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]:
        """Yield (x_gene, cat, cont, y, w, gene_keys) in sequential order."""
        for start in range(0, self.n_rows, batch_size):
            end = min(start + batch_size, self.n_rows)
            batch_idx = np.arange(start, end, dtype=np.int64)
            x_g, cat, cont, y, w = self._make_batch(batch_idx)
            if self._gene_keys is not None:
                gks: list[str] = self._gene_keys[batch_idx].tolist()
            else:
                gks = [""] * (end - start)
            yield x_g, cat, cont, y, w, gks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_batch(
        self,
        batch_idx: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = self.device
        emb_idx = torch.as_tensor(
            self._gene_emb_idx[batch_idx].astype(np.int64), dtype=torch.long, device=dev
        )
        x_gene = self._emb_matrix[emb_idx]  # (B, gene_dim) — vectorized GPU gather
        cat = torch.as_tensor(
            self._ce_cat[batch_idx], dtype=torch.long, device=dev
        )
        cont = torch.as_tensor(
            self._ce_cont[batch_idx], dtype=torch.float32, device=dev
        )
        y = torch.as_tensor(self._fit[batch_idx], dtype=torch.float32, device=dev)
        w = torch.as_tensor(self._weight[batch_idx], dtype=torch.float32, device=dev)
        return x_gene, cat, cont, y, w
