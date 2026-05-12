"""Dataset builders for S5 row-quality policy comparisons.

S5 uses a fixed model input:
  - gene embedding (ProteomeLM, layer8)
  - role-blind chemistry multihot (one bit per canonical_id)

This module prepares reusable experiment-level chemistry features and
row-level index arrays for train/val/test partitions.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from src.data.preprocessing.build_experiment_chemistry import experiment_uid


log = logging.getLogger(__name__)


def _load_concatenated_embeddings(
    orgs: list[str], embedding_dir: Path
) -> tuple[np.ndarray, dict[str, int]]:
    """Concatenate and L2-normalize embedding bundles.

    Returns:
        (embedding_matrix [n_genes, d], gene_key_to_index)
    """
    matrices: list[np.ndarray] = []
    idx_to_key: list[str] = []
    for org in sorted(set(orgs)):
        pt = embedding_dir / f"{org}_proteomelm.pt"
        if not pt.exists():
            log.warning("Missing embedding bundle for org=%s", org)
            continue
        bundle = torch.load(pt, map_location="cpu", weights_only=False)
        emb = bundle["embeddings"].to(torch.float32).numpy()
        matrices.append(emb)
        idx_to_key.extend(bundle["group_labels"])
    if not matrices:
        return np.zeros((0, 1152), dtype=np.float32), {}
    big = np.concatenate(matrices, axis=0)
    norms = np.maximum(np.linalg.norm(big, axis=1, keepdims=True), 1e-9)
    big = (big / norms).astype(np.float32)
    key_to_idx = {k: i for i, k in enumerate(idx_to_key)}
    return big, key_to_idx


def add_experiment_id(fitness_df: pd.DataFrame) -> pd.DataFrame:
    """Attach S4-compatible experiment_id to fitness rows.

    To avoid hashing every fitness row directly, hash only unique
    (orgId, setName, seqindex, media) keys and merge back.
    """
    required = ["orgId", "setName", "seqindex", "media"]
    missing = [c for c in required if c not in fitness_df.columns]
    if missing:
        raise ValueError(f"fitness_df missing experiment-id columns: {missing}")
    key_df = fitness_df[required].drop_duplicates().copy()
    key_df["experiment_id"] = key_df.apply(experiment_uid, axis=1)
    out = fitness_df.merge(key_df, on=required, how="left", validate="many_to_one")
    if out["experiment_id"].isna().any():
        raise ValueError("Failed to assign experiment_id to all fitness rows")
    return out


def _load_vocab_map(canonical_vocab_json_path: Path) -> dict[str, int]:
    payload = json.loads(canonical_vocab_json_path.read_text())
    return {str(k): int(v) for k, v in payload["canonical_id_to_index"].items()}


def build_or_load_experiment_multihot(
    *,
    chemistry_parquet_path: Path,
    canonical_vocab_json_path: Path,
    target_experiment_ids: list[str],
    cache_dir: Path,
    sparse_cache_filename: str = "multihot.npz",
) -> tuple[sp.csr_matrix, dict[str, int], int]:
    """Build role-blind multihot vectors keyed by experiment_id.

    Returns:
        (csr_matrix [n_experiments, vocab_size], experiment_id_to_row, vocab_size)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    sparse_path = cache_dir / sparse_cache_filename
    row_ids_path = cache_dir / "multihot_experiment_ids.json"

    vocab_map = _load_vocab_map(canonical_vocab_json_path)
    vocab_size = max(vocab_map.values()) + 1 if vocab_map else 0

    if sparse_path.exists() and row_ids_path.exists():
        mat = sp.load_npz(sparse_path).tocsr()
        row_ids = json.loads(row_ids_path.read_text())
        exp_to_row = {str(eid): i for i, eid in enumerate(row_ids)}
        return mat, exp_to_row, vocab_size

    chemistry_df = pd.read_parquet(chemistry_parquet_path, columns=["experiment_id", "canonical_id"])
    wanted = pd.Index(sorted(set(target_experiment_ids)))
    chemistry_df = chemistry_df[chemistry_df["experiment_id"].isin(wanted)].copy()
    chemistry_df["canonical_id"] = chemistry_df["canonical_id"].astype(str)
    chemistry_df["col_idx"] = chemistry_df["canonical_id"].map(vocab_map)
    chemistry_df = chemistry_df.dropna(subset=["col_idx"]).copy()
    chemistry_df["col_idx"] = chemistry_df["col_idx"].astype(np.int32)

    # Role-blind: duplicates collapse to 1 regardless of medium/stressor role.
    chemistry_df = chemistry_df.drop_duplicates(subset=["experiment_id", "col_idx"])
    row_ids = wanted.tolist()
    exp_to_row = {eid: i for i, eid in enumerate(row_ids)}
    chemistry_df["row_idx"] = chemistry_df["experiment_id"].map(exp_to_row).astype(np.int32)

    data = np.ones(len(chemistry_df), dtype=np.float32)
    mat = sp.csr_matrix(
        (data, (chemistry_df["row_idx"].to_numpy(), chemistry_df["col_idx"].to_numpy())),
        shape=(len(row_ids), vocab_size),
    )
    sp.save_npz(sparse_path, mat)
    row_ids_path.write_text(json.dumps(row_ids))
    return mat, exp_to_row, vocab_size


@dataclass
class S5RowBatch:
    """Row-level arrays for one partition."""

    gene_idx: np.ndarray
    exp_idx: np.ndarray
    y: np.ndarray
    abs_t: np.ndarray
    cor12: np.ndarray
    org_id: np.ndarray
    gene_key: np.ndarray
    row_index: np.ndarray


def make_row_batch(
    fitness_df: pd.DataFrame,
    *,
    gene_key_to_idx: dict[str, int],
    experiment_id_to_row: dict[str, int],
    fit_col: str = "fit",
) -> S5RowBatch:
    """Convert a fitness frame into index arrays consumed by S5 datasets."""
    required = ["gene_key", "experiment_id", "t", "cor12", "orgId", fit_col]
    missing = [c for c in required if c not in fitness_df.columns]
    if missing:
        raise ValueError(f"fitness_df missing columns for S5 batch: {missing}")

    work = fitness_df.copy()
    work["gene_idx"] = work["gene_key"].map(gene_key_to_idx)
    work["exp_idx"] = work["experiment_id"].map(experiment_id_to_row)
    work = work.dropna(subset=["gene_idx", "exp_idx", fit_col]).copy()
    if len(work) == 0:
        raise ValueError("No rows remain after joining embeddings and experiment multihot.")

    return S5RowBatch(
        gene_idx=work["gene_idx"].astype(np.int32).to_numpy(),
        exp_idx=work["exp_idx"].astype(np.int32).to_numpy(),
        y=pd.to_numeric(work[fit_col], errors="coerce").astype(np.float32).to_numpy(),
        abs_t=np.abs(pd.to_numeric(work["t"], errors="coerce")).astype(np.float32).to_numpy(),
        cor12=pd.to_numeric(work["cor12"], errors="coerce").astype(np.float32).to_numpy(),
        org_id=work["orgId"].astype(str).to_numpy(),
        gene_key=work["gene_key"].astype(str).to_numpy(),
        row_index=work.index.to_numpy(dtype=np.int64),
    )


class S5TorchDataset(Dataset):
    """Torch dataset backed by row indices into dense embedding/chem tables."""

    def __init__(
        self,
        row_batch: S5RowBatch,
        *,
        embedding_matrix: np.ndarray,
        chemistry_matrix: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        self.row_batch = row_batch
        self.embedding_matrix = embedding_matrix
        self.chemistry_matrix = chemistry_matrix
        if weights is None:
            weights = np.ones(len(row_batch.y), dtype=np.float32)
        if len(weights) != len(row_batch.y):
            raise ValueError("weights must match number of row targets")
        self.weights = weights.astype(np.float32)

    def __len__(self) -> int:
        return int(len(self.row_batch.y))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = self.row_batch.gene_idx[idx]
        e = self.row_batch.exp_idx[idx]
        return {
            "gene_emb": torch.from_numpy(self.embedding_matrix[g]).to(torch.float32),
            "chem_multihot": torch.from_numpy(self.chemistry_matrix[e]).to(torch.float32),
            "y": torch.tensor(self.row_batch.y[idx], dtype=torch.float32),
            "weight": torch.tensor(self.weights[idx], dtype=torch.float32),
            "abs_t": torch.tensor(self.row_batch.abs_t[idx], dtype=torch.float32),
            "cor12": torch.tensor(self.row_batch.cor12[idx], dtype=torch.float32),
        }


def dense_chemistry_from_csr(mat: sp.csr_matrix) -> np.ndarray:
    """Materialize experiment-level chemistry features as dense float32."""
    return mat.toarray().astype(np.float32, copy=False)

