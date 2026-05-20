"""Shared building blocks for Tier 1 experiments.

T1-A through T1-E all need:
  - canonical fitness + S4 artifact loading
  - locked split application
  - S5 quality weights
  - gene embeddings
  - a model vehicle that accepts a (gene_emb, condition_tensor) pair

This module collects the reusable pieces so each Tier-1 experiment is just
a thin orchestrator on top.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.datasets.build_s5_dataset import (
    S5RowBatch,
    S5TorchDataset,
    _load_concatenated_embeddings,
    add_experiment_id,
    build_or_load_experiment_multihot,
    dense_chemistry_from_csr,
    make_row_batch,
)
from src.experiments.stage5.run import (
    compute_train_thresholds,
    compute_weighted_full_weights,
)


@dataclass
class T1Inputs:
    """Everything a T1 experiment needs after the shared setup."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    train_batch: S5RowBatch
    val_batch: S5RowBatch
    embedding_matrix: np.ndarray            # (n_genes, 1152)
    gene_key_to_idx: dict[str, int]
    chemistry_dense: np.ndarray             # (n_experiments, 425) multihot
    exp_to_row: dict[str, int]
    exp_id_to_media: dict[str, str]         # experiment_id -> media string
    weighted_weights: np.ndarray            # S5 weighted_full weights per train row
    spearman_m: int
    spearman_vmin: float
    locked_protocol_id: str
    artifact_id: str


def load_t1_inputs(
    *,
    fitness_path: Path,
    feature_contract_path: Path,
    locked_protocol_path: Path,
    eval_policy_path: Path,
    embedding_dir: Path,
    cache_dir: Path,
    embedding_filename_suffix: str = "_proteomelm.pt",
) -> T1Inputs:
    """Load canonical fitness, S4 artifact, locked split, S5 quality weights, embeddings."""
    feature_contract = yaml.safe_load(feature_contract_path.read_text())
    artifact_id = str(feature_contract["artifact_id"])
    artifact_root = Path("data_contract/preprocessing") / artifact_id
    chemistry_parquet = artifact_root / feature_contract["experiment_chemistry_table"]["path"]
    canonical_vocab_path = artifact_root / "canonical_id_vocab.json"

    eval_policy = yaml.safe_load(eval_policy_path.read_text())
    locked_protocol = yaml.safe_load(locked_protocol_path.read_text())
    protocol_id = str(locked_protocol["protocol_id"])
    spearman_m = int(eval_policy["spearman_eligibility"]["m"])
    spearman_vmin = float(
        eval_policy["spearman_eligibility"]["v_min_value_per_protocol"][protocol_id]
    )

    fitness_df = pd.read_parquet(fitness_path)
    fitness_df = add_experiment_id(fitness_df)
    val_orgs = set(locked_protocol["val_org_ids"])
    test_orgs = set(locked_protocol["test_org_ids"])
    train_mask = ~(fitness_df["orgId"].isin(val_orgs | test_orgs))
    val_mask = fitness_df["orgId"].isin(val_orgs)
    train_df = fitness_df.loc[train_mask].copy()
    val_df = fitness_df.loc[val_mask].copy()

    all_orgs = fitness_df["orgId"].astype(str).unique().tolist()
    embedding_matrix, gene_key_to_idx = _load_concatenated_embeddings(
        all_orgs, embedding_dir, filename_suffix=embedding_filename_suffix,
    )

    exp_ids = fitness_df["experiment_id"].astype(str).unique().tolist()
    chem_csr, exp_to_row, chem_len = build_or_load_experiment_multihot(
        chemistry_parquet_path=chemistry_parquet,
        canonical_vocab_json_path=canonical_vocab_path,
        target_experiment_ids=exp_ids,
        cache_dir=cache_dir / artifact_id,
        sparse_cache_filename="multihot.npz",
    )
    chemistry_dense = dense_chemistry_from_csr(chem_csr)

    train_batch = make_row_batch(
        train_df, gene_key_to_idx=gene_key_to_idx, experiment_id_to_row=exp_to_row
    )
    val_batch = make_row_batch(
        val_df, gene_key_to_idx=gene_key_to_idx, experiment_id_to_row=exp_to_row
    )

    # S5 weighted_full weights (locked quality policy)
    thresholds = compute_train_thresholds(
        train_df, cor12_quantile=0.25, abs_t_quantile=0.25
    )
    weighted_weights = compute_weighted_full_weights(
        train_batch.cor12, train_batch.abs_t, thresholds
    )

    # experiment_id -> media (for media_id encoder vocab)
    exp_id_to_media = (
        fitness_df[["experiment_id", "media"]]
        .drop_duplicates(subset=["experiment_id"])
        .set_index("experiment_id")["media"]
        .astype(str)
        .to_dict()
    )

    return T1Inputs(
        train_df=train_df,
        val_df=val_df,
        train_batch=train_batch,
        val_batch=val_batch,
        embedding_matrix=embedding_matrix,
        gene_key_to_idx=gene_key_to_idx,
        chemistry_dense=chemistry_dense,
        exp_to_row=exp_to_row,
        exp_id_to_media=exp_id_to_media,
        weighted_weights=weighted_weights,
        spearman_m=spearman_m,
        spearman_vmin=spearman_vmin,
        locked_protocol_id=protocol_id,
        artifact_id=artifact_id,
    )


def build_media_id_chemistry_matrix(
    inputs: T1Inputs,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build a (n_experiments, 1) int32 matrix of media-vocabulary indices.

    Vocab is built from train experiments only (the locked train rows). Val/test
    experiments whose `media` string is not in the train vocab map to index 0
    (`<UNK>`). This is the operationalization of H-DATA-01 for the media-id arm.

    Returns:
        chem_matrix_int: (n_experiments, 1) int32 matrix, aligned with
            inputs.exp_to_row (same first dim as inputs.chemistry_dense).
        media_vocab: {media_str: idx} where idx >= 1; idx 0 is reserved for <UNK>.
    """
    # Train medias only
    train_exp_ids = pd.Series(inputs.train_batch.exp_idx).map(
        {v: k for k, v in inputs.exp_to_row.items()}
    )
    train_medias = sorted(
        set(inputs.exp_id_to_media[eid] for eid in train_exp_ids if eid in inputs.exp_id_to_media)
    )
    media_vocab = {m: i + 1 for i, m in enumerate(train_medias)}  # idx 0 = UNK

    n_exp = inputs.chemistry_dense.shape[0]
    chem_matrix_int = np.zeros((n_exp, 1), dtype=np.int32)
    for exp_id, row in inputs.exp_to_row.items():
        media = inputs.exp_id_to_media.get(exp_id)
        if media is None:
            chem_matrix_int[row, 0] = 0  # UNK
        else:
            chem_matrix_int[row, 0] = media_vocab.get(media, 0)  # UNK if not in train
    return chem_matrix_int, media_vocab


class T1ConcatLinearMLP(torch.nn.Module):
    """Generic concat-linear MLP that accepts either a precomputed condition
    vector (multihot path) or an integer index (media_id path).

    The two arms produce the same output dim (`condition_dim`) so the head
    is identical; only the encoder differs.
    """

    def __init__(
        self,
        *,
        gene_dim: int,
        condition_encoder: torch.nn.Module,
        condition_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.condition_encoder = condition_encoder
        in_dim = int(gene_dim) + int(condition_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_dim, int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, gene_emb: torch.Tensor, condition_input: torch.Tensor) -> torch.Tensor:
        cond_vec = self.condition_encoder(condition_input)
        x = torch.cat([gene_emb, cond_vec], dim=1)
        return self.head(x).squeeze(1)


class _MultihotPassthrough(torch.nn.Module):
    """Identity encoder for the multihot arm — input is already the condition vec."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()


class _MediaIdEmbedding(torch.nn.Module):
    """Encoder for the media-id arm — looks up a learnable embedding per media."""

    def __init__(self, n_media_vocab: int, embed_dim: int) -> None:
        super().__init__()
        # idx 0 = <UNK>, no padding (we want UNK to be a learnable token).
        self.embedding = torch.nn.Embedding(int(n_media_vocab), int(embed_dim))

    def forward(self, media_idx_input: torch.Tensor) -> torch.Tensor:
        # Input is (B, 1) int — squeeze to (B,)
        idx = media_idx_input.long().reshape(-1)
        return self.embedding(idx)


def make_multihot_model(*, gene_dim: int, chem_dim: int, hidden_dim: int, dropout: float) -> T1ConcatLinearMLP:
    return T1ConcatLinearMLP(
        gene_dim=gene_dim,
        condition_encoder=_MultihotPassthrough(),
        condition_dim=chem_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def make_media_id_model(
    *,
    gene_dim: int,
    n_media_vocab: int,
    embed_dim: int,
    hidden_dim: int,
    dropout: float,
) -> T1ConcatLinearMLP:
    return T1ConcatLinearMLP(
        gene_dim=gene_dim,
        condition_encoder=_MediaIdEmbedding(n_media_vocab, embed_dim),
        condition_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
