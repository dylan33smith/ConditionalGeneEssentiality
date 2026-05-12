"""Unit tests for S5 quality-policy components."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.datasets.build_s5_dataset import build_or_load_experiment_multihot
from src.experiments.stage5.run import (
    compute_strict_slice_mask,
    compute_train_thresholds,
    compute_weighted_full_weights,
)
from src.models.concat_linear import ConcatLinearMLP


def test_weighted_full_weights_include_cor12_and_abs_t() -> None:
    thresholds = {
        "cor12_median_train": 0.4,
        "abs_t_median_train": 2.0,
        "cor12_q_train": 0.2,
        "abs_t_q_train": 1.0,
    }
    cor12 = np.array([0.4, 0.2, 0.8], dtype=np.float32)
    abs_t = np.array([2.0, 1.0, 4.0], dtype=np.float32)
    w = compute_weighted_full_weights(cor12, abs_t, thresholds)
    assert np.allclose(w, np.array([1.0, 0.25, 1.0], dtype=np.float32))


def test_strict_slice_mask_uses_both_thresholds() -> None:
    thresholds = {
        "cor12_median_train": 0.4,
        "abs_t_median_train": 2.0,
        "cor12_q_train": 0.3,
        "abs_t_q_train": 1.5,
    }
    cor12 = np.array([0.5, 0.2, 0.5], dtype=np.float32)
    abs_t = np.array([1.0, 2.0, 2.0], dtype=np.float32)
    keep = compute_strict_slice_mask(cor12, abs_t, thresholds)
    assert keep.tolist() == [False, False, True]


def test_train_only_thresholds_ignore_val_rows() -> None:
    train_df = pd.DataFrame({"cor12": [0.1, 0.2, 0.3], "t": [1.0, 2.0, 3.0]})
    val_df = pd.DataFrame({"cor12": [999.0], "t": [999.0]})
    th_train = compute_train_thresholds(train_df, cor12_quantile=0.25, abs_t_quantile=0.25)
    th_concat = compute_train_thresholds(
        pd.concat([train_df, val_df], ignore_index=True),
        cor12_quantile=0.25,
        abs_t_quantile=0.25,
    )
    assert th_train["cor12_median_train"] != th_concat["cor12_median_train"]
    assert th_train["abs_t_median_train"] != th_concat["abs_t_median_train"]


def test_multihot_cache_is_stable(tmp_path: Path) -> None:
    chemistry = pd.DataFrame(
        {
            "experiment_id": ["e1", "e1", "e2"],
            "canonical_id": ["A", "B", "A"],
        }
    )
    chem_path = tmp_path / "chem.parquet"
    chemistry.to_parquet(chem_path, index=False)
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"canonical_id_to_index": {"<UNK>": 0, "A": 1, "B": 2}}))
    cache = tmp_path / "cache"
    m1, exp_to_row_1, _ = build_or_load_experiment_multihot(
        chemistry_parquet_path=chem_path,
        canonical_vocab_json_path=vocab_path,
        target_experiment_ids=["e1", "e2"],
        cache_dir=cache,
    )
    m2, exp_to_row_2, _ = build_or_load_experiment_multihot(
        chemistry_parquet_path=chem_path,
        canonical_vocab_json_path=vocab_path,
        target_experiment_ids=["e1", "e2"],
        cache_dir=cache,
    )
    assert exp_to_row_1 == exp_to_row_2
    assert np.array_equal(m1.toarray(), m2.toarray())


def test_concat_linear_forward_shape() -> None:
    model = ConcatLinearMLP(gene_dim=4, chemistry_dim=3, hidden_dim=8, dropout=0.0)
    gene = np.random.randn(5, 4).astype(np.float32)
    chem = np.random.randn(5, 3).astype(np.float32)
    import torch

    out = model(torch.from_numpy(gene), torch.from_numpy(chem))
    assert tuple(out.shape) == (5,)

