"""Unit tests for S4 Option D feature-contract preprocessing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import pandas as pd
import yaml

from src.data.preprocessing.build_experiment_chemistry import (
    apply_chemistry_vocab_trim,
    build_experiment_chemistry_long,
    experiment_uid,
)
from src.data.preprocessing.fit_condition_scalers import fit_log1p_scaler
from src.data.preprocessing.fit_condition_vocab import (
    build_representation_mode_per_canonical,
    fit_canonical_id_vocab,
    fit_canonical_id_vocab_from_chemistry_long,
)
from src.data.preprocessing.fit_metadata_vocabs import fit_metadata_field_vocab
from src.data.preprocessing.transform_conditions import (
    apply_amount_scaler,
    chemistry_row_unknown_rates,
    load_experiment_chemistry,
)


def _components_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Media": ["M1", "M1", "M2", "M2", "M3", "M4", "M4"],
            "Canonical_ID": ["C_A", "C_B", "C_A", "C_C", "C_VAL_ONLY", "C_A", "C_A"],
            "Include_in_ml": [True, True, True, True, True, True, True],
            "Decomposition_type": ["direct", "mix", "direct", "extract", "mix", "salt", "salt"],
            "Amount": [1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 1.0],
        }
    )


def test_vocab_fit_only_on_train_rows_no_leakage() -> None:
    train_rows = pd.DataFrame({"media": ["M1", "M1", "M2", "M2"]})
    components = _components_df()
    vocab = fit_canonical_id_vocab(
        train_rows_df=train_rows,
        components_df=components,
        prevalence_threshold=0.0,
    )
    assert "C_VAL_ONLY" not in vocab["canonical_id_to_index"]


def test_fit_canonical_id_vocab_from_chemistry_long_uses_experiment_counts() -> None:
    chem = pd.DataFrame(
        {
            "experiment_id": ["e1", "e1", "e2", "e3"],
            "canonical_id": ["X", "Y", "X", "Z"],
            "role": ["medium", "stressor", "medium", "medium"],
            "amount": [1.0, 1.0, 1.0, 1.0],
        }
    )
    # n_train_exp=3; threshold 0.7 => min_exp=2 => only X (in e1+e2) survives
    v = fit_canonical_id_vocab_from_chemistry_long(chem, prevalence_threshold=0.7)
    assert "X" in v["canonical_id_to_index"]
    assert "Y" not in v["canonical_id_to_index"]
    assert "Z" not in v["canonical_id_to_index"]


def test_unk_index_is_zero_in_media_vocab() -> None:
    train_rows = pd.DataFrame({"media": ["M1", "M2"]})
    components = _components_df()
    vocab = fit_canonical_id_vocab(
        train_rows_df=train_rows,
        components_df=components,
        prevalence_threshold=0.0,
    )
    assert vocab["canonical_id_vocab"][0] == "<UNK>"
    assert vocab["canonical_id_to_index"]["<UNK>"] == 0


def test_prevalence_threshold_applied_media_vocab() -> None:
    train_rows = pd.DataFrame({"media": ["M1"] * 900 + ["M2"] * 100})
    components = pd.DataFrame(
        {
            "Media": ["M1", "M2"],
            "Canonical_ID": ["C_COMMON", "C_RARE"],
            "Include_in_ml": [True, True],
        }
    )
    vocab = fit_canonical_id_vocab(
        train_rows_df=train_rows,
        components_df=components,
        prevalence_threshold=0.2,
    )
    assert "C_COMMON" in vocab["canonical_id_to_index"]
    assert "C_RARE" not in vocab["canonical_id_to_index"]


def test_experiment_uid_missing_media_hashes_like_empty_string() -> None:
    explicit_empty = pd.Series(
        {"orgId": "O1", "setName": "S", "seqindex": "IT1", "media": ""}
    )
    nan_media = pd.Series(
        {"orgId": "O1", "setName": "S", "seqindex": "IT1", "media": np.nan}
    )
    assert experiment_uid(explicit_empty) == experiment_uid(nan_media)


def test_experiment_uid_stable() -> None:
    row = pd.Series(
        {
            "orgId": "O1",
            "setName": "S1",
            "seqindex": "IT1",
            "media": "LB",
        }
    )
    a = experiment_uid(row)
    b = experiment_uid(row)
    assert a == b
    assert len(a) == 64


def test_chemistry_prevalence_zero_keeps_all_train_canonicals() -> None:
    chem = pd.DataFrame(
        {
            "experiment_id": ["e1", "e2"],
            "canonical_id": ["Rare", "Rare"],
            "role": ["medium", "stressor"],
            "amount": [1.0, 2.0],
        }
    )
    v = fit_canonical_id_vocab_from_chemistry_long(chem, prevalence_threshold=0.0)
    assert "Rare" in v["canonical_id_to_index"]


def test_apply_chemistry_vocab_trim_stressor_vs_medium_unk() -> None:
    df = pd.DataFrame(
        {
            "experiment_id": ["e1", "e1"],
            "canonical_id": ["RARE_M", "RARE_S"],
            "role": ["medium", "stressor"],
            "amount": [1.0, 1.0],
        }
    )
    kept: set[str] = set()  # force all OOV
    out = apply_chemistry_vocab_trim(
        df,
        kept_canonicals=kept,
        unk_token="<UNK>",
        unk_stressor_token="<UNK_STRESSOR>",
    )
    assert out.iloc[0]["canonical_id"] == "<UNK>"
    assert out.iloc[1]["canonical_id"] == "<UNK_STRESSOR>"


def test_log1p_scaler_train_only() -> None:
    train = np.array([0.0, 1.0, 3.0, 7.0])
    val = np.array([100.0, 200.0])
    train_scaler = fit_log1p_scaler(train)
    val_scaler = fit_log1p_scaler(val)
    assert train_scaler["mean"] != val_scaler["mean"]
    assert train_scaler["std"] != val_scaler["std"]


def test_metadata_vocabs_have_unk_and_missing_indices() -> None:
    values = pd.Series(["aerobic", None, "anaerobic", "aerobic"])
    payload = fit_metadata_field_vocab(values, field_name="oxygen", prevalence_threshold=0.0)
    assert payload["vocab"][0] == "<UNK>"
    assert payload["vocab"][1] == "<MISSING>"
    assert payload["vocab_to_index"]["<UNK>"] == 0
    assert payload["vocab_to_index"]["<MISSING>"] == 1


def test_build_experiment_chemistry_resolves_and_dedupes(tmp_path: Path) -> None:
    comps = pd.DataFrame(
        {
            "Media": ["M1"],
            "Canonical_ID": ["C1"],
            "Include_in_ml": [True],
            "Amount": [2.0],
            "Decomposition_type": ["direct"],
        }
    )
    exp = pd.DataFrame(
        [
            {
                "orgId": "O1",
                "setName": "setA",
                "seqindex": "IT1",
                "media": "M1",
                "condition_1": "S1",
                "concentration_1": 0.5,
                "condition_2": None,
                "concentration_2": None,
                "condition_3": None,
                "concentration_3": None,
                "condition_4": None,
                "concentration_4": None,
            }
        ]
    )
    res = {"S1": "C_STRESS"}
    long_df = build_experiment_chemistry_long(exp, comps, res, include_in_ml_only=True)
    uid = experiment_uid(exp.iloc[0])
    assert (long_df["experiment_id"] == uid).all()
    roles = set(long_df["role"])
    assert roles == {"medium", "stressor"}


def test_apply_amount_scaler_roundtrip_file(tmp_path: Path) -> None:
    art = tmp_path / "amount_scaler.json"
    payload = {"transform": "log1p", "mean": 0.5, "std": 1.0, "n_train": 3, "n_finite": 3}
    art.write_text(json.dumps(payload))
    out = apply_amount_scaler(np.array([1.0, np.nan]), art)
    assert out.shape == (2,)
    assert np.isfinite(out[0])


def test_chemistry_unknown_rates() -> None:
    df = pd.DataFrame(
        {
            "canonical_id": ["<UNK>", "<UNK_STRESSOR>", "A"],
        }
    )
    r = chemistry_row_unknown_rates(df, unk_token="<UNK>", unk_stressor_token="<UNK_STRESSOR>")
    assert r["unk_medium_fraction"] == pytest.approx(1 / 3)
    assert r["unk_stressor_fraction"] == pytest.approx(1 / 3)


def test_load_experiment_chemistry_reads_parquet(tmp_path: Path) -> None:
    art = tmp_path / "bundle"
    art.mkdir()
    df = pd.DataFrame(
        [{"experiment_id": "a", "canonical_id": "X", "role": "medium", "amount": 1.0, "log1p_amount": 0.1}]
    )
    df.to_parquet(art / "experiment_chemistry.parquet", index=False)
    loaded = load_experiment_chemistry(art)
    assert len(loaded) == 1


def test_build_representation_mode_per_canonical_stressor_only_for_workbook_absent(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    mode_yaml = repo_root / "data_contract" / "representation_mode_mapping.yaml"
    comps = pd.DataFrame(
        {
            "Media": ["M1"],
            "Canonical_ID": ["C_workbook"],
            "Include_in_ml": [True],
            "Decomposition_type": ["direct"],
        }
    )
    out = build_representation_mode_per_canonical(
        ["C_workbook", "C_stressor_only"],
        comps,
        mode_yaml,
    )
    by_id = out.set_index("canonical_id").to_dict("index")
    assert by_id["C_workbook"]["dominant_mode"] == "physical"
    assert by_id["C_stressor_only"]["dominant_mode"] == "stressor"
    assert by_id["C_stressor_only"]["prop_stressor"] == pytest.approx(1.0)


def test_stressor_resolution_yaml_roundtrip(tmp_path: Path) -> None:
    from src.data.preprocessing.build_experiment_chemistry import load_stressor_resolution_map

    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "map": {"A": "B", "C": "<NEW>"}}))
    m = load_stressor_resolution_map(p)
    assert m == {"A": "B"}
