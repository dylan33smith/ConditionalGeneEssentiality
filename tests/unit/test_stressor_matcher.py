"""Tests for stressor ↔ workbook string matching."""
from __future__ import annotations

import pandas as pd

from src.data.preprocessing.stressor_matcher import (
    build_candidate_report,
    build_ratified_mapping_from_report,
    build_workbook_stressor_index,
    exact_match_after_normalize,
    fuzzy_top_canonicals,
    normalize_chemical_name,
)


def test_normalize_case_and_whitespace() -> None:
    assert normalize_chemical_name("  Ethanol  ") == "ethanol"


def test_normalize_strips_hydrate_suffix() -> None:
    assert normalize_chemical_name("Nickel (II) chloride hexahydrate") == "nickel (ii) chloride"


def test_workbook_index_exact_round_trip() -> None:
    df = pd.DataFrame(
        {
            "Canonical_ID": ["C_ETOH", "C_NACL"],
            "Compound_name": ["Ethanol", "Sodium chloride"],
            "Source_row_component": ["ethanol 200 proof", "NaCl"],
            "Include_in_ml": [True, True],
        }
    )
    idx = build_workbook_stressor_index(df)
    assert idx["ethanol"] == "C_ETOH"
    assert idx["sodium chloride"] == "C_NACL"


def test_exact_match_after_normalize() -> None:
    idx = {"ethanol": "C_ETOH", "sodium sulfate": "C_SO4"}
    assert exact_match_after_normalize("Ethanol", idx) == "C_ETOH"
    assert exact_match_after_normalize("SODIUM SULFATE decahydrate", idx) == "C_SO4"


def test_fuzzy_prefers_best_canonical() -> None:
    idx = {
        "ethanol": "C1",
        "ethyl alcohol": "C1",
        "methanol": "C2",
    }
    top = fuzzy_top_canonicals("Ethanol 99%", idx, k=2)
    assert top[0][0] == "C1"
    assert top[0][1] >= 0.5


def test_candidate_report_exact_and_fuzzy_columns() -> None:
    df = pd.DataFrame(
        {
            "Canonical_ID": ["CX"],
            "Compound_name": ["Widget acid"],
            "Source_row_component": ["Widget-acid sodium salt"],
            "Include_in_ml": [True],
        }
    )
    stressors = {"Widget acid sodium salt": 10, "zzz_nonexistent_compound_999": 2}
    rep = build_candidate_report(stressors, df, auto_fuzzy_threshold=0.95)
    row_exact = rep[rep["stressor"] == "Widget acid sodium salt"].iloc[0]
    assert row_exact["match_type"] == "exact_after_norm"
    assert bool(row_exact["auto_apply"]) is True
    row_none = rep[rep["stressor"] == "zzz_nonexistent_compound_999"].iloc[0]
    assert row_none["match_type"] == "none"
    assert bool(row_none["auto_apply"]) is False


def test_ratified_mapping_includes_exact_only_by_default() -> None:
    df = pd.DataFrame(
        {
            "Canonical_ID": ["A", "B"],
            "Compound_name": ["Alpha", "Beta"],
            "Source_row_component": ["Alpha", "Beta"],
            "Include_in_ml": [True, True],
        }
    )
    stressors = {"alpha monohydrate": 5, "gamma": 1}
    rep = build_candidate_report(stressors, df, auto_fuzzy_threshold=0.95)
    m = build_ratified_mapping_from_report(rep, fuzzy_auto_threshold=0.95)
    assert m["alpha monohydrate"] == "A"
    assert "gamma" not in m
