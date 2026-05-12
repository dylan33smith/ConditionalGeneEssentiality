"""Unit tests for S3 split-lock selection and payload emission."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.experiments.stage3.run import (
    build_diagnostic_protocols_payload,
    build_locked_protocol_payload,
    select_primary_protocol,
)


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def _load_json(path: Path):
    return json.loads(path.read_text())


def test_selection_rule_picks_multi_org_balanced_on_real_artifacts() -> None:
    root = Path(__file__).resolve().parents[2]
    candidates = _load_yaml(root / "data_contract/splits/candidate_protocols.yaml")
    baselines = _load_json(root / "artifacts/baselines/baselines_per_protocol.json")
    eval_policy = _load_yaml(root / "data_contract/policy/eval_policy.yaml")
    winner, table = select_primary_protocol(candidates, baselines, eval_policy)
    assert winner == "multi_org_balanced"
    assert any(r["protocol_id"] == "multi_org_balanced" for r in table)


def test_selection_rule_with_synthetic_candidates() -> None:
    candidates = {
        "candidates": [
            {
                "protocol_id": "p1",
                "support": {"val_rows": 2000},
                "chemistry_overlap": {"val_canonical_id_seen_rate": 1.0, "val_seen_rate": 0.4},
            },
            {
                "protocol_id": "p2",
                "support": {"val_rows": 3000},
                "chemistry_overlap": {"val_canonical_id_seen_rate": 1.0, "val_seen_rate": 0.5},
            },
        ]
    }
    baselines = {
        "candidates": {
            "p1": {
                "power": {"bootstrap_ci": {"n_genes_used": 1000}, "spearman_role": "primary"},
                "baselines": {
                    "embedding_nn": {"rmse": 0.6},
                    "global_train_mean": {"rmse": 0.7},
                },
            },
            "p2": {
                "power": {"bootstrap_ci": {"n_genes_used": 1200}, "spearman_role": "primary"},
                "baselines": {
                    "embedding_nn": {"rmse": 0.8},
                    "global_train_mean": {"rmse": 0.7},
                },
            },
        }
    }
    eval_policy = {"spearman_eligibility": {"m": 5}}
    winner, _ = select_primary_protocol(candidates, baselines, eval_policy)
    assert winner == "p1"


def test_selection_rule_no_winner_raises() -> None:
    candidates = {
        "candidates": [
            {
                "protocol_id": "p_bad",
                "support": {"val_rows": 50},
                "chemistry_overlap": {"val_canonical_id_seen_rate": 0.5, "val_seen_rate": 0.0},
            }
        ]
    }
    baselines = {
        "candidates": {
            "p_bad": {
                "power": {"bootstrap_ci": {"n_genes_used": 10}, "spearman_role": "diagnostic_only"},
                "baselines": {
                    "embedding_nn": {"rmse": 1.0},
                    "global_train_mean": {"rmse": 0.9},
                },
            }
        }
    }
    eval_policy = {"spearman_eligibility": {"m": 5}}
    try:
        select_primary_protocol(candidates, baselines, eval_policy)
    except ValueError as e:
        assert "No S3 primary protocol" in str(e)
    else:
        raise AssertionError("Expected ValueError for no eligible S3 protocol")


def test_emit_payload_is_deterministic() -> None:
    winner = {"protocol_id": "multi_org_balanced", "val_org_ids": ["v2", "v1"], "test_org_ids": ["t1"]}
    table = [{"protocol_id": "multi_org_balanced", "eligible": True}]
    a = build_locked_protocol_payload(
        winner=winner,
        selection_table=table,
        candidate_sha256="a",
        baselines_sha256="b",
        eval_policy_sha256="c",
    )
    b = build_locked_protocol_payload(
        winner=winner,
        selection_table=table,
        candidate_sha256="a",
        baselines_sha256="b",
        eval_policy_sha256="c",
    )
    assert a == b
    assert a["split_manifest_sha256"] == b["split_manifest_sha256"]


def test_diagnostic_payload_includes_homology_recipe() -> None:
    payload = build_diagnostic_protocols_payload(
        candidates_payload={
            "candidates": [
                {"protocol_id": "multi_org_balanced", "chemistry_overlap": {"val_seen_rate": 1.0}},
                {"protocol_id": "low_overlap_stress", "chemistry_overlap": {"val_seen_rate": 0.0}},
            ]
        },
        primary_id="multi_org_balanced",
        enable_homology_diagnostic=True,
    )
    hom = [p for p in payload["protocols"] if p["id"] == "homology_diagnostic"][0]
    assert hom["cosine_threshold"] == 0.85
    assert len(hom["bin_edges"]) == 4
    assert "S3-DEC-001" in hom["decision_ref"]
