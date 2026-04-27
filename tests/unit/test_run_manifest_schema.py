"""Validate that example run-manifest payloads pass / fail the v1 JSON schema."""
import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

SCHEMA_PATH = Path("data_contract/schemas/run_manifest_v1.schema.json")


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text())


def _valid_manifest():
    """Minimal valid manifest for shape testing."""
    return {
        "run_id": "20260427_120000_S0_smoke_s0_abc1234",
        "experiment_id": "S0_smoke",
        "stage_or_tier": "S0",
        "git_sha": "abc1234",
        "code_sha": "abc1234567890",
        "seed": 0,
        "data_contract_checksums": {
            "feba_db_sha256": "0" * 64,
            "workbook_v4_sha256": "1" * 64,
            "workbook_v4_sheet": "Media_Components_ML",
            "embedding_manifest_id": "emb_v0",
            "canonical_manifest_id": "canon_v0",
        },
        "split_protocol": {
            "protocol_id": "candidate_smoke",
            "split_manifest_sha256": "2" * 64,
        },
        "preprocessing_artifact_id": "preproc_smoke",
        "config_snapshot_path": ".hydra/config.yaml",
        "scored_rowset": {
            "scored_rowset_hash": "3" * 64,
            "n_rows_scored": 100,
            "n_genes_eligible": 10,
            "inclusion_counters": {
                "pre_join": 200,
                "post_join": 150,
                "post_filter": 120,
                "post_split": 100,
            },
        },
        "metrics": {"rmse": 0.5, "mae": 0.4, "within_gene_spearman": None},
        "null_baseline_deltas": {
            "global_train_mean": {
                "baseline_rmse": 0.6, "baseline_mae": 0.5,
                "delta_rmse": -0.1, "delta_mae": -0.1,
            },
            "additive_baseline": {
                "baseline_rmse": 0.55, "baseline_mae": 0.45,
                "delta_rmse": -0.05, "delta_mae": -0.05,
            },
        },
        "unknown_category_rate": 0.0,
    }


def test_valid_manifest_passes(schema):
    jsonschema.validate(_valid_manifest(), schema)


def test_missing_required_field_fails(schema):
    bad = _valid_manifest()
    del bad["git_sha"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_wrong_sheet_name_fails(schema):
    bad = _valid_manifest()
    bad["data_contract_checksums"]["workbook_v4_sheet"] = "Media_Components"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_invalid_sha_fails(schema):
    bad = _valid_manifest()
    bad["data_contract_checksums"]["feba_db_sha256"] = "not_a_sha"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
