"""S4 — Feature Contract handler (Option D: chemistry long + metadata wide)."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from omegaconf import DictConfig

from src.data.ingestion.checksums import file_sha256
from src.data.preprocessing.build_experiment_chemistry import (
    build_experiment_chemistry_long,
    experiment_uid,
    load_stressor_resolution_map,
)
from src.data.preprocessing.build_experiment_metadata import (
    fit_and_build_experiment_metadata_table,
)
from src.data.preprocessing.fit_condition_scalers import (
    fit_log1p_scaler,
    record_bounded_reference_stats,
)
from src.data.preprocessing.fit_condition_vocab import (
    build_representation_mode_per_canonical,
    build_representation_mode_per_medium,
    fit_canonical_id_vocab_from_chemistry_long,
)
from src.data.preprocessing.transform_conditions import apply_amount_scaler, apply_metadata_encoding


log = logging.getLogger(__name__)


def _canonical_sha_for_dict(d: dict[str, Any]) -> str:
    payload = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_components(path: Path, sheet: str, include_in_ml_only: bool) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    if include_in_ml_only and "Include_in_ml" in df.columns:
        df = df[df["Include_in_ml"] == True].copy()  # noqa: E712
    return df


def _compute_artifact_id(per_file_hashes: dict[str, str]) -> str:
    return _canonical_sha_for_dict(per_file_hashes)[:16]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main(cfg: DictConfig) -> None:
    s4 = cfg.stage.s4
    locked_path = Path(str(s4.locked_protocol_path))
    feature_contract_path = Path(str(s4.output_feature_contract))
    out_root = Path(str(s4.output_artifact_dir))
    components_path = Path(str(s4.components_path))
    components_sheet = str(s4.components_sheet)
    mode_mapping_path = Path(str(s4.representation_mode_mapping_path))
    experiments_path = Path(str(s4.experiments_path))
    stressor_resolution_path = Path(str(s4.stressor_resolution_path))
    unk_stressor_token = str(s4.unk_stressor_token)

    log.info("=" * 60)
    log.info("S4 Feature Contract (Option D)")
    log.info("=" * 60)

    locked = yaml.safe_load(locked_path.read_text())
    val_orgs = set(locked["val_org_ids"])
    test_orgs = set(locked["test_org_ids"])
    held_out = val_orgs | test_orgs

    experiments_df = pd.read_parquet(experiments_path)
    train_mask = ~experiments_df["orgId"].isin(held_out)
    eval_mask = experiments_df["orgId"].isin(held_out)
    if not train_mask.any():
        raise ValueError("No train experiments after applying locked split.")
    log.info("Train experiments=%d, eval experiments=%d", int(train_mask.sum()), int(eval_mask.sum()))

    resolution_map = load_stressor_resolution_map(stressor_resolution_path)
    components_df = _load_components(
        components_path, components_sheet, include_in_ml_only=bool(s4.include_in_ml_only)
    )

    chemistry_long = build_experiment_chemistry_long(
        experiments_df,
        components_df,
        resolution_map,
        include_in_ml_only=bool(s4.include_in_ml_only),
    )

    train_uid_set = {experiment_uid(r) for _, r in experiments_df.loc[train_mask].iterrows()}
    eval_uid_set = {experiment_uid(r) for _, r in experiments_df.loc[eval_mask].iterrows()}

    train_chem = chemistry_long[chemistry_long["experiment_id"].isin(train_uid_set)].copy()

    chem_prev = float(s4.trimming.chemistry_prevalence_threshold)
    vocab_payload = fit_canonical_id_vocab_from_chemistry_long(
        train_chem,
        prevalence_threshold=chem_prev,
        unk_token=str(s4.unk_token),
        unk_stressor_token=unk_stressor_token,
        unk_index=int(s4.unk_index),
    )
    train_canonical_vocab = set(train_chem["canonical_id"].astype(str).unique())

    # Raw chemistry in parquet (no prevalence-based ID replacement; T1 handles OOV).
    chemistry_for_out = chemistry_long.copy()

    # Amount scaler: train-only finite non-negative amounts from train chemistry (raw IDs)
    train_slice = chemistry_for_out[chemistry_for_out["experiment_id"].isin(train_uid_set)]
    amt_train = pd.to_numeric(train_slice["amount"], errors="coerce").to_numpy(dtype=float)
    amount_scaler = fit_log1p_scaler(amt_train)

    staging_dir = out_root / "_tmp_s4"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "metadata_vocabs").mkdir(parents=True, exist_ok=True)

    _write_json(staging_dir / "amount_scaler.json", amount_scaler)
    amt_path = staging_dir / "amount_scaler.json"
    log1p_amounts = apply_amount_scaler(
        pd.to_numeric(chemistry_for_out["amount"], errors="coerce").to_numpy(dtype=float),
        amt_path,
    )
    chemistry_out = chemistry_for_out.assign(log1p_amount=log1p_amounts.astype("float32"))[
        ["experiment_id", "canonical_id", "role", "amount", "log1p_amount"]
    ]
    chemistry_out.to_parquet(staging_dir / "experiment_chemistry.parquet", index=False)

    train_media_names = (
        experiments_df.loc[train_mask, "media"].dropna().astype(str).unique().tolist()
    )
    comp_train = components_df[components_df["Media"].astype(str).isin(set(train_media_names))].copy()
    amount_workbook = pd.to_numeric(comp_train.get("Amount", pd.Series([], dtype=float)), errors="coerce")
    # bounded_ref already from chemistry train amounts; keep second record from workbook for audit?
    # Contract expects bounded_reference_stats from Amount domain — keep workbook-only for parity with S4-DEC-001
    bounded_ref_workbook = record_bounded_reference_stats(amount_workbook.to_numpy())
    _write_json(staging_dir / "bounded_reference_stats.json", bounded_ref_workbook)

    mode_df = build_representation_mode_per_medium(
        components_df=components_df,
        mode_mapping_yaml=mode_mapping_path,
    )
    mode_df.to_parquet(staging_dir / "representation_mode_per_media.parquet", index=False)

    canon_mode_df = build_representation_mode_per_canonical(
        chemistry_canonical_ids=chemistry_out["canonical_id"].astype(str).unique().tolist(),
        components_df=components_df,
        mode_mapping_yaml=mode_mapping_path,
    )
    canon_mode_df.to_parquet(staging_dir / "representation_mode_per_canonical.parquet", index=False)

    metadata_fields = dict(s4.metadata_fields)
    numeric_fields = dict(s4.numeric_metadata_fields)
    metadata_prevalence = float(s4.trimming.metadata_prevalence_threshold)

    meta_df, meta_vocab_payloads, numeric_scalers = fit_and_build_experiment_metadata_table(
        experiments_df,
        train_mask,
        categorical_specs=metadata_fields,
        numeric_source_cols=numeric_fields,
        metadata_prevalence_threshold=metadata_prevalence,
        unk_token=str(s4.unk_token),
        missing_token=str(s4.missing_token),
        unk_index=int(s4.unk_index),
        missing_index=int(s4.missing_index),
    )
    meta_df.to_parquet(staging_dir / "experiment_metadata.parquet", index=False)
    _write_json(staging_dir / "numeric_metadata_scalers.json", numeric_scalers)

    metadata_unknown_rates: dict[str, float] = {}
    for field, payload in meta_vocab_payloads.items():
        path = staging_dir / "metadata_vocabs" / f"{field}.json"
        _write_json(path, payload)
        if field in metadata_fields:
            src = metadata_fields[field]
            eval_vals = (
                experiments_df.loc[eval_mask, src]
                if src in experiments_df.columns
                else pd.Series([], dtype=object)
            )
            _, unk_rate = apply_metadata_encoding(eval_vals.reset_index(drop=True), path)
            metadata_unknown_rates[field] = float(unk_rate)

    _write_json(staging_dir / "canonical_id_vocab.json", vocab_payload)

    # Eval chemistry: fraction of rows whose canonical_id never appears in train slice
    eval_chem_out = chemistry_out[chemistry_out["experiment_id"].isin(eval_uid_set)]
    n_eval_rows = max(len(eval_chem_out), 1)
    oov_mask = ~eval_chem_out["canonical_id"].astype(str).isin(train_canonical_vocab)
    eval_chemistry_row_fraction_not_in_train_vocab = float(oov_mask.sum() / n_eval_rows)

    shutil.copy2(stressor_resolution_path, staging_dir / "stressor_to_canonical_id.yaml")

    files_for_manifest = [
        staging_dir / "canonical_id_vocab.json",
        staging_dir / "experiment_chemistry.parquet",
        staging_dir / "experiment_metadata.parquet",
        staging_dir / "representation_mode_per_media.parquet",
        staging_dir / "representation_mode_per_canonical.parquet",
        staging_dir / "amount_scaler.json",
        staging_dir / "bounded_reference_stats.json",
        staging_dir / "numeric_metadata_scalers.json",
        staging_dir / "stressor_to_canonical_id.yaml",
    ] + sorted((staging_dir / "metadata_vocabs").glob("*.json"))
    per_file_hashes = {str(p.relative_to(staging_dir)): file_sha256(p) for p in files_for_manifest}
    artifact_id = _compute_artifact_id(per_file_hashes)
    final_artifact_dir = out_root / artifact_id
    if final_artifact_dir.exists():
        shutil.rmtree(final_artifact_dir)
    staging_dir.rename(final_artifact_dir)

    manifest_payload = {
        "artifact_id": artifact_id,
        "per_file_sha256": per_file_hashes,
    }
    _write_json(final_artifact_dir / "artifact_manifest.json", manifest_payload)
    per_file_hashes["artifact_manifest.json"] = file_sha256(final_artifact_dir / "artifact_manifest.json")

    feature_contract = {
        "status": "locked",
        "emitted_by": "stage4",
        "schema_version": "s4_option_d_v1",
        "artifact_id": artifact_id,
        "locked_protocol_id": locked["protocol_id"],
        "locked_split_manifest_sha256": locked["split_manifest_sha256"],
        "unk_policy": {
            "unk_token": str(s4.unk_token),
            "unk_index": int(s4.unk_index),
            "missing_token": str(s4.missing_token),
            "missing_index": int(s4.missing_index),
            "unk_stressor_token": unk_stressor_token,
        },
        "stressor_resolution": {
            "source_path": str(stressor_resolution_path),
            "artifact_copy": "stressor_to_canonical_id.yaml",
        },
        "canonical_id_vocab": {
            "n_pretrim": vocab_payload["n_canonical_pretrim"],
            "n_posttrim": vocab_payload["n_canonical_posttrim"],
            "vocab_size_with_specials": len(vocab_payload["canonical_id_vocab"]),
            "chemistry_prevalence_threshold": chem_prev,
            "prevalence_unit": "distinct_train_experiments_per_canonical_id",
            "chemistry_parquet_policy": "raw_canonical_id_strings; no S4 prevalence replacement — OOV/rare bucketing deferred to T1",
        },
        "experiment_chemistry_table": {
            "path": "experiment_chemistry.parquet",
            "columns": ["experiment_id", "canonical_id", "role", "amount", "log1p_amount"],
        },
        "experiment_metadata_table": {
            "path": "experiment_metadata.parquet",
            "categorical_columns": [f"{k}_idx" for k in metadata_fields],
            "numeric_columns": [f"{k}_z" for k in numeric_fields] + [f"{k}_is_finite" for k in numeric_fields],
        },
        "representation_mode": {
            "modes": ["physical", "mix", "extract", "in_silico", "stressor"],
            "source_mapping_path": str(mode_mapping_path),
            "per_media_table": "representation_mode_per_media.parquet",
            "per_canonical_table": "representation_mode_per_canonical.parquet",
        },
        "concentration_policy": {
            "note": "units_1..units_4 are NOT consumed in S4. concentration_* values are pooled without unit conversion — do NOT use raw concentrations for cross-experiment comparisons or magnitude claims until T1 normalizes units.",
        },
        "numeric_registry": {
            "Amount": {
                "default_transform": "log1p",
                "fitted_scaler_file": "amount_scaler.json",
                "available_alternatives": list(s4.numeric_transforms.alternatives),
                "bounded_reference_file": "bounded_reference_stats.json",
            },
            "metadata_numeric": {
                col: {"fitted_scaler": "numeric_metadata_scalers.json", "field_key": col}
                for col in numeric_fields
            },
        },
        "metadata_registry": {
            field: {
                "source_column": source_col,
                "encoding": "categorical_index",
                "vocab_file": f"metadata_vocabs/{field}.json",
                "unknown_rate_eval": metadata_unknown_rates.get(field, 0.0),
                "prevalence_threshold": metadata_prevalence,
            }
            for field, source_col in metadata_fields.items()
        },
        "unknown_category_rates": {
            "chemistry_eval_rows": {
                "fraction_canonical_id_not_in_train_vocab": eval_chemistry_row_fraction_not_in_train_vocab,
            },
            "metadata_eval": metadata_unknown_rates,
        },
        "artifact_manifest": {
            "path": f"data_contract/preprocessing/{artifact_id}/artifact_manifest.json",
            "sha256": per_file_hashes["artifact_manifest.json"],
        },
    }
    feature_contract_path.parent.mkdir(parents=True, exist_ok=True)
    feature_contract_path.write_text(yaml.safe_dump(feature_contract, sort_keys=False, default_flow_style=False))
    log.info("S4 complete.")
    log.info("  artifact_id=%s", artifact_id)
    log.info("  wrote %s", feature_contract_path)
