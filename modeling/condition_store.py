"""Load versioned per-(orgId, expName) condition encodings from manifest + Parquet."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from condition_spec import ce_cat_column, ce_cont_column


def _float_cell(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, float) and math.isnan(x):
        return 0.0
    try:
        v = float(x)
        if math.isnan(v):
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


class ExperimentConditionEncoding:
    """Lookup table keyed by (orgId, expName) with integer cat codes + numeric fields."""

    def __init__(self, manifest_path: Path, *, repo_root: Path) -> None:
        mp = Path(manifest_path)
        if not mp.is_absolute():
            mp = repo_root / mp
        if not mp.is_file():
            raise FileNotFoundError(f"Condition manifest not found: {mp}")
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        self._manifest_path = mp
        self.encoding_id = str(manifest["encoding_id"])
        self._canonical_experiments_sha256_expected = manifest.get("canonical_experiments_sha256_expected")
        rel_parquet = str(manifest["parquet_relative"])
        pq_path = repo_root / rel_parquet
        if not pq_path.is_file():
            raise FileNotFoundError(f"Condition encoding Parquet missing: {pq_path}")

        self._cat_fields: tuple[str, ...] = tuple(str(x) for x in manifest["categorical_fields"])
        self._num_fields: tuple[str, ...] = tuple(str(x) for x in manifest["numeric_fields"])
        self._cat_cols: tuple[str, ...] = tuple(ce_cat_column(f) for f in self._cat_fields)
        self._cont_cols: tuple[str, ...] = tuple(ce_cont_column(f) for f in self._num_fields)

        raw_max: dict[str, Any] = manifest["cat_field_max_id"]
        self.cat_field_max_ids: dict[str, int] = {str(k): int(raw_max[k]) for k in self._cat_fields}

        cols = ["orgId", "expName", *self._cat_cols, *self._cont_cols]
        table = pq.read_table(pq_path, columns=cols)
        names = table.column_names
        for c in cols:
            if c not in names:
                raise RuntimeError(f"Condition Parquet missing column {c!r}")

        self._by_key: dict[tuple[str, str], tuple[tuple[int, ...], tuple[float, ...]]] = {}
        d = table.to_pydict()
        n = len(d["orgId"])
        for i in range(n):
            o = d["orgId"][i]
            e = d["expName"][i]
            if o is None or e is None:
                continue
            org = str(o)
            exp = str(e)
            cats = tuple(int(d[c][i]) for c in self._cat_cols)
            conts = tuple(_float_cell(d[c][i]) for c in self._cont_cols)
            self._by_key[(org, exp)] = (cats, conts)

        self._parquet_path = pq_path
        self._n_indexed = len(self._by_key)

    @property
    def cat_field_order(self) -> tuple[str, ...]:
        return self._cat_fields

    @property
    def n_cont_fields(self) -> int:
        return len(self._num_fields)

    def manifest_summary(self) -> dict[str, Any]:
        return {
            "encoding_id": self.encoding_id,
            "n_experiments_indexed": self._n_indexed,
            "parquet_path": str(self._parquet_path),
            "manifest_path": str(self._manifest_path),
            "canonical_experiments_sha256_expected": self._canonical_experiments_sha256_expected,
        }

    def key_set(self) -> frozenset[tuple[str, str]]:
        return frozenset(self._by_key.keys())

    def has(self, org_id: str, exp_name: str) -> bool:
        return (org_id, exp_name) in self._by_key

    def encode(self, org_id: str, exp_name: str) -> tuple[tuple[int, ...], tuple[float, ...]] | None:
        return self._by_key.get((org_id, exp_name))
