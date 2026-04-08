#!/usr/bin/env python3
"""Build canonical Parquet tables (M2): fitness × experiment long + media composition sidecars.

Usage (from repo root):
  python data_processing/build_canonical_v0.py
  python data_processing/build_canonical_v0.py --max-rowid 800000   # smoke test

Outputs (under data/derived/canonical/v0/, gitignored with data/):
  - fitness_experiment_long.parquet
  - experiments.parquet
  - media_master.parquet
  - media_components_long.parquet

Committed manifest:
  docs/canonical_build_manifest_v0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from paths import DERIVED_CANONICAL_V0, FEBA_DB, MEDIA_XLSX, REPO_ROOT

CHUNK_ROWS = 400_000
M0_MANIFEST = REPO_ROOT / "docs" / "data_inputs_manifest_M0.json"
OUT_MANIFEST = REPO_ROOT / "docs" / "canonical_build_manifest_v0.json"


def connect_ro() -> sqlite3.Connection:
    if not FEBA_DB.is_file():
        raise FileNotFoundError(FEBA_DB)
    return sqlite3.connect(f"file:{FEBA_DB.as_posix()}?mode=ro", uri=True)


def sqlite_table_column_kinds(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    """Map column name -> float (numeric) or text (SQLite TEXT / BLOB names)."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    kinds: dict[str, str] = {}
    for row in cur.fetchall():
        _cid, name, col_type, *_rest = row
        ct = (col_type or "").upper()
        if any(x in ct for x in ("INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC")):
            kinds[name] = "float"
        else:
            kinds[name] = "text"
    return kinds


def fitness_long_column_kinds(conn: sqlite3.Connection) -> dict[str, str]:
    g = sqlite_table_column_kinds(conn, "GeneFitness")
    e = sqlite_table_column_kinds(conn, "Experiment")
    out: dict[str, str] = {}
    for k in ("orgId", "locusId", "expName", "fit", "t"):
        out[k] = g[k]
    for k, v in e.items():
        if k in ("orgId", "expName"):
            continue
        out[k] = v
    out["gene_key"] = "text"
    out["abs_t"] = "float"
    out["has_media_composition"] = "bool"
    return out


def apply_sqlite_kinds(df: pd.DataFrame, kinds: dict[str, str]) -> pd.DataFrame:
    """Apply stable dtypes from SQLite PRAGMA so every chunk matches one Parquet schema."""
    df = df.copy()
    for c, kind in kinds.items():
        if c not in df.columns:
            continue
        if kind == "float":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        elif kind == "bool":
            df[c] = df[c].astype("bool")
        else:
            s = df[c].replace("", pd.NA).replace(r"^\s*$", pd.NA, regex=True)

            def _cell(x):
                if x is None:
                    return pd.NA
                if isinstance(x, float) and np.isnan(x):
                    return pd.NA
                if pd.isna(x):
                    return pd.NA
                return str(x)

            df[c] = s.map(_cell).astype("string")
    return df


def experiment_select_sql(conn: sqlite3.Connection) -> str:
    cur = conn.execute("PRAGMA table_info(Experiment)")
    cols = [row[1] for row in cur.fetchall()]
    skip = {"orgId", "expName"}
    exp_cols = [c for c in cols if c not in skip]
    parts = ["gf.orgId", "gf.locusId", "gf.expName", "gf.fit", "gf.t"]
    parts += [f"e.{c}" for c in exp_cols]
    return ", ".join(parts)


def enrich_chunk(df: pd.DataFrame, media_with_components: set[str]) -> pd.DataFrame:
    df = df.copy()
    df["gene_key"] = df["orgId"].astype(str) + ":" + df["locusId"].astype(str)
    df["abs_t"] = pd.to_numeric(df["t"], errors="coerce").astype(float).abs()
    med = df["media"].map(lambda x: str(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) and not pd.isna(x) else "")
    df["has_media_composition"] = med.isin(media_with_components)
    return df


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_m0_input_hashes() -> dict:
    if not M0_MANIFEST.is_file():
        return {}
    data = json.loads(M0_MANIFEST.read_text(encoding="utf-8"))
    out = {}
    for item in data.get("required_inputs", []):
        out[item["path"]] = item.get("sha256")
    return out


def coerce_excel_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Excel → Parquet: normalize object columns only."""
    df = df.copy()
    for c in list(df.columns):
        if df[c].dtype != object:
            continue
        s = df[c].replace("", pd.NA).replace(r"^\s*$", pd.NA, regex=True)
        num = pd.to_numeric(s, errors="coerce")
        non_null_mask = s.notna()
        if int(non_null_mask.sum()) == 0:
            df[c] = pd.Series(pd.NA, index=df.index, dtype="string")
            continue
        converted = int((num.notna() & non_null_mask).sum())
        if converted == int(non_null_mask.sum()):
            df[c] = num.astype("float64")
        else:
            df[c] = s.map(
                lambda x: pd.NA
                if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x)
                else str(x)
            ).astype("string")
    return df


def write_small_parquet(df: pd.DataFrame, path: Path, *, sqlite_kinds: dict[str, str] | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_kinds is not None:
        df = apply_sqlite_kinds(df, sqlite_kinds)
    else:
        df = coerce_excel_object_columns(df)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    return len(df)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-rowid",
        type=int,
        default=None,
        help="Stop after this GeneFitness.rowid (inclusive). For smoke tests.",
    )
    args = parser.parse_args()

    DERIVED_CANONICAL_V0.mkdir(parents=True, exist_ok=True)

    media_comp = pd.read_excel(MEDIA_XLSX, sheet_name="Media_Components", header=0)
    media_master = pd.read_excel(MEDIA_XLSX, sheet_name="Media", header=0)

    media_with_components = set(media_comp["Media"].dropna().astype(str).unique())

    n_media_comp = write_small_parquet(media_comp, DERIVED_CANONICAL_V0 / "media_components_long.parquet")
    n_media_master = write_small_parquet(media_master, DERIVED_CANONICAL_V0 / "media_master.parquet")

    conn = connect_ro()
    try:
        kinds_exp = sqlite_table_column_kinds(conn, "Experiment")
        kinds_long = fitness_long_column_kinds(conn)

        experiments = pd.read_sql_query("SELECT * FROM Experiment", conn)
        n_exp = write_small_parquet(
            experiments, DERIVED_CANONICAL_V0 / "experiments.parquet", sqlite_kinds=kinds_exp
        )

        cur = conn.execute("SELECT MAX(rowid) FROM GeneFitness")
        max_rowid = int(cur.fetchone()[0])
        cur = conn.execute(
            "SELECT COUNT(*) FROM GeneFitness gf INNER JOIN Experiment e ON "
            "gf.orgId = e.orgId AND gf.expName = e.expName"
        )
        expected_join_rows = int(cur.fetchone()[0])

        effective_max = max_rowid if args.max_rowid is None else min(max_rowid, args.max_rowid)
        select_list = experiment_select_sql(conn)

        out_path = DERIVED_CANONICAL_V0 / "fitness_experiment_long.parquet"
        if out_path.is_file():
            out_path.unlink()

        writer: pq.ParquetWriter | None = None
        total_written = 0
        t0 = time.perf_counter()
        row_start = 1
        while row_start <= effective_max:
            row_end = min(row_start + CHUNK_ROWS - 1, effective_max)
            q = f"""
            SELECT {select_list}
            FROM GeneFitness gf
            INNER JOIN Experiment e ON gf.orgId = e.orgId AND gf.expName = e.expName
            WHERE gf.rowid BETWEEN {row_start} AND {row_end}
            """
            chunk = pd.read_sql_query(q, conn)
            if chunk.empty:
                row_start = row_end + 1
                continue
            chunk = enrich_chunk(chunk, media_with_components)
            chunk = apply_sqlite_kinds(chunk, kinds_long)
            if chunk["orgId"].isna().any() or chunk["expName"].isna().any():
                raise RuntimeError("Null join keys in output chunk")
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            total_written += len(chunk)
            elapsed = time.perf_counter() - t0
            rate = total_written / elapsed if elapsed > 0 else 0
            print(
                f"rowid {row_start}-{row_end}: +{len(chunk)} rows "
                f"(total {total_written}, {rate:,.0f} rows/s)",
                flush=True,
            )
            row_start = row_end + 1
        if writer is not None:
            writer.close()
    finally:
        conn.close()

    if args.max_rowid is not None and total_written != expected_join_rows:
        print(
            f"Note: partial build (--max-rowid); wrote {total_written} / {expected_join_rows} rows.",
            file=sys.stderr,
        )
    elif args.max_rowid is None and total_written != expected_join_rows:
        raise RuntimeError(f"Row count mismatch: wrote {total_written}, expected {expected_join_rows}")

    outputs_meta = []
    for rel_name in [
        "fitness_experiment_long.parquet",
        "experiments.parquet",
        "media_master.parquet",
        "media_components_long.parquet",
    ]:
        p = DERIVED_CANONICAL_V0 / rel_name
        st = p.stat()
        outputs_meta.append(
            {
                "path": str(p.relative_to(REPO_ROOT)),
                "bytes": st.st_size,
                "sha256": file_sha256(p),
                "rows": 0,
            }
        )

    for o in outputs_meta:
        name = Path(o["path"]).name
        if name == "fitness_experiment_long.parquet":
            o["rows"] = total_written
        elif name == "experiments.parquet":
            o["rows"] = n_exp
        elif name == "media_master.parquet":
            o["rows"] = n_media_master
        elif name == "media_components_long.parquet":
            o["rows"] = n_media_comp

    manifest = {
        "manifest_version": 1,
        "build_id": "canonical_v0",
        "description": "GeneFitness INNER JOIN Experiment; media composition from Excel sidecars.",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "chunk_rows": CHUNK_ROWS,
        "max_rowid_effective": effective_max,
        "inputs_expected_sha256_from_M0": load_m0_input_hashes(),
        "outputs": outputs_meta,
        "dtypes_policy": "SQLite PRAGMA table_info: INT/REAL → float64; TEXT → pandas string; derived gene_key text, abs_t float64, has_media_composition bool.",
        "columns_fitness_long_note": "gf orgId,locusId,expName,fit,t plus all Experiment columns except orgId/expName; plus gene_key, abs_t, has_media_composition.",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest {OUT_MANIFEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
