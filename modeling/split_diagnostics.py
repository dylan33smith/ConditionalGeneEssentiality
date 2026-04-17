"""Split diagnostics: organism protocol + chemistry overlap vs train (Media_Components)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from condition_store import ExperimentConditionEncoding
from data import ArmName, iter_filtered_row_dicts
from embedding_store import EmbeddingStore


def _norm_workbook_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def load_media_to_components(workbook: Path) -> dict[str, set[str]]:
    """Map medium name -> set of component names from Media_Components sheet."""
    mc = pd.read_excel(workbook, sheet_name="Media_Components")
    out: dict[str, set[str]] = {}
    for _, r in mc.iterrows():
        m = _norm_workbook_text(r.get("Media"))
        c = _norm_workbook_text(r.get("Component"))
        if not m or not c:
            continue
        out.setdefault(m, set()).add(c)
    return out


def train_seen_components(
    experiments_parquet: Path,
    train_orgs: set[str],
    media_to_components: dict[str, set[str]],
) -> set[str]:
    """Union of all components appearing in media used by at least one train-organism experiment."""
    ex = pd.read_parquet(experiments_parquet, columns=["orgId", "media"])
    ex = ex[ex["orgId"].isin(train_orgs)]
    seen: set[str] = set()
    for raw in ex["media"].dropna().unique():
        m = _norm_workbook_text(raw)
        if not m:
            continue
        seen.update(media_to_components.get(m, set()))
    return seen


def val_component_ood_stats(
    parquet_path: Path,
    val_orgs: set[str],
    train_seen_components: set[str],
    media_to_components: dict[str, set[str]],
    arm: ArmName,
    embed_store: EmbeddingStore,
    condition_store: ExperimentConditionEncoding,
    *,
    strict_min_cor12: float,
    strict_min_abs_t: float,
    cor12_floor: float,
    weight_t_scale: float,
    max_rows: int | None = None,
) -> dict[str, float | int | str]:
    """Fraction of scored val rows (same filters as training) with any component absent from train media."""
    n_rows = 0
    n_any_unseen = 0
    w_sum = 0.0
    w_unseen = 0.0
    for row in iter_filtered_row_dicts(
        parquet_path,
        val_orgs,
        arm,
        embed_store,
        condition_store,
        strict_min_cor12=strict_min_cor12,
        strict_min_abs_t=strict_min_abs_t,
        cor12_floor=cor12_floor,
        weight_t_scale=weight_t_scale,
        max_rows=max_rows,
    ):
        media = _norm_workbook_text(row.get("media"))
        w = float(row["weight"])
        comps = media_to_components.get(media, set()) if media else set()
        if not comps:
            any_unseen = True
        else:
            any_unseen = any(c not in train_seen_components for c in comps)
        n_rows += 1
        w_sum += w
        if any_unseen:
            n_any_unseen += 1
            w_unseen += w
    if n_rows == 0:
        return {
            "n_val_rows_chemistry_audit": 0,
            "val_frac_rows_any_unseen_component": float("nan"),
            "val_frac_weight_on_unseen_component_rows": float("nan"),
        }
    return {
        "n_val_rows_chemistry_audit": n_rows,
        "val_frac_rows_any_unseen_component": n_any_unseen / n_rows,
        "val_frac_weight_on_unseen_component_rows": w_unseen / w_sum if w_sum > 0 else float("nan"),
    }


def compute_split_chemistry_report(
    *,
    experiments_parquet: Path,
    media_workbook: Path,
    fitness_parquet: Path,
    train_orgs: set[str],
    val_orgs: set[str],
    arm: ArmName,
    embed_store: EmbeddingStore,
    condition_store: ExperimentConditionEncoding,
    strict_min_cor12: float,
    strict_min_abs_t: float,
    cor12_floor: float,
    weight_t_scale: float,
    max_val_rows: int | None,
) -> dict[str, object] | None:
    if not media_workbook.is_file():
        return None
    if not experiments_parquet.is_file():
        return None
    media_map = load_media_to_components(media_workbook)
    t_seen = train_seen_components(experiments_parquet, train_orgs, media_map)
    stats = val_component_ood_stats(
        fitness_parquet,
        val_orgs,
        t_seen,
        media_map,
        arm,
        embed_store,
        condition_store,
        strict_min_cor12=strict_min_cor12,
        strict_min_abs_t=strict_min_abs_t,
        cor12_floor=cor12_floor,
        weight_t_scale=weight_t_scale,
        max_rows=max_val_rows,
    )
    return {
        "split_axis": "organism",
        "media_workbook_chemistry_audit": str(media_workbook),
        "experiments_parquet_for_train_media": str(experiments_parquet),
        "n_distinct_components_in_train_media": len(t_seen),
        "chemistry_audit_note": (
            "train_seen_components = union of Media_Components rows for media strings appearing in "
            "canonical experiments.parquet for train orgs. Val row is 'unseen' if its medium maps to "
            "no components OR any component is not in that train set. Counts use the same val filters "
            "as the training iterator (embedding + condition encoding + arm + non-blank media)."
        ),
        **stats,
    }
