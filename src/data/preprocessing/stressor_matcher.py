"""Match stressor strings from experiments to workbook Canonical_ID entries.

Used by S4 Option D to avoid duplicate canonical slots for surface-form variants
(e.g. ``Ethanol`` vs workbook ``Ethanol``). See OPEN-001 / S4-DEC-002.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

# Longest / most specific suffixes first (iterative stripping order).
_CHEMICAL_SUFFIX_TOKENS: tuple[str, ...] = (
    "pentahydrate",
    "hexahydrate",
    "decahydrate",
    "tetrahydrate",
    "trihydrate",
    "dihydrate",
    "monohydrate",
    "hydrate",
    "dihydrochloride",
    "hydrochloride",
    "hydrochloride hydrate",
    "hydrochloride monohydrate",
    "sulfate salt",
    "sodium salt",
    "disodium salt",
    "trisodium salt",
    "potassium salt",
    "lithium salt",
    "calcium salt",
    "magnesium salt",
    "sodium",
    "salt",
)


def normalize_chemical_name(s: str) -> str:
    """Case-fold, collapse whitespace, strip trailing chemical suffix tokens."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = str(s).strip().lower()
    t = re.sub(r"\s+", " ", t)
    changed = True
    while changed and t:
        changed = False
        for suf in _CHEMICAL_SUFFIX_TOKENS:
            for sep in (" ", ", ", ","):
                suffix = sep + suf
                if t.endswith(suffix):
                    t = t[: -len(suffix)].rstrip(" ,;")
                    changed = True
                    break
            if changed:
                break
    return t.strip()


def _token_sort_ratio(a: str, b: str) -> float:
    ta = " ".join(sorted(a.split()))
    tb = " ".join(sorted(b.split()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return float(SequenceMatcher(None, ta, tb).ratio())


def _try_import_rapidfuzz_token_sort_ratio() -> Any | None:
    try:
        from rapidfuzz import fuzz  # type: ignore[import-not-found]

        return fuzz.token_sort_ratio
    except ImportError:
        return None


_rapidfuzz_tsr = _try_import_rapidfuzz_token_sort_ratio()


def build_workbook_stressor_index(components_df: pd.DataFrame) -> dict[str, str]:
    """Map normalized surface strings -> Canonical_ID (deterministic on ties).

    Indexes ``Canonical_ID``, ``Compound_name``, and ``Source_row_component`` when present.
    """
    df = components_df.copy()
    if "Include_in_ml" in df.columns:
        df = df[df["Include_in_ml"] == True].copy()  # noqa: E712
    canon_col = "Canonical_ID"
    rows: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        cid = str(row[canon_col]).strip()
        for col in ("Canonical_ID", "Compound_name", "Source_row_component"):
            if col not in row.index:
                continue
            v = row[col]
            if pd.isna(v):
                continue
            ns = normalize_chemical_name(str(v))
            if ns:
                rows.append((ns, cid))
    rows.sort(key=lambda x: (x[0], x[1]))
    out: dict[str, str] = {}
    for ns, cid in rows:
        out.setdefault(ns, cid)
    return out


def exact_match_after_normalize(query: str, index: dict[str, str]) -> str | None:
    nq = normalize_chemical_name(query)
    if not nq:
        return None
    return index.get(nq)


def fuzzy_top_canonicals(
    query: str,
    index: dict[str, str],
    *,
    k: int = 3,
) -> list[tuple[str, float]]:
    """Return up to ``k`` (canonical_id, score) pairs sorted by descending score."""
    nq = normalize_chemical_name(query)
    if not nq:
        return []
    best_per_canon: dict[str, float] = {}
    if _rapidfuzz_tsr is not None:
        for ns, cid in index.items():
            score = _rapidfuzz_tsr(nq, ns) / 100.0
            prev = best_per_canon.get(cid, 0.0)
            if score > prev:
                best_per_canon[cid] = score
    else:
        for ns, cid in index.items():
            score = _token_sort_ratio(nq, ns)
            prev = best_per_canon.get(cid, 0.0)
            if score > prev:
                best_per_canon[cid] = score
    ranked = sorted(best_per_canon.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:k]


@dataclass(frozen=True)
class StressorMatchRow:
    stressor: str
    train_freq: int
    match_type: str
    top1_canonical_id: str
    top1_score: float
    top2_canonical_id: str
    top2_score: float
    top3_canonical_id: str
    top3_score: float
    auto_apply: bool


def build_candidate_report(
    stressors_with_freq: dict[str, int],
    components_df: pd.DataFrame,
    *,
    auto_fuzzy_threshold: float = 0.95,
    fuzzy_report_min_score: float = 0.35,
) -> pd.DataFrame:
    """One row per unique stressor string with exact/fuzzy workbook matches."""
    index = build_workbook_stressor_index(components_df)
    records: list[dict[str, Any]] = []
    for stressor in sorted(stressors_with_freq.keys(), key=lambda s: (-stressors_with_freq[s], s)):
        freq = int(stressors_with_freq[stressor])
        exact = exact_match_after_normalize(stressor, index)
        if exact is not None:
            mt = "exact_after_norm"
            t1, s1 = exact, 1.0
            t2, s2 = "", 0.0
            t3, s3 = "", 0.0
            auto = True
        else:
            fuzzy = fuzzy_top_canonicals(stressor, index, k=3)
            if fuzzy and fuzzy[0][1] >= fuzzy_report_min_score:
                mt = "fuzzy"
                (t1, s1) = fuzzy[0]
                t2, s2 = (fuzzy[1] if len(fuzzy) > 1 else ("", 0.0))
                t3, s3 = (fuzzy[2] if len(fuzzy) > 2 else ("", 0.0))
                auto = bool(s1 >= auto_fuzzy_threshold)
            else:
                mt = "none"
                t1, s1 = "", 0.0
                t2, s2 = "", 0.0
                t3, s3 = "", 0.0
                auto = False
        records.append(
            {
                "stressor": stressor,
                "train_freq": freq,
                "match_type": mt,
                "top1_canonical_id": t1,
                "top1_score": round(s1, 6),
                "top2_canonical_id": t2,
                "top2_score": round(s2, 6),
                "top3_canonical_id": t3,
                "top3_score": round(s3, 6),
                "auto_apply": auto,
            }
        )
    return pd.DataFrame.from_records(records)


def build_ratified_mapping_from_report(
    report_df: pd.DataFrame,
    *,
    fuzzy_auto_threshold: float = 0.95,
) -> dict[str, str]:
    """Build stressor string -> Canonical_ID map for ratified YAML (auto tier).

    Includes: all ``exact_after_norm`` rows and fuzzy rows with
    ``top1_score >= fuzzy_auto_threshold`` (default 0.95). Everything else is
    left unmapped so the chemistry builder uses ``normalize_chemical_name`` as a
    new slot unless the human fills ``<NEW>`` entries in the draft YAML.
    """
    out: dict[str, str] = {}
    for row in report_df.itertuples(index=False):
        stressor = str(row.stressor)
        mt = str(row.match_type)
        s1 = float(row.top1_score)
        t1 = str(row.top1_canonical_id)
        if mt == "exact_after_norm" and t1:
            out[stressor] = t1
        elif mt == "fuzzy" and t1 and s1 >= fuzzy_auto_threshold:
            out[stressor] = t1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def build_draft_yaml_payload(
    report_df: pd.DataFrame,
    *,
    ratified: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Draft YAML structure: explicit map + placeholders for human completion."""
    ratified = ratified or build_ratified_mapping_from_report(report_df)
    pending: dict[str, str] = {}
    for row in report_df.itertuples(index=False):
        s = str(row.stressor)
        if s not in ratified:
            pending[s] = "<NEW>"
    return {
        "version": 1,
        "description": "Maps condition_* stressor display strings to workbook Canonical_ID. "
        "Omit keys to fall back to normalize_chemical_name(stressor) as a new slot.",
        "map": {**ratified, **pending},
    }
