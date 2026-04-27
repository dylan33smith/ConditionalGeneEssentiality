"""Candidate protocol generation for S1 → S3.

Generates 3-5 candidate val/test partitions stratified by chemistry overlap
and support, per REFACTORPLAN §7 S1. Each candidate includes the full schema
defined in data_contract/splits/candidate_protocols.yaml (rationale, support,
chemistry_overlap, representation_mode_proportions, optional homology summary).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.stage1 import analyses as A


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chemistry_seen_rate(val_orgs: list[str], train_orgs: list[str],
                       org_chem: pd.DataFrame) -> dict[str, float]:
    """Compute Canonical_ID seen-rate for val orgs.

    seen_rate = |val_chemistry ∩ train_chemistry| / |val_chemistry|
    """
    binary = (org_chem > 0).astype(bool)
    val_chem = set()
    for o in val_orgs:
        if o in binary.index:
            val_chem |= set(binary.columns[binary.loc[o].values])
    train_chem = set()
    for o in train_orgs:
        if o in binary.index:
            train_chem |= set(binary.columns[binary.loc[o].values])
    if not val_chem:
        return {"seen": 0.0, "unseen": 0.0, "n_val_chem": 0, "n_seen": 0}
    seen = val_chem & train_chem
    return {
        "seen": float(len(seen) / len(val_chem)),
        "unseen": float(1 - len(seen) / len(val_chem)),
        "n_val_chem": len(val_chem),
        "n_seen": len(seen),
    }


def media_seen_rate(val_orgs: list[str], train_orgs: list[str],
                    org_media: pd.DataFrame) -> dict[str, float]:
    return chemistry_seen_rate(val_orgs, train_orgs, org_media)  # same shape, different unit


def split_support(fit_df: pd.DataFrame, val_orgs: list[str], test_orgs: list[str]) -> dict[str, int]:
    val_set = set(val_orgs)
    test_set = set(test_orgs)
    train_mask = ~fit_df["orgId"].isin(val_set | test_set)
    val_mask = fit_df["orgId"].isin(val_set)
    test_mask = fit_df["orgId"].isin(test_set)
    return {
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "val_n_genes": int(fit_df.loc[val_mask, "gene_key"].nunique()),
        "test_n_genes": int(fit_df.loc[test_mask, "gene_key"].nunique()),
        "val_n_conditions": int(fit_df.loc[val_mask, "expName"].nunique()),
        "test_n_conditions": int(fit_df.loc[test_mask, "expName"].nunique()),
    }


def representation_mode_proportions_for_split(fit_df: pd.DataFrame,
                                              val_orgs: list[str],
                                              test_orgs: list[str],
                                              mode_per_org: pd.DataFrame) -> dict:
    """Aggregated representation_mode proportions for train/val/test partitions.

    mode_per_org rows: orgId + mode columns, values are weighted row-counts.
    """
    mode_cols = [c for c in mode_per_org.columns if c != "orgId"]
    val_set, test_set = set(val_orgs), set(test_orgs)

    def part(orgs: set) -> dict:
        sub = mode_per_org[mode_per_org["orgId"].isin(orgs)]
        totals = sub[mode_cols].sum()
        s = float(totals.sum())
        if s <= 0:
            return {c: 0.0 for c in mode_cols}
        return {c: float(totals[c] / s) for c in mode_cols}

    train_set = set(mode_per_org["orgId"]) - val_set - test_set
    return {
        "train": part(train_set),
        "val": part(val_set),
        "test": part(test_set),
    }


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    protocol_id: str
    description: str
    val_org_ids: list[str]
    test_org_ids: list[str]
    selection_strategy: str
    notes: str = ""
    homology_summary: dict | None = None

    def to_dict(self, support: dict, chem_overlap: dict, mode_props: dict) -> dict:
        out: dict[str, Any] = {
            "protocol_id": self.protocol_id,
            "description": self.description,
            "val_org_ids": list(self.val_org_ids),
            "test_org_ids": list(self.test_org_ids),
            "selection_strategy": self.selection_strategy,
            "support": support,
            "chemistry_overlap": chem_overlap,
            "representation_mode_proportions": mode_props,
            "notes": self.notes,
        }
        if self.homology_summary is not None:
            out["homology_summary"] = self.homology_summary
        return out


def generate_candidates(fit_df: pd.DataFrame, components: pd.DataFrame,
                       org_chem: pd.DataFrame,
                       org_media: pd.DataFrame,
                       mode_per_org: pd.DataFrame,
                       homology_by_org: dict[str, pd.DataFrame] | None = None,
                       homology_high_sim_threshold: float = 0.85,
                       *,
                       min_val_rows: int = 50_000,
                       min_val_genes: int = 1_000) -> list[dict]:
    """Generate 3-5 candidate split protocols stratified by overlap and support.

    Strategy:
      1. largest-by-rows         val=biggest org, test=second-biggest
      2. mid-overlap-stratified  multi-org val with ~50-70% chemistry overlap
      3. high-overlap-easy       multi-org val with >80% overlap
      4. low-overlap-stress      single-org val with the LOWEST overlap (stress test)
      5. multi-org-balanced      4 val orgs spanning the overlap distribution
    """
    rows = A.rows_per_organism(fit_df)
    org_to_rows = dict(zip(rows["orgId"], rows["n_rows"]))
    eligible = [o for o in rows["orgId"] if org_to_rows.get(o, 0) >= min_val_rows]

    # Per-org overlap when held out individually
    all_orgs = sorted(org_chem.index)
    overlap_per_org = {}
    for o in all_orgs:
        train = [x for x in all_orgs if x != o]
        rate = chemistry_seen_rate([o], train, org_chem)
        overlap_per_org[o] = rate["seen"]

    candidates: list[Candidate] = []

    # 1. largest-by-rows
    top2 = list(rows["orgId"].head(2))
    if len(top2) >= 2:
        candidates.append(Candidate(
            protocol_id="largest_by_rows",
            description=f"Single largest val ({top2[0]}) + second-largest test ({top2[1]})",
            val_org_ids=[top2[0]],
            test_org_ids=[top2[1]],
            selection_strategy="largest-by-rows",
            notes="High val/test mass; chemistry overlap may be high or low depending on organism.",
        ))

    # Build a mid-overlap multi-org val by picking orgs in overlap range
    eligible_with_overlap = [(o, overlap_per_org.get(o, 0.0)) for o in eligible
                              if o not in (top2[:1] + top2[1:2])]
    mid = [o for o, ov in eligible_with_overlap if 0.5 <= ov <= 0.7]
    high = [o for o, ov in eligible_with_overlap if ov > 0.85]
    low_pool = sorted(eligible_with_overlap, key=lambda x: x[1])

    # 2. mid-overlap-stratified
    if len(mid) >= 2:
        val4 = mid[:2]
        test1 = mid[2] if len(mid) > 2 else mid[0]
        candidates.append(Candidate(
            protocol_id="mid_overlap_stratified",
            description=f"Multi-org val ({val4}) chosen for ~50-70% chemistry overlap with train",
            val_org_ids=val4,
            test_org_ids=[test1] if test1 not in val4 else [],
            selection_strategy="stratified-overlap-mid",
            notes="Goldilocks zone: not too easy, not impossible.",
        ))

    # 3. high-overlap-easy
    if len(high) >= 2:
        val2 = high[:2]
        test1 = high[2] if len(high) > 2 else high[0]
        candidates.append(Candidate(
            protocol_id="high_overlap_easy",
            description=f"Multi-org val ({val2}) with >85% chemistry overlap; sanity-check protocol",
            val_org_ids=val2,
            test_org_ids=[test1] if test1 not in val2 else [],
            selection_strategy="stratified-overlap-high",
            notes="Easy protocol — establishes whether the model can solve the in-distribution case.",
        ))

    # 4. low-overlap-stress
    if low_pool:
        stress = low_pool[0][0]
        # pick a test from the next-lowest, ensuring distinct
        test_pick = next((o for o, _ in low_pool[1:] if o != stress), None)
        candidates.append(Candidate(
            protocol_id="low_overlap_stress",
            description=f"Stress test: lowest-overlap val ({stress}, {low_pool[0][1]:.0%} chemistry overlap)",
            val_org_ids=[stress],
            test_org_ids=[test_pick] if test_pick else [],
            selection_strategy="low-overlap-stress",
            notes=("Reported but NOT promotion-gating per §7 S2 — likely impossible; "
                   "diagnostic only."),
        ))

    # 5. multi-org-balanced (4 orgs spanning the overlap distribution)
    if len(eligible_with_overlap) >= 5:
        sorted_pool = sorted(eligible_with_overlap, key=lambda x: x[1])
        # take 4 quartile-spaced
        quartiles_idx = [int(len(sorted_pool) * q) for q in (0.15, 0.40, 0.65, 0.90)]
        quartile_orgs = [sorted_pool[i][0] for i in quartiles_idx]
        # ensure distinct from val candidates already chosen
        # simple test pick from remaining
        chosen_set = set(quartile_orgs)
        test_pool = [o for o, _ in sorted_pool if o not in chosen_set]
        test_pick = test_pool[len(test_pool) // 2] if test_pool else None
        candidates.append(Candidate(
            protocol_id="multi_org_balanced",
            description=f"Four val orgs spanning overlap quartiles ({quartile_orgs})",
            val_org_ids=quartile_orgs,
            test_org_ids=[test_pick] if test_pick else [],
            selection_strategy="quartile-stratified",
            notes="Balanced view across the overlap distribution.",
        ))

    # ----- enrich each candidate with metrics ---------------------------------
    out: list[dict] = []
    for cand in candidates:
        all_orgs_set = set(fit_df["orgId"].dropna().unique())
        train_orgs = sorted(all_orgs_set - set(cand.val_org_ids) - set(cand.test_org_ids))

        media_overlap_val = media_seen_rate(cand.val_org_ids, train_orgs, org_media)
        chem_overlap_val = chemistry_seen_rate(cand.val_org_ids, train_orgs, org_chem)
        media_overlap_test = media_seen_rate(cand.test_org_ids, train_orgs, org_media) if cand.test_org_ids else {"seen": 0.0, "unseen": 0.0, "n_val_chem": 0, "n_seen": 0}
        chem_overlap_test = chemistry_seen_rate(cand.test_org_ids, train_orgs, org_chem) if cand.test_org_ids else {"seen": 0.0, "unseen": 0.0, "n_val_chem": 0, "n_seen": 0}

        chem_overlap = {
            "val_seen_rate": media_overlap_val["seen"],
            "val_unseen_rate": media_overlap_val["unseen"],
            "test_seen_rate": media_overlap_test["seen"],
            "test_unseen_rate": media_overlap_test["unseen"],
            "val_canonical_id_seen_rate": chem_overlap_val["seen"],
            "val_canonical_id_unseen_rate": chem_overlap_val["unseen"],
            "test_canonical_id_seen_rate": chem_overlap_test["seen"],
            "test_canonical_id_unseen_rate": chem_overlap_test["unseen"],
        }

        support = split_support(fit_df, cand.val_org_ids, cand.test_org_ids)
        mode_props = representation_mode_proportions_for_split(
            fit_df, cand.val_org_ids, cand.test_org_ids, mode_per_org
        )

        # Optional homology
        if homology_by_org is not None:
            val_high = []
            test_high = []
            for o in cand.val_org_ids:
                if o in homology_by_org:
                    h = homology_by_org[o]
                    val_high.append(float((h["max_cosine"] > homology_high_sim_threshold).mean()))
            for o in cand.test_org_ids:
                if o in homology_by_org:
                    h = homology_by_org[o]
                    test_high.append(float((h["max_cosine"] > homology_high_sim_threshold).mean()))
            cand.homology_summary = {
                "val_high_sim_fraction": float(np.mean(val_high)) if val_high else None,
                "test_high_sim_fraction": float(np.mean(test_high)) if test_high else None,
                "similarity_threshold": homology_high_sim_threshold,
            }

        out.append(cand.to_dict(support, chem_overlap, mode_props))

    return out
