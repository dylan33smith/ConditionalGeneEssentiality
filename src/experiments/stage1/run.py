"""S1 — Data Characterization handler.

Loads the canonical fitness table, v4 components, and embeddings; runs the
six analysis groups; emits 24 required + 1 optional figures, the candidate
protocols YAML, and the tier report scaffolding.

Per REFACTORPLAN §7 S1.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml
from omegaconf import DictConfig

from src.experiments.stage1 import analyses as A
from src.experiments.stage1 import candidates as C
from src.experiments.stage1 import figures as F


log = logging.getLogger(__name__)

CANDIDATE_OUT = Path("data_contract/splits/candidate_protocols.yaml")
TIER_REPORT_OUT = Path("research_log/tier_reports/s1_data_characterization.md")


def main(cfg: DictConfig) -> None:
    log.info("=" * 60)
    log.info("S1 Data Characterization")
    log.info("=" * 60)

    homology_threshold_sigma = float(cfg.stage.s1.get("homology_diagnostic_threshold_sigma", 0.5))
    min_org_support = int(cfg.stage.s1.get("min_org_support_rows", 50_000))

    # ---- Load data ---------------------------------------------------------
    log.info("[1/8] loading canonical fitness")
    fit_df = A.load_fitness()
    log.info("    fit rows=%d, organisms=%d", len(fit_df), fit_df["orgId"].nunique())

    log.info("[2/8] loading v4 Media_Components_ML (Include_in_ml=True)")
    components = A.load_v4_components()
    log.info("    %d component rows across %d media (%d Canonical_IDs)",
             len(components), components["Media"].nunique(),
             components["Canonical_ID"].nunique())

    # ---- Compute analyses --------------------------------------------------
    log.info("[3/8] overlap & coverage matrices")
    org_media = A.org_media_matrix(fit_df)
    org_chem = A.org_chemical_matrix(fit_df, components)
    ubiquity = A.chemical_ubiquity(org_chem)
    log.info("    org × media: %s, org × chem: %s",
             org_media.shape, org_chem.shape)

    log.info("[4/8] support, sparsity, quality")
    rows_per_org = A.rows_per_organism(fit_df)
    cond_per_gene = A.conditions_per_gene(fit_df)
    org_media_rows = A.org_media_row_counts(fit_df)
    genes_per_org = A.genes_per_organism(fit_df)
    per_exp_cor12 = A.per_experiment_cor12(fit_df)
    chem_cov = A.chemistry_coverage_by_org(fit_df, components)
    emb_cov = A.embedding_coverage_by_org(fit_df)
    mode_per_org = A.representation_mode_per_org(fit_df, components)
    log.info("    cond_per_gene rows=%d", len(cond_per_gene))

    # ---- Homology (per organism vs all others) -----------------------------
    log.info("[5/8] homology analysis (cosine to nearest train gene per organism)")
    all_orgs = sorted(fit_df["orgId"].dropna().unique())
    homology_by_org: dict = {}
    homology_orgs = list(rows_per_org["orgId"].head(int(cfg.stage.s1.get("n_homology_orgs", 12))))
    for org in homology_orgs:
        train_orgs = [o for o in all_orgs if o != org]
        log.info("    %s: comparing against %d train orgs (subsampled)", org, len(train_orgs))
        hom = A.cross_org_max_cosine(org, train_orgs, max_train_genes=2000)
        homology_by_org[org] = hom

    # H-HOMO-01 effect-size proxy: per-org median deviation in σ units
    if homology_by_org:
        all_cos = np.concatenate([df["max_cosine"].to_numpy()
                                   for df in homology_by_org.values()])
        global_median = float(np.median(all_cos))
        global_std = float(np.std(all_cos))
        max_dev = max(
            abs(float(df["max_cosine"].median()) - global_median) / max(global_std, 1e-9)
            for df in homology_by_org.values()
        )
        h_homo_01_triggered = max_dev > homology_threshold_sigma
        log.info("    H-HOMO-01: max per-org median deviation = %.3fσ (threshold %.1fσ) → %s",
                 max_dev, homology_threshold_sigma,
                 "TRIGGERED" if h_homo_01_triggered else "not triggered")
    else:
        h_homo_01_triggered = False
        max_dev = 0.0

    # ---- Generate candidate protocols --------------------------------------
    log.info("[6/8] generating candidate protocols")
    candidate_dicts = C.generate_candidates(
        fit_df=fit_df,
        components=components,
        org_chem=org_chem,
        org_media=org_media,
        mode_per_org=mode_per_org,
        homology_by_org=homology_by_org if homology_by_org else None,
        min_val_rows=min_org_support,
    )
    log.info("    generated %d candidates", len(candidate_dicts))
    for c in candidate_dicts:
        log.info("      %s: val=%s, val_chem_seen=%.2f, val_rows=%d",
                 c["protocol_id"], c["val_org_ids"],
                 c["chemistry_overlap"]["val_canonical_id_seen_rate"],
                 c["support"]["val_rows"])

    # ---- Write candidates YAML ---------------------------------------------
    log.info("[7/8] emitting %s", CANDIDATE_OUT)
    out_payload = {
        "status": "populated",
        "emitted_by": "stage1",
        "h_homo_01_triggered": bool(h_homo_01_triggered),
        "h_homo_01_max_deviation_sigma": float(max_dev),
        "homology_threshold_sigma": homology_threshold_sigma,
        "min_org_support_rows": min_org_support,
        "candidates": candidate_dicts,
    }
    CANDIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_OUT.write_text(yaml.safe_dump(out_payload, sort_keys=False,
                                            default_flow_style=False))

    # ---- Generate figures --------------------------------------------------
    log.info("[8/8] generating 24 required + 1 optional figures")
    F.fig_01_org_media_overlap_heatmap(org_media)
    F.fig_02_org_canonical_id_overlap_heatmap(org_chem)
    F.fig_03_org_pair_jaccard_distribution(org_chem)
    F.fig_04_bipartite_org_media_top(org_media)
    F.fig_05_rows_per_organism_bar(rows_per_org)
    F.fig_06_conditions_per_gene_cdf(cond_per_gene)
    F.fig_07_conditions_per_gene_violin(cond_per_gene)
    F.fig_08_org_media_row_count_heatmap(org_media_rows)
    F.fig_09_genes_per_organism_bar(genes_per_org)
    F.fig_10_fit_distribution_per_org_violin(fit_df)
    F.fig_11_t_stat_distribution_per_org(fit_df)
    F.fig_12_cor12_distribution_per_experiment(per_exp_cor12)
    F.fig_13_fit_qq_plot_global(fit_df)
    F.fig_14_chemistry_mapped_unmapped_by_org(chem_cov)
    F.fig_15_embedding_coverage_by_org(emb_cov)
    F.fig_16_canonical_id_prevalence_distribution(ubiquity)
    F.fig_17_chemistry_seen_unseen_rate_per_protocol(candidate_dicts)
    F.fig_18_embedding_cosine_to_nearest_train_per_protocol(homology_by_org, candidate_dicts)
    F.fig_19_homology_similarity_by_org(homology_by_org)
    F.fig_20_representation_mode_proportions_per_org(mode_per_org)
    F.fig_21_representation_mode_per_protocol(candidate_dicts)
    F.fig_22_chemical_ubiquity_histogram(ubiquity)
    F.fig_23_organism_topN_chemical_heatmap(org_chem)
    F.fig_24_chemical_coverage_curve(ubiquity)
    F.fig_25_media_chemistry_umap(components, fit_df)
    log.info("    24 + 1 figures written under research_log/figures/stage1/")

    # ---- Save key analyses as CSV for downstream ---------------------------
    F.fig_dir().joinpath("_data").mkdir(exist_ok=True)
    rows_per_org.to_csv(F.fig_dir() / "_data" / "rows_per_organism.csv", index=False)
    chem_cov.to_csv(F.fig_dir() / "_data" / "chemistry_mapped_coverage.csv", index=False)
    emb_cov.to_csv(F.fig_dir() / "_data" / "embedding_coverage.csv", index=False)
    ubiquity.to_csv(F.fig_dir() / "_data" / "chemical_ubiquity.csv", index=False)
    mode_per_org.to_csv(F.fig_dir() / "_data" / "representation_mode_per_org.csv", index=False)

    log.info("S1 complete.")
    log.info("  candidates → %s", CANDIDATE_OUT)
    log.info("  figures    → %s", F.FIG_DIR)
    log.info("  next: write %s and S1-DEC-001", TIER_REPORT_OUT)
