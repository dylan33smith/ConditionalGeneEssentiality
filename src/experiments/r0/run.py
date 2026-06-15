"""R0 — Data Characterization handler.

Invoked by src/cli/run_experiment.py when stage_or_tier == "R0".
Runs all 10 analyses, emits figures + CSVs + candidate protocols + report.
"""
from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from src.experiments.r0 import analyses as A
from src.experiments.r0 import candidates as C
from src.experiments.r0 import figures as F


log = logging.getLogger(__name__)

REPORT_OUT = Path("research_log/tier_reports/r0_data_characterization.md")


def main(cfg: DictConfig) -> None:
    log.info("=" * 60)
    log.info("R0 — Data Characterization (Ranking Regime)")
    log.info("=" * 60)

    r0_cfg = cfg.get("r0", {})
    m_grid = list(r0_cfg.get("m_min_grid", [3, 5, 10, 15, 20, 30, 50]))
    tail_grid = list(r0_cfg.get("tail_min_grid",
                                [0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.50]))
    iqr_grid = list(r0_cfg.get("iqr_min_grid", [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]))
    split_preview = r0_cfg.get("split_preview", {})
    holdout_fractions = list(split_preview.get(
        "holdout_fractions", [0.10, 0.15, 0.20, 0.25, 0.30]))
    min_holdout = int(split_preview.get("min_holdout", 3))
    max_holdout = int(split_preview.get("max_holdout", 30))
    max_orgs_noise = int(r0_cfg.get("noise_floor_max_orgs", 24))
    seed = int(r0_cfg.get("seed", 0))

    log.info("[1/12] loading canonical fitness")
    fit_df = A.load_fitness()
    log.info("    rows=%d, organisms=%d, experiments=%d",
             len(fit_df), fit_df["orgId"].nunique(), fit_df["expName"].nunique())

    log.info("[2/12] A: conditions per gene (m_g)")
    cond_per_gene = A.conditions_per_gene(fit_df)

    log.info("[3/12] B: IQR per gene")
    iqr_df = A.iqr_per_gene(fit_df)

    log.info("[4/12] D: replicate noise floor (max_orgs=%d)", max_orgs_noise)
    noise_df = A.replicate_noise_floor(fit_df, max_orgs=max_orgs_noise)
    noise_summary = A.per_org_noise_summary(noise_df)
    log.info("    noise rows=%d across %d orgs", len(noise_df),
             noise_df["orgId"].nunique() if not noise_df.empty else 0)

    log.info("[5/12] E: signal-to-noise per gene")
    snr_df = A.signal_to_noise(iqr_df, noise_summary)

    log.info("[6/12] F: eligibility frontier (m_grid=%s, tail_grid=%s)",
             m_grid, tail_grid)
    frontier = A.eligibility_frontier(iqr_df, m_grid, tail_grid, metric="tail_g")
    # Diagnostic frontier on legacy IQR metric — for cross-comparison
    frontier_iqr = A.eligibility_frontier(iqr_df, m_grid, iqr_grid, metric="iqr_g")
    frontier_iqr.to_csv("research_log/figures/r0_data/06b_eligibility_frontier_iqr_diagnostic.csv",
                         index=False)

    log.info("[7/12] G: experiment structure per org")
    exp_struct = A.experiment_structure(fit_df)

    # H (low-IQR Jaccard across orgs) is DEFERRED: gene_key is org-prefixed
    # in the canonical data (e.g. 'Keio:14146' vs 'BFirm:BPHYT_RS00020'), so
    # the gene-sets of any two orgs are disjoint by construction and the
    # Jaccard is identically zero off-diagonal. H-R0-03 requires a homology
    # mapping (analog of S3's 0.85 cosine cutoff) to be testable; that's
    # out of R0 scope. Tracked in ARCHITECTURE.md as deferred.

    log.info("[8/12] I: condition discriminability")
    disc = A.condition_discriminability(fit_df)

    log.info("[9/12] J: split feasibility preview (fractions=%s)", holdout_fractions)
    split_df = A.split_feasibility(
        fit_df, holdout_fractions, seed=seed,
        min_holdout=min_holdout, max_holdout=max_holdout,
        replicate_group_together=True,
    )

    log.info("[10/12] K: expGroup coverage (R-LOCK-2 stratification feasibility)")
    eg_coverage = A.expgroup_coverage(fit_df)
    eg_summary = A.expgroup_summary(eg_coverage, min_holdout_per_group=2)

    all_orgs = sorted(cond_per_gene["orgId"].unique())

    log.info("[11/12] writing figures")
    F.fig_01_m_per_gene(cond_per_gene)
    F.fig_02_iqr_per_gene(iqr_df)
    F.fig_03_joint_m_iqr(iqr_df)
    F.fig_04_replicate_noise(noise_df, noise_summary, all_orgs=all_orgs)
    F.fig_05_snr(snr_df, all_orgs=all_orgs)
    F.fig_06_eligibility_frontier(frontier)
    # fig 07 dropped — see analysis comment above
    F.fig_08_experiment_structure(exp_struct)
    F.fig_09_condition_discriminability(disc)
    F.fig_10_split_feasibility(split_df)
    F.fig_11_expgroup_coverage(eg_coverage, eg_summary)

    log.info("[12/12] emitting candidate protocols")
    elig_cands = C.build_eligibility_candidates(frontier, noise_summary)
    split_cands = C.build_split_candidates(split_df)
    C.write_candidates(elig_cands, split_cands)
    log.info("    %d eligibility candidates, %d split candidates",
             len(elig_cands), len(split_cands))

    _write_report(cond_per_gene, iqr_df, noise_summary, frontier,
                  exp_struct, elig_cands, split_cands)
    log.info("DONE. Figures: research_log/figures/r0_data/  "
             "Candidates: data_contract/ranking/r0_candidates.yaml  "
             "Report: %s", REPORT_OUT)


def _write_report(cond_per_gene, iqr_df, noise_summary, frontier,
                  exp_struct, elig_cands, split_cands) -> None:
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# R0 — Data Characterization Report",
        "",
        "Source: `R0-A_data_characterization`. See `ARCHITECTURE.md` for spec.",
        "",
        "## Per-organism summary",
        "",
        f"- Orgs analyzed: {cond_per_gene['orgId'].nunique()}",
        f"- Total (org, gene) pairs: {len(cond_per_gene):,}",
        f"- Total experiments: {int(exp_struct['n_experiments'].sum()) if not exp_struct.empty else 'NA'}",
        "",
        "## IQR summary",
        "",
        f"- Median IQR_g across all (org, gene): {iqr_df['iqr_g'].median():.3f}",
        f"- 25th percentile IQR_g: {iqr_df['iqr_g'].quantile(0.25):.3f}",
        f"- 75th percentile IQR_g: {iqr_df['iqr_g'].quantile(0.75):.3f}",
        f"- Fraction of (org, gene) with IQR < 0.10: "
        f"{float((iqr_df['iqr_g'] < 0.10).mean()):.3f} "
        f"(distribution is moderately centered, not a bimodal housekeeping spike — "
        f"truly non-conditional genes are a small minority)",
        "",
        "## Replicate noise floor",
        "",
    ]
    if noise_summary.empty:
        lines.append("- No replicate data computed.")
    else:
        n_with_noise = len(noise_summary)
        n_total = cond_per_gene["orgId"].nunique()
        lines.append(f"- Median cross-replicate Spearman across orgs: "
                     f"{noise_summary['median'].median():.3f}")
        lines.append(f"- Range: [{noise_summary['median'].min():.3f}, "
                     f"{noise_summary['median'].max():.3f}]")
        lines.append(f"- Orgs with replicate data: **{n_with_noise} of {n_total}**. "
                     f"The other {n_total - n_with_noise} orgs ran each (expDesc, media) "
                     f"once (no biological replicates) and default to the IQR floor "
                     f"(0.10) in R-LOCK-1.")

    lines += [
        "",
        "## Experiment structure (why and what)",
        "",
        "**Why this matters.** The ranking task ranks distinct *conditions* within a gene,",
        "but the raw data is at the `expName` (assay) level — and most conditions were",
        "run multiple times as biological replicates (different `expName`s sharing the same",
        "`(expDesc, media)`). Three downstream decisions depend on this structure:",
        "",
        "1. **Split protocol (R-LOCK-2):** holding out random `expName`s would silently leak",
        "   conditions into train via their replicate siblings. The split MUST hold out at",
        "   the condition level. The DvH numbers (757 assays → 248 conditions; 99% of",
        "   conditions have replicates) show how severe this would be.",
        "2. **Replicate handling (R-LOCK-3):** train can treat each replicate as an",
        "   independent noisy observation of the same target; val must mean-pool",
        "   replicates before per-gene Spearman or the metric is inflated by intra-condition",
        "   noise.",
        "3. **Eligibility (R-LOCK-1):** `m_g` (conditions per gene) is at the condition",
        "   level, not the `expName` level — otherwise replicate count would inflate",
        "   it and under-filter low-condition genes.",
        "",
        "**What's reported (per org, in `08_experiment_structure.csv`):**",
        "",
        "- `n_experiments` — distinct assays.",
        "- `n_conditions` — distinct `(expDesc, media)` pairs (the ranking unit).",
        "- `n_media` — distinct media base names (coarser than condition).",
        "- `exp_per_cond_p50, p90, max` — replicate-depth distribution. p50≥2 means",
        "  replication is the norm; p50=1 means most conditions are singletons.",
        "- `frac_cond_with_replicates` — fraction with `exp_per_cond ≥ 2`. High values",
        "  (DvH 0.99, Caulo 0.98, Pedo557 0.93) make the leakage risk above critical;",
        "  low values (Miya 0.22) make the constraint mostly a no-op.",
        "",
        "## Deferred",
        "",
        "- **H-R0-03** (low-IQR gene overlap across orgs): figure 07 was dropped "
        "because `gene_key` is org-prefixed in the canonical data, making the "
        "Jaccard zero off-diagonal by construction. Testing this hypothesis "
        "requires a cross-org homology mapping (analog of S3's 0.85 cosine "
        "cutoff). Not blocking R-LOCK decisions; rationale for `weighted_all` "
        "training rests on the model-stability argument (predict-flat genes "
        "anchor the loss).",
    ]

    lines += [
        "",
        "## Candidate eligibility protocols",
        "",
        "| metric | m_min | threshold | coverage | median eligible/org |",
        "|---|---|---|---|---|",
    ]
    for c in elig_cands[:10]:
        lines.append(f"| {c.get('metric','tail_g')} | {c['m_min']} | {c['threshold']:.2f} | "
                     f"{c['coverage']:.2f} | "
                     f"{c['median_eligible_genes_per_org']:.0f} |")

    lines += [
        "",
        "## Candidate split protocols (fraction-based holdout)",
        "",
        "| split_id | frac | coverage | median val genes/org | median val conditions/org |",
        "|---|---|---|---|---|",
    ]
    for s in split_cands:
        lines.append(f"| {s['split_id']} | {s['holdout_fraction']:.2f} | "
                     f"{s['coverage']:.2f} | "
                     f"{s['median_val_genes_per_org']:.0f} | "
                     f"{s.get('median_val_conditions_per_org', 0):.0f} |")

    lines += [
        "",
        "## Figures",
        "",
        "See `research_log/figures/r0_data/` for the produced plots:",
        "",
        "- 01 m_g per gene per org · 02 IQR_g per gene per org · 03 joint (m_g, IQR_g)",
        "- 04 replicate noise floor · 05 SNR · 06 eligibility frontier",
        "- (07 dropped — see Deferred section)",
        "- 08 experiment structure · 09 condition discriminability",
        "- 10 split feasibility · 11 expGroup coverage",
        "",
        "## Next",
        "",
        "- R-LOCK-1 selects an eligibility candidate (or proposes a new one).",
        "- R-LOCK-2 selects a split candidate (likely refined with stratification).",
        "- R-LOCK-3 implements the RankingBatch contract.",
        "- R-LOCK-4 finalizes metric + baseline + noise-floor reporting.",
    ]
    with REPORT_OUT.open("w") as f:
        f.write("\n".join(lines))
