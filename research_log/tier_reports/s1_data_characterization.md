# Stage 1 Report — Data Characterization

**Status:** approved (2026-04-27). See [`decisions/stage1/S1-DEC-001.md`](../decisions/stage1/S1-DEC-001.md).

---

## TL;DR

- **Scale.** 27,410,721 fitness rows × 48 organisms × 7,552 experiments × 112 distinct media in v4. Median organism has ~570k rows; the largest (Btheta) has 2.1M, the smallest <50k.
- **Chemistry overlap is high at the Canonical_ID level (~95–100%) but sparse at the media-name level (~0–20%).** Different organisms use different media *recipes* drawn from a common chemistry vocabulary. This means an organism-holdout split must use *Canonical_ID*-level overlap as the primary metric — media-name overlap will mislead.
- **`H-HOMO-01` triggered.** Maximum per-organism deviation in nearest-train-gene cosine similarity is **1.6σ**, far above the 0.5σ threshold. **S3 must add a homology-aware diagnostic protocol.**
- **Representation-mode mapping ratified** (4 modes: physical, mix, extract, in_silico). `direct + salt → physical`, `mix → mix`, `extract → extract`, `unrecoverable → in_silico`. v4 currently has zero `unrecoverable` rows so the in_silico mode is reserved but unused.
- **4 candidate split protocols** emitted (one fewer than the planned 5; the `mid_overlap_stratified` band [50–70% chemistry overlap] was empty because v4 organisms are bimodal: chemistry overlap is either ~95–100% or near zero at media level).
- **Hard-gate decisions taken:** mapped/unmapped policy = explicit `<UNK>`; min support threshold = 50,000 rows; H-HOMO-01 trigger = required diagnostic at S3; representation_mode_mapping.yaml ratified.

---

## 1. Overlap analyses (figures 01–04)

The pairwise organism × organism overlap structure has two surprising properties:

- **Media-name overlap is very low.** Most organism pairs share zero or one media name. See [`01_org_media_overlap_heatmap.png`](../figures/stage1/01_org_media_overlap_heatmap.png).
- **Canonical_ID overlap is very high.** Most organism pairs share 90+ chemistry components. See [`02_org_canonical_id_overlap_heatmap.png`](../figures/stage1/02_org_canonical_id_overlap_heatmap.png).

The Jaccard distribution ([`03_org_pair_jaccard_distribution.png`](../figures/stage1/03_org_pair_jaccard_distribution.png)) confirms this: at the Canonical_ID level most pairs are tightly clustered around high Jaccard (~0.7–1.0); at the media-name level most pairs sit at near-zero Jaccard.

**Interpretation:** organism-holdout splits will see almost-complete chemistry coverage *as long as we encode at the Canonical_ID level*. A media-name encoder would generalize poorly; this is a strong (and concrete) argument for `H-ENC-01` (multihot beats media-id).

The bipartite top-degree graph ([`04_bipartite_org_media_top.png`](../figures/stage1/04_bipartite_org_media_top.png)) shows that a small set of media (LB, M9 variants, RCH2, mineral-defined media) account for most cross-organism sharing.

## 2. Support and sparsity (figures 05–09)

- **Rows per organism** ([`05_rows_per_organism_bar.png`](../figures/stage1/05_rows_per_organism_bar.png)): heavy-tailed; Btheta and DvH each have ~2M rows, the smallest 5 organisms each have <100k. Setting `min_org_support_rows = 50,000` includes ~40 of 48 organisms.
- **Conditions per gene** ([`06_conditions_per_gene_cdf.png`](../figures/stage1/06_conditions_per_gene_cdf.png), [`07_conditions_per_gene_violin.png`](../figures/stage1/07_conditions_per_gene_violin.png)): median gene is observed in 100+ conditions in well-studied organisms; the long tail goes >500. **Spearman eligibility `m=5` is comfortably satisfied for almost every gene** — `H-EVAL-01` will be powered.
- **Org × media row counts** ([`08_org_media_row_count_heatmap.png`](../figures/stage1/08_org_media_row_count_heatmap.png)) is highly sparse: most (org, media) cells are empty.
- **Genes per organism** ([`09_genes_per_organism_bar.png`](../figures/stage1/09_genes_per_organism_bar.png)): ranges ~1.5k–9k unique gene_keys per organism.

## 3. Quality and noise (figures 10–13)

- **`fit` distributions per organism** ([`10_fit_distribution_per_org_violin.png`](../figures/stage1/10_fit_distribution_per_org_violin.png)) are heavy-tailed and centered near zero, with substantial per-organism variation in spread. This **confirms the heteroscedastic-noise policy from L4** — RMSE+MAE co-primary is the right call.
- **`|t|` distributions** ([`11_t_stat_distribution_per_org.png`](../figures/stage1/11_t_stat_distribution_per_org.png)) show the classic Tn-seq long-right-tail; some experiments have very few high-|t| rows (mostly genes near neutral).
- **`cor12` distributions** ([`12_cor12_distribution_per_experiment.png`](../figures/stage1/12_cor12_distribution_per_experiment.png)) show heterogeneous experiment quality across organisms — useful input to S5's row-quality policy.
- **`fit` QQ plot vs Normal** ([`13_fit_qq_plot_global.png`](../figures/stage1/13_fit_qq_plot_global.png)) shows clear S-shaped curvature at the tails. **Huber loss is well-motivated for T4** (`H-LOSS-01`); MSE will be unfairly punished by the tails.

## 4. Modality coverage (figures 14–16)

- **Chemistry coverage of media** ([`14_chemistry_mapped_unmapped_by_org.png`](../figures/stage1/14_chemistry_mapped_unmapped_by_org.png)): per organism, the fraction of media that have v4 component data. Most organisms have 30–80% mapping; some have <10%. Unmapped media must be handled with explicit `<UNK>` per `H-DATA-01`.
- **Embedding coverage** ([`15_embedding_coverage_by_org.png`](../figures/stage1/15_embedding_coverage_by_org.png)): each organism's fitness rows are very largely (>90%) covered by ProteomeLM embeddings. The remaining drop is acceptable inner-join loss.
- **Canonical_ID prevalence** ([`16_canonical_id_prevalence_distribution.png`](../figures/stage1/16_canonical_id_prevalence_distribution.png)): U-shaped — many chemicals appear in >40 organisms (the ubiquitous "core") AND many appear in only 1–3 organisms (the long tail). The middle is sparse. This shape **directly informs S4's prevalence-based feature trimming**.

## 5. OOD and homology diagnostics (figures 17–19)

- **Per-protocol chemistry seen/unseen** ([`17_chemistry_seen_unseen_rate_per_protocol.png`](../figures/stage1/17_chemistry_seen_unseen_rate_per_protocol.png)): all 4 candidate protocols have val Canonical_ID seen-rate ≥95% — the chemistry overlap will not be the limiting factor for any of these splits. This is good (we won't accidentally pick an "impossible" protocol).
- **Embedding cosine to nearest train gene per protocol** ([`18_embedding_cosine_to_nearest_train_per_protocol.png`](../figures/stage1/18_embedding_cosine_to_nearest_train_per_protocol.png)): protocols differ substantially. `largest_by_rows` (Btheta) sees lower max cosines than `multi_org_balanced`, indicating the easier protocols also enjoy higher train similarity for val genes.
- **Per-organism homology** ([`19_homology_similarity_by_org.png`](../figures/stage1/19_homology_similarity_by_org.png)): organisms have substantially different similarity profiles relative to all-other-orgs. **`H-HOMO-01` triggered: max per-org median deviation = 1.606σ** (threshold was 0.5σ). This is a strong signal that homology-aware evaluation must be added at S3.

## 6. Representation-mode audit (figures 20–21)

- **Per-organism mode proportions** ([`20_representation_mode_proportions_per_org.png`](../figures/stage1/20_representation_mode_proportions_per_org.png)): organisms vary from ~80% physical (defined-media-heavy) to ~70% extract (LB-rich). Three modes appear in practice: `physical`, `mix`, `extract`. The fourth mode `in_silico` does not appear because v4 has zero `unrecoverable` rows (this is intentional — v4 explicitly resolved the prior LB Miller in-silico issue).
- **Per-protocol-partition mode proportions** ([`21_representation_mode_per_protocol.png`](../figures/stage1/21_representation_mode_per_protocol.png)): for `largest_by_rows`, the val partition (Btheta) has 76% physical / 24% extract / 0% mix — substantially different from train (40% extract / 37% mix / 23% physical). This is **the H-ENC-05 risk made concrete**: organisms have systematically different mode mixes, so a model that doesn't have explicit mode flags could spuriously learn organism shortcuts disguised as chemistry.

## 7. Cross-organism chemistry coverage (figures 22–24)

- **Chemical ubiquity histogram** ([`22_chemical_ubiquity_histogram.png`](../figures/stage1/22_chemical_ubiquity_histogram.png)): bimodal — most Canonical_IDs are either ubiquitous (used by 30+ organisms) or organism-specific (1–3 orgs). The "middle" range is sparse.
- **Organism × top-100 chemicals heatmap** ([`23_organism_topN_chemical_heatmap.png`](../figures/stage1/23_organism_topN_chemical_heatmap.png)): the top-100 chemicals cover essentially every organism's needs. This visually confirms that **at the Canonical_ID level there is no chemistry-overlap problem** for any organism-holdout protocol.
- **Coverage curve** ([`24_chemical_coverage_curve.png`](../figures/stage1/24_chemical_coverage_curve.png)): the curve has a clear "knee" around chemical ~50–60. Above the knee, chemicals are used by 30+ organisms; below the knee, the long tail. **S4's prevalence threshold should sit just below the knee.**

## 8. Optional / exploratory (figure 25)

- **Media chemistry PCA** ([`25_media_chemistry_pca.png`](../figures/stage1/25_media_chemistry_pca.png)): media cluster by chemistry as expected. Defined-mineral media occupy one cluster; LB + nutrient-rich media occupy another. Coloring by #organisms shows that *some* clusters have media used by many organisms while other clusters are organism-specific. Exploratory only — does not gate any decision.

---

## Hard-gate decisions

| Decision | Outcome | Reference |
|---|---|---|
| Mapped/unmapped chemistry policy (`H-DATA-01`) | **Explicit `<UNK>` token** at index 0 in feature contract; unknown_category_rate logged per run. Reasoning: significant fraction of organisms have unmapped media; silent drop would skew evaluation. | S4 will use this; `apply_unk_policy` already implements it. |
| Min support threshold for val/test orgs (`H-DATA-02`) | **`min_val_rows = 50,000`** (current default = 10,000 in run; bumped here based on rows-per-org distribution). Excludes 8 of 48 organisms. | candidate_protocols.yaml |
| `H-HOMO-01` trigger | **TRIGGERED** at 1.606σ. S3 must add a homology-aware diagnostic protocol. | candidate_protocols.yaml `h_homo_01_triggered: true` |
| Representation mode mapping | **Ratified** as proposed. See S1-DEC-001. | data_contract/representation_mode_mapping.yaml |

## Outputs

- [`data_contract/splits/candidate_protocols.yaml`](../../data_contract/splits/candidate_protocols.yaml) — 4 candidates
- [`research_log/figures/stage1/`](../figures/stage1/) — 24 required + 1 optional figures (PNG + sibling CSV)
- [`research_log/decisions/stage1/S1-DEC-001.md`](../decisions/stage1/S1-DEC-001.md) — representation_mode mapping ratification
- This report

## Open items carried into S2

- Vectorize `fit_additive_baseline` (TODO from S0); now blocking S2 since S2 fits additive on full-scale data.
- The 4 candidate protocols all have val_canonical_id_seen_rate ≥0.95. If S3 wants a "low-overlap stress test" with substantial unseen chemistry, the current candidates won't provide one. May need to construct a synthetic protocol that holds out specific Canonical_IDs in addition to organisms.
- The `mid_overlap_stratified` candidate slot is empty. Plan called for 3–5 candidates; we have 4. Acceptable but worth noting that v4 organisms are bimodal in chemistry overlap (no 50–70% band).

## Open items carried into S4

- `representation_mode_mapping.yaml` is now `status: ratified`. S4 consumes it to build the mode tag per medium.
- Use the chemical coverage curve knee (~rank 50–60) as S4's prevalence-trimming threshold.
- Make sure the explicit `<UNK>` policy applies to *any* val/test medium with no v4 component data; this is the operationalization of `H-DATA-01`.
