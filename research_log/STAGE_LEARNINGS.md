# Stage learnings log

**Purpose.** This file records **project-specific conclusions** from each pipeline stage
(reproducibility, data-sheet, split lock, policy lock, tiers). It complements:

- `docs/REFACTORPLAN.md` — what we *plan* to do and how stages gate promotion.
- `research_log/decisions/<stage>/` — formal decision records (S0-DEC-001, …).

**How to use it.** After a stage closes (or when figure data is regenerated), append or
revise the corresponding section with:

1. **What we measured** (artifact paths, figure CSV names).
2. **What we learned** (bullet takeaways tied to this repo’s data).
3. **What we decided or deferred** (link to decision file if one exists).
4. **Open risks** (what could still invalidate the takeaway).

**Figure conventions.** Stage 1 figures live under `research_log/figures/stage1/`.
Numeric files `NN_*.csv` are plot-ready tables. Companion tables under
`research_log/figures/stage1/_data/` are sometimes the **authoritative** export when a
numbered CSV is malformed (noted explicitly below).

---

## Stage S0 — Reproducibility & governance

**Sources.**

- `research_log/decisions/stage0/S0-DEC-001.md`
- Smoke runs (example): `artifacts/runs/20260427_122651_S0_smoke_s0/run_manifest.json`

**Learnings.**

- The S0 smoke pipeline produces a `run_manifest.json` that **validates** against
  `data_contract/schemas/run_manifest_v1.schema.json`.
- Two reruns at the same seed and smoke row cap produced **bit-identical**
  `smoke_digest` values — reproducibility of the harness itself is established for the
  declared smoke configuration.
- Smoke scoring used **15,936** val rows (80/20 split of 80k sampled rows); this is
  sufficient to prove harness integrity, **not** to characterize biology.

**Implications.**

- Downstream stages may trust **manifest shape + digest reproducibility** as a platform
  gate; biological conclusions still require S1+ evidence.

**Open risks.**

- Full-scale runs can still fail for operational reasons (I/O, nondeterministic GPU
  ops). Any nondeterminism must be documented in manifests if observed.

---

## Stage S1 — Data-sheet & split-feasibility diagnostics

**Sources (primary).**

- `research_log/figures/stage1/01_*.csv` … `25_*.csv`
- Clean companion tables: `research_log/figures/stage1/_data/*.csv`

### Cross-cutting S1 conclusions (from the figure set as a whole)

1. **Media-name overlap is sparse across organism pairs**, but **canonical chemistry
   overlap is dense**. Split design must prioritize chemistry overlap diagnostics over
   naive media-string overlap.
2. **Row-count imbalance across organisms is extreme (~92× top vs bottom)** while
   **gene-count imbalance is modest (~5×)** — most row imbalance comes from experiment
   density, not “more genes.”
3. **Within-gene condition coverage is organism-dependent.** Some organisms have median
   conditions per gene in the hundreds; others in the single digits — Spearman and
   ranking metrics are **not comparable** across organisms without eligibility gating.
4. **Chemistry workbook mapping is strong on average (~97% mapped)** but **weak for a
   small set of organisms** (50% mapped in the worst case observed) — mapped/unmapped
   cohort reporting is mandatory.
5. **Embedding join coverage is high on average (~98.7% covered)** but **`azobra` is a
   clear outlier (~13.6% rows without embeddings)** — denominator parity and per-org
   join audits are mandatory.
6. **`fit` is heavy-tailed (strong left tail)** — RMSE-only optimization or reporting
   would overweight extremes; MAE + tail diagnostics + Huber policy tests are
   justified empirically.
7. **Declared split protocols remain largely chemistry-“seen”** (even `low_overlap_stress`
   is ~95% seen-chemistry in the exported table). Claims about “novel chemistry
   generalization” need a harder protocol than currently exported, or narrower wording.
8. **Homology / embedding-nearest-neighbor difficulty varies a lot by protocol**
   (median max-cosine ~0.51 for `largest_by_rows` vs ~0.92 for `multi_org_balanced`).
   Protocol comparison must stratify or disclose this axis.
9. **Representation-mode composition (`extract` / `mix` / `physical`) shifts by protocol
   and split side** — mode-shift confounding is a first-class fairness risk, not a
   hypothetical edge case.

---

### Figure `01_org_media_overlap_heatmap.csv`

| Field | Content |
| --- | --- |
| **What it is** | Pairwise matrix: count of **shared media name strings** between organisms. |
| **How to read** | Off-diagonal cell (A,B) = intersection size of media sets. |
| **Project-specific takeaway** | Off-diagonal overlaps are **mostly zero**: median **0**, **67.7%** of pairs share **0** media names, max shared count **21**. |
| **Implication** | Organism holdout without chemistry diagnostics can accidentally become a **media-ID cold start** problem. |
| **Drives** | Stage S3 protocol selection: require chemistry overlap reporting, not only organism IDs. |

---

### Figure `02_org_canonical_id_overlap_heatmap.csv`

| Field | Content |
| --- | --- |
| **What it is** | Pairwise matrix: count of **shared canonical chemical IDs** between organisms. |
| **Project-specific takeaway** | Chemistry overlap is **dense**: **0%** of pairs have zero overlap (min **4** shared IDs), median **69**, mean **~57.5**, max **82**. |
| **Implication** | “No shared media strings” does **not** imply “no shared chemistry.” |
| **Drives** | Tier 1 encoding: canonical chemistry features are the right primary object; media-name features are secondary / shortcut-prone. |

---

### Figure `03_org_pair_jaccard_distribution.csv`

| Field | Content |
| --- | --- |
| **What it is** | Distribution of pairwise Jaccard similarity between organism chemistry sets. |
| **Project-specific takeaway** | Wide but skewed high: min **0.095**, median **0.854**, p90 **0.974**, max **1.0** (n = **1128** pairs). |
| **Implication** | Many organism pairs are chemically similar, but a **non-trivial tail** of harder pairs exists. |
| **Drives** | Evaluation reporting: stratify metrics by overlap bucket, not only a single headline number. |

---

### Figure `04_bipartite_org_media_top.csv`

| Field | Content |
| --- | --- |
| **What it is** | Heaviest edges in the organism ↔ media bipartite graph (`weight` ≈ mass). |
| **Project-specific takeaway** | Dominant edges include `Btheta`–`Varel_Bryant_medium_Glucose` (**239**), `DvH`–`MoYLS4` (**224**), `DvH`–`MoLS4` (**168**), `Marino`–`marine_broth_2216` (**159**), `psRCH2`–`LB` (**138**). |
| **Implication** | A model can score well by fitting **hubs**; conditional-essentiality claims need **hub vs tail** performance slices. |
| **Drives** | Tier reports: always include mass-weighted vs equal-weight summaries where feasible. |

---

### Figure `05_rows_per_organism_bar.csv`

| Field | Content |
| --- | --- |
| **What it is** | Total fitness **rows per organism** (wide single-row table for bar plotting). |
| **Project-specific takeaway** | Imbalance is extreme: top `Btheta` **2,104,545** rows vs bottom `RalstoniaUW163` **22,770** → ratio **~92.4×**. |
| **Implication** | Global micro-averaged metrics are dominated by a few organisms unless counterweighted. |
| **Drives** | Stage S5 / reporting: per-organism panels + macro-averages remain mandatory. |

---

### Figure `06_conditions_per_gene_cdf.csv` + `07_conditions_per_gene_violin.csv`

| Field | Content |
| --- | --- |
| **What they are** | `06`: ECDF of distinct conditions per gene, by organism. `07`: underlying `n_conditions` per gene for violin plots. |
| **Project-specific takeaway (from `07`)** | Median conditions/gene spans **~6 (`RalstoniaUW163`) to ~757 (`DvH`)**. For the worst Ralstonia organisms, **every gene** has ≤10 conditions (fraction **1.0** at ≤10 for those orgs). |
| **Implication** | Spearman across conditions is **undefined or noise-dominated** for many genes in low-support organisms unless `m` and variability gates are strict. |
| **Drives** | Stage S2: finalize `spearman_eligibility_policy_id` (`m`, `v_min`, exclusion counts) before tier promotions. |

---

### Figure `08_org_media_row_count_heatmap.csv`

| Field | Content |
| --- | --- |
| **What it is** | Organism × media matrix of **row counts** (top media columns). |
| **Project-specific takeaway** | Matrix is **60** media columns × **48** organisms, but each organism uses only a **tiny** number of those columns nonzeros: median **4** nonzero media/org (min **1**, max **8**). |
| **Implication** | Even within “top media,” usage is **highly concentrated** — media-ID shortcuts are structurally easy. |
| **Drives** | Tier 1: prioritize chemistry / decomposition-aware features over raw media-id one-hot as the scientific default. |

---

### Figure `09_genes_per_organism_bar.csv`

| Field | Content |
| --- | --- |
| **What it is** | Distinct **genes measured** per organism. |
| **Project-specific takeaway** | Top `Cup4G11` **6384** genes vs bottom `Methanococcus_S2` **1244** → ratio **~5.1×** (far less than row imbalance). |
| **Implication** | Row imbalance is driven primarily by **experiment/replicate density**, not catalog size alone. |
| **Drives** | Diagnostics: when an organism looks “bad,” check **conditions/gene** and **rows/gene**, not only gene count. |

---

### Figure `10_fit_distribution_per_org_violin.csv`

| Field | Content |
| --- | --- |
| **What it is** | All `(orgId, fit)` pairs for distribution/violin plots (very large). |
| **Project-specific takeaway (global quantiles)** | Median `fit` **~−0.014**; p1 **~−3.044**; p5 **~−0.857**; p95 **~0.464**; p99 **~1.012** — sharp central mass + heavy left tail. |
| **Implication** | RMSE is tail-sensitive; MAE and Huber are not luxuries for this dataset. |
| **Drives** | Locked plan items: RMSE+MAE co-primary; Huber vs MSE ablation in policy stage; tail slices in reports. |

---

### Figure `11_t_stat_distribution_per_org.csv`

| Field | Content |
| --- | --- |
| **What it is** | Per-row `|t|` distribution (gene-level confidence in `fit`). |
| **Project-specific takeaway (global quantiles)** | p5 **~0.060**, p25 **~0.319**, median **~0.700**, p75 **~1.282**, p95 **~3.244**. |
| **Implication** | A large mass of rows are **low-confidence**; equal weighting will wash noise into gradients unless policy addresses it. |
| **Drives** | Stage S5: weighted-full vs strict-slice is an empirical necessity, not a style choice. |

---

### Figure `12_cor12_distribution_per_experiment.csv`

| Field | Content |
| --- | --- |
| **What it is** | ECDF of experiment-level `cor12` (replicate agreement), grouped by organism. |
| **Project-specific takeaway (pooled quantiles)** | Median **~0.232**; p25 **~0.171**; p5 **~0.119** — many experiments are only modestly reproducible at replicate level. |
| **Implication** | Experiment-quality weighting/filtering should be **ECDF-driven**, not arbitrary constants. |
| **Drives** | Stage S5: document weight functional form + floors/caps from these distributions. |

---

### Figure `13_fit_qq_plot_global.csv`

| Field | Content |
| --- | --- |
| **What it is** | QQ-plot sample: empirical `fit` quantiles vs normal reference. |
| **Project-specific takeaway** | Tail mismatch is severe (bottom tail mean sample quantile **~−4.38** vs theoretical **~−2.67**; top tail lighter than normal). |
| **Implication** | Gaussian noise assumptions are **not** safe for inference or naive interval interpretation. |
| **Drives** | Metric policy: require tail diagnostics; treat “Gaussian CI stories” as invalid without calibration evidence. |

---

### Figure `14_chemistry_mapped_unmapped_by_org.csv`

| Field | Content |
| --- | --- |
| **What it is** | Per organism: fraction of experiments with mapped vs unmapped chemistry. |
| **Project-specific takeaway** | Mean mapped **~0.9735**. Worst: `Ddia6719` **50%** unmapped; `DdiaME23` **37.5%**; `Dda3937` **28.6%**; `psRCH2` **11.1%**. |
| **Implication** | Chemistry-first modeling is **conditionally valid** — validity depends on organism cohort. |
| **Drives** | Stage S3: mapped/unmapped cohort metrics in every protocol report; careful val/test organism selection. |

---

### Figure `15_embedding_coverage_by_org.csv`

| Field | Content |
| --- | --- |
| **What it is** | Per organism: fraction of rows with an embedding join hit. |
| **Project-specific takeaway** | Mean covered **~0.9869**. Worst: `azobra` **86.45%** covered (**13.55%** dropped); `Keio` **94.62%**. |
| **Implication** | Embedding join is a **first-class data processing step** with organism-specific loss. |
| **Drives** | Denominator parity: model and nulls must use identical post-join row ids; log per-org uncovered rates in manifests. |

---

### Figure `16_canonical_id_prevalence_distribution.csv`

| Field | Content |
| --- | --- |
| **What it is** | Histogram: for each integer value 1..48, how many Canonical_IDs appear in exactly that many organisms. |
| **How to read** | Each bin is one possible "ubiquity" value. Tall bars at the extremes = bimodal. |
| **Project-specific takeaway** | Bimodal U-shape: **12** chemicals used by 1 organism (rare tail), **20** chemicals used by 46 organisms (the "core"), middle values are sparse. Median = 38 organisms; mean ≈ 29.6. |
| **Implication** | Canonical_IDs are either "core utilities" (in almost every medium) or "organism-specific oddities" (in 1–3 media). Very few are "moderately shared." This shape directly informs S4's prevalence-based trimming policy. |
| **Drives** | Tier 1 / S4: train-only prevalence trimming with threshold sitting in the middle (where there's almost no data anyway); explicit tail-vs-core reporting in T1 ablations. |

**Note (resolved 2026-04-27).** Initial export used a single-group violin which is degenerate for this 1-D integer distribution; replaced with `save_histogram` helper. Authoritative companion at `_data/chemical_ubiquity.csv`.

---

### Figure `17_chemistry_seen_unseen_rate_per_protocol.csv`

| Field | Content |
| --- | --- |
| **What it is** | For each split protocol: fraction of val/test **supervision mass** on chemistry seen vs unseen in train. |
| **Project-specific takeaway** | `high_overlap_easy` and `multi_org_balanced`: **100%** seen on val and test. `largest_by_rows`: val **98.61%** seen, test **97.62%** seen. `low_overlap_stress`: val **94.74%**, test **95.12%** seen. |
| **Implication** | Current protocols are still **predominantly in-distribution chemistry**; “hard chemistry OOD” is not yet exercised by these numbers alone. |
| **Drives** | Stage S3: if OOD chemistry claims are needed, add a **new** protocol with materially higher unseen rates, or narrow scientific claims. |

---

### Figure `18_embedding_cosine_to_nearest_train_per_protocol.csv`

| Field | Content |
| --- | --- |
| **What it is** | Val/test genes: max cosine similarity to nearest train gene embedding, by protocol. |
| **Project-specific takeaway (medians)** | `largest_by_rows` **0.506**; `high_overlap_easy` **0.799**; `multi_org_balanced` **0.921**. |
| **Implication** | Protocols differ strongly in **homology / embedding-neighborhood ease**, independent of model quality. |
| **Drives** | Homology diagnostics (`H-HOMO-01`): mandatory stratified reporting; optional homology-masked protocol if effect size triggers plan. |

---

### Figure `19_homology_similarity_by_org.csv`

| Field | Content |
| --- | --- |
| **What it is** | Same cosine-nearest-train signal with per-`orgId` breakdown (many rows). |
| **Project-specific takeaway (distribution of `max_cosine`, n = 59632)** | Median **~0.779**; p75 **~0.918**; p90 **~0.962** — overall high neighborhood similarity mass. |
| **Implication** | Organism-holdout alone does **not** remove sequence-similarity transfer; headline metrics can be optimistic. |
| **Drives** | Stage S3 organism pool design: include similarity regimes in val/test selection or reporting. |

---

### Figure `20_representation_mode_proportions_per_org.csv`

| Field | Content |
| --- | --- |
| **What it is** | Per organism: fraction of chemistry decomposition mass in `extract` / `mix` / `physical` modes. |
| **Project-specific takeaway** | Highly variable: e.g. `Btheta` is **76%** `physical` / **24%** `extract` / **0%** `mix`; `pseudo3_N2E3` is **39%** / **49%** / **13%**. |
| **Implication** | Mode composition is a real covariate; models can exploit **mode mixtures** if not controlled in encoding + reporting. |
| **Drives** | Tier 1 (`H-ENC-05`): explicit mode indicators / ablations; never silently merge incompatible decompositions. |

---

### Figure `21_representation_mode_per_protocol.csv`

| Field | Content |
| --- | --- |
| **What it is** | Representation-mode fractions **conditional on split side** (`train|val|test`) for each protocol. |
| **Project-specific takeaway** | Large shifts across protocol and split side exist (e.g. `multi_org_balanced|test` **~81%** `extract`; `largest_by_rows|val` **~76%** `physical` with **0%** `mix`; `low_overlap_stress|val` shows **0%** `extract` in that slice). |
| **Implication** | Protocol comparisons can be **confounded by representation-mode shifts** — not just chemistry overlap. |
| **Drives** | Promotion guardrail: reject wins explained primarily by mode imbalance between train and val/test slices. |

---

### Figure `22_chemical_ubiquity_histogram.csv`

| Field | Content |
| --- | --- |
| **What it is** | Same data as fig 16 — chemical ubiquity histogram — but kept separately because the cross-organism coverage block (figs 22–24) is a self-contained narrative deliverable. |
| **Note (resolved 2026-04-27)** | Same fix as fig 16; rebuilt with `save_histogram`. |
| **Drives** | Cross-organism coverage block of the tier report; informs S4 trimming threshold (knee from fig 24). |

---

### Figure `23_organism_topN_chemical_heatmap.csv`

| Field | Content |
| --- | --- |
| **What it is** | Heatmap: organism × top-N canonical chemicals (counts / intensities). |
| **Project-specific takeaway** | Shape **48 × 100**; row-sum over displayed chemicals: median **~7286.5**, min **64**, max **36534**. |
| **Implication** | Organisms have very different **top-chemical signatures** — another organism-shortcut surface. |
| **Drives** | Split diagnostics + baseline hierarchy: include chemistry-stratified slices and organism-balanced reporting where possible. |

---

### Figure `24_chemical_coverage_curve.csv`

| Field | Content |
| --- | --- |
| **What it is** | Rank curve of chemicals by `n_organisms` / `n_experiments`. |
| **Project-specific takeaway** | Ultra-head-heavy: top experiment counts at **7352 / 7348 / …** levels vs tail chemicals at only **2–4** experiments (from tail inspection). |
| **Implication** | A raw full chemistry vector is dominated by **a few universal ions/metabolites** plus a long sparse tail. |
| **Drives** | Tier 1: define **core vs tail** chemistry reporting; train-only trimming sensitivity must be pre-registered. |

---

### Figure `25_media_chemistry_pca.csv`

| Field | Content |
| --- | --- |
| **What it is** | 2D PCA embedding of media chemistry vectors (`x`, `y`) with `label` = medium. |
| **Project-specific takeaway** | **112** media points spanning a wide PCA bounding box (x roughly **−2.65 … 4.99**, y **−3.04 … 1.96**). |
| **Implication** | Media chemistry space is genuinely high-diversity; split protocols can land on **isolated** neighborhoods — worth visual check when locking S3. |
| **Drives** | Stage S3: explicitly check whether val/test media fall in train-covered neighborhoods in PCA (or similar) space for the primary protocol. |

---

### Discussion appendix (post-S1 review, 2026-04-27)

Issues raised during stakeholder review of the S1 figures, with project-level
implications captured here so they don't get lost.

**A. Scientific scope: "predicts in known media" vs "generalizes to novel chemistry."**
All four candidate protocols are ≥95% chemistry-seen at the Canonical_ID level
(fig 17). With v4 as it stands (~112 Canonical_IDs, mostly widely shared) we
can't honestly claim "generalizes to any chemistry." We CAN claim
"predicts conditional gene essentiality given growth in a known medium drawn
from the v4 vocabulary" — a real, narrower, publishable claim. Going beyond
this requires either (a) holdouts of specific Canonical_IDs (not just organisms),
(b) chemical-fingerprint encoders (Morgan/RDKit) that can represent unseen
chemicals structurally, or (c) external test data with novel media.
**Pre-S2 decision: keep current scope; document the narrower claim explicitly
in tier reports; note fingerprint encoding as a deliberate deferral.**

**B. Decomposition-source flags (`H-ENC-05`) — scrutiny.**
The flags exist to distinguish "L-Alanine added directly at known concentration"
from "L-Alanine inferred to be present because the medium contains yeast extract."
The two are chemically identical molecules but the *quantity* differs (a defined
1 g/L dose vs a batch-to-batch yeast-extract estimate). Whether models actually
need to distinguish them is empirically open: if extract-derived and direct-derived
amino acids predict fitness similarly, decomp flags are noise; if they predict
differently, the flag is load-bearing. **T1-E will test this empirically; if
the flag adds nothing, we drop it and the feature contract simplifies.**

**C. "Core-chemistry subset" experiment — proposed for T1.**
Fig 24's coverage curve has a steep head (a few chemicals appearing in ~46/48
organisms — these are likely water/glucose/NaCl/MgSO4-class core utilities)
and a long tail. A natural ablation: drop the top-K most-ubiquitous Canonical_IDs
and see whether the model's val performance changes. If it doesn't, those
chemicals are constants carrying no signal and S4's trimming should remove
them. If performance drops, they're informative even when ubiquitous.
**Add as `T1-F_core_chemistry_ablation` to the T1 experiment list. Hypothesis
candidate: `H-ENC-06` — the most-ubiquitous chemicals carry no information
beyond the global mean and can be dropped without harm.**

**D. Embedding coverage and the "unusable rows" question.**
Per fig 15: ~98.7% of fitness rows have embeddings. The dropped ~1.3% have no
ProteomeLM vector for the gene → genuinely unusable for an embedding-based model.

**Investigation (2026-04-27): aaseqs and embeddings are 1:1.** Both contain
exactly 221,030 gene_keys; every aaseqs entry has an embedding and vice versa.
The 2,484 fitness genes missing embeddings are **also absent from aaseqs**
— they have no protein sequence in our database at all. Likely causes: Tn-seq
insertions in non-protein-coding regions (intergenic, RNAs, pseudogenes),
genome-version drift between the fitness data and the aaseqs dump, or
upstream pipeline gaps. There is no recovery path from current data;
re-running ProteomeLM on aaseqs would produce the same 221,030 embeddings
we already have.

**Pre-S2 decision: inner-join and drop. Log per-organism uncovered fraction
in run manifest; flag azobra (~13.6% loss) as the worst case.** Recovery
would require either obtaining a more complete aaseqs dump (out of scope)
or accepting fallback embeddings (deferred).

**E. Chemistry coverage and the "unmapped media" question.**
Per fig 14: significant per-organism variation in fraction-of-media-mapped
(0–80%). For unmapped media, we have a row with valid `fit` but no chemistry
vector. **Per H-DATA-01 the policy is: keep the row, encode chemistry as
explicit `<UNK>` (multihot vector of all zeros + a "missing chemistry" mask
flag). The non-chemistry features (organism, condition_1..4, temperature, pH,
aerobic) still inform the model. Log unknown_category_rate per run.**
The model still learns from the row but is signaled that chemistry information
is unavailable.

**F. The 4 candidate protocols — why these specifically.**

| Protocol | Strategy | Why this one |
|---|---|---|
| `largest_by_rows` | val=Btheta, test=DvH | Statistical-power maximizer. The two largest organisms by row count yield val/test sets of ~2M each. Worst case if model fails here, fails everywhere. |
| `high_overlap_easy` | val=[Putida, psRCH2], test=pseudo3_N2E3 | Sanity-check protocol: chemistry overlap >85%. If our model can't beat additive on the easiest possible org-holdout, no point doing harder ones. |
| `low_overlap_stress` | val=SynE, test=Magneto | The lowest-overlap eligible val org. SynE is a cyanobacterium — different lifestyle from most. Diagnostic only; reported but not promotion-gating. |
| `multi_org_balanced` | val=[pseudo3_N2E3, BFirm, ANA3, SyringaeB728a_mexBdelta], test=PV4 | Quartile-spaced over the overlap distribution. Balanced view; closest to "publishable benchmark" framing. |

**Why not other configurations?** The candidate generator implements 5 strategies
(largest, mid-overlap, high-overlap, low-overlap, multi-org-balanced) and only
emits the ones that match. The `mid_overlap_stratified` (50–70% chemistry overlap)
strategy emitted nothing because v4 organisms are bimodal in overlap — none fall
in that band. Strategies we intentionally did not include: random val org,
LOOO (deferred to optional robustness protocol), Canonical_ID-level holdouts
(deferred per scope discussion A above). If S3 wants to add a synthetic
"hard chemistry holdout" protocol, that goes in S3's selection process
not back in S1.

**G. Representation modes — why we keep tracking them despite the user
question.** The honest answer: we track them as a cheap insurance policy.
The mode tags cost almost nothing to compute and emit, and the H-ENC-05 test
is one T1 ablation that either confirms they matter or kills the feature
permanently. Continuing to track them through S4 is *not* a commitment to use
them in the model; it's a commitment to give T1-E something to test.

---

## Stage S2 — (pending)

*No learnings recorded here yet.* When Stage S2 closes, paste:

- locked primary protocol id,
- Spearman eligibility policy,
- homology diagnostic disposition,
- additive baseline (`H-BASE-01`) numbers on the locked row set.

---

## Stage S3 — (pending)

*No learnings recorded here yet.*

---

## Stage S5 — Data-quality policy lock (pending)

*No learnings recorded here yet.*

---

## Tiers T1–T4 — (pending)

*No learnings recorded here yet.*

---

## Maintenance checklist

- [ ] After **S1** figure regeneration, fix `16_*.csv` and `22_*.csv` or remove them from
      publication paths in favor of `_data/chemical_ubiquity.csv`.
- [ ] After **S3** split lock, add a short subsection summarizing **chosen protocol** vs
      **diagnostic protocols** with key rates from `17_*.csv` and `18_*.csv`.
- [ ] After each **tier promotion**, add 3–6 bullets: what improved, what did not, and
      which figure diagnostics would falsify the promoted default.
