# memory.md — permanent ledger

**NEVER read this file whole. Grep it.** It is append-only and unbounded.
`docs/plan.md` exists so a new session does not have to come here.

Newest first. Entries are compressed; the full record for each is in
`archive_docs/` at the path named in its `Source:` line.

**Correction rule:** historical lines are never deleted or edited. A refuted line
is prefixed `[INCORRECT] - ` verbatim, with `[CORRECTION - YYYY-MM-DD]:` directly
below it. Grep `[INCORRECT]` for everything this project has been wrong about.

---

## 2026-08-25 — P-FIX-docs-system: adopted the six-file documentation system

**Goal.** Replace the scattered doc set (README + `docs/project_memory/**` +
`research_log/**`) with six files split by access pattern, plus a verifier.

**Method.** Audited every doc-like file in the tree; mapped each to one of the six;
moved originals unchanged to `archive_docs/`; moved manuscript material to `paper/`;
renamed loose artifacts `DEPRECATED_*`; wrote `tests/test_docs_contract.py`.

**Result.** The audit surfaced five live contradictions, three of them inside
`data_contract/ranking/metric_contract.yaml` — the human-facing spec that
`src/ranking/eval/` implements (no code loads it). All are corrected below and the
contract is amended to v3. Two further corrections (5 and 6) record errors made
*during* this session: a false staleness claim about the dashboard, and a class of
failure the number-probe cannot detect at all. Two stale git worktrees were removed
after rescuing the sole unique file they held, now at
`archive_docs/v4_workbook_audit.md`, which existed in no commit.

**Source:** `archive_docs/` (all originals), `archive_docs/CLAUDE.md.orig`,
`archive_docs/README.md.orig`.

### Correction 5 — the dashboard was never stale (a claim made during this session)
[INCORRECT] - the dashboard is known-stale, tracked, and excluded until it becomes a generated view
[CORRECTION - 2026-08-25]: False. `dashboard/build.py` and `dashboard/content/experiments.json`
state the cold-gene CI caveat correctly and explicitly ("beats the null — but NDCG CIs
overlap"; "on the PRIMARY metric ... the 95% CIs OVERLAP ... An earlier claim of
CI-confirmation rested on Spearman alone and was corrected"). The dashboard was added to a
`KNOWN_STALE` allowlist in `tests/test_docs_contract.py` on the strength of 15 probe hits
that were all false positives. **Root cause of the false positives:** the correction probe
extracted numbers from `[INCORRECT]` lines, but a retracted line routinely contains numbers
that are still perfectly valid — what was retracted was the *interpretation*. The probe now
excludes any number restated anywhere else in this ledger as live fact. The allowlist has
been deleted entirely; nothing is exempt from the correction audit. The dashboard's real
weakness is structural (unverified hand-maintained duplication), not factual.

### Correction 6 — a gap the number-probe cannot see, by construction
A retracted *value* can be hunted by grep. A value that is **real but means the wrong thing
in the wrong context** cannot: it survives every "is this still true?" check, because it is
still true — just not of what it is being quoted against. `0.3214` (the median, unfiltered
variant) is the type case: a genuine, currently-computed quantity that is wrong only when
placed beside a model score and called "the ceiling" without that qualifier. The correction probe is blind to it. `tests/test_docs_contract.py::
test_contested_values_are_never_quoted_bare` covers this class via a `CONTESTED_VALUES`
registry: value -> (context it must not appear in, annotation that makes it safe, why).
Currently registered: `0.3214` (median/unfiltered) and `0.358` (8-high-rep-org subset).

### Correction 1 — the replicate ceiling
[INCORRECT] - noise_floor.primary.measured_value_full_val_23_orgs: 0.3214   # USE THIS — all 23 replicate orgs, real cold-condition val, n=74,804 genes
[CORRECTION - 2026-08-25]: "USE THIS" is wrong. 0.3214 is MEDIAN-aggregated over
ALL val genes; the model and every baseline are MEAN-aggregated over
PARITY-ELIGIBLE val genes. The comparable ceiling is 0.393 (n=45,943), the
`per_gene_mean` field of `task_relevant_noise_floor` with `eligible_genes` supplied.
0.3214 differs on two axes at once and is valid against nothing in the standard
report. Confirmed by the promotion arithmetic: `0.15 x (0.658 - 0.481) = 0.0266`
reproduces the documented ndcg_at_5 gate, whereas `0.15 x (0.3214 - 0.1694) = 0.0228`
reproduces the superseded Spearman gate. 0.3214 is retained as
`replicate_ceiling_spearman_unfiltered_median`, status DIAGNOSTIC. Caveat that must
travel with the 0.393 figure: eligibility selects high-spread genes, which replicate
better, so it is a ceiling for *eligible* genes, not a statement about the assay.
See `docs/terms.md`.

### Correction 2 — the promotion delta
[INCORRECT] - delta_concrete_spearman: 0.023          # promote if Δ-Spearman ≥ 0.023 vs chem-kNN
[CORRECTION - 2026-08-25]: Superseded, and never a real disagreement with the
documented ΔNDCG@5 ≳ 0.026 — the same formula evaluated against a different ceiling
(see Correction 1). The gate is on `ndcg_at_5` (PRIMARY since 2026-06-24), not
Spearman. `metric_contract.yaml` was never updated when the primary metric changed.

### Correction 3 — what the bottleneck is
[INCORRECT] - => the bottleneck is NOT model capacity/expressiveness; it is the OBJECTIVE (pointwise MSE for a ranking task) and/or the local-vs-global nature of the signal ... Strongly supports R-LOSS-before-R2.
[CORRECTION - 2026-08-25]: The objective half was refuted by R-LOSS (2026-06-06) and
R-TOPK (2026-06-17): pointwise Huber remained best and every ranking loss, including
top-5-truncated NDCG-matched ones, failed to beat the gate. Only the local-vs-global
half survived, and 2026-07-06 reframed even that as a property of the benchmark
rather than of the model. This refuted claim still sits in
`data_contract/ranking/metric_contract.yaml` — the correction never propagated to the
file the pipeline reads. Fix tracked as `P-FIX-metric-contract`.

### Correction 4 — chem_knn point values
[INCORRECT] - chem_knn:    [0.222, 0.377, 0.431, 0.481, 0.142, 0.446]   # STRONG bar / GATE (n=45,769)
[CORRECTION - 2026-08-25]: Not a definitional conflict but a stale snapshot. This row
is the 2026-05-25 measurement. The current locked values are Spearman 0.2402 /
ndcg_at_5 0.4852 (n=39,356 parity common set) in
`data_contract/ranking/reval_baseline.json`. The `reference_baselines_all_org` table
is a historical snapshot and must be read as one, not as current.

---

## 2026-08-25 — FRAMING LOCKED: foundational benchmark paper, not a corrective one

**Decision (project lead).** The paper is framed as **foundational, not corrective**:
*conditional essentiality is a distinct problem from essentiality, and here is how to measure
it.* Central document created at `paper/PAPER.md`.

**Why foundational rather than corrective.** Bernett et al. retrained eight published models;
no published methods exist for this task, so that structure is unavailable. Rather than treat
this as a deficit, the absence of prior work becomes the asset: we are preventing a field from
starting wrong rather than correcting one that already has. Benchmark-and-baselines papers are
an established genre and need no prior victims. The template's structure transfers completely;
only the rhetorical stance changes.

**The hypothesis, as locked:** conditional gene essentiality is a different prediction problem
from gene essentiality, not an extension of it, and the standard way to pose it measures the
wrong thing. The quantity of interest is the gene x condition interaction; it is abundant
(~48% of variance), largely not measurement noise, and almost entirely unpredicted by frozen
protein embeddings plus condition chemistry -- the apparent performance of such models coming
from main effects and from per-gene memorization that a random condition split silently
permits.

**Structural devices adopted.** (1) The introduction contrasts conditional against
non-conditional essentiality, exploiting the fact that `a_g` IS the non-conditional quantity
measurable in the same data -- enabling a twin-target experiment with everything else held
fixed. (2) The three-scenario device from the template, for which we have evidence on all
three: underpowered models (rejected), label noise (rejected), benchmark design (supported
twice). (3) The paper ends by opening a problem rather than closing one.

**Gold-standard design extended by the project lead: SIMILARITY CONTROL.** The splits as built
hold out random genes and exact compounds, with no control for near-duplicates -- precisely the
sequence-similarity leakage Bernett et al. removed. Three axes added to the plan: homology-
controlled gene clusters, scaffold-controlled compound clusters, and fitness-profile similarity
(stratified rather than split on, to stay leakage-free, since profiles are built from labels).
This was a real gap in the benchmark design and is now the strongest part of it.

**Also recorded:** the RB-TnSeq assay cannot measure unconditionally-essential genes at all --
they yield no viable insertion mutants and never enter the library -- so `a_g` is "average
fitness cost among assayable genes", not textbook essentiality. Stated as a limitation of the
whole transposon-based conditional-essentiality literature, not just this work.

---

## 2026-08-25 — P-ANL-variance-decomposition: the interaction is the LARGEST component

**Goal.** Measure, for the first time, how `var(fit)` actually divides among the terms of
`fit(g,c) = mu + a_g + b_c + I(g,c) + noise`. The project has asserted for months that the
target is noise-dominated and the signal small. It had never been measured.

**Method.** Keio+Caulo+MR1, 1,957,960 measurements, 10,883 genes, 317 conditions.
Main effects by alternating means (sum-to-zero); noise variance from 655,846 replicated
cells via `var(rep_A - rep_B) = 2 sigma^2`.

| component | variance | share |
|---|---|---|
| `a_g` gene main effect | 0.2714 | 40.6% |
| `b_c` condition main effect | 0.0115 | **1.7%** |
| **`I(g,c)` true interaction** | **0.3201** | **47.9%** |
| noise (replicate-estimated) | 0.0654 | 9.8% |
| total | 0.6683 | 100% |

**This inverts a standing assumption.** The interaction is the LARGEST single component --
1.18x the gene main effect -- and 83% of what remains after removing main effects is real
signal, not noise. The dataset is not signal-poor. The problem is that our models capture
very little of the signal that is demonstrably there.

**The apparent contradiction with the modest replicate ceiling (NDCG@5 ~0.66) is itself a
finding.** Total interaction variance is large, but the per-gene RANKING task is
noise-limited for most genes. The most likely reconciliation is that interaction variance
is CONCENTRATED -- a minority of gene x condition cells carry large effects while most are
near-flat -- so aggregate variance is dominated by a few strong hits while the median
gene's condition ordering is mostly noise. If true, this reframes the task: not "predict a
weak signal everywhere" but "find the sparse strong interactions". **Not yet verified;**
the concentration claim needs measuring directly.

**`b_c` is nearly negligible (1.7%).** Condition harshness barely varies once gene identity
is accounted for. Any framing that leans on "eligibility suppresses `b_c`" is leaning on
something that was already small.

---

## 2026-08-25 — P-EVL-random-baseline: chance on `ndcg_at_5` is ~0.16, and we had never measured it

**Gap found.** Every reported number is compared to `chem_null` as "the floor", but
`chem_null` is a *learned population profile*, not chance. The project has never reported
what a RANDOM predictor scores -- an omission that is awkward in a paper about honest
baselines.

**Measured (Keio, 3,000 genes with >=5 conditions, cell-pooled, 5 permutations/gene):**

| predictor | `ndcg_at_5` |
|---|---|
| random per-gene scoring | **0.1611** |
| constant predictor (all ties) | 0.1572 |
| `chem_null` (reported) | 0.3426 |
| `model` (reported) | 0.4608 |
| `chem_knn` (reported) | 0.5091 |

**Reassuring:** `chem_null` sits well above chance (+0.18), so it is doing real work and the
existing comparisons are not against a disguised random baseline. **But the metric's floor is
0.16, not 0, so raw NDCG@5 values overstate how much of the task is solved:** `chem_knn`
covers 41.5% of the distance from chance to a perfect 1.0, and the model 35.7%.

**Caveat:** indicative only -- one organism, all genes with >=5 conditions rather than the
eligibility-filtered val set. Needs computing on the exact reported gene set and added to
the standard report as a permanent row.

---

## 2026-08-25 — P-EVL-leave-compound-out FIRST RESULT: the lookup's margin collapses

**Hypothesis.** The primary condition-holdout split leaves nearly every held-out condition
a chemically near neighbour in train, and that neighbour is what chem-kNN interpolates from.
Remove it by holding out whole stressor compounds and the lookup's advantage should shrink.

**Method.** `leave_compound_out`, 20% of stressor compounds held out GLOBALLY, Keio+Caulo+MR1,
split_seed 0, model_seed 0, on the regenerated embeddings. 9,534 common eligible val genes.

| `ndcg_at_5` | condition_holdout | leave_compound_out |
|---|---|---|
| `chem_null` | 0.3426 | 0.4284 |
| `model` | 0.4608 | 0.4505 |
| `linear_mf` | 0.5122 | 0.4621 |
| `chem_knn` | 0.5091 | 0.4704 |
| **lookup margin (`chem_knn` - `model`)** | **0.0483** | **0.0199** |

**Result — the predicted signature.** The lookup's advantage over the learned model falls
by **59%** (0.0483 -> 0.0199) once the manufactured near neighbour is removed. `linear_mf`,
the other method that exploits per-gene structure, falls further still (0.5122 -> 0.4621)
and no longer leads. This is the first EMPIRICAL evidence for the 2026-07-06 rigging
argument, which until now was an argument rather than a measurement.

Corroborating detail: `chem_null`'s within-gene Spearman drops to **0.0089** on this split,
essentially zero, confirming the population profile carries no gene-specific ordering — the
gene-specific component is what the split is stressing, and it is exactly the component that
shrinks.

**Caveats that must travel with this number.**
1. **The two splits have different val sets**, so absolute values are not directly
   comparable across columns. Only the WITHIN-split lookup margin is a fair comparison.
2. `chem_null` RISES sharply (0.3426 -> 0.4284), meaning the leave-compound-out val
   conditions are systematically easier for a population profile. That needs explaining
   before the result is published -- most likely the held-out compounds' conditions are
   less idiosyncratic than a random condition sample, which would mean the split is easier
   in a way unrelated to near-neighbour removal. **Do not publish this result until that
   is diagnosed.**
3. Three organisms, one split seed, one model seed. The full 23-organism, 3-seed run is
   required before any claim.

**Status.** Directional and strong, not yet publishable. -> `P-EVL-leave-compound-out-full`
and `P-ANL-lco-null-rise` on the board.

---

## 2026-08-25 — P-FIX-data-repin RESULT: pipeline restored; regenerated embeddings are BETTER

**Method.** Repointed `pipeline.EMB_DIR` at the regenerated
`data/processed/PLM_embeddings_layer8/` and ran the R-EVAL fast gate (Keio+Caulo+MR1,
split_seed 0, model_seed 0) against the baseline pinned on the lost 48-organism release.

**The gate FAILED, as designed, and the failure is fully attributable.**

| method | old (lost release) | new (regenerated) | delta | uses the embedding? |
|---|---|---|---|---|
| `model` | 0.4468 | **0.4608** | **+0.0140** | YES |
| `chem_knn` | 0.5091 | 0.5091 | -0.0000 | no |
| `linear_mf` | 0.5122 | 0.5122 | +0.0000 | no |
| `chem_null` | 0.3426 | 0.3426 | -0.0000 | no |

Common eligible val genes: 9,152 before and 9,152 after — identical.

**Interpretation.** Every quantity that does not touch the embedding reproduces
BIT-EXACTLY, including the gene count. That isolates the change completely: the canonical
data, the split, the eligibility filter and all three baselines are reproducing the lost
release exactly for these organisms, and the only thing that moved is the model, which is
the only consumer of the embedding. **The regenerated ProteomeLM-L8 embeddings are better
for this task than the originals, by +0.0140 `ndcg_at_5` and +0.0128 Spearman.**

**What does NOT change.** The ordering is unchanged and the conclusion stands:
`chem_knn` (0.5091) still beats the model (0.4608); the gap narrows from 0.0623 to 0.0484
but does not close. Note also that on this 3-organism subset `linear_mf` (0.5122) edges
`chem_knn` — true in the old baseline too, and an instance of the known small-subset
artifact; it reverses on the full 23-organism panel.

**Consequence for the paper.** Every headline number understates the model slightly and
must be re-measured on the full panel. The direction is favourable to the model and
therefore cannot be accused of flattering the negative result — worth stating explicitly,
because a negative result produced on a *weaker* version of one's own model is exactly what
a reviewer should suspect.

**Caveat retained.** ProteomeLM is proteome-contextual and this runs on the 62-organism
release, so the improvement may reflect richer proteome context rather than a better
encoder per se. Not diagnosed further; recorded as an open question.

**Action.** Fast baseline re-pinned with the movement attributed. The full 23-organism
re-pin is queued.

---

## 2026-08-25 — P-ANL-defensibility-audit: a durable finding does not survive its own primary metric

**Goal.** Before writing the paper, audit every load-bearing claim for whether it is
defensible on the PRIMARY metric and on the data as it now stands.

**Headline finding.** Durable Finding #2 -- carried in the synthesis, the README and the
dashboard since June -- is a Spearman-only claim that is FALSE on `ndcg_at_5`.

[INCORRECT] - **The signal is gene-idiosyncratic and local.** Population/cross-gene structure ≈ 0 within-gene signal; per-gene history carries most of it.
[CORRECTION - 2026-08-25]: True on the SECONDARY metric only. On parity-locked values
(n=39,356): `within_gene_spearman_mean` chem_null 0.0235 vs model 0.1522, so chem_null is
15.4% of the model and "approximately zero" is fair. But on the PRIMARY metric
`ndcg_at_5`, chem_null is 0.3434 against model 0.4319 -- **79.5% of the model's score**.
The population condition profile is one of the strongest single components of
top-5 retrieval performance, not a negligible one. The correct statement is metric-
specific: *population structure carries almost no full-list ordering information, but it
carries most of the top-5 retrieval performance; only the increment above it is
gene-specific.* Measured on the span the model must actually cover, chem_null -> chem_knn
is 0.1419 `ndcg_at_5`, of which the model captures 62.4%.

**Why this recurred.** Identical failure mode to the 2026-06-24 correction: a claim
established while Spearman was primary, never re-checked after `ndcg_at_5` was made
primary on 2026-06-24. Making a metric primary does not retroactively re-derive the
findings that were reached under the old one. Every durable finding stated before
2026-06-24 needs the same re-check; this is now a plan item.

**Consequence for the paper.** The abstract cannot say "population structure carries no
within-gene signal" without a metric qualifier, and Figure 2 must show chem_null high on
the NDCG panel and near-zero on the Spearman panel -- which is itself an informative
result about what the two metrics measure, and worth a paragraph rather than a footnote.

**Also audited (full list in `paper/PAPER_OUTLINE_2026-08-25.md` §6b).** Twelve further
claims were checked; the unresolved vulnerabilities with no current mitigation are: the
eligibility threshold has never been sensitivity-tested; the split uses a single seed so
all reported variance is model-init variance and not split variance; chem-kNN's `k` was
never tuned, so the gate's strength is unquantified; the relevance definition
`max(0, -fit)` was never varied; and the linear-MF "fitness-aware representation" argument
is weaker than stated because that latent is transductive, not inductive.

**Source:** `paper/PAPER_OUTLINE_2026-08-25.md`.

---

## 2026-08-25 — P-FIX-data-repin + paper items A-H: implementation pass

**Goal.** Work the paper improvement list A-H (plus K, the ceiling-parity defect found
during the documentation pass).

**A -- data footing. The blocker was misdiagnosed; it is recoverable.**
[INCORRECT] - ProteomeLM-L8 is now formally RETIRED and its replacement, ESM-C, is a different representation, not a restoration.
[CORRECTION - 2026-08-25]: ProteomeLM-L is a *proteome-context model applied on top of
per-protein ESM-C embeddings* -- a deterministic function of (ESM-C 600M embeddings,
frozen checkpoint). All three inputs survived: the checkpoint at
`/data/ds85/huggingface_cache/models--Bitbol-Lab--ProteomeLM-L`, the model source at
`~/projects/ProteomeLM`, and the protein sequences at `data/raw/aaseqs`. The embeddings
are therefore REGENERABLE, and `src/data/encode_proteomelm_layers.py` already existed to
do it. ProteomeLM is restored to Status PRIMARY in `docs/terms.md`.

Two obstacles surfaced en route, both now fixed and both worth remembering:
1. The 2026-07-06 ESM-C run produced a plain a plain locusId-to-tensor mapping, not the
   the documented two-key bundle the ProteomeLM script expects. Added
   `_unpack_esmc_bundle`, which accepts both and sorts by locusId so row order is
   deterministic rather than dependent on dict insertion order.
2. **The ESM-C variant is load-bearing and was wrong.** ProteomeLM-L requires 1152-d
   input, i.e. ESM-C **600M**. The 2026-07-06 run generated ESM-C **300M** (960-d),
   chosen for the shared-embedding proposal. The mismatch surfaces only as a matmul
   shape error deep in the forward pass. ESM-C 600M is being regenerated to
   `data/processed/ESMC_embeddings_600m/`. The directory name `ESMC_embeddings` records
   no variant, which is precisely how this was missed -- see `docs/bugs.md`.

**Caveat that must reach the paper.** ProteomeLM is proteome-CONTEXTUAL: it attends across
an organism's whole proteome. The regeneration runs on the 62-organism release, whose
proteomes may differ in composition from the 48-organism ones. So regenerated embeddings
are not guaranteed bit-identical to the lost ones even for genes whose sequence is
unchanged, and any number that moves must be attributed rather than absorbed.

**K -- the ceiling was never computed by the pipeline at all.** Root cause of the parity
defect: `runner.METHODS` is `(model, chem_knn, linear_mf, chem_null)` and the evaluation
never called a ceiling function. The published ceiling came from a one-off 2026-05-25
measurement on a different gene set. Fixed: `retrieval_noise_floor` now takes
`eligible_genes` and an optional per-gene return, and the pipeline computes the ceiling on
the parity common gene set and emits it in the results row.

**B -- paired bootstrap.** Added `paired_hierarchical_bootstrap_ci`: joins two methods on
their common genes, bootstraps the PER-GENE DELTA org->gene, and returns a CI plus a
two-sided p. Cancels the shared per-gene difficulty that makes two marginal CIs overlap.
A unit test constructs the exact situation -- marginal CIs overlap, paired delta CI
excludes zero -- so the distinction is asserted, not just asserted-about.

**C -- leave-compound-out.** `materialize_leave_compound_out` holds out whole stressor
compounds GLOBALLY (not per organism -- a compound held out in one organism but present in
another leaks through the shared chemistry encoder and shared weights). Medium components
are never held out; removing a background nutrient changes what the assay is. Optional
`compound_groups` supports scaffold-level holdout. `assert_no_compound_leakage` verifies
the guarantee rather than trusting it. 370 distinct stressor compounds are available.

**D -- similarity-stratified diagnostic.** `nearest_train_condition_distance` computes, per
val gene, the chemical distance from its val conditions to its OWN train conditions;
`similarity_stratified_report` buckets by that distance and reports every method in every
bucket. This converts "the split manufactures near neighbours" from an argument into a
measurement.

**E -- the untested model families.** `src/ranking/eval/extra_baselines.py` adds
`gbdt_predict` (sklearn HistGradientBoostingRegressor on embedding+chemistry, with
train-only PCA and row subsampling, both documented concessions that can only hurt the tree
arm) and `resmem_predict` (the lookup memorizes the model's RESIDUAL, and degrades to the
pure model on a cold gene by construction). Verified by grep that no tree or
learned-nonparametric baseline had ever existed in this codebase.

**F -- de-meaned and denoised targets.** `src/ranking/targets.py`. `fit_additive_effects`
estimates mu/a_g/b_c on train rows by alternating means, under a sum-to-zero constraint --
without it mu silently absorbs mean(a)+mean(b) and effect magnitudes are not comparable
across fits. `apply_demeaning` raises rather than silently substituting 0 when b_c is
unestimable, which on the cold-column primary split is ALWAYS: b_c cannot be estimated for
a condition with no training rows. That constraint is structural and cannot be engineered
away; the paper must state it.

**G -- external validation, reframed.** Integrating MtbTnDB was scoped as days of work.
A stronger and cheaper option was already on disk: every locked design decision was made on
the same 23 replicate organisms, and the 62-organism release contains 39 more that informed
no decision. `configs/experiment/P-EVL_external_orgs.yaml` runs the locked pipeline on
those 39 unchanged. Honest scope: this controls for benchmark-design overfitting, NOT for
platform or protocol effects, since it is the same database and assay.

**H -- related work, citations verified.** The first draft claimed there was no field of
overclaimers to position against. That was wrong: a critique literature exists in this exact
domain -- a ridge baseline beating DeepDEP on cancer dependency (rho 0.276 vs 0.137, all
gene sets), "Deep-learning-based gene perturbation effect prediction does not yet outperform
simple linear baselines" (no model beat a no-change baseline), and a Sci Rep 2024 robust
evaluation. Cross-field memorization citations verified: Grinsztajn NeurIPS 2022, Dacrema
RecSys 2019, Khandelwal ICLR 2020, Feldman STOC 2020. Written into the outline as §2b.

**Tests.** 170 unit tests pass, including 24 new ones across the five new modules.

**Source:** `paper/PAPER_OUTLINE_2026-08-25.md`.

---

## 2026-08-25 — P-ANL-paper-outline: outline for the negative-result paper

**Goal.** Assess Bernett/Blumenthal/List, *Cracking the black box of deep
sequence-based protein-protein interaction prediction* (Brief. Bioinform. 25(2),
2024) as a genre template, and outline this project's paper.

**Result.** Genre fits (rigorous negative -> benchmark contribution); structure does
not (no field of published overclaimers to retrain; our own models are the subjects).
Distinguishing asset is the empirical replicate ceiling, which Bernett lacks. Main
missing experiment is the leave-compound-out split. Blocking item is the data
footing. Ten-section outline, seven figures, ten improvement items.

**Source:** `paper/PAPER_OUTLINE_2026-08-25.md`.

---

## 2026-07-06 — canonical parquet rebuilt on the 62-organism release

Rebuilt `data/derived/canonical/v0/` on the restored 62-org `feba.db` +
`media_composition_v5.xlsx`, excluding 94 wholly-undefined-media experiments.
`fitness_experiment_long.parquet` = 33,433,466 rows (was 27,410,721); 9,438
experiments; media_master 191; media_components_long 4,078. Manifest at
`data/canonical_build_manifest_v0_62org.json`.

**Source:** `archive_docs/project_memory/progress.md`.

---

## 2026-07-06 — canonical chemistry filled + ESM-C generated + Phase-0 PASS

Filled `Media_Components_ML` for all 69 new media (4,406 -> 8,622 rows, 135 canonical
IDs); 64% of new components reused existing decomposition rules; ~11 new small-molecule
IDs; ~6 non-nutrient additives flagged `Include_in_ml=False`. Fixed a dedup bug (the
reuse-map had aggregated each ingredient's decomposition across all media, causing 6x
row inflation). Generated mean-pooled ESM-C 300M (960-d) for 279,140 proteins / 62 orgs
in 24 min. **Phase-0 gate PASSED:** orthologs cluster in ESM-C space — ortholog cosine
median 0.92-0.99 vs random ~0.70, AUROC 0.97-0.998 across close and distant pairs
including Keio<->M.tb, cosine monotone in sequence identity. Caveat: mean-pooled ESM-C
is anisotropic (random cosine ~0.7) — centre/whiten before similarity use.

**Source:** `archive_docs/project_memory/progress.md`.

---

## 2026-07-06 — first-principles re-examination: the warm benchmark is rigged (H1 retired)

**Goal.** Re-derive the problem from scratch, taking no prior conclusion at face value.

**Result.** Amends the framing of the entire warm-task program without changing any
measurement. (1) `fit(g,c) = mu + a_g + b_c + I(g,c) + noise`; within-gene ranking of
eligible genes is predicting the pure interaction `I(g,c)`, a target chosen by the
metric rather than discovered. (2) Two hypotheses had been conflated — **H1** "beat
chem-kNN" (a benchmark) vs **H2** "do embedding + chemistry contain generalizable
`I(g,c)` signal" (the science). (3) The warm contest is rigged three ways: the kNN is
handed the target gene's own labels at inference; eligibility pre-selects history-rich
genes; a random condition split leaves every held-out condition a chemically near
measured neighbour. So H1's "no" is expected, not a verdict on deep learning. (4) The
family that could beat the kNN — learned non-parametric (EASE / learned-metric kNN /
TabR / canonical ResMem), plus de-meaned and denoised targets — was never built. But a
learned-*local* winner would confirm memorization, not refute it. (5) The replicate
ceiling is the ceiling of the single-noisy-measurement task; denoising raises it.

**Decision.** Demote chem_knn from "the gate you must beat" to "a reference for what
memorization achieves". Adopt leave-compound-out + cold_gene, scored against chem_null
as a fraction of the denoised ceiling. No code or behaviour change; R-EVAL untouched.

[INCORRECT] - **Conclusion.** Three orthogonal fusion designs converging on ≈ 0, on top of the earlier encoder (R1), capacity, and objective (R-LOSS) sweeps all failing, is strong evidence the chem-kNN ceiling is **real, not a fusion-design artifact**. A global model — alone or hybridized — adds nothing promotable over local lookup on the warm task.
[CORRECTION - 2026-07-06]: Re-scoped, not withdrawn. The measurements stand. The
conclusion is restated as: *no GLOBAL model beats a lookup that has been handed the
gene's own answer key on a task selected to reward exactly that.* That is expected and
low-significance, and it is not the scientific question. The live science is H2 on
honest splits.

**Source:** `paper/SCIENTIFIC_SYNTHESIS.md` §9.

---

## 2026-07-06 — data-root loss and live re-download (newer 62-org release)

The external data root disappeared mid-session: `feba.db`, the canonical parquet and
all embeddings gone, symlink intact but target missing. Re-downloaded the raw DB from
the live Fitness Browser — integrity OK, 62 orgs / 9,532 exps / 33.8M rows, sha
`b6627137...`. This is a **newer, larger release** than the pinned 48-org DB (sha
`627f2097...`). **Consequence: every headline number was computed on the old release
and will shift on the new one.** ProteomeLM embeddings were not recoverable.

**Source:** `archive_docs/project_memory/progress.md`, `docs/bugs.md`.

---

## 2026-07-06 — formal plan: cross-organism shared gene-condition embedding

Implementation-ready form of the ortholog-rewiring reframe as a shared-embedding
factorization `fit_o(g,c) ~ a_g^o + b_c^o + (s_g + d_g^o) . v_c`, with a conserved
sequence prior `s_g = phi(emb)`, an org-specific fitness-learned deviation `d` (the
rewiring signal), and a shared condition vector `v_c`. Chose ESM-C over ProteomeLM
(orthologs must be similar by construction so `d` isolates rewiring) and over Evo2
(DNA diverges faster). Data grounding: ~117 canonical conditions in >=10 orgs; 179k
genes with a cross-org ortholog; DvH<->Miya 1,784 orthologs; M. tuberculosis only 8%
orthologous, so it helps cross-org condition transfer but not the rewiring map.

**Source:** `archive_docs/PROPOSAL_crossorg_shared_embedding.md`.

---

## 2026-07-06 — project dashboard built

Dependency-free static dashboard from a pure-Python generator over a content model:
13 per-experiment extracts + 8 durable learnings. Pages: overview, key result, data,
timeline, learnings, directions, status, limitations, references. **Known weakness:**
hand-maintained, duplicates ledger content, and nothing verifies it.

**Source:** `dashboard/README.md`.

---

## 2026-06-29 — adversarial hypothesis dialogue -> five-direction slate

Two-agent Socratic exercise: a first-principles generator blind to all project
code/docs produced 12 directions; a critic with full prior-work access pressure-tested
them. Decisive filter: *does a chemistry-kNN's retrieved neighbour structurally contain
the target's answer?* Headline candidate: **ortholog conditional-response divergence**
(conserved vs rewired essentiality across orthologs) — the one direction that cleanly
escapes, because the target is a cross-organism comparison and within-gene cross-org
transfer is ~0, so an organism-local lookup cannot compute it. Second: fitness-fingerprint
MoA sensor, escaping only on the structurally-distant slice under leave-compound-and-
organism-out.

**Source:** `archive_docs/DIRECTIONS_adversarial_slate_2026-06-29.md`.

---

## 2026-06-25 — R-DARK spike: signal real, dark-genome enrichment falsified

**Hypothesis.** Among unannotated (hypothetical/DUF) genes, some show chemically-specific
essentiality the lookup cannot explain, and orphans are enriched for these.

**Method.** Studentized lookup residuals on replicated cells only (`n_rep>=2`), deviation
required to reproduce across a gene's two replicate halves (|z|>3 both, concordant),
within-condition replicate-shuffle null. Inline kNN verified against the harness to ~1e-16.

**Result — split verdict.** (1) Reproducible chemically-unexpected conditional essentiality
is REAL: confirmed orphan surprises run ~10x the null in all three orgs tested
(794/76, 2235/293, 2404/234; emp_p=0.002).
[INCORRECT] - the dark genome is special / enriched for surprising essentiality
[CORRECTION - 2026-06-25]: FALSIFIED. Orphans are *less* surprising than annotated genes
in all three orgs (rate ~0.02-0.04 vs ~0.06-0.10; label_p=1.0). A confirmed
chemically-unexpected reproducible phenotype is approximately a `SpecificPhenotype`,
already catalogued in `feba.db` (38,525 rows), so novelty is thin.

**Source:** `archive_docs/PROPOSAL_dark_genome_residual_miner.md`.

---

## 2026-06-25 — literature diligence: the wall is a named cross-field regime

Citation-verified multi-agent sweep across 8 fields. **Finding:** "local memorization
beats global parametric learning" is a well-characterized regime, not a leak or a bug —
kNN-LM ("Generalization through Memorization"), Feldman long-tail necessity, tabular
DL losing to trees, recsys kNN-beats-neural, QSAR applicability domain. chem_knn winning
is the EXPECTED outcome. Ranked categorically-different techniques not yet tried: ResMem,
Correct-and-Smooth, adaptive Meta-k gate, EASE, TabR.

**Source:** `archive_docs/LITERATURE_local_vs_global_memorization.md`.

---

## 2026-06-24 — R-EVL-cold-gene confidence: NDCG@5 made primary, correcting a Spearman-only read

**Method.** The harness only ever bootstrapped Spearman. Added a hierarchical (org->gene)
bootstrap CI for `ndcg_at_5`; `ndcg_at_5` point values unchanged, so R-EVAL stayed
bit-exact (0.4468 / 0.5091).

[INCORRECT] - R-COLD confirmatory: full 23-org cold-gene Spearman is CI-disjoint (model 0.0735 [0.0523, 0.1034] vs chem-NULL 0.0359 [0.0230, 0.0514]), upgrading confidence from per-seed point estimates to disjoint 95% CIs.
[CORRECTION - 2026-06-24]: That read judged disjointness on the SECONDARY metric alone.
On the PRIMARY metric the CIs OVERLAP — `ndcg_at_5` model 0.2748 [0.2424, 0.3179] vs
chem_null 0.2447 [0.2118, 0.2823]. The +0.030 is a consistent point-estimate win
(disjoint across all 3 seeds) but is NOT CI-significant. `ndcg_at_5` has higher per-gene
variance and therefore a wider CI, so a Spearman-only confidence read overstates
significance — precisely why `ndcg_at_5` is now primary project-wide. Remaining: a
paired/pooled bootstrap on the per-gene `ndcg_at_5` delta, which cancels shared per-gene
variance and is more powerful than two marginal CIs.

Also noted: multi-seed bounds are the mean of per-seed CIs — a summary band, not a
pooled bootstrap.

**Source:** `archive_docs/project_memory/progress.md`; `archive_docs/decisions/rcold/R-COLD-DEC-001.md`.

---

## 2026-06-23 — R-COLD (`R-EVL-cold-gene`): first positive — the embedding generalizes

**Hypothesis (H-R-COLD-01).** chem_knn wins on the primary split only because it retrieves
each gene's own history — pure within-gene memorization that says nothing about whether the
frozen embedding carries transferable signal.

**Method.** `cold_gene` split holds out whole genes per organism (zero train rows), making
chem_knn structurally inapplicable and linear_mf unlearnable, leaving chem_null as the only
applicable baseline. 23 orgs, 3 model seeds, n=11,761 eligible cold val genes, denominator
parity restricted to {model, chem_null}.

**Result.** chem_knn coverage **0.0000** (confirmed inapplicable). Model beats chem_null on
genes it never trained on: `ndcg_at_5` **0.2748 vs 0.2447** (delta +0.0301), Spearman
**0.0735 vs 0.0359** (delta +0.0376), every model seed [0.2732, 0.2746, 0.2765] above the
constant gate. Fast 3-org set reproduces sign and magnitude (+0.0271).

Full hierarchical (org->gene) 95% CIs, both metrics — these values stand; only the
significance reading of them was later corrected (see 2026-06-24):

| metric | model [95% CI] | chem_null [95% CI] | disjoint? |
|---|---|---|---|
| `ndcg_at_5` (PRIMARY) | 0.2748 [0.2424, 0.3179] | 0.2447 [0.2118, 0.2823] | NO — overlap |
| `within_gene_spearman_mean` | 0.0735 [0.0523, 0.1034] | 0.0359 [0.0230, 0.0514] | yes |

**Meaning.** The frozen embedding carries transferable gene-specific conditional-response
signal. This reframes the warm negative as a MEMORIZATION gap, not an embedding-value gap.
Why the T-regime prior did not apply: T-regime measured cross-ORGANISM transfer (~0.045);
cold_gene is cross-GENE transfer *within* known organisms, a much easier ask.

**Scope.** NOT a promotion — chem_null is a weaker bar than the chem_knn gate; no headline
number moves. See the 2026-06-24 correction above for the CI status.

**Source:** `archive_docs/decisions/rcold/R-COLD-DEC-001.md`.

---

## 2026-06-18 — R-AUG (`R-TRN-org-augmentation`): negative transfer

**Hypothesis (H-R-AUG-01).** Training the shared encoder on all 48 embedded organisms
(eval held to the locked 23) narrows the model-to-gate gap.

**Result.** REJECTED, sign reversed. `ndcg_at_5` 0.4166 (aug_48org) vs 0.4319 (base_23org),
delta **-0.0152**; Spearman 0.1522 -> 0.1270; every aug seed below every base seed. The
chem_knn gate was bit-identical across arms (drift 0.000000), so the A/B is clean. The extra
organisms' conditional structure does not transfer and pulls shared weights off the eval
orgs. Closes the "add more organisms / external Tn-seq data" line for the within-org headline.
Note: measured warm-only — the cold-gene regime was never retested and may differ.

**Source:** `archive_docs/decisions/raug/R-AUG-DEC-001.md`.

---

## 2026-06-17 — R-TOPK: top-k-truncated losses (objective axis closed)

**Hypothesis (H-R-TOPK-01).** A loss truncating NDCG gain to the top-5, matching the metric
exactly, beats the gate.

**Result.** REJECTED. 5 arms x 3 seeds x 23 orgs: no loss beats the gate (0.4852);
`pointwise_huber` stays best (0.4319); truncated variants fall BELOW their untruncated forms
(lambdarank_top5 0.4239, approxndcg_top5 0.3695). Found and fixed an approxndcg_top5 training
freeze — the top-k gate reused the score temperature, causing vanishing gradient; fixed with
`gate_temp=2.0`. With pointwise, pairwise, whole-list and top-k-truncated all tested, the
objective axis is closed.

**Source:** `archive_docs/decisions/rloss/R-TOPK-DEC-001.md`.

---

## 2026-06-17 — RankingBatch wired into training; R-EVAL re-pinned

Replaced hand-rolled batching with the tested `RankingBatch` samplers. `ndcg_at_5` moved
0.4347 -> 0.4319 (delta -0.0028, within tolerance), Spearman 0.1509 -> 0.1522; chem_knn
bit-exact. Three-lens adversarial review clean — movement is RNG variance from a different
shuffle stream, not a bug. `reval_baseline.json` re-pinned.

**Source:** `archive_docs/project_memory/progress.md`.

---

## 2026-06-15 — ranking-branch cleanup: reorganize, prune, modular runner

Six gated steps, each verified bit-exact by R-EVAL plus pytest: built the R-EVAL regression
harness; extracted the model into `src/ranking/models.py`; decoupled ranking from the T-tier;
relocated pipeline and trainers; added the shared `runner.py` (ArmSpec + run_experiment +
standardized report); pruned the T-regime and dead code (130 files, -13.4k lines).
**Learned:** training is deterministic, so the gate is bit-exact, which made every refactor
verifiable.

**Source:** `archive_docs/PRUNED_INDEX.md`, `archive_docs/project_memory/progress.md`.

---

## 2026-06-06 — R-HYBRID-B: three learned fusions converge on zero

**Hypothesis (H-R-HYBRID-01 round 2).** A learned hybrid of the global model and chem_knn
extracts more complementary signal than R-HYBRID-A's static ensemble.

**Result.** REJECTED. Full 23-org, 3 seeds, denominator parity: residual +0.0002 `ndcg_at_5`
/ -0.0049 Spearman; retrieval-augmented -0.0059 / -0.0109; learned gating -0.0081 / -0.0181.
The learned hybrids extracted LESS than the crude static ensemble. Best (residual) is a tie
that sign-flips across seeds (+0.0048 / -0.0076 / +0.0033). Learned gating settled at mean
alpha ~0.46 but every unit of weight on the model hurt. A Keio-only dev run gave residual a
misleading +0.042 that vanished on full multi-org eval — the known single-org-ceiling artifact.

**Source:** `archive_docs/decisions/rhybrid/R-HYBRID-DEC-002.md`.

---

## 2026-06-06 — R-HYBRID-A: static z-score ensemble, first thing to edge past the lookup

Fixed convex combination `alpha.z(kNN) + (1-alpha).z(model)` with honest held-out alpha
selection: **+0.008 `ndcg_at_5`** over chem_knn. Real complementary signal, but below the
promotion delta. Not promotable.

**Source:** `archive_docs/decisions/rhybrid/R-HYBRID-DEC-001.md`.

---

## 2026-06-06 — R-LOSS (`R-TRN-loss-family`): the objective is not the lever

**Hypothesis (H-R-LOSS-01).** Swapping pointwise MSE for a ranking loss beats the gate.

**Result.** REJECTED. MSE, Huber, RankNet, LambdaRank, ListMLE, ApproxNDCG: Huber marginally
best (`ndcg_at_5` 0.435); LambdaRank reaches good NDCG but collapses Spearman (top-focused);
ApproxNDCG worst. None beat chem_knn.

**Source:** `archive_docs/decisions/rloss/R-LOSS-DEC-001.md`.

---

## 2026-06-06 — R-CONF: the negative is noise-robust

**Hypothesis (H-R-CONF-01).** Label noise, not structure, explains the model's deficit.

**Result.** REJECTED as an explanation. Stratifying eligible val genes by `abs_t`, chem_knn
beats the model in EVERY confidence quartile and at EVERY per-cell |t| threshold. Noise does
depress measured performance — every method and the replicate ceiling rise monotonically with
confidence (ceiling 0.59 -> 0.83) — and accounts for part of the deficit (the gap narrows
0.055 -> 0.040) but never closes it. Confidence-weighted training does not help (delta ~-0.004).
So part of the model-to-ceiling gap is irreducible biological noise and the rest is structural.
This pre-empts the most likely reviewer objection.

**Source:** `archive_docs/decisions/rconf/R-CONF-DEC-001.md`.

---

## 2026-06-06 — R1 (`R-TRN-chemistry-encoder`) + capacity: neither is the lever

**Hypothesis (H-R-CHEM-01).** Structural fingerprints (Morgan / RDKit / MACCS) outperform the
425-d multihot chemistry encoder.

**Result.** REJECTED. 6 arms x 3 seeds x 23 orgs: no arm differs meaningfully; multihot
marginally best; none beat chem_knn. Separately, **capacity is not the lever either**:
linear_mf (~13k params, bilinear) is statistically indistinguishable from the deep model
(2.5M params, nonlinear). Because linear_mf's free per-gene latent is a representation learned
entirely from fitness data, this also shows the warm bottleneck is **not** the embedding's
fitness-blindness — swapping a fitness-blind representation for a fitness-aware one does not
beat the lookup.

**Source:** `archive_docs/decisions/r1/R1-DEC-001.md`.

---

## 2026-05-25 — R-LOCK-1..4: the benchmark instrument

Locked the measuring instrument every R-tier experiment inherits: `condition_holdout` split
(frac 0.20, seed 0; held-out conditions 100% disjoint from train, so the task is inductive
cold-start matrix completion, not warm collaborative filtering); `tail_g` eligibility with
per-organism thresholds and train weight `w_g`; denominator parity; retrieval-primary metrics
with hierarchical org->gene bootstrap and BH-FDR; three matched baselines plus a replicate
ceiling. Dataset: 48 organisms, 182,447 (orgId, gene) pairs, 7,500 expNames, ~4,200 distinct
conditions; 23 organisms carry a reliable replicate noise floor and are the headline subset.
Per-org cross-replicate Spearman varies 5.4x across organisms.

**Source:** `archive_docs/decisions/r_lock/`, `archive_docs/audits/2026-05-25_ranking_pivot_audit.md`.

---

## 2026-05 — the reframe: from cross-organism regression to within-organism ranking

Two changes after T7-prep diagnostics. Generalization claim narrowed from cross-organism to
**within-organism** (a known organism's response to novel condition combinations). Metric
changed from RMSE to **per-gene ranking**, the quantity a biologist actually uses.

**Source:** `paper/SCIENTIFIC_SYNTHESIS.md` §2.

---

## 2026-04/05 — T-regime: cross-organism transfer of the conditional signal is ~0

**Hypothesis (§0, the foundational bet).** A frozen, proteome-contextualized, sequence-derived
embedding plus a condition-chemistry encoding contains enough information to predict conditional
gene essentiality, generalizing across genes and organisms.

**Result.** Substantially undermined for the cross-organism case. RMSE/MAE optimized acceptably
across T1-T6, but within-gene ranking of conditions was **~0.045 Spearman** against a ~0.43
replicate floor — indistinguishable from random.

**Mechanism (the pivotal diagnosis).** RMSE is dominated by the **gene mean**. A model can fit
gene means and the population condition effect and score decent RMSE while learning nothing
about the gene-by-condition interaction. Later confirmed: chem_null scores ~0 within-gene
Spearman. Two contributing causes for the cross-org failure: homology distance (held-out
organisms' genes lie far from any training gene in embedding space) and the embedding's
fitness-blindness — ProteomeLM beat raw ESM-C by only ~0.006 RMSE, so proteome-contextual
machinery added almost no conditional signal.

**Durable finding.** Predicting the LEVEL of fitness is easy and gene-mean-dominated; predicting
the within-gene conditional ORDERING is the real problem, and cross-organism transfer of it is ~0.

**Source:** `archive_docs/decisions/tier1/` through `tier5/`, `archive_docs/tier_reports/`.

---

## 2026-04 — S0-S5: data characterization, baselines, split lock, feature contract, quality policy

Foundational stage regime: smoke tests, data characterization, baseline establishment, split
locking, the frozen S4 feature contract (425-d multihot,
`data_contract/preprocessing/de21504134c84a6c/`), and the S5 quality policy.

**Source:** `archive_docs/decisions/stage0/` through `stage5/`, `archive_docs/tier_reports/`.
