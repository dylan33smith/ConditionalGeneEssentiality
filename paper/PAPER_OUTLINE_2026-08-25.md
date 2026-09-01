# Paper outline — the rigorous-negative / benchmark paper

**Date:** 2026-08-25 · **Status:** planning draft, not a decision
**Prompted by:** PI supplied Bernett, Blumenthal & List, *Cracking the black box of deep
sequence-based protein–protein interaction prediction*, Brief. Bioinform. 25(2):bbae076 (2024)
as a genre example.

---

## Status update 3 — the leave-compound-out pilot (D5 is now answered)

The template's strongest move is "we fixed the benchmark and the results changed." The
first draft could not make it. **Now it can, provisionally.**

Holding out 20% of stressor compounds globally (Keio+Caulo+MR1, seed 0), the lookup's
margin over the learned model falls by **59%**:

| `ndcg_at_5` | condition_holdout | leave_compound_out |
|---|---|---|
| `chem_null` | 0.3426 | 0.4284 |
| `model` | 0.4608 | 0.4505 |
| `linear_mf` | 0.5122 | 0.4621 |
| `chem_knn` | 0.5091 | 0.4704 |
| **`chem_knn` − `model`** | **0.0483** | **0.0199** |

`linear_mf` — the other per-gene-structure exploiter — falls furthest and loses its lead.
`chem_null`'s Spearman drops to 0.0089, confirming the gene-specific component is what the
split stresses.

**Two caveats that gate publication.** (i) The two splits have different val sets, so only
the within-split margin is comparable. (ii) `chem_null` RISES sharply (0.3426 → 0.4284),
meaning the leave-compound-out conditions are systematically easier for a population
profile. If the held-out compounds are simply less idiosyncratic, part of the margin
collapse is a difficulty artifact rather than near-neighbour removal. **This must be
diagnosed before the result is used** (`P-ANL-lco-null-rise`). Three organisms, one seed.

---

## Status update 2 — 2026-08-25, after implementing items A–H

**A is RESOLVED and the earlier diagnosis was wrong.** ProteomeLM-L is a proteome-context
model applied on top of ESM-C 600M embeddings — a deterministic function of (ESM-C 600M,
frozen checkpoint). All inputs survived, so the embeddings were regenerated, not lost. The
regression gate then failed *as designed*, and the failure was fully attributable: only the
model moved (+0.0140 NDCG@5), while chem-kNN, linear-MF and chem-NULL were bit-exact on an
identical 9,152-gene common set. **The regenerated embeddings are better than the
originals.** Every headline number therefore *understates* the model, which is worth saying
in the paper: a negative result produced on a weaker version of one's own model is exactly
what a reviewer should suspect, and here the correction runs the other way.

**B, C, D, E, F, K implemented and unit-tested** (24 new tests): paired hierarchical
bootstrap on the per-gene delta; ceiling computed on the parity gene set (the runner had
never computed a ceiling at all — that was the root cause of the parity defect);
leave-compound-out materializer with a leakage assertion; similarity-stratified diagnostic;
GBDT and ResMem baselines; de-meaned and denoised targets.

**G reframed and configured.** Integrating MtbTnDB was days of work; a stronger option was
already on disk. Every design decision was locked on the same 23 organisms, and the release
contains 39 more that informed none. Running the locked pipeline on those unchanged
controls for benchmark-design overfitting — though not for platform effects, since it is
the same assay.

**H done, and it changes the positioning.** A critique literature exists in this exact
domain (see §2b). We are converging evidence, not a lone negative.

**One durable finding did not survive the audit** — see §6b, U10. "Population structure
carries approximately zero within-gene signal" is true on Spearman and false on the primary
metric, where chem-NULL reaches 79.5% of the model's NDCG@5. The abstract must be
metric-qualified.

---

## Status update — 2026-08-25, after the documentation pass

The six-file documentation adoption audited every number in the project. Three things
changed for this paper; the thesis and structure did not.

**1. Table 1 as drafted below was NOT parity-consistent — corrected in §4.** It mixed
three denominators in one table: chem-NULL at n=54,593, model/linear-MF/chem-kNN at
n=39,356, and the ceiling at n=45,943. In a paper whose central methodological argument
is denominator parity, that is the single worst thing a reviewer could find. The
parity-locked values now live in `data_contract/ranking/reval_baseline.json` and
`docs/plan.md`, and §4 has been rebuilt from them.

**2. NEW BLOCKER — the replicate ceiling has never been computed on the parity gene
set.** `reval_baseline.json` contains no ceiling entry at all. Every "fraction of
achievable" figure in the literature-facing story mixes n=45,943 against n=39,356. This
blocks **Figure 2**, the paper's money figure, as currently conceived. Promoted to a
first-tier improvement item (**K**).

**3. The ceiling definition is settled, and it validates the gate.** The comparable
ceiling is the MEAN over parity-eligible val genes, not the median over all val genes.
Confirmation: the promotion delta is `0.15 x (ceiling - chem_kNN)`, and only the
parity-eligible ceiling reproduces the documented NDCG@5 gate. The Methods section can
now state the gate's derivation as verified arithmetic rather than a stipulated number.

Also relevant, smaller: the media-chemistry provenance audit was recovered from a stale
worktree (`archive_docs/v4_workbook_audit.md`) and answers a likely Methods question --
the media decompositions trace to primary literature (DSMZ 380, ATCC 1293, Neidhardt
1974, Miller 1972, and others), not to generated recipes. And a claim that the training
objective was the bottleneck, which R-LOSS and R-TOPK refuted, was still being asserted
in the metric contract; it has been retracted in place.

---

## 0. Is the Bernett paper a good template?

**What that paper does.** Sequence-based deep PPI predictors report near-perfect accuracy.
Bernett et al. show that (i) random train/test splits leak — test pairs share proteins and
high sequence similarity with train; (ii) under leakage the models learn only *node degree*
and *sequence similarity*, not interaction biology; (iii) they build a leakage-controlled
"gold standard" split (sequence-similarity redundancy removal + partitioning so no protein
crosses the split, degree/hub balancing) and retrain the published models; (iv) performance
collapses to ~random. Conclusion: the task is unsolved for proteins dissimilar to studied
ones, and the field needs the corrected benchmark.

**Where the analogy is strong (use this):**
- Same core move: *the evaluation, not the model, was the story.* §9 of SCIENTIFIC_SYNTHESIS
  independently rederived Bernett's central argument — a random split manufactures near
  neighbours and hands a lookup the answer key.
- Same "what is the model actually learning?" decomposition. Bernett: node degree + sequence
  similarity. Ours: `fit(g,c)=μ+a_g+b_c+I(g,c)+ε` — RMSE is dominated by `a_g`, the population
  condition profile is `b_c`, and only `I(g,c)` is the biology. That decomposition is the
  paper's intellectual spine, exactly as degree/similarity is theirs.
- Same deliverable shape: a corrected benchmark + honest baselines + a call on what is
  actually unsolved.

**Where it does NOT fit (must be handled differently):**
1. **No field of overclaimers to knock down.** Bernett retrained 8 *published* models. There is
   no established leaderboard for "conditional essentiality from Tn-seq + PLM embeddings." The
   models that lose here are *ours*. "My model didn't work" is not a paper. The contribution
   must be **the benchmark + the decomposition + the ceiling**, with our models as instruments.
   *Mitigations:* (a) borrow the adjacent overclaiming literature (DepMap-style dependency
   prediction, where "the gene mean is a strong baseline" is an established critique;
   chemical-genetics ML); (b) run at least one off-the-shelf/external method so the result is
   not architecture-specific (see Improvement G).
2. **Their headline is "collapses to random"; ours is subtler.** Ours never collapses — it just
   never beats a trivial lookup, and the *direction* of failure is diagnostic. That is a harder
   story to sell but a more informative one. Lead with the diagnosis, not the loss.
3. **We have something they don't: an empirical noise ceiling.** Biological-replicate agreement
   (NDCG@5 ≈ 0.66, Spearman ≈ 0.39) converts "our numbers are low" into "the achievable range is
   [0.31, 0.66] and here is where each method sits in it." This is our single biggest asset —
   it forecloses "you just didn't try hard enough."
4. **They fixed the benchmark and showed the corrected result. We have not yet.** The
   leave-compound-out split does not exist in the codebase. That is the main missing experiment
   and the direct analogue of their gold-standard dataset.

**The one thing that will kill the paper if mishandled.** Our central negative (§3–§7: "no model
beats chem-kNN") was measured under a formulation we have since concluded is rigged (§9). If the
paper leads with that as the finding, a sharp reviewer makes the §9 argument and the paper dies.
**We must be the ones who demolish our own benchmark, inside the paper.** That self-demolition is
precisely what makes it a Bernett-genre paper rather than a failed-model report.

**Verdict.** Use it as a *genre* template (rigorous negative → benchmark contribution), not a
structural one. Our narrative arc is different: not "leakage → collapse → fix," but
**"decomposition → the benchmark rewards memorization → remove the crutch and real signal
appears → here is the corrected benchmark."**

---

## 1. Premise / thesis

**Working title options**
- *Memorization, not generalization: what benchmarks for conditional gene essentiality actually measure*
- *A gene's own history beats its sequence: decomposing conditional essentiality prediction in bacterial chemical genetics*
- *Conditional gene essentiality prediction is memorization-dominated*

**Thesis (one sentence).** On genome-scale bacterial chemical-genetics data, the natural way to
pose "predict conditional gene essentiality" makes a trivial per-gene chemistry lookup the top
method; decomposing the target and controlling the split shows that protein-language-model
embeddings do carry real, small, generalizable gene-specific conditional signal, and that headline
numbers on this task mostly measure memorization against an unacknowledged noise ceiling.

**Contributions**
- **C1 — A benchmark instrument.** Pre-registered, leakage-controlled within-organism
  conditional-essentiality ranking benchmark: cold-condition holdout, spread-based gene
  eligibility, denominator parity across all methods, retrieval-first primary metric (NDCG@5),
  hierarchical (organism→gene) bootstrap CIs, three matched baselines, and an empirical
  replicate-noise ceiling. 48 organisms / 182k gene records / ~4.2k conditions.
- **C2 — A diagnosis of what the benchmark rewards.** The `μ+a_g+b_c+I(g,c)` decomposition; proof
  that population/cross-gene structure carries ≈0 within-gene signal; the demonstration that a
  random condition split manufactures near neighbours so a lossless per-gene lookup is *expected*
  to win; and a five-axis sweep (encoder, objective, capacity, hybridization, training-organism
  volume) showing that no global parametric model in this family beats it.
- **C3 — The split that separates memorization from generalization, and a corrected benchmark.**
  On held-out *genes*, chem-kNN is structurally inapplicable (coverage 0.000) and the learned model
  beats the population baseline — the failure is memorization dominance, not absent signal. We
  propose leave-compound-out + cold-gene, scored as a fraction of the (denoised) ceiling against a
  chem-NULL bar.

---

## 2. Section-by-section outline, with which existing results go where

### 1. Introduction
Conditional (context-specific) essentiality matters: antibiotic targets, MoA, gene function for the
unannotated majority. PLMs promised transferable gene representations. Frame the question as: *do
sequence-derived representations plus condition chemistry contain generalizable gene×condition
interaction signal?* State up front that the answer depends almost entirely on how the benchmark is
built, and that this paper is about establishing which version of the question is being answered.

### 2. Data and task formalization
- **Data (R0/S1).** feba.db Fitness Browser; 48 organisms, 182,447 (organism, gene) pairs, 7,500
  experiments, ~4,200 distinct conditions; 23 organisms have replicate structure sufficient for a
  noise floor (the headline eval subset).
- **The decomposition.** `fit(g,c)=μ+a_g+b_c+I(g,c)+ε`. Within-gene ranking fixes `g` (kills `a_g`);
  eligibility selects genes where `b_c` is small; therefore the task *is* predicting `I(g,c)`.
  **Key point for the reader:** this target was *chosen by the metric*, not discovered — and most
  reported performance on essentiality tasks lives in `a_g`, not `I`.
- **Evidence:** the T-regime cross-organism regression result — RMSE optimized acceptably while
  within-gene Spearman was ≈0.045 against a ≈0.43 replicate floor. RMSE is gene-mean-dominated.
  chem-NULL (population condition profile) scores ≈0.02 within-gene Spearman. → *Fig 1.*

### 3. The benchmark instrument (Methods-heavy; R-LOCK-1..4)
Split (within-org condition holdout, frac 0.20 — held-out conditions are 100% disjoint from train,
so this is inductive/cold-start matrix completion, not warm collaborative filtering); eligibility
(`tail_g = p95−p5`, per-organism thresholds, train weight `w_g`, with the documented reason `tail_g`
beats IQR); denominator parity; metrics (NDCG@5 primary, within-gene Spearman secondary, both with
hierarchical org→gene bootstrap CI, BH-FDR); baselines (chem-NULL, chem-kNN, linear inductive-MF);
the replicate-noise ceiling. Emphasize the **pre-registration**: locked gates + a dated decision
ledger with hypotheses recorded before results.

### 4. Result 1 — a trivial local baseline beats every learned global model (warm split)
**Table 1** (23 organisms, 3 seeds, identical eligible val genes):

| method | NDCG@5 | within-gene Spearman | n |
|---|---|---|---|
| chem-NULL (population profile) | 0.3434 | 0.0235 | 39,356 |
| deep model (frozen emb ⊕ chem) | 0.4319 | 0.1522 | 39,356 |
| linear-MF (learned free gene latents) | 0.4290 | 0.1426 | 39,356 |
| **chem-kNN (gene's own history)** | **0.4852** | **0.2402** | 39,356 |
| replicate ceiling | *pending* | *pending* | — |

*Provenance: `data_contract/ranking/reval_baseline.json`, tag `full`, condition-holdout
split, seed 0, model_seeds [0,1,2], 23 replicate organisms, MEAN over the identical
parity-eligible val gene set. The ceiling row is deliberately blank: the only measured
values (NDCG@5 0.658 / Spearman 0.393) are on n=45,943, a different gene set, so quoting
them here would reproduce the very parity violation this paper argues against. See
improvement K.*

Note two things the corrected numbers change. The floor is higher than previously
reported (0.3434, not 0.31), so the model's margin over chem-NULL is 0.089 rather than
~0.12. And model-vs-linear-MF is 0.0029 — even tighter than the "statistically
indistinguishable" claim suggested. Neither alters a conclusion; both make the
capacity-is-not-the-lever argument stronger.

Then the axes, each a pre-registered rejected hypothesis:
- **Encoder (R1):** multihot vs Morgan/RDKit/MACCS, 6 arms × 3 seeds × 23 orgs — no difference; none beat the gate.
- **Objective (R-LOSS, R-TOPK):** MSE, Huber, RankNet, LambdaRank, ListMLE, ApproxNDCG, plus top-5-truncated NDCG-matched variants. Huber best (0.432); truncated variants fall *below* their untruncated forms.
- **Capacity:** linear-MF (~13k params, bilinear) ≈ deep model (2.5M params, nonlinear).
- **Representation:** linear-MF's free latent is a *fitness-aware* gene representation learned from scratch — it does not beat the lookup either. So the bottleneck is not the embedding's fitness-blindness.
- **Hybridization (R-HYBRID A/B):** static z-ensemble +0.008 (below the 0.026 promotion delta); three learned fusions (residual / retrieval-augmented / gating) converge on ≈0 or negative; learned gating settles at α≈0.46 but every unit of weight on the model hurts.
- **Training volume (R-AUG):** training on all 48 embedded organisms *hurts* (0.4166 vs 0.4319, every aug seed below every base seed) — negative transfer.
→ *Fig 2 (the money figure: method ladder against the ceiling band), Fig 3 (small multiples of the five failed axes).*

### 5. Result 2 — the negative is not a label-noise artifact (R-CONF)
Stratify eligible val genes by measurement confidence (`abs_t`, Wetmore moderated t). chem-kNN beats
the model in *every* confidence quartile and at *every* per-cell |t| threshold. Noise does depress
everything (the ceiling itself rises 0.59→0.83 with confidence) and narrows the gap (0.055→0.040)
but never closes it. Confidence-weighted training does not help (Δ≈−0.004). *This section exists to
pre-empt the most likely reviewer objection.* → *Fig 4.*

### 6. Result 3 — what the benchmark actually measures (the §9 self-critique; the pivot)
The three ways the warm contest is rigged: the kNN is handed the target gene's own labels at
inference; eligibility pre-selects history-rich genes; a random condition split leaves every
held-out condition a chemically near measured neighbour. Given a high-rank idiosyncratic target, a
lossless non-parametric reader of history beating a lossy parametric model is the *expected*
outcome — a named cross-field regime (kNN-LM "generalization through memorization"; Feldman's
long-tail necessity; tabular DL losing to trees; kNN-beats-neural in recsys; QSAR applicability
domain). So "no model beats the kNN" must be restated as **"no global model beats a lookup handed
the gene's own answer key on a task selected to reward exactly that."**
**Needs a new experiment to be a measurement rather than an argument:** the *similarity-stratified*
diagnostic — bucket held-out (gene, condition) cells by the gene's chemical distance to its nearest
in-train condition, and show kNN dominance decaying (and the model overtaking) in the far bucket.
→ *Fig 5 (to run).*

### 7. Result 4 — remove the crutch and the embedding wins (R-COLD; the positive)
Cold-gene split (whole genes held out): chem-kNN coverage **0.0000**, linear-MF unlearnable, so
chem-NULL is the only applicable bar. 23 orgs / 3 seeds / n=11,761 eligible cold val genes:

| method | NDCG@5 [95% CI] | Spearman [95% CI] | coverage |
|---|---|---|---|
| chem-kNN / linear-MF | — | — | 0% (inapplicable) |
| chem-NULL | 0.2447 [0.2118, 0.2823] | 0.0359 [0.0230, 0.0514] | 100% |
| model | 0.2748 [0.2424, 0.3179] | 0.0735 [0.0523, 0.1034] | 100% |
| Δ | +0.0301 | +0.0376 | |

Report honestly: point estimates disjoint across all 3 seeds; **NDCG@5 CIs overlap**; only the
secondary Spearman CI is disjoint. Note why the T-regime prior didn't apply (cross-*gene* transfer
within a known organism is a much easier ask than cross-*organism* transfer). → *Fig 6.*
**Improvement required before this can be the paper's payoff:** paired/pooled bootstrap on the
per-gene NDCG@5 Δ (cancels shared per-gene variance).

### 8. Result 5 — the corrected benchmark (NEW WORK; the Bernett gold-standard analogue)
Leave-compound-out (scaffold / chemical-class holdout) as the primary split, plus cold-gene; bar =
chem-NULL; scores reported as a fraction of the *denoised* ceiling. This neutralizes the
near-neighbour gift so every method must generalize. Re-run the model, the baselines, and (see
Improvement E) at least one learned-*local* method under it. → *Fig 7 (to run).*

### 9. Discussion
What the community should take away: (i) report the interaction, not the level — most reported
performance on essentiality tasks is `a_g`; (ii) always report a lookup baseline that is handed the
entity's own history, because that is the memorization floor; (iii) always report an empirical noise
ceiling and score as a fraction of it; (iv) random splits over conditions/compounds manufacture near
neighbours — use compound-class holdout; (v) chemical-genetics fitness data sits in the local
memorization regime, so the useful engineering answer for a *known* gene is a lookup, and learned
models earn their keep only in the cold regimes. What would move the needle: inductive fitness-aware
representations targeted at cold genes/compounds; denoised targets; cross-organism ortholog transfer.

### 10. Limitations
Single data source (feba.db); one PLM family; modest ceiling; cold-gene positive not CI-confirmed on
the primary metric; multi-seed bounds as summary bands; the eligibility filter is a design choice
that shapes the target.

---

## 2b. Related work (item H — citations verified 2026-08-25)

**The genre: rigorous negatives that fix a benchmark.** Bernett, Blumenthal & List
(*Brief. Bioinform.* 25(2):bbae076, 2024) is the template the PI supplied. Adjacent and
directly relevant: Dacrema, Cremonesi & Jannach, "Are We Really Making Much Progress? A
Worrying Analysis of Recent Neural Recommendation Approaches" (RecSys '19) — neural
recommenders repeatedly fail to beat well-tuned kNN baselines, and weak baselines
propagate through a literature.

**The critique literature in OUR OWN domain — this is the positioning, and it exists.**
The first draft claimed there was no field of overclaimers to knock down. That was too
pessimistic:
- *"Ridge regression baseline model outperforms deep learning method for cancer genetic
  dependency prediction"* (bioRxiv 2023.11.29.569083) — DeepDEP reaches mean per-gene
  rho 0.137 while a ridge baseline reaches 0.276, and ridge wins on **all** gene sets.
  A direct analogue of our result in the human-cell-line dependency setting.
- *"Deep-learning-based gene perturbation effect prediction does not yet outperform
  simple linear baselines"* (PMID 40759747, 2025) — no model beat a "no change" baseline
  on double-perturbation prediction. The analogue of our chem-NULL finding.
- *"Robust evaluation of deep learning-based representation methods for survival and gene
  essentiality prediction on bulk RNA-seq data"* (*Sci. Rep.* 2024, s41598-024-67023-8) —
  finds gains highly task- and architecture-dependent, and calls for rigorous evaluation
  guidelines.

**Position the paper against these explicitly.** They establish that "simple baseline
beats deep model" is a recurring, publishable finding in essentiality/perturbation
prediction. What none of them supplies, and what this paper adds, is *why*: a
decomposition (`mu + a_g + b_c + I(g,c)`) that says which component the metric is actually
scoring, an empirical noise ceiling that bounds what any method could achieve, and a split
design that separates memorization from generalization. Those three are the contribution;
the negative result itself is now table stakes.

**The memorization regime (why our negative was predictable).** Khandelwal et al., kNN-LM
"Generalization through Memorization" (ICLR 2020); Feldman, "Does Learning Require
Memorization? A Short Tale about a Long Tail" (STOC 2020) — long-tail examples *must* be
memorized for optimal generalization; Grinsztajn, Oyallon & Varoquaux, "Why do tree-based
models still outperform deep learning on typical tabular data?" (NeurIPS 2022) — with
explicit inductive-bias reasons (robustness to uninformative features, data orientation,
irregular functions) that apply directly to a frozen embedding concatenated with a sparse
chemistry vector.

**Data and methods.** Wetmore et al. (2015) for RB-TnSeq and the moderated `t`; Price et
al. (*Nature* 2018) for the Fitness Browser corpus; ProteomeLM (Bitbol Lab) and ESM-C
(EvolutionaryScale) for the representations.

**Still to verify before submission:** exact venue/volume for the two bioRxiv/PubMed items
above, the ESM-C citation form, and the ResMem/EASE/TabR references if those arms are
reported.

---

## 3. Figure list
- **F1** Task + decomposition schematic; RMSE-is-gene-mean evidence.
- **F2** *The money figure.* Method ladder (chem-NULL / model / linear-MF / chem-kNN) with the replicate-ceiling band, warm and cold panels side by side.
- **F3** Five failed axes as small multiples (encoder, objective, capacity, hybrid, org volume).
- **F4** Confidence stratification: all methods and the ceiling rising, gap never closing.
- **F5** *(to run)* Similarity-stratified NDCG@5 — kNN dominance vs chemical distance to nearest in-train condition.
- **F6** Cold-gene: kNN coverage → 0; model > chem-NULL with CIs.
- **F7** *(to run)* Leave-compound-out results.

---

## 4. What has to be improved before this is submittable
Ordered by severity.

**A. BLOCKING — the data footing. [STATUS 2026-08-25: unchanged, but sharper than stated.]**
Every headline number was computed on the 48-organism `feba.db` (sha `627f2097…`) plus
ProteomeLM-L8 embeddings. **Both were lost** in the 2026-07-06 data-root incident. The
documentation pass made the consequence precise: ProteomeLM-L8 is now formally `RETIRED`
(`docs/terms.md`) and its replacement, ESM-C, is **a different representation, not a
restoration**. So this is not "re-pin the data" — the model's central input no longer exists,
and re-running produces numbers that are *not comparable* to anything in the ledger. Either
recover the pinned release, or accept that the paper reports ESM-C results and every historical
number becomes context rather than evidence. The current data root is a *newer 62-organism* release with
ESM-C embeddings. Either restore the pinned 48-org release (Figshare fallbacks noted in bugs.md) or
recompute the entire paper on one pinned, archived release and re-pin `reval_baseline.json`.
Nothing else should start until this is resolved — otherwise every table in the paper is
unreproducible from the code as shipped.

**B. Statistical rigor of the one positive result. [STATUS: open, tracked as `P-ANL-cold-gene-paired-ci`.]** R-COLD's NDCG@5 CIs overlap. Run the
paired/pooled bootstrap on the per-gene Δ. Also replace the "mean of per-seed CIs" multi-seed band
with a proper pooled bootstrap — a reviewer will catch that. If the paired test does not clear, the
cold-gene claim must be stated as directional only, which materially weakens the payoff.

**C. Build the leave-compound-out split. [STATUS: open, confirmed.** `build_ranking_split.py` has `materialize_condition_holdout`, `materialize_cold_gene` and `materialize_cell_holdout` — no compound-holdout materializer exists.] No LOPO materializer exists
(`build_ranking_split.py` has condition-holdout and cold-gene only). This is the paper's
gold-standard-dataset analogue and the highest-value new experiment.

**D. Run the similarity-stratified diagnostic. [STATUS: open, tracked as `P-ANL-similarity-stratified`.]** Cheap — no training. It converts §9's rigging
argument from rhetoric into a measurement, and is probably the most persuasive figure in the paper.

**E. Add learned-*local* baselines to make the negative airtight. [STATUS: open, confirmed.** A grep for EASE / TabR / ResMem / LightGBM / XGBoost / CatBoost / random-forest across `src/` returns nothing — no learned-nonparametric and no tree baseline has ever been run.] Every model we beat was
global/compressing. Add EASE / learned-metric kNN / TabR / canonical ResMem. Note the trap: a
learned-local winner *confirms* the memorization finding rather than refuting it — so this is cheap
insurance either way, and it closes the obvious "you only tried global models" review.

**F. De-meaned and denoised targets. [STATUS: open, confirmed** — no de-meaning or replicate-averaging path exists in `src/`. Now sharpened: `docs/terms.md:interaction_target` records that de-meaning and denoising change what "the target" means, and that the ceiling moves with it.] Model `I(g,c)` directly (subtract fitted `a_g`, `b_c`), and
denoise the target (replicate averaging / t-shrinkage / low-rank). Needed to report "fraction of
achievable ceiling" and to address §9.5 — the 0.66 ceiling is the ceiling of the *single noisy
measurement* task, and denoising raises it.

**G. External validation + a non-ours method. [STATUS: open, confirmed** — MtbTnDB and *A. baumannii* appear only as a comment in `R-AUG_train_org_augmentation.yaml`; nothing is wired in.] Bernett retrained eight published models; we
currently benchmark only our own. Minimum viable answer: (i) run the harness on one external
chemical-genetics dataset (MtbTnDB, A. baumannii) to show the finding is not a feba artifact; and
(ii) add a strong generic tabular baseline — **gradient-boosted trees on (embedding ⊕ chemistry)**,
which the tabular-DL literature says is the natural rival and which we never ran. Its absence is a
real hole.

**H. Related work and citation verification. [STATUS: open, unchanged.]** Needs a proper positioning section: PLM-based
essentiality/fitness prediction, chemical-genetics ML, DepMap gene-mean-baseline critiques, cold-start
recsys, kNN-LM / Feldman / tabular-DL-vs-trees. Several claims in
`LITERATURE_local_vs_global_memorization.md` still need citation verification before they can be cited.

**I. Scope discipline — one paper, not three. [STATUS: enforced.** The reframe proposals are in `archive_docs/`; `paper/` holds only this outline and the narrative.] Keep the ortholog-rewiring reframe
(`PROPOSAL_crossorg_shared_embedding.md`), the MoA sensor, and the dark-genome miner **out**. They
are a second paper. This one is the benchmark/characterization paper; the reframes are its
"future work" paragraph.

**J. Play up the pre-registration. [STATUS: strengthened.]** A dated decision ledger with hypotheses recorded before results
(35 dated decision records, now `archive_docs/decisions/**`), locked gates, and a bit-exact
regression harness is unusual and is exactly the credibility a negative-result paper needs. Two
things strengthen it since the first draft. (i) The promotion delta is now **verified arithmetic**,
not a stipulated number: `0.15 x (0.658 - 0.481) = 0.0266` reproduces the documented gate exactly,
so Methods can show the derivation. (ii) The ledger now carries **8 dated in-place corrections**,
including a self-critique that retired the project's original headline framing and two errors caught
during the documentation pass itself. Present that as a feature, not an embarrassment — a project
that can enumerate what it got wrong is the one whose negative results are worth believing. Still
required: verify hypothesis-before-run for each experiment cited, then say so explicitly.

**K. NEW, first-tier — compute the ceiling on the parity gene set.** `reval_baseline.json` has no
ceiling entry; the only measured values are on n=45,943 while every method is on n=39,356. Until
this is fixed, **Figure 2 cannot be drawn honestly** and no "fraction of achievable" number can be
quoted. Cheap — the harness already computes it, it just needs `eligible_genes` passed and the
result pinned. Tracked as `P-EVL-ceiling-parity`. Do this immediately after A.

**L. NEW — rebuild every table from the pinned artifact, not from prose.** Table 1 was assembled by
hand from three sources and inherited three denominators. Every table in the manuscript should be
generated from `reval_baseline.json` (or its successor) by script, so a number cannot drift from
its artifact. This is also the paper's own methodological claim applied to itself.

**M. NEW, resolved — media-chemistry provenance is documented.** `archive_docs/v4_workbook_audit.md`
(recovered from a stale worktree, where it existed in no commit) verifies that the media
decompositions trace to primary literature rather than being generated. Cite it in Methods; it
pre-empts a reasonable reviewer question about where 191 media compositions came from.

---

## 6b. Defensibility audit — where the claims are unsteady (2026-08-25)

Written adversarially: for each load-bearing claim, what would a hostile reviewer say, and
is there an answer? Items with a fix in hand are marked; items without are plan entries.

### Resolved or in hand

| # | vulnerability | status |
|---|---|---|
| U1 | The cold-gene positive is not CI-significant on `ndcg_at_5` | `paired_hierarchical_bootstrap_ci` implemented; needs running (`P-ANL-cold-gene-paired-ci`) |
| U2 | The ceiling is on a different gene set than the methods | fixed in the pipeline; ceiling now computed on the parity common set |
| U3 | Every number is on a data release that no longer exists | ProteomeLM regenerated from ESM-C 600M + frozen checkpoint; re-pin pending |
| U4 | "No model beats the lookup" covers only the models we tried | GBDT and ResMem added; EASE/TabR still absent |
| U6 | Every design decision was made on the same 23 organisms | 39-organism external panel configured |

### Unresolved — these are the ones that can still sink a review

**U5. The eligibility threshold is an untested researcher degree of freedom.**
`tail_g` and its per-organism threshold decide which genes are scored, which target the
metric measures, AND how high the ceiling sits (high-spread genes replicate better). It has
never been varied. If the method ordering changes materially at a different threshold, the
headline is fragile; if it does not, that is a strong robustness result worth a
supplementary figure. **Right now we do not know which.** -> `P-ANL-eligibility-sensitivity`

**U7. One split seed. All reported variance is model-init variance, not split variance.**
`model_seeds=[0,1,2]` varies only network initialisation; `split_seed=0` throughout. Every
CI and every "disjoint across all 3 seeds" statement therefore describes initialisation
noise on ONE partition of the data. A reviewer will ask what happens on split seeds 1 and 2,
and the honest answer today is that we have not looked. -> `P-EVL-split-seed-variance`

**U8. The gate's strength is unquantified.** chem-kNN's `k` was fixed at 5 and never tuned.
A baseline that is accidentally weak makes the negative less interesting; one that is
accidentally strong makes it overstated. Dacrema et al. (RecSys 2019) is precisely about
under-tuned baselines, so citing that paper while shipping an untuned baseline is an
obvious opening. -> `P-EVL-knn-sensitivity`

**U9. The relevance function is a choice that defines the task.** NDCG's gain uses
`max(0, -fit)`. That decision alone determines what "top stressor" means, and it was never
varied (e.g. a `t`-thresholded binary relevance, or a rank-based gain). -> `P-ANL-relevance-sensitivity`

**U10. RESOLVED AS AN ERROR, and it changes the abstract.** "Population structure carries
approximately zero within-gene signal" is true on Spearman (chem_null is 15.4% of the
model) and FALSE on the primary metric (79.5%). See the correction in `docs/memory.md`.
The paper must state this per-metric, and should treat the divergence as a finding: the two
metrics disagree about how much of the task the population profile solves.

**U11. The linear-MF argument is weaker than the project has been stating.** The claim is
"a fitness-aware representation does not help, so fitness-blindness is not the bottleneck."
But linear-MF's per-gene latent is **transductive** — free parameters fit to that gene's own
training rows. It is not an inductive fitness-aware representation and cannot be computed
for an unseen gene. So the evidence rules out *transductive* fitness-aware representations
on the warm split; it says nothing about an inductive one, which is exactly what a fine-tuned
encoder would provide. Weaken the claim in the text.

**U12. Multi-seed CIs are the mean of per-seed CIs, not a pooled bootstrap.** Documented in
`docs/terms.md` but unfixed. A summary band presented as a confidence interval is the kind
of thing a statistically-minded reviewer will catch immediately. -> `P-FIX-pooled-ci`

**U13. Regenerated ProteomeLM may not reproduce the historical numbers.** ProteomeLM is
proteome-CONTEXTUAL, and the regeneration runs on the 62-organism release whose proteome
composition may differ from the 48-organism one. Any movement must be attributed, not
absorbed. If the numbers move materially, the honest paper reports the NEW numbers and
describes the old ones as a prior release.

### The meta-lesson for the paper's own methods section

U10 and the 2026-06-24 correction are the same mistake twice: a finding established under
one primary metric, never re-derived when the primary metric changed. Every durable finding
predating 2026-06-24 needs the same audit. This is worth stating in the paper — it is
concrete evidence for the paper's own thesis that *what you measure decides what you
conclude*, and it happened to us, in writing, twice.

---

## 0b. Detailed comparison with Bernett et al. — what transfers and what does not

Written after implementing items A-H, so it reflects what we can actually claim.

### Five things that genuinely transfer

**S1. The thesis shape.** Both papers say: *the reported result is a property of the
benchmark, not of the method.* That is the sentence the abstract turns on in both cases.

**S2. A named shortcut.** They identify what the models actually key on — node degree and
sequence similarity. We identify ours: the target gene's own measured history, plus
chemically near neighbours that a random condition split manufactures. Naming the shortcut
is what turns "the numbers are wrong" into "here is the mechanism."

**S3. A corrected split as the deliverable.** They redundancy-filter by sequence similarity
and partition so no protein crosses the split. We hold out whole stressor compounds
globally and whole genes. In both cases the artefact is the paper's product, not just the
argument.

**S4. Re-running methods unchanged on the corrected benchmark.** Neither paper is entitled
to retune anything after fixing the split.

**S5. The closing move.** Both end by scoping what remains unsolved rather than proposing a
better model.

### Seven divergences — and what each forces us to do differently

**D1. They knock down eight published models; our models are our own.**
This is the structural difference and it bounds the claim. They can write "the field's
methods collapse"; the most we can write is "no method in the families we tested wins."
*Mitigations now in place:* GBDT and ResMem widen it beyond our architecture; the external
39-organism panel guards against panel-specific tuning; and §2b shows a critique literature
already exists in this domain (ridge beating DeepDEP; linear baselines beating perturbation
models), so we position as **converging evidence**, not a lone negative. **Do not overclaim
field-level generality.**

**D2. Their failure is catastrophic; ours is graded — and we have a positive they don't.**
Their models fall to chance. Ours does not: it sits at roughly 62% of the span between the
population floor and the lookup, and on held-out *genes* it beats the population baseline.
So we cannot borrow their rhetorical frame ("the task is unsolved"). Ours is: *the reported
margin is mostly memorization; the generalizable component is real but small.* Harder to
sell, more accurate, and the cold-gene positive gives the paper an affirmative claim that
Bernett's has no counterpart for.

**D3. We have an empirical ceiling; they cannot have one.** PPI labels are binary and
nominally noiseless, so "best achievable" is undefined for them. Our replicate ceiling
converts "0.43 is low" into "0.43 out of an achievable 0.66." This is our strongest
methodological asset and it has no analogue in the template. **Lead with it.**

**D4. Their leakage is real leakage; ours is not — and conflating them would be fatal.**
In their setting the same protein appears on both sides of the split: that is a defect. In
ours, no row is duplicated and no label is copied. chem-kNN reads *other* labels of the
target gene, which is a legitimate method using legitimate information. **chem-kNN is not
cheating.** The claim must be that the *benchmark* is constructed so that memorization
suffices, which makes it a poor instrument for the scientific question — not that the
baseline is illegitimate. Getting this wrong invites the reviewer response "you are calling
a fair baseline a cheat," and that response would be correct.

**D5. They demonstrate that fixing the benchmark changes the answer; we have not yet.**
They retrained on the corrected split and showed collapse. Our leave-compound-out split is
built and wired but **has not been run**. Until it has, we are proposing a fix without
demonstrating its effect — which is exactly the weakest form of this paper. This is the
single highest-value remaining experiment.

**D6. Scope.** They audit a field's benchmarks; we audit one dataset on one platform. The
generality claim must be narrower, and the external-organism panel should be described as
controlling for benchmark-design overfitting, **not** for platform effects.

**D7. Our self-critique is part of the contribution.** We have a dated ledger of eight
in-place corrections, including two — the 2026-06-24 CI read and the U10 population-signal
claim — where our own conclusions did not survive a change of primary metric. Bernett has
no equivalent. Presented properly this is not an embarrassment but the paper's thesis
demonstrated on itself: *what you measure decides what you conclude.*

### The one-sentence positioning

> Bernett et al. showed that a field's benchmark was leaking and that fixing it erased the
> results. We show something adjacent but distinct: a benchmark that leaks nothing can still
> measure the wrong thing, because it can be built so that memorizing an entity's own
> history is sufficient — and we supply the decomposition, the noise ceiling and the splits
> needed to tell memorization and generalization apart.

---

## 5. Venue
Bernett landed in *Briefings in Bioinformatics*. Comparable homes: *Briefings in Bioinformatics*,
*Bioinformatics* (Analysis/Application Note), *NAR Genomics & Bioinformatics*, *PLOS Computational
Biology*, *mSystems* (microbiology angle). Benchmark/ML framing also fits MLCB or an ICLR/NeurIPS
workshop track. Recommendation: **BiB or NAR GaB.**

## 6. Honest assessment of publishability
**As a benchmark + characterization paper with the corrected split included: yes.**
**As "we tried N models and they all lost to a kNN": no.**
The distance between those two is items C, D, E, F, G and now K — mostly cheap experiments, roughly
4–8 weeks of work, after A is resolved. B determines whether the paper ends on a positive note or a
purely diagnostic one.

**Revised ordering after the documentation pass:** A (data footing) -> K (ceiling parity) ->
B (paired CI) -> D (similarity-stratified, no training) -> C (leave-compound-out) -> E/F/G ->
L (generate tables from artifacts) -> H. K moved up because it is cheap and it gates the paper's
central figure; D stays early because it needs no training and carries the most argumentative
weight per hour spent.
