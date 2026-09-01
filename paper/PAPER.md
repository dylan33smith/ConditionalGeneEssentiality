# PAPER.md — the central working document

**What this is.** The framing, hypothesis, structure, findings and claim boundaries for the
paper. Not the manuscript. Everything we write draws from here.
`PAPER_OUTLINE_2026-08-25.md` is the companion working file: section-by-section outline,
the defensibility audit, the Bernett comparison, and verified related work.

**Last updated:** 2026-08-25

---

## 1. The hypothesis

> **Conditional gene essentiality is a different prediction problem from gene essentiality,
> not an extension of it — and the standard way to pose it measures the wrong thing.** The
> quantity of interest is the gene×condition interaction. It is abundant (≈48% of variance),
> it is largely not measurement noise, and it is almost entirely unpredicted by frozen
> protein embeddings plus condition chemistry: the apparent performance of such models comes
> from main effects and from per-gene memorization that a random condition split silently
> permits.

Three parts, and the paper must deliver all three:
- **A positive claim about the data.** The interaction is large and real.
- **A negative claim about methods.** Our model families barely capture it.
- **A mechanism for the discrepancy.** Main effects plus memorization, both invited by the
  standard task formulation.

---

## 2. The opening move

> Essentiality prediction reports high accuracy. Conditional essentiality appears to be the
> same problem with an extra argument. We show it is not: the two differ in what fraction of
> variance is predictable from sequence, in what a random split rewards, and in what the
> metric actually scores.

---

## 3. The central framework — foundational, not corrective

Bernett et al. wrote a **corrective** paper: a field went wrong, here is the fix, here are
eight published models collapsing on a repaired benchmark.

We cannot write that paper: nobody has published methods for this task to knock down. So we
write the **foundational** one instead — *this task is about to become popular, and here is
the benchmark, the baselines, the ceiling, and the three ways it will go wrong if posed
naively.* Benchmark-and-baselines papers are an established, well-cited genre and require no
prior victims.

The absence of prior work is therefore an **asset, not a deficit**: we are preventing a
field from starting wrong rather than correcting one that already has. Bernett's *structure*
still transfers completely; only the rhetorical stance changes.

**We end by opening a problem, not closing one.** The interaction is abundant and largely
unpredicted, which makes this an open problem rather than a solved or hopeless one — and we
supply the instrument for measuring progress on it.

---

## 4. Introduction — the contrast with non-conditional essentiality

This is the paper's spine.

Classical gene essentiality prediction — *"is gene g essential?"* — is a mature field with
reported AUROCs in the 0.85–0.95 range from sequence, network and comparative-genomics
features. Conditional essentiality — *"is gene g essential in condition c?"* — looks like a
modest extension: add an argument.

It is not, and the decomposition shows why. Writing

```
fit(g, c) = mu + a_g + b_c + I(g, c) + noise
```

`a_g` **is the non-conditional quantity**, measurable in the very same data. So the two
problems can be compared with everything else held fixed: same features, same model, same
protocol, two targets.

- **Target 1 — `a_g`:** how detrimental knocking out gene g is on average. The
  non-conditional problem. Expected to be comparatively easy.
- **Target 2 — `I(g,c)`:** the interaction. The actual conditional problem.

**That twin-target experiment is the paper's centerpiece** (`P-TRN-main-effect-vs-interaction`).
It turns an abstract argument into a two-bar figure a reviewer cannot misread. It does not
exist yet and should be run early.

### The caveat that must be stated precisely

RB-TnSeq (Random Barcode Transposon-site Sequencing) measures fitness by disrupting genes
with barcoded transposon insertions and tracking which mutants drop out of a growing pool.
A gene that is required under **all** conditions produces no viable insertion mutants, so it
never enters the library and has no data at all.

Therefore `a_g` is **"average fitness cost among assayable genes"**, not textbook
essentiality: the unconditionally-essential genes are structurally absent from the dataset.
Two consequences, both worth stating rather than hiding:
1. The contrast is "average fitness cost vs interaction", not "essentiality vs interaction".
   The contrast still holds; the label must be honest.
2. It is a selection effect worth naming — the most critical genes in each genome are
   invisible to this assay, which is itself a limitation of the whole conditional-essentiality
   literature built on transposon data.

### Why the task is about to become popular

Two things are converging, which is why a benchmark now is worth more than a benchmark later:

- **Protein language models** (ESM, ESM-C, ProteomeLM, ProtT5) — models trained on large
  protein-sequence corpora that emit a fixed-length embedding per protein. They have become
  the default way to featurize a gene.
- **Perturbation atlases** — large datasets measuring what happens when you perturb genes
  across many contexts: DepMap (CRISPR knockouts across ~1,000 cancer cell lines),
  Perturb-seq (CRISPR crossed with single-cell RNA-seq), Cell Painting, and the RB-TnSeq
  Fitness Browser corpus used here.

Feeding embeddings plus context features into a model to predict perturbation outcomes is
the obvious next move, and it is already happening in the human-cell-line setting. Our
verified related work shows that setting is *already* producing the same failure mode: a
ridge baseline beats DeepDEP on cancer dependency prediction, and linear baselines beat
deep models on perturbation-effect prediction. We supply the decomposition and the splits
those papers lacked.

---

## 5. The variance decomposition — the finding that drives the framing

### The equation

`fit(g, c) = mu + a_g + b_c + I(g, c) + noise`

- `mu` — the grand mean fitness across everything.
- `a_g` — **gene main effect.** How costly is losing gene g, averaged over conditions.
  Non-conditional. This is the classical quantity.
- `b_c` — **condition main effect.** How harsh is condition c, averaged over genes.
- `I(g,c)` — **the interaction.** Is gene g *specifically* needed for condition c, beyond
  what g's general importance and c's general harshness already explain. **This is the only
  term that is genuinely about conditional essentiality.**
- `noise` — measurement error.

### How the numbers were obtained

Keio + Caulobacter + MR1; 1,957,960 measurements; 10,883 genes; 317 conditions.

1. Fit `mu`, `a_g`, `b_c` by alternating means over three iterations (the design is
   unbalanced — genes are not measured at identical condition sets — so a single pass leaves
   main-effect mass in the residual), under a sum-to-zero constraint on `a` and `b` so the
   parameters are identified.
2. Residual `= fit − mu − a_g − b_c`. This is `I(g,c) + noise`.
3. Estimate noise separately from replicates: for 655,846 cells measured twice,
   `var(rep_A − rep_B) = 2·sigma²`, so `sigma² = var(diff)/2 = 0.0654`.
4. `var(I) = var(residual) − sigma²`.

### The result

| component | variance | share |
|---|---|---|
| `a_g` gene main effect | 0.2714 | 40.6% |
| `b_c` condition main effect | 0.0115 | **1.7%** |
| **`I(g,c)` true interaction** | **0.3201** | **47.9%** |
| noise | 0.0654 | 9.8% |
| total | 0.6683 | 100.0% |

The components sum to 100.0%, i.e. the cross-covariances are ~0, which is what the
alternating-means fit is designed to achieve and a useful internal check.

### Why this inverts the project's standing story

For months the project's narrative was: *the target is intrinsically noisy, the achievable
ceiling is modest, the signal is weak — so a modest result is the honest ceiling of what is
possible.* That framing made the negative feel like a property of biology.

The decomposition says otherwise:
- The interaction is the **largest single component**, 1.18× the gene main effect.
- Noise is only **9.8%** — five times smaller than the interaction.
- **83%** of what survives main-effect removal is real signal.

So the data is **not** signal-poor. The honest statement becomes: *there is a great deal of
real gene×condition signal in this dataset, and our models capture very little of it.* That
converts a shrug into an indictment — a much stronger and more useful paper.

### Conservative direction of the estimate

The main effects were fit on all rows, letting `a_g` absorb the maximum possible variance.
That makes `var(a_g)` an upper bound and `var(I)` a **lower** bound — the conservative
direction for our claim. Counter-caveat: if replicates share batch effects they are not
independent, `sigma²` is underestimated, and `var(I)` is correspondingly overestimated.
Both should be stated. Three organisms only; the full-panel version is a plan item.

---

## 6. The decomposition versus the replicate ceiling — how they relate

These two numbers appear to contradict each other and the resolution is itself a finding.

|  | variance decomposition | replicate ceiling |
|---|---|---|
| what it measures | share of `var(fit)` that is interaction, across ALL cells | how well replicate A's *ordering* of a gene's conditions predicts replicate B's |
| unit | one number over the whole dataset | a per-gene rank statistic, averaged over eligible val genes |
| says | interaction:noise ≈ 4.9 : 1 — lots of signal | `ndcg_at_5` ≈ 0.66 — modest reproducibility |

They are not in conflict because they aggregate differently. Global variance is dominated by
whichever cells have the biggest effects. Per-gene ranking asks a much harder question of
*every* eligible gene, including the many whose conditions are all nearly flat and whose
ordering is therefore mostly noise.

**The reconciliation hypothesis: interaction variance is CONCENTRATED.** A minority of
gene×condition cells carry large effects; most are near-flat. Aggregate variance reflects the
few strong hits, while the median gene's condition ordering is noise-limited.

**If confirmed, this reframes the task.** Not "predict a weak signal everywhere" but **"find
sparse strong interactions"** — a different and more tractable problem statement, with
direct consequences for objectives (see §11: hurdle models, imbalance-aware ranking).

### The experiment that tests it — `P-ANL-interaction-concentration`

Run before anything else; it is cheap and it decides the framing.

1. **Cell-level concentration.** Compute `I(g,c)` for every cell. Report the share of total
   interaction variance held by the top 1%, 5% and 10% of cells by `|I|`, plus a Gini
   coefficient. Uniform spread → Gini near 0; concentrated → Gini near 1.
2. **Per-gene signal-to-noise.** For each gene, `SNR_g = (var_c(residual) − sigma²) / sigma²`.
   Plot the distribution. The concentration hypothesis predicts a long right tail with a mass
   of genes near zero.
3. **The link that closes the argument.** For each gene compute BOTH `SNR_g` and its
   replicate-reproducibility (per-gene replicate Spearman, the per-gene ingredient of the
   ceiling). Plot one against the other. They should be strongly monotone. That plot
   *quantitatively explains* the ceiling — it shows the modest ceiling is the average over a
   population of genes most of which have little interaction signal to reproduce.
4. **Stratified ceiling and stratified performance.** Bin genes by `SNR_g`; report the
   ceiling and every method within each bin. Prediction: in the top-SNR bin the ceiling is
   high and the methods have real headroom; in the bottom bin everything collapses toward
   chance. If so, the honest headline becomes performance on the high-SNR genes, and the
   rest is correctly described as unmeasurable rather than unpredicted.

This single experiment converts the paper's weakest number (a modest ceiling that invites
"so the task is hopeless") into its most informative one.

---

## 7. The three scenarios — the Bernett device, with evidence for all three

> A chemistry-similarity lookup outperforms every learned model on within-organism
> conditional essentiality ranking. We systematically examine three explanations.

**S1 — The models are underpowered.** The first thing anyone assumes: wrong encoder, wrong
objective, insufficient capacity.
→ **REJECTED.** Six chemistry encoders (multihot vs Morgan/RDKit/MACCS, 6 arms × 3 seeds ×
23 organisms); six loss functions plus top-5-truncated NDCG-matched variants; a
13k-parameter bilinear model statistically indistinguishable from a 2.5M-parameter nonlinear
one; three orthogonal learned hybrids converging on zero; doubling the training organisms
*hurts*. Gradient-boosted trees and ResMem now added to close the family gap.

**S2 — The labels are too noisy for any model to do better.**
→ **REJECTED.** The lookup beats the model in *every* confidence quartile and at every
per-cell |t| threshold; noise depresses all methods and the ceiling together but never closes
the gap. Now also quantitatively: noise is 9.8% of variance against an interaction of 47.9%.

**S3 — The benchmark rewards memorization.** The lookup is handed the target gene's own
labels at inference, and a random condition split leaves nearly every held-out condition a
chemically near neighbour in training.
→ **SUPPORTED, by two independent removals.** Remove the own-history (cold-gene split) and
the learned model beats the population baseline on genes it has never seen. Remove the near
neighbour (leave-compound-out) and the lookup's margin over the model collapses by **59%**
(0.0483 → 0.0199).

This is a stronger three-way than the template's, because two of the three are rejections we
can *show* rather than assumptions we dismiss.

---

## 8. What we can claim

1. The variance decomposition with the noise term empirically separated. **No equivalent
   exists for bacterial chemical genetics.**
2. An empirical replicate ceiling for the interaction task.
3. A per-gene chemistry lookup is a strong baseline that must be reported — and a random
   floor of ≈0.16 `ndcg_at_5` that must be reported beside it.
4. Random condition splits manufacture near neighbours; compound holdout removes them and
   the lookup's advantage collapses.
5. Frozen protein-language-model embeddings carry *some* transferable gene-specific
   conditional signal (cold-gene; directional, not yet CI-confirmed).
6. The negative is not explained by encoder, objective, capacity, hybridization, training
   volume, or label noise.
7. Regenerated embeddings *improved* the model (+0.0140 `ndcg_at_5`) with every
   embedding-independent quantity bit-exact — so the negative is not an artifact of a weak
   model instance. Worth stating: the correction ran in the direction that *disfavours* our
   own negative result.

## 9. What we cannot claim

1. **"Deep learning fails at conditional essentiality."** Only that these families fail on
   this data.
2. **"The task is unsolvable."** The decomposition actively contradicts it.
3. **"Embeddings are useless."** Cold-gene refutes it.
4. Anything about a field's published methods — there are none to test.
5. CI-significance on the cold-gene positive. Not yet.
6. That the leave-compound-out collapse is purely near-neighbour removal. `chem_null` rises
   there (0.3426 → 0.4284), so part may be split difficulty. **Blocks publication of that
   result until diagnosed.**
7. Generality beyond RB-TnSeq / Fitness Browser.
8. **"Population structure carries ≈0 within-gene signal."** Spearman-only; false on the
   primary metric, where `chem_null` reaches 79.5% of the model's `ndcg_at_5`.

---

## 10. The gold-standard dataset

Bernett's artifact: redundancy-filter by sequence similarity, partition so no protein crosses
the split, balance node degree. Ours is the same idea applied to a richer corpus — 62
organisms, ~9,400 experiments, 33.4M measurements, 370 stressor compounds with canonical
chemistry and fingerprints, replicate structure for a noise ceiling, and media compositions
traced to primary literature.

### Four generalization axes

| split | holds out | tests | status |
|---|---|---|---|
| `condition_holdout` | conditions at random | the naive setting — shipped as the **negative control**, not omitted | built |
| `leave_compound_out` | whole stressor compounds | chemical generalization | built |
| `cold_gene` | whole genes | gene generalization | built |
| `cold_organism` | whole organisms | phylogenetic generalization | **missing under this harness** |

Shipping the naive split *as a control* and showing the same method score differently across
all four **is** the contribution.

### Similarity control — the part we had missed

A holdout is only honest if the held-out item has no near-duplicate in training. This is
exactly Bernett's sequence-similarity redundancy removal, and our splits currently do not do
it. Three axes, in increasing order of strength:

**(a) Gene similarity — homology-controlled `cold_gene`.**
Currently `cold_gene` holds out *random* genes. If a paralog or close ortholog of a held-out
gene remains in training, the embedding transfers trivially and "generalization to unseen
genes" overstates itself. **Fix:** cluster genes by sequence identity (MMseqs2/CD-HIT, or
`feba.db`'s `Ortholog` table, 2.84M rows) and hold out whole clusters. **Also report
stratified by max identity to any training gene** — that curve is more informative than any
single number and is the direct analogue of Bernett's headline figure.

**(b) Chemical-structure similarity — scaffold-controlled `leave_compound_out`.**
Currently exact compounds are held out, so a held-out compound may have a near-identical
analogue in training (two tetracyclines, two aminoglycosides). **Fix:** cluster by Morgan
fingerprint Tanimoto or Murcko scaffold and hold out clusters. Fingerprints already exist
(`canonical_fingerprints.npz`, ~77% SMILES coverage); the materializer already accepts a
`compound_groups` argument for exactly this.

**(c) Fitness-profile similarity — the strongest and most novel control.**
Two compounds can be chemically unrelated yet produce nearly identical genome-wide fitness
profiles because they hit the same pathway. Holding out a compound whose *phenotypic* twin
remains in training still leaks, and no structural filter catches it.

**Design caution.** Fitness profiles are built from the labels, so using them to *construct*
the split means the split was chosen using label information. Two options:
- *Conservative and preferred:* do not split on it — **stratify on it.** After splitting,
  compute for each held-out condition its maximum fitness-profile correlation to any
  **training** condition, and report performance stratified by that. This generalizes the
  similarity-stratified diagnostic and is leakage-free.
- *Aggressive:* cluster **training** conditions by profile and hold out whole clusters,
  disclosing that split construction used training-label information. Makes the benchmark
  harder, which is the conservative direction, but must be disclosed.

Recommendation: ship the stratified version as the headline and the aggressive split as a
supplementary robustness check.

### Everything else the release must contain

The eligibility rule; denominator parity; the four methods plus `random` and `constant`
floors plus the replicate ceiling; the mandatory per-organism breakdown; hierarchical
org→gene bootstrap CIs; the pinned data manifest and code.

---

## 11. Algorithms worth trying

If concentration is confirmed, the model class should change — every model tried so far
assumes a dense signal and a symmetric loss.

- **Two-stage / hurdle models.** Classify "is this cell a hit?", then regress magnitude
  among hits. Directly matches a sparse-strong-interaction structure. Nothing run so far
  does this.
- **Imbalance-aware ranking objectives.** If hits are a few percent of cells, pointwise
  Huber over all cells is the wrong objective — and this may be the real reason the ranking
  losses underperformed, rather than "ranking losses do not help".
- **Explicit interaction models on the de-meaned target.** Factorization machines, or a
  bilinear head trained on `I(g,c)` directly, so no capacity is spent re-learning `a_g`.
  The de-meaning machinery now exists.
- **EASE and TabR.** The learned-local family we cite but have not run. Cheap, and closes
  S1 completely.
- **Chemistry-aware condition kernels.** Tanimoto-kernel ridge regression or a GP — an
  interpretable middle ground between a fixed kNN and an opaque MLP, and a natural fit for
  a benchmark paper.
- **Deprioritized:** fine-tuning the encoder, until the twin-target experiment says whether
  sequence carries interaction information at all.

---

## 12. Title and abstract skeleton

**Title direction:** *Conditional gene essentiality is not gene essentiality: a
decomposition, a benchmark, and three ways to get it wrong.*

**Abstract skeleton:**
1. Essentiality prediction is mature and reports high accuracy.
2. Conditional essentiality looks like an extension of it.
3. We decompose the measurement and find the gene×condition interaction is the largest
   variance component and mostly not measurement noise.
4. Yet no learned model we test beats a per-gene chemistry lookup.
5. We examine three explanations — underpowered models, label noise, benchmark design — and
   only the third survives.
6. We release four generalization splits with similarity control, a random floor, and a
   replicate ceiling.
7. The interaction is abundant and largely unpredicted: an open problem, not a solved or a
   hopeless one.

---

## 13. Pointers

- Detailed section outline, defensibility audit, Bernett comparison, verified related work:
  `paper/PAPER_OUTLINE_2026-08-25.md`
- The scientific narrative and its self-corrections: `paper/SCIENTIFIC_SYNTHESIS.md`
- Every result and correction, dated: `docs/memory.md`
- Definitions and what changes each number's meaning: `docs/terms.md`
- The work queue: `docs/plan.md`
