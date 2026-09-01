# Five research directions — adversarial hypothesis dialogue (2026-06-29)

**Method.** Two-agent Socratic process. *Prof. Aria Vance* (cross-disciplinary first-principles
hypothesis generator, **blind** to all project code/docs, sees only `data/raw/feba.db` +
canonical parquets) generated 12 directions. *Dr. Elena Reyes* (Socratic critic, **full** access
to prior work) opened with the nearest-neighbor-wall briefing, flagged leakage landmines, sorted
the 12 into KILL/WOUNDED/LIVE, and posed a decisive test. Vance refined under fire with fresh SQL.

**The decisive test (Reyes → Vance).** *"Name the one idea whose target a chemistry-similarity
lookup STRUCTURALLY CANNOT compute, and prove it — state exactly what near-neighbor the lookup
retrieves and why that neighbor does not already contain the answer."* An idea escapes the
memorization wall **iff its target is not a smooth function of `{fit(g, c′) : c′ chemically near c,
same organism}`** — because that set is exactly what the chem-kNN retrieves.

---

## The slate (ranked by paper-worthiness × feasibility)

### 1. Conserved vs. rewired conditional essentiality across orthologs — **THE PROJECT**
The cleanest wall-escape. Target = a **cross-organism comparison** of a gene's conditional
knockout response. The lookup retrieves an *organism-A-local* neighbor; the answer lives in
*organism B*; cross-organism within-gene transfer is ≈0.045 Spearman (≈random) → the lookup
**cannot** compute it. The cross-org≈0 result flips from liability to **signal generator**: against
a null of zero conserved within-gene structure, every ortholog whose response *does* conserve above
the replicate-noise floor is a discovery.
- **Measurable / baseline.** Per ortholog pair (matched compound), conserved-response score =
  matched-condition fit-vector correlation, gated above the replicate-noise floor (~0.39 Spearman).
  Then predict *which* orthologs conserve. **Must beat:** sequence-identity (bitscore ratio) alone.
- **Leakage guard.** Orthology from the **raw `Ortholog` table only** (never `ConservedCofit`);
  correlations **train-only**; replicate-noise gate on every conservation call.
- **N (verified).** Caulo↔MR1 (distant) = 328 orthologs at ratio>0.5 with matched conditions;
  same-species pairs (4× *P. fluorescens*, DvH/Miya, 2× *Methanococcus*) = thousands; 48 stress
  compounds in ≥10 organisms.
- **Good / bad.** Good: a replicate-gated population of sequence-conserved / phenotype-diverged
  orthologs, enriched for regulators, with conservation predictable beyond sequence identity.
  Bad: divergence fully tracks sequence divergence → clean publishable null on genotype-phenotype
  rigidity.

### 2. Fitness fingerprint as a mechanism sensor for structurally-novel compounds
Fitness as **input**, MoA class as **target**. Escape is real **only on the structurally-distant
slice**: where a compound's nearest *chemical* neighbor has a *different* MoA, structure-kNN
retrieves the wrong-mechanism neighbor and the genome-wide fitness fingerprint carries information
structure does not.
- **Measurable / baseline.** Incremental AUROC of the fitness-fingerprint MoA classifier **over a
  structure-only Morgan-kNN**, on the structurally-distant held-out compounds, under
  **leave-compound-AND-organism-out**. MoA labels curated independently of this DB.
- **Leakage guard.** Joint leave-compound-and-organism-out; report post-split N *before* claiming.
- **Risk.** N collapses after the joint split — the live risk is N, not the concept.
- **Good / bad.** Good: fitness adds real AUROC over structure on distant compounds. Bad: N too
  small / no increment → the negative is the paper.

### 3. The local-memorization characterization (weaponize the negative result) — methods backbone
Does not escape the wall; **characterizes** it. A rigorous statement that chemical-genetic fitness
sits in the named local-memorization regime (kNN-LM, Feldman, tabular-DL-vs-trees, recsys), with
the replicate-noise ceiling pinned and a proof that population/cross-gene structure carries ≈0
within-gene signal. Applicability-domain-aware (similarity-stratified) evaluation as the deliverable.
- **Measurable / baseline.** Replicate concordance = ceiling; signal/(signal+noise) per t-bin and
  condition-class; global-model plateau vs local lookup vs ceiling; manufactured-near-neighbor
  inflation of random splits shown explicitly.
- **Good / bad.** Good: a transferable characterization of *when* local lookups beat global models
  in biological fitness data. Bad: too dataset-specific → lower impact, still sound.

### 4. Genome → substrate-utilization, cross-organism — **ON PROBATION**
Predict whether a *never-assayed* organism grows on a C/N source (155 C / 79 N) from genome content
— a held-out-genome target the lookup cannot reach.
- **Blocker.** Must validate against an **independently measured growth phenotype**, never KEGG
  derived from the same data (circular). *Confirm such an external growth table exists first* — this
  is why it ranks 4th.
- **Baseline.** Ortholog presence/absence of canonical catabolic genes.

### 5. De-leaked condition-resolved genetic-interaction networks (conditional epistasis)
Which gene pairs become co-essential **only** under specific stress (condition-resolved modules)?
Biologically rich, but advances **only** with a strict **train-only, per-condition,
replicate-noise-gated** co-essentiality rebuild — the naive baseline (`Cofit`) *is* the leak
(cofit-only scores 0.73–0.75, above the replicate ceiling). Corroborated as a viable escape by the
earlier reframe search (de-leaked co-essentiality → functional linkage).

---

## Demoted / retracted this round
- **Dark-genome-by-condition** — RETRACTED (premise falsified: unannotated genes are *less*
  surprising than annotated; leaky co-fitness). Matches the prior R-DARK spike result.
- **Dose-response Hill-curve mechanism discriminator** — DEMOTED to boutique validation: only 7
  cells reach ≥5 concentrations and dose-count conflates with replicate-count → Hill params not
  reproducibly identifiable genome-wide. SynE NaCl 10-point is the one clean case study.
- **Specialist↔generalist breadth law** — needs an *independent* target (e.g. HGT status) and
  assay-coverage correction to be more than a histogram.

**Recommendation:** pursue **#1 (ortholog conditional-response divergence)** as the headline — it is
the only direction that cleanly passes the lookup-cannot-compute test and is genuinely new (not in
the prior reframe slate); **#2** as the strong translational second; **#3** as the methods backbone
that frames either.
