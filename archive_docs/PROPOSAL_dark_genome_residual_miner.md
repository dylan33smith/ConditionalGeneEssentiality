# Proposal: Dark-genome residual miner

**Status:** SPIKED (Steps 1–3, 2026-06-25) — **signal is real, but the dark-genome *enrichment*
premise is NOT supported.** See "Spike result" below. **Type:** reframe — a *different question*, not a
better ranker.

## Spike result (Steps 1–3, Keio/Caulo/MR1) — 2026-06-25

Code: `src/experiments/rdark/spike.py` (v1) + `spike_repro.py` (v2). v1's naive top "surprises"
were single-measurement (`n_rep=1`) artifacts (the σ/low-replication risk, as predicted), and its
gene-permutation null tested *predictability* not *surprise-beyond-noise*. **v2 fixed both**:
restrict to replicated cells (`n_rep≥2`), require the deviation to reproduce across a gene's two
replicate halves (concordant sign, both |z|>3 vs `σ(condition)`), and use a within-condition
replicate-shuffle null. Inline kNN verified against the canonical harness to ~1e-16.

| org | replicated cells (orphan) | CONFIRMED orphan surprises real / null | emp_p | confirmed RATE orphan vs annotated | label_p |
|---|---|---|---|---|---|
| Keio | 204,606 (35,748) | **794 / 75.9** | 0.0020 | 0.0222 vs 0.0595 | 1.0000 |
| Caulo | 284,832 (63,984) | **2235 / 293** | 0.0020 | 0.0349 vs 0.0576 | 1.0000 |
| MR1 | 166,408 (65,296) | **2404 / 234** | 0.0020 | 0.0368 vs 0.1020 | 1.0000 |

**Two findings:**
1. **Reproducible, chemically-unexpected conditional-essentiality IS real** — confirmed surprises run
   ~10× the replicate-shuffle null in every org (emp_p at the 500-perm floor). The residual carries
   genuine signal, not measurement noise. (Kills the "it's all artifacts" worry.)
2. **The dark-genome ENRICHMENT premise FAILS** — orphans are *less* surprising than annotated genes
   in all three orgs (label_p = 1.0, i.e. significantly quieter, not richer). "Unknown genes harbor
   the surprising biology" is not supported; the phenomenon is broader and stronger in annotated genes.

**Implication / re-scope:** a "confirmed chemically-unexpected reproducible phenotype" is close to a
*specific phenotype*, which feba.db's `SpecificPhenotype` (38,525 rows) already catalogs — so the
novelty over existing annotation is thin. The headline "dark genome is special" is dead. The only
surviving angle is narrow: the orphan subset of *chemically-unexpected* (residual-defined, not just
large-|fit|) reproducible phenotypes, **gated on whether cross-organism ortholog concordance (Step 4)
yields a corroborated subset** the existing annotation misses. Borderline; proceed to Step 4 only if
that nugget is wanted — do not headline dark-genome enrichment.

---
 **Companion docs:** [REFRAME_CANDIDATES.md](REFRAME_CANDIDATES.md) (ranked #1),
[LITERATURE_local_vs_global_memorization.md](LITERATURE_local_vs_global_memorization.md) (why the
ranking objective is memorization-capped). **Effort:** spike = days; no model training, no GPU.

## The idea (the flip)

We are memorization-capped: chem-kNN beats every learned model at *predicting* a gene's fitness.
So **stop predicting fitness**. Instead, use chem-kNN as a model of *what is chemically predictable*
and mine its **residual** — the part of a gene's conditional essentiality the lookup **cannot**
explain — restricted to **unannotated ("dark genome") genes**. Each large residual is a concrete
hypothesis: *unknown gene → the specific condition that unexpectedly makes it essential.*

**Why it escapes the wall (by construction):** the memorization wall is a statement about
prediction accuracy. Here chem-kNN is the *null model* and its error *is* the target — "the lookup
beats learning" is structurally inapplicable. This is the one reframe the wall cannot recapture, and
it turns the project's central negative result into the contribution.

This is a **discovery task**, not a ranking task — NDCG@5 / the chem-kNN promotion gate do **not**
apply. The deliverable is a ranked, cross-organism-corroborated nomination list of dark genes, each
with a stressing condition.

## Verified data grounding (confirmed against feba.db / parquet, 2026-06-25)

| Piece | Source | Notes |
|---|---|---|
| Observed fitness, gene×condition | `data/derived/canonical/v0/fitness_experiment_long.parquet`: `fit`, `gene_key` (=`orgId:locusId`), `expName`, `expDesc`/`media`/`temperature`→`condition_key` | replicates = multiple `expName` per `condition_key`; loaders already build this matrix |
| Per-cell noise (option A) | parquet `t` / `abs_t` = Wetmore moderated-t = `fit / SE` | per-measurement noise already in the data |
| Replicate σ(condition) (option B) | std of `fit` across replicate `expName` sharing a `condition_key` | ~72–75% of conditions; **singletons need a σ fallback (see gotchas)** |
| chem-kNN expectation | `chemistry_knn_predict(train, query, cond_features, k=5, exclude_self=True)` — `src/ranking/eval/harness.py:337` | **exists**; `exclude_self` gives leave-one-condition-out so the expectation never includes the scored cell |
| Replicate pairing helper | `retrieval_noise_floor` (same harness) | reuse for the σ machinery |
| Dark-genome filter | `Gene` table join on `(orgId, locusId)`: keep `type=1` (protein-coding) AND `desc` null/empty or ~`%hypothetical%`/`%DUF%`/`%uncharacterized%`/`%unknown function%` | verified: 221,005 protein-coding; **54,891 orphan-like**. Optional stricter: also require no hit in `BestHitSwissProt`/`SEEDAnnotation`/`Reannotation` |
| Cross-org validator (PRIMARY) | `Ortholog` table `(orgId1, locusId1, orgId2, locusId2, ratio)` | **2,838,750** rows; chem-kNN never reads orthology → independent evidence |
| Specific-phenotype validator (SECONDARY) | `SpecificPhenotype` `(orgId, expName, locusId)`, join `expName`→condition via `Experiment` | 38,525 rows; **partially circular** (derived from the same `fit`/`t`) → supporting only |
| Leakage to AVOID | `Cofit`, `ConservedCofit`, `SpecOG` (split-blind / positives-only) | see `docs/project_memory/bugs.md` |

## Method / data flow

```
parquet (fit, t, expName, gene_key, condition_key)
  ├─ pool replicates ───────────────► observed fit  +  σ(condition)
  ├─ chemistry_knn_predict(exclude_self=True) ─► expected fit          [harness, exists]
  └─ r* = (observed − expected) / σ          (studentized residual)
        │
        join Gene (type=1, desc=orphan) ─► keep dark-genome cells
        │
        rank by |r*| ─► candidate list:  (orphan gene → stressing condition)
        │
        ├─ scramble/sign-flip null ........... GO/NO-GO GATE
        ├─ Ortholog concordance (deg-preserving null) ... PRIMARY evidence
        └─ SpecificPhenotype enrichment ...... secondary (semi-circular)
```

**Exists already:** data loaders, `chemistry_knn_predict(exclude_self=...)`, replicate pairing.
**New (all lightweight, no training):** residual + studentization, σ-fallback for singleton
conditions, the `Gene`-join orphan filter, the ortholog concordance test, the scramble null.

## Evaluation (no NDCG — this is discovery)

1. **Go/no-go gate (cheap, run first):** scramble within each condition (swap replicate labels /
   sign-flip), recompute residuals, and check whether the top *real* orphan residuals stand above
   the top *scrambled* residuals. If indistinguishable → the signal is noise → stop, near-zero cost.
2. **Primary evidence — cross-organism ortholog concordance:** for each top hit, compute its
   ortholog's residual *independently* in another organism; test (degree-preserving permutation
   null) whether real ortholog pairs agree on the stressing condition more than random pairs. This is
   non-circular because chem-kNN never reads orthology.
3. **Secondary — SpecificPhenotype enrichment:** corroboration only; do not headline (semi-circular).

## Honest baseline + ceiling

- **Baseline / gate:** the scramble null. The bar is simply "real surprises ≠ noise surprises,"
  then "orthologs corroborate."
- **Ceiling:** modest — a curated list of tens-to-low-hundreds of dark genes with a proposed stressor
  each, not a large quantitative effect. Value is biological (hypothesis generation for the dark
  genome) + methodological (negative-result-as-discovery-tool). Publishable at small effect; the
  framing holds even if the list is short.

## Gotchas (the only real judgment calls)

- **σ-fallback for singleton conditions** is load-bearing: too small → noisy singletons fake hits;
  too large → discard ~25% of conditions. Decide the fallback (pooled over `expGroup` / per-org σ)
  deliberately and sanity-check both failure modes.
- **Use `exclude_self=True`** so the expectation is genuine leave-one-condition-out (no trivial
  self-inclusion).
- **Lead with ortholog concordance, not SpecificPhenotype** (the latter is derived from the same fit).
- **Orphan definition** — start permissive (`desc` unknown), report sensitivity to the stricter
  "no hit in SwissProt/SEED/Reannotation" definition.

## Phased plan

1. **Spike (days):** residual pass + orphan filter on 2–3 organisms (e.g. Keio/Caulo/MR1) → run the
   scramble go/no-go. Decision point: signal vs noise.
2. **If it survives:** ortholog concordance across the full replicate-floor org set → the nomination
   list + the permutation-null significance.
3. **Write-up:** the dark-gene nomination list (gene, stressor, cross-org corroboration) + the method.
