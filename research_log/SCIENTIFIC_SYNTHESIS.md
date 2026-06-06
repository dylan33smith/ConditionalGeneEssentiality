# Scientific Synthesis — Conditional Gene Essentiality

**Purpose.** The canonical project-state narrative, organized by the *scientific
questions* we have asked and answered — not by gate/stage/tier. Read this to
understand what is durably established, the mechanisms behind each result, and
the strategic fork we now face (esp. the fitness-aware-embedding decision).

**Last updated:** 2026-06-06 (after R-HYBRID-A + R-HYBRID-B; modeling thread
concluded, characterization scope adopted — R-HYBRID-DEC-002 approved).

For the operational pipeline and per-decision detail see `docs/REFACTORPLAN.md`
(T-regime), `docs/RPLAN.md` (R-regime), and `research_log/decisions/`.

---

## 0. The foundational bet

> A **frozen, sequence-derived** protein embedding (ProteomeLM) + a
> **condition / media-chemistry encoding** contains enough information to
> predict **conditional gene essentiality**, and this generalizes across genes
> and organisms.

Nearly every experiment since has been, in effect, a stress-test of this single
assumption. It is now substantially undermined. Understanding *how* is the key
to every forward decision.

---

## 1. Q: Can frozen embeddings + chemistry predict fitness for held-out ORGANISMS?

**(T-regime: pointwise MSE/MAE regression, cross-organism holdout.)**

**Answer: No — not the part that matters.** RMSE/MAE were optimized acceptably
across T1–T6, but the biologically meaningful quantity — the *within-gene
ranking* of conditions — was **≈ 0.045 Spearman** vs a ~0.43 replicate noise
floor under cross-organism holdout. That is statistically indistinguishable
from random.

**Mechanism (the pivotal diagnosis).** RMSE is dominated by the **gene-mean** —
each gene's baseline fitness level. A model can fit gene means and the
*population* condition effect (which conditions hurt *most* genes) and score
decent RMSE, while learning **nothing** about the **gene × condition
interaction** — the actual signal of interest. We later confirmed the
population effect is worthless for ranking: predicting the population condition
profile (chem-NULL baseline) scores ≈ 0 within-gene Spearman.

**Two reasons identified for the cross-org failure specifically:**
1. **Homology distance** (S1, H-HOMO-01 at 1.6σ): held-out organisms' genes lie
   far from any training gene in embedding space, so the model cannot place them.
2. **Fitness-blindness of the embedding** (established later in the R-regime):
   ProteomeLM encodes protein sequence / structure / proteome-context — not how
   a gene *responds to stressors*. Cross-org compounds this with distribution
   shift. ProteomeLM beat raw ESM-C by only RMSE ~0.006, so even the
   proteome-*contextual* machinery added almost no conditional signal.

**Durable finding:** predicting the *level* of fitness is easy and gene-mean-
dominated; predicting the within-gene *conditional ordering* is the real, hard
problem, and **cross-organism transfer of it is ≈ 0**.

---

## 2. The reframe (why the R-regime exists)

Two changes after T7-prep diagnostics:
- **Generalization claim narrowed** from cross-organism to **within-organism**
  (predict a *known* organism's gene response to *novel condition
  combinations*).
- **Metric changed** from RMSE to **per-gene ranking** (within-gene Spearman +
  NDCG@k) — the quantity a biologist actually uses ("which conditions stress
  this gene most?").

A structural fact this surfaced: under condition-holdout the held-out conditions
are **100% disjoint from train** (cold columns), so this is **inductive
(cold-start) matrix completion with chemistry as column-features** — not the
standard warm-column collaborative-filtering setup. (Vanilla matrix
factorization cannot even be applied to the primary split.)

---

## 3. Q: Within a known organism, can we rank a gene's conditions?

**(R-regime, condition-holdout, the primary task.)**

**Answer: Yes — but a simple LOCAL baseline wins.** Full-val numbers (23
replicate orgs, eligible val genes):

| method | Spearman | NDCG@5 | what it is |
|---|---|---|---|
| chem-NULL | 0.02 | 0.31 | population condition profile (no gene specificity) |
| deep model (frozen emb + chem) | 0.13 | 0.42 | the "intended" model (R1) |
| linear-MF (learned free gene latents) | 0.14 | 0.43 | a *learned, fitness-aware* gene rep |
| **chem-kNN** | **0.24** | **0.485** | gene's OWN history, chemistry lookup |
| noise floor (ceiling) | 0.39 | 0.66 | biological-replicate agreement |

The achievable ceiling is **modest** (replicates only agree at NDCG@5 ~0.66 /
Spearman ~0.39) — this is an intrinsically noisy biological target.

---

## 4. Q: Is the ENCODER, the OBJECTIVE, or CAPACITY the lever?

Tested independently; **all three say no.**

- **Encoder** (R1: multihot vs Morgan / RDKit / MACCS fingerprints, 6 arms ×
  3 seeds × 23 orgs): no arm differs meaningfully; none beat chem-kNN. Multihot
  was marginally best — fingerprints did not help (same direction as T6 under
  MSE+cross-org). *H-R-CHEM-01 rejected (provisional; R1-DEC-001).*
- **Objective** (R-LOSS: MSE, Huber, RankNet, LambdaRank, ListMLE, ApproxNDCG):
  Huber marginally best (NDCG@5 0.435); ranking losses do **not** help —
  LambdaRank reaches good NDCG@5 but collapses Spearman (top-focused),
  ApproxNDCG is worst. None beat chem-kNN. *H-R-LOSS-01 rejected (R-LOSS-DEC-001).*
- **Capacity**: linear-MF (≈13k params, bilinear) ≈ deep model (2.5M params,
  nonlinear). Adding capacity/nonlinearity buys nothing.

**Mechanism (the synthesized finding).** Every *global parametric* model
plateaus at NDCG@5 ≈ 0.42–0.44; a *local non-parametric* chemistry-kNN reaches
0.485. The signal is **local and gene-idiosyncratic**:
- The population / cross-gene pattern carries ≈ 0 within-gene signal (chem-NULL).
- So all the signal is in each gene's *own deviations* from the bulk.
- A global model minimizing average error over ~54k genes captures the bulk and
  **averages the idiosyncratic deviations away** (they look like cross-gene
  noise). kNN **preserves** them by reading each gene's actual history and
  interpolating by chemistry similarity.

**Analogy.** Global model = a doctor predicting a never-before-seen patient from
their "type." kNN = a doctor holding that patient's full chart, predicting the
new situation from the most similar past situations *for that patient*. When the
patient's own history is available and idiosyncratic (our warm-rows setup), the
second wins — not by being smarter, but by having patient-specific information
the first compressed away.

---

## 5. Pressure-test: would a FITNESS-AWARE embedding change the answer?

The instinct is "every failure traces to the fitness-blind embedding, so train a
fitness-aware one." **We already have indirect evidence against this for the warm
task**, and it must temper expectations:

**The linear-MF result is already a fitness-aware-embedding experiment.** Its
free per-gene latent `U[g]` is a representation learned *entirely from the
fitness data* — it discards ProteomeLM and learns a fitness-aware gene vector
from scratch. Head-to-head:
- Frozen *sequence* embedding (deep model): NDCG@5 **0.42**
- Learned *fitness-aware* latent (linear-MF): NDCG@5 **0.43**
- → swapping a fitness-blind rep for a fitness-aware one did **not** beat kNN.

So the *warm-task* bottleneck is **not** the embedding's fitness-blindness — it
is the **global-vs-local** structure. Even a perfect fitness-aware gene vector,
used in a global model, loses to local lookup, because the warm task is
fundamentally about *memorizing each gene's own history*, which any global
representation compresses away.

**Where a fitness-aware embedding could still matter: only the COLD case.** For
an unseen gene, free latents are undefined; you must *compute* a representation
from features — i.e., an **inductive** fitness-aware embedding (e.g., fine-tuned
ProteomeLM). But that is the **cold-gene / cold-org** regime, exactly where the
prior is worst (T-stage cross-org ≈ 0.045). And note: on a cold gene, chem-kNN
itself **vanishes** (no history to look up), so a hybrid reduces to the global
model alone — back in the weak regime.

**Honest verdict:** fitness-aware embeddings are unlikely to help the warm task
(linear-MF shows the rep isn't the warm bottleneck) and their only real target —
cold transfer — is the hardest regime with the most discouraging prior. High
effort, high risk, upside concentrated in the regime we already flagged as
near-intractable.

---

## 6. What is DURABLY established

1. **Conditional essentiality has a real but modest within-gene signal** —
   ceiling NDCG@5 ~0.66 / Spearman ~0.39 even for biological replicates. A noisy
   target, not a clean one.
2. **The signal is gene-idiosyncratic and local.** Population/cross-gene
   structure ≈ 0 within-gene signal; per-gene history carries most of it.
3. **Cross-organism transfer of the conditional signal is ≈ 0.** Frozen protein
   embeddings carry no transferable conditional-response information.
4. **On the within-org task, a chemistry-similarity kNN (NDCG@5 0.485) is a
   strong baseline that learned global models do not beat** — across encoders,
   objectives, capacities, and both frozen *and* learned-fitness-aware gene
   representations.
5. **The task is memorization-dominated.** The useful product — "rank a known
   gene's stressors in a known organism" — is achievable (~0.485 NDCG@5), via
   lookup. Deep learning, as configured, adds nothing over it.

---

## 7. R-HYBRID resolved the fork: the hybrid does NOT beat kNN

We ran R-HYBRID in two rounds on the warm task. **Both failed to beat the
chem-kNN gate by a promotable margin**, and the second round — three orthogonal
*learned* fusions — converged on ≈ 0, which is the decisive evidence.

- **R-HYBRID-A (static z-score ensemble; R-HYBRID-DEC-001).** A fixed convex
  combination `α·z(kNN)+(1−α)·z(model)` with honest held-out α-selection gave
  +0.008 NDCG@5 over kNN — real complementary signal, but **below** the ~0.026
  promotion delta. First thing to edge past kNN, but not promotable.
- **R-HYBRID-B (three learned hybrids; R-HYBRID-DEC-002, approved).** Residual,
  retrieval-augmented, and learned-gating models, full 23-org eval, 3 seeds,
  denominator parity:

  | model | Δ NDCG@5 | Δ Spearman | honest held-out Δ NDCG@5 |
  |---|---|---|---|
  | residual | +0.0002 | −0.0049 | +0.0009 |
  | retrieval-augmented | −0.0059 | −0.0109 | −0.0057 |
  | learned gating | −0.0081 | −0.0181 | −0.0081 |

  The learned hybrids extracted **less** complementary signal than the crude
  static ensemble. The best (residual) is a statistical tie that **sign-flips
  across seeds** (+0.0048 / −0.0076 / +0.0033) — within split-noise, not a
  stable gain. Learned gating settled at mean α ≈ 0.46 but every bit of weight
  it placed on the model *hurt*: the data says "trust the lookup." (A Keio-only
  dev run gave residual a misleading +0.042 that vanished on full multi-org
  eval — the known single-org-ceiling artifact.)

**Conclusion.** Three orthogonal fusion designs converging on ≈ 0, on top of the
earlier encoder (R1), capacity, and objective (R-LOSS) sweeps all failing, is
strong evidence the chem-kNN ceiling is **real, not a fusion-design artifact**.
A global model — alone or hybridized — adds nothing promotable over local lookup
on the warm task.

### Remaining paths

| Path | What it is | Status |
|---|---|---|
| **Characterization (ADOPTED)** | Publish the rigorous negative: memorization-dominated task, strong-baseline benchmark, pre-registered evidence that learned global models don't beat chemistry-similarity lookup | **This is the scope.** Already supported by R1 + capacity + R-LOSS + R-HYBRID-A/B. |
| Cold-gene diagnostic | Quantify how far chem-kNN degrades on unseen genes — the one regime a global model could in principle help — to bound the value of any future modeling | Optional, characterization (not a gate). The next concrete measurement. |
| R-EMB (fitness-aware embedding) | Fine-tune / jointly-learn an inductive fitness-aware gene representation | Deferred. linear-MF shows it won't help warm; its only target (cold transfer) has a discouraging prior (T-stage 0.045). Eyes-open moonshot only. |

**The decision (2026-06-06):** the modeling thread is concluded. The honest,
publishable result is the rigorous negative — a pre-registered benchmark showing
the within-org conditional-essentiality ranking task is memorization-dominated,
with a simple chemistry-similarity lookup as the strong baseline that learned
global models (across encoders, objectives, capacities, and frozen *and*
learned-fitness-aware representations, alone *and* hybridized with the lookup)
cannot surpass.

**What the cold-gene split is for (corrected framing):** it is a *diagnostic*,
not a target. chem-kNN vanishes on cold genes, so the split measures the
**learned/global** component alone — i.e., it decomposes performance into
memorization (warm, kNN-dominated) vs transferable generalization (cold,
global-only). Expected to be poor (consistent with T-stage); valuable for the
paper's "is this just memorization?" question; optional to run.
