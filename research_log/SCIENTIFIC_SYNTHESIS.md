# Scientific Synthesis — Conditional Gene Essentiality

**Purpose.** The canonical project-state narrative, organized by the *scientific
questions* we have asked and answered — not by gate/stage/tier. Read this to
understand what is durably established, the mechanisms behind each result, and
the strategic fork we now face (esp. the fitness-aware-embedding decision).

**Last updated:** 2026-06-06 (after R1-DEC-001 + R-LOSS-DEC-001).

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

## 7. The strategic fork

| Path | What it is | Upside | Risk / cost | Evidence-based prior |
|---|---|---|---|---|
| **R-HYBRID** | Global model + local kNN (residual / retrieval / ensemble) on the WARM task; fix kNN's sparse-neighborhood & cross-gene blind spots | Beat the 0.485 lookup → a real model contribution | Cheap, reuses everything | The only path with a clear positive-result mechanism |
| **Characterization** | No new modeling; publish the rigorous negative: memorization-dominated task, strong-baseline benchmark, why learned models don't win | Honest, credible (pre-registration), publishable at a benchmark/methods venue | Low effort; modest contribution | Already largely supported by current results |
| **R-EMB (fitness-aware embedding)** | Fine-tune / jointly-learn an inductive fitness-aware gene representation | Could move cold-gene/cold-org off the floor (transferable biology) | High effort; biggest lift | linear-MF says it won't help warm; cold prior discouraging (T-stage 0.045) |

**Recommended order:** **R-HYBRID first** — cheapest, decisive, and the only
path with a positive-result mechanism. If it beats kNN, that is the contribution.
If it merely ties kNN, that is strong evidence the task is pure lookup, which
(a) makes the **characterization** paper the honest scope, and (b) tells us a
global model isn't the answer — at which point the **embedding moonshot** (R-EMB)
is the only remaining lever, to be entered with eyes open about its cold-regime
prior.

**What the cold-gene split is for (corrected framing):** it is a *diagnostic*,
not a target. chem-kNN vanishes on cold genes, so the split measures the
**learned/global** component alone — i.e., it decomposes performance into
memorization (warm, kNN-dominated) vs transferable generalization (cold,
global-only). Expected to be poor (consistent with T-stage); valuable for the
paper's "is this just memorization?" question; optional to run.
