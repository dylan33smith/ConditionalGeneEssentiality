# Project memory — Decisions

Why the project is built the way it is. Architecture and approach choices with
their rationale. Update this when a structural/scientific decision is made or
changed. The formal promotion-gate record lives in `research_log/decisions/**`;
this file is the distilled, fast-to-read "why."

---

## Scientific / modeling decisions

### Reframed from cross-organism regression (T) to within-organism ranking (R)
The original objective predicted a gene's continuous fitness across **held-out
organisms** (the "T-regime"). RMSE optimized fine, but the meaningful quantity —
within-gene *ordering* of conditions — was ≈ random (~0.045 Spearman) because
cross-organism transfer of the conditional signal is ≈ 0 and RMSE is gene-mean-
dominated. **Decision:** narrow the claim to *within-organism* and switch the
metric to *ranking* (within-gene Spearman + NDCG@5). The whole T-regime was later
pruned (see `docs/PRUNED_INDEX.md`).

### chem-kNN is the gate, not just a baseline
A chemistry-similarity kNN (predict a gene's fit at a novel condition from its own
fit at the chemically-nearest *seen* conditions) scores NDCG@5 ~0.485 and **beats
every learned global model** (encoders, objectives, capacity, frozen and learned
reps, and static+learned hybrids). **Decision:** a learned model only earns
promotion if it beats chem-kNN by ΔNDCG@5 ≳ 0.026 with disjoint hierarchical-
bootstrap CIs. Rationale: the task is memorization-dominated and local; a global
mapping averages the gene-idiosyncratic signal away. (R1-DEC-001; SCIENTIFIC_SYNTHESIS.)

### Locked model = AdapterResidualMLP (frozen embedding + learnable adapter)
A learnable adapter over the **frozen** ProteomeLM-L8 embedding, concatenated with
the chemistry vector, then a residual-MLP head. It was the first config to gain
over frozen-only, and added capacity beyond it (deeper/wider, or free latents via
linear-MF) buys nothing. **Decision:** keep the embedding frozen; the bottleneck
is global-vs-local structure, not representation capacity.

### Encoder = 425-d multihot; loss = pointwise Huber (carried-forward base)
Morgan/RDKit/MACCS fingerprints did not beat the multihot chemistry encoder
(R1-DEC-001). Among objectives, Huber slightly beats MSE (robust to fit outliers)
and the ranking losses (RankNet/LambdaRank/ListMLE/ApproxNDCG) did **not** beat
the gate (R-LOSS-DEC-001). **Decision:** carry `pointwise_huber + multihot_425` as
the locked base. The ranking losses are **kept** as the substrate for the top-k
objective (they target the top of the list, which is the next experiment).

### Eligibility / metrics protocol (R-LOCK series)
- **R-LOCK-1:** rank only genes whose fitness has real spread (`tail_g = p95−p5`
  over a per-org threshold); weight train rows by `w_g`. Ranking flat/near-constant
  genes is meaningless.
- **R-LOCK-2:** within-org condition-holdout split, fraction 0.20, seed 0,
  replicate-grouped + expGroup-stratified.
- **R-LOCK-4:** co-primary within-gene Spearman + NDCG@5 (k=5); hierarchical
  org→gene bootstrap (genes within an org are correlated); BH-FDR across arms.

---

## Engineering / structure decisions (from the cleanup)

### `src/ranking/` is a self-contained package; experiments are thin specs
The reusable core (models, losses, eval, data, pipeline, train, runner) lives in
`src/ranking/`. CLI handlers in `src/experiments/<R*>/run.py` just declare arms and
call the shared runner. **Why:** a new test should be a declarative `ArmSpec`, not a
copy-pasted pipeline. Layering is strict: `experiments → ranking → data`; nothing in
`src/data` or `src/ranking` imports upward into `src/experiments`.

### Regression-gate-first; prune by deletion (not archive), learnings preserved
The cleanup built the `R-EVAL` bit-exact gate **first**, then gated every refactor
step against it. The legacy T-regime + dead code was **deleted** (−13.4k lines),
not archived in-tree, because git history + the `refactor` branch + the decision
ledger already preserve it; `docs/PRUNED_INDEX.md` maps each deleted component to
where its learning lives. Reusable substrate (loss family, `RankingBatch` sampler
contract, eval/baseline harness) was kept even though those experiments produced
negative results — they are the foundation for the top-k work.

### Condition-key helpers live in the data layer
`_condition_key` (= `(expDesc, media, temperature)`), `_normalize_string_keys`, and
`load_fitness` live in `src/data/datasets/conditions.py` — the data layer — because
both the data modules and the ranking pipeline consume them. **Why:** keeps
`src/ranking` and `src/data` free of any dependency on `src/experiments`.

### Eval: harness is canonical; contract's flat bootstrap kept distinct on purpose
`src/ranking/eval/harness.py` is the live, canonical eval (it produced the published
numbers). `contract.py` holds the R-LOCK-4 promotion-gate helpers. During cleanup
the duplicated per-gene-correlation was deduped (contract delegates to harness;
verified equivalent). But contract's **flat** bootstrap and harness's
**hierarchical** org→gene bootstrap are *genuinely different methods*, not
duplicates — they were **not** merged, because doing so would silently swap a flat
CI for a hierarchical one. Use the hierarchical CI for org-clustered promotion
decisions.

### `RankingBatch` adopted as the typed contract, sampler-swap deferred
`src/data/datasets/ranking_batch.py` provides pointwise/pairwise/listwise samplers
for the top-k objective. During cleanup we kept the current training numerically
identical (the gate stayed bit-exact). **Decision:** actually wiring the samplers
into training is the *first top-k modeling step* (it will move the numbers by
design), not part of the behavior-preserving cleanup.
