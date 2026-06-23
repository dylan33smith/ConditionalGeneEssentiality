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
the locked base. The ranking losses are **kept** in the registry as a tested
substrate, but the objective axis is now closed (see R-TOPK below).

### The objective axis is closed — even top-k-direct losses lose (R-TOPK)
Post-RankingBatch, added NDCG@5-truncated losses (`lambdarank_top5`,
`approxndcg_top5`) that optimize *exactly* the promotion metric. They still do not
beat the gate and land **below** plain pointwise_huber (0.4239 / 0.3695 vs 0.4319;
gate 0.4852). **Decision:** keep pointwise_huber; stop pursuing the loss/objective
as the lever. The gap is structural (local vs global), not a surrogate-choice
problem. (R-TOPK-DEC-001.)

### More training organisms HURT — negative transfer (R-AUG)
Training the global model on all 48 feba organisms (eval held to the locked 23,
chem-kNN gate bit-identical) made it **worse**, not better: NDCG@5 0.4319→0.4166
(Δ−0.0152), Spearman 0.1522→0.1270, disjoint across all 3 seeds. **Decision:** do
NOT augment training with more organisms for the within-org headline; keep the
23-org training set. Rationale: the conditional signal does not transfer across
organisms (cf. T-regime ≈ random), so pooling heterogeneous orgs pulls the shared
weights off the eval orgs — classic negative transfer. This closes the
training-data-VOLUME axis and, by extension, shelves the external-Tn-seq-dataset
idea (more-distant organisms would only worsen the transfer) for this objective.
The **cold-gene** regime (where chem-kNN has no within-gene history to retrieve)
is the one remaining place a global model could win. (R-AUG-DEC-001.)

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

### `RankingBatch` wired into training (2026-06-17)
`src/data/datasets/ranking_batch.py` provides pointwise/pairwise/listwise samplers
for the top-k objective. `src/ranking/train.py` now builds batches through them
(`PointwiseSampler` for pointwise; `ListwiseSampler` for the ranking losses) and
the hand-rolled `_build_gene_groups`/`_make_batch` was deleted — one tested
batching path. **Why:** top-focused losses need list/pair-structured batches, and
this removes a duplication; it is the prerequisite for the top-k loss experiment.
**Result:** behavior-preserving at headline scale — the locked `pointwise_huber`
arm moved within gate tolerance (23-org/3-seed: NDCG@5 0.4347→0.4319, Spearman
0.1509→0.1522; chem-kNN bit-exact). A 3-lens adversarial review confirmed the
index/mask/loss wiring is correct and the small movement is RNG/batch-composition
variance (different shuffle stream), not a logic change. `PairwiseSampler` is wired
as available but the current pairwise losses still derive pairs from the listwise
[B,L]; switching them to native pairwise batches is a follow-up.
