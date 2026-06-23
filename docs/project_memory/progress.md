# Project memory — Progress

Where the project actually is. Read this first when picking the work back up.
The dated log below is append-only (newest first) — never rewrite past entries.

---

## Where we left off (2026-06-18)

- **Branch:** `topk-loss` (off `ranking` at `c416ecf`). Holds the top-k loss
  variants + the R-AUG experiment + their decisions/memory. **Not yet merged** to
  `ranking` — awaiting the go-ahead (adds new loss code + the R-AUG handler).
- **Repo:** cleaned + modular — self-contained `src/ranking/` core, shared runner,
  `R-EVAL` regression gate. Training flows through the `RankingBatch` samplers.
- **Data:** byte-for-byte canonical parquet from `feba.db`. `data` is an untracked,
  gitignored machine-local symlink (recreate per worktree — a checkout can drop
  it). 48 organisms total; 23 have a reliable replicate noise floor (the headline
  eval subset); all 48 have ProteomeLM-L8 embeddings. See `bugs.md`.
- **Baseline (23-org/3-seed):** model NDCG@5 **0.4319** / Spearman **0.1522**;
  chem-kNN gate NDCG@5 **0.4852** / Spearman **0.2402**. Fast gate model 0.4468 /
  kNN 0.5091.
- **Two experiments just landed (both NEGATIVE — the gate stands):**
  - **R-TOPK** (R-TOPK-DEC-001): top-k-truncated NDCG losses do NOT beat the gate
    and fall *below* pointwise_huber (lambdarank_top5 0.4239, approxndcg_top5
    0.3695). Objective axis closed.
  - **R-AUG** (R-AUG-DEC-001): training the model on all 48 orgs (eval still 23,
    gate bit-identical) made it **worse** — NDCG@5 0.4319→**0.4166** (Δ−0.0152),
    Spearman 0.1522→0.1270, disjoint across all 3 seeds. **Negative transfer.**
    The model→gate gap is NOT a data-volume problem; it is structural.
- **Open loose ends:** (1) merge `topk-loss` → `ranking` + push (needs nod);
  (2) the cold-gene diagnostic is the designated next experiment.

## Next tasks

1. **Cold-gene diagnostic** — `materialize_cold_gene` already exists. Quantify how
   far chem-kNN degrades on held-out *whole genes* (the one regime a global model
   could win, since kNN has no within-gene history to retrieve). This is now the
   only open lever after objective (R-LOSS/R-TOPK), encoder/capacity (R1), hybrids
   (R-HYBRID), and training-org volume (R-AUG) all failed to beat the gate.
2. **Merge `topk-loss` → `ranking`** (top-k losses + R-AUG handler + decisions),
   then delete the feature branch.

### Lower-priority follow-ups
- Migrate `r1/run.py` and `rconf/run.py` onto the shared runner (reval/rloss/raug
  already are).
- External Tn-seq datasets (MtbTnDB, A. baumannii — see 2026-06-18 survey):
  shelved. R-AUG's negative transfer makes more-distant organisms a worse bet for
  the within-org headline; revisit only if the cold-gene regime shows promise.

---

## Log

### 2026-06-18 — R-AUG: train-organism augmentation (NEGATIVE — negative transfer)
Tested whether training the global model on all 48 embedded organisms (eval still
the locked 23, gate held bit-identical) narrows the model→chem-kNN gap. It does
the opposite: aug_48org NDCG@5 **0.4166** vs base_23org **0.4319** (Δ**−0.0152**),
Spearman 0.1522→0.1270, with every aug seed below every base seed (disjoint). The
chem-kNN gate is bit-identical across arms (drift 0.000000), so the A/B is clean.
**Negative transfer:** the extra organisms' conditional structure doesn't transfer
(cf. T-regime ≈ random) and pulls the shared weights off the eval orgs. The
model→gate gap is structural, not a training-data-volume problem. Closes the "add
more organisms / external Tn-seq datasets" line for the within-org headline.
Decision: `research_log/decisions/raug/R-AUG-DEC-001.md`. **Implementation:** added
`R1Data.baseline_train` + `prepare_r_aug_data` (model trains on the union, val +
eligibility + ALL baselines stay locked to the 23 eval orgs) + the `R-AUG` handler
/ config. Reproduce: `+experiment=R-AUG_train_org_augmentation` (artifacts in
`artifacts/runs/raug/`). Context: a 2026-06-18 web survey of external Tn-seq data
(MtbTnDB, A. baumannii, Sphingobium SYK-6, Nichols E.coli) — shelved by this
result.

### 2026-06-17 — R-TOPK: top-k-truncated losses (NEGATIVE — objective axis closed)
Post-RankingBatch, retested whether a loss that truncates NDCG gain to the top-5
(matching the metric exactly) beats the gate. New losses `lambdarank_top5` +
`approxndcg_top5` added to the registry (+ unit test). 5 arms × 3 seeds × 23 orgs:
no loss beats the gate (0.4852); pointwise_huber stays best (0.4319), and the
truncated variants fall BELOW their untruncated forms (lambdarank_top5 0.4239,
approxndcg_top5 0.3695). Found+fixed an approxndcg_top5 training freeze (top-k gate
reused the score temperature → vanishing gradient; fixed with `gate_temp=2.0`).
Decision: `research_log/decisions/rloss/R-TOPK-DEC-001.md`. Reproduce:
`+experiment=R-TOPK_loss`.

### 2026-06-17 — Wire `RankingBatch` into training (first top-k step)
Replaced the hand-rolled batching in `src/ranking/train.py` with the tested
`RankingBatch` samplers (`PointwiseSampler` for pointwise; `ListwiseSampler` for
the ranking losses); deleted `_build_gene_groups`/`_make_batch`. Loss interfaces
unchanged. **R-EVAL result:** 23-org/3-seed locked `pointwise_huber` arm moved
NDCG@5 0.4347→**0.4319** (Δ−0.0028, within tol), Spearman 0.1509→**0.1522**;
chem-kNN bit-exact (0.4852/0.2402). Fast single-seed gate moved more (NDCG@5
0.4516→0.4468) — seed noise from the different shuffle stream. 3-lens adversarial
review CLEAN (index/mask/loss correct; movement is RNG variance, not a bug).
`reval_baseline.json` re-set to the new numbers (fast + full). pytest 70 (ranking
core) green. Note: a branch checkout dropped the gitignored `data` symlink in the
worktree — recreate it (`ln -sfn <real data root> data`) per `bugs.md`.

### 2026-06-17 — `data` symlink incident + byte-exact recovery
While running the 23-org headline anchor, the `data` link resolved to a self-loop
and the canonical parquet was unreachable. Root cause + permanent fix in `bugs.md`
(gitignore `/data` + keep `data` untracked). **Recovery:** rebuilt all four
canonical parquets from `feba.db` via `archive/data_processing/build_canonical_v0.py`;
`fitness_experiment_long.parquet` verified byte-identical to the original
(27,410,721 rows, 805,220,433 bytes, sha256 `9b981201…` = manifest). No permanent
loss. Re-launched the 23-org headline anchor (confirming ~0.435/0.485).

### 2026-06-15 — Documentation consolidation (this scheme)
Consolidated the scattered top-level docs into `README.md` (single source of truth
for current state) + `docs/project_memory/{decisions,bugs,progress}.md` (modular AI
working memory). Added the Memory Protocol hook to `CLAUDE.md`. Deleted the
now-redundant `ARCHITECTURE.md` / `PLAN.md` / `PROGRESS.md` (folded in).

### 2026-06-15 — Layering fixes (post-cleanup polish)
- **Fix A** — condition-key helpers extracted from `src/experiments/r0/analyses.py`
  into `src/data/datasets/conditions.py` (data layer). Net: nothing in `src/data` or
  `src/ranking` imports upward into `src/experiments`.
- **Fix B** — deduped `eval/contract.py`'s per-gene-correlation against the
  canonical `harness.per_gene_correlations` (verified equivalent; test_ranking_metrics
  17/17). Kept contract's flat bootstrap distinct from harness's hierarchical one
  (different methods; merging would swap a flat CI for a hierarchical one).
Regression: R-EVAL fast gate bit-exact (Δ=0.0000); pytest 119 passed.

### 2026-06-15 — Ranking-branch cleanup (reorganize + aggressive prune + modular runner)
**Goal:** clean `ranking` branch as the basis for the top-k objective. Each step
gated by R-EVAL (bit-exact) + pytest:
1. R-EVAL regression harness (fast gate: model 0.4516 / chem-kNN 0.5091 on
   Keio+Caulo+MR1; baseline in `data_contract/ranking/reval_baseline.json`).
2. Extracted the model (`ResidualBlock` + `AdapterResidualMLP`) into `src/ranking/models.py`.
3. Decoupled ranking from the T-tier (fingerprint loader, eval `harness`+`contract`,
   loss family moved into `src/ranking`); no ranking module imports any `tier*`.
4. Relocated pipeline + trainers into `src/ranking/{pipeline,train}.py`; added
   `src/ranking/runner.py` (ArmSpec + run_experiment + standardized report). `reval`
   and `rloss` handlers became thin runner specs.
5. Pruned the T-regime + dead code (130 files / −13.4k lines): stage0-5, tier1-6,
   diagnostics, `src/{train,models,domain,training,evaluation}`, failed hybrids, dead
   data utils, 51 T-regime tests, T*/stage* configs, root cruft; deregistered
   S0-S5/T1-T6 handlers. Learnings preserved in the ledger + SCIENTIFIC_SYNTHESIS;
   mapping in `docs/PRUNED_INDEX.md`.
6. Docs scheme (since superseded by the consolidation above).

**Learned:** training is deterministic → the gate is bit-exact, which made every
refactor verifiable. Kept the reusable top-k substrate (loss family, `RankingBatch`,
eval/baseline harness); deleted only single-use glue and the T-regime.

### Earlier (T-regime + R-regime exploration)
Full history is in the decision ledger `research_log/decisions/**` and the narrative
`research_log/SCIENTIFIC_SYNTHESIS.md`. Highlights: T-regime cross-org regression
failed (within-gene ranking ≈ random); reframed to within-org ranking (R); R1
(encoder), R-LOSS (objective), capacity, and R-HYBRID-A/B (model+kNN hybrids) all
failed to beat the chem-kNN gate; R-CONF showed the negative is noise-robust.
