# Project memory — Progress

Where the project actually is. Read this first when picking the work back up.
The dated log below is append-only (newest first) — never rewrite past entries.

---

## Where we left off (2026-06-23)

- **Branch:** `ranking`. The R-COLD cold-gene diagnostic (handler +
  `prepare_cold_gene_data` + config + decision/memory) is **committed directly on
  `ranking`** (the work landed here, not on the stale `cold-gene` placeholder
  branch, which held a divergent uncommitted draft in its worktree). The earlier
  `topk-loss` branch is already merged + pushed at `b935485`.
- **Repo:** cleaned + modular — self-contained `src/ranking/` core, shared runner,
  `R-EVAL` regression gate. Training flows through the `RankingBatch` samplers.
- **Data:** byte-for-byte canonical parquet from `feba.db`. `data` is an untracked,
  gitignored machine-local symlink (recreate per worktree — a checkout can drop
  it). 48 organisms total; 23 have a reliable replicate noise floor (the headline
  eval subset); all 48 have ProteomeLM-L8 embeddings. See `bugs.md`.
- **Warm-split baseline (23-org/3-seed):** model NDCG@5 **0.4319** / Spearman
  **0.1522**; chem-kNN gate NDCG@5 **0.4852** / Spearman **0.2402**. Fast gate
  model 0.4468 / kNN 0.5091 (R-EVAL still bit-exact after the R-COLD changes).
- **R-COLD just landed — FIRST POSITIVE (R-COLD-DEC-001):** on the cold_gene split
  (whole genes held out → chem-kNN coverage **0.0000**, linear-MF inapplicable, so
  **chem-NULL is the gate**), the model BEATS chem-NULL on genes it never trained
  on: 23-org/3-seed NDCG@5 **0.2748 vs 0.2447** (Δ**+0.0301**), Spearman **0.0735
  vs 0.0359** (Δ**+0.0376**), n=11,761, disjoint across all 3 seeds. **The
  embedding carries transferable gene-specific signal.** This reframes the warm
  negative as a MEMORIZATION gap (kNN retrieves a gene's own history), not an
  "embeddings are useless" result. NOT a promotion vs the locked chem-kNN gate —
  chem-NULL is a weaker bar; no headline number changes.
- **Prior axes (all NEGATIVE on the warm split):** R-TOPK (objective), R-AUG
  (training-org volume → negative transfer), R1 (encoder/capacity), R-HYBRID.

## Next tasks

1. **DONE — committed on `ranking`** (`0db7f1a` R-COLD; CI surfacing in a
   follow-up commit). R-EVAL bit-exact (0.4468/0.5091). **Not yet pushed** —
   awaiting the nod.
2. **Bootstrap CI on the cold-gene Δ** — DONE for Spearman: the runner now
   surfaces the harness's hierarchical org→gene CI + a model-vs-gate disjointness
   flag. Full 23-org Spearman is **disjoint** (model 0.0735 [0.0523, 0.1034] vs
   chem-NULL 0.0359 [0.0230, 0.0514], thin margin); the fast 3-org set OVERLAPS
   (needs the full panel). REMAINING: extend the bootstrap to NDCG@5 — the
   headline metric still has no CI.
3. **Re-open encoder/capacity + training-org volume IN THE COLD-GENE REGIME.**
   R-AUG's negative transfer was measured warm-only; more-diverse organisms may
   HELP inductive (cold-start-over-genes) generalization even though they hurt the
   warm headline. This is the live lever now.

### Lower-priority follow-ups
- Migrate `r1/run.py` and `rconf/run.py` onto the shared runner (reval/rloss/raug
  already are).
- External Tn-seq datasets (MtbTnDB, A. baumannii — see 2026-06-18 survey):
  un-shelved as a COLD-GENE candidate. R-AUG closed them for the warm headline, but
  the cold-gene positive makes more-distant organisms worth revisiting for the
  inductive regime.

---

## Log

### 2026-06-24 — Runner: surface the hierarchical-bootstrap Spearman CI (R-COLD confirmatory)
The runner computed the harness's hierarchical (org→gene) Spearman CI per method
but dropped it (not in `METRIC_KEYS`). Added `spearman_ci_low/high` to the schema
(now in the CSV) and a model-vs-gate **disjointness flag** in `standardized_report`
(`_ci_disjoint`). This delivers the R-COLD confirmatory step: full 23-org cold-gene
Spearman is **CI-disjoint** (model 0.0735 [0.0523, 0.1034] vs chem-NULL 0.0359
[0.0230, 0.0514], thin margin), upgrading confidence from per-seed point estimates
to disjoint 95% CIs. The fast 3-org set OVERLAPS (0.1155 [0.0614, 0.1938] vs 0.0813
[0.0269, 0.1564]) — CI-significance needs the full org panel. Caveats: multi-seed
bounds are the mean of per-seed CIs (a summary band, not a pooled bootstrap), and
the harness bootstraps only Spearman — NDCG@5 (the headline metric) still has no CI
(the last confirmatory gap). Additive schema change; R-EVAL value-gate unaffected.

### 2026-06-23 — R-COLD: cold-gene diagnostic (FIRST POSITIVE — embedding generalizes)
Built + ran the designated cold-gene (inductive-over-genes) diagnostic. The
primary split is transductive over genes, so chem-kNN wins by retrieving a gene's
OWN history; the cold_gene split holds out WHOLE genes (zero train rows), which
makes chem-kNN structurally inapplicable (**coverage 0.0000**) and linear-MF
unlearnable (no per-gene latent), leaving **chem-NULL** (population condition
profile, gene-identity-free) as the only applicable baseline → the gate. **Result
(23-org/3-seed, n=11,761):** model NDCG@5 **0.2748** vs chem-NULL **0.2447**
(Δ**+0.0301**), Spearman **0.0735** vs **0.0359** (Δ**+0.0376**), with every model
seed [0.2732, 0.2746, 0.2765] disjoint above the gate. Fast gate (Keio+Caulo+MR1):
model 0.2905 vs 0.2633 (Δ+0.0271), same sign/magnitude. **The frozen ProteomeLM
embedding carries transferable gene-specific conditional-response signal** — the
first positive global-model result. Reframes the warm-split negative as a
MEMORIZATION gap, not an embedding-value gap. NOT a promotion vs the locked
chem-kNN gate (chem-NULL is a weaker bar; no headline number moves). Decision:
`research_log/decisions/rcold/R-COLD-DEC-001.md`. **Implementation:** parameterized
`prepare_r1_data` (split_fn / compute_mf / parity_pred_cols) + thin
`prepare_cold_gene_data`; `R1Data.parity_pred_cols` restricts denominator parity to
{model, chem-NULL}; coverage diagnostics + an all-NaN-baseline guard in
`_metrics_for_pred` (inapplicable baseline → NaN, not a row-order artifact);
configurable gate in the runner; `R-COLD` handler + `R-COLD_cold_gene.yaml` + CLI
registration + a unit test. **R-EVAL fast gate bit-exact after the change** (model
0.4468 / chem-kNN 0.5091); unit tests 125 passed. Reproduce:
`+experiment=R-COLD_cold_gene` (artifacts in `artifacts/runs/rcold/`).

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
