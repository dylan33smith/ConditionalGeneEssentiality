# bugs.md — [Symptom] -> [Proven fix]

Grep by symptom. Grouped under subject headings. Every entry has been fixed and
verified at least once; do not re-derive.

---

## Data / git / environment

### `OSError: [Errno 40] Too many levels of symbolic links` on the canonical parquet
**Symptom:** the `data` link resolves to itself; the canonical parquet is unreachable.
**Root cause:** `.gitignore` had `/data/` (matches a *directory* named data) but not a
`data` *symlink* — a symlink is a file. A stray `git add -A` committed the link, and a
later checkout clobbered it into a self-loop.
**Proven fix (both parts required):** gitignore the symlink form as well as the
directory form, and keep `data` untracked. Never run `git add -A` while a `data`
symlink is present.
**Recovery:** the canonical parquet is rebuildable from `feba.db` via
`archive/data_processing/build_canonical_v0.py`; verify against the manifest SHA256.
**Related:** switching branches in a worktree can DELETE the gitignored `data` symlink.
Recreate it per worktree with `ln -sfn <real data root> data`.

### Infinite recursion when walking `artifacts/`
**Symptom (2026-08-25):** `find`, `rglob`, `copytree` and the docs verifier recurse
forever under `artifacts/`.
**Root cause:** `artifacts/artifacts` was a symlink pointing back at
`artifacts/` — the same self-loop class as the `data` incident above, but live and
unrecorded for two months.
**Proven fix:** `rm artifacts/artifacts`. Removing a symlink does not touch the target.
**Rule:** any recursive walk over `artifacts/` or `data/` must pass `-P` / `follow_symlinks=False`.

### The entire data ROOT vanished mid-session
**Symptom (2026-07-06):** `feba.db`, the canonical parquet and all embeddings gone;
the `data` symlink intact but its target missing.
**Root cause:** unknown — the sibling project directory the symlink points at disappeared.
**Recovery:** re-download the raw DB from the live Fitness Browser with a browser
user-agent (LBL blocks the default UA); the download is resumable.
**IMPORTANT:** the re-downloaded DB is a **newer release, not a byte-match** — 62
organisms / 9,532 experiments / 33.8M rows, versus the pinned 48-organism release. Every
number computed on the old release will shift. Re-pin `reval_baseline.json` before
treating any headline number as current.
**Not recoverable from the Fitness Browser:** `media_composition.xlsx` (reconstructed as
v5 from `feba.db`'s own `MediaComponents` table) and the ProteomeLM embeddings (retired;
replaced by ESM-C).
**Lesson:** the data root is a single point of failure. A checksummed backup of `feba.db`
plus the embeddings is cheap insurance.

### `orgs=null` silently gives different, lower numbers than the headline
**Symptom:** an R-EVAL run reports numbers well below the published values.
**Root cause:** `orgs=null` means ALL organisms. The headline is the 23-organism
replicate subset.
**Proven fix:** pass the explicit 23-organism list — the exact command is in `docs/data.md`.

### A file written by a heredoc lands in the wrong directory
**Symptom (2026-08-25):** a file intended for `research_log/` was created one level deeper,
under `research_log/decisions/`.
**Root cause:** the shell's working directory persists between tool calls. An earlier
`cd research_log/decisions && grep ...` left the shell there; the next heredoc used a
relative path.
**Proven fix:** open every shell command with an absolute `cd` to the repo root, or use
absolute paths in redirections. Never rely on inherited cwd.

### LibreOffice / `soffice` is non-functional in this environment
**Proven fix:** manipulate `.xlsx` with `openpyxl`, never by shelling out to `soffice`.

---

## Training / determinism

### The regression gate is bit-exact — rely on it
Training is deterministic, so `R-EVAL` reproduces to the last digit. Any movement at all
after a refactor is a real behavioural change, not noise. This is what made the 130-file
cleanup verifiable. **Do not "absorb" a moved number.**

### A ranking loss trains but scores far below its pointwise counterpart
**Root cause:** batching. Pointwise losses need ROW-batched samplers; pairwise/listwise
losses need GENE-batched ones, or the loss sees no within-gene comparisons.
**Proven fix:** dispatch by loss family — pointwise -> `PointwiseSampler`,
pairwise/listwise -> `ListwiseSampler`.

### ApproxNDCG-top5 does not train (loss flat from step 1)
**Root cause:** the top-k gate reused the score temperature, so the smooth-sort gradient
vanished.
**Proven fix:** give the gate its own temperature, `gate_temp=2.0`.

### LambdaRank collapses full-list Spearman
**Not a bug.** LambdaRank is top-focused by construction; it optimizes the head of the
list at the expense of the tail. Expected behaviour — this is why
`within_gene_kendall_mean` is carried as a no-regression guard.

### OOM while materializing per-row chemistry
**Proven fix:** the memory-safe per-experiment gather — build a small per-*experiment*
chemistry table and join, instead of materializing chemistry per row.

---

## Evaluation / metrics

### A ceiling or baseline number disagrees with another copy of "the same" number
**Symptom:** `noise_floor` appears as 0.3214, 0.358 and 0.393 in the same project;
`chem_knn` as 0.481 and 0.4852.
**Root cause:** the name is underspecified. These differ by **aggregator** (MEAN vs
MEDIAN), **denominator** (parity-eligible val genes vs all val genes), **organism
subset**, and in one case simply by **measurement date**.
**Proven fix:** every reported number carries its aggregator, gene set and split. The
comparable ceiling is the MEAN over parity-eligible val genes, matching how the model and
baselines are summarized. See `replicate_ceiling_spearman` in `docs/terms.md` and the
2026-08-25 corrections in `docs/memory.md`.

### `retrieval_noise_floor` returns a misleadingly low precision@1
**Root cause:** median aggregation over a binary per-gene metric collapses to 0.
**Proven fix:** aggregate per-gene metrics with MEAN. Already fixed in
`src/ranking/eval/harness.py`; the same reasoning governs the Spearman ceiling.

### `condition_chemistry` returns a stale or 0-feature chemistry map
**Root cause:** the cache was namespaced by nothing, so a different org-set read another
org-set's cache (e.g. Keio reading an 8-org cache).
**Proven fix:** namespace the cache directory by an org-set hash.

### A per-condition-train-mean baseline returns NaN for every val row
**Not a bug — a split property.** Under `condition_holdout` the val conditions are 100%
cold, so no train mean exists for them.
**Proven fix:** use the split-specific baselines — `chem_null` (nearest-condition
profile), `chem_knn`, `linear_mf`. Vanilla matrix factorization applies only to a
cell-holdout diagnostic, never the primary split.

### A baseline reports 0 rather than "inapplicable"
**Root cause:** on `cold_gene`, `chem_knn` has no own-gene history and `linear_mf` has no
learnable latent. An all-NaN prediction column can look like zero skill or produce a
row-order artifact.
**Proven fix:** the all-NaN-baseline guard in `_metrics_for_pred` returns NaN, plus an
explicit `coverage` diagnostic. Coverage 0.0000 means undefined, not bad.

### `feba.db` `Cofit` is TARGET LEAKAGE — never use it as a gene-gene edge
**What it is:** a precomputed table of Pearson correlations between gene fitness profiles,
computed over **all** experiments (split-blind).
**Why it leaks:** a `Cofit` correlation spans the held-out val conditions, so a cold
gene's top co-fitness neighbour's train profile reconstructs that gene's own held-out
answer key.
**Smoking gun:** cofit-only `ndcg_at_5` reached 0.73-0.75 — *above even the warm chem_knn
reference* — which is physically impossible without a leak. The rank-1 cofit neighbour's
train profile correlates with the val gene's true held-out profile at median r ~0.56.
**Rule:** never read `Cofit` (or any all-experiment co-fitness matrix) into a `cold_gene`
or `condition_holdout` method. Co-fitness must be recomputed TRAIN-ONLY. On `cold_gene` a
train-only co-fitness edge is undefined, so the frozen embedding is the only
leakage-free gene-gene edge source there.
**Also leaky:** `ConservedCofit`, and `SpecOG` (positives-only, no denominator, and its
`ogId` split leaks the target).

### A single-organism dev run shows a large gain that vanishes on the full panel
**Root cause:** the single-org-ceiling artifact. A Keio-only run gave the residual hybrid
+0.042; on the full 23-organism eval it was +0.0002.
**Proven fix:** never promote on a single-organism run. The per-organism breakdown is
required on every scoring.

### `RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x960 and 1152x1152)`
**Symptom:** ProteomeLM-L fails deep in its forward pass when fed ESM-C embeddings.
**Root cause:** ESM-C ships in two sizes. ESM-C **300M** is 960-d; ProteomeLM-L requires
**600M**, which is 1152-d. The 2026-07-06 run generated 300M (chosen for the
shared-embedding proposal) into a directory called `ESMC_embeddings`, whose name records
no variant, so the mismatch was invisible until the matmul failed.
**Proven fix:** generate with `--model esmc_600m` into a variant-named directory
(`ESMC_embeddings_600m`). Put the varying parameter IN the path -- this is the naming rule
existing precisely to prevent this.
**Rule:** never name an embedding directory after the model FAMILY. Name it after the
checkpoint.

### `KeyError: 'embeddings'` loading an ESM-C bundle
**Symptom:** `encode_proteomelm_layers.py` cannot read the ESM-C files.
**Root cause:** two on-disk layouts exist -- the documented two-key bundle, and a plain
locusId-to-tensor mapping written by the 2026-07-06 regeneration.
**Proven fix:** `_unpack_esmc_bundle` accepts both. It sorts by locusId so row order is
deterministic; relying on dict insertion order would make the embedding matrix depend on
how the file happened to be written.

### `mu` is not identifiable in an additive gene/condition fit
**Symptom:** the fitted grand mean does not match the generative one, and effect
magnitudes are not comparable between two fits.
**Root cause:** `fit = mu + a_g + b_c` is invariant under `(mu+k, a-k, b)`. Without a
constraint, `mu` silently absorbs `mean(a) + mean(b)`.
**Proven fix:** impose sum-to-zero on `a` and `b` after fitting, so `mu` is the grand mean
and the effects are deviations. Implemented in `src/ranking/targets.py`.

### `b_c` cannot be estimated on the primary split
**Not a bug -- a structural constraint.** Under `condition_holdout` the val conditions are
100% cold, so no training row exists at any val condition and the condition main effect is
unestimable there. `apply_demeaning` raises rather than substituting 0, because a silent 0
would understate the target and make the resulting numbers quietly incomparable. De-mean by
gene only on that split, or predict `b_c` from chemistry and accept that the target then
embeds the chem_null baseline.

---

## Documentation

### A correction is recorded but the old claim keeps being asserted
**Symptom:** `metric_contract.yaml` still states that the bottleneck is the training
objective, months after R-LOSS and R-TOPK refuted it; `bugs.md` recorded "use MEAN, not
MEDIAN" while the contract kept pointing at the median value.
**Root cause:** corrections flow into `memory.md` and nowhere else. Nothing propagates
them outward to the documents people actually read.
**Proven fix:** after writing any `[CORRECTION - ...]`, grep the distinctive **number or
phrase** — not the topic word — across `docs/`, `src/`, `scripts/`, `CLAUDE.md`, `paper/`
and the assistant memory directory. `tests/test_docs_contract.py::test_corrections_do_not_survive_elsewhere`
enforces this.

### Stale git worktrees produce phantom grep hits and hide unique files
**Symptom (2026-08-25):** greps over the repo returned 2-4 copies of every doc; two
worktrees held divergent versions of `SCIENTIFIC_SYNTHESIS.md` and `CLAUDE.md`.
**Root cause:** two abandoned worktrees under `.claude/worktrees/`.
**Proven fix:** `git worktree list` before trusting any repo-wide grep; exclude
`.claude/` from doc scans. **Check for uncommitted and untracked files before removing a
worktree** — one held a 266-line audit that existed in no commit and would have been lost.
