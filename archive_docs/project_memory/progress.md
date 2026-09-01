# Project memory — Progress

Where the project actually is. Read this first when picking the work back up.
The dated log below is append-only (newest first) — never rewrite past entries.

---

## Where we left off (2026-06-23)

> **Update 2026-07-06 (this session).** Two material changes — see the 2026-07-06 log entries:
> (1) **Data-root loss + recovery.** The external data root vanished mid-session; the raw fitness
> DB was re-downloaded from the live Fitness Browser, but it's a **newer 62-org release** (not the
> pinned 48-org one) — so the parquet needs rebuilding, embeddings regenerating, and the R-EVAL gate
> re-pinning before any headline number below is current. Details + recovery commands in `bugs.md`.
> `media_composition.xlsx` is still missing (blocks the parquet media columns).
> (2) **Project dashboard** built at `dashboard/` (generator `build.py` + content model
> `dashboard/content/{experiments,learnings}.json`) — a navigable working reference / deck served via
> `python -m http.server` + SSH tunnel. The content JSON is the persistent plain-language synthesis of
> every experiment (hypothesis · how-tested · explained numbers · analysis · meaning).

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

> **STRATEGIC NOTE (2026-06-25):** the warm "beat chem-kNN" objective is
> memorization-capped and that is now understood as a *named cross-field regime*,
> not a fixable modeling gap — see
> [research_log/LITERATURE_local_vs_global_memorization.md](../../research_log/LITERATURE_local_vs_global_memorization.md).
> Direction is under review by the project lead: (i) reframe-and-publish the
> characterization + cold-gene generalization + a principled local+global hybrid,
> (ii) ask a *different question* in the conditional-essentiality space (reframe
> search in flight), or (iii) wind down. **Do not invest in new global-model
> architectures for the warm ranking task** pending that decision.

### Candidate experiments (NOT yet committed — awaiting the lead's go-ahead)
- **[Reframe — TOP PICK] Dark-genome residual miner.** A *different question*, not a
  better ranker: among unannotated (hypothetical/DUF) genes, which show a
  chemically-specific essentiality the chem-kNN/population lookup CANNOT explain
  (studentized residual), corroborated by cross-organism ortholog concordance?
  Makes the lookup the *null model* → the memorization wall is structurally
  inapplicable. Spike = days; go/no-go is a replicate-swap null. Full ranked
  reframe slate + the honest verdict on the organism-level GNN idea (transductive +
  cold-gene arms = same wall; only the cold-CHEMICAL arm escapes, and only if it
  beats a fingerprint-kNN kill-switch) →
  [research_log/REFRAME_CANDIDATES.md](../../research_log/REFRAME_CANDIDATES.md).
  **Full implementation-ready plan (verified data grounding + data flow + go/no-go):**
  [research_log/PROPOSAL_dark_genome_residual_miner.md](../../research_log/PROPOSAL_dark_genome_residual_miner.md).
- **[Reframe — 2nd] De-leaked co-essentiality → functional linkage** (train-only
  correlation as feature; rehabilitates the `Cofit` footgun; scope to incremental
  AUPRC over embedding cosine on the hard slice).
- **[Option A] Similarity-stratified NDCG@5 diagnostic + ResMem.** (a) Bucket
  held-out (gene, condition) cells by the gene's distance to its nearest in-train
  condition; expect chem-kNN to dominate the near bucket and collapse in the far
  bucket where the model should win — this *empirically* answers "are we cheating?"
  and locates the real headroom (no training). (b) **ResMem** (kNN memorizes the
  MLP's *residual*, not a rival fitness prediction; [arXiv:2302.01576]) as the first
  principled hybrid — low effort, reuses the locked model + chem-kNN datastore,
  preserves the cold-gene win by construction. Both are hours, not days. Rationale +
  the full technique menu (Correct-and-Smooth, adaptive Meta-k gate, EASE, TabR) in
  the literature doc above.

1. **DONE — committed on `ranking`** (`0db7f1a` R-COLD; CI surfacing in a
   follow-up commit). R-EVAL bit-exact (0.4468/0.5091). **Not yet pushed** —
   awaiting the nod.
2. **Bootstrap CI on the cold-gene Δ** — DONE for BOTH metrics (NDCG@5 primary +
   Spearman). Honest result: **NDCG@5 CIs OVERLAP** (model 0.2748 [0.2424, 0.3179]
   vs chem-NULL 0.2447 [0.2118, 0.2823]) — the +0.030 is NOT CI-significant on the
   primary metric; only the secondary Spearman is disjoint (0.0735 [0.0523, 0.1034]
   vs 0.0359 [0.0230, 0.0514]). So the cold-gene win is directional (3/3 seeds) +
   Spearman-CI-disjoint, but not NDCG@5-CI-disjoint. REMAINING: a paired/pooled
   bootstrap on the per-gene NDCG@5 **Δ** (cancels shared per-gene variance, more
   powerful than two marginal CIs) to firm up the primary-metric claim.
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

### 2026-07-06 — Canonical parquet REBUILT on 62-org release (pipeline data re-armed)
Rebuilt `data/derived/canonical/v0/` on the restored 62-org `feba.db` + `media_composition_v5.xlsx`,
excluding the 94 wholly-undefined-media experiments. Faithful to the archived build's schema logic
(adapted script at `scratchpad/rebuild_canonical.py`; real data paths; row-count assert passed).
**Result:** `fitness_experiment_long.parquet` = **33,433,466 rows** (vs original 27.4M), experiments
= **9,438** (9,532 − 94 excluded), media_master = 191, media_components_long = 4,078; manifest at
`data/canonical_build_manifest_v0_62org.json`. All three data blockers are now cleared (data restored,
media v5 verified, embeddings + parquet built). **Remaining before R-EVAL can run:** the pipeline reads
ProteomeLM embeddings (lost in the data incident) — either regenerate ProteomeLM or switch the pipeline's
embedding source to the freshly-generated ESM-C, then re-pin the gate on the new release.

### 2026-07-06 — Canonical chemistry filled + ESM-C done + Phase-0 PASS (orthologs cluster in ESM-C)
Completed the media/embedding prep and ran the make-or-break Phase-0 check. **(1) Canonical chemistry.**
Filled the `Media_Components_ML` sheet of `media_composition_v5.xlsx` for all 69 new media
(4,406→8,622 rows, ~61/media; 135 canonical IDs). 64% of distinct new components reused existing
decomposition rules; the 42 uncovered mapped mostly to EXISTING canonical IDs (amino acids, ions, salts→ions,
`xan`, `inost`, `nta`) with ~11 new small-molecule IDs (lactose, cellobiose, fructose, maltose, ethanol,
DTT, glutathione, pyridoxamine, thiosulfate, iodide, vanadium) and ~6 non-nutrient additives flagged
`Include_in_ml=False` (agar, Tween 80, Tyloxapol, kanamycin, cycloheximide, BSA). Fixed a dedup bug
(reuse-map had aggregated each ingredient's decomposition across all media → 6× row inflation). All existing
rows in all three sheets preserved byte-for-byte. **Wholly-undefined media flagged + experiment-exclusion
list emitted** (`data/excluded_undefined_media.json`): Potato tuber (53), Potato stem (22), Potato Dextrose
Broth (18), Clovis xylem sap (1) = **94 experiments to exclude**. Not done: only Clovis xylem sap needed a
new undefined flag (Potato* were already handled in v4). **(2) ESM-C.** Generated `esmc_300m` (960-d)
mean-pooled embeddings for all 279,140 proteins / 62 orgs in 24 min → `data/processed/ESMC_embeddings/`.
**(3) Phase-0 sanity test — PASS.** Do orthologs cluster in ESM-C space? YES, decisively: ortholog cosine
median 0.92–0.99 vs random ~0.70, **AUROC 0.97–0.998** across close AND distant pairs (incl. Keio↔M.tb),
cosine monotone in sequence identity. The conserved-sequence-prior assumption holds → the shared-embedding
rewiring plan is viable. Minor: mean-pooled ESM-C is anisotropic (random cosine ~0.7) → center/whiten before
φ. Remaining rebuild blocker: run the canonical parquet build on the 62-org DB + v5 (point archived
`build_canonical_v0.py` at real data paths), then re-pin R-EVAL.

### 2026-07-06 — Pipeline unblock: media file filled (v5, full coverage) + ESM-C embeddings generated
Two of the three post-data-loss rebuild blockers cleared. **(1) Media composition.** The curated media
file was found at `data/media_composition_v4.xlsx` (6 sheets, 121 media). Against the 62-org DB, 70 media
used by experiments were missing from it — but **67 have composition directly in `feba.db`'s
`MediaComponents` table** (513 media; the Fitness Browser ships it — the "go to the papers" worry was
mostly unfounded), 2 are trivially derived (`R2A_0.5%agar` = R2A + agar; `ZMB_ALS_noFolicAcid` = ZMB_ALS
− folic acid), and only **`Clovis xylem sap`** is a genuinely undefined plant fluid (in-planta; added as a
composition-less Media row, `Include_in_ml=False`). Wrote **`data/media_composition_v5.xlsx`** (191 media,
4,078 component rows) — existing rows preserved **byte-for-byte**, columns identical, and **zero
experiment media now uncovered**. Registered `MEDIA_XLSX_V5` in `archive/repo_paths.py`. NB the archived
`build_canonical_v0.py` path layout (REPO_ROOT=archive/) needs the real data-root paths pointed at
feba.db + v5 at actual rebuild time. Follow-ups NOT done: the `Media_Components_ML` canonical-chemistry
sheet (needed for the nutrient-profile featurizer, not the basic build) and the documentation-only
`Experiments` sheet. **(2) ESM-C embeddings.** `esm` 3.2.3 + 4× A40 present; generating mean-pooled
**esmc_300m (960-d)** embeddings from the re-downloaded `aaseqs` (279,140 proteins, 62 orgs) → per-org
`data/processed/ESMC_embeddings/<org>_esmc.pt` (fp16, resumable), ~187 prot/s. Chosen over ProteomeLM
(proteome-contextualized → may not preserve ortholog similarity, the conserved-prior requirement) and
Evo2 (DNA diverges faster); Phase-0 will empirically compare ortholog cosine similarity. Remaining
blocker for a live pipeline: run the parquet rebuild on the 62-org DB + v5, then re-pin R-EVAL.

### 2026-07-06 — Formal plan: cross-organism shared gene–condition embedding (Direction 1, buildable)
Wrote [research_log/PROPOSAL_crossorg_shared_embedding.md](../../research_log/PROPOSAL_crossorg_shared_embedding.md)
— the implementation-ready form of the ortholog-rewiring reframe, as a shared-embedding factorization:
`fit_o(g,c) ≈ a_g^o + b_c^o + (s_g + δ_g^o)·v_c` with a **conserved sequence prior** `s_g=φ(emb)`, an
**org-specific fitness-learned deviation** `δ` (the λ-regularized rewiring signal), and a **shared
condition vector** `v_c`. De-meaning + denoising are built in. Key design decisions, grounded in the
restored 62-org DB: (1) **ESM-C (context-free) over ProteomeLM** — orthologs must be similar by
construction so `δ` isolates rewiring; ProteomeLM is proteome-contextualized (may break this); **Evo2
worse** (DNA diverges faster). Phase-0 empirical gate on ortholog cosine similarity. (2) condition
features = **learned `v_c` + a nutrient-profile inductive bridge** (concentration-weighted fingerprint
mixture) so novel/cold conditions are placeable; multihot as baseline. (3) **train on all genes, compare
only orthologs.** Two outputs: **Mode A** (fair cross-org condition transfer vs chem-NULL — a prediction
the lookup can't make) + **Mode B** (rewiring map). Data grounding: ~117 canonical conditions in ≥10
orgs; 179k genes with a cross-org ortholog (~52k across ≥10 orgs); DvH↔Miya 1,784 orthologs (rewiring
pilot), Caulo↔MR1 164; M. tuberculosis only 8% orthologous → helps Mode A not Mode B. Phased plan +
go/no-go gates + leakage discipline in the doc. Blocked on the parquet rebuild (needs
`media_composition.xlsx` or a media-less workaround) + ESM-C generation from `aaseqs`.

### 2026-07-06 — First-principles re-examination: the warm benchmark is RIGGED (retire H1, test H2)
Re-derived the problem from scratch, taking no prior conclusion at face value. Result **amends the
framing** of the whole warm-task program (SCIENTIFIC_SYNTHESIS §9 added; README central finding +
dashboard updated). Key points: (1) `fit(g,c)=μ+a_g+b_c+I(g,c)+noise`; within-gene ranking of eligible
genes = predicting the pure interaction `I(g,c)`. (2) Two conflated hypotheses — **H1** "beat chem-kNN"
(a benchmark) vs **H2** "do embedding+chemistry contain generalizable `I(g,c)` signal" (the science).
(3) The warm contest is **rigged** three ways (kNN handed the gene's own labels; eligibility selects
history-rich genes; random split manufactures near-neighbors), so H1's "no" is *expected*, not a deep
verdict on DL — and R-COLD (no own-history) already shows the model wins → H2 is a partial **yes**.
(4) The family that could beat the kNN (learned non-parametric: EASE / learned-metric kNN / TabR /
canonical ResMem; + de-meaned + denoised targets) was **never built** — but a learned-*local* winner
would only *confirm* memorization, so that sweep serves to make the negative airtight, not to "win".
(5) The 0.66 ceiling is the *single-noisy-measurement* ceiling; **denoising raises it**; the
kNN→ceiling gap (0.175) ≫ model→kNN gap (0.055) and is an unquantified noise-vs-structure mix.
**Decision framing:** demote chem-kNN from "the gate" to "a reference for what memorization achieves";
adopt an honest benchmark — **leave-compound-out (scaffold) + cold-gene, scored vs the ceiling, bar =
chem-NULL** — and move the real effort to the **reframed questions** (cold-condition, fitness-as-feature
→ function/MoA, matrix-structure/co-essentiality, cross-organism rewiring). No code/behavior change;
R-EVAL untouched. Next concrete diagnostics (cheap, unblock after parquet rebuild): de-mean + denoise +
re-run the model; similarity-stratified NDCG; a learned condition metric (EASE); flip primary task to
leave-compound-out.

### 2026-07-06 — Data-root loss + live re-download (NEWER 62-org release; reproduction caveat)
The external data root `/home/ds85/projects/GeneEssentiality/` disappeared mid-session (feba.db +
parquet + embeddings all gone; symlink intact but target missing). Re-downloaded the raw fitness DB
from the **live** Fitness Browser (`fit.genomics.lbl.gov/cgi_data/feba.db`, browser-UA, resumable):
integrity OK, **62 orgs / 9,532 exps / 33.8M rows**, sha `b6627137…` — a *newer, larger* release than
the pinned 48-org db (sha `627f2097…`). **Consequence:** headline numbers (model 0.432 / kNN 0.485 /
ceiling 0.66 etc.) were computed on the OLD db; they will shift on the new one — rebuild the canonical
parquet, regenerate ProteomeLM embeddings, and **re-pin `reval_baseline.json`** before treating them as
current. Blocker: `media_composition.xlsx` (parquet sidecar) is gone and not on the Fitness Browser.
Full incident + recovery commands + Figshare pinned-release fallbacks in `bugs.md`. No code/behavior
change; R-EVAL not re-run (data mid-restore).

### 2026-07-06 — Project dashboard (navigable working reference + deck) at `dashboard/`
Built a dependency-free static dashboard rendered by a pure-Python generator (`dashboard/build.py`)
from a content model: `dashboard/content/experiments.json` (13 per-experiment extracts: hypothesis ·
**how-we-tested** · every number explained in plain language · analysis · meaning · verification badge)
+ `learnings.json` (8 enriched durable learnings). Content was extracted from the decision ledger
(`research_log/decisions/**`) + synthesis via a fan-out workflow; this JSON is the **persistent
plain-language synthesis** (the user asked that the richer accounts persist). Pages: Overview · Key
result (8 grouped comparison tables — cross-org / warm / encoder / loss / hybrid / aug / confidence /
cold-gene) · Data · Timeline (scrollable, filterable, deep-linkable expandable cards) · Learnings ·
Directions · Status · Limitations&verification · References. Features: verdict filters, expand/collapse
all, `#exp-<id>` deep links, glossary tooltips, light/dark, presentation mode, print/PDF. Served with
`dashboard/serve.sh` (127.0.0.1:8080) + `ssh -L 8080:localhost:8080`. Every claim carries a
verification badge (code / artifact / web / ledger). Next: extract `build.py` into a reusable
project-dashboard **skill**; add command-palette search. No project-code/behavior change; R-EVAL untouched.

### 2026-06-29 — Adversarial hypothesis dialogue → 5-direction slate (blind generator + Socratic critic)
Ran a two-agent reframe exercise: a first-principles hypothesis generator **blind** to all
code/docs (raw `feba.db` only) produced 12 directions; a Socratic critic with full prior-work
access opened with the NN-wall briefing, flagged leakage landmines, and pressure-tested all 12; the
generator then refined under fire with fresh SQL. The decisive filter: *does a chemistry-kNN's
retrieved neighbor structurally contain the target's answer?* Full slate +
verdicts → [research_log/DIRECTIONS_adversarial_slate_2026-06-29.md](../../research_log/DIRECTIONS_adversarial_slate_2026-06-29.md).
**Headline new candidate (NOT in the prior reframe slate): #1 ortholog conditional-response
divergence** (conserved vs. rewired essentiality across orthologs) — the one direction that cleanly
escapes the wall, because the target is a cross-organism comparison and within-gene conditional
transfer is ≈0, so the organism-local lookup cannot compute it; the cross-org≈0 result becomes the
signal generator. N verified ample (328 hi-conf orthologs even for a distant Caulo↔MR1 pair; 48
stress compounds in ≥10 orgs). Strong second: #2 fitness-fingerprint MoA sensor (escape real only on
the structurally-distant slice, leave-compound-AND-org-out). #3 local-memorization characterization =
methods backbone. #4 substrate-utilization on probation (needs an external growth table — confirm
first). #5 de-leaked conditional epistasis (corroborates the earlier co-essentiality reframe).
Retracted/demoted: dark-genome-by-condition (premise falsified, matches R-DARK), dose-response
(only 7 cells ≥5 doses — too thin for genome-wide Hill fits), breadth-law (needs independent
target). No code/behavior change; R-EVAL untouched. Awaiting the lead's pick.

### 2026-06-25 — R-DARK spike (Steps 1–3): signal REAL, dark-genome enrichment FALSIFIED
Built + ran the dark-genome residual-miner spike (`src/experiments/rdark/spike.py` v1,
`spike_repro.py` v2; uncommitted). v1 surfaced that naive top "surprises" are `n_rep=1`
single-measurement artifacts and that a gene-permutation null tests predictability, not
surprise — so v2 does the honest test: replicated cells only (`n_rep≥2`), deviation must
reproduce across a gene's two replicate halves (|z|>3 both, concordant), within-condition
replicate-shuffle null; inline kNN verified vs the harness to ~1e-16. **Result (Keio/
Caulo/MR1):** (1) reproducible chemically-unexpected conditional-essentiality is REAL —
confirmed orphan surprises run ~10× the null in every org (794/76, 2235/293, 2404/234;
emp_p=0.002). (2) BUT orphans are NOT enriched — they are *less* surprising than annotated
genes in all 3 orgs (rate ~0.02–0.04 vs ~0.06–0.10; label_p=1.0). The "dark genome is
special" hook is dead; a confirmed chemically-unexpected reproducible phenotype ≈ a
`SpecificPhenotype` (already cataloged in feba.db, 38,525 rows), so novelty is thin. Only
surviving angle: orphan subset of residual-defined surprises gated on Step-4 cross-org
ortholog concordance — borderline. Full result + re-scope in
[research_log/PROPOSAL_dark_genome_residual_miner.md](../../research_log/PROPOSAL_dark_genome_residual_miner.md).
No project-code/behavior change; R-EVAL untouched.

### 2026-06-25 — Literature diligence on the wall + reframe search (strategic pivot under review)
After two evolutionary-search rounds (round 1: all 3 approaches collapsed to the
(frozen-emb, chemistry) parametric attractor; round 2: GENE-NW gene-axis retrieval is
the first CI-disjoint *paired* cold-gene positive, but its signal = R-COLD's embedding
gene-specificity, and the genuinely-new co-fitness channel is **target leakage** — now
in `bugs.md`), the project lead questioned whether the whole direction is viable.
Ran a citation-verified multi-agent literature sweep (8 fields). **Finding:** our
"local memorization (chem-kNN) beats global parametric learning" wall is a *named,
well-characterized cross-field regime* (kNN-LM "Generalization through Memorization";
Feldman long-tail necessity; tabular DL-loses-to-trees; recsys kNN-beats-neural /
Wide&Deep; QSAR applicability-domain). chem-kNN winning is the EXPECTED outcome, not a
leak/cheat. Full writeup +ranked categorically-different techniques (ResMem,
Correct-and-Smooth, adaptive Meta-k gate, EASE/TabR) + dead-ends + realistic ceiling:
`research_log/LITERATURE_local_vs_global_memorization.md`. Recorded **Option A**
(similarity-stratified diagnostic + ResMem) as an uncommitted candidate. Launched a
second multi-agent search (with verifiers) for *reframes of the question itself* —
different feasible questions in the conditional-essentiality space (transpose/MoA,
essentiality-as-feature/function-inference, organism-level bipartite GNN [user idea],
residual/discovery, typology/conservation) — results pending. No code/behavior change;
R-EVAL untouched.

### 2026-06-24 — NDCG@5 made PRIMARY project-wide + NDCG@5 CI (CORRECTS the Spearman-only read)
Per project-lead direction, NDCG@5 is now held above within-gene Spearman
*everywhere* (CLAUDE.md "Primary metric", metric_contract `metric_primacy`,
README/reval/runner all report NDCG first). Substantive fix: the harness only ever
bootstrapped Spearman, so the prior "confirmatory" entry below judged disjointness
on Spearman alone. Added a hierarchical (org→gene) bootstrap CI for **NDCG@5** in
`_metrics_for_pred` (ndcg_at_5 point value unchanged → R-EVAL bit-exact: 0.4468 /
0.5091) and made the runner/handler lead CIs + the disjointness verdict with NDCG@5.
**Corrected result (full 23-org):** on the PRIMARY metric the model vs chem-NULL
**NDCG@5 CIs OVERLAP** — 0.2748 [0.2424, 0.3179] vs 0.2447 [0.2118, 0.2823] — so the
+0.030 is a consistent point-estimate win (3/3 seeds disjoint) but NOT
CI-significant; only the secondary Spearman is disjoint. Lesson: NDCG@5's per-gene
variance gives it a wider CI than Spearman, so a Spearman-only confidence read
overstates significance — exactly why NDCG@5 is primary. Next: paired/pooled
bootstrap on the per-gene NDCG@5 Δ (cancels shared variance) to firm it up.

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
