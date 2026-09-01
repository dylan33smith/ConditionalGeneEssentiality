## Decision: META-DEC-001

### Header
- decision_id: META-DEC-001
- stage_or_tier: meta (plan governance)
- date: 2026-04-27
- owner: project lead
- status: approved
- related_experiments: none (planning decision)
- related_hypotheses: none (this decision constrains the structure under which all H-* are tested)

### Assumption Under Test
- assumption_statement: Plan v1's stage ordering and tier scoping are coherent
  enough to begin Stage 0 implementation without revision.
- assumption_type: governance
- why_it_matters: every downstream stage and tier consumes the plan's structure;
  ordering errors propagate.

### Pre-Registered Test Plan
- comparison: Plan v1 (as committed at git `91a4c71`, archived at
  `archive/docs/REFACTORPLAN_v1.md`) vs revised Plan v2.
- evaluation method: structural critique by an independent researcher review pass —
  inspect for circular dependencies between stages, overlapping ownership of
  decisions across tiers, hypotheses without concrete experiments, missing
  schema/contract artifacts, configuration framework consistency.
- promotion rule: Plan v1 is rejected if any of the following are found:
  (a) a stage requires output from a stage ordered later than itself,
  (b) two tiers claim the same decision space without disambiguation,
  (c) hypotheses are listed without concrete owning experiments,
  (d) required artifacts are referenced but not specified.

### Evidence Summary
Plan v1 review surfaced six material defects:

1. **Circular stage ordering.** v1 ordered Stage 0.5 (Evaluation Trustworthiness)
   *before* Stage 1 (Data Characterization). But Stage 0.5 requires "null-baseline
   suite for each protocol" and "metric power report" — both of which need
   candidate protocols emitted by Stage 1. Stage 0.5 cannot complete its
   deliverables until Stage 1 has run. Fail (a).

2. **Tier-1 / Stage-2.5 ownership overlap.** v1 Stage 2.5 hard-gated a
   "default condition-feature bundle" decision (chemistry-only vs
   chemistry+metadata vs chemistry+metadata+extract flags). v1 Tier 1 also
   hard-gated this, via experiments 1A/1C plus required ablations. Two stages
   own the same decision. Fail (b).

3. **Tier-2 / Tier-3 ownership overlap.** v1 Tier 2 included condition-gated
   gene features (Exp 2C). v1 Tier 3 included FiLM-like / attention-like
   interaction blocks (Exp 3B). These are the same modeling concept at
   different scales. Fail (b).

4. **Orphan hypotheses.** v1 listed 21 H-* hypotheses, but H-TRAIN-01
   (balanced sampling), H-TRAIN-02 (curriculum), and H-SPLIT-02 (random vs
   stratified selection) had no concrete owning experiment. Fail (c).

5. **Missing run-manifest schema.** v1 referenced "run manifest schema" as a
   Stage-0 deliverable and listed minimum fields in Stage 0.5 and Stage 2.5,
   but specified no JSON Schema artifact. Fail (d).

6. **Configuration framework / dependency mismatch.** v1 configs used
   Hydra-style `defaults: - base/data` composition syntax, but `pyproject.toml`
   did not list Hydra. Fail (d).

Additional smaller findings (documented in plan-revision discussion):
- Frozen ProteomeLM embeddings were a tacit assumption, not a hypothesis.
  Promoted to H-EMB-01 in v2.
- Additive baseline `fit ~ a + α[gene] + β[condition]` was listed as an
  *optional* check in v1 Stage 0.5, despite being the most informative null
  for the project's research question (gene×condition interaction). Promoted
  to required gate (H-BASE-01) in v2.
- Spearman variability threshold `v_min` was unpinned — every tier could
  retroactively pick a flattering value. Pinned in v2 (cross-gene IQR p25,
  frozen in S2).
- Stage-to-stage handoff was prose, not artifacts. v2 makes every stage emit
  a named file consumed by downstream stages.

### Decision
- decision_outcome: **promote Plan v2; reject Plan v1.**
- rationale: Plan v1 contains structural defects in (a) stage ordering and
  (b) decision ownership that would force re-litigation during execution.
  Plan v2 fixes ordering to S0 → S1 → S2 → S3 → S4 → S5 → T1–T4, gives every
  decision a single owner, maps every retained hypothesis to exactly one
  experiment, adds the run-manifest JSON Schema, commits to Hydra as the
  config framework, and converts stage handoffs to files.
- risks_remaining:
  - Plan v2 has not yet been exercised end-to-end. The first time S0 actually
    emits a run manifest may surface schema gaps. Acceptable risk; iterate
    via the same ledger mechanism with explicit version bump
    (`run_manifest_v2.schema.json`).
  - The legacy media parquets at `data/derived/canonical/v0/media_master.parquet`
    and `media_components_long.parquet` reflect v1 (45 media). v4 has 120+ media.
    Plan v2 marks these deprecated and instructs S1/S4 to read v4 directly.
    Worth flagging as a future cleanup: regenerate the parquets from v4 or
    delete them outright.
- next_action: implement Stage 0 (S0) per Plan v2. First S0 work item is
  v4 schema verification + checksum manifest + smoke pipeline.

### Reproducibility Attachments
- v1 plan archived: `archive/docs/REFACTORPLAN_v1.md` (837 lines)
- v2 plan: `docs/REFACTORPLAN.md`
- code_sha at decision time: `f11f4d4`
- preflight verification: feba.db, canonical parquets, embedding bundle all load
  successfully (verified 2026-04-27); columns match v2 contract; legacy media
  parquets confirmed stale and marked deprecated in
  `data_contract/schemas/canonical_tables.schema.json`.
