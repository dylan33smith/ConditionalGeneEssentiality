## Decision: R-COLD-DEC-001

### Header
- decision_id: R-COLD-DEC-001
- stage_or_tier: R-COLD
- regime: R
- date: 2026-06-23
- owner: project lead
- status: approved
- approved_date: 2026-06-23
- related_experiments: [R-COLD_cold_gene]
- related_hypotheses: [H-R-COLD-01]
- related_decisions: [R-AUG-DEC-001, R1-DEC-001]

### Assumption Under Test
- assumption_statement: H-R-COLD-01 — every prior lever (R-LOSS/R-TOPK objective,
  R1 encoder/capacity, R-HYBRID model+kNN, R-AUG training-org volume) failed to
  beat chem-kNN on the PRIMARY split, which is transductive over genes: each val
  gene also has TRAIN rows, so chem-kNN retrieves the gene's OWN history and that
  per-gene memorization is unbeatable. The cold_gene split holds out WHOLE genes
  (zero train rows), removing memorization by construction. If the frozen
  ProteomeLM embedding carries gene-specific conditional-response signal that
  GENERALIZES to unseen genes, the global model should beat the only baseline that
  still applies there — chem-NULL, the population condition profile.
- assumption_type: model / signal
- why_it_matters: This is the single regime left open after the warm-split axes
  closed (R-AUG-DEC-001 explicitly named it). On the primary split chem-kNN wins
  by retrieving within-gene history; that tells us nothing about whether the
  embedding has inductive value, because the strong baseline is memorization, not
  generalization. Cold genes are the only clean test of "does the embedding add
  gene-specific signal a global model can transfer to genes it never saw."

### Pre-Registered Test Plan
- comparison: locked arm (pointwise_huber + multihot_425 + T5-A) trained and
  evaluated on the `cold_gene` split (`materialize_cold_gene`, fraction 0.20).
  Held-out whole genes per org; the model early-stops on within-gene Spearman over
  the held-out (cold) val genes — no leakage (cold genes never appear in train).
- baselines + applicability under cold_gene:
    - chem-kNN  : predicts a gene's held-out conditions from that gene's OWN train
                  conditions → cold gene has none → **0% coverage (inapplicable)**.
    - linear-MF : needs a learned free per-gene latent U[g] → unlearnable for an
                  unseen gene → **inapplicable** (skipped, `compute_mf=False`).
    - chem-NULL : train-gene-MEAN fit at the chemically-nearest train condition —
                  gene-identity-free → **applies → it is the gate.**
  Denominator parity is restricted to {model, chem-NULL} (the two methods that can
  score a held-out gene) via `R1Data.parity_pred_cols`; chem-kNN/MF coverage is
  reported explicitly to make their inapplicability quantitative, not implied.
- metrics_primary: [ndcg_at_5, within_gene_spearman]
- promotion_rule (diagnostic): the model must beat chem-NULL on cold genes,
  consistently across seeds (ΔNDCG@5 > 0, per-seed disjoint), to establish that
  the embedding carries transferable gene-specific signal.

### Evidence Summary

**Full headline — 23 replicate orgs, 3 seeds, identical eligible cold val genes
(n = 11,761), split seed 0, 8 epochs:**

| method | within-gene Spearman | NDCG@5 | prec@5 | coverage |
|---|---|---|---|---|
| chem-kNN | — | — | — | **0.0000 (inapplicable)** |
| linear-MF | — | — | — | inapplicable (skipped) |
| chem-NULL (GATE) | 0.0359 | 0.2447 | 0.1261 | 1.0000 |
| **model (frozen emb + chem)** | **0.0735** | **0.2748** | **0.1505** | 1.0000 |
| **Δ (model − chem-NULL)** | **+0.0376** | **+0.0301** | **+0.0244** | — |

**Per-seed (disjoint — the win is robust, not seed noise):**
- model NDCG@5: [0.2732, 0.2746, 0.2765] · chem-NULL NDCG@5: 0.2447 (constant —
  it is model-independent). Every model seed sits well above the gate.
- model Spearman: [0.0712, 0.0756, 0.0738] · chem-NULL Spearman: 0.0359.

**Fast confirmation (Keio+Caulo+MR1, seed 0, n = 1,912):** model NDCG@5 0.2905 /
Spearman 0.1155 vs chem-NULL 0.2633 / 0.0813 → Δ +0.0271 / +0.0342. Same sign,
same magnitude as the headline.

**Findings:**
1. **The model beats the population baseline on genes it never trained on.**
   ΔNDCG@5 +0.0301, ΔSpearman +0.0376, disjoint across all 3 seeds. H-R-COLD-01
   **accepted.** This is the FIRST positive global-model result in the project.
2. **chem-kNN is structurally inapplicable here (0% coverage), confirming the
   premise.** Cold genes have no own-gene history to retrieve; the warm-split gate
   simply cannot play. So the model→chem-kNN gap on the primary split is a
   MEMORIZATION gap (kNN retrieves the gene's own offsets), not an
   "embedding-carries-nothing" gap — when memorization is removed, the embedding's
   generalization is the best available signal.
3. **The absolute level is modest and the regime is genuinely harder.** Cold-gene
   NDCG@5 (~0.27) sits far below the warm split (~0.43) for every method; the
   embedding adds real but small gene-specificity over the population average, not
   a large effect.

### Decision
- decision_outcome: **accept H-R-COLD-01 (diagnostic, directional).** The frozen
  ProteomeLM embedding carries transferable, gene-specific conditional-response
  signal: on held-out whole genes the global model beats the only applicable
  baseline (chem-NULL) by ΔNDCG@5 ≈ +0.030 / ΔSpearman ≈ +0.038, with point
  estimates disjoint across all 3 seeds. STRENGTH CAVEAT (see Confidence): the
  PRIMARY-metric (NDCG@5) bootstrap CIs OVERLAP — the result is directionally
  robust and Spearman-CI-disjoint, but not yet CI-significant on NDCG@5. Treat as
  a strong lead to pursue, not a closed/confirmed win.
- rationale:
  1. It reframes the project's central finding. The warm-split negative
     (R1/R-LOSS/R-TOPK/R-HYBRID/R-AUG) is a memorization-dominance result, not an
     "embeddings are useless" result: the model loses to chem-kNN only because
     chem-kNN memorizes each gene's own history, which is unavailable for unseen
     genes. The cold-gene regime is where the embedding's value is visible.
  2. It opens the one path forward the warm axes had closed: the inductive
     (cold-start over genes) objective is where the global model is not just
     competitive but BEST, and where more-diverse training data (R-AUG's negative
     was warm-only) could plausibly HELP rather than hurt.
- scope / what this does NOT establish:
  - This is **not** a promotion against the locked R1 gate. The gate is chem-NULL
    (the only applicable baseline on cold genes), a WEAKER bar than chem-kNN; the
    +0.030 here is not comparable to the ΔNDCG@5 ≳ 0.026-vs-chem-kNN promotion
    rule. No headline number changes.
  - Confidence (full 23-org bootstrap CI, both metrics now computed):

    | metric | model [95% CI] | chem-NULL [95% CI] | disjoint? |
    |---|---|---|---|
    | **NDCG@5 (PRIMARY)** | 0.2748 [0.2424, 0.3179] | 0.2447 [0.2118, 0.2823] | **NO (overlap)** |
    | Spearman (secondary) | 0.0735 [0.0523, 0.1034] | 0.0359 [0.0230, 0.0514] | YES |

    HONEST READ: on the **primary** metric (NDCG@5) the model and chem-NULL CIs
    **OVERLAP** — the +0.0301 is a consistent point-estimate win (disjoint across
    all 3 seeds) but is NOT CI-significant. Only the secondary Spearman is
    CI-disjoint (NDCG@5's per-gene variance gives it a wider CI). The fast 3-org
    set is overlap on BOTH metrics. So the cold-gene positive is **directionally
    robust (3/3 seeds) and Spearman-CI-disjoint, but does not clear a strict
    disjoint-CI bar on the primary metric.** An earlier note here claimed the win
    was CI-confirmed on the strength of Spearman alone — that was corrected once
    NDCG@5 got its own CI (exactly why NDCG@5 is held primary).
- recommended next work:
  1. DONE — the runner now bootstraps + surfaces the hierarchical (org→gene) CI for
     BOTH NDCG@5 (primary) and Spearman, with a model-vs-gate disjointness flag.
     Result: NDCG@5 CIs OVERLAP (not significant), Spearman disjoint. To firm up
     the primary-metric claim, a paired/pooled bootstrap on the per-gene NDCG@5
     **Δ** (model − chem-NULL) — which cancels shared per-gene variance and is more
     powerful than comparing two marginal CIs — is the right next test.
  2. Re-open the encoder/capacity and training-org-volume axes IN THE COLD-GENE
     REGIME (R-AUG's negative transfer was measured warm; diverse organisms may
     help inductive generalization). 
  3. Decide whether to reframe the project objective toward cold-start-over-genes,
     where the embedding-based model leads.
- reproduce:
    - fast : `python -m src.cli.run_experiment +experiment=R-COLD_cold_gene`
    - full : `+experiment=R-COLD_cold_gene experiment.tag=full
      experiment.model_seeds=[0,1,2] experiment.orgs=[...23 replicate orgs...]`
  artifacts in `artifacts/runs/rcold/`.
- implementation: `prepare_cold_gene_data` (pipeline.py; `split_fn=materialize_cold_gene`,
  `compute_mf=False`, `parity_pred_cols=["model_pred","null_pred"]`), the `R-COLD`
  handler (`src/experiments/rcold/run.py`), config `R-COLD_cold_gene.yaml`,
  configurable gate in the runner, and an all-NaN-baseline guard in
  `_metrics_for_pred` (an inapplicable baseline reports NaN, not a row-order
  artifact). Primary-split path is unchanged — R-EVAL fast gate reproduces
  bit-exactly (model 0.4468 / chem-kNN 0.5091).
