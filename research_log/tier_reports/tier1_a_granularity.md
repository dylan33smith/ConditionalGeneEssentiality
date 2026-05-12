# Tier 1-A Report — Granularity Test (H-ENC-01)

**Status:** completed (2026-05-12). Outcome: **no_winner** between the two
locked arms. See [`decisions/tier1/T1-DEC-001.md`](../decisions/tier1/T1-DEC-001.md).

---

## TL;DR

- **Both encoders beat the locked S2 best-non-global baseline** (embedding_nn,
  RMSE 0.5884) by ~0.07 RMSE. H-BASE-01 RMSE gate passes for both arms.
- **Between the two arms, no winner.** Mean RMSE gap = 0.0028 (multihot
  better), threshold = 0.0212. Mean MAE gap = 0.0059 (media_id better),
  threshold = 0.005. RMSE and MAE disagree (the H-METRIC-01 regime under
  heavy-tailed residuals).
- **Per-organism**: multihot wins on RMSE on every val org (ANA3, BFirm,
  pseudo3_N2E3, SyringaeB728a_mexBdelta), each by 0.002–0.006. Consistent
  but all below threshold.
- **Homology-bin crossover at cosine ≈ 0.70**: for val genes with **low**
  similarity to training, media_id wins; for val genes with **high**
  similarity, multihot wins. Real regime-specific story.
- **Structural caveat (locked into the protocol, not the arms).**
  `multi_org_balanced` has val_seen_rate = 1.0 at the media-name level —
  every val medium also appears in train. This neutralizes the chemistry
  encoder's main S1-predicted advantage (generalizing across
  organism-unique medium names). The H-ENC-01 hypothesis as written is
  effectively un-stress-tested by this protocol; a fair test would use a
  protocol where val medium names are *not* in train.

## Setup

- **Hypothesis:** H-ENC-01 (decomposed chemistry > coarse medium name).
  See REFACTORPLAN §4. Scope-clarification noted: T1-A measured gap
  conflates representation effect with chemistry-scope effect (per the
  2026-05-12 hypothesis revision).
- **Locked split:** `multi_org_balanced` (val =
  {pseudo3_N2E3, BFirm, ANA3, SyringaeB728a_mexBdelta}, test = {PV4}).
  val_rows = 2,165,234. train_rows = 24,764,047 after locked split.
- **Training recipe:** S5 weighted_full (cor12 × abs_t weights).
- **Feature substrate:** S4 Option D, `artifact_id = de21504134c84a6c`.
- **Model:** shallow concat-linear MLP, gene_emb (1152) ⊕
  condition_vec (425) → hidden(256) → 1. Same hidden capacity, lr, batch
  size, epochs for both arms.
- **Seeds:** {0, 1, 2}; epochs = 8 (no early stop).

## Two arms

| Arm | Condition encoder | Per-experiment input |
|---|---|---|
| **A1: media_id** | `nn.Embedding(114, 425)` — 113 train medias + UNK | media-string idx (int) |
| **A2: multihot_canonical_id** | Identity (passthrough) over locked 425-dim multihot (medium + stressor) | 425-dim float vector |

Same condition output dim (425) for both. UNK at idx 0 for media not in
train; multi_org_balanced happens to have 0% UNK rate because every val
medium is in train.

## Headline metrics

| Arm | RMSE mean ± std | MAE mean ± std |
|---|---:|---:|
| media_id | 0.5177 ± 0.0010 | 0.2895 ± 0.0013 |
| multihot_canonical_id | 0.5149 ± 0.0021 | 0.2954 ± 0.0017 |
| **gap (media_id − multihot)** | **+0.0028** | **−0.0059** |
| S2 threshold | 0.0212 | 0.0050 |

S2-locked baselines on `multi_org_balanced` (for context):
- global_train_mean: 0.6307 RMSE / 0.3159 MAE
- additive_baseline: 0.6320 RMSE / 0.3159 MAE
- embedding_nn (best non-global, per S2-DEC-001): **0.5884 RMSE / 0.3247 MAE**

Both arms beat embedding_nn by ~0.07 RMSE → H-BASE-01 gate passes for both.

## Bootstrap CIs (seed 0, 1000 resamples by row)

| Arm | RMSE 95% CI | MAE 95% CI |
|---|---|---|
| multihot_canonical_id | [0.5110, 0.5146] | [0.2935, 0.2945] |
| media_id | [0.5169, 0.5207] | [0.2888, 0.2899] |

CIs do not overlap on RMSE → multihot is statistically below media_id
on RMSE within seed 0. CIs do not overlap on MAE in the opposite
direction → media_id is statistically below multihot on MAE within
seed 0. Both gaps reverse interpretation depending on metric: the
**H-METRIC-01 disagreement regime under heavy tails**.

## Per-organism RMSE (best epoch, mean across seeds)

| Organism | media_id RMSE | multihot RMSE | gap |
|---|---:|---:|---:|
| ANA3 | 0.7558 | 0.7535 | +0.0023 |
| BFirm | 0.5140 | 0.5099 | +0.0041 |
| pseudo3_N2E3 | 0.4099 | 0.4079 | +0.0020 |
| SyringaeB728a_mexBdelta | 0.4071 | 0.4009 | +0.0062 |

**Multihot wins on RMSE on every organism**, consistently by 0.002–0.006.
The pattern is stable but every individual gap is below the S2 threshold.

## Homology-bin breakdown (seed 0 predictions)

| Cosine similarity bin | n_rows | multihot RMSE | media_id RMSE | gap (multihot − media_id) |
|---|---:|---:|---:|---:|
| [0.00, 0.50) | 16,572 | 0.4060 | **0.3984** | +0.0076 (media_id wins) |
| [0.50, 0.70) | 515,176 | 0.4464 | **0.4361** | +0.0103 (media_id wins) |
| [0.70, 0.85) | 532,555 | **0.4984** | 0.5102 | −0.0118 (multihot wins) |
| [0.85, 1.01) | 1,095,699 | **0.5489** | 0.5586 | −0.0097 (multihot wins) |

**Clear crossover at cosine ≈ 0.70.** For val genes with low similarity
to training (no close homolog), media_id wins by ~0.01 RMSE. For val
genes with high similarity (close homolog in train), multihot wins by
~0.01 RMSE. This is a real regime-specific effect even though the
overall gap averages out.

Caveat: RMSE rises with similarity bin because target magnitude rises
with conservation (well-conserved genes have larger-magnitude fitness
effects). Inter-arm gap *within* a bin is the comparable quantity.

## Figures

| # | Figure | What it shows |
|---|---|---|
| 01 | `01_val_metrics_per_arm.png` | RMSE+MAE bars per arm with cross-seed error |
| 02 | `02_per_org_val_rmse_per_arm.png` | Per-val-organism RMSE for both arms |
| 03 | `03_homology_bin_metrics_per_arm.png` | The homology-bin crossover |
| 04 | `04_train_val_curves_per_arm.png` | Train/val RMSE+MAE by epoch per seed |

## Promotion outcome

**`no_winner`** — neither arm beats the other by the S2-locked threshold
on both co-primaries. The decision rule per S2-DEC-001 requires both
RMSE and MAE gaps to exceed threshold in the same direction; here they
exceed in opposite directions.

## Interpretation

1. **The protocol neutralizes the H-ENC-01 stress test.** S1 figs 01–02
   showed that media-name overlap is sparse (median 0 shared media per
   org pair) while chemistry overlap is dense. But `multi_org_balanced`
   happens to be the protocol where val_seen_rate = 1.0 at the media
   level — every val medium is in train. So the media_id encoder never
   has to fall back to UNK. The S1-predicted advantage of multihot
   (generalizing across organism-unique medium names) is not exercised.
2. **For the locked-protocol "useful encoder" comparison, the two arms
   are tied.** Both produce a 425-dim condition vector with similar
   expressive capacity; for val media all seen in train, the media_id
   encoder can learn full per-medium effects and is competitive.
3. **The homology bin crossover is a real and informative finding.**
   For low-similarity val genes (the harder generalization regime),
   media_id wins by ~0.01 RMSE. For high-similarity val genes, multihot
   wins. One plausible mechanism: media_id has 425 trainable parameters
   per medium, giving it more capacity to learn the medium's mean
   fitness pattern when gene info is uninformative. Multihot's fixed
   chemistry vector is more general (good for similarity-driven
   generalization) but less direct.
4. **MAE-vs-RMSE disagreement is the H-METRIC-01 case.** Multihot
   produces a tail-resistant fit (lower RMSE) but a slightly worse
   central-mass fit (higher MAE) than media_id. This is exactly the
   regime S2-DEC-001 designed the co-primary policy for.

## What this implies for downstream tiers

- **The H-ENC-01 question is not yet answered.** The locked protocol
  doesn't stress-test the chemistry-vs-name distinction in the way S1's
  structural argument predicts. To get a clean test, we'd want a
  protocol where val media are *not* in train (e.g.,
  `largest_by_rows` where val_seen_rate = 0%). That's the proper
  follow-up.
- **T1-B, T1-C, T1-D, T1-E proceed with the multihot encoder** as the
  locked schema (because it beats embedding_nn baseline by enough to
  pass H-BASE-01 and there's a small consistent per-org RMSE advantage).
  The media_id encoder is retained as a diagnostic-only baseline for
  any future protocol that does have unseen val medium names.
- **The homology-bin crossover finding goes into the open-issues list**
  as a candidate driver for fusion-tier (T2) experiments. If multihot's
  advantage is concentrated in the high-similarity bins, a gating
  architecture that adaptively weights chemistry vs. medium identity by
  homology might do better than either arm alone.

## Outputs

- [`artifacts/runs/t1a/t1a_summary.json`](../../artifacts/runs/t1a/t1a_summary.json) — full numerical results
- [`artifacts/runs/t1a/t1a_metrics.parquet`](../../artifacts/runs/t1a/t1a_metrics.parquet) — per-epoch metrics
- [`artifacts/runs/t1a/t1a_summaries.parquet`](../../artifacts/runs/t1a/t1a_summaries.parquet) — per-(arm, seed) best-epoch summary
- [`research_log/figures/tier1_a/`](../figures/tier1_a/) — 4 figures (PNG + sibling CSV)
- [`research_log/decisions/tier1/T1-DEC-001.md`](../decisions/tier1/T1-DEC-001.md) — formal acceptance

## Open items carried into T1-B and beyond

- **Re-run T1-A on a protocol with unseen val media** (largest_by_rows
  is the natural choice — val_seen_rate=0). This would actually test
  the H-ENC-01 hypothesis as originally written. Filed as a follow-up
  in REFACTORPLAN §12.
- **Homology-bin gating** as a T2 candidate.
- **Investigate why media_id's MAE advantage exists.** Possibly:
  the high-capacity per-medium embedding lets the model learn medium
  means more efficiently, reducing central-mass error. Worth checking
  whether a wider chemistry projection (e.g., 425 → 1024) closes the
  gap.
