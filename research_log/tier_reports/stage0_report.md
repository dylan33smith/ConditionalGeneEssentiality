# Stage 0 Report — Reproducibility & Governance

**Status:** approved (2026-04-27). See [`decisions/stage0/S0-DEC-001.md`](../decisions/stage0/S0-DEC-001.md).

## Summary

The S0 acceptance gate has passed. The smoke pipeline runs end-to-end, emits a
manifest validating against `run_manifest_v1.schema.json`, and reproduces
bit-identically across fixed-seed reruns.

## Required outputs delivered

| Output | Location |
|---|---|
| v4 schema verification | `data_contract/v4_schema_verification.json` |
| Run manifest schema | `data_contract/schemas/run_manifest_v1.schema.json` |
| End-to-end smoke pipeline | `src/experiments/stage0/run.py` |
| Test harness | `tests/` (26 passed, 2 stubbed for S2/S4) |
| Decision-ledger template | `research_log/decisions/decision_template.md` |

## Acceptance gate evidence

- **Reproducibility**: two reruns of `+stage=s0_reproducibility train.seed=0`
  produced identical smoke digests
  (`4796b8b1824f75f4630e9eeb846c9a054e87c094eb5da3ef21c8f8858b9bdd3f`).
- **Schema validation**: manifest passes `jsonschema.validate`.
- **Tests green**: full suite passed including
  `test_s0_smoke_reproducibility.py` (4 tests).
- **v4 verification**: 4,333 rows × 10 expected columns confirmed; 5 distinct
  `Decomposition_type` values discovered (`direct`, `extract`, `mix`, `salt`,
  `unrecoverable`) — these inform S4's `representation_mode` mapping.

## Numbers (smoke scale, n=80,000)

| Predictor | RMSE | MAE |
|---|---|---|
| global train mean | 0.6486 | 0.3201 |
| additive baseline | 0.7097 | 0.3973 |

The additive baseline is **worse** than the global mean at this scale because
it overfits with thousands of gene_keys and thousands of expNames over a 64k
train sample. This is expected to invert at full S2 scale (millions of train
rows). Promotion-eligible additive baselines will be computed in S2 on the
locked S3 split.

## Open items carried into S1

- Implement S1 data characterization → emit
  `data_contract/splits/candidate_protocols.yaml`.
- Vectorize `fit_additive_baseline` before S2 (TODO flagged in source).
- Decide whether to delete or regenerate the deprecated legacy media parquets.
