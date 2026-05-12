# Stage 3 Report — Split Protocol Lock

**Status:** approved (2026-04-28). See [`decisions/stage3/S3-DEC-001.md`](../decisions/stage3/S3-DEC-001.md).

---

## TL;DR

- Primary protocol locked to `multi_org_balanced` via deterministic rule `s3_v1_power_driven`.
- Secondary stress protocol is `low_overlap_stress` (`reported_not_gating`).
- Because `H-HOMO-01` triggered in S1, S3 emits a locked homology diagnostic recipe:
  - primary diagnostic: similarity-bin stratification
  - secondary diagnostic: masked homology-clean metric subset
  - cosine cutoff: `0.85`

---

## Inputs consumed

- `data_contract/splits/candidate_protocols.yaml` (S1 output)
- `artifacts/baselines/baselines_per_protocol.json` (S2 output)
- `data_contract/policy/eval_policy.yaml` (S2 output)

## Selection rule used

S3 applied `s3_v1_power_driven`:

1. Eligibility:
   - Spearman role = `primary`
   - bootstrap `n_genes_used >= 200`
   - `val_rows >= m * 200` where `m` is from `eval_policy.yaml`
   - val Canonical_ID seen-rate `>= 0.90`
2. Ranking:
   - prefer protocols where embedding-NN beats global RMSE
   - tiebreak by higher `n_genes_used`
   - then by higher `val_rows`

This acts as the explicit override for the literal 30–80% overlap band in
REFACTORPLAN S3, which is degenerate on this dataset due to near-saturated
Canonical_ID overlap across all candidates.

## Candidate criterion table

| Protocol | Eligible | n_genes_used | val_rows | val Canonical seen-rate | NN beats global RMSE? |
|---|:---:|---:|---:|---:|:---:|
| `largest_by_rows` | yes | 3041 | 2104545 | 0.986 | no |
| `high_overlap_easy` | yes | 6095 | 2605550 | 1.000 | no |
| `low_overlap_stress` | yes | 1424 | 244971 | 0.947 | no |
| `multi_org_balanced` | yes | 13804 | 2165234 | 1.000 | **yes** |

## Locked outputs

- `data_contract/splits/locked_protocol.yaml`
  - `protocol_id: multi_org_balanced`
  - `seed: 0`
  - `seed_set: [0, 1, 2]`
  - `split_manifest_sha256: 74095fbfec6b34897fe02a146a46b290bdcd4c23661315bf6a2ef82abb57e74f`
- `data_contract/splits/diagnostic_protocols.yaml`
  - `secondary_stress`: `low_overlap_stress`
  - `homology_diagnostic`: `primary_bin_stratified_plus_secondary_masked_subset`,
    threshold `0.85`, bins `[0.0, 0.5, 0.85, 1.0]`

## Figure references informing the lock

- S1 figure 17: chemistry seen/unseen per protocol
- S1 figure 18: embedding cosine to nearest train per protocol
- S2 figure 04: bootstrap CI / protocol power evidence

## Next action

Proceed to S4 (Feature Contract) using `multi_org_balanced` as the single
promotion-gating protocol, with `low_overlap_stress` and homology diagnostics
reported as non-gating diagnostics.
