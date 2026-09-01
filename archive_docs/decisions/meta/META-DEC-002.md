## Decision: META-DEC-002

### Header
- decision_id: META-DEC-002
- stage_or_tier: meta (project-scope governance)
- date: 2026-04-27
- owner: project lead
- status: approved
- related_experiments: [S1_data_characterization (figure 17)]
- related_hypotheses: [H-SPLIT-01]

### Assumption Under Test
- assumption_statement: The project's generalization claim must be bounded by
  what the v4 data can honestly support. Specifically: with v4 chemistry overlap
  ≥95% in every candidate protocol, we cannot claim "generalizes to any
  chemistry" — only "generalizes to organisms and condition contexts within
  the v4 vocabulary."
- assumption_type: data / scope
- why_it_matters: scope creep at write-up time leads to overclaimed
  generalization, which is a fast path to a paper that doesn't replicate.
  Locking the bounded scope now prevents that.

### Pre-Registered Test Plan
- comparison: examine fig 17 (`chemistry_seen_unseen_rate_per_protocol`) and
  the candidate-protocol chemistry overlap rates emitted by S1.
- success criteria:
  - if any candidate protocol has val/test Canonical_ID seen-rate < 50%, the
    "any chemistry" claim is testable on the data we have.
  - if every candidate has seen-rate ≥ 95%, we cannot test that claim and
    must lock a narrower one.

### Evidence Summary
S1 fig 17 results (from `data_contract/splits/candidate_protocols.yaml`):

| Protocol | val_canonical_id_seen_rate | test_canonical_id_seen_rate |
|---|---:|---:|
| `largest_by_rows` | 98.61% | 97.62% |
| `high_overlap_easy` | 100.00% | 100.00% |
| `low_overlap_stress` | 94.74% | 95.12% |
| `multi_org_balanced` | 100.00% | 100.00% |

All four protocols are ≥94.7% chemistry-seen. No candidate exists with
substantial unseen chemistry mass. v4 has only ~112 Canonical_IDs and the
chemical ubiquity histogram (fig 16) shows most are widely shared.

The "any chemistry" claim is **not testable** with current data. To test it
would require either:
1. **Chemical-fingerprint encoders** (Morgan / RDKit / RDF) that represent
   unseen chemicals via shared substructure with seen ones — currently
   deferred per REFACTORPLAN §12.
2. **Canonical_ID-level holdouts** in addition to organism holdouts — also
   deferred per REFACTORPLAN §12.
3. **External validation set** with novel media — also deferred.

### Decision
- decision_outcome: **lock the narrower scope as REFACTORPLAN §2 L7.**

Locked statement (verbatim, REFACTORPLAN §2 L7):

> "Given a gene and a growth medium drawn from a known chemistry vocabulary,
> our model predicts conditional gene essentiality — including for organisms
> not seen during training, and conditions structured differently from those
> the gene appeared in during training."

- rationale: this is a real, narrower, publishable claim that is testable
  on v4 data. It correctly highlights what *is* novel (organism generalization,
  novel condition structure within seen chemistry) and excludes what we
  cannot honestly demonstrate (chemistry generalization).
- risks_remaining:
  - A reviewer may push for novel-chemistry generalization. Response:
    cite §12 deferred experiments and the trigger conditions; framing
    is intentional.
  - If the model fails even at this narrower claim, the L7 lock makes it
    visible early rather than hiding behind ambitious framing.
- next_action: L7 already added to `docs/REFACTORPLAN.md` §2 (L7 row).
  Mirrored in `CLAUDE.md` so it's loaded every session. This entry
  documents the formal decision; no code changes required.

### Reproducibility Attachments
- evidence figure: `research_log/figures/stage1/17_chemistry_seen_unseen_rate_per_protocol.png`
- supporting data: `data_contract/splits/candidate_protocols.yaml`
  (chemistry_overlap fields per candidate)
- CLAUDE.md scope section
- REFACTORPLAN.md §2 L7 row
- REFACTORPLAN.md §12 (Deferred Experiments) — chemistry-axis subsection lists
  the three paths that would unlock the broader claim.
- code_sha at decision time: 15a9df1
