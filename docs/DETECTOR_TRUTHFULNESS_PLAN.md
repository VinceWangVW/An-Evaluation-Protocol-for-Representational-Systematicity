# Phase 22: Detector-Truthfulness Validation

## Goal

`Phase 22` validates the detector, not the real `G2` family. The question is:

> If a `G2`-style argument-locus causal kind exists in a model-native family, can the frozen closure pipeline recover it blind?

The repo framing stays frozen while this runs:

- `G1_binding`: bounded hypothesis-final success
- `G2_distributed_geometry`: `Phase 18` family-final negative
- no target-panel expansion
- no second architecture
- no new real `H1` family sweep

## Families

For each class:

- `interaction`
- `cross_step_bridge`
- `query_source_relation`

we construct `4` families with `2` seeds each:

- `predicate_sanity_h2`
- `argument_h2`
- `argument_h1`
- `argument_null`

These are model-native constrained fine-tune families, not protocol-side planted patches.

## Decision Rule

`Phase 22` may end in only one of two states:

- `detectability_validated`
  - every class passes family validity, anchorability, observability, `H2`, `H1`, and null specificity
  - then a preserved `Phase 18` real-family regression allows an ontology-final upgrade
- `retain_phase18`
  - any class fails after Stage A and its class-local Stage B fallback

## Why This Exists

`Phase 18` already supports the paper's main bounded claim. `Phase 22` is an epistemic upgrade path: it tries to show that the detector is truth-tracking for argument-locus kinds before we use the detector to justify any stronger ontology-style negative.
