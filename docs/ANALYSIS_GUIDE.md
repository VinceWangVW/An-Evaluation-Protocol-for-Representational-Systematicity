# Analysis guide

BOUNDED-RS uses one decision function, `release_analysis.adjudicate`, for
checkpoint audits, walkthroughs, sensitivity, bootstrap and transfer comparison.
The active profile is stored in `configs/release/decision_profile.json`.

## Features and decision records

The L1 signal schema includes:

```text
b0_gen_clause_f1
m1_semantic_role_accuracy
m1_surface_control_gap
m2_advantage_mean
m2_permutation_p_value
m3_selective_impairment_clause_f1
m4_advantage_mean
realism_pass_rate
linkage_pass_rate
evidence_coverage
audit_ready
state_anchor_ready
```

Readiness fields are booleans. Other fields are finite scalars. Accuracy, rate
and probability values lie in [0,1]. Required evidence is checked before a
stronger state is licensed; an observed zero is a numeric measurement.

For raw feature x, boundary tau, positive scale s and direction d in {-1,+1},
the support transform is:

```text
support(x) = sigmoid(d * (x - tau) / s)
```

Let P be the weighted presence score, U the weighted use score and B the
behavior support. Presence weights are 0.60 role readout and 0.40 controlled
surface gap. Use weights are 0.60 intervention advantage and 0.40 permutation
support. With behavior prior b=0.04, state scores are:

```text
Causal-use:   b*B     + (1-b)*(P + U)/2
Present-only: b*B     + (1-b)*(P + 1-U)/2
Unsupported:  b*(1-B) + (1-b)*(2-P-U)/2
```

The diagnostic Undetermined score is `1 - max(the evidential state scores)`;
the actual abstention decision follows coverage, margin and validity gates.
The winning evidential state must lead by more than 0.10, coverage must exceed
0.80, and readiness must hold. Causal-use additionally requires P>0.5, U>0.5,
realism>0.60, linkage>0.50, selective necessity>0.05 and repair advantage>0.245002.
Present-only requires P>0.5 and U<0.5; Unsupported requires P<0.5 and U<0.5.
Conflicting evidence yields Undetermined. Configurations declare all thresholds
and scales explicitly.

```bash
python3 scripts/run_release_analysis.py adjudicate \
  --signals SIGNALS.json --output runs/decision.json
```

L2 contains the raw features, sigmoid parameters, group scores, state scores,
margin, gate decisions, profile version and final state. No runner supplies a
state-score constant in place of this computation.

## Matched-set evidence

`scripts/run_matched_audit.py` accepts canonical JSONL and a checkpoint. Each
matched set must include its declared easy source, hard target and semantic
contrasts. Per-item operation records retain source/target IDs, role, gold and
generated logical forms, control identity and effect. Presence readouts include
role labels and surface/filler/predicate controls.

Decision features average matched-set estimates. Readouts use leave-one-item-out
prediction within each set; global readout diagnostics and the held-out
behavioral anchor are reported separately. Audit summaries and cluster
bootstrap use this same aggregation rule.

Interchange compares transfer toward the exact counterfactual gold form.
Necessity measures output degradation under removal of a role signal relative
to controls. Repair patches a same-meaning source into the hard target and
compares the resulting output with controlled alternatives. The full repair
pair denominator and baseline-error subset are separate fields. Sample counts
and measurement coverage travel with the L1 vector.

## Sensitivity

`configs/release/sensitivity_grid.json` declares six blocks, each with three
levels: score boundaries, scales, mixture weights, validity/abstention gates,
causal-effect gates and Geometry closure thresholds. The Cartesian grid has
729 configurations.

The panel JSON contains disjoint `calibration` and `targets` arrays. Every row
has `id` and `signals`; calibration also has an `expected_state`. Compatibility
is determined only from calibration labels. Optional `closure` records are
re-evaluated under each compatible configuration.

## Cluster bootstrap

Each input JSONL row contains a unique `matched_set_id` and the measured
per-set `signals`. The algorithm resamples complete clusters, preserves pairing,
recomputes effect means and a one-sided paired sign-permutation p-value, and
reapplies the profile. P-values are recomputed rather than averaged. The default
is 1,000 replicates and 199 permutations per replicate. The output records the
RNG seed, sampled indices, intervals and state frequencies.

Geometry rows can include `closure_objects` and identical `closure_context`.
Each object requires stable ID/class/step/template descriptors, discovery and
necessity gaps, null effect, realism and object-specific signals. Context
declares the complete `frozen_basis`, observability and presence-route results.
Duplicate IDs, nonfinite measurements and changing descriptors are rejected.

## Calibration and transfer

`seed-plan` and `transfer-plan` declare target training seeds and independent
calibration controls. Calibration uses shifted seeds, causal training,
decoder-cross-attention scaling 0.25 for a present-only regime, and permuted
targets for an unsupported regime. Regime labels are intended control states;
the audit computes their decisions from evidence.

`transfer-compare` computes frozen and recalibrated target decisions. Feature
boundaries are refitted from the labeled low/high calibration-group means;
weights, scales and validity gates remain frozen. Target values never enter
the boundary estimates.

## Known-truth controls and component panel

`truth-controls` executes a computational experiment in which a designated
role-like scalar or distributed bridge is causal, present but unused, or null.
Discovery searches candidates and separate confirmation measures readout,
interchange, necessity and repair effects.

`component-panel` defines 41 computational cases: seven causal and 34 noncausal
conditions spanning unused signals, nonspecific interventions, realism and
linkage failures, wrong-role operations and weak shortcuts. Each condition
generates observations and control comparisons. `component-rules` applies
recoverability-implies-causal, ungated intervention and full adjudication to the
same known-truth evidence. Output retains the model context, causal labels,
per-case effects, predictions, numerators, denominators and abstentions.

## SLOG input contract

Normalize upstream SLOG rows into JSONL with `item_id`, `input_text`,
`target_form`, `split` and `role_signature`. Include original train/dev rows
and the selected held-out pairs. Preserve the benchmark revision and license.
The selection file pins:

```json
{
  "source_revision": "UPSTREAM_REVISION",
  "source_sha256": "SHA256_OF_NORMALIZED_SOURCE",
  "selection_rationale": "STRUCTURAL_SELECTION_RULE",
  "pairs": [
    {"pair_id": "pair-001", "source_id": "ID_A", "target_id": "ID_B", "changed_roles": ["agent"]}
  ]
}
```

Use 96 distinct endpoint pairs. Supported contrasts are identical-meaning
surface changes or one exact agent/theme/predicate substitution preserving
the remaining input and LF content. Duplicate/reversed endpoint pairs are
rejected. Output includes the selection report and `_canonical.jsonl` rows.

## Software checks

Test data and expected software outputs are under `tests/fixtures/analysis/`.
They support fast deterministic checks of schemas, scoring, calibration and
resampling. Experiment runs write their measured outputs to their selected run
directories, together with configuration and provenance.
