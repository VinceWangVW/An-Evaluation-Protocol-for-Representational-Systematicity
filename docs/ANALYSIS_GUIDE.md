# Analysis guide

BOUNDED-RS provides analyses for parameter sensitivity, training seeds,
matched-set uncertainty, truth controls, component rules and transfer. See
[artifact availability](../REPRODUCIBILITY.md#artifact-availability) for the
released reports and the input records required by each workflow.

The commands below run from the repository root with Python 3.11 or later.
Statistical analyses and the small engineered fixture use only the standard
library. Model training needs the main project dependencies, public T5 downloads,
and suitable compute. The five-seed and transfer commands produce plans by default;
`--execute` actually trains models and runs discovery and generation interventions.

## Decision profile and trace

`src/release_analysis.py` defines the versioned
`camera-ready-implementation-v1` decision profile. The sigmoid transforms,
feature weights, state formulas and stop rules are specified below. Legacy
scorers and their frozen configurations remain available for the corresponding
reports. The complete scorer underlying the paper's reported state scores is
unavailable, so numerical results from different profiles should be tracked
separately.

Required signal keys are:

```text
b0_gen_clause_f1, m1_semantic_role_accuracy, m1_surface_control_gap,
m2_advantage_mean, m2_permutation_p_value,
m3_selective_impairment_clause_f1, m4_advantage_mean,
realism_pass_rate, linkage_pass_rate, evidence_coverage,
audit_ready, state_anchor_ready
```

The last two values are booleans. Other values are finite scalars; probability,
accuracy and rate values lie in [0,1]. A zero is observed evidence. A missing or
null required value yields `Undetermined` and an explicit missing-feature list.

Each feature uses `sigmoid(direction * (value - boundary) / scale)`. Presence is
0.60 role readout + 0.40 surface gap; use is 0.60 interchange advantage + 0.40
permutation support. Let these group scores be P and U and behavior be B. With
b=0.04, the three evidential state scores are:

```text
Causal-use:   b*B     + (1-b)*(P + U)/2
Present-only: b*B     + (1-b)*(P + 1-U)/2
Unsupported:  b*(1-B) + (1-b)*(2-P-U)/2
```

The diagnostic Undetermined score is `1 - max(the three state scores)`; abstention
is governed by the actual stop rules, not competition with this diagnostic score.
The top evidential state must lead the next by **more than** 0.10, coverage must
exceed 0.80 and both readiness flags must be true. Causal-use additionally requires
P>0.5, U>0.5, realism>0.60, linkage>0.50, selective necessity>0.05 and repair
advantage>0.245002. Present-only requires P>0.5 and U<0.5; Unsupported requires
P<0.5 and U<0.5. Conflicting evidence yields Undetermined.

```bash
python3 scripts/run_release_analysis.py adjudicate \
  --signals measured_signals.json --output runs/release/decision.json
```

The output contains raw features, every sigmoid transform, group scores, state
scores, all gate decisions and the decision margin. Missing required
measurements yield an explicit abstention.

## Sensitivity grid

`configs/release/sensitivity_grid.json` declares a six-block grid with
three levels each: boundaries, scales, mixture weights, validity/abstention gates,
causal-effect gates and Geometry closure thresholds. Grid provenance is retained
in the output alongside the full configuration-by-configuration decisions.

The panel JSON has disjoint `calibration` and `targets` arrays. Each row contains
`id` and `signals`; calibration additionally needs an `expected_state` of
Unsupported, Present-only or Causal-use. A configuration is compatible only when
every intended calibration state is preserved. Target evidence never determines
compatibility. Optional `closure` records are rechecked for every compatible
configuration using the same trial profile.

```bash
python3 scripts/run_release_analysis.py sensitivity \
  --panel configs/release/implementation_fixture_panel.json \
  --output runs/release/implementation_sensitivity.json
```

The checked-in synthetic test fixture and its generated report exercise all 729
configurations.
To audit measured data, replace the panel file without changing the already
declared grid, or supply another independently prespecified grid with `--grid`.

## Five independent seeds and T5-base

```bash
python3 scripts/run_release_analysis.py seed-plan \
  --plan-dir runs/release/five_seeds \
  --output runs/release/five_seeds_plan.json
python3 scripts/run_release_analysis.py transfer-plan \
  --model google-t5/t5-base --seeds 0 1 2 \
  --plan-dir runs/release/t5base \
  --output runs/release/t5base_plan.json
```

Append `--execute` to perform the model runs. The default T5-small seeds are
0–4. For each family and seed the generated panel contains a target checkpoint and independent
calibration controls with seed+1000: a fully trained causal control, an
inference-only 0.25 decoder-cross-attention control reusing that control checkpoint,
and a permuted-target unsupported control. These regimes declare intended states;
their measured evidence determines the resulting decisions.

Both model sizes use 8 epochs, AdamW learning rate 5e-5, batch size 8 and greedy
decoding. The relative-depth mapping uses the final encoder layer (`layers: [-1]`)
and the family-operation interfaces. Head and rank defaults are specified by
the family operations. Geometry controls apply the declared 0.75 feature mask.

Each plan executes actual training, then separate Binding and Geometry discovery
and generation-intervention runners, then exports evidence:

```bash
python3 scripts/run_release_analysis.py collect-evidence \
  --panel runs/release/t5base/binding_panel.json \
  --output runs/release/t5base/binding_evidence.json
python3 scripts/run_release_analysis.py transfer-compare \
  --panel runs/release/t5base/binding_evidence.json \
  --output runs/release/t5base/binding_frozen_vs_recalibrated.json
```

`transfer-compare` applies the frozen profile, then recalibrates **feature
boundaries only** using labeled calibration controls and reruns the same targets.
Boundary values are midpoints of the low/high control-group feature means.
Scales, weights, validity gates and state semantics remain frozen. Target IDs must
be disjoint from calibration IDs. This control-only calibration rule is fixed
before evaluating target states.
Repeat with the Geometry panel for operation-level states; a family-final negative
also requires complete frozen object closure.

The collector reuses `hypotheses.audit` to derive existing L1 signals and reads raw
M2/M4 realism and M5 item-role linkage records when available. Its
linkage rate is the fraction of observed intended item-role entries with positive
selective necessity or positive repair advantage. Its coverage is the fraction of
required numeric slots actually measured. Missing raw realism/linkage records
remain missing and force abstention.

## Matched-set bootstrap and frozen closure

```bash
python3 scripts/run_release_analysis.py bootstrap --rows measured_per_set.jsonl \
  --replicates 1000 --seed 13 --expected-sets 96 \
  --output runs/release/bootstrap.json
```

Each JSONL row contains a unique `matched_set_id` and `signals` with **per-set
aggregate measurements** in the required schema. Supply raw per-set signed
interchange advantages, not rounded panel averages. The algorithm resamples 96
clusters with replacement, preserves the entire paired evidence vector of each
set, recomputes feature means and a one-sided paired sign-permutation p-value,
then reruns adjudication. It never averages p-values. Boolean readiness is combined
conservatively by conjunction. Percentile 95% intervals, all sampled indices,
replicate states and RNG seed are returned. `--permutations` defaults to 199;
increase it for a finer Monte Carlo p-value resolution. Record the resampling
seed and permutation count with each analysis.

For Geometry resampling, every cluster additionally needs `closure_objects` and
identical `closure_context`. Each object requires `object_id`, `object_class`,
`step`, `template`, `discovery_gap`, `necessity_gap`, `null_effect`,
`realism_pass_rate`, and complete object-specific `signals` for a positive result.
The context contains strict boolean `sealed`, `complete`,
`present_route_complete`, `present_route_found`, plus `frozen_basis`: the exact
prespecified list of object IDs/classes/steps/templates. Duplicate objects,
changing descriptors and nonfinite metrics fail validation. Missing frozen basis
coverage or incomplete evidence yields Undetermined.

Closure accepts a class only after the declared survivor count and step/template
coverage conditions pass. A surviving necessity candidate must still pass the
complete object-specific adjudication chain before it can license Causal-use.
An unresolved candidate prevents Unsupported. A family-final negative additionally
requires completed direct and slice-local Present-only routes, encoded explicitly
in the context after those tests have run. The closure function consumes those
completed operation results.

## Truth controls and component rules

```bash
python3 scripts/run_release_analysis.py truth-controls --seeds 0 1 2 \
  --output runs/release/implementation_truth_controls.json
python3 scripts/run_release_analysis.py component-rules --rows known_truth_units.jsonl \
  --output runs/release/component_rules.json
```

The first command executes a small known-truth computational fixture: a role-like
scalar or a distributed linear bridge carries a binary variable, while the
generation rule either uses that signal, uses a separate route, or has matched
null content. Discovery scans 12 candidates; independent confirmation measures
recoverability and swap, removal and repair effects. A paired sign-permutation
test operates on the measured intervention effects. This linear fixture checks
the analysis pipeline; natural-model detector truthfulness requires the
model-based workflows below.

The repository also retains its model-based truthfulness infrastructure under
`scripts/run/build_truthfulness_benchmarks.py`,
`scripts/run/train_truthfulness_families.py` and
`scripts/run/run_detector_truthfulness.py`; see the corresponding experiment
configuration and detector-truthfulness documentation. It is distinct from this
linear test fixture.

For component rules each input row has `id`, complete `signals`, `truth` equal to
`causal` or `noncausal`, and a nonempty `truth_provenance` explaining the known
causal construction. All rules see the same rows: recoverability licenses causal
use if P>0.5; ungated intervention if U>0.5; full adjudication uses the complete
profile. Reports include every per-unit prediction, numerator, denominator and
abstention.

## Prespecified SLOG adapter

SLOG is an external MIT-licensed benchmark, not the in-repository synthetic
generator. Obtain the original SLOG distribution and preserve its revision and
license. Normalize its original rows into JSONL with `item_id`, `input_text`,
`target_form`, `split`, and `role_signature`; role metadata must match the parser's
extraction from the original logical forms. Include original training and
development-control records, not only selected evaluation pairs.

Freeze a selection JSON **before looking at model outcomes**:

```json
{
  "source_revision": "an exact upstream revision",
  "source_sha256": "SHA256 of the normalized source JSONL bytes",
  "selection_rationale": "a structural rule chosen before outcomes",
  "pairs": [
    {"pair_id": "pair-001", "source_id": "upstream-id-a", "target_id": "upstream-id-b", "changed_roles": ["agent"]}
  ]
}
```

Supply 96 distinct pairs for the full command. Reversals/duplicate endpoint pairs
do not count as independent pairs. Selection endpoints must be original held-out
records. Supported contrasts are surface-only (identical canonical logical forms)
or exactly one agent/theme/predicate substitution that preserves all remaining
input and LF content. Single-role substitutions require one unambiguous occurrence
of the changed filler in each input and LF. This intentionally narrow adapter
rejects cases it cannot establish as exact contrasts.

```bash
python3 scripts/run_release_analysis.py slog-adapt \
  --source data/raw/slog/normalized.jsonl --selection slog_selection.json \
  --output runs/release/slog_selection_report.json
cp runs/release/slog_selection_report_canonical.jsonl data/processed/slog_selected.jsonl
python3 scripts/run_release_analysis.py transfer-plan \
  --model google-t5/t5-small --benchmark slog_selected --seeds 0 1 2 \
  --plan-dir runs/release/slog --output runs/release/slog_plan.json
```

Append `--execute` to train and run the full evidence chain. The adapter verifies
the source SHA256, preserves original training/dev rows and selected held-out
rows, adds source/pair/counterpart provenance, and derives canonical surface
features used by discovery. Preserve the selected pair IDs, structural rationale
and source revision with each transfer experiment. The original 96-pair selection
is not bundled; see the artifact availability table.

## Verification

```bash
python3 -m pytest tests/test_release_analysis.py -q
```

Tests check strict gates, unavailable evidence, target-independent compatibility
and recalibration, deterministic cluster resampling, complete frozen closure,
truth provenance, executable family plans and exact SLOG contrasts. Synthetic
test reports are stored separately from the paper-reported ledger.
