# Running BOUNDED-RS

Run commands from the repository root after installing `.[test,metadata]`.
Each experiment records its input hashes, checkpoint, seed, configuration and
decision profile alongside its generated evidence.

## Data

```bash
python3 scripts/build_release_data.py --seed 13
python3 scripts/build_release_data.py --seed 13 --check
python3 scripts/validate_release_data.py --mlcroissant
```

The main dataset contains 96 seven-item sets. Interchange pairs differ in one
role filler; repair pairs share the gold meaning. Cluster identity is
`metadata.counterpart_family_id`. Generalization easy sources intentionally
belong to dev; use the matched-set ID for resampling. See [the data card](docs/DATA_CARD.md).

## Binding checkpoint audit

```bash
python3 scripts/run_matched_audit.py \
  --data data/processed/phase11_systematicity_v2.jsonl \
  --checkpoint PATH_TO_CHECKPOINT --output-dir runs/audit
```

The Binding audit evaluates baseline predictions and complete intervention/control
panels. It writes:

- `raw_operations.jsonl`: per-item and role intervention outputs, gold outputs,
  control identities and measured effects;
- `L1_per_set.jsonl`: the sufficient-evidence vector for each matched set;
- `L2_decisions.jsonl`: feature transformations, group/state scores, gates and decision;
- `summary.json`: aggregate evidence, sample counts, configuration and run identity.

The repair denominator includes all declared repair pairs, and baseline-error
eligibility is recorded separately. Necessity compares target-specific output
degradation with the strongest declared control. Neither an empty subset nor a
zero-valued observation is substituted for another measurement.

## MS-047

```bash
python3 scripts/run_walkthrough.py --output-dir outputs/walkthrough
python3 scripts/run_walkthrough.py --checkpoint PATH_TO_CHECKPOINT \
  --output-dir runs/walkthrough
```

`configs/experiments/ms047.json` specifies predicate, role fillers, alternate
fillers, modifiers and construction. The program builds seven inputs with
gold logical forms and interchange/repair pairs. Checkpoint execution uses
the same matched-set audit and decision function as the main testbed.

## Five seeds and T5-base

```bash
python3 scripts/run_release_analysis.py seed-plan \
  --seeds 0 1 2 3 4 --plan-dir runs/five_seeds \
  --output runs/five_seeds/plan.json --execute
python3 scripts/run_release_analysis.py transfer-plan \
  --model google-t5/t5-base --seeds 0 1 2 --plan-dir runs/t5base \
  --output runs/t5base/plan.json --execute
```

The plans train target checkpoints and independent causal, decoder-scaled and
permuted-target calibration controls, then run family-specific evidence
operations. Default training is 8 epochs with batch size 8, learning rate
5e-5, and greedy decoding. Seeds for calibration and targets are disjoint.
Model interfaces support public pretrained identifiers and local checkpoints.
Use CPU for small execution checks and a suitable GPU for full audits.

To prepare or execute the combined experiment suite:

```bash
python3 scripts/run_release_analysis.py suite-plan \
  --config configs/experiments/analysis_suite.json --plan-dir runs/experiments \
  --output runs/experiments/plan.json
```

Add `--execute` to run the commands. The suite connects training, Binding
matched-set audits, Geometry-specific evidence, sensitivity, per-seed bootstrap,
transfer comparison and the known-truth computational experiments. The suite
configuration also accepts a selected SLOG benchmark.

## Sensitivity and bootstrap

```bash
python3 scripts/run_release_analysis.py sensitivity \
  --panel runs/PANEL.json --grid configs/release/sensitivity_grid.json \
  --output runs/sensitivity.json
python3 scripts/run_release_analysis.py bootstrap \
  --rows runs/audit/L1_per_set.jsonl --expected-sets 96 --replicates 1000 \
  --seed 13 --output runs/bootstrap.json
```

The six three-level parameter blocks generate 729 configurations. Calibration
compatibility requires every labeled calibration control to retain its intended
state; targets do not affect that filter. The report records compatibility and
all target decisions for each configuration.

Bootstrap resamples matched sets with replacement while preserving the paired
feature vector. It recomputes effect means and the paired permutation statistic,
then reruns adjudication. Outputs contain the sampled indices, 95% percentile
intervals, state counts and configuration. Optional Geometry object records are
resampled together under an identical frozen basis.

## Truth controls and component rules

```bash
python3 scripts/run_release_analysis.py truth-controls --seeds 0 1 2 \
  --output runs/truth_controls.json
python3 scripts/run_release_analysis.py component-panel \
  --output runs/component_panel.json
python3 scripts/run_release_analysis.py component-rules \
  --rows KNOWN_TRUTH_ROWS.jsonl --output runs/component_rules.json
```

Engineered computational controls define causal, equally recoverable-unused and
null conditions. Independent discovery and confirmation samples support known
truth labels. The 41-case panel specifies seven causal cases and 34 noncausal
failure modes; each case generates evidence before the three inference rules
are compared. Reports identify the computational model and retain the measured
effects and per-case decisions.

The model-based detector workflows are available through:

```bash
make detector-truthfulness-build
make detector-truthfulness-train
make detector-truthfulness-run
```

## Transfer calibration and SLOG

```bash
python3 scripts/run_release_analysis.py transfer-compare \
  --panel CALIBRATION_AND_TARGET_PANEL.json --output runs/transfer_comparison.json
python3 scripts/run_release_analysis.py slog-adapt \
  --source NORMALIZED_SLOG.jsonl --selection FROZEN_SELECTION.json \
  --expected-pairs 96 --output runs/slog_selection.json
cp runs/slog_selection_canonical.jsonl data/processed/slog_selected.jsonl
python3 scripts/run_release_analysis.py transfer-plan \
  --model google-t5/t5-small --benchmark slog_selected --seeds 0 1 2 \
  --plan-dir runs/slog --output runs/slog/plan.json --execute
```

Transfer comparison refits boundaries using independent calibration controls
and compares frozen/refitted rules on the same targets. Target values do not
enter calibration. SLOG selection pins the normalized source hash, upstream
revision, original pair IDs and structural rationale. Exact contrasts preserve
the remaining input and logical-form content; original training and dev rows
are retained for model training and controls.

## Geometry closure

```bash
make family-closure-observability
make family-closure-catalogs
make family-closure-h2
make family-closure-h1
make family-closure-finalize
```

Closure uses a frozen object basis, sealed step observability, discovery,
object-specific intervention evidence and direct/slice-local presence routes.
A stronger family decision requires complete coverage of the declared basis.

## Verification

```bash
python3 -m pytest -q
make release-check
```

[The analysis guide](docs/ANALYSIS_GUIDE.md) defines schemas and the scoring
formulas. [Validation](docs/VALIDATION.md) describes the test environment.
