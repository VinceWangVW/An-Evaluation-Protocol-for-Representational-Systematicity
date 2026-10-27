# Implementation reproducibility

This release makes the synthetic testbed and implementation inspectable and
executable. It does not claim to regenerate the historical training results.
All paths below are relative to the repository root. Use `python3`, or activate
the virtual environment described in the README.

## What is reproducible from the release

| Component | Release status | Verification / requirements |
| --- | --- | --- |
| Canonical seed-13 testbed | Exact regeneration verified against the existing 672-row file | `make data-check`; 96 sets, exact pairs and hashes |
| Croissant core and RAI metadata | Official reader validation and full 672-record local loading | `make metadata-check` after installing `.[metadata]` |
| Training and generation interventions | Source, target-panel builder and executable command chain supplied | `seed-plan --execute`; requires public model download and compute |
| Decision scoring | Explicit versioned implementation profile, complete transform and gate records | New profile is not presented as the unavailable historical scorer |
| Sensitivity | Six blocks × three levels, target-independent calibration filter implemented | Executable fixture; historical 487-compatible result is reported-only |
| Five training seeds | Independent seed orchestration implemented | Defaults 0–4 are new defaults; expensive runs not performed for this release |
| Cluster bootstrap | Paired per-set resampling and effect intervals implemented | Requires actual per-set measurements; no resampling of paper aggregate numbers |
| Engineered truth controls | Independent planted linear fixture executed | Verifies implementation logic; does not reproduce the reported T5 truth-control experiment |
| Component comparison | Three inference rules applied to shared known-truth records | Historical 41-unit panel is unavailable; counts are reported-only |
| T5-base transfer | Model-parametric train/evidence plan and calibration-only refitting/comparison implemented | Historical per-seed measurements and historical recalibrated profile unavailable |
| SLOG transfer | Source-pinned exact-pair validation and normalized pipeline input implemented | Original benchmark and explicit selection required; historical 96-pair selector unavailable |
| MS-047 | Reported metrics and unknowns transcribed; actual generated example supplied separately | No claim that GEN-047 is historical MS-047 |
| Historical final-stage artifacts | Preserved measurements with portable path metadata | Included summaries are inspectable; omitted weights/raw records cannot be recovered from summaries |

`paper_reported` means a value stated by the supplied manuscript or rebuttal.
`executed_implementation_fixture` means new synthetic software verification.
`missing_inputs`/`unavailable` means evidence was not supplied. A saved plan is
`plan_only` until its commands actually execute successfully. None of these
statuses is interchangeable with a newly measured scientific result.

## Data and metadata

```bash
python3 scripts/build_release_data.py --seed 13 --check
python3 scripts/validate_release_data.py --mlcroissant
```

The builder also accepts `--output-root /tmp/bounded-rs-data` for isolated
regeneration. It rejects a seed change that silently reuses the versioned
canonical release. See [DATA_CARD.md](docs/DATA_CARD.md) for split and cluster
semantics: generalization easy sources intentionally belong to dev, so dev/gen
are not independent cluster partitions.

## Scoring, missing evidence and walkthrough

```bash
python3 scripts/run_walkthrough.py --output-dir outputs/walkthrough
python3 scripts/run_release_analysis.py adjudicate \
  --signals MY_MEASURED_SIGNALS.json --output outputs/decision.json
```

The release profile records sigmoid boundary, scale and direction for each
feature. Presence and use features use explicit weighted averages; behavior
has a 4% prior share. State formulas, required metrics and the differences from
legacy frozen configurations are documented in [ANALYSIS_GUIDE.md](docs/ANALYSIS_GUIDE.md).
The strict realism/linkage/coverage/margin comparisons require values above
0.60/0.50/0.80/0.10, respectively. Equality fails the gate. Missing required
features abstain, and zero-valued observations are not treated as missing.

The MS-047 source provides only rounded aggregates, not the complete
experimental record. The walkthrough preserves missing controls and p-values
as null. Its new recomputation therefore returns Undetermined, while retaining
the paper's reported Causal-use state in the separate provenance ledger.
This disagreement is an evidence-availability statement, not a new experimental
contradiction of the reported result.

## Fresh model runs

```bash
python3 scripts/run_release_analysis.py seed-plan \
  --seeds 0 1 2 3 4 --plan-dir runs/release/five_seeds \
  --output runs/release/five_seeds_plan.json --execute
python3 scripts/run_release_analysis.py collect-evidence \
  --panel runs/release/five_seeds/binding_panel.json \
  --output runs/release/five_seeds_binding_evidence.json
python3 scripts/run_release_analysis.py collect-evidence \
  --panel runs/release/five_seeds/geometry_panel.json \
  --output runs/release/five_seeds_geometry_evidence.json
```

Plans train each declared seed with 8 epochs, batch size 8, learning rate 5e-5,
greedy decoding, and the canonical `phase11_systematicity_v2` testbed. The
Geometry control transformation uses a feature-mask keep ratio of 0.75.
Each family uses the existing discovery and generation operations. Checkpoints
are fresh outputs under `runs/`; the repository does not distribute trained
weights. Summary-only evidence may lack validity quantities, in which case
the collector reports the gap and abstains.

For T5-base, replace the first command by:

```bash
python3 scripts/run_release_analysis.py transfer-plan \
  --model google-t5/t5-base --seeds 0 1 2 \
  --plan-dir runs/release/t5base --output runs/release/t5base_plan.json --execute
```

Generated plans include independent calibration controls and target evidence.
`transfer-compare --panel MY_CALIBRATION_AND_TARGET_PANEL.json --output
outputs/transfer_comparison.json` compares frozen and calibration-only refitted
profiles. Targets cannot influence the refitted boundaries. The guide documents
the generated panel file and required evidence.

CPU fallback is supported for implementation checks. Full pretrained T5 audits
are substantially more expensive than the checks here and normally require a
GPU. Runtime depends on hardware and checkpoint size; no new full-run timing
or benchmark accuracy is asserted by this release.

## Sensitivity, bootstrap and component analysis

```bash
python3 scripts/run_release_analysis.py sensitivity \
  --panel MY_CALIBRATION_AND_TARGET_PANEL.json \
  --grid configs/release/sensitivity_grid.json --output outputs/sensitivity.json
python3 scripts/run_release_analysis.py bootstrap \
  --rows MY_96_MEASURED_SET_RECORDS.jsonl --replicates 1000 \
  --expected-sets 96 --seed 13 --output outputs/bootstrap.json
python3 scripts/run_release_analysis.py component-rules \
  --rows MY_KNOWN_TRUTH_PANEL.jsonl --output outputs/component_rules.json
python3 scripts/run_release_analysis.py truth-controls \
  --seeds 0 1 2 --output outputs/implementation_truth_controls.json
```

See the guide for schemas. Calibration and target identities must be disjoint;
calibration compatibility cannot use target outcomes. Cluster bootstrap requires
one measured sufficient-evidence record per matched set and recomputes the
paired permutation statistic. A whole-family negative additionally requires
complete frozen-basis evidence and completed direct/slice-local presence routes;
an aggregate effect or a single object cannot establish closure.

## SLOG adapter

```bash
python3 scripts/run_release_analysis.py slog-adapt \
  --source NORMALIZED_UPSTREAM_SLOG.jsonl --selection MY_FROZEN_SELECTION.json \
  --expected-pairs 96 --output outputs/slog_selected.json
```

The selector records the benchmark revision, source-file SHA-256, structural
rationale, original IDs and declared changed roles. The adapter checks those
identities and contrasts, and writes normalized canonical rows alongside the
selection report. Obtain original SLOG data under its own license; no synthetic
replacement is labeled as SLOG. See the guide for the explicit training-input
and exact-counterpart requirements before using the selected benchmark with
`transfer-plan --benchmark slog_selected`.

## Existing Geometry closure and detector branches

The original source and frozen final-stage reports remain available:

```bash
make family-closure-observability
make family-closure-catalogs
make family-closure-h2
make family-closure-h1
make family-closure-finalize
make detector-truthfulness-build
make detector-truthfulness-train
make detector-truthfulness-run
```

These commands may require newly trained checkpoints and generated intermediate
records referenced by their configurations. Portable `runs/...` references are
output locations, not bundled model files. The detector-competition targets
provide an additional legacy diagnostic branch. Their failure to validate a
detector must not be read as ontology-level absence of Geometry.

## Validation and maintenance

```bash
python3 -m pytest -q
make release-check
```

The automated checks cover deterministic data and exact pairs, integrity and
provenance, missing evidence and gate equality, analysis input contracts,
portability, and existing closure/detector behavior. The model smoke uses an
offline randomly initialized tiny T5 only to exercise code; it establishes no
scientific result. [VALIDATION.md](docs/VALIDATION.md) records what was executed
for this release. When data or public paths change, regenerate manifests and
Croissant hashes and rerun the official reader before publication.
