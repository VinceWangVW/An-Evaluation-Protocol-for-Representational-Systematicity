# Running BOUNDED-RS

This guide covers data generation, model audits, statistical analyses and report
inspection. All paths are relative to the repository root. Use `python3`, or
activate the virtual environment described in the README.

## Artifact availability

| Component | Included artifacts | Execution / availability |
| --- | --- | --- |
| Canonical seed-13 testbed | 672 rows, 96 matched sets, exact pairs and checksums | Exact regeneration verified with `make data-check` |
| Croissant metadata | Core fields, schema, license, authors and RAI documentation | `make metadata-check` validates and loads all 672 records |
| Training and interventions | Target-panel builder and executable command chain | `seed-plan --execute`; requires model download and compute; trained weights are not bundled |
| Decision scoring | Versioned profile, feature transforms, state formulas and gates | The release profile is explicit; the complete scorer used for the paper's reported state scores is unavailable |
| Sensitivity | Six-block grid and target-independent calibration filtering | Synthetic test fixture included; original grid levels and per-configuration records for the reported 487-compatible result are unavailable |
| Training seeds | Independent-seed orchestration | Default seeds 0–4; original five-seed checkpoints and complete run records are unavailable |
| Cluster bootstrap | Paired cluster resampling, effect intervals and permutation testing | Requires per-set measurements; original 96-set observations are unavailable |
| Truth controls | Planted linear test fixture and model-based detector workflows | Fixture outputs are software checks; original matched T5 truth-control records are unavailable |
| Component comparison | Three rules applied to shared known-truth records | Original 41-unit panel unavailable; aggregate findings indexed as `paper_reported` |
| T5-base transfer | Model-parametric audit, calibration-only refitting and comparison | Original per-seed measurements and recalibrated profile unavailable |
| SLOG transfer | Source-pinned exact-pair adapter and model pipeline | Requires upstream data and a declared selector; original 96-pair selector unavailable |
| MS-047 | Reported metrics, feature transforms and a separate generated example | Original seven inputs and some raw controls/p-values unavailable; GEN-047 is a separate navigation alias |
| Frozen final-stage reports | Experiment summaries with portable path metadata | Scientific values preserved; omitted weights/raw records are not represented by summaries |

Outputs carry explicit provenance: `paper_reported` identifies findings stated
in the manuscript or rebuttal; `executed_implementation_fixture` identifies a
synthetic software test; `missing_inputs`/`unavailable` identifies absent evidence.
A generated command plan is `plan_only` until it executes successfully. These
statuses distinguish report inspection, software validation and model experiments.

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

The decision profile records sigmoid boundary, scale and direction for each
feature. Presence and use features use explicit weighted averages; behavior
has a 4% prior share. State formulas, required metrics and the differences from
legacy frozen configurations are documented in [ANALYSIS_GUIDE.md](docs/ANALYSIS_GUIDE.md).
The strict realism/linkage/coverage/margin comparisons require values above
0.60/0.50/0.80/0.10, respectively. Equality fails the gate. Missing required
features abstain, and zero-valued observations are not treated as missing.

The MS-047 summary contains rounded aggregate metrics. Missing controls and
p-values remain null, so the incomplete decision record returns Undetermined.
The reported Causal-use finding is retained separately in the provenance ledger.
GEN-047 supplies a seven-input example from the released generator; it is not
assigned the paper's MS-047 measurements.

## Model audits

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
are written under `runs/`. Summary-only evidence may lack validity quantities,
in which case the collector reports the gap and abstains.

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

CPU execution is supported. Full pretrained T5 audits normally require a GPU;
runtime depends on hardware, checkpoint size and the number of intervention
conditions.

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

## Geometry closure and detector validation

Run the closure and detector workflows with:

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

These workflows consume the checkpoints and intermediate records referenced by
their configurations. The detector-competition targets provide an additional
diagnostic branch. Detector conclusions remain bounded by the declared object
classes and controls.

## Validation and maintenance

```bash
python3 -m pytest -q
make release-check
```

The automated checks cover deterministic data and exact pairs, integrity and
provenance, missing evidence and gate equality, analysis input contracts,
portability, and existing closure/detector behavior. The model smoke uses an
offline randomly initialized tiny T5 to check training, generation and ablation.
[VALIDATION.md](docs/VALIDATION.md) records what was executed for this release. When data or public paths change, regenerate manifests and
Croissant hashes and rerun the official reader before publication.
