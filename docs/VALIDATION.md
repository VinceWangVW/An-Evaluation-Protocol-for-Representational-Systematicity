# Camera-ready implementation validation

Verified on **2026-10-27**. This record concerns implementation and artifact
integrity, not new pretrained-model scientific results.

## Environment

- Python 3.13.14 on macOS arm64, CPU execution.
- PyTorch 2.14.1; Transformers 4.57.6; sentencepiece 0.2.2.
- pytest 9.1.1; mlcroissant 1.1.1.
- Editable package installation: `python3 -m pip install -e '.[test,metadata]'`.
- `python3 -m pip check`: no broken requirements.

These are the versions used for this check, not a historical experiment lockfile.
The package declares Python >=3.11; other supported environments should run the
same checks before a long audit.

## Executed checks

```bash
python3 -m pytest -q
make release-check PYTHON=python3
git diff --check
```

The complete suite passed **46 tests**. The release target passed deterministic
data checking, ordinary validation, official Croissant validation and loading,
the provenance-aware walkthrough, sensitivity fixture, truth-control fixture,
and the full test suite. All 15 original model/closure/detector entry points also
accepted `--help` after import repairs.

Dataset regeneration is byte-identical to the existing canonical `v2` JSONL:
672 items, 96 complete sets, 448/128/96 item splits, 2,016 validated template
candidates, 192 interchange pairs and 96 repair pairs (68 surface-distinct).
The official Croissant reader loaded every one of the 672 local CSV records.
Hashes and all exact pair/split invariants passed.

The offline model test created and trained a real tiny T5 with locally generated
tokenizer/checkpoint files. It verified finite loss and a parameter update,
checkpoint reload, greedy generation, hidden-state extraction, targeted encoder
ablation reaching the following block, and restoration/hook cleanup. No network
download was needed. It also checks forwarding of all relation-barrier options.

Portability checks cover released source/configs/reports and import/CLI behavior.
An exact recursive comparison against the pre-release Git HEAD confirmed that
the 450 normalized historical reports changed only checkpoint-root strings and
endpoint metadata. Scientific values, counts, seeds, booleans and decisions were
preserved. A separate linkage-loader regression verifies both the emitted M5
field name and backward compatibility without treating observed zero as missing.

## Executed analysis fixtures and plans

The generated reports in `reports/release/` retain explicit fixture provenance:

- `implementation_sensitivity.json`: all 729 new implementation-grid settings
  evaluated; 729 compatible on the synthetic calibration fixture. This does not
  reproduce the paper's 487-compatible historical grid.
- `implementation_truth_controls.json`: 18 engineered linear cases; 16 correct
  evidential states and two conservative null abstentions. The embedded
  component-rule comparison uses those same known-truth fixture cases. This is
  not the historical matched T5 experiment or the historical 41-unit panel.
- `implementation_bootstrap.json`: 1,000 cluster replicates on eight explicitly
  synthetic sufficient-evidence fixture clusters. It verifies execution and
  resampling logic, not the historical 96-set confidence intervals.
- `implementation_transfer_compare.json`: frozen and control-only refitted
  rules compared on the synthetic panel; tests verify target independence.

Five-seed and three-seed T5-base plans were materialized and parsed. Their
commands include actual training, family-specific discovery/interventions and
evidence export. Missing-checkpoint evidence collection was also executed and
reported missing inputs. Full pretrained runs were not executed for this
implementation release.

The historical MS-047 walkthrough remains incomplete because its original raw
inputs, controls and p-value were not supplied. The release records known
sigmoid transforms and control differences while abstaining on the incomplete
recomputation; it does not associate the published metrics with GEN-047.

## Publication verification

The versioned files and Croissant hashes must agree after a push. Public URL
checks are separate from local reader validation. Preserve the release commit
and `data/release/manifest.json` together when citing or extending an audit.
