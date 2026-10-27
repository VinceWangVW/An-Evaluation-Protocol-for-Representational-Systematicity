# BOUNDED-RS

**A Bounded Evaluation Protocol and Testbed for Representational Systematicity**

Boyu Cao and Wenshuo Wang · NeurIPS 2026 Evaluations and Datasets Track

BOUNDED-RS evaluates whether a recoverable internal signal causally supports
systematic generation. The protocol combines matched semantic-parsing examples,
recoverability readouts, interchange, necessity, repair and validity controls,
then assigns one of four states: **Unsupported, Present-only, Causal-use,
Undetermined**. Binding and Geometry specify the representational objects being
tested; Geometry family-level decisions additionally use a declared closure basis.

Code and synthetic data are distributed under [MIT](LICENSE).
See [CITATION.cff](CITATION.cff) and [third-party attribution](THIRD_PARTY_NOTICES.md).

## Installation

Use Python 3.11 or later:

```bash
git clone https://github.com/VinceWangVW/An-Evaluation-Protocol-for-Representational-Systematicity.git
cd An-Evaluation-Protocol-for-Representational-Systematicity
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[test,metadata]'
```

Model runs accept public Hugging Face identifiers or local checkpoints. T5-small
and T5-base are supported through the same encoder–decoder interfaces. The data
builder and statistical analysis tools can run without a GPU.

## Testbed

The primary testbed is [phase11_systematicity_v2.jsonl](data/processed/phase11_systematicity_v2.jsonl):
**96 complete matched sets and 672 items**, generated with **seed 13**. Each set
contains an easy source, a hard target and five controlled semantic contrasts.
There are 448 training, 128 development and 96 generalization items, 192
interchange pairs and 96 repair pairs. Passive relative-clause constructions
are held out from training.

```bash
python3 scripts/build_release_data.py --seed 13
python3 scripts/build_release_data.py --seed 13 --check
python3 scripts/validate_release_data.py --mlcroissant
```

[The data card](docs/DATA_CARD.md) documents construction, split semantics, uses
and limitations. [Croissant metadata](croissant.json), [checksums](data/release/manifest.json),
flat CSV and exact pair membership are included in `data/release/`.

## Run a Binding audit

Training uses 8 epochs, AdamW learning rate `5e-5`, batch size 8 and greedy
decoding. A Binding checkpoint audit records baseline outputs, role/filler readouts,
every intervention and its controls, per-set L1 evidence, and L2 decisions.

```bash
python3 scripts/run_matched_audit.py \
  --data data/processed/phase11_systematicity_v2.jsonl \
  --checkpoint PATH_TO_CHECKPOINT --output-dir runs/audit
```

Repair is evaluated on the declared source–target pairs. Reports include both
the full pair count and the subset with a baseline target error. Interchange,
necessity and repair effects are computed against their measured controls;
all scores and gates are saved with the decision profile.

## MS-047 walkthrough

The [MS-047 design](configs/experiments/ms047.json) defines a seven-item panel
using chef/poet/praise, role and filler substitutions, modifier attachment and
an active-to-passive relative-clause contrast. Its inputs, gold logical forms
and exact pairs are distributed in [data/walkthrough](data/walkthrough).

```bash
# Prepare the panel and execution plan.
python3 scripts/run_walkthrough.py --output-dir outputs/walkthrough
# Execute the same audit and scoring chain on a checkpoint.
python3 scripts/run_walkthrough.py --checkpoint PATH_TO_CHECKPOINT \
  --output-dir runs/walkthrough
```

## Robustness, controls and transfer

| Experiment | Entry point | Main outputs |
| --- | --- | --- |
| Six-block sensitivity grid | `sensitivity` | All 729 configurations, calibration compatibility, target decisions |
| Five training seeds | `seed-plan --seeds 0 1 2 3 4` | Independent checkpoints and complete audit records |
| Matched-set bootstrap | `bootstrap --replicates 1000` | Sampled clusters, effect intervals and decision distributions |
| Engineered truth controls | `truth-controls` | Controlled causal/present/null conditions and measured decisions |
| 41-case component comparison | `component-panel` | Known-truth cases, raw control effects and three rule comparisons |
| T5-base transfer | `transfer-plan --model google-t5/t5-base` | Audits through the same model/evidence interfaces |
| Frozen and recalibrated rules | `transfer-compare` | Control-only calibration and paired target decisions |
| SLOG transfer | `slog-adapt`, then `transfer-plan` | Source-pinned exact contrasts and benchmark audit |
| Geometry closure | `make family-closure-*` | Observability, object catalogs, gates and family-final decision |

```bash
python3 scripts/run_release_analysis.py seed-plan \
  --seeds 0 1 2 3 4 --plan-dir runs/five_seeds \
  --output runs/five_seeds/plan.json
python3 scripts/run_release_analysis.py transfer-plan \
  --model google-t5/t5-base --seeds 0 1 2 --plan-dir runs/t5base \
  --output runs/t5base/plan.json
```

Plans expose the commands and configurations before execution. Add `--execute`
to run them. `make experiment-plan` prepares the experiment suite in
`configs/experiments/analysis_suite.json`. Geometry uses its object-matched
operations and closure workflow. See [Running BOUNDED-RS](REPRODUCIBILITY.md) and the
[analysis guide](docs/ANALYSIS_GUIDE.md) for the full experiment suite, schemas,
scoring formulas and output layout.

## Repository layout and tests

`src/` contains data, model, intervention and decision code. `configs/` declares
experiments and profiles. `reports/` and `artifacts/` contain experiment reports
and intermediate objects; new runs write their outputs under the chosen run
directory. [The report index](reports/index.json) identifies report groups and
preserved reference records by their source and status. Software test inputs
and expected outputs are in `tests/fixtures/`.

```bash
python3 -m pytest -q
make release-check
```

Tests cover exact semantic pairs, data integrity, real CPU T5 execution and
targeted intervention, control bookkeeping, shared scoring, calibration,
cluster resampling and closure. See [validation](docs/VALIDATION.md) for the
verification environment.
