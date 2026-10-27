# BOUNDED-RS

**A Bounded Evaluation Protocol and Testbed for Representational Systematicity**
Boyu Cao and Wenshuo Wang — NeurIPS 2026 Evaluations and Datasets Track

BOUNDED-RS distinguishes a recoverable internal signal from one that causally
supports systematic generation. Its three layers are a matched semantic-parsing
testbed, family-specific recoverability/interchange/necessity/repair operations,
and evidence-to-claim adjudication into **Unsupported, Present-only, Causal-use,
or Undetermined**. Binding and Geometry denote the declared operationalizations;
the finite Geometry closure basis does not cover every geometric mechanism.

Code and original synthetic data: [MIT](LICENSE). Attribution and external asset
licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Citation: [CITATION.cff](CITATION.cff).
The public artifact is this repository; no anonymous mirror is required.

## Start here

The main testbed is **`data/processed/phase11_systematicity_v2.jsonl`**, already
checked in: **96 complete sets × 7 = 672 items**, generated with **seed 13**.
It has 448 training, 128 development and 96 generalization items, 192 interchange
pairs and 96 repair pairs. Passive relative-clause constructions are held out
from training. The earlier 480-item `v1` and COGS configurations are legacy
experiments; they are not the paper's main testbed.

The data can be inspected and regenerated without a GPU or ML dependencies:

```bash
git clone https://github.com/VinceWangVW/An-Evaluation-Protocol-for-Representational-Systematicity.git
cd An-Evaluation-Protocol-for-Representational-Systematicity
python3 scripts/build_release_data.py --seed 13 --check
python3 scripts/validate_release_data.py
# To rebuild files explicitly:
python3 scripts/build_release_data.py --seed 13
```

[The data card](docs/DATA_CARD.md) describes construction, exact contrasts,
splits, intended use and limitations. [The manifest](data/release/manifest.json)
records SHA-256 hashes. [Croissant metadata](croissant.json) includes the public
download URLs, schema, license, authors and minimal Responsible AI fields.
`data/release/items.csv` is an equivalent flat representation; exact membership
and pairs are in `data/release/matched_sets.jsonl`.

## Installation and implementation checks

Use Python 3.11 or later. Full model runs use PyTorch and public T5 checkpoints.
The deterministic data builder and release analysis tools use the standard
library; the complete test suite also exercises model code.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[test,metadata]'
python3 -m pytest -q
python3 scripts/validate_release_data.py --mlcroissant
make release-check
```

The `metadata` extra installs the official Croissant reader. Its check validates
the metadata and loads all 672 records using the matching local CSV. Public URL
availability is a separate network check. Model runs download the original T5
assets from Hugging Face unless a local checkpoint is supplied. No mirror or
private training-server path is required. [Portability notes](docs/PORTABILITY.md)
describe normalization of path metadata in historical reports.

## Paper workflow and rerun entry points

The main model setup is T5-small, 8 epochs, AdamW learning rate `5e-5`, batch size
8 and greedy decoding. Training is followed by candidate discovery, generation
interventions, validity checks, and adjudication; Geometry family negatives
additionally require observability and exhaustive frozen-basis closure.

| Stage | Implementation / command | Output |
| --- | --- | --- |
| Data | `python3 scripts/build_release_data.py --seed 13` | Canonical JSONL, CSV, matched pairs, manifest, Croissant |
| Train and collect intervention evidence | `seed-plan` with `--execute` below | Fresh checkpoints in `runs/`, behavior/discovery/protocol summaries and raw intermediate records |
| Export decision evidence | `collect-evidence` below | L1 signals, provenance, missing-feature list, L2 trace |
| Score a supplied record | `adjudicate --signals INPUT.json --output OUTPUT.json` | Raw feature transforms, group/state scores, strict gates and final decision |
| Geometry closure | Existing `make family-closure-*` targets | Observability, catalogs, necessity/presence gates and family-final report |
| Detector validation | Existing `make detector-truthfulness-*` and `make detector-competition-*` targets | Natural-model detector validation reports |

For the five-seed implementation run:

```bash
# Writes inspectable commands/configs; no training until --execute is supplied.
python3 scripts/run_release_analysis.py seed-plan \
  --seeds 0 1 2 3 4 --plan-dir runs/release/five_seeds \
  --output runs/release/five_seeds_plan.json
# Execute the generated train -> discovery -> intervention chain:
python3 scripts/run_release_analysis.py seed-plan \
  --seeds 0 1 2 3 4 --plan-dir runs/release/five_seeds \
  --output runs/release/five_seeds_plan.json --execute
python3 scripts/run_release_analysis.py collect-evidence \
  --panel runs/release/five_seeds/binding_panel.json \
  --output runs/release/five_seeds_binding_evidence.json
```

Seeds `0..4` are explicit release defaults, not recovered historical seed IDs.
These fresh runs exercise the implementation; they do not guarantee the paper's
numerical outcomes. Do not turn operation-level Geometry scores into a
family-final claim without running the frozen closure branch. All commands and
input schemas are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and
[ANALYSIS_GUIDE.md](docs/ANALYSIS_GUIDE.md).

## Rebuttal analyses

`python3 scripts/run_release_analysis.py --help` lists the runnable interfaces:

| Analysis | Subcommand | Important input / scope |
| --- | --- | --- |
| Six-block, 729-point sensitivity grid | `sensitivity` | Independent labeled calibration panel and target records; release levels are explicit implementation defaults |
| Independent training seeds | `seed-plan` | Full training/discovery/intervention commands, executed with `--execute` |
| Matched-set bootstrap | `bootstrap` | Measured per-set evidence; resample clusters, not individual items; default 1,000 replicates |
| Engineered truth controls | `truth-controls` | Executes an independent linear planted-signal fixture with blind discovery and separate confirmation |
| Component-rule comparison | `component-rules` | Supplied known-truth units and shared evidence for all three rules |
| T5-base / selected SLOG transfer | `transfer-plan` | Same training/evidence interfaces, explicit model and benchmark |
| Frozen vs. recalibrated transfer rules | `transfer-compare` | Refit boundaries on independent calibration controls only, then compare target states |
| SLOG exact-contrast adapter | `slog-adapt` | Original normalized SLOG records plus a hash-pinned, prespecified pair selector |

Small executable examples:

```bash
python3 scripts/run_release_analysis.py sensitivity \
  --panel configs/release/implementation_fixture_panel.json \
  --output outputs/implementation_sensitivity.json
python3 scripts/run_release_analysis.py truth-controls \
  --output outputs/implementation_truth_controls.json
python3 scripts/run_release_analysis.py transfer-plan \
  --model google-t5/t5-base --seeds 0 1 2 \
  --plan-dir runs/release/t5base --output runs/release/t5base_plan.json
python3 scripts/run_walkthrough.py --output-dir outputs/walkthrough
```

## Evidence provenance and MS-047

This is an **implementation release**, not a claim that missing historical
experiments were reconstructed. Original frozen reports in `reports/` and
`artifacts/` are preserved, with path/endpoint metadata normalized. Newly
executed fixture outputs in `reports/release/implementation_*` verify the
implementation and are labeled as fixtures. Values transcribed from the paper
and rebuttal are separately labeled `paper_reported` in
`reports/release/paper_reported_summary.json`.

The provided source/rebuttal do not contain the historical seven MS-047 inputs,
remaining filler/predicate controls, complete raw L1/L2 records, full historical
grid, 41-unit comparison panel or SLOG pair selector. These are recorded as
unavailable instead of being fabricated. The generator's index 47 has
Caleb/Opal/contact, while the rebuttal's example uses chef/poet/praise, outside
the generator vocabulary. New data aliases use **GEN-000–GEN-095** to avoid
asserting a false historical mapping.

`scripts/run_walkthrough.py` emits the reported MS-047 ledger, seven actual
released index-47 inputs **as a separate example**, and recomputed feature/decision
records. The incomplete reported evidence correctly abstains. A new complete
measured record can be supplied with `--evidence FILE.json`. See
[the reproducibility status table](REPRODUCIBILITY.md) before treating any file
as empirical evidence for a paper claim.
