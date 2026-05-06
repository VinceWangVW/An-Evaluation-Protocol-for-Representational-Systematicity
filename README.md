# Anonymous GitHub repo for From Behavioral Systematicity to Causal Use: An Evaluation Protocol for Representational Systematicity

## Overview

This repository contains the code, configuration, and frozen result artifacts for the ED Track submission *From Behavioral Systematicity to Causal Use: An Evaluation Protocol for Representational Systematicity*.

The repository is organized around a small number of reproducible workflow layers:

- `src/`: core analysis, intervention, evaluation, and data-processing modules
- `scripts/run/`: command-line entry points for the main experimental workflows
- `configs/`: frozen experiment and protocol configuration files
- `data/`: processed benchmark inputs and task metadata
- `reports/` and `artifacts/`: frozen outputs used to inspect and reproduce the reported conclusions
- `tests/`: regression and workflow-level validation

The main top-level workflows exposed in this repository are:

- `family closure`: family-level closure for `G2_distributed_geometry`
- `detector truthfulness`: detector-truthfulness validation
- `detector competition`: detector ontology competition

## Repository Layout

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── Makefile
├── src/
│   ├── analysis/
│   ├── data/
│   ├── discovery/
│   ├── eval/
│   ├── hypotheses/
│   ├── interventions/
│   ├── models/
│   └── utils/
├── scripts/run/
├── configs/
├── data/
├── reports/
├── artifacts/
├── docs/
└── tests/
```

## Included Results

The repository ships with the frozen final-stage outputs needed to inspect the current conclusions:

- `reports/FAMILY_CLOSURE_STATUS.md`
- `reports/DETECTOR_TRUTHFULNESS_STATUS.md`
- `reports/DETECTOR_COMPETITION_STATUS.md`
- `reports/family_closure/`
- `reports/detector_truthfulness/`
- `reports/detector_competition/`

The strongest supported result in this codebase remains:

- `G1_binding`: bounded hypothesis-final success
- `G2_distributed_geometry`: family-final negative under the frozen `Phase 18` closure rule

The detector competition does not upgrade the project to ontology-final; instead, it sharpens the detector-side diagnosis by showing that no detector ontology truth-validates under the frozen outer protocol.

## Environment Setup

### 1. Clone the repository

```bash
git clone <REPO_URL>
cd An-Evaluation-Protocol-for-Representational-Systematicity
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Or, equivalently:

```bash
make install
```

## How To Use

### Run tests

```bash
make test
```

### Rebuild detector competition benchmarks

```bash
make detector-competition-build
```

### Train or materialize detector competition families

```bash
make detector-competition-train
```

### Run the full detector ontology competition

```bash
make detector-competition-run
```

### Re-run the family-closure branch

```bash
make family-closure-observability
make family-closure-catalogs
make family-closure-h2
make family-closure-h1
make family-closure-finalize
```

## Notes On Compute

The heavy training and protocol materialization steps were run on a separate GPU host. The included `reports/` and `artifacts/` directories are the frozen outputs from those completed runs.

If you want to reproduce the long-running branches end to end, you should expect to need:

- a recent Python 3.11 environment
- a working local or remote GPU environment
- enough disk space for cached reports and artifacts

## Key Entry Points

- `scripts/run/run_observability_audit.py`
- `scripts/run/build_family_closure_catalogs.py`
- `scripts/run/gate_family_h2.py`
- `scripts/run/gate_direct_h1.py`
- `scripts/run/run_family_closure.py`
- `scripts/run/run_detector_truthfulness.py`
- `scripts/run/run_detector_competition.py`
- `scripts/run/run_discovery.py`
- `scripts/run/run_protocol.py`
- `scripts/run/train_target_families.py`
- `scripts/run/materialize_target_runs.py`

## Documentation

Additional design documents are in `docs/`:

- `FAMILY_CLOSURE_PLAN.md`
- `DETECTOR_TRUTHFULNESS_PLAN.md`
- `DETECTOR_ONTOLOGY_COMPETITION_PLAN.md`
