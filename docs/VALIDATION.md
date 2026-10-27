# Release validation

Validated on 2026-10-27 with Python 3.13.14 on macOS arm64, PyTorch 2.14.1,
Transformers 4.57.6, pytest 9.1.1 and mlcroissant 1.1.1. The package supports
Python 3.11 or later.

```bash
python3 -m pip install -e '.[test,metadata]'
python3 -m pip check
python3 -m pytest -q
make release-check
git diff --check
```

Validation covers:

- exact seed-13 regeneration, 96 complete matched sets, 672 items and split counts;
- 192 interchange pairs, 96 repair pairs and logical-form invariants;
- Croissant metadata parsing and loading of every CSV record;
- local tiny-T5 training, checkpoint reload, generation and targeted intervention;
- complete intervention/control bookkeeping and separate repair denominators;
- one scoring function for audit, walkthrough and statistical analyses;
- strict gates, zero-valued observations and evidence coverage;
- target-independent calibration and exhaustive sensitivity enumeration;
- paired matched-set resampling and frozen-basis closure;
- known-truth computational controls and component-rule comparisons;
- source-pinned SLOG contrasts and public path portability.

Software inputs and expected outputs live in `tests/fixtures/`. The model
execution checks create local tokenizers and tiny checkpoints, so they run
without network access. Full pretrained-model experiments use the training,
audit and analysis commands documented in [Running BOUNDED-RS](../REPRODUCIBILITY.md).

Retain the run configuration, checkpoint identity, input hashes, random seeds,
L1 evidence and L2 decisions with each experiment. When public data files change,
regenerate the manifest and Croissant checksums before publication.
