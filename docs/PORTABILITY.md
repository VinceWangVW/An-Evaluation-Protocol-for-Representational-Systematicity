# Portable execution

Run commands from the repository root. Dataset and configuration paths are
repository-relative; model checkpoints can be public Hugging Face identifiers
such as `google-t5/t5-small` or local checkpoint directories. Training writes
checkpoints under `runs/` and experiment plans declare their output locations.

Hugging Face uses its public service by default. An endpoint can be configured
explicitly with `HF_ENDPOINT` or an available `--hf-endpoint` option. Protocol
runners use the current configuration when accessing model assets.

Adapters select CUDA when available and otherwise use CPU. Small execution
checks work on CPU; full pretrained audits benefit from a GPU. CPU precision
handling avoids CUDA autocast. Data generation and analysis of recorded evidence
can run without downloading model weights.

Import the required analysis module directly, for example
`analysis.family_closure`. Audit loaders consume the dataset, behavior,
discovery, intervention and linkage files identified in the experiment plan.
Generated reports retain model, dataset, profile and seed provenance.
