# Portability and historical path normalization

Run the documented commands from the repository root. Data, configurations,
reports, and generated checkpoints use repository-relative locations. Public
pretrained backbones use Hugging Face identifiers such as `google-t5/t5-small`
and `google-t5/t5-base`; a local fine-tuned checkpoint remains a local checkpoint
and must be trained or supplied before a model audit can run.

No mirror is selected by the released configurations. The Hugging Face libraries
use their public service by default. Users can explicitly select another endpoint
through their own `HF_ENDPOINT` environment variable or an available
`--hf-endpoint` option. The protocol runner does not restore an endpoint from an
old discovery report.

The model adapters and training model automatically select CUDA when available
and otherwise use CPU. CPU execution is functional but full model audits can be
slow. Precision handling avoids CUDA autocast on CPU. The release checks and
analysis over supplied evidence do not require downloading model weights.

## Historical reports

The camera-ready cleanup normalized 450 existing JSON reports. The original
machine-specific project-root prefix was removed from checkpoint paths, leaving
paths such as `runs/phase11_targets/<checkpoint>`. Endpoint metadata identifying
the original machine's mirror was redacted to JSON `null`. A null endpoint in
these historical reports means that the release omits the original endpoint;
it does not establish which service was used for the original run.

This is metadata normalization only. A recursive comparison confirmed that every
numeric and boolean value, including all scientific measurements, seeds, counts,
scores, and gate decisions, was unchanged. The reports retain their original
scientific status and are not fresh execution results. Relative checkpoint paths
are provenance identifiers, not evidence that checkpoint files are distributed.

## Import repairs

The `analysis` package no longer imports unavailable exploratory modules at
initialization. Import a released module explicitly, for example
`analysis.family_closure`. The state-analysis loader imports the existing
`analysis.behavior_analysis` implementation. It still requires the raw linkage
file when calculating an audit row; absent evidence is not replaced by fabricated
records.
