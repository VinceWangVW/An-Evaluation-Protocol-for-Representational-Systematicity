# Phase 23: Detector Ontology Competition Challenge

`Phase 23` is a detector-truthfulness branch, not a continuation of the paper mainline. It keeps the outer `Phase 18` protocol frozen and runs a competition among three detector ontologies:

- `D1 = argument_locus`
- `D2 = predicate_conditioned_relation`
- `D3 = cross_step_bridge`

Each detector gets a model-native truthfulness benchmark with:

- `predicate_sanity_h2`
- `detector_h2`
- `detector_h1`
- `detector_null`

and, if needed, a detector-local `Stage B` fallback with:

- `detector_h2_stageb`
- `detector_h1_stageb`
- `detector_null_stageb`

The branch only upgrades beyond `Phase 18` if:

1. at least one detector truth-validates, and
2. every truth-validated detector independently agrees on `real G2 = T-supported / family-final`.

Otherwise the strongest paper claim stays at `retain_phase18`.
