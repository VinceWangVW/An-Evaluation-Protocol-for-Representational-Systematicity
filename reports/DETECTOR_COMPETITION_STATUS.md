# Phase 23 Status

## Frozen Framing
- `G1_binding`: bounded hypothesis-final success
- `G2_distributed_geometry`: `Phase 18` family-final negative under the frozen closure rule.
- `Phase 23` is a separate detector-ontology competition branch.

## Benchmarks
- `benchmark_count = 3`
- `argument_locus`: `canonical_example_count = 336`, `coverage = 1.0`
- `predicate_conditioned_relation`: `canonical_example_count = 336`, `coverage = 1.0`
- `cross_step_bridge`: `canonical_example_count = 336`, `coverage = 1.0`

## Panels
- `argument_locus`: `stage_a_target_count = 8`, `stage_b_target_count = 6`
- `predicate_conditioned_relation`: `stage_a_target_count = 8`, `stage_b_target_count = 6`
- `cross_step_bridge`: `stage_a_target_count = 8`, `stage_b_target_count = 6`

## Execution
- Full remote `Phase 23` competition completed with `--materialize-max-parallel 4`
- Observed GPU utilization during the main protocol/materialization window was approximately `82-90%`
- `Stage A` and `Stage B` both executed before the final competition verdict was written

## Family Validity
- `argument_locus`: `passed = False`, `stage = stage_b`, `failure_reasons = ['predicate_sanity_h2:seed1', 'detector_h2:seed1', 'detector_h1:seed1', 'detector_h1:not_weaker_than_h2:seed1', 'predicate_sanity_h2:seed2', 'detector_h2:seed2', 'detector_h1:seed2', 'detector_h1:not_weaker_than_h2:seed2']`
- `predicate_conditioned_relation`: `passed = False`, `stage = stage_b`, `failure_reasons = ['predicate_sanity_h2:seed1', 'detector_h2:seed1', 'detector_h1:seed1', 'detector_h1:not_weaker_than_h2:seed1', 'predicate_sanity_h2:seed2', 'detector_h2:seed2', 'detector_h1:seed2', 'detector_h1:not_weaker_than_h2:seed2']`
- `cross_step_bridge`: `passed = False`, `stage = stage_b`, `failure_reasons = ['predicate_sanity_h2:seed1', 'detector_h2:seed1', 'detector_h1:seed1', 'detector_h1:not_weaker_than_h2:seed1', 'predicate_sanity_h2:seed2', 'detector_h2:seed2', 'detector_h1:seed2', 'detector_h1:not_weaker_than_h2:seed2']`

## Anchorability
- `argument_locus`: `passed = True`, `usable_anchor_count = 2`
- `predicate_conditioned_relation`: `passed = True`, `usable_anchor_count = 2`
- `cross_step_bridge`: `passed = True`, `usable_anchor_count = 2`

## Observability
- `argument_locus`: `passed = False`, `predicate_sanity_passed = True`, `detector_families_passed = False`, `detector_specific_failure = True`
- `predicate_conditioned_relation`: `passed = True`, `predicate_sanity_passed = True`, `detector_families_passed = True`, `detector_specific_failure = False`
- `cross_step_bridge`: `passed = False`, `predicate_sanity_passed = True`, `detector_families_passed = False`, `detector_specific_failure = True`

## Detector H2
- `argument_locus`: `passed = False`, `stage = stage_b`, `null_passed_count = 0`
- `predicate_conditioned_relation`: `passed = False`, `stage = stage_b`, `null_passed_count = 1`
- `cross_step_bridge`: `passed = False`, `stage = stage_b`, `null_passed_count = 0`

## Detector H1
- `argument_locus`: `positive_passed = False`, `stage = stage_b`, `null_false_positive_rate = 0.0`
- `predicate_conditioned_relation`: `positive_passed = False`, `stage = stage_b`, `null_false_positive_rate = 0.0`
- `cross_step_bridge`: `positive_passed = False`, `stage = stage_b`, `null_false_positive_rate = 0.0`

## Stage B
- `triggered = True`
- `failed_detectors = ['argument_locus', 'predicate_conditioned_relation', 'cross_step_bridge']`
- `stage_b_executed = True`

## Real-Family Regression
- `executed = False`
- `all_validated_detectors_agree_negative = False`

## Competition Verdict
- `decision = retain_phase18`
- `strongest_claim_decision = retain_phase18`
- `validated_detectors = []`
