# Phase 22 Status

## Frozen Framing
- `G1_binding`: bounded hypothesis-final success
- `G2_distributed_geometry`: `Phase 18` family-final negative unless detector truthfulness is validated.
- `Phase 22` validates the detector rather than reopening the real `G2` family.

## Benchmarks
- `benchmark_count = 3`
- `interaction`: `canonical_example_count = 336`, `coverage = 1.0`
- `cross_step_bridge`: `canonical_example_count = 336`, `coverage = 1.0`
- `query_source_relation`: `canonical_example_count = 336`, `coverage = 1.0`

## Panels
- `interaction`: `stage_a_target_count = 8`, `stage_b_target_count = 6`
- `cross_step_bridge`: `stage_a_target_count = 8`, `stage_b_target_count = 6`
- `query_source_relation`: `stage_a_target_count = 8`, `stage_b_target_count = 6`

## Family Validity
- `interaction`: `passed = False`, `stage = stage_b`, `failure_reasons = ['predicate_sanity_h2:seed1', 'argument_h2:seed1', 'argument_h1:seed1', 'argument_h1:not_weaker_than_h2:seed1', 'predicate_sanity_h2:seed2', 'argument_h2:seed2', 'argument_h1:seed2', 'argument_h1:not_weaker_than_h2:seed2']`
- `cross_step_bridge`: `passed = False`, `stage = stage_b`, `failure_reasons = ['predicate_sanity_h2:seed1', 'argument_h2:seed1', 'argument_h1:seed1', 'argument_h1:not_weaker_than_h2:seed1', 'predicate_sanity_h2:seed2', 'argument_h2:seed2', 'argument_h1:seed2', 'argument_h1:not_weaker_than_h2:seed2']`
- `query_source_relation`: `passed = False`, `stage = stage_b`, `failure_reasons = ['predicate_sanity_h2:seed1', 'argument_h2:seed1', 'argument_h1:seed1', 'argument_h1:not_weaker_than_h2:seed1', 'predicate_sanity_h2:seed2', 'argument_h2:seed2', 'argument_h1:seed2', 'argument_h1:not_weaker_than_h2:seed2']`

## Anchorability
- `interaction`: `passed = True`, `usable_anchor_count = 2`
- `cross_step_bridge`: `passed = True`, `usable_anchor_count = 2`
- `query_source_relation`: `passed = True`, `usable_anchor_count = 2`

## Observability
- `interaction`: `passed = False`, `predicate_sanity_passed = True`, `argument_families_passed = False`, `argument_specific_failure = True`
- `cross_step_bridge`: `passed = False`, `predicate_sanity_passed = True`, `argument_families_passed = False`, `argument_specific_failure = True`
- `query_source_relation`: `passed = False`, `predicate_sanity_passed = True`, `argument_families_passed = False`, `argument_specific_failure = True`

## Argument H2
- `interaction`: `passed = False`, `stage = stage_b`, `null_passed_count = 0`
- `cross_step_bridge`: `passed = False`, `stage = stage_b`, `null_passed_count = 0`
- `query_source_relation`: `passed = False`, `stage = stage_b`, `null_passed_count = 0`

## Argument H1
- `interaction`: `positive_passed = False`, `stage = stage_b`, `null_false_positive_rate = 0.0`
- `cross_step_bridge`: `positive_passed = False`, `stage = stage_b`, `null_false_positive_rate = 0.0`
- `query_source_relation`: `positive_passed = False`, `stage = stage_b`, `null_false_positive_rate = 0.0`

## Stage B
- `triggered = True`
- `failed_classes = ['interaction', 'cross_step_bridge', 'query_source_relation']`
- `stage_b_executed = True`

## Real-Family Regression
- `preserved_phase18_t_final = True`
- `executed = False`

## Detector Verdict
- `detectability_validated = False`
- `decision = retain_phase18`
- `failure_reasons = ['interaction:family_construction_failed', 'interaction:observability_failed', 'interaction:argument_specific_observability_failure', 'interaction:argument_h2_failed', 'interaction:argument_h1_failed', 'cross_step_bridge:family_construction_failed', 'cross_step_bridge:observability_failed', 'cross_step_bridge:argument_specific_observability_failure', 'cross_step_bridge:argument_h2_failed', 'cross_step_bridge:argument_h1_failed', 'query_source_relation:family_construction_failed', 'query_source_relation:observability_failed', 'query_source_relation:argument_specific_observability_failure', 'query_source_relation:argument_h2_failed', 'query_source_relation:argument_h1_failed']`
