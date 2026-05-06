from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .metrics import identity_stability, l2_norm, leave_one_out_centroid_accuracy
from .spans import build_probe_targets


def run_candidate_discovery(
    adapter: Any,
    candidates: list[dict[str, Any]],
    layer_indices: list[int],
    max_length: int = 256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    span_records: list[dict[str, Any]] = []
    requested_span_count = 0
    found_probe_ids: set[str] = set()
    namespace_coverage: Counter[str] = Counter()

    for candidate in candidates:
        probes = build_probe_targets(candidate)
        requested_span_count += len(probes)
        traced = adapter.trace_text(
            input_text=str(candidate["input_text"]),
            probes=probes,
            layer_indices=layer_indices,
            max_length=max_length,
        )
        for vector_record in traced["span_vectors"]:
            found_probe_ids.add(str(vector_record["probe_id"]))
            namespace_coverage[str(vector_record["namespace"])] += 1
            span_records.append(
                {
                    "item_id": candidate["item_id"],
                    "benchmark_name": candidate["benchmark_name"],
                    "split": candidate["split"],
                    "generalization_case": candidate["generalization_case"],
                    "stratum": candidate["stratum"],
                    "failure_category": candidate.get("failure_category"),
                    "input_text": candidate["input_text"],
                    "target_form": candidate["target_form"],
                    **vector_record,
                }
            )

    summary = summarize_discovery(
        adapter=adapter,
        candidates=candidates,
        span_records=span_records,
        requested_span_count=requested_span_count,
        found_probe_count=len(found_probe_ids),
        namespace_coverage=dict(sorted(namespace_coverage.items())),
    )
    return span_records, summary


def summarize_discovery(
    adapter: Any,
    candidates: list[dict[str, Any]],
    span_records: list[dict[str, Any]],
    requested_span_count: int,
    found_probe_count: int,
    namespace_coverage: dict[str, int],
) -> dict[str, Any]:
    candidate_counts = Counter(str(candidate["stratum"]) for candidate in candidates)
    layer_buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in span_records:
        layer_buckets[int(record["layer_index"])].append(record)

    layers = {
        str(layer_index): summarize_layer(records)
        for layer_index, records in sorted(layer_buckets.items(), key=lambda item: item[0])
    }

    return {
        "model": {
            "name_or_path": adapter.model_name_or_path,
            "family": adapter.family,
            "device": adapter.device,
            "prompt_style": adapter.prompt_style,
        },
        "candidate_counts": {
            "selected": len(candidates),
            "by_stratum": dict(sorted(candidate_counts.items())),
        },
        "span_coverage": {
            "requested": requested_span_count,
            "found": found_probe_count,
            "coverage_rate": round(found_probe_count / requested_span_count, 6)
            if requested_span_count
            else None,
            "by_namespace": namespace_coverage,
        },
        "layers": layers,
    }


def summarize_layer(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["namespace"])].append(record)

    semantic_role = grouped.get("semantic_role", [])
    surface_position = grouped.get("surface_position", [])
    semantic_summary = leave_one_out_centroid_accuracy(semantic_role)
    surface_summary = leave_one_out_centroid_accuracy(surface_position)
    semantic_by_stratum = {
        stratum: leave_one_out_centroid_accuracy(stratum_records)
        for stratum, stratum_records in _group_by_stratum(semantic_role).items()
    }

    semantic_accuracy = semantic_summary.get("loo_centroid_accuracy")
    surface_accuracy = surface_summary.get("loo_centroid_accuracy")
    control_gap = None
    if semantic_accuracy is not None and surface_accuracy is not None:
        control_gap = round(float(semantic_accuracy) - float(surface_accuracy), 6)

    return {
        "vector_count": len(records),
        "avg_vector_norm": round(sum(l2_norm(list(record["vector"])) for record in records) / len(records), 6)
        if records
        else None,
        "semantic_role": semantic_summary,
        "semantic_role_by_stratum": semantic_by_stratum,
        "surface_position": surface_summary,
        "surface_control_gap": control_gap,
        "filler_identity": identity_stability(grouped.get("filler_identity", [])),
        "predicate_identity": identity_stability(grouped.get("predicate_identity", [])),
    }


def _group_by_stratum(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["stratum"])].append(record)
    return dict(sorted(grouped.items()))
