from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from discovery.spans import build_probe_targets

from .probes import accuracy, build_centroid_probes


def run_phase4_protocol(
    adapter: Any,
    candidate_records: list[dict[str, Any]],
    reference_records: list[dict[str, Any]],
    matched_sets: list[dict[str, Any]],
    span_records: list[dict[str, Any]],
    layer_index: int,
    max_length: int = 256,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    probes = build_centroid_probes(span_records, layer_index=layer_index)
    span_index = index_span_records(span_records=span_records, layer_index=layer_index)
    candidate_index = {str(record["item_id"]): record for record in reference_records}
    live_span_cache: dict[str, dict[str, Any]] = {}

    m2_records = run_structure_preserving_interchange(
        adapter=adapter,
        probes=probes,
        candidate_index=candidate_index,
        matched_sets=matched_sets,
        span_index=span_index,
        live_span_cache=live_span_cache,
        layer_index=layer_index,
        max_length=max_length,
    )
    m3_records = run_selective_necessity(
        adapter=adapter,
        probes=probes,
        candidate_records=candidate_records,
        span_index=span_index,
        layer_index=layer_index,
        max_length=max_length,
    )
    m4_records = run_failure_rescue(
        adapter=adapter,
        probes=probes,
        candidate_index=candidate_index,
        matched_sets=matched_sets,
        span_index=span_index,
        live_span_cache=live_span_cache,
        layer_index=layer_index,
        max_length=max_length,
    )
    linkage_records = build_item_level_linkage(m3_records=m3_records, m4_records=m4_records)
    summary = summarize_phase4(
        probes=probes,
        span_records=span_records,
        layer_index=layer_index,
        m2_records=m2_records,
        m3_records=m3_records,
        m4_records=m4_records,
        linkage_records=linkage_records,
    )
    return m2_records, m3_records, m4_records, linkage_records, summary


def run_structure_preserving_interchange(
    adapter: Any,
    probes: dict[str, Any],
    candidate_index: dict[str, dict[str, Any]],
    matched_sets: list[dict[str, Any]],
    span_index: dict[tuple[str, str], dict[str, Any]],
    live_span_cache: dict[str, dict[str, Any]],
    layer_index: int,
    max_length: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for match in matched_sets:
        donor_item = candidate_index[str(match["members"]["pass"])]
        target_item = candidate_index[str(match["members"]["fail"])]
        target_probes = build_probe_targets(target_item)
        donor_span_map = ensure_item_span_map(
            adapter=adapter,
            item=donor_item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        target_span_map = ensure_item_span_map(
            adapter=adapter,
            item=target_item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        for role in ("agent", "theme"):
            donor_filler = donor_span_map.get(f"filler::{role}")
            target_filler = target_span_map.get(f"filler::{role}")
            target_role = target_span_map.get(role)
            if donor_filler is None or target_filler is None or target_role is None:
                continue
            if str(donor_filler["label"]) == str(target_filler["label"]):
                continue

            intervened = adapter.intervene_text(
                input_text=str(target_item["input_text"]),
                probes=target_probes,
                layer_index=layer_index,
                patch_specs=[
                    {
                        "token_indices": list(target_filler["token_indices"]),
                        "replacement_vector": list(donor_filler["vector"]),
                    }
                ],
                max_length=max_length,
            )
            intervened_index = {suffix_from_probe_id(record["probe_id"]): record for record in intervened["span_vectors"]}
            patched_filler = intervened_index.get(f"filler::{role}")
            patched_role = intervened_index.get(role)
            if patched_filler is None or patched_role is None:
                continue

            filler_prediction = probes["filler_identity"].predict(list(patched_filler["vector"]))
            role_prediction = probes["semantic_role"].predict(list(patched_role["vector"]))
            records.append(
                {
                    "module": "M2",
                    "match_level": match["match_level"],
                    "match_key": list(match["match_key"]),
                    "donor_item_id": donor_item["item_id"],
                    "target_item_id": target_item["item_id"],
                    "target_stratum": target_item["stratum"],
                    "role": role,
                    "donor_filler_label": donor_filler["label"],
                    "target_filler_label": target_filler["label"],
                    "baseline_filler_prediction": probes["filler_identity"].predict(list(target_filler["vector"]))["label"],
                    "patched_filler_prediction": filler_prediction["label"],
                    "baseline_role_prediction": probes["semantic_role"].predict(list(target_role["vector"]))["label"],
                    "patched_role_prediction": role_prediction["label"],
                    "filler_changed_to_donor": filler_prediction["label"] == str(donor_filler["label"]),
                    "role_preserved": role_prediction["label"] == role,
                    "joint_success": filler_prediction["label"] == str(donor_filler["label"]) and role_prediction["label"] == role,
                }
            )
    return records


def run_selective_necessity(
    adapter: Any,
    probes: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    span_index: dict[tuple[str, str], dict[str, Any]],
    layer_index: int,
    max_length: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in candidate_records:
        item_id = str(item["item_id"])
        target_probes = build_probe_targets(item)
        predicate_record = span_index.get((item_id, "predicate"))
        if predicate_record is None:
            continue

        for role in ("agent", "theme"):
            filler_record = span_index.get((item_id, f"filler::{role}"))
            role_record = span_index.get((item_id, role))
            if filler_record is None or role_record is None:
                continue

            intervened = adapter.intervene_text(
                input_text=str(item["input_text"]),
                probes=target_probes,
                layer_index=layer_index,
                ablation_specs=[{"token_indices": list(filler_record["token_indices"])}],
                max_length=max_length,
            )
            intervened_index = {suffix_from_probe_id(record["probe_id"]): record for record in intervened["span_vectors"]}
            ablated_role = intervened_index.get(role)
            ablated_filler = intervened_index.get(f"filler::{role}")
            ablated_predicate = intervened_index.get("predicate")
            if ablated_role is None or ablated_filler is None or ablated_predicate is None:
                continue

            baseline_role_correct = probes["semantic_role"].predict(list(role_record["vector"]))["label"] == role
            baseline_filler_correct = (
                probes["filler_identity"].predict(list(filler_record["vector"]))["label"] == str(filler_record["label"])
            )
            baseline_predicate_correct = (
                probes["predicate_identity"].predict(list(predicate_record["vector"]))["label"] == str(predicate_record["label"])
            )
            ablated_role_correct = probes["semantic_role"].predict(list(ablated_role["vector"]))["label"] == role
            ablated_filler_correct = (
                probes["filler_identity"].predict(list(ablated_filler["vector"]))["label"] == str(filler_record["label"])
            )
            ablated_predicate_correct = (
                probes["predicate_identity"].predict(list(ablated_predicate["vector"]))["label"] == str(predicate_record["label"])
            )

            records.append(
                {
                    "module": "M3",
                    "item_id": item_id,
                    "stratum": item["stratum"],
                    "role": role,
                    "baseline_role_correct": baseline_role_correct,
                    "ablated_role_correct": ablated_role_correct,
                    "baseline_filler_correct": baseline_filler_correct,
                    "ablated_filler_correct": ablated_filler_correct,
                    "baseline_predicate_correct": baseline_predicate_correct,
                    "ablated_predicate_correct": ablated_predicate_correct,
                    "semantic_role_drop": int(baseline_role_correct) - int(ablated_role_correct),
                    "filler_identity_drop": int(baseline_filler_correct) - int(ablated_filler_correct),
                    "predicate_control_drop": int(baseline_predicate_correct) - int(ablated_predicate_correct),
                    "selective_impairment": (
                        int(baseline_role_correct) - int(ablated_role_correct)
                    ) - (
                        int(baseline_predicate_correct) - int(ablated_predicate_correct)
                    ),
                }
            )
    return records


def run_failure_rescue(
    adapter: Any,
    probes: dict[str, Any],
    candidate_index: dict[str, dict[str, Any]],
    matched_sets: list[dict[str, Any]],
    span_index: dict[tuple[str, str], dict[str, Any]],
    live_span_cache: dict[str, dict[str, Any]],
    layer_index: int,
    max_length: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for match in matched_sets:
        donor_item = candidate_index[str(match["members"]["pass"])]
        target_item = candidate_index[str(match["members"]["fail"])]
        target_probes = build_probe_targets(target_item)
        donor_span_map = ensure_item_span_map(
            adapter=adapter,
            item=donor_item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        target_span_map = ensure_item_span_map(
            adapter=adapter,
            item=target_item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        for role in ("agent", "theme"):
            donor_role = donor_span_map.get(role)
            target_role = target_span_map.get(role)
            if donor_role is None or target_role is None:
                continue

            baseline_prediction = probes["semantic_role"].predict(list(target_role["vector"]))
            intervened = adapter.intervene_text(
                input_text=str(target_item["input_text"]),
                probes=target_probes,
                layer_index=layer_index,
                patch_specs=[
                    {
                        "token_indices": list(target_role["token_indices"]),
                        "replacement_vector": list(donor_role["vector"]),
                    }
                ],
                max_length=max_length,
            )
            intervened_index = {suffix_from_probe_id(record["probe_id"]): record for record in intervened["span_vectors"]}
            rescued_role = intervened_index.get(role)
            if rescued_role is None:
                continue
            rescued_prediction = probes["semantic_role"].predict(list(rescued_role["vector"]))
            baseline_correct = baseline_prediction["label"] == role
            rescued_correct = rescued_prediction["label"] == role
            records.append(
                {
                    "module": "M4",
                    "donor_item_id": donor_item["item_id"],
                    "target_item_id": target_item["item_id"],
                    "target_failure_category": target_item["failure_category"],
                    "role": role,
                    "baseline_role_prediction": baseline_prediction["label"],
                    "rescued_role_prediction": rescued_prediction["label"],
                    "baseline_correct": baseline_correct,
                    "rescued_correct": rescued_correct,
                    "rescued_from_incorrect": (not baseline_correct) and rescued_correct,
                }
            )
    return records


def build_item_level_linkage(
    m3_records: list[dict[str, Any]],
    m4_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    linkage: dict[tuple[str, str], dict[str, Any]] = {}

    for record in m3_records:
        key = (str(record["item_id"]), str(record["role"]))
        linkage[key] = {
            "item_id": record["item_id"],
            "role": record["role"],
            "stratum": record["stratum"],
            "semantic_role_drop": record["semantic_role_drop"],
            "filler_identity_drop": record["filler_identity_drop"],
            "predicate_control_drop": record["predicate_control_drop"],
            "selective_impairment": record["selective_impairment"],
            "rescued_from_incorrect": None,
        }

    for record in m4_records:
        key = (str(record["target_item_id"]), str(record["role"]))
        linkage.setdefault(
            key,
            {
                "item_id": record["target_item_id"],
                "role": record["role"],
                "stratum": "fail",
                "semantic_role_drop": None,
                "filler_identity_drop": None,
                "predicate_control_drop": None,
                "selective_impairment": None,
                "rescued_from_incorrect": None,
            },
        )
        linkage[key]["rescued_from_incorrect"] = bool(record["rescued_from_incorrect"])

    return list(linkage.values())


def summarize_phase4(
    probes: dict[str, Any],
    span_records: list[dict[str, Any]],
    layer_index: int,
    m2_records: list[dict[str, Any]],
    m3_records: list[dict[str, Any]],
    m4_records: list[dict[str, Any]],
    linkage_records: list[dict[str, Any]],
) -> dict[str, Any]:
    layer_records = [record for record in span_records if int(record["layer_index"]) == layer_index]
    semantic_role_records = [record for record in layer_records if record["namespace"] == "semantic_role"]
    filler_identity_records = [record for record in layer_records if record["namespace"] == "filler_identity"]
    predicate_records = [record for record in layer_records if record["namespace"] == "predicate_identity"]

    semantic_baseline = accuracy(semantic_role_records, probes["semantic_role"])
    filler_baseline = accuracy(filler_identity_records, probes["filler_identity"])
    predicate_baseline = accuracy(predicate_records, probes["predicate_identity"])

    m2_joint = mean(float(record["joint_success"]) for record in m2_records) if m2_records else None
    m3_semantic_drop = mean(float(record["semantic_role_drop"]) for record in m3_records) if m3_records else None
    m3_control_drop = mean(float(record["predicate_control_drop"]) for record in m3_records) if m3_records else None
    m3_selective = mean(float(record["selective_impairment"]) for record in m3_records) if m3_records else None
    m4_rescue = mean(float(record["rescued_from_incorrect"]) for record in m4_records) if m4_records else None

    return {
        "baseline_probe_accuracy": {
            "semantic_role": semantic_baseline,
            "filler_identity": filler_baseline,
            "predicate_identity": predicate_baseline,
        },
        "modules": {
            "M2_structure_preserving_interchange": {
                "count": len(m2_records),
                "joint_success_rate": round(m2_joint, 6) if m2_joint is not None else None,
                "role_preservation_rate": round(
                    mean(float(record["role_preserved"]) for record in m2_records), 6
                )
                if m2_records
                else None,
                "donor_transfer_rate": round(
                    mean(float(record["filler_changed_to_donor"]) for record in m2_records), 6
                )
                if m2_records
                else None,
            },
            "M3_selective_necessity": {
                "count": len(m3_records),
                "semantic_role_drop": round(m3_semantic_drop, 6) if m3_semantic_drop is not None else None,
                "predicate_control_drop": round(m3_control_drop, 6) if m3_control_drop is not None else None,
                "selective_impairment": round(m3_selective, 6) if m3_selective is not None else None,
            },
            "M4_failure_rescue": {
                "count": len(m4_records),
                "rescued_from_incorrect_rate": round(m4_rescue, 6) if m4_rescue is not None else None,
            },
            "M5_item_level_linkage": {
                "count": len(linkage_records),
                "fail_items_with_rescue_signal": sum(
                    1 for record in linkage_records if bool(record.get("rescued_from_incorrect"))
                ),
            },
        },
    }


def ensure_item_span_map(
    adapter: Any,
    item: dict[str, Any],
    span_index: dict[tuple[str, str], dict[str, Any]],
    live_span_cache: dict[str, dict[str, Any]],
    layer_index: int,
    max_length: int,
) -> dict[str, Any]:
    item_id = str(item["item_id"])
    if item_id in live_span_cache:
        return live_span_cache[item_id]

    existing = {
        suffix: record
        for (record_item_id, suffix), record in span_index.items()
        if record_item_id == item_id
    }
    if existing:
        live_span_cache[item_id] = existing
        return existing

    probes = build_probe_targets(item)
    traced = adapter.trace_text(
        input_text=str(item["input_text"]),
        probes=probes,
        layer_indices=[layer_index],
        max_length=max_length,
    )
    traced_map = {suffix_from_probe_id(record["probe_id"]): record for record in traced["span_vectors"]}
    live_span_cache[item_id] = traced_map
    return traced_map


def index_span_records(
    span_records: list[dict[str, Any]],
    layer_index: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed = {}
    for record in span_records:
        if int(record["layer_index"]) != layer_index:
            continue
        indexed[(str(record["item_id"]), suffix_from_probe_id(str(record["probe_id"])))] = record
    return indexed


def suffix_from_probe_id(probe_id: str) -> str:
    _, _, suffix = probe_id.partition("::")
    return suffix
