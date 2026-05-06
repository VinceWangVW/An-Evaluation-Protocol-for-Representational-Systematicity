from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

CORE_ROLES = ("agent", "theme")
ALL_ROLE_KEYS = ("agent", "theme", "recipient")
CONTROL_STOPWORDS = {
    "a",
    "an",
    "and",
    "beside",
    "by",
    "did",
    "in",
    "into",
    "of",
    "on",
    "the",
    "to",
    "was",
    "were",
    "who",
}

GEOMETRY_PROTOCOL_FAMILY = "geometry"
BINDING_PROTOCOL_FAMILY = "binding"
GEOMETRY_ROUTING_UNIT_CLASS = "routing"
GEOMETRY_SUBSPACE_UNIT_CLASS = "subspace"
GEOMETRY_TRAJECTORY_UNIT_CLASS = "trajectory"
GEOMETRY_STORE_OBJECT_CLASS = "store"
GEOMETRY_ROUTE_OBJECT_CLASS = "route"
GEOMETRY_READ_OBJECT_CLASS = "read"
GEOMETRY_INTERACTION_OBJECT_CLASS = "interaction"
GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS = "cross_step_bridge"
GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS = "query_source_relation"
GEOMETRY_PLANT_KIND_OFF = "off"
GEOMETRY_PLANT_KIND_H2_POSITIVE = "h2_positive"
GEOMETRY_PLANT_KIND_H1_POSITIVE = "h1_positive"
GEOMETRY_PLANT_KIND_NULL_DECOY = "null_decoy"
GEOMETRY_RELATION_KEEP_RATIO = 0.35
GEOMETRY_RESCUE_RELATION_SCALE = 1.75
GEOMETRY_NECESSITY_RELATION_SCALE = 0.0
GEOMETRY_SUBSPACE_KEEP_RATIO = 0.35
GEOMETRY_GENERIC_CONTROL_KEEP_RATIO = 0.35
GEOMETRY_GENERIC_DECODER_GAIN_SCALE = 1.75
GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO = 0.5
GEOMETRY_TRAJECTORY_RESCUE_SCALE = 1.25
GEOMETRY_TRAJECTORY_NOISE_STD = 0.25


def ensure_item_span_map(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .analyze import ensure_item_span_map as _ensure_item_span_map

    return _ensure_item_span_map(*args, **kwargs)


def index_span_records(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .analyze import index_span_records as _index_span_records

    return _index_span_records(*args, **kwargs)


def score_prediction(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from eval.metrics import score_prediction as _score_prediction

    return _score_prediction(*args, **kwargs)


def infer_operationalization_family(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if "geometry" in normalized:
        return GEOMETRY_PROTOCOL_FAMILY
    return BINDING_PROTOCOL_FAMILY


def infer_geometry_unit_class(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "auto", GEOMETRY_ROUTING_UNIT_CLASS}:
        return GEOMETRY_ROUTING_UNIT_CLASS
    if "subspace" in normalized:
        return GEOMETRY_SUBSPACE_UNIT_CLASS
    if "trajectory" in normalized:
        return GEOMETRY_TRAJECTORY_UNIT_CLASS
    if "routing" in normalized or "path" in normalized:
        return GEOMETRY_ROUTING_UNIT_CLASS
    raise ValueError(f"Unknown geometry unit class: {value}")


def geometry_operationalization_label(unit_class: str) -> str:
    return f"geometry_{infer_geometry_unit_class(unit_class)}"


def infer_geometry_object_class(value: str | None) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "auto", "none", "null"}:
        return None
    if normalized in {
        GEOMETRY_STORE_OBJECT_CLASS,
        GEOMETRY_ROUTE_OBJECT_CLASS,
        GEOMETRY_READ_OBJECT_CLASS,
        GEOMETRY_INTERACTION_OBJECT_CLASS,
        GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS,
        GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS,
    }:
        return normalized
    if "bridge" in normalized:
        return GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS
    if "query" in normalized or "relation" in normalized:
        return GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS
    if "interaction" in normalized or "event" in normalized:
        return GEOMETRY_INTERACTION_OBJECT_CLASS
    if "store" in normalized:
        return GEOMETRY_STORE_OBJECT_CLASS
    if "route" in normalized or "routing" in normalized or "path" in normalized:
        return GEOMETRY_ROUTE_OBJECT_CLASS
    if "read" in normalized or "readout" in normalized:
        return GEOMETRY_READ_OBJECT_CLASS
    raise ValueError(f"Unknown geometry object class: {value}")


def infer_geometry_plant_kind(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"", GEOMETRY_PLANT_KIND_OFF}:
        return GEOMETRY_PLANT_KIND_OFF
    if normalized in {
        GEOMETRY_PLANT_KIND_H2_POSITIVE,
        GEOMETRY_PLANT_KIND_H1_POSITIVE,
        GEOMETRY_PLANT_KIND_NULL_DECOY,
    }:
        return normalized
    raise ValueError(f"Unknown geometry plant kind: {value}")


def parse_geometry_object_spec(
    geometry_object_class: str | None,
    geometry_object_spec: str | None,
) -> dict[str, Any] | None:
    object_class = infer_geometry_object_class(geometry_object_class)
    if object_class is None:
        return None
    raw = str(geometry_object_spec or "").strip()
    if not raw:
        return {
            "object_class": object_class,
            "object_id": f"{object_class}_default",
        }
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"object_id": raw}
    if not isinstance(parsed, dict):
        parsed = {"object_id": str(parsed)}
    spec = dict(parsed)
    spec["object_class"] = object_class
    spec["object_id"] = str(spec.get("object_id") or raw or f"{object_class}_default")
    return spec


def geometry_object_id(
    geometry_object_class: str | None,
    geometry_object_spec: str | None,
) -> str | None:
    parsed = parse_geometry_object_spec(geometry_object_class, geometry_object_spec)
    if parsed is None:
        return None
    return str(parsed["object_id"])


PLANT_SPEC_REQUIRED_KEYS: tuple[str, ...] = (
    "plant_id",
    "object_class",
    "object_spec",
    "base_run",
    "head_scale",
    "basis_scale",
    "store_mode",
    "source_mode",
    "intervention_scope",
    "default_attenuation",
)


def parse_geometry_plant_spec(
    geometry_plant_kind: str | None,
    geometry_plant_spec: str | None,
) -> dict[str, Any] | None:
    plant_kind = infer_geometry_plant_kind(geometry_plant_kind)
    if plant_kind == GEOMETRY_PLANT_KIND_OFF:
        return None
    raw = str(geometry_plant_spec or "").strip()
    if not raw:
        return {
            "plant_kind": plant_kind,
            "plant_id": f"{plant_kind}_default",
        }
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"plant_id": raw}
    if not isinstance(parsed, dict):
        parsed = {"plant_id": str(parsed)}
    spec = dict(parsed)
    spec["plant_kind"] = plant_kind
    spec["plant_id"] = str(spec.get("plant_id") or raw or f"{plant_kind}_default")
    return spec


def normalize_geometry_plant_spec(
    geometry_plant_kind: str | None,
    geometry_plant_spec: str | None,
) -> dict[str, Any] | None:
    spec = parse_geometry_plant_spec(geometry_plant_kind, geometry_plant_spec)
    if spec is None:
        return None
    normalized = dict(spec)
    for key in PLANT_SPEC_REQUIRED_KEYS:
        normalized.setdefault(key, None)
    normalized["plant_kind"] = infer_geometry_plant_kind(spec.get("plant_kind"))
    normalized["plant_id"] = str(normalized.get("plant_id") or f"{normalized['plant_kind']}_default")
    normalized["object_class"] = infer_geometry_object_class(normalized.get("object_class"))
    object_spec = normalized.get("object_spec")
    if isinstance(object_spec, str):
        normalized["object_spec"] = normalize_geometry_object_spec(normalized["object_class"], object_spec)
    elif isinstance(object_spec, dict):
        normalized["object_spec"] = normalize_geometry_object_spec(normalized["object_class"], json.dumps(object_spec))
    else:
        normalized["object_spec"] = None
    normalized["base_run"] = str(normalized.get("base_run") or "")
    normalized["head_scale"] = float(normalized.get("head_scale") or 0.0)
    normalized["basis_scale"] = float(normalized.get("basis_scale") or 0.0)
    normalized["store_mode"] = str(normalized.get("store_mode") or "none")
    normalized["source_mode"] = str(normalized.get("source_mode") or "correct")
    normalized["intervention_scope"] = str(normalized.get("intervention_scope") or "full_object_scope")
    normalized["default_attenuation"] = float(normalized.get("default_attenuation") or 0.0)
    return normalized


STORE_SELECTOR_TYPE_DEFAULTS: dict[str, str] = {
    "store_semantic_role_centroid": "semantic_role_bundle",
    "store_filler_identity_gap": "filler_bundle",
    "store_predicate_identity_gap": "predicate_bundle",
    "store_semantic_filler_combo": "sparse_geometry_bundle",
    "store_role_residual_bundle": "semantic_role_bundle",
    "store_filler_bundle": "filler_bundle",
    "store_predicate_bundle": "predicate_bundle",
    "store_sparse_geometry_bundle": "sparse_geometry_bundle",
}

ROUTE_SELECTOR_TYPE_DEFAULTS: dict[str, str] = {
    "route_relation_barrier": "head_gate",
    "route_path_consumption": "head_bundle",
    "route_trajectory_barrier": "relation_window",
    "route_wrong_path_sensitivity": "consumption_site",
    "route_head_gate": "head_gate",
    "route_head_bundle": "head_bundle",
    "route_relation_window": "relation_window",
    "route_consumption_site": "consumption_site",
}

READ_SELECTOR_TYPE_DEFAULTS: dict[str, str] = {
    "read_decoder_subspace_remove": "alignment_projection",
    "read_decoder_subspace_keep": "feature_bundle",
    "read_low_rank_access": "low_rank_basis",
    "read_pre_readout_alignment": "prelogit_basis",
    "read_prelogit_basis": "prelogit_basis",
    "read_low_rank_basis": "low_rank_basis",
    "read_feature_bundle": "feature_bundle",
    "read_alignment_projection": "alignment_projection",
}

INTERACTION_REQUIRED_SPEC_KEYS: tuple[str, ...] = (
    "decoder_step_key",
    "source_span_keys",
    "head_selector",
    "read_basis",
    "store_condition",
)
BRIDGE_REQUIRED_SPEC_KEYS: tuple[str, ...] = (
    "source_step_key",
    "target_step_key",
    "source_span_keys",
    "head_selector",
    "bridge_basis",
    "store_condition",
)
QUERY_SOURCE_REQUIRED_SPEC_KEYS: tuple[str, ...] = (
    "query_step_key",
    "source_span_keys",
    "head_selector",
    "query_basis",
    "read_basis",
    "store_condition",
)


def normalize_geometry_object_spec(
    geometry_object_class: str | None,
    geometry_object_spec: str | None,
) -> dict[str, Any] | None:
    spec = parse_geometry_object_spec(geometry_object_class, geometry_object_spec)
    if spec is None:
        return None
    object_class = str(spec["object_class"])
    object_id = str(spec["object_id"])
    selector_payload = dict(spec.get("selector_payload") or {})
    if object_class == GEOMETRY_STORE_OBJECT_CLASS:
        selector_type = str(spec.get("selector_type") or STORE_SELECTOR_TYPE_DEFAULTS.get(object_id) or "semantic_role_bundle")
        matched_control_type = str(spec.get("matched_control_type") or "same_layer_non_geometry_ablation")
    elif object_class == GEOMETRY_ROUTE_OBJECT_CLASS:
        selector_type = str(spec.get("selector_type") or ROUTE_SELECTOR_TYPE_DEFAULTS.get(object_id) or "head_gate")
        matched_control_type = str(spec.get("matched_control_type") or "wrong_route_or_wrong_read_drop")
    elif object_class == GEOMETRY_INTERACTION_OBJECT_CLASS:
        normalized = dict(spec)
        normalized.setdefault("layer_index", 0)
        normalized["decoder_step_key"] = str(spec.get("decoder_step_key") or "")
        normalized["source_span_keys"] = [str(value) for value in list(spec.get("source_span_keys") or [])]
        normalized["head_selector"] = dict(spec.get("head_selector") or {"mode": "explicit", "indices": []})
        normalized["read_basis"] = dict(spec.get("read_basis") or {"mode": "explicit", "basis_ids": []})
        normalized["store_condition"] = dict(spec.get("store_condition") or {"mode": "none"})
        normalized["matched_control_type"] = str(spec.get("matched_control_type") or "wrong_event_same_step")
        for key in INTERACTION_REQUIRED_SPEC_KEYS:
            normalized.setdefault(key, [] if key == "source_span_keys" else {})
        return normalized
    elif object_class == GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS:
        normalized = dict(spec)
        normalized.setdefault("layer_index", 0)
        normalized["source_step_key"] = str(spec.get("source_step_key") or "")
        normalized["target_step_key"] = str(spec.get("target_step_key") or "")
        normalized["source_span_keys"] = [str(value) for value in list(spec.get("source_span_keys") or [])]
        normalized["head_selector"] = dict(spec.get("head_selector") or {"mode": "explicit", "indices": []})
        normalized["bridge_basis"] = dict(spec.get("bridge_basis") or {"mode": "explicit", "basis_ids": []})
        normalized["store_condition"] = dict(spec.get("store_condition") or {"mode": "none"})
        normalized["matched_control_type"] = str(spec.get("matched_control_type") or "wrong_event_same_cardinality")
        for key in BRIDGE_REQUIRED_SPEC_KEYS:
            normalized.setdefault(key, [] if key == "source_span_keys" else {})
        return normalized
    elif object_class == GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS:
        normalized = dict(spec)
        normalized.setdefault("layer_index", 0)
        normalized["query_step_key"] = str(spec.get("query_step_key") or "")
        normalized["source_span_keys"] = [str(value) for value in list(spec.get("source_span_keys") or [])]
        normalized["head_selector"] = dict(spec.get("head_selector") or {"mode": "explicit", "indices": []})
        normalized["query_basis"] = dict(spec.get("query_basis") or {"mode": "explicit", "basis_ids": []})
        normalized["read_basis"] = dict(spec.get("read_basis") or {"mode": "explicit", "basis_ids": []})
        normalized["store_condition"] = dict(spec.get("store_condition") or {"mode": "none"})
        normalized["matched_control_type"] = str(spec.get("matched_control_type") or "wrong_event_same_cardinality")
        for key in QUERY_SOURCE_REQUIRED_SPEC_KEYS:
            normalized.setdefault(key, [] if key == "source_span_keys" else {})
        return normalized
    else:
        selector_type = str(spec.get("selector_type") or READ_SELECTOR_TYPE_DEFAULTS.get(object_id) or "alignment_projection")
        matched_control_type = str(spec.get("matched_control_type") or "wrong_route_or_wrong_read_drop")
    normalized = dict(spec)
    normalized["selector_type"] = selector_type
    normalized["selector_payload"] = selector_payload
    normalized["matched_control_type"] = matched_control_type
    return normalized


def geometry_object_behavior_key(
    geometry_object_class: str | None,
    geometry_object_spec: str | None,
) -> str | None:
    spec = normalize_geometry_object_spec(geometry_object_class, geometry_object_spec)
    if spec is None:
        return None
    object_class = str(spec["object_class"])
    if object_class in {
        GEOMETRY_INTERACTION_OBJECT_CLASS,
        GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS,
        GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS,
    }:
        return None
    selector_type = str(spec["selector_type"])
    if object_class == GEOMETRY_STORE_OBJECT_CLASS:
        return {
            "semantic_role_bundle": "store_semantic_role_centroid",
            "filler_bundle": "store_filler_identity_gap",
            "predicate_bundle": "store_predicate_identity_gap",
            "sparse_geometry_bundle": "store_semantic_filler_combo",
        }.get(selector_type, "store_semantic_role_centroid")
    if object_class == GEOMETRY_ROUTE_OBJECT_CLASS:
        return {
            "head_gate": "route_relation_barrier",
            "head_bundle": "route_path_consumption",
            "relation_window": "route_trajectory_barrier",
            "consumption_site": "route_wrong_path_sensitivity",
        }.get(selector_type, "route_relation_barrier")
    return {
        "prelogit_basis": "read_pre_readout_alignment",
        "low_rank_basis": "read_low_rank_access",
        "feature_bundle": "read_decoder_subspace_keep",
        "alignment_projection": "read_decoder_subspace_remove",
    }.get(selector_type, "read_decoder_subspace_remove")


def run_generation_phase4_protocol(
    adapter: Any,
    candidate_records: list[dict[str, Any]],
    reference_records: list[dict[str, Any]],
    matched_sets: list[dict[str, Any]],
    span_records: list[dict[str, Any]],
    layer_index: int,
    max_length: int = 256,
    max_new_tokens: int = 192,
    num_beams: int = 1,
    non_role_policy: str = "contentful_first",
    strongest_control_aggregation: str = "max",
    strongest_control_priority: tuple[str, ...] = ("wrong_role", "non_role", "shuffled"),
    preferred_counterpart_strata: tuple[str, ...] = ("pass", "control"),
    counterpart_same_split_only: bool = False,
    interchange_target_strata: tuple[str, ...] = ("fail",),
    rescue_target_strata: tuple[str, ...] = ("fail",),
    operationalization_family: str = BINDING_PROTOCOL_FAMILY,
    geometry_unit_class: str = GEOMETRY_ROUTING_UNIT_CLASS,
    geometry_object_class: str | None = None,
    geometry_object_spec: str | None = None,
    geometry_plant_kind: str = GEOMETRY_PLANT_KIND_OFF,
    geometry_plant_spec: str | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    protocol_family = infer_operationalization_family(operationalization_family)
    geometry_unit = infer_geometry_unit_class(geometry_unit_class)
    span_index = index_span_records(span_records=span_records, layer_index=layer_index)
    realism_bank = build_realism_bank(span_records=span_records, layer_index=layer_index)
    live_span_cache: dict[str, dict[str, Any]] = {}
    generation_cache: dict[str, dict[str, Any]] = {}
    control_span_cache: dict[tuple[str, int, int], dict[str, Any] | None] = {}
    control_window_cache: dict[tuple[str, int, int], dict[str, Any] | None] = {}
    counterpart_index = build_exact_counterpart_index(reference_records)
    interchange_targets = select_protocol_targets(
        candidate_records=candidate_records,
        matched_sets=matched_sets,
        target_strata=interchange_target_strata,
    )
    rescue_targets = select_protocol_targets(
        candidate_records=candidate_records,
        matched_sets=matched_sets,
        target_strata=rescue_target_strata,
    )

    m2_records = run_generation_interchange(
        adapter=adapter,
        target_records=interchange_targets,
        counterpart_index=counterpart_index,
        span_index=span_index,
        live_span_cache=live_span_cache,
        generation_cache=generation_cache,
        control_span_cache=control_span_cache,
        control_window_cache=control_window_cache,
        realism_bank=realism_bank,
        layer_index=layer_index,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        non_role_policy=non_role_policy,
        strongest_control_aggregation=strongest_control_aggregation,
        strongest_control_priority=strongest_control_priority,
        preferred_counterpart_strata=preferred_counterpart_strata,
        counterpart_same_split_only=counterpart_same_split_only,
        operationalization_family=protocol_family,
        seed=seed,
    )
    m3_records = run_generation_necessity(
        adapter=adapter,
        candidate_records=candidate_records,
        span_index=span_index,
        live_span_cache=live_span_cache,
        generation_cache=generation_cache,
        control_window_cache=control_window_cache,
        realism_bank=realism_bank,
        layer_index=layer_index,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        non_role_policy=non_role_policy,
        strongest_control_aggregation=strongest_control_aggregation,
        num_beams=num_beams,
        operationalization_family=protocol_family,
        geometry_unit_class=geometry_unit,
        geometry_object_class=geometry_object_class,
        geometry_object_spec=geometry_object_spec,
        seed=seed,
    )
    m4_records = run_generation_rescue(
        adapter=adapter,
        target_records=rescue_targets,
        counterpart_index=counterpart_index,
        span_index=span_index,
        live_span_cache=live_span_cache,
        generation_cache=generation_cache,
        control_span_cache=control_span_cache,
        control_window_cache=control_window_cache,
        realism_bank=realism_bank,
        layer_index=layer_index,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        non_role_policy=non_role_policy,
        strongest_control_aggregation=strongest_control_aggregation,
        strongest_control_priority=strongest_control_priority,
        preferred_counterpart_strata=preferred_counterpart_strata,
        counterpart_same_split_only=counterpart_same_split_only,
        operationalization_family=protocol_family,
        geometry_unit_class=geometry_unit,
        geometry_object_class=geometry_object_class,
        geometry_object_spec=geometry_object_spec,
        seed=seed,
    )
    linkage_records = build_generation_item_level_linkage(m3_records=m3_records, m4_records=m4_records)
    summary = summarize_generation_phase4(
        candidate_records=candidate_records,
        m2_records=m2_records,
        m3_records=m3_records,
        m4_records=m4_records,
        linkage_records=linkage_records,
        operationalization_family=protocol_family,
        geometry_unit_class=geometry_unit,
        geometry_object_class=geometry_object_class,
        geometry_object_spec=geometry_object_spec,
        geometry_plant_kind=geometry_plant_kind,
        geometry_plant_spec=geometry_plant_spec,
        seed=seed,
    )
    return m2_records, m3_records, m4_records, linkage_records, summary


def select_protocol_targets(
    candidate_records: list[dict[str, Any]],
    matched_sets: list[dict[str, Any]],
    target_strata: tuple[str, ...] = ("fail",),
) -> list[dict[str, Any]]:
    if not target_strata:
        return []

    selected_target_ids: list[str] = []
    for match in matched_sets:
        members = dict(match.get("members") or {})
        for stratum in target_strata:
            item_id = str(members.get(stratum) or "")
            if item_id:
                selected_target_ids.append(item_id)

    selected_target_id_set = set(selected_target_ids)
    target_records = [
        record for record in candidate_records if str(record.get("stratum")) in set(str(label) for label in target_strata)
    ]
    prioritized = [record for record in target_records if str(record["item_id"]) in selected_target_id_set]
    fallback = [record for record in target_records if str(record["item_id"]) not in selected_target_id_set]
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in prioritized + fallback:
        item_id = str(record["item_id"])
        if item_id in seen:
            continue
        seen.add(item_id)
        ordered.append(record)
    return ordered


def geometry_necessity_actual_overrides(item_id: str, role: str, seed: int, unit_class: str) -> dict[str, Any]:
    unit = infer_geometry_unit_class(unit_class)
    if unit == GEOMETRY_SUBSPACE_UNIT_CLASS:
        return {
            "encoder_geometry_transform": "feature_mask",
            "encoder_geometry_seed": stable_seed(seed, "geometry_u1_encmask", item_id, role),
            "encoder_geometry_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
            "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
            "decoder_cross_attention_bottleneck_seed": stable_seed(seed, "geometry_u1_bottleneck", item_id, role),
            "decoder_cross_attention_bottleneck_mode": "remove",
            "decoder_readout_subspace_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
            "decoder_readout_subspace_seed": stable_seed(seed, "geometry_u1_readout", item_id, role),
            "decoder_readout_subspace_mode": "remove",
        }
    if unit == GEOMETRY_TRAJECTORY_UNIT_CLASS:
        return {
            "encoder_geometry_transform": "gaussian_noise",
            "encoder_geometry_seed": stable_seed(seed, "geometry_u2_noise", item_id, role),
            "encoder_geometry_noise_std": GEOMETRY_TRAJECTORY_NOISE_STD,
            "decoder_cross_attention_layer_keep_ratio": GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO,
            "decoder_cross_attention_layer_seed": stable_seed(seed, "geometry_u2_layers", item_id, role),
            "decoder_cross_attention_layer_policy": "alternating",
            "decoder_cross_attention_relation_barrier_keep_ratio": GEOMETRY_RELATION_KEEP_RATIO,
            "decoder_cross_attention_relation_barrier_seed": stable_seed(seed, "geometry_u2_relation", item_id, role),
            "decoder_cross_attention_relation_barrier_scale": GEOMETRY_NECESSITY_RELATION_SCALE,
        }
    return {
        "decoder_cross_attention_relation_barrier_keep_ratio": GEOMETRY_RELATION_KEEP_RATIO,
        "decoder_cross_attention_relation_barrier_seed": stable_seed(seed, "geometry_m3_relation", item_id, role),
        "decoder_cross_attention_relation_barrier_scale": GEOMETRY_NECESSITY_RELATION_SCALE,
        "decoder_readout_subspace_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
        "decoder_readout_subspace_seed": stable_seed(seed, "geometry_m3_readout", item_id, role),
        "decoder_readout_subspace_mode": "remove",
    }


def geometry_necessity_control_overrides(item_id: str, role: str, seed: int, unit_class: str) -> dict[str, dict[str, Any]]:
    unit = infer_geometry_unit_class(unit_class)
    if unit == GEOMETRY_SUBSPACE_UNIT_CLASS:
        return {
            "wrong_role": {
                "encoder_geometry_transform": "orthogonal_preserve",
                "encoder_geometry_seed": stable_seed(seed, "geometry_u1_orth", item_id, role),
                "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_GENERIC_CONTROL_KEEP_RATIO,
                "decoder_cross_attention_bottleneck_seed": stable_seed(seed, "geometry_u1_keep", item_id, role),
                "decoder_cross_attention_bottleneck_mode": "keep",
            },
            "non_role": {
                "decoder_readout_keep_ratio": GEOMETRY_GENERIC_CONTROL_KEEP_RATIO,
            },
        }
    if unit == GEOMETRY_TRAJECTORY_UNIT_CLASS:
        return {
            "wrong_role": {
                "decoder_cross_attention_layer_keep_ratio": GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO,
                "decoder_cross_attention_layer_seed": stable_seed(seed, "geometry_u2_wrong_layers", item_id, role),
                "decoder_cross_attention_layer_policy": "top",
            },
            "non_role": {
                "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_GENERIC_CONTROL_KEEP_RATIO,
                "decoder_cross_attention_bottleneck_seed": stable_seed(seed, "geometry_u2_bottleneck", item_id, role),
                "decoder_cross_attention_bottleneck_mode": "keep",
            },
        }
    return {
        "wrong_role": {
            "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_GENERIC_CONTROL_KEEP_RATIO,
            "decoder_cross_attention_bottleneck_seed": stable_seed(seed, "geometry_m3_bottleneck", item_id, role),
            "decoder_cross_attention_bottleneck_mode": "keep",
        },
        "non_role": {
            "decoder_readout_keep_ratio": GEOMETRY_GENERIC_CONTROL_KEEP_RATIO,
        },
    }


def geometry_rescue_actual_overrides(item_id: str, role: str, seed: int, unit_class: str) -> dict[str, Any]:
    unit = infer_geometry_unit_class(unit_class)
    if unit == GEOMETRY_SUBSPACE_UNIT_CLASS:
        return {
            "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
            "decoder_cross_attention_bottleneck_seed": stable_seed(seed, "geometry_u1_m4_bottleneck", item_id, role),
            "decoder_cross_attention_bottleneck_mode": "keep",
            "decoder_readout_subspace_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
            "decoder_readout_subspace_seed": stable_seed(seed, "geometry_u1_m4_readout", item_id, role),
            "decoder_readout_subspace_mode": "keep",
        }
    if unit == GEOMETRY_TRAJECTORY_UNIT_CLASS:
        return {
            "decoder_cross_attention_layer_keep_ratio": GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO,
            "decoder_cross_attention_layer_seed": stable_seed(seed, "geometry_u2_m4_layers", item_id, role),
            "decoder_cross_attention_layer_policy": "alternating",
            "decoder_cross_attention_scale": GEOMETRY_TRAJECTORY_RESCUE_SCALE,
        }
    return {
        "decoder_cross_attention_relation_barrier_keep_ratio": GEOMETRY_RELATION_KEEP_RATIO,
        "decoder_cross_attention_relation_barrier_seed": stable_seed(seed, "geometry_m4_relation", item_id, role),
        "decoder_cross_attention_relation_barrier_scale": GEOMETRY_RESCUE_RELATION_SCALE,
    }


def geometry_rescue_wrong_path_overrides(item_id: str, role: str, seed: int, unit_class: str) -> dict[str, Any]:
    unit = infer_geometry_unit_class(unit_class)
    if unit == GEOMETRY_SUBSPACE_UNIT_CLASS:
        return {
            "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
            "decoder_cross_attention_bottleneck_seed": stable_seed(seed, "geometry_u1_wrong_bottleneck", item_id, role),
            "decoder_cross_attention_bottleneck_mode": "keep",
            "decoder_readout_subspace_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
            "decoder_readout_subspace_seed": stable_seed(seed, "geometry_u1_wrong_readout", item_id, role),
            "decoder_readout_subspace_mode": "keep",
        }
    if unit == GEOMETRY_TRAJECTORY_UNIT_CLASS:
        return {
            "decoder_cross_attention_layer_keep_ratio": GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO,
            "decoder_cross_attention_layer_seed": stable_seed(seed, "geometry_u2_wrong_layers", item_id, role),
            "decoder_cross_attention_layer_policy": "top",
            "decoder_cross_attention_scale": GEOMETRY_TRAJECTORY_RESCUE_SCALE,
        }
    return {
        "decoder_cross_attention_relation_barrier_keep_ratio": GEOMETRY_RELATION_KEEP_RATIO,
        "decoder_cross_attention_relation_barrier_seed": stable_seed(seed, "geometry_m4_wrong_path", item_id, role),
        "decoder_cross_attention_relation_barrier_scale": GEOMETRY_RESCUE_RELATION_SCALE,
    }


def geometry_rescue_generic_decoder_gain_overrides(item_id: str, role: str, seed: int, unit_class: str) -> dict[str, Any]:
    del item_id, role, seed, unit_class
    return {
        "decoder_cross_attention_scale": GEOMETRY_GENERIC_DECODER_GAIN_SCALE,
    }


def _geometry_object_spec_id(
    geometry_object_class: str | None,
    geometry_object_spec: str | None,
) -> str | None:
    return geometry_object_id(geometry_object_class, geometry_object_spec)


def _unique_span_records(records: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for record in records:
        if record is None:
            continue
        token_indices = tuple(int(index) for index in list(record.get("token_indices") or []))
        if not token_indices or token_indices in seen:
            continue
        seen.add(token_indices)
        unique.append(record)
    return unique


def _ablation_specs_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"token_indices": list(record["token_indices"])} for record in _unique_span_records(records)]


def _patch_specs_from_records(
    *,
    target_records: list[dict[str, Any]],
    donor_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for target_record, donor_record in zip(_unique_span_records(target_records), _unique_span_records(donor_records)):
        specs.append(
            {
                "token_indices": list(target_record["token_indices"]),
                "replacement_vector": list(donor_record["vector"]),
            }
        )
    return specs


def _store_site_keys_for_object(selector_type: str, role: str) -> list[str]:
    if selector_type == "predicate_bundle":
        return ["predicate"]
    if selector_type == "sparse_geometry_bundle":
        return [role, "predicate"]
    if selector_type == "filler_bundle":
        return [f"filler::{role}"]
    return [role]


def _store_wrong_site_keys_for_object(selector_type: str, role: str) -> list[str]:
    if selector_type == "predicate_bundle":
        return [role]
    if selector_type == "sparse_geometry_bundle":
        return [opposite_role(role), "predicate"]
    if selector_type == "filler_bundle":
        return [f"filler::{opposite_role(role)}"]
    return [opposite_role(role)]


def _store_target_records(
    item_span_map: dict[str, Any],
    *,
    selector_type: str,
    role: str,
) -> list[dict[str, Any]]:
    return _unique_span_records([item_span_map.get(key) for key in _store_site_keys_for_object(selector_type, role)])


def _store_wrong_records(
    item_span_map: dict[str, Any],
    *,
    selector_type: str,
    role: str,
) -> list[dict[str, Any]]:
    return _unique_span_records([item_span_map.get(key) for key in _store_wrong_site_keys_for_object(selector_type, role)])


def _store_donor_records(
    counterpart_span_map: dict[str, Any],
    *,
    selector_type: str,
    role: str,
) -> list[dict[str, Any]]:
    return _unique_span_records([counterpart_span_map.get(key) for key in _store_site_keys_for_object(selector_type, role)])


def _store_wrong_donor_records(
    counterpart_span_map: dict[str, Any],
    *,
    selector_type: str,
    role: str,
) -> list[dict[str, Any]]:
    return _unique_span_records([counterpart_span_map.get(key) for key in _store_wrong_site_keys_for_object(selector_type, role)])


def geometry_generic_path_only_overrides(item_id: str, role: str, seed: int) -> dict[str, Any]:
    return geometry_rescue_actual_overrides(item_id=item_id, role=role, seed=seed, unit_class=GEOMETRY_ROUTING_UNIT_CLASS)


def geometry_route_object_necessity_actual_overrides(item_id: str, role: str, seed: int, object_id: str) -> dict[str, Any]:
    if object_id == "route_path_consumption":
        return {
            "decoder_cross_attention_relation_barrier_keep_ratio": GEOMETRY_RELATION_KEEP_RATIO,
            "decoder_cross_attention_relation_barrier_seed": stable_seed(seed, object_id, item_id, role, "relation"),
            "decoder_cross_attention_relation_barrier_scale": GEOMETRY_NECESSITY_RELATION_SCALE,
            "decoder_cross_attention_layer_keep_ratio": GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO,
            "decoder_cross_attention_layer_seed": stable_seed(seed, object_id, item_id, role, "layers"),
            "decoder_cross_attention_layer_policy": "alternating",
        }
    if object_id == "route_trajectory_barrier":
        return {
            "decoder_cross_attention_layer_keep_ratio": GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO,
            "decoder_cross_attention_layer_seed": stable_seed(seed, object_id, item_id, role, "layers"),
            "decoder_cross_attention_layer_policy": "alternating",
            "encoder_geometry_transform": "gaussian_noise",
            "encoder_geometry_seed": stable_seed(seed, object_id, item_id, role, "noise"),
            "encoder_geometry_noise_std": GEOMETRY_TRAJECTORY_NOISE_STD,
        }
    if object_id == "route_wrong_path_sensitivity":
        return {
            "decoder_cross_attention_layer_keep_ratio": GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO,
            "decoder_cross_attention_layer_seed": stable_seed(seed, object_id, item_id, role, "layers"),
            "decoder_cross_attention_layer_policy": "top",
            "decoder_cross_attention_relation_barrier_keep_ratio": GEOMETRY_RELATION_KEEP_RATIO,
            "decoder_cross_attention_relation_barrier_seed": stable_seed(seed, object_id, item_id, role, "relation"),
            "decoder_cross_attention_relation_barrier_scale": GEOMETRY_NECESSITY_RELATION_SCALE,
        }
    return {
        "decoder_cross_attention_relation_barrier_keep_ratio": GEOMETRY_RELATION_KEEP_RATIO,
        "decoder_cross_attention_relation_barrier_seed": stable_seed(seed, object_id, item_id, role, "relation"),
        "decoder_cross_attention_relation_barrier_scale": GEOMETRY_NECESSITY_RELATION_SCALE,
    }


def geometry_route_object_necessity_control_overrides(item_id: str, role: str, seed: int, object_id: str) -> dict[str, dict[str, Any]]:
    if object_id == "route_path_consumption":
        return {
            "wrong_role": {
                "decoder_cross_attention_layer_keep_ratio": GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO,
                "decoder_cross_attention_layer_seed": stable_seed(seed, object_id, item_id, role, "wrong_layers"),
                "decoder_cross_attention_layer_policy": "top",
            },
            "non_role": {
                "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_GENERIC_CONTROL_KEEP_RATIO,
                "decoder_cross_attention_bottleneck_seed": stable_seed(seed, object_id, item_id, role, "bottleneck"),
                "decoder_cross_attention_bottleneck_mode": "keep",
            },
        }
    if object_id == "route_trajectory_barrier":
        return {
            "wrong_role": {
                "decoder_cross_attention_layer_keep_ratio": GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO,
                "decoder_cross_attention_layer_seed": stable_seed(seed, object_id, item_id, role, "wrong_layers"),
                "decoder_cross_attention_layer_policy": "bottom",
            },
            "non_role": {
                "decoder_readout_keep_ratio": GEOMETRY_GENERIC_CONTROL_KEEP_RATIO,
            },
        }
    if object_id == "route_wrong_path_sensitivity":
        return {
            "wrong_role": {
                "decoder_cross_attention_layer_keep_ratio": GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO,
                "decoder_cross_attention_layer_seed": stable_seed(seed, object_id, item_id, role, "wrong_layers"),
                "decoder_cross_attention_layer_policy": "alternating",
            },
            "non_role": {
                "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_GENERIC_CONTROL_KEEP_RATIO,
                "decoder_cross_attention_bottleneck_seed": stable_seed(seed, object_id, item_id, role, "bottleneck"),
                "decoder_cross_attention_bottleneck_mode": "keep",
            },
        }
    return {
        "wrong_role": {
            "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_GENERIC_CONTROL_KEEP_RATIO,
            "decoder_cross_attention_bottleneck_seed": stable_seed(seed, object_id, item_id, role, "bottleneck"),
            "decoder_cross_attention_bottleneck_mode": "keep",
        },
        "non_role": {
            "decoder_readout_keep_ratio": GEOMETRY_GENERIC_CONTROL_KEEP_RATIO,
        },
    }


def geometry_read_object_necessity_actual_overrides(item_id: str, role: str, seed: int, object_id: str) -> dict[str, Any]:
    if object_id == "read_low_rank_access":
        return {
            "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
            "decoder_cross_attention_bottleneck_seed": stable_seed(seed, object_id, item_id, role, "bottleneck"),
            "decoder_cross_attention_bottleneck_mode": "remove",
        }
    if object_id == "read_pre_readout_alignment":
        return {
            "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
            "decoder_cross_attention_bottleneck_seed": stable_seed(seed, object_id, item_id, role, "bottleneck"),
            "decoder_cross_attention_bottleneck_mode": "remove",
            "decoder_readout_subspace_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
            "decoder_readout_subspace_seed": stable_seed(seed, object_id, item_id, role, "readout"),
            "decoder_readout_subspace_mode": "remove",
        }
    if object_id == "read_decoder_subspace_keep":
        return {
            "decoder_readout_keep_ratio": GEOMETRY_GENERIC_CONTROL_KEEP_RATIO,
        }
    return {
        "decoder_readout_subspace_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
        "decoder_readout_subspace_seed": stable_seed(seed, object_id, item_id, role, "readout"),
        "decoder_readout_subspace_mode": "remove",
    }


def geometry_read_object_necessity_control_overrides(item_id: str, role: str, seed: int, object_id: str) -> dict[str, dict[str, Any]]:
    if object_id == "read_low_rank_access":
        return {
            "wrong_role": {
                "decoder_readout_subspace_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
                "decoder_readout_subspace_seed": stable_seed(seed, object_id, item_id, role, "wrong_readout"),
                "decoder_readout_subspace_mode": "remove",
            },
            "non_role": {
                "decoder_cross_attention_relation_barrier_keep_ratio": GEOMETRY_RELATION_KEEP_RATIO,
                "decoder_cross_attention_relation_barrier_seed": stable_seed(seed, object_id, item_id, role, "relation"),
                "decoder_cross_attention_relation_barrier_scale": GEOMETRY_NECESSITY_RELATION_SCALE,
            },
        }
    if object_id == "read_pre_readout_alignment":
        return {
            "wrong_role": {
                "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
                "decoder_cross_attention_bottleneck_seed": stable_seed(seed, object_id, item_id, role, "wrong_bottleneck"),
                "decoder_cross_attention_bottleneck_mode": "remove",
            },
            "non_role": {
                "decoder_readout_keep_ratio": GEOMETRY_GENERIC_CONTROL_KEEP_RATIO,
            },
        }
    return {
        "wrong_role": {
            "decoder_cross_attention_relation_barrier_keep_ratio": GEOMETRY_RELATION_KEEP_RATIO,
            "decoder_cross_attention_relation_barrier_seed": stable_seed(seed, object_id, item_id, role, "relation"),
            "decoder_cross_attention_relation_barrier_scale": GEOMETRY_NECESSITY_RELATION_SCALE,
        },
        "non_role": {
            "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_GENERIC_CONTROL_KEEP_RATIO,
            "decoder_cross_attention_bottleneck_seed": stable_seed(seed, object_id, item_id, role, "bottleneck"),
            "decoder_cross_attention_bottleneck_mode": "keep",
        },
    }


def geometry_route_object_rescue_overrides(item_id: str, role: str, seed: int, object_id: str) -> dict[str, Any]:
    if object_id == "route_path_consumption":
        return {
            "decoder_cross_attention_relation_barrier_keep_ratio": GEOMETRY_RELATION_KEEP_RATIO,
            "decoder_cross_attention_relation_barrier_seed": stable_seed(seed, object_id, item_id, role, "relation"),
            "decoder_cross_attention_relation_barrier_scale": GEOMETRY_RESCUE_RELATION_SCALE,
            "decoder_cross_attention_layer_keep_ratio": GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO,
            "decoder_cross_attention_layer_seed": stable_seed(seed, object_id, item_id, role, "layers"),
            "decoder_cross_attention_layer_policy": "alternating",
        }
    if object_id == "route_trajectory_barrier":
        return {
            "decoder_cross_attention_layer_keep_ratio": GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO,
            "decoder_cross_attention_layer_seed": stable_seed(seed, object_id, item_id, role, "layers"),
            "decoder_cross_attention_layer_policy": "alternating",
            "decoder_cross_attention_scale": GEOMETRY_TRAJECTORY_RESCUE_SCALE,
        }
    if object_id == "route_wrong_path_sensitivity":
        return {
            "decoder_cross_attention_layer_keep_ratio": GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO,
            "decoder_cross_attention_layer_seed": stable_seed(seed, object_id, item_id, role, "layers"),
            "decoder_cross_attention_layer_policy": "top",
            "decoder_cross_attention_scale": GEOMETRY_TRAJECTORY_RESCUE_SCALE,
        }
    return {
        "decoder_cross_attention_relation_barrier_keep_ratio": GEOMETRY_RELATION_KEEP_RATIO,
        "decoder_cross_attention_relation_barrier_seed": stable_seed(seed, object_id, item_id, role, "relation"),
        "decoder_cross_attention_relation_barrier_scale": GEOMETRY_RESCUE_RELATION_SCALE,
    }


def geometry_route_wrong_object_rescue_overrides(item_id: str, role: str, seed: int, object_id: str) -> dict[str, Any]:
    if object_id in {"route_path_consumption", "route_wrong_path_sensitivity"}:
        return {
            "decoder_cross_attention_layer_keep_ratio": GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO,
            "decoder_cross_attention_layer_seed": stable_seed(seed, object_id, item_id, role, "wrong_layers"),
            "decoder_cross_attention_layer_policy": "bottom",
            "decoder_cross_attention_scale": GEOMETRY_TRAJECTORY_RESCUE_SCALE,
        }
    if object_id == "route_trajectory_barrier":
        return {
            "decoder_cross_attention_layer_keep_ratio": GEOMETRY_TRAJECTORY_LAYER_KEEP_RATIO,
            "decoder_cross_attention_layer_seed": stable_seed(seed, object_id, item_id, role, "wrong_layers"),
            "decoder_cross_attention_layer_policy": "top",
            "decoder_cross_attention_scale": GEOMETRY_TRAJECTORY_RESCUE_SCALE,
        }
    return {
        "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_GENERIC_CONTROL_KEEP_RATIO,
        "decoder_cross_attention_bottleneck_seed": stable_seed(seed, object_id, item_id, role, "wrong_bottleneck"),
        "decoder_cross_attention_bottleneck_mode": "keep",
    }


def geometry_read_object_rescue_overrides(item_id: str, role: str, seed: int, object_id: str) -> dict[str, Any]:
    if object_id == "read_low_rank_access":
        return {
            "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
            "decoder_cross_attention_bottleneck_seed": stable_seed(seed, object_id, item_id, role, "bottleneck"),
            "decoder_cross_attention_bottleneck_mode": "keep",
        }
    if object_id == "read_pre_readout_alignment":
        return {
            "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
            "decoder_cross_attention_bottleneck_seed": stable_seed(seed, object_id, item_id, role, "bottleneck"),
            "decoder_cross_attention_bottleneck_mode": "keep",
            "decoder_readout_subspace_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
            "decoder_readout_subspace_seed": stable_seed(seed, object_id, item_id, role, "readout"),
            "decoder_readout_subspace_mode": "keep",
        }
    if object_id == "read_decoder_subspace_keep":
        return {
            "decoder_readout_keep_ratio": GEOMETRY_GENERIC_CONTROL_KEEP_RATIO,
        }
    return {
        "decoder_readout_subspace_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
        "decoder_readout_subspace_seed": stable_seed(seed, object_id, item_id, role, "readout"),
        "decoder_readout_subspace_mode": "keep",
    }


def geometry_read_wrong_object_rescue_overrides(item_id: str, role: str, seed: int, object_id: str) -> dict[str, Any]:
    if object_id == "read_low_rank_access":
        return {
            "decoder_readout_subspace_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
            "decoder_readout_subspace_seed": stable_seed(seed, object_id, item_id, role, "wrong_readout"),
            "decoder_readout_subspace_mode": "keep",
        }
    if object_id == "read_pre_readout_alignment":
        return {
            "decoder_cross_attention_relation_barrier_keep_ratio": GEOMETRY_RELATION_KEEP_RATIO,
            "decoder_cross_attention_relation_barrier_seed": stable_seed(seed, object_id, item_id, role, "wrong_relation"),
            "decoder_cross_attention_relation_barrier_scale": GEOMETRY_RESCUE_RELATION_SCALE,
        }
    return {
        "decoder_cross_attention_bottleneck_keep_ratio": GEOMETRY_SUBSPACE_KEEP_RATIO,
        "decoder_cross_attention_bottleneck_seed": stable_seed(seed, object_id, item_id, role, "wrong_bottleneck"),
        "decoder_cross_attention_bottleneck_mode": "keep",
    }


def geometry_object_necessity_plan(
    *,
    item: dict[str, Any],
    item_span_map: dict[str, Any],
    role: str,
    unit_class: str,
    geometry_object_class: str | None,
    geometry_object_spec: str | None,
    non_role_window: dict[str, Any] | None,
    seed: int,
) -> dict[str, Any]:
    object_class = infer_geometry_object_class(geometry_object_class)
    object_id = _geometry_object_spec_id(geometry_object_class, geometry_object_spec)
    object_spec = normalize_geometry_object_spec(geometry_object_class, geometry_object_spec)
    selector_type = str((object_spec or {}).get("selector_type") or "")
    behavior_key = geometry_object_behavior_key(geometry_object_class, geometry_object_spec)
    if object_class == GEOMETRY_STORE_OBJECT_CLASS and object_id is not None:
        actual_records = _store_target_records(item_span_map, selector_type=selector_type, role=role)
        wrong_records = _store_wrong_records(item_span_map, selector_type=selector_type, role=role)
        return {
            "object_id": object_id,
            "actual_ablation_specs": _ablation_specs_from_records(actual_records),
            "actual_overrides": None,
            "wrong_ablation_specs": _ablation_specs_from_records(wrong_records),
            "wrong_overrides": None,
            "non_role_ablation_specs": (
                [{"token_indices": list(non_role_window["token_indices"])}] if non_role_window is not None else []
            ),
            "non_role_overrides": None,
            "actual_label": f"{object_id}_ablation",
            "wrong_label": "same_layer_non_geometry_ablation",
            "non_role_label": "non_role_window_ablation",
        }
    if object_class == GEOMETRY_ROUTE_OBJECT_CLASS and object_id is not None and behavior_key is not None:
        return {
            "object_id": object_id,
            "actual_ablation_specs": [],
            "actual_overrides": geometry_route_object_necessity_actual_overrides(str(item["item_id"]), role, seed, behavior_key),
            "wrong_ablation_specs": [],
            "wrong_overrides": geometry_route_object_necessity_control_overrides(str(item["item_id"]), role, seed, behavior_key)["wrong_role"],
            "non_role_ablation_specs": [],
            "non_role_overrides": geometry_route_object_necessity_control_overrides(str(item["item_id"]), role, seed, behavior_key)["non_role"],
            "actual_label": f"{object_id}_necessity",
            "wrong_label": "wrong_route_control",
            "non_role_label": "generic_barrier_control",
        }
    if object_class == GEOMETRY_READ_OBJECT_CLASS and object_id is not None and behavior_key is not None:
        return {
            "object_id": object_id,
            "actual_ablation_specs": [],
            "actual_overrides": geometry_read_object_necessity_actual_overrides(str(item["item_id"]), role, seed, behavior_key),
            "wrong_ablation_specs": [],
            "wrong_overrides": geometry_read_object_necessity_control_overrides(str(item["item_id"]), role, seed, behavior_key)["wrong_role"],
            "non_role_ablation_specs": [],
            "non_role_overrides": geometry_read_object_necessity_control_overrides(str(item["item_id"]), role, seed, behavior_key)["non_role"],
            "actual_label": f"{object_id}_necessity",
            "wrong_label": "wrong_read_control",
            "non_role_label": "generic_barrier_control",
        }
    return {
        "object_id": object_id,
        "actual_ablation_specs": [],
        "actual_overrides": geometry_necessity_actual_overrides(str(item["item_id"]), role, seed, unit_class),
        "wrong_ablation_specs": [],
        "wrong_overrides": geometry_necessity_control_overrides(str(item["item_id"]), role, seed, unit_class)["wrong_role"],
        "non_role_ablation_specs": [],
        "non_role_overrides": geometry_necessity_control_overrides(str(item["item_id"]), role, seed, unit_class)["non_role"],
        "actual_label": (
            "subspace_remove"
            if infer_geometry_unit_class(unit_class) == GEOMETRY_SUBSPACE_UNIT_CLASS
            else "trajectory_barrier"
            if infer_geometry_unit_class(unit_class) == GEOMETRY_TRAJECTORY_UNIT_CLASS
            else "relation_barrier_plus_readout_remove"
        ),
        "wrong_label": (
            "orthogonal_preserve_plus_bottleneck_keep"
            if infer_geometry_unit_class(unit_class) == GEOMETRY_SUBSPACE_UNIT_CLASS
            else "wrong_layer_policy"
            if infer_geometry_unit_class(unit_class) == GEOMETRY_TRAJECTORY_UNIT_CLASS
            else "cross_attention_bottleneck"
        ),
        "non_role_label": (
            "decoder_readout_feature_mask"
            if infer_geometry_unit_class(unit_class) == GEOMETRY_SUBSPACE_UNIT_CLASS
            else "cross_attention_bottleneck"
            if infer_geometry_unit_class(unit_class) == GEOMETRY_TRAJECTORY_UNIT_CLASS
            else "decoder_readout_feature_mask"
        ),
    }


def geometry_object_rescue_plan(
    *,
    item_id: str,
    role: str,
    unit_class: str,
    geometry_object_class: str | None,
    geometry_object_spec: str | None,
    target_role: dict[str, Any],
    donor_role: dict[str, Any],
    wrong_role_record: dict[str, Any],
    non_role_record: dict[str, Any],
    target_span_map: dict[str, Any],
    counterpart_span_map: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    object_class = infer_geometry_object_class(geometry_object_class)
    object_id = _geometry_object_spec_id(geometry_object_class, geometry_object_spec)
    object_spec = normalize_geometry_object_spec(geometry_object_class, geometry_object_spec)
    selector_type = str((object_spec or {}).get("selector_type") or "")
    behavior_key = geometry_object_behavior_key(geometry_object_class, geometry_object_spec)
    generic_gain_overrides = geometry_rescue_generic_decoder_gain_overrides(item_id, role, seed, unit_class)
    generic_path_overrides = geometry_generic_path_only_overrides(item_id, role, seed)
    if object_class == GEOMETRY_STORE_OBJECT_CLASS and object_id is not None:
        target_records = _store_target_records(target_span_map, selector_type=selector_type, role=role)
        donor_records = _store_donor_records(counterpart_span_map, selector_type=selector_type, role=role)
        wrong_donor_records = _store_wrong_donor_records(counterpart_span_map, selector_type=selector_type, role=role)
        actual_overrides = geometry_generic_path_only_overrides(item_id, role, seed)
        return {
            "object_id": object_id,
            "actual_overrides": actual_overrides,
            "actual_patch_specs": _patch_specs_from_records(target_records=target_records, donor_records=donor_records),
            "object_only_overrides": None,
            "object_only_patch_specs": _patch_specs_from_records(target_records=target_records, donor_records=donor_records),
            "wrong_object_overrides": actual_overrides,
            "wrong_object_patch_specs": _patch_specs_from_records(target_records=target_records, donor_records=wrong_donor_records),
            "non_role_overrides": actual_overrides,
            "non_role_patch_specs": [
                {
                    "token_indices": list(target_role["token_indices"]),
                    "replacement_vector": list(non_role_record["vector"]),
                }
            ],
            "path_only_overrides": generic_path_overrides,
            "wrong_path_overrides": geometry_rescue_wrong_path_overrides(item_id, role, seed, GEOMETRY_ROUTING_UNIT_CLASS),
            "generic_gain_overrides": generic_gain_overrides,
            "actual_label": f"{object_id}_plus_alignment",
            "object_only_label": f"{object_id}_reinstate_only",
            "wrong_object_label": "wrong_object_reinstate",
            "path_only_label": "generic_path_only",
            "wrong_path_label": "wrong_relation_path_only",
        }
    if object_class == GEOMETRY_ROUTE_OBJECT_CLASS and object_id is not None and behavior_key is not None:
        object_overrides = geometry_route_object_rescue_overrides(item_id, role, seed, behavior_key)
        return {
            "object_id": object_id,
            "actual_overrides": object_overrides,
            "actual_patch_specs": [
                {
                    "token_indices": list(target_role["token_indices"]),
                    "replacement_vector": list(donor_role["vector"]),
                }
            ],
            "object_only_overrides": object_overrides,
            "object_only_patch_specs": [],
            "wrong_object_overrides": object_overrides,
            "wrong_object_patch_specs": [
                {
                    "token_indices": list(target_role["token_indices"]),
                    "replacement_vector": list(wrong_role_record["vector"]),
                }
            ],
            "non_role_overrides": object_overrides,
            "non_role_patch_specs": [
                {
                    "token_indices": list(target_role["token_indices"]),
                    "replacement_vector": list(non_role_record["vector"]),
                }
            ],
            "path_only_overrides": generic_path_overrides,
            "wrong_path_overrides": geometry_route_wrong_object_rescue_overrides(item_id, role, seed, behavior_key),
            "generic_gain_overrides": generic_gain_overrides,
            "actual_label": f"{object_id}_plus_alignment",
            "object_only_label": f"{object_id}_reinstate_only",
            "wrong_object_label": "wrong_route_reinstate",
            "path_only_label": "generic_path_only",
            "wrong_path_label": "wrong_route_reinstate_only",
        }
    if object_class == GEOMETRY_READ_OBJECT_CLASS and object_id is not None and behavior_key is not None:
        object_overrides = geometry_read_object_rescue_overrides(item_id, role, seed, behavior_key)
        return {
            "object_id": object_id,
            "actual_overrides": object_overrides,
            "actual_patch_specs": [
                {
                    "token_indices": list(target_role["token_indices"]),
                    "replacement_vector": list(donor_role["vector"]),
                }
            ],
            "object_only_overrides": object_overrides,
            "object_only_patch_specs": [],
            "wrong_object_overrides": object_overrides,
            "wrong_object_patch_specs": [
                {
                    "token_indices": list(target_role["token_indices"]),
                    "replacement_vector": list(wrong_role_record["vector"]),
                }
            ],
            "non_role_overrides": object_overrides,
            "non_role_patch_specs": [
                {
                    "token_indices": list(target_role["token_indices"]),
                    "replacement_vector": list(non_role_record["vector"]),
                }
            ],
            "path_only_overrides": generic_path_overrides,
            "wrong_path_overrides": geometry_read_wrong_object_rescue_overrides(item_id, role, seed, behavior_key),
            "generic_gain_overrides": generic_gain_overrides,
            "actual_label": f"{object_id}_plus_alignment",
            "object_only_label": f"{object_id}_reinstate_only",
            "wrong_object_label": "wrong_read_reinstate",
            "path_only_label": "generic_path_only",
            "wrong_path_label": "wrong_read_reinstate_only",
        }
    actual_overrides = geometry_rescue_actual_overrides(item_id, role, seed, unit_class)
    return {
        "object_id": object_id,
        "actual_overrides": actual_overrides,
        "actual_patch_specs": [
            {
                "token_indices": list(target_role["token_indices"]),
                "replacement_vector": list(donor_role["vector"]),
            }
        ],
        "object_only_overrides": None,
        "object_only_patch_specs": [
            {
                "token_indices": list(target_role["token_indices"]),
                "replacement_vector": list(donor_role["vector"]),
            }
        ],
        "wrong_object_overrides": actual_overrides,
        "wrong_object_patch_specs": [
            {
                "token_indices": list(target_role["token_indices"]),
                "replacement_vector": list(wrong_role_record["vector"]),
            }
        ],
        "non_role_overrides": actual_overrides,
        "non_role_patch_specs": [
            {
                "token_indices": list(target_role["token_indices"]),
                "replacement_vector": list(non_role_record["vector"]),
            }
        ],
        "path_only_overrides": actual_overrides,
        "wrong_path_overrides": geometry_rescue_wrong_path_overrides(item_id, role, seed, unit_class),
        "generic_gain_overrides": generic_gain_overrides,
        "actual_label": (
            "path_plus_alignment"
            if infer_geometry_unit_class(unit_class) == GEOMETRY_ROUTING_UNIT_CLASS
            else "subspace_plus_alignment"
            if infer_geometry_unit_class(unit_class) == GEOMETRY_SUBSPACE_UNIT_CLASS
            else "trajectory_plus_alignment"
        ),
        "object_only_label": "correct_alignment_only",
        "wrong_object_label": "wrong_alignment_plus_path",
        "path_only_label": (
            "relation_consumption_boost_only"
            if infer_geometry_unit_class(unit_class) == GEOMETRY_ROUTING_UNIT_CLASS
            else "subspace_only"
            if infer_geometry_unit_class(unit_class) == GEOMETRY_SUBSPACE_UNIT_CLASS
            else "trajectory_only"
        ),
        "wrong_path_label": (
            "wrong_relation_path_boost_only"
            if infer_geometry_unit_class(unit_class) == GEOMETRY_ROUTING_UNIT_CLASS
            else "wrong_subspace_only"
            if infer_geometry_unit_class(unit_class) == GEOMETRY_SUBSPACE_UNIT_CLASS
            else "wrong_trajectory_only"
        ),
    }


def run_generation_interchange(
    adapter: Any,
    target_records: list[dict[str, Any]],
    counterpart_index: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    span_index: dict[tuple[str, str], dict[str, Any]],
    live_span_cache: dict[str, dict[str, Any]],
    generation_cache: dict[str, dict[str, Any]],
    control_span_cache: dict[tuple[str, int, int], dict[str, Any] | None],
    control_window_cache: dict[tuple[str, int, int], dict[str, Any] | None],
    realism_bank: dict[str, list[list[float]]],
    layer_index: int,
    max_length: int,
    max_new_tokens: int,
    num_beams: int,
    non_role_policy: str,
    strongest_control_aggregation: str,
    strongest_control_priority: tuple[str, ...],
    preferred_counterpart_strata: tuple[str, ...],
    counterpart_same_split_only: bool,
    operationalization_family: str,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for target_item in target_records:
        target_span_map = ensure_item_span_map(
            adapter=adapter,
            item=target_item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        baseline = ensure_generation_record(
            adapter=adapter,
            item=target_item,
            generation_cache=generation_cache,
            max_new_tokens=max_new_tokens,
        )
        for role in CORE_ROLES:
            target_filler = target_span_map.get(f"filler::{role}")
            if target_filler is None:
                continue
            counterpart = find_exact_role_counterpart(
                record=target_item,
                role=role,
                counterpart_index=counterpart_index,
                preferred_strata=preferred_counterpart_strata,
                same_split_only=counterpart_same_split_only,
            )
            if counterpart is None:
                continue

            counterpart_span_map = ensure_item_span_map(
                adapter=adapter,
                item=counterpart,
                span_index=span_index,
                live_span_cache=live_span_cache,
                layer_index=layer_index,
                max_length=max_length,
            )
            donor_filler = counterpart_span_map.get(f"filler::{role}")
            wrong_role = opposite_role(role)
            wrong_role_filler = counterpart_span_map.get(f"filler::{wrong_role}")
            non_role_record = ensure_non_role_span_record(
                adapter=adapter,
                item=counterpart,
                item_span_map=counterpart_span_map,
                desired_token_count=len(list(target_filler["token_indices"])),
                layer_index=layer_index,
                max_length=max_length,
                control_span_cache=control_span_cache,
                control_window_cache=control_window_cache,
                non_role_policy=non_role_policy,
                seed=stable_seed(seed, "non_role", target_item["item_id"], counterpart["item_id"], role),
            )
            if donor_filler is None or wrong_role_filler is None or non_role_record is None:
                continue

            actual = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                patch_specs=[
                    {
                        "token_indices": list(target_filler["token_indices"]),
                        "replacement_vector": list(donor_filler["vector"]),
                    }
                ],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            wrong_role_output = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                patch_specs=[
                    {
                        "token_indices": list(target_filler["token_indices"]),
                        "replacement_vector": list(wrong_role_filler["vector"]),
                    }
                ],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            non_role_output = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                patch_specs=[
                    {
                        "token_indices": list(target_filler["token_indices"]),
                        "replacement_vector": list(non_role_record["vector"]),
                    }
                ],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            shuffled_vector = shuffle_vector(
                list(donor_filler["vector"]),
                seed=stable_seed(seed, "m2", target_item["item_id"], role),
            )
            shuffled_output = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                patch_specs=[
                    {
                        "token_indices": list(target_filler["token_indices"]),
                        "replacement_vector": shuffled_vector,
                    }
                ],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )

            counterfactual_gold = str(counterpart["target_form"])
            actual_metrics = score_prediction(actual["generated_text"], counterfactual_gold)
            wrong_role_metrics = score_prediction(wrong_role_output["generated_text"], counterfactual_gold)
            non_role_metrics = score_prediction(non_role_output["generated_text"], counterfactual_gold)
            shuffled_metrics = score_prediction(shuffled_output["generated_text"], counterfactual_gold)
            strongest_control = aggregate_control_value(
                {
                    "wrong_role": float(wrong_role_metrics["clause_f1"]),
                    "non_role": float(non_role_metrics["clause_f1"]),
                    "shuffled": float(shuffled_metrics["clause_f1"]),
                },
                method=strongest_control_aggregation,
                priority=strongest_control_priority,
            )
            strongest_control_exact = aggregate_control_value(
                {
                    "wrong_role": int(bool(wrong_role_metrics["exact_match"])),
                    "non_role": int(bool(non_role_metrics["exact_match"])),
                    "shuffled": int(bool(shuffled_metrics["exact_match"])),
                },
                method=strongest_control_aggregation,
                priority=strongest_control_priority,
            )

            records.append(
                {
                    "module": "M2_generation_interchange",
                    "target_item_id": target_item["item_id"],
                    "target_failure_category": target_item["failure_category"],
                    "generalization_case": target_item["generalization_case"],
                    "role": role,
                    "baseline_prediction": baseline["prediction"],
                    "baseline_metrics": baseline["metrics"],
                    "counterpart_item_id": counterpart["item_id"],
                    "counterpart_stratum": counterpart["stratum"],
                    "counterpart_gold": counterfactual_gold,
                    "actual_prediction": actual["generated_text"],
                    "actual_metrics": actual_metrics,
                    "wrong_role_prediction": wrong_role_output["generated_text"],
                    "wrong_role_metrics": wrong_role_metrics,
                    "non_role_prediction": non_role_output["generated_text"],
                    "non_role_metrics": non_role_metrics,
                    "shuffled_prediction": shuffled_output["generated_text"],
                    "shuffled_metrics": shuffled_metrics,
                    "actual_counterfactual_clause_f1": float(actual_metrics["clause_f1"]),
                    "wrong_role_counterfactual_clause_f1": float(wrong_role_metrics["clause_f1"]),
                    "non_role_counterfactual_clause_f1": float(non_role_metrics["clause_f1"]),
                    "shuffled_counterfactual_clause_f1": float(shuffled_metrics["clause_f1"]),
                    "strongest_control_counterfactual_clause_f1": round(strongest_control, 6),
                    "actual_counterfactual_exact_match": bool(actual_metrics["exact_match"]),
                    "wrong_role_counterfactual_exact_match": bool(wrong_role_metrics["exact_match"]),
                    "non_role_counterfactual_exact_match": bool(non_role_metrics["exact_match"]),
                    "shuffled_counterfactual_exact_match": bool(shuffled_metrics["exact_match"]),
                    "strongest_control_counterfactual_exact_match": bool(strongest_control_exact),
                    "actual_vs_wrong_role_advantage_clause_f1": round(
                        float(actual_metrics["clause_f1"]) - float(wrong_role_metrics["clause_f1"]),
                        6,
                    ),
                    "actual_vs_non_role_advantage_clause_f1": round(
                        float(actual_metrics["clause_f1"]) - float(non_role_metrics["clause_f1"]),
                        6,
                    ),
                    "actual_vs_shuffled_advantage_clause_f1": round(
                        float(actual_metrics["clause_f1"]) - float(shuffled_metrics["clause_f1"]),
                        6,
                    ),
                    "actual_vs_strongest_control_advantage_clause_f1": round(
                        float(actual_metrics["clause_f1"]) - strongest_control,
                        6,
                    ),
                    **build_patch_realism_snapshot(
                        target_site_key=f"filler::{role}",
                        target_vector=list(target_filler["vector"]),
                        actual_vector=list(donor_filler["vector"]),
                        wrong_role_vector=list(wrong_role_filler["vector"]),
                        non_role_vector=list(non_role_record["vector"]),
                        shuffled_vector=shuffled_vector,
                        realism_bank=realism_bank,
                    ),
                }
            )
    return records


def run_generation_necessity(
    adapter: Any,
    candidate_records: list[dict[str, Any]],
    span_index: dict[tuple[str, str], dict[str, Any]],
    live_span_cache: dict[str, dict[str, Any]],
    generation_cache: dict[str, dict[str, Any]],
    control_window_cache: dict[tuple[str, int, int], dict[str, Any] | None],
    realism_bank: dict[str, list[list[float]]],
    layer_index: int,
    max_length: int,
    max_new_tokens: int,
    non_role_policy: str,
    strongest_control_aggregation: str,
    num_beams: int,
    operationalization_family: str,
    geometry_unit_class: str,
    geometry_object_class: str | None,
    geometry_object_spec: str | None,
    seed: int,
) -> list[dict[str, Any]]:
    if infer_operationalization_family(operationalization_family) == GEOMETRY_PROTOCOL_FAMILY:
        return run_generation_geometry_necessity(
            adapter=adapter,
            candidate_records=candidate_records,
            span_index=span_index,
            live_span_cache=live_span_cache,
            generation_cache=generation_cache,
            control_window_cache=control_window_cache,
            layer_index=layer_index,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            non_role_policy=non_role_policy,
            strongest_control_aggregation=strongest_control_aggregation,
            num_beams=num_beams,
            unit_class=geometry_unit_class,
            geometry_object_class=geometry_object_class,
            geometry_object_spec=geometry_object_spec,
            seed=seed,
        )

    records: list[dict[str, Any]] = []
    for item in candidate_records:
        item_span_map = ensure_item_span_map(
            adapter=adapter,
            item=item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        baseline = ensure_generation_record(
            adapter=adapter,
            item=item,
            generation_cache=generation_cache,
            max_new_tokens=max_new_tokens,
        )
        for role in CORE_ROLES:
            role_record = item_span_map.get(role)
            wrong_role_record = item_span_map.get(opposite_role(role))
            non_role_window = ensure_non_role_window(
                adapter=adapter,
                item=item,
                item_span_map=item_span_map,
                desired_token_count=len(list(role_record["token_indices"])) if role_record is not None else 0,
                max_length=max_length,
                control_window_cache=control_window_cache,
                non_role_policy=non_role_policy,
                seed=stable_seed(seed, "necessity", item["item_id"], role),
            )
            if role_record is None or wrong_role_record is None or non_role_window is None:
                continue

            role_ablated = adapter.generate_with_intervention(
                input_text=str(item["input_text"]),
                layer_index=layer_index,
                ablation_specs=[{"token_indices": list(role_record["token_indices"])}],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            wrong_role_ablated = adapter.generate_with_intervention(
                input_text=str(item["input_text"]),
                layer_index=layer_index,
                ablation_specs=[{"token_indices": list(wrong_role_record["token_indices"])}],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            non_role_ablated = adapter.generate_with_intervention(
                input_text=str(item["input_text"]),
                layer_index=layer_index,
                ablation_specs=[{"token_indices": list(non_role_window["token_indices"])}],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            role_metrics = score_prediction(role_ablated["generated_text"], str(item["target_form"]))
            wrong_role_metrics = score_prediction(wrong_role_ablated["generated_text"], str(item["target_form"]))
            non_role_metrics = score_prediction(non_role_ablated["generated_text"], str(item["target_form"]))
            baseline_metrics = dict(baseline["metrics"])

            role_drop = float(baseline_metrics["clause_f1"]) - float(role_metrics["clause_f1"])
            wrong_role_drop = float(baseline_metrics["clause_f1"]) - float(wrong_role_metrics["clause_f1"])
            non_role_drop = float(baseline_metrics["clause_f1"]) - float(non_role_metrics["clause_f1"])
            strongest_control_drop = aggregate_control_value(
                {
                    "wrong_role": wrong_role_drop,
                    "non_role": non_role_drop,
                },
                method=strongest_control_aggregation,
                priority=("wrong_role", "non_role"),
            )

            records.append(
                {
                    "module": "M3_generation_necessity",
                    "item_id": item["item_id"],
                    "stratum": item["stratum"],
                    "generalization_case": item["generalization_case"],
                    "role": role,
                    "baseline_prediction": baseline["prediction"],
                    "baseline_metrics": baseline_metrics,
                    "role_ablated_prediction": role_ablated["generated_text"],
                    "role_ablated_metrics": role_metrics,
                    "wrong_role_ablated_prediction": wrong_role_ablated["generated_text"],
                    "wrong_role_ablated_metrics": wrong_role_metrics,
                    "non_role_ablated_prediction": non_role_ablated["generated_text"],
                    "non_role_ablated_metrics": non_role_metrics,
                    "role_ablation_clause_f1_drop": round(role_drop, 6),
                    "wrong_role_clause_f1_drop": round(wrong_role_drop, 6),
                    "non_role_clause_f1_drop": round(non_role_drop, 6),
                    "strongest_control_clause_f1_drop": round(strongest_control_drop, 6),
                    "role_vs_wrong_role_gap_clause_f1": round(role_drop - wrong_role_drop, 6),
                    "role_vs_non_role_gap_clause_f1": round(role_drop - non_role_drop, 6),
                    "role_vs_strongest_control_gap_clause_f1": round(role_drop - strongest_control_drop, 6),
                }
            )
    return records


def run_generation_geometry_necessity(
    adapter: Any,
    candidate_records: list[dict[str, Any]],
    span_index: dict[tuple[str, str], dict[str, Any]],
    live_span_cache: dict[str, dict[str, Any]],
    generation_cache: dict[str, dict[str, Any]],
    control_window_cache: dict[tuple[str, int, int], dict[str, Any] | None],
    layer_index: int,
    max_length: int,
    max_new_tokens: int,
    non_role_policy: str,
    strongest_control_aggregation: str,
    num_beams: int,
    unit_class: str,
    geometry_object_class: str | None,
    geometry_object_spec: str | None,
    seed: int,
) -> list[dict[str, Any]]:
    unit = infer_geometry_unit_class(unit_class)
    object_class = infer_geometry_object_class(geometry_object_class)
    object_id = _geometry_object_spec_id(geometry_object_class, geometry_object_spec)
    operationalization = geometry_operationalization_label(unit)
    records: list[dict[str, Any]] = []
    for item in candidate_records:
        item_span_map = ensure_item_span_map(
            adapter=adapter,
            item=item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        baseline = ensure_generation_record(
            adapter=adapter,
            item=item,
            generation_cache=generation_cache,
            max_new_tokens=max_new_tokens,
        )
        for role in CORE_ROLES:
            role_record = item_span_map.get(role)
            if role_record is None:
                continue

            non_role_window = ensure_non_role_window(
                adapter=adapter,
                item=item,
                item_span_map=item_span_map,
                desired_token_count=len(list(role_record["token_indices"])),
                max_length=max_length,
                control_window_cache=control_window_cache,
                non_role_policy=non_role_policy,
                seed=stable_seed(seed, "geometry_phase16_non_role", item["item_id"], role),
            )
            plan = geometry_object_necessity_plan(
                item=item,
                item_span_map=item_span_map,
                role=role,
                unit_class=unit,
                geometry_object_class=object_class,
                geometry_object_spec=geometry_object_spec,
                non_role_window=non_role_window,
                seed=seed,
            )

            actual_output = adapter.generate_with_intervention(
                input_text=str(item["input_text"]),
                layer_index=layer_index,
                ablation_specs=list(plan["actual_ablation_specs"]),
                control_overrides=plan["actual_overrides"],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            bottleneck_output = adapter.generate_with_intervention(
                input_text=str(item["input_text"]),
                layer_index=layer_index,
                ablation_specs=list(plan["wrong_ablation_specs"]),
                control_overrides=plan["wrong_overrides"],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            readout_mask_output = adapter.generate_with_intervention(
                input_text=str(item["input_text"]),
                layer_index=layer_index,
                ablation_specs=list(plan["non_role_ablation_specs"]),
                control_overrides=plan["non_role_overrides"],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )

            actual_metrics = score_prediction(actual_output["generated_text"], str(item["target_form"]))
            bottleneck_metrics = score_prediction(bottleneck_output["generated_text"], str(item["target_form"]))
            readout_mask_metrics = score_prediction(readout_mask_output["generated_text"], str(item["target_form"]))
            baseline_metrics = dict(baseline["metrics"])

            actual_drop = float(baseline_metrics["clause_f1"]) - float(actual_metrics["clause_f1"])
            bottleneck_drop = float(baseline_metrics["clause_f1"]) - float(bottleneck_metrics["clause_f1"])
            readout_mask_drop = float(baseline_metrics["clause_f1"]) - float(readout_mask_metrics["clause_f1"])
            strongest_control_drop = aggregate_control_value(
                {
                    "wrong_role": bottleneck_drop,
                    "non_role": readout_mask_drop,
                },
                method=strongest_control_aggregation,
                priority=("wrong_role", "non_role"),
            )

            records.append(
                {
                    "module": "M3_generation_necessity",
                    "operationalization": operationalization,
                    "geometry_object_class": object_class,
                    "geometry_object_id": plan["object_id"] or object_id,
                    "geometry_object_spec": geometry_object_spec,
                    "actual_intervention_label": plan["actual_label"],
                    "wrong_role_control_label": plan["wrong_label"],
                    "non_role_control_label": plan["non_role_label"],
                    "item_id": item["item_id"],
                    "stratum": item["stratum"],
                    "generalization_case": item["generalization_case"],
                    "role": role,
                    "baseline_prediction": baseline["prediction"],
                    "baseline_metrics": baseline_metrics,
                    "role_ablated_prediction": actual_output["generated_text"],
                    "role_ablated_metrics": actual_metrics,
                    "wrong_role_ablated_prediction": bottleneck_output["generated_text"],
                    "wrong_role_ablated_metrics": bottleneck_metrics,
                    "non_role_ablated_prediction": readout_mask_output["generated_text"],
                    "non_role_ablated_metrics": readout_mask_metrics,
                    "role_ablation_clause_f1_drop": round(actual_drop, 6),
                    "wrong_role_clause_f1_drop": round(bottleneck_drop, 6),
                    "non_role_clause_f1_drop": round(readout_mask_drop, 6),
                    "strongest_control_clause_f1_drop": round(strongest_control_drop, 6),
                    "role_vs_wrong_role_gap_clause_f1": round(actual_drop - bottleneck_drop, 6),
                    "role_vs_non_role_gap_clause_f1": round(actual_drop - readout_mask_drop, 6),
                    "role_vs_strongest_control_gap_clause_f1": round(actual_drop - strongest_control_drop, 6),
                }
            )
    return records


def run_generation_rescue(
    adapter: Any,
    target_records: list[dict[str, Any]],
    counterpart_index: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    span_index: dict[tuple[str, str], dict[str, Any]],
    live_span_cache: dict[str, dict[str, Any]],
    generation_cache: dict[str, dict[str, Any]],
    control_span_cache: dict[tuple[str, int, int], dict[str, Any] | None],
    control_window_cache: dict[tuple[str, int, int], dict[str, Any] | None],
    realism_bank: dict[str, list[list[float]]],
    layer_index: int,
    max_length: int,
    max_new_tokens: int,
    num_beams: int,
    non_role_policy: str,
    strongest_control_aggregation: str,
    strongest_control_priority: tuple[str, ...],
    preferred_counterpart_strata: tuple[str, ...],
    counterpart_same_split_only: bool,
    operationalization_family: str,
    geometry_unit_class: str,
    geometry_object_class: str | None,
    geometry_object_spec: str | None,
    seed: int,
) -> list[dict[str, Any]]:
    if infer_operationalization_family(operationalization_family) == GEOMETRY_PROTOCOL_FAMILY:
        return run_generation_geometry_rescue(
            adapter=adapter,
            target_records=target_records,
            counterpart_index=counterpart_index,
            span_index=span_index,
            live_span_cache=live_span_cache,
            generation_cache=generation_cache,
            control_span_cache=control_span_cache,
            control_window_cache=control_window_cache,
            realism_bank=realism_bank,
            layer_index=layer_index,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            non_role_policy=non_role_policy,
            strongest_control_aggregation=strongest_control_aggregation,
            strongest_control_priority=strongest_control_priority,
            preferred_counterpart_strata=preferred_counterpart_strata,
            counterpart_same_split_only=counterpart_same_split_only,
            unit_class=geometry_unit_class,
            geometry_object_class=geometry_object_class,
            geometry_object_spec=geometry_object_spec,
            seed=seed,
        )

    records: list[dict[str, Any]] = []
    for target_item in target_records:
        target_span_map = ensure_item_span_map(
            adapter=adapter,
            item=target_item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        baseline = ensure_generation_record(
            adapter=adapter,
            item=target_item,
            generation_cache=generation_cache,
            max_new_tokens=max_new_tokens,
        )
        for role in CORE_ROLES:
            target_role = target_span_map.get(role)
            if target_role is None:
                continue
            counterpart = find_exact_role_counterpart(
                record=target_item,
                role=role,
                counterpart_index=counterpart_index,
                preferred_strata=preferred_counterpart_strata,
                same_split_only=counterpart_same_split_only,
            )
            if counterpart is None:
                continue
            counterpart_span_map = ensure_item_span_map(
                adapter=adapter,
                item=counterpart,
                span_index=span_index,
                live_span_cache=live_span_cache,
                layer_index=layer_index,
                max_length=max_length,
            )
            donor_role = counterpart_span_map.get(role)
            wrong_role_record = counterpart_span_map.get(opposite_role(role))
            non_role_record = ensure_non_role_span_record(
                adapter=adapter,
                item=counterpart,
                item_span_map=counterpart_span_map,
                desired_token_count=len(list(target_role["token_indices"])),
                layer_index=layer_index,
                max_length=max_length,
                control_span_cache=control_span_cache,
                control_window_cache=control_window_cache,
                non_role_policy=non_role_policy,
                seed=stable_seed(seed, "non_role_rescue", target_item["item_id"], counterpart["item_id"], role),
            )
            if donor_role is None or wrong_role_record is None or non_role_record is None:
                continue

            actual = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                patch_specs=[
                    {
                        "token_indices": list(target_role["token_indices"]),
                        "replacement_vector": list(donor_role["vector"]),
                    }
                ],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            wrong_role_output = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                patch_specs=[
                    {
                        "token_indices": list(target_role["token_indices"]),
                        "replacement_vector": list(wrong_role_record["vector"]),
                    }
                ],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            non_role_output = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                patch_specs=[
                    {
                        "token_indices": list(target_role["token_indices"]),
                        "replacement_vector": list(non_role_record["vector"]),
                    }
                ],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            shuffled_vector = shuffle_vector(
                list(donor_role["vector"]),
                seed=stable_seed(seed, "m4", target_item["item_id"], role),
            )
            shuffled_output = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                patch_specs=[
                    {
                        "token_indices": list(target_role["token_indices"]),
                        "replacement_vector": shuffled_vector,
                    }
                ],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            actual_metrics = score_prediction(actual["generated_text"], str(target_item["target_form"]))
            wrong_role_metrics = score_prediction(wrong_role_output["generated_text"], str(target_item["target_form"]))
            non_role_metrics = score_prediction(non_role_output["generated_text"], str(target_item["target_form"]))
            shuffled_metrics = score_prediction(shuffled_output["generated_text"], str(target_item["target_form"]))
            baseline_metrics = dict(baseline["metrics"])

            actual_gain = float(actual_metrics["clause_f1"]) - float(baseline_metrics["clause_f1"])
            wrong_role_gain = float(wrong_role_metrics["clause_f1"]) - float(baseline_metrics["clause_f1"])
            non_role_gain = float(non_role_metrics["clause_f1"]) - float(baseline_metrics["clause_f1"])
            shuffled_gain = float(shuffled_metrics["clause_f1"]) - float(baseline_metrics["clause_f1"])
            strongest_control_gain = aggregate_control_value(
                {
                    "wrong_role": wrong_role_gain,
                    "non_role": non_role_gain,
                    "shuffled": shuffled_gain,
                },
                method=strongest_control_aggregation,
                priority=strongest_control_priority,
            )
            strongest_control_rescued = aggregate_control_value(
                {
                    "wrong_role": int((not bool(baseline_metrics["exact_match"])) and bool(wrong_role_metrics["exact_match"])),
                    "non_role": int((not bool(baseline_metrics["exact_match"])) and bool(non_role_metrics["exact_match"])),
                    "shuffled": int((not bool(baseline_metrics["exact_match"])) and bool(shuffled_metrics["exact_match"])),
                },
                method=strongest_control_aggregation,
                priority=strongest_control_priority,
            )

            records.append(
                {
                    "module": "M4_generation_rescue",
                    "target_item_id": target_item["item_id"],
                    "target_failure_category": target_item["failure_category"],
                    "generalization_case": target_item["generalization_case"],
                    "role": role,
                    "counterpart_item_id": counterpart["item_id"],
                    "counterpart_stratum": counterpart["stratum"],
                    "baseline_prediction": baseline["prediction"],
                    "baseline_metrics": baseline_metrics,
                    "actual_prediction": actual["generated_text"],
                    "actual_metrics": actual_metrics,
                    "wrong_role_prediction": wrong_role_output["generated_text"],
                    "wrong_role_metrics": wrong_role_metrics,
                    "non_role_prediction": non_role_output["generated_text"],
                    "non_role_metrics": non_role_metrics,
                    "shuffled_prediction": shuffled_output["generated_text"],
                    "shuffled_metrics": shuffled_metrics,
                    "actual_clause_f1_gain": round(actual_gain, 6),
                    "wrong_role_clause_f1_gain": round(wrong_role_gain, 6),
                    "non_role_clause_f1_gain": round(non_role_gain, 6),
                    "shuffled_clause_f1_gain": round(shuffled_gain, 6),
                    "strongest_control_clause_f1_gain": round(strongest_control_gain, 6),
                    "actual_rescued_from_incorrect": (not bool(baseline_metrics["exact_match"])) and bool(actual_metrics["exact_match"]),
                    "wrong_role_rescued_from_incorrect": (not bool(baseline_metrics["exact_match"])) and bool(wrong_role_metrics["exact_match"]),
                    "non_role_rescued_from_incorrect": (not bool(baseline_metrics["exact_match"])) and bool(non_role_metrics["exact_match"]),
                    "shuffled_rescued_from_incorrect": (not bool(baseline_metrics["exact_match"])) and bool(shuffled_metrics["exact_match"]),
                    "strongest_control_rescued_from_incorrect": bool(strongest_control_rescued),
                    "actual_vs_wrong_role_advantage_clause_f1": round(actual_gain - wrong_role_gain, 6),
                    "actual_vs_non_role_advantage_clause_f1": round(actual_gain - non_role_gain, 6),
                    "actual_vs_shuffled_advantage_clause_f1": round(actual_gain - shuffled_gain, 6),
                    "actual_vs_strongest_control_advantage_clause_f1": round(actual_gain - strongest_control_gain, 6),
                    **build_patch_realism_snapshot(
                        target_site_key=role,
                        target_vector=list(target_role["vector"]),
                        actual_vector=list(donor_role["vector"]),
                        wrong_role_vector=list(wrong_role_record["vector"]),
                        non_role_vector=list(non_role_record["vector"]),
                        shuffled_vector=shuffled_vector,
                        realism_bank=realism_bank,
                    ),
                }
            )
    return records


def run_generation_geometry_rescue(
    adapter: Any,
    target_records: list[dict[str, Any]],
    counterpart_index: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    span_index: dict[tuple[str, str], dict[str, Any]],
    live_span_cache: dict[str, dict[str, Any]],
    generation_cache: dict[str, dict[str, Any]],
    control_span_cache: dict[tuple[str, int, int], dict[str, Any] | None],
    control_window_cache: dict[tuple[str, int, int], dict[str, Any] | None],
    realism_bank: dict[str, list[list[float]]],
    layer_index: int,
    max_length: int,
    max_new_tokens: int,
    num_beams: int,
    non_role_policy: str,
    strongest_control_aggregation: str,
    strongest_control_priority: tuple[str, ...],
    preferred_counterpart_strata: tuple[str, ...],
    counterpart_same_split_only: bool,
    unit_class: str,
    geometry_object_class: str | None,
    geometry_object_spec: str | None,
    seed: int,
) -> list[dict[str, Any]]:
    unit = infer_geometry_unit_class(unit_class)
    object_class = infer_geometry_object_class(geometry_object_class)
    object_id = _geometry_object_spec_id(geometry_object_class, geometry_object_spec)
    operationalization = geometry_operationalization_label(unit)
    records: list[dict[str, Any]] = []
    for target_item in target_records:
        target_span_map = ensure_item_span_map(
            adapter=adapter,
            item=target_item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        baseline = ensure_generation_record(
            adapter=adapter,
            item=target_item,
            generation_cache=generation_cache,
            max_new_tokens=max_new_tokens,
        )
        for role in CORE_ROLES:
            target_role = target_span_map.get(role)
            if target_role is None:
                continue
            counterpart = find_exact_role_counterpart(
                record=target_item,
                role=role,
                counterpart_index=counterpart_index,
                preferred_strata=preferred_counterpart_strata,
                same_split_only=counterpart_same_split_only,
            )
            if counterpart is None:
                continue
            counterpart_span_map = ensure_item_span_map(
                adapter=adapter,
                item=counterpart,
                span_index=span_index,
                live_span_cache=live_span_cache,
                layer_index=layer_index,
                max_length=max_length,
            )
            donor_role = counterpart_span_map.get(role)
            wrong_role_record = counterpart_span_map.get(opposite_role(role))
            non_role_record = ensure_non_role_span_record(
                adapter=adapter,
                item=counterpart,
                item_span_map=counterpart_span_map,
                desired_token_count=len(list(target_role["token_indices"])),
                layer_index=layer_index,
                max_length=max_length,
                control_span_cache=control_span_cache,
                control_window_cache=control_window_cache,
                non_role_policy=non_role_policy,
                seed=stable_seed(seed, "geometry_non_role_rescue", target_item["item_id"], counterpart["item_id"], role),
            )
            if donor_role is None or wrong_role_record is None or non_role_record is None:
                continue

            item_id = str(target_item["item_id"])
            plan = geometry_object_rescue_plan(
                item_id=item_id,
                role=role,
                unit_class=unit,
                geometry_object_class=object_class,
                geometry_object_spec=geometry_object_spec,
                target_role=target_role,
                donor_role=donor_role,
                wrong_role_record=wrong_role_record,
                non_role_record=non_role_record,
                target_span_map=target_span_map,
                counterpart_span_map=counterpart_span_map,
                seed=seed,
            )

            path_plus_alignment = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                patch_specs=list(plan["actual_patch_specs"]),
                control_overrides=plan["actual_overrides"],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            alignment_only_output = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                patch_specs=list(plan["object_only_patch_specs"]),
                control_overrides=plan["object_only_overrides"],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            wrong_role_output = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                patch_specs=list(plan["wrong_object_patch_specs"]),
                control_overrides=plan["wrong_object_overrides"],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            non_role_output = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                patch_specs=list(plan["non_role_patch_specs"]),
                control_overrides=plan["non_role_overrides"],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            shuffled_vector = shuffle_vector(
                list(donor_role["vector"]),
                seed=stable_seed(seed, "geometry_m4", target_item["item_id"], role),
            )
            shuffled_output = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                patch_specs=[
                    {
                        "token_indices": list(target_role["token_indices"]),
                        "replacement_vector": shuffled_vector,
                    }
                ],
                control_overrides=plan["actual_overrides"],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            shuffled_alignment_output = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                patch_specs=[
                    {
                        "token_indices": list(target_role["token_indices"]),
                        "replacement_vector": shuffled_vector,
                    }
                ],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            wrong_path_output = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                control_overrides=plan["wrong_path_overrides"],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            generic_decoder_gain_output = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                control_overrides=plan["generic_gain_overrides"],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            path_only_output = adapter.generate_with_intervention(
                input_text=str(target_item["input_text"]),
                layer_index=layer_index,
                control_overrides=plan["path_only_overrides"],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )

            actual_metrics = score_prediction(path_plus_alignment["generated_text"], str(target_item["target_form"]))
            alignment_only_metrics = score_prediction(alignment_only_output["generated_text"], str(target_item["target_form"]))
            wrong_role_metrics = score_prediction(wrong_role_output["generated_text"], str(target_item["target_form"]))
            non_role_metrics = score_prediction(non_role_output["generated_text"], str(target_item["target_form"]))
            shuffled_metrics = score_prediction(shuffled_output["generated_text"], str(target_item["target_form"]))
            shuffled_alignment_metrics = score_prediction(
                shuffled_alignment_output["generated_text"],
                str(target_item["target_form"]),
            )
            wrong_path_metrics = score_prediction(wrong_path_output["generated_text"], str(target_item["target_form"]))
            generic_decoder_gain_metrics = score_prediction(
                generic_decoder_gain_output["generated_text"],
                str(target_item["target_form"]),
            )
            path_only_metrics = score_prediction(path_only_output["generated_text"], str(target_item["target_form"]))
            baseline_metrics = dict(baseline["metrics"])

            actual_gain = float(actual_metrics["clause_f1"]) - float(baseline_metrics["clause_f1"])
            alignment_only_gain = float(alignment_only_metrics["clause_f1"]) - float(baseline_metrics["clause_f1"])
            wrong_role_gain = float(wrong_role_metrics["clause_f1"]) - float(baseline_metrics["clause_f1"])
            non_role_gain = float(non_role_metrics["clause_f1"]) - float(baseline_metrics["clause_f1"])
            shuffled_gain = float(shuffled_metrics["clause_f1"]) - float(baseline_metrics["clause_f1"])
            shuffled_alignment_gain = float(shuffled_alignment_metrics["clause_f1"]) - float(baseline_metrics["clause_f1"])
            wrong_path_gain = float(wrong_path_metrics["clause_f1"]) - float(baseline_metrics["clause_f1"])
            generic_decoder_gain = float(generic_decoder_gain_metrics["clause_f1"]) - float(baseline_metrics["clause_f1"])
            path_only_gain = float(path_only_metrics["clause_f1"]) - float(baseline_metrics["clause_f1"])

            control_values = {
                "alignment_only": alignment_only_gain,
                "wrong_role": wrong_role_gain,
                "non_role": non_role_gain,
                "shuffled": shuffled_gain,
                "shuffled_alignment": shuffled_alignment_gain,
                "wrong_path": wrong_path_gain,
                "generic_decoder_gain": generic_decoder_gain,
                "path_only": path_only_gain,
            }
            strongest_priority = tuple(strongest_control_priority)
            for control_name in (
                "alignment_only",
                "shuffled_alignment",
                "wrong_path",
                "generic_decoder_gain",
                "path_only",
            ):
                if control_name not in strongest_priority:
                    strongest_priority = strongest_priority + (control_name,)
            strongest_control_gain = aggregate_control_value(
                control_values,
                method=strongest_control_aggregation,
                priority=strongest_priority,
            )
            strongest_control_rescued = aggregate_control_value(
                {
                    "alignment_only": int(
                        (not bool(baseline_metrics["exact_match"])) and bool(alignment_only_metrics["exact_match"])
                    ),
                    "wrong_role": int((not bool(baseline_metrics["exact_match"])) and bool(wrong_role_metrics["exact_match"])),
                    "non_role": int((not bool(baseline_metrics["exact_match"])) and bool(non_role_metrics["exact_match"])),
                    "shuffled": int((not bool(baseline_metrics["exact_match"])) and bool(shuffled_metrics["exact_match"])),
                    "shuffled_alignment": int(
                        (not bool(baseline_metrics["exact_match"])) and bool(shuffled_alignment_metrics["exact_match"])
                    ),
                    "wrong_path": int(
                        (not bool(baseline_metrics["exact_match"])) and bool(wrong_path_metrics["exact_match"])
                    ),
                    "generic_decoder_gain": int(
                        (not bool(baseline_metrics["exact_match"])) and bool(generic_decoder_gain_metrics["exact_match"])
                    ),
                    "path_only": int((not bool(baseline_metrics["exact_match"])) and bool(path_only_metrics["exact_match"])),
                },
                method=strongest_control_aggregation,
                priority=strongest_priority,
            )

            records.append(
                {
                    "module": "M4_generation_rescue",
                    "operationalization": operationalization,
                    "geometry_object_class": object_class,
                    "geometry_object_id": plan["object_id"] or object_id,
                    "geometry_object_spec": geometry_object_spec,
                    "actual_intervention_label": plan["actual_label"],
                    "alignment_only_control_label": plan["object_only_label"],
                    "path_only_control_label": plan["path_only_label"],
                    "wrong_path_control_label": plan["wrong_path_label"],
                    "generic_decoder_gain_control_label": "generic_cross_attention_gain_only",
                    "shuffled_alignment_control_label": "shuffled_alignment_only",
                    "target_item_id": target_item["item_id"],
                    "target_failure_category": target_item["failure_category"],
                    "generalization_case": target_item["generalization_case"],
                    "role": role,
                    "counterpart_item_id": counterpart["item_id"],
                    "counterpart_stratum": counterpart["stratum"],
                    "baseline_prediction": baseline["prediction"],
                    "baseline_metrics": baseline_metrics,
                    "actual_prediction": path_plus_alignment["generated_text"],
                    "actual_metrics": actual_metrics,
                    "alignment_only_prediction": alignment_only_output["generated_text"],
                    "alignment_only_metrics": alignment_only_metrics,
                    "wrong_role_prediction": wrong_role_output["generated_text"],
                    "wrong_role_metrics": wrong_role_metrics,
                    "non_role_prediction": non_role_output["generated_text"],
                    "non_role_metrics": non_role_metrics,
                    "shuffled_prediction": shuffled_output["generated_text"],
                    "shuffled_metrics": shuffled_metrics,
                    "shuffled_alignment_prediction": shuffled_alignment_output["generated_text"],
                    "shuffled_alignment_metrics": shuffled_alignment_metrics,
                    "wrong_path_prediction": wrong_path_output["generated_text"],
                    "wrong_path_metrics": wrong_path_metrics,
                    "generic_decoder_gain_prediction": generic_decoder_gain_output["generated_text"],
                    "generic_decoder_gain_metrics": generic_decoder_gain_metrics,
                    "path_only_prediction": path_only_output["generated_text"],
                    "path_only_metrics": path_only_metrics,
                    "actual_clause_f1_gain": round(actual_gain, 6),
                    "alignment_only_clause_f1_gain": round(alignment_only_gain, 6),
                    "wrong_role_clause_f1_gain": round(wrong_role_gain, 6),
                    "non_role_clause_f1_gain": round(non_role_gain, 6),
                    "shuffled_clause_f1_gain": round(shuffled_gain, 6),
                    "shuffled_alignment_clause_f1_gain": round(shuffled_alignment_gain, 6),
                    "wrong_path_clause_f1_gain": round(wrong_path_gain, 6),
                    "generic_decoder_gain_clause_f1_gain": round(generic_decoder_gain, 6),
                    "path_only_clause_f1_gain": round(path_only_gain, 6),
                    "strongest_control_clause_f1_gain": round(strongest_control_gain, 6),
                    "actual_rescued_from_incorrect": (not bool(baseline_metrics["exact_match"])) and bool(actual_metrics["exact_match"]),
                    "alignment_only_rescued_from_incorrect": (
                        (not bool(baseline_metrics["exact_match"])) and bool(alignment_only_metrics["exact_match"])
                    ),
                    "wrong_role_rescued_from_incorrect": (not bool(baseline_metrics["exact_match"])) and bool(wrong_role_metrics["exact_match"]),
                    "non_role_rescued_from_incorrect": (not bool(baseline_metrics["exact_match"])) and bool(non_role_metrics["exact_match"]),
                    "shuffled_rescued_from_incorrect": (not bool(baseline_metrics["exact_match"])) and bool(shuffled_metrics["exact_match"]),
                    "shuffled_alignment_rescued_from_incorrect": (
                        (not bool(baseline_metrics["exact_match"])) and bool(shuffled_alignment_metrics["exact_match"])
                    ),
                    "wrong_path_rescued_from_incorrect": (
                        (not bool(baseline_metrics["exact_match"])) and bool(wrong_path_metrics["exact_match"])
                    ),
                    "generic_decoder_gain_rescued_from_incorrect": (
                        (not bool(baseline_metrics["exact_match"])) and bool(generic_decoder_gain_metrics["exact_match"])
                    ),
                    "path_only_rescued_from_incorrect": (not bool(baseline_metrics["exact_match"])) and bool(path_only_metrics["exact_match"]),
                    "strongest_control_rescued_from_incorrect": bool(strongest_control_rescued),
                    "actual_vs_alignment_only_advantage_clause_f1": round(actual_gain - alignment_only_gain, 6),
                    "actual_vs_wrong_role_advantage_clause_f1": round(actual_gain - wrong_role_gain, 6),
                    "actual_vs_non_role_advantage_clause_f1": round(actual_gain - non_role_gain, 6),
                    "actual_vs_shuffled_advantage_clause_f1": round(actual_gain - shuffled_gain, 6),
                    "actual_vs_shuffled_alignment_advantage_clause_f1": round(
                        actual_gain - shuffled_alignment_gain,
                        6,
                    ),
                    "actual_vs_wrong_path_advantage_clause_f1": round(actual_gain - wrong_path_gain, 6),
                    "actual_vs_generic_decoder_gain_advantage_clause_f1": round(
                        actual_gain - generic_decoder_gain,
                        6,
                    ),
                    "actual_vs_path_only_advantage_clause_f1": round(actual_gain - path_only_gain, 6),
                    "actual_vs_strongest_control_advantage_clause_f1": round(actual_gain - strongest_control_gain, 6),
                    **build_patch_realism_snapshot(
                        target_site_key=role,
                        target_vector=list(target_role["vector"]),
                        actual_vector=list(donor_role["vector"]),
                        wrong_role_vector=list(wrong_role_record["vector"]),
                        non_role_vector=list(non_role_record["vector"]),
                        shuffled_vector=shuffled_vector,
                        realism_bank=realism_bank,
                    ),
                }
            )
    return records


def build_generation_item_level_linkage(
    m3_records: list[dict[str, Any]],
    m4_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for record in m3_records:
        key = (str(record["item_id"]), str(record["role"]))
        rows[key] = {
            "item_id": record["item_id"],
            "role": record["role"],
            "stratum": record["stratum"],
            "generalization_case": record["generalization_case"],
            "role_ablation_clause_f1_drop": record["role_ablation_clause_f1_drop"],
            "wrong_role_clause_f1_drop": record["wrong_role_clause_f1_drop"],
            "non_role_clause_f1_drop": record["non_role_clause_f1_drop"],
            "role_vs_strongest_control_gap_clause_f1": record["role_vs_strongest_control_gap_clause_f1"],
            "actual_rescued_from_incorrect": None,
            "rescue_vs_strongest_control_advantage_clause_f1": None,
        }
    for record in m4_records:
        key = (str(record["target_item_id"]), str(record["role"]))
        rows.setdefault(
            key,
            {
                "item_id": record["target_item_id"],
                "role": record["role"],
                "stratum": "fail",
                "generalization_case": record["generalization_case"],
                "role_ablation_clause_f1_drop": None,
                "wrong_role_clause_f1_drop": None,
                "non_role_clause_f1_drop": None,
                "role_vs_strongest_control_gap_clause_f1": None,
                "actual_rescued_from_incorrect": None,
                "rescue_vs_strongest_control_advantage_clause_f1": None,
            },
        )
        rows[key]["actual_rescued_from_incorrect"] = bool(record["actual_rescued_from_incorrect"])
        rows[key]["rescue_vs_strongest_control_advantage_clause_f1"] = record[
            "actual_vs_strongest_control_advantage_clause_f1"
        ]
    return list(rows.values())


def summarize_generation_phase4(
    candidate_records: list[dict[str, Any]],
    m2_records: list[dict[str, Any]],
    m3_records: list[dict[str, Any]],
    m4_records: list[dict[str, Any]],
    linkage_records: list[dict[str, Any]],
    operationalization_family: str,
    geometry_unit_class: str,
    geometry_object_class: str | None,
    geometry_object_spec: str | None,
    geometry_plant_kind: str,
    geometry_plant_spec: str | None,
    seed: int,
) -> dict[str, Any]:
    candidate_by_stratum: dict[str, int] = {}
    for record in candidate_records:
        candidate_by_stratum[str(record["stratum"])] = candidate_by_stratum.get(str(record["stratum"]), 0) + 1

    return {
        "candidate_counts_by_stratum": dict(sorted(candidate_by_stratum.items())),
        "operationalization_family": infer_operationalization_family(operationalization_family),
        "geometry_unit_class": infer_geometry_unit_class(geometry_unit_class),
        "geometry_object_class": infer_geometry_object_class(geometry_object_class),
        "geometry_object_spec": geometry_object_spec,
        "geometry_plant_kind": infer_geometry_plant_kind(geometry_plant_kind),
        "geometry_plant_spec": normalize_geometry_plant_spec(geometry_plant_kind, geometry_plant_spec),
        "modules": {
            "M2_generation_interchange": summarize_interchange_module(
                records=m2_records,
                seed=stable_seed(seed, "summary", "m2"),
            ),
            "M3_generation_necessity": summarize_necessity_module(
                records=m3_records,
                seed=stable_seed(seed, "summary", "m3"),
            ),
            "M4_generation_rescue": summarize_rescue_module(
                records=m4_records,
                seed=stable_seed(seed, "summary", "m4"),
            ),
            "M5_item_level_linkage": {
                "count": len(linkage_records),
                "fail_items_with_rescue_advantage": sum(
                    1
                    for record in linkage_records
                    if str(record.get("stratum")) == "fail"
                    and float(record.get("rescue_vs_strongest_control_advantage_clause_f1") or 0.0) > 0.0
                ),
            },
        },
    }


def summarize_interchange_module(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "actual_mean": None,
            "wrong_role_mean": None,
            "non_role_mean": None,
            "shuffled_mean": None,
            "strongest_control_mean": None,
            "actual_binary_rate": None,
            "strongest_control_binary_rate": None,
            "actual_vs_wrong_role_advantage_mean": None,
            "actual_vs_non_role_advantage_mean": None,
            "actual_vs_shuffled_advantage_mean": None,
            "actual_vs_strongest_control_advantage_mean": None,
            "bootstrap_ci": None,
            "actual_vs_strongest_control_permutation_p_value": None,
            "actual_vs_wrong_role_permutation_p_value": None,
            "actual_vs_non_role_permutation_p_value": None,
            "actual_vs_shuffled_permutation_p_value": None,
        }
    strongest_diffs = [float(record["actual_vs_strongest_control_advantage_clause_f1"]) for record in records]
    wrong_role_diffs = [float(record["actual_vs_wrong_role_advantage_clause_f1"]) for record in records]
    non_role_diffs = [float(record["actual_vs_non_role_advantage_clause_f1"]) for record in records]
    shuffled_diffs = [float(record["actual_vs_shuffled_advantage_clause_f1"]) for record in records]
    return {
        "count": len(records),
        "actual_mean": round(mean(float(record["actual_counterfactual_clause_f1"]) for record in records), 6),
        "wrong_role_mean": round(mean(float(record["wrong_role_counterfactual_clause_f1"]) for record in records), 6),
        "non_role_mean": round(mean(float(record["non_role_counterfactual_clause_f1"]) for record in records), 6),
        "shuffled_mean": round(mean(float(record["shuffled_counterfactual_clause_f1"]) for record in records), 6),
        "strongest_control_mean": round(
            mean(float(record["strongest_control_counterfactual_clause_f1"]) for record in records),
            6,
        ),
        "actual_binary_rate": round(
            mean(float(bool(record["actual_counterfactual_exact_match"])) for record in records),
            6,
        ),
        "strongest_control_binary_rate": round(
            mean(float(bool(record["strongest_control_counterfactual_exact_match"])) for record in records),
            6,
        ),
        "actual_vs_wrong_role_advantage_mean": round(mean(wrong_role_diffs), 6),
        "actual_vs_non_role_advantage_mean": round(mean(non_role_diffs), 6),
        "actual_vs_shuffled_advantage_mean": round(mean(shuffled_diffs), 6),
        "actual_vs_strongest_control_advantage_mean": round(mean(strongest_diffs), 6),
        "bootstrap_ci": bootstrap_mean_ci(strongest_diffs, seed=seed),
        "actual_vs_strongest_control_permutation_p_value": paired_sign_permutation_p_value(strongest_diffs, seed=seed),
        "actual_vs_wrong_role_permutation_p_value": paired_sign_permutation_p_value(wrong_role_diffs, seed=stable_seed(seed, "wrong_role")),
        "actual_vs_non_role_permutation_p_value": paired_sign_permutation_p_value(non_role_diffs, seed=stable_seed(seed, "non_role")),
        "actual_vs_shuffled_permutation_p_value": paired_sign_permutation_p_value(shuffled_diffs, seed=stable_seed(seed, "shuffled")),
    }


def summarize_necessity_module(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "role_ablation_clause_f1_drop": None,
            "wrong_role_clause_f1_drop": None,
            "non_role_clause_f1_drop": None,
            "strongest_control_clause_f1_drop": None,
            "role_vs_wrong_role_gap_clause_f1": None,
            "role_vs_non_role_gap_clause_f1": None,
            "role_vs_strongest_control_gap_clause_f1": None,
            "bootstrap_ci": None,
            "role_vs_strongest_control_permutation_p_value": None,
            "role_vs_wrong_role_permutation_p_value": None,
            "role_vs_non_role_permutation_p_value": None,
        }
    strongest_diffs = [float(record["role_vs_strongest_control_gap_clause_f1"]) for record in records]
    wrong_role_diffs = [float(record["role_vs_wrong_role_gap_clause_f1"]) for record in records]
    non_role_diffs = [float(record["role_vs_non_role_gap_clause_f1"]) for record in records]
    return {
        "count": len(records),
        "operationalization_counts": dict(
            Counter(str(record.get("operationalization") or "span") for record in records)
        ),
        "role_ablation_clause_f1_drop": round(mean(float(record["role_ablation_clause_f1_drop"]) for record in records), 6),
        "wrong_role_clause_f1_drop": round(mean(float(record["wrong_role_clause_f1_drop"]) for record in records), 6),
        "non_role_clause_f1_drop": round(mean(float(record["non_role_clause_f1_drop"]) for record in records), 6),
        "strongest_control_clause_f1_drop": round(
            mean(float(record["strongest_control_clause_f1_drop"]) for record in records),
            6,
        ),
        "role_vs_wrong_role_gap_clause_f1": round(mean(wrong_role_diffs), 6),
        "role_vs_non_role_gap_clause_f1": round(mean(non_role_diffs), 6),
        "role_vs_strongest_control_gap_clause_f1": round(mean(strongest_diffs), 6),
        "bootstrap_ci": bootstrap_mean_ci(strongest_diffs, seed=seed),
        "role_vs_strongest_control_permutation_p_value": paired_sign_permutation_p_value(strongest_diffs, seed=seed),
        "role_vs_wrong_role_permutation_p_value": paired_sign_permutation_p_value(wrong_role_diffs, seed=stable_seed(seed, "wrong_role")),
        "role_vs_non_role_permutation_p_value": paired_sign_permutation_p_value(non_role_diffs, seed=stable_seed(seed, "non_role")),
    }


def summarize_rescue_module(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "actual_mean": None,
            "wrong_role_mean": None,
            "non_role_mean": None,
            "shuffled_mean": None,
            "strongest_control_mean": None,
            "actual_binary_rate": None,
            "strongest_control_binary_rate": None,
            "actual_vs_wrong_role_advantage_mean": None,
            "actual_vs_non_role_advantage_mean": None,
            "actual_vs_shuffled_advantage_mean": None,
            "actual_vs_strongest_control_advantage_mean": None,
            "bootstrap_ci": None,
            "actual_vs_strongest_control_permutation_p_value": None,
            "actual_vs_wrong_role_permutation_p_value": None,
            "actual_vs_non_role_permutation_p_value": None,
            "actual_vs_shuffled_permutation_p_value": None,
        }
    strongest_diffs = [float(record["actual_vs_strongest_control_advantage_clause_f1"]) for record in records]
    alignment_only_diffs = [
        float(record["actual_vs_alignment_only_advantage_clause_f1"])
        for record in records
        if record.get("actual_vs_alignment_only_advantage_clause_f1") is not None
    ]
    wrong_role_diffs = [float(record["actual_vs_wrong_role_advantage_clause_f1"]) for record in records]
    non_role_diffs = [float(record["actual_vs_non_role_advantage_clause_f1"]) for record in records]
    shuffled_diffs = [float(record["actual_vs_shuffled_advantage_clause_f1"]) for record in records]
    shuffled_alignment_diffs = [
        float(record["actual_vs_shuffled_alignment_advantage_clause_f1"])
        for record in records
        if record.get("actual_vs_shuffled_alignment_advantage_clause_f1") is not None
    ]
    wrong_path_diffs = [
        float(record["actual_vs_wrong_path_advantage_clause_f1"])
        for record in records
        if record.get("actual_vs_wrong_path_advantage_clause_f1") is not None
    ]
    generic_decoder_gain_diffs = [
        float(record["actual_vs_generic_decoder_gain_advantage_clause_f1"])
        for record in records
        if record.get("actual_vs_generic_decoder_gain_advantage_clause_f1") is not None
    ]
    alignment_only_values = [
        float(record["alignment_only_clause_f1_gain"])
        for record in records
        if record.get("alignment_only_clause_f1_gain") is not None
    ]
    path_only_values = [
        float(record["path_only_clause_f1_gain"])
        for record in records
        if record.get("path_only_clause_f1_gain") is not None
    ]
    wrong_path_values = [
        float(record["wrong_path_clause_f1_gain"])
        for record in records
        if record.get("wrong_path_clause_f1_gain") is not None
    ]
    generic_decoder_gain_values = [
        float(record["generic_decoder_gain_clause_f1_gain"])
        for record in records
        if record.get("generic_decoder_gain_clause_f1_gain") is not None
    ]
    shuffled_alignment_values = [
        float(record["shuffled_alignment_clause_f1_gain"])
        for record in records
        if record.get("shuffled_alignment_clause_f1_gain") is not None
    ]
    path_only_diffs = [
        float(record["actual_vs_path_only_advantage_clause_f1"])
        for record in records
        if record.get("actual_vs_path_only_advantage_clause_f1") is not None
    ]
    return {
        "count": len(records),
        "operationalization_counts": dict(
            Counter(str(record.get("operationalization") or "span") for record in records)
        ),
        "actual_mean": round(mean(float(record["actual_clause_f1_gain"]) for record in records), 6),
        "alignment_only_mean": round(mean(alignment_only_values), 6) if alignment_only_values else None,
        "wrong_role_mean": round(mean(float(record["wrong_role_clause_f1_gain"]) for record in records), 6),
        "non_role_mean": round(mean(float(record["non_role_clause_f1_gain"]) for record in records), 6),
        "shuffled_mean": round(mean(float(record["shuffled_clause_f1_gain"]) for record in records), 6),
        "shuffled_alignment_mean": round(mean(shuffled_alignment_values), 6) if shuffled_alignment_values else None,
        "wrong_path_mean": round(mean(wrong_path_values), 6) if wrong_path_values else None,
        "generic_decoder_gain_mean": round(mean(generic_decoder_gain_values), 6) if generic_decoder_gain_values else None,
        "path_only_mean": round(mean(path_only_values), 6) if path_only_values else None,
        "strongest_control_mean": round(mean(float(record["strongest_control_clause_f1_gain"]) for record in records), 6),
        "actual_binary_rate": round(mean(float(bool(record["actual_rescued_from_incorrect"])) for record in records), 6),
        "strongest_control_binary_rate": round(
            mean(float(bool(record["strongest_control_rescued_from_incorrect"])) for record in records),
            6,
        ),
        "actual_vs_alignment_only_advantage_mean": round(mean(alignment_only_diffs), 6) if alignment_only_diffs else None,
        "actual_vs_wrong_role_advantage_mean": round(mean(wrong_role_diffs), 6),
        "actual_vs_non_role_advantage_mean": round(mean(non_role_diffs), 6),
        "actual_vs_shuffled_advantage_mean": round(mean(shuffled_diffs), 6),
        "actual_vs_shuffled_alignment_advantage_mean": round(mean(shuffled_alignment_diffs), 6)
        if shuffled_alignment_diffs
        else None,
        "actual_vs_wrong_path_advantage_mean": round(mean(wrong_path_diffs), 6) if wrong_path_diffs else None,
        "actual_vs_generic_decoder_gain_advantage_mean": round(mean(generic_decoder_gain_diffs), 6)
        if generic_decoder_gain_diffs
        else None,
        "actual_vs_path_only_advantage_mean": round(mean(path_only_diffs), 6) if path_only_diffs else None,
        "actual_vs_strongest_control_advantage_mean": round(mean(strongest_diffs), 6),
        "bootstrap_ci": bootstrap_mean_ci(strongest_diffs, seed=seed),
        "actual_vs_strongest_control_permutation_p_value": paired_sign_permutation_p_value(strongest_diffs, seed=seed),
        "actual_vs_wrong_role_permutation_p_value": paired_sign_permutation_p_value(wrong_role_diffs, seed=stable_seed(seed, "wrong_role")),
        "actual_vs_non_role_permutation_p_value": paired_sign_permutation_p_value(non_role_diffs, seed=stable_seed(seed, "non_role")),
        "actual_vs_shuffled_permutation_p_value": paired_sign_permutation_p_value(shuffled_diffs, seed=stable_seed(seed, "shuffled")),
    }


def ensure_generation_record(
    adapter: Any,
    item: dict[str, Any],
    generation_cache: dict[str, dict[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
    item_id = str(item["item_id"])
    if item_id in generation_cache:
        return generation_cache[item_id]
    prediction = adapter.generate_text(str(item["input_text"]), max_new_tokens=max_new_tokens)
    record = {
        "prediction": prediction,
        "metrics": score_prediction(prediction, str(item["target_form"])),
    }
    generation_cache[item_id] = record
    return record


def build_exact_counterpart_index(
    reference_records: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in reference_records:
        grouped[counterpart_group_key(record)].append(record)
    return grouped


def counterpart_group_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    roles = dict(record.get("role_signature") or {})
    return (
        str(record.get("benchmark_name") or ""),
        str(record.get("generalization_case") or ""),
        abstract_roles_in_text(str(record["input_text"]), roles),
        abstract_roles_in_text(str(record["target_form"]), roles),
    )


def find_exact_role_counterpart(
    record: dict[str, Any],
    role: str,
    counterpart_index: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    preferred_strata: tuple[str, ...] = ("pass", "control"),
    same_split_only: bool = False,
) -> dict[str, Any] | None:
    roles = dict(record.get("role_signature") or {})
    target_value = str(roles.get(role) or "")
    if not target_value:
        return None

    candidates = counterpart_index.get(counterpart_group_key(record), [])
    filtered = []
    for candidate in candidates:
        if str(candidate["item_id"]) == str(record["item_id"]):
            continue
        candidate_roles = dict(candidate.get("role_signature") or {})
        candidate_value = str(candidate_roles.get(role) or "")
        if not candidate_value or candidate_value.lower() == target_value.lower():
            continue
        if same_split_only and str(candidate.get("split")) != str(record.get("split")):
            continue
        if any(
            str(candidate_roles.get(other_role) or "").lower() != str(roles.get(other_role) or "").lower()
            for other_role in ALL_ROLE_KEYS
            if other_role != role and roles.get(other_role)
        ):
            continue
        filtered.append(candidate)

    if not filtered:
        return None

    preferred_order = {label: index for index, label in enumerate(preferred_strata)}
    filtered.sort(
        key=lambda candidate: (
            preferred_order.get(str(candidate.get("stratum")), len(preferred_order)),
            0 if str(candidate.get("split")) == str(record.get("split")) else 1,
            str(candidate.get("item_id")),
        )
    )
    return filtered[0]


def abstract_roles_in_text(text: str, roles: dict[str, Any]) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    for role in ALL_ROLE_KEYS:
        filler = str(roles.get(role) or "")
        if not filler:
            continue
        normalized = re.sub(
            rf"(?<!\w){re.escape(filler)}(?!\w)",
            "{" + role.upper() + "}",
            normalized,
            flags=re.IGNORECASE,
        )
    return normalized


def ensure_non_role_span_record(
    adapter: Any,
    item: dict[str, Any],
    item_span_map: dict[str, Any],
    desired_token_count: int,
    layer_index: int,
    max_length: int,
    control_span_cache: dict[tuple[str, int, int], dict[str, Any] | None],
    control_window_cache: dict[tuple[str, int, int], dict[str, Any] | None],
    non_role_policy: str = "contentful_first",
    seed: int = 0,
) -> dict[str, Any] | None:
    cache_key = (str(item["item_id"]), layer_index, desired_token_count)
    if cache_key in control_span_cache:
        return control_span_cache[cache_key]

    window = ensure_non_role_window(
        adapter=adapter,
        item=item,
        item_span_map=item_span_map,
        desired_token_count=desired_token_count,
        max_length=max_length,
        control_window_cache=control_window_cache,
        non_role_policy=non_role_policy,
        seed=seed,
    )
    if window is None:
        control_span_cache[cache_key] = None
        return None

    probe = {
        "probe_id": f"{item['item_id']}::control::non_role::{desired_token_count}",
        "namespace": "control_span",
        "label": "non_role",
        "surface_text": window["surface_text"],
        "start_char": window["start_char"],
        "end_char": window["end_char"],
    }
    traced = adapter.trace_text(
        input_text=str(item["input_text"]),
        probes=[probe],
        layer_indices=[layer_index],
        max_length=max_length,
    )
    record = traced["span_vectors"][0] if traced["span_vectors"] else None
    control_span_cache[cache_key] = record
    return record


def ensure_non_role_window(
    adapter: Any,
    item: dict[str, Any],
    item_span_map: dict[str, Any],
    desired_token_count: int,
    max_length: int,
    control_window_cache: dict[tuple[str, int, int], dict[str, Any] | None],
    non_role_policy: str = "contentful_first",
    seed: int = 0,
) -> dict[str, Any] | None:
    cache_key = (str(item["item_id"]), max_length, desired_token_count)
    if cache_key in control_window_cache:
        return control_window_cache[cache_key]

    if desired_token_count <= 0:
        control_window_cache[cache_key] = None
        return None

    token_metadata = adapter.prompt_token_metadata(str(item["input_text"]), max_length=max_length)
    prompt_text = str(token_metadata["prompt_text"])
    input_text = str(item["input_text"])
    input_offset = max(0, prompt_text.find(input_text))
    offsets = [list(offset) for offset in token_metadata["offsets"]]
    tokens = list(token_metadata["tokens"])
    occupied = {
        int(token_index)
        for span_record in item_span_map.values()
        for token_index in list(span_record.get("token_indices") or [])
    }
    candidates: list[tuple[int, int, dict[str, Any]]] = []

    for start_index in range(0, max(0, len(offsets) - desired_token_count + 1)):
        token_indices = list(range(start_index, start_index + desired_token_count))
        if any(token_index in occupied for token_index in token_indices):
            continue
        window_offsets = [offsets[token_index] for token_index in token_indices]
        if any(token_end <= token_start for token_start, token_end in window_offsets):
            continue
        start_char = int(window_offsets[0][0])
        end_char = int(window_offsets[-1][1])
        if start_char < input_offset or end_char > input_offset + len(input_text):
            continue
        surface_text = prompt_text[start_char:end_char]
        if not re.search(r"[A-Za-z]", surface_text):
            continue
        control_tokens = tokens[start_index : start_index + desired_token_count]
        contentful = any(
            normalize_control_token(token_text) not in CONTROL_STOPWORDS
            for token_text in control_tokens
            if re.search(r"[A-Za-z]", token_text)
        )
        candidates.append(
            (
                0 if contentful else 1,
                start_index,
                {
                    "token_indices": token_indices,
                    "surface_text": surface_text.strip(),
                    "start_char": start_char - input_offset,
                    "end_char": end_char - input_offset,
                },
            )
        )

    if non_role_policy == "random_window":
        rng = random.Random(seed)
        selected = rng.choice(candidates)[2] if candidates else None
    elif non_role_policy == "nearest_window":
        occupied_centers = sorted(occupied)
        candidates.sort(
            key=lambda item: (
                min(abs(((item[2]["token_indices"][0] + item[2]["token_indices"][-1]) / 2) - center) for center in occupied_centers)
                if occupied_centers
                else 0,
                item[0],
                item[1],
            )
        )
        selected = candidates[0][2] if candidates else None
    else:
        candidates.sort(key=lambda item: (item[0], item[1]))
        selected = candidates[0][2] if candidates else None
    control_window_cache[cache_key] = selected
    return selected


def normalize_control_token(token_text: str) -> str:
    return re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", token_text).lower()


def opposite_role(role: str) -> str:
    return "theme" if role == "agent" else "agent"


def shuffle_vector(vector: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    shuffled = list(vector)
    rng.shuffle(shuffled)
    return shuffled


def stable_seed(*parts: object) -> int:
    payload = "||".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def build_realism_bank(span_records: list[dict[str, Any]], layer_index: int) -> dict[str, list[list[float]]]:
    bank: dict[str, list[list[float]]] = defaultdict(list)
    for record in span_records:
        if int(record["layer_index"]) != layer_index:
            continue
        probe_id = str(record.get("probe_id") or "")
        _, _, suffix = probe_id.partition("::")
        if not suffix:
            continue
        bank[suffix].append(list(record["vector"]))
    return dict(bank)


def build_patch_realism_snapshot(
    target_site_key: str,
    target_vector: list[float],
    actual_vector: list[float],
    wrong_role_vector: list[float],
    non_role_vector: list[float],
    shuffled_vector: list[float],
    realism_bank: dict[str, list[list[float]]],
) -> dict[str, float | None]:
    target_bank = list(realism_bank.get(target_site_key) or [])
    payload: dict[str, float | None] = {}
    for prefix, vector in (
        ("actual", actual_vector),
        ("wrong_role", wrong_role_vector),
        ("non_role", non_role_vector),
        ("shuffled", shuffled_vector),
    ):
        payload[f"{prefix}_replacement_norm"] = round(vector_l2_norm(vector), 6)
        payload[f"{prefix}_cosine_to_target"] = round(cosine_similarity(vector, target_vector), 6)
        payload[f"{prefix}_min_cosine_distance_to_target_bank"] = min_cosine_distance(vector, target_bank)
    return payload


def vector_l2_norm(vector: list[float]) -> float:
    return sum(float(value) * float(value) for value in vector) ** 0.5


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = vector_l2_norm(left)
    right_norm = vector_l2_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(float(lv) * float(rv) for lv, rv in zip(left, right))
    return dot / (left_norm * right_norm)


def min_cosine_distance(vector: list[float], bank: list[list[float]]) -> float | None:
    if not bank:
        return None
    return round(min(1.0 - cosine_similarity(vector, candidate) for candidate in bank), 6)


def aggregate_control_value(
    values: dict[str, float | int],
    method: str = "max",
    priority: tuple[str, ...] = ("wrong_role", "non_role", "shuffled"),
) -> float:
    numeric_values = {key: float(value) for key, value in values.items()}
    if not numeric_values:
        return 0.0
    if method == "mean":
        return round(mean(numeric_values.values()), 6)
    if method == "priority":
        for key in priority:
            if key in numeric_values:
                return round(float(numeric_values[key]), 6)
        return round(float(next(iter(numeric_values.values()))), 6)
    return round(max(numeric_values.values()), 6)


def bootstrap_mean_ci(values: list[float], n_boot: int = 500, seed: int = 0) -> dict[str, float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        boot_means.append(mean(sample))
    boot_means.sort()
    lower_index = max(0, int(0.025 * len(boot_means)) - 1)
    upper_index = min(len(boot_means) - 1, int(0.975 * len(boot_means)) - 1)
    return {
        "low": round(float(boot_means[lower_index]), 6),
        "high": round(float(boot_means[upper_index]), 6),
    }


def paired_sign_permutation_p_value(values: list[float], n_perm: int = 1000, seed: int = 0) -> float | None:
    if not values:
        return None
    observed = abs(mean(values))
    rng = random.Random(seed)
    exceed = 0
    for _ in range(n_perm):
        permuted = [value if rng.random() < 0.5 else -value for value in values]
        if abs(mean(permuted)) >= observed:
            exceed += 1
    return round((exceed + 1) / (n_perm + 1), 6)
