from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from interventions.generation import (
    GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS,
    GEOMETRY_INTERACTION_OBJECT_CLASS,
    GEOMETRY_PLANT_KIND_H1_POSITIVE,
    GEOMETRY_PLANT_KIND_H2_POSITIVE,
    GEOMETRY_PLANT_KIND_NULL_DECOY,
    GEOMETRY_PLANT_KIND_OFF,
    GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS,
    aggregate_control_value,
    infer_geometry_plant_kind,
    normalize_geometry_plant_spec,
    stable_seed,
)

from . import interaction_objects as p17
from .family_closure import (
    ALIGNER_BASELINE,
    ALIGNER_BRIDGE,
    ALIGNER_NAMES,
    ALIGNER_SLOT,
    BRIDGE_OBJECT_CLASS,
    QUERY_SOURCE_OBJECT_CLASS,
    _align_with_selected,
    _margin_mean_for_positions,
    _object_view,
    case_condition_direct_h1,
    load_json,
)


H2_ITEM_RECOVERY_MIN_RECALL = 0.8
H1_RUN_RECOVERY_MIN_RECALL = 0.67
NULL_DIRECT_H1_MAX_FALSE_POSITIVE_RATE = 0.2

GENERIC_GAIN_OVERLAP_MAX = 0.2
H2_NECESSITY_MIN_GAP = p17.H2_NECESSITY_MIN_GAP
H1_RESCUE_MIN_GAP = p17.H1_RESCUE_MIN_GAP
H1_GENERIC_ADVANTAGE_MIN = p17.H1_GENERIC_ADVANTAGE_MIN
H1_RESCUE_MIN_BINARY_RATE = p17.H1_RESCUE_MIN_BINARY_RATE

CLASS_ORDER: tuple[str, ...] = (
    GEOMETRY_INTERACTION_OBJECT_CLASS,
    BRIDGE_OBJECT_CLASS,
    QUERY_SOURCE_OBJECT_CLASS,
)

STEP_KEYS: tuple[str, ...] = ("predicate_value", "agent_value", "theme_value")


def _hidden_size_for_adapter(adapter: Any) -> int:
    config = getattr(adapter.model, "config", None)
    for name in ("d_model", "hidden_size", "n_embd"):
        value = getattr(config, name, None)
        if value:
            return int(value)
    lm_head = getattr(adapter.model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if getattr(weight, "shape", None):
        return int(weight.shape[-1])
    return 512


def _num_heads_for_adapter(adapter: Any) -> int:
    config = getattr(adapter.model, "config", None)
    for name in ("num_heads", "num_attention_heads"):
        value = getattr(config, name, None)
        if value:
            return int(value)
    return 8


def _intended_step_key(object_class: str, object_spec: dict[str, Any]) -> str:
    if object_class == GEOMETRY_INTERACTION_OBJECT_CLASS:
        return str(object_spec["decoder_step_key"])
    if object_class == BRIDGE_OBJECT_CLASS:
        return str(object_spec["target_step_key"])
    return str(object_spec["query_step_key"])


def _supports_store(object_spec: dict[str, Any]) -> bool:
    return str(dict(object_spec.get("store_condition") or {}).get("mode") or "none") != "none"


def _class_label(object_class: str) -> str:
    return {
        GEOMETRY_INTERACTION_OBJECT_CLASS: "C1",
        BRIDGE_OBJECT_CLASS: "C2",
        QUERY_SOURCE_OBJECT_CLASS: "C3",
    }[object_class]


def protocol_kwargs_from_plant_spec(
    *,
    geometry_plant_kind: str,
    geometry_plant_spec: dict[str, Any] | str | None,
) -> dict[str, Any]:
    normalized = normalize_geometry_plant_spec(
        geometry_plant_kind,
        geometry_plant_spec if isinstance(geometry_plant_spec, str) else json.dumps(geometry_plant_spec or {}),
    )
    return {
        "geometry_plant_kind": infer_geometry_plant_kind(geometry_plant_kind),
        "geometry_plant_spec": normalized,
    }


def _top1_anchor(catalog: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        dict(row)
        for row in catalog
        if bool(row.get("non_flat_survivor"))
        and float(row.get("generic_gain_overlap") or 0.0) <= GENERIC_GAIN_OVERLAP_MAX
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: float(row.get("dev_h2_contrast") or 0.0) - float(row.get("dev_h0_contrast") or 0.0),
        reverse=True,
    )
    return eligible[0]


def _family_spec(
    *,
    object_class: str,
    object_spec: dict[str, Any],
    plant_kind: str,
    base_run: str,
    physical_family: str,
    head_scale: float,
    basis_scale: float,
    store_mode: str,
    source_mode: str,
    intervention_scope: str,
    default_attenuation: float,
) -> dict[str, Any]:
    plant_spec = normalize_geometry_plant_spec(
        plant_kind,
        json.dumps(
            {
                "plant_id": f"{object_spec['object_id']}::{plant_kind}::{physical_family}",
                "object_class": object_class,
                "object_spec": dict(object_spec),
                "base_run": base_run,
                "head_scale": head_scale,
                "basis_scale": basis_scale,
                "store_mode": store_mode,
                "source_mode": source_mode,
                "intervention_scope": intervention_scope,
                "default_attenuation": default_attenuation,
            }
        ),
    )
    return {
        "plant_kind": plant_kind,
        "physical_family": physical_family,
        "base_run": base_run,
        "geometry_plant_spec": plant_spec,
    }


def build_planted_family_manifest(
    *,
    class_catalogs: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    classes: list[dict[str, Any]] = []
    h2_bases = list(config.get("h2_base_runs") or [])
    h1_probes = list(config.get("h1_probe_runs") or [])
    for object_class in CLASS_ORDER:
        catalog = [dict(row) for row in class_catalogs.get(object_class) or []]
        anchor = _top1_anchor(catalog)
        if anchor is None:
            classes.append(
                {
                    "object_class": object_class,
                    "class_label": _class_label(object_class),
                    "anchor_found": False,
                    "anchor_failure_reason": "no_non_generic_survivor",
                    "h2_positive_families": [],
                    "h1_positive_families": [],
                    "null_decoy_families": [],
                }
            )
            continue
        supports_store = _supports_store(anchor)
        h2_positive = [
            _family_spec(
                object_class=object_class,
                object_spec=anchor,
                plant_kind=GEOMETRY_PLANT_KIND_H2_POSITIVE,
                base_run=str(base["run_id"]),
                physical_family=str(base["run_id"]),
                head_scale=1.75,
                basis_scale=1.5,
                store_mode="matched" if supports_store else "none",
                source_mode="correct",
                intervention_scope="full_object_scope",
                default_attenuation=0.0,
            )
            for base in h2_bases
        ]
        h1_positive = [
            _family_spec(
                object_class=object_class,
                object_spec=anchor,
                plant_kind=GEOMETRY_PLANT_KIND_H1_POSITIVE,
                base_run=str(probe["run_id"]),
                physical_family=str(probe["physical_family"]),
                head_scale=1.25,
                basis_scale=1.1,
                store_mode="rescue_only" if supports_store else "none",
                source_mode="correct",
                intervention_scope="metric_scope_only",
                default_attenuation=0.5,
            )
            for probe in h1_probes
        ]
        null_decoy = []
        for family in h2_positive + h1_positive:
            recipe = dict(family["geometry_plant_spec"])
            null_decoy.append(
                _family_spec(
                    object_class=object_class,
                    object_spec=anchor,
                    plant_kind=GEOMETRY_PLANT_KIND_NULL_DECOY,
                    base_run=str(family["base_run"]),
                    physical_family=str(family["physical_family"]),
                    head_scale=float(recipe["head_scale"]),
                    basis_scale=float(recipe["basis_scale"]),
                    store_mode="wrong_or_none",
                    source_mode="wrong_or_shuffled",
                    intervention_scope=str(recipe["intervention_scope"]),
                    default_attenuation=float(recipe["default_attenuation"]),
                )
            )
        classes.append(
            {
                "object_class": object_class,
                "class_label": _class_label(object_class),
                "anchor_found": True,
                "anchor_object_id": str(anchor["object_id"]),
                "anchor_object_spec": anchor,
                "h2_positive_families": h2_positive,
                "h1_positive_families": h1_positive,
                "null_decoy_families": null_decoy,
            }
        )
    return {
        "class_count": len(classes),
        "classes": classes,
    }


def summarize_planted_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = list(manifest.get("classes") or [])
    anchor_found = [row for row in rows if bool(row.get("anchor_found"))]
    return {
        "class_count": len(rows),
        "anchor_found_count": len(anchor_found),
        "anchor_missing_classes": [str(row["object_class"]) for row in rows if not bool(row.get("anchor_found"))],
        "anchors": {
            str(row["object_class"]): str(row.get("anchor_object_id") or "")
            for row in anchor_found
        },
        "class_rows": rows,
    }


def _bundle_family_kind(entry: dict[str, Any]) -> str:
    kind = str(entry["plant_kind"])
    if kind == GEOMETRY_PLANT_KIND_NULL_DECOY:
        return "null_decoy"
    if kind == GEOMETRY_PLANT_KIND_H1_POSITIVE:
        return "h1_positive"
    return "h2_positive"


def _head_indices_for_mode(adapter: Any, view: dict[str, Any], *, source_mode: str, object_id: str, salt: str) -> list[int]:
    selected = [int(value) for value in list(view["head_indices"])]
    if source_mode != "wrong_or_shuffled":
        return selected
    return p17._shuffled_indices(
        _num_heads_for_adapter(adapter),
        selected,
        len(selected),
        stable_seed(object_id, salt, "heads"),
    )


def _basis_indices_for_mode(adapter: Any, view: dict[str, Any], *, source_mode: str, generic: bool, object_id: str, salt: str) -> list[int]:
    if generic and list(view.get("generic_basis_indices") or []):
        return [int(value) for value in list(view["generic_basis_indices"])]
    selected = [int(value) for value in list(view["basis_indices"])]
    if source_mode != "wrong_or_shuffled":
        return selected
    return p17._shuffled_indices(
        _hidden_size_for_adapter(adapter),
        selected,
        len(selected),
        stable_seed(object_id, salt, "basis"),
    )


def _source_indices_for_mode(view: dict[str, Any], *, source_mode: str) -> list[int]:
    if source_mode == "wrong_or_shuffled":
        return [int(value) for value in list(view.get("wrong_source_token_indices") or [])]
    return [int(value) for value in list(view.get("source_token_indices") or [])]


def _step_positions_for_scope(view: dict[str, Any], scope: str, *, wrong_event: bool = False) -> list[int]:
    if scope == "metric_scope_only":
        return [int(value) for value in list(view["metric_positions"])]
    if wrong_event and list(view.get("wrong_intervention_positions") or []):
        return [int(value) for value in list(view["wrong_intervention_positions"])]
    return [int(value) for value in list(view["intervention_positions"])]


def _positive_scales(plant_spec: dict[str, Any]) -> tuple[float, float, float]:
    head_scale = float(plant_spec["head_scale"])
    basis_scale = float(plant_spec["basis_scale"])
    attenuation = float(plant_spec["default_attenuation"] or 0.0)
    if str(plant_spec["plant_kind"]) == GEOMETRY_PLANT_KIND_H1_POSITIVE:
        return head_scale * max(0.0, 1.0 - attenuation), basis_scale * max(0.0, 1.0 - attenuation), max(0.0, 1.0 - attenuation)
    return head_scale, basis_scale, 1.0


def _teacher_forced_with_mode(
    *,
    adapter: Any,
    input_text: str,
    target_text: str,
    layer_index: int,
    max_length: int,
    view: dict[str, Any],
    object_spec: dict[str, Any],
    plant_spec: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    object_id = str(object_spec["object_id"])
    source_mode = str(plant_spec["source_mode"])
    scope = str(plant_spec["intervention_scope"])
    head_scale, basis_scale, encoder_scale = _positive_scales(plant_spec)
    head_indices = _head_indices_for_mode(adapter, view, source_mode=source_mode, object_id=object_id, salt=mode)
    basis_indices = _basis_indices_for_mode(
        adapter,
        view,
        source_mode=source_mode,
        generic=(mode == "generic_gain"),
        object_id=object_id,
        salt=mode,
    )
    step_indices = _step_positions_for_scope(view, scope, wrong_event=mode == "wrong_event")
    encoder_token_indices = _source_indices_for_mode(view, source_mode=source_mode)
    store_mode = str(plant_spec["store_mode"])
    head_mask = None
    basis_mode = None
    basis_edit_indices: list[int] = []
    basis_edit_scale = basis_scale
    encoder_indices: list[int] | None = None
    encoder_edit_scale: float | None = None

    if mode == "baseline":
        head_mask = p17._build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=head_indices, selected_scale=head_scale)
        basis_mode = "boost"
        basis_edit_indices = basis_indices
        if store_mode in {"matched", "rescue_only"} and encoder_token_indices:
            encoder_indices = encoder_token_indices
            encoder_edit_scale = encoder_scale
    elif mode == "remove":
        head_mask = p17._build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=[int(v) for v in list(view["head_indices"])], selected_scale=0.0)
        basis_mode = "remove"
        basis_edit_indices = [int(value) for value in list(view["basis_indices"])]
        if store_mode in {"matched", "rescue_only"} and list(view.get("source_token_indices") or []):
            encoder_indices = [int(value) for value in list(view["source_token_indices"])]
            encoder_edit_scale = 0.0
    elif mode == "wrong_event":
        head_mask = p17._build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=[int(v) for v in list(view["head_indices"])], selected_scale=0.0)
        basis_mode = "remove"
        basis_edit_indices = [int(value) for value in list(view["basis_indices"])]
        if store_mode in {"matched", "rescue_only"} and list(view.get("wrong_source_token_indices") or []):
            encoder_indices = [int(value) for value in list(view["wrong_source_token_indices"])]
            encoder_edit_scale = 0.0
    elif mode == "shuffled_head":
        shuffled = _head_indices_for_mode(adapter, view, source_mode="wrong_or_shuffled", object_id=object_id, salt="shuffled_head")
        head_mask = p17._build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=shuffled, selected_scale=0.0)
        basis_mode = "remove"
        basis_edit_indices = [int(value) for value in list(view["basis_indices"])]
    elif mode == "shuffled_basis":
        head_mask = p17._build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=[int(v) for v in list(view["head_indices"])], selected_scale=0.0)
        basis_mode = "remove"
        basis_edit_indices = _basis_indices_for_mode(
            adapter,
            view,
            source_mode="wrong_or_shuffled",
            generic=False,
            object_id=object_id,
            salt="shuffled_basis",
        )
    elif mode == "generic_gain":
        basis_mode = "boost"
        basis_edit_indices = _basis_indices_for_mode(
            adapter,
            view,
            source_mode=source_mode,
            generic=True,
            object_id=object_id,
            salt="generic_gain",
        )
        basis_edit_scale = basis_scale
        step_indices = [int(value) for value in list(view["metric_positions"])]
    elif mode == "reinstate_only":
        head_mask = p17._build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=[int(v) for v in list(view["head_indices"])], selected_scale=float(plant_spec["head_scale"]))
        basis_mode = "boost"
        basis_edit_indices = [int(value) for value in list(view["basis_indices"])]
        basis_edit_scale = float(plant_spec["basis_scale"])
    elif mode == "reinstate_plus_store":
        head_mask = p17._build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=[int(v) for v in list(view["head_indices"])], selected_scale=float(plant_spec["head_scale"]))
        basis_mode = "boost"
        basis_edit_indices = [int(value) for value in list(view["basis_indices"])]
        basis_edit_scale = float(plant_spec["basis_scale"])
        if _supports_store(object_spec) and list(view.get("source_token_indices") or []):
            encoder_indices = [int(value) for value in list(view["source_token_indices"])]
            encoder_edit_scale = 1.0
    elif mode == "wrong_object":
        wrong_heads = _head_indices_for_mode(adapter, view, source_mode="wrong_or_shuffled", object_id=object_id, salt="wrong_object")
        head_mask = p17._build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=wrong_heads, selected_scale=float(plant_spec["head_scale"]))
        basis_mode = "boost"
        basis_edit_indices = _basis_indices_for_mode(
            adapter,
            view,
            source_mode="wrong_or_shuffled",
            generic=False,
            object_id=object_id,
            salt="wrong_object",
        )
        basis_edit_scale = float(plant_spec["basis_scale"])
        if _supports_store(object_spec) and list(view.get("wrong_source_token_indices") or []):
            encoder_indices = [int(value) for value in list(view["wrong_source_token_indices"])]
            encoder_edit_scale = 1.0
    elif mode == "path_only":
        head_mask = p17._build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=[int(v) for v in list(view["head_indices"])], selected_scale=float(plant_spec["head_scale"]))
    else:
        raise ValueError(f"Unsupported plant mode: {mode}")

    return p17._teacher_forced_forward(
        adapter=adapter,
        input_text=input_text,
        target_text=target_text,
        layer_index=layer_index,
        max_length=max_length,
        head_mask=head_mask,
        encoder_token_indices=encoder_indices,
        encoder_scale=encoder_edit_scale,
        step_indices=step_indices if basis_mode is not None and basis_edit_indices else None,
        basis_indices=basis_edit_indices if basis_mode is not None else None,
        basis_mode=basis_mode,
        basis_scale=basis_edit_scale,
    )


def evaluate_planted_observability(
    *,
    bundle: dict[str, Any],
    object_class: str,
    object_spec: dict[str, Any],
    plant_spec: dict[str, Any],
    aligner_name: str,
    max_length: int = 256,
) -> dict[str, Any]:
    adapter, layer_index = p17.build_adapter_from_bundle(bundle)
    span_index = p17._index_span_records(bundle["span_records"], layer_index=layer_index)
    live_span_cache: dict[str, dict[str, Any]] = {}
    intended_step = _intended_step_key(object_class, object_spec)
    rows: list[dict[str, Any]] = []
    for item in bundle["candidate_records"]:
        item_span_map = p17._ensure_item_span_map(
            adapter=adapter,
            item=item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        forward = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
        )
        baseline_alignments = _align_with_selected(
            aligner_name=aligner_name,
            tokenizer=adapter.tokenizer,
            target_text=str(item["target_form"]),
            record=item,
            forward=forward,
            item_span_map=item_span_map,
            layer_index=layer_index,
        )
        view = _object_view(object_class=object_class, spec=object_spec, alignments=baseline_alignments, item_span_map=item_span_map)
        planted_forward = forward
        if aligner_name == ALIGNER_BRIDGE and view is not None:
            planted_forward = _teacher_forced_with_mode(
                adapter=adapter,
                input_text=str(item["input_text"]),
                target_text=str(item["target_form"]),
                layer_index=layer_index,
                max_length=max_length,
                view=view,
                object_spec=object_spec,
                plant_spec=plant_spec,
                mode="baseline",
            )
        alignments = _align_with_selected(
            aligner_name=aligner_name,
            tokenizer=adapter.tokenizer,
            target_text=str(item["target_form"]),
            record=item,
            forward=planted_forward,
            item_span_map=item_span_map,
            layer_index=layer_index,
        )
        aligned_steps = {str(key) for key in dict(alignments.get("alignments") or {}).keys()}
        rows.append(
            {
                "item_id": str(item["item_id"]),
                "intended_step_aligned": intended_step in aligned_steps,
                "unintended_step_aligned": any(step != intended_step for step in aligned_steps),
            }
        )
    intended_rate = round(mean(int(bool(row["intended_step_aligned"])) for row in rows), 6) if rows else 0.0
    unintended_rate = round(mean(int(bool(row["unintended_step_aligned"])) for row in rows), 6) if rows else 0.0
    return {
        "aligner": aligner_name,
        "intended_step_key": intended_step,
        "row_count": len(rows),
        "intended_step_alignment_rate": intended_rate,
        "unintended_step_alignment_rate": unintended_rate,
        "passed": intended_rate >= 0.8 and unintended_rate < 0.2,
    }


def select_planted_aligner(
    *,
    per_aligner_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    passed = [row for row in per_aligner_rows if bool(row.get("passed"))]
    if passed:
        ranked = sorted(
            passed,
            key=lambda row: (float(row["intended_step_alignment_rate"]) - float(row["unintended_step_alignment_rate"])),
            reverse=True,
        )
        selected = dict(ranked[0])
        return {
            "selected_aligner": str(selected["aligner"]),
            "passed": True,
            "selected_summary": selected,
        }
    ranked = sorted(
        per_aligner_rows,
        key=lambda row: (float(row["intended_step_alignment_rate"]) - float(row["unintended_step_alignment_rate"])),
        reverse=True,
    )
    return {
        "selected_aligner": str(ranked[0]["aligner"]) if ranked else ALIGNER_BASELINE,
        "passed": False,
        "selected_summary": dict(ranked[0]) if ranked else {},
    }


def evaluate_planted_h2_bundle(
    *,
    bundle: dict[str, Any],
    object_class: str,
    object_spec: dict[str, Any],
    plant_spec: dict[str, Any],
    aligner_name: str,
    max_length: int = 256,
) -> dict[str, Any]:
    adapter, layer_index = p17.build_adapter_from_bundle(bundle)
    span_index = p17._index_span_records(bundle["span_records"], layer_index=layer_index)
    live_span_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for item in bundle["candidate_records"]:
        item_span_map = p17._ensure_item_span_map(
            adapter=adapter,
            item=item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        prealign_forward = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
        )
        alignments = _align_with_selected(
            aligner_name=aligner_name,
            tokenizer=adapter.tokenizer,
            target_text=str(item["target_form"]),
            record=item,
            forward=prealign_forward,
            item_span_map=item_span_map,
            layer_index=layer_index,
        )
        view = _object_view(object_class=object_class, spec=object_spec, alignments=alignments, item_span_map=item_span_map)
        if view is None:
            continue
        baseline = _teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=plant_spec,
            mode="baseline",
        )
        metric_positions = [int(value) for value in list(view["metric_positions"])]
        baseline_margin = _margin_mean_for_positions(baseline, metric_positions, [])
        actual_remove = _teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=plant_spec,
            mode="remove",
        )
        wrong_event = _teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=plant_spec,
            mode="wrong_event",
        )
        shuffled_head = _teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=plant_spec,
            mode="shuffled_head",
        )
        shuffled_basis = _teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=plant_spec,
            mode="shuffled_basis",
        )
        generic_gain = _teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=plant_spec,
            mode="generic_gain",
        )
        actual_drop = baseline_margin - _margin_mean_for_positions(actual_remove, metric_positions, [])
        wrong_event_drop = baseline_margin - _margin_mean_for_positions(wrong_event, metric_positions, [])
        shuffled_head_drop = baseline_margin - _margin_mean_for_positions(shuffled_head, metric_positions, [])
        shuffled_basis_drop = baseline_margin - _margin_mean_for_positions(shuffled_basis, metric_positions, [])
        generic_drop = baseline_margin - _margin_mean_for_positions(generic_gain, metric_positions, [])
        strongest = aggregate_control_value(
            {
                "wrong_event_same_cardinality": wrong_event_drop,
                "shuffled_head_same_cardinality": shuffled_head_drop,
                "shuffled_basis_same_cardinality": shuffled_basis_drop,
                "generic_decoder_gain": generic_drop,
            },
            method="max",
            priority=(
                "wrong_event_same_cardinality",
                "shuffled_head_same_cardinality",
                "shuffled_basis_same_cardinality",
                "generic_decoder_gain",
            ),
        )
        gap = float(actual_drop) - float(strongest)
        rows.append(
            {
                "item_id": str(item["item_id"]),
                "necessity_gap": round(gap, 6),
                "passed_item": bool(gap >= H2_NECESSITY_MIN_GAP),
            }
        )
    recall = round(mean(int(bool(row["passed_item"])) for row in rows), 6) if rows else 0.0
    return {
        "base_run": str(plant_spec["base_run"]),
        "row_count": len(rows),
        "item_recovery_recall": recall,
        "passed": recall >= H2_ITEM_RECOVERY_MIN_RECALL,
        "rows": rows,
    }


def _counterpart_role_for_object(object_class: str, object_spec: dict[str, Any]) -> str:
    step_key = _intended_step_key(object_class, object_spec)
    return "theme" if step_key == "theme_value" else "agent"


def evaluate_planted_h1_run(
    *,
    bundle: dict[str, Any],
    object_class: str,
    object_spec: dict[str, Any],
    plant_spec: dict[str, Any],
    aligner_name: str,
    physical_family: str,
    max_length: int = 256,
) -> dict[str, Any]:
    adapter, layer_index = p17.build_adapter_from_bundle(bundle)
    span_index = p17._index_span_records(bundle["span_records"], layer_index=layer_index)
    live_span_cache: dict[str, dict[str, Any]] = {}
    counterpart_index = p17.build_exact_counterpart_index(bundle["candidate_records"])
    rows: list[dict[str, Any]] = []
    for item in bundle["candidate_records"]:
        if str(item.get("stratum")) != "fail":
            continue
        item_span_map = p17._ensure_item_span_map(
            adapter=adapter,
            item=item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        prealign_forward = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
        )
        alignments = _align_with_selected(
            aligner_name=aligner_name,
            tokenizer=adapter.tokenizer,
            target_text=str(item["target_form"]),
            record=item,
            forward=prealign_forward,
            item_span_map=item_span_map,
            layer_index=layer_index,
        )
        view = _object_view(object_class=object_class, spec=object_spec, alignments=alignments, item_span_map=item_span_map)
        if view is None:
            continue
        counterpart = p17.find_exact_role_counterpart(
            record=item,
            role=_counterpart_role_for_object(object_class, object_spec),
            counterpart_index=counterpart_index,
            preferred_strata=("pass", "control"),
            same_split_only=False,
        )
        if counterpart is None:
            continue
        baseline = _teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=plant_spec,
            mode="baseline",
        )
        metric_positions = [int(value) for value in list(view["metric_positions"])]
        baseline_summary = p17._token_margin_summary(
            logits=baseline["logits"],
            target_ids=baseline["target_ids"],
            step_positions=metric_positions,
            compare_ids=p17._counterpart_compare_ids(
                tokenizer=adapter.tokenizer,
                step_key=str(view["metric_step_key"]),
                target_positions=metric_positions,
                counterpart_record=counterpart,
            ),
        )
        if bool(baseline_summary["exact_match"]):
            continue
        actual_only = _teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=plant_spec,
            mode="reinstate_only",
        )
        actual_plus_store = _teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=plant_spec,
            mode="reinstate_plus_store",
        )
        wrong_object = _teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=plant_spec,
            mode="wrong_object",
        )
        path_only = _teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=plant_spec,
            mode="path_only",
        )
        generic_gain = _teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=plant_spec,
            mode="generic_gain",
        )

        def margin_gain(forward: dict[str, Any]) -> tuple[float, bool]:
            summary = p17._token_margin_summary(
                logits=forward["logits"],
                target_ids=forward["target_ids"],
                step_positions=metric_positions,
                compare_ids=p17._counterpart_compare_ids(
                    tokenizer=adapter.tokenizer,
                    step_key=str(view["metric_step_key"]),
                    target_positions=metric_positions,
                    counterpart_record=counterpart,
                ),
            )
            return float(summary["margin_mean"] or 0.0) - float(baseline_summary["margin_mean"] or 0.0), bool(summary["exact_match"])

        actual_only_gain, actual_only_exact = margin_gain(actual_only)
        actual_plus_store_gain, actual_plus_store_exact = margin_gain(actual_plus_store)
        wrong_gain, _ = margin_gain(wrong_object)
        path_gain, _ = margin_gain(path_only)
        generic_gain_value, _ = margin_gain(generic_gain)
        strongest_control = aggregate_control_value(
            {
                "wrong_object_reinstate": wrong_gain,
                "path_only": path_gain,
                "generic_decoder_gain": generic_gain_value,
            },
            method="max",
            priority=("wrong_object_reinstate", "path_only", "generic_decoder_gain"),
        )
        best_actual = max(actual_only_gain, actual_plus_store_gain)
        rows.append(
            {
                "item_id": str(item["item_id"]),
                "generalization_case": str(item["generalization_case"]),
                "physical_family": physical_family,
                "object_class": object_class,
                "best_object_actual_gain": round(best_actual, 6),
                "generic_decoder_gain": round(generic_gain_value, 6),
                "strongest_control_gain": round(float(strongest_control), 6),
                "rescue_gap": round(best_actual - float(strongest_control), 6),
                "object_exact_match": bool(actual_only_exact or actual_plus_store_exact),
            }
        )
    rescue_gap = round(mean(float(row["rescue_gap"]) for row in rows), 6) if rows else 0.0
    exact_rate = round(mean(int(bool(row["object_exact_match"])) for row in rows), 6) if rows else 0.0
    best_actual = round(mean(float(row["best_object_actual_gain"]) for row in rows), 6) if rows else 0.0
    generic_gain = round(mean(float(row["generic_decoder_gain"]) for row in rows), 6) if rows else 0.0
    return {
        "physical_family": physical_family,
        "row_count": len(rows),
        "run_recovered": bool(rows) and rescue_gap >= H1_RESCUE_MIN_GAP and exact_rate >= H1_RESCUE_MIN_BINARY_RATE,
        "mean_rescue_gap": rescue_gap,
        "aggregated_exact_match_rate": exact_rate,
        "generic_advantage": round(best_actual - generic_gain, 6),
        "rows": rows,
    }


def summarize_planted_h1_class(
    *,
    object_class: str,
    plant_kind: str,
    object_id: str,
    run_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    recovered_runs = [row for row in run_rows if bool(row.get("run_recovered"))]
    family_coverage = sorted({str(row["physical_family"]) for row in recovered_runs})
    recall = round(len(recovered_runs) / max(len(run_rows), 1), 6) if run_rows else 0.0
    generic_advantages = [float(row.get("generic_advantage") or 0.0) for row in run_rows]
    exact_rates = [float(row.get("aggregated_exact_match_rate") or 0.0) for row in run_rows]
    exact_rate = round(mean(exact_rates), 6) if exact_rates else 0.0
    passed = bool(run_rows) and (
        recall >= H1_RUN_RECOVERY_MIN_RECALL
        and len(family_coverage) >= 2
        and max(generic_advantages or [0.0]) >= H1_GENERIC_ADVANTAGE_MIN
        and sum(1 for value in generic_advantages if value >= 0.0) >= 2
        and exact_rate >= H1_RESCUE_MIN_BINARY_RATE
    )
    return {
        "object_class": object_class,
        "object_id": object_id,
        "plant_kind": plant_kind,
        "run_count": len(run_rows),
        "run_recovery_recall": recall,
        "physical_family_coverage": family_coverage,
        "aggregated_exact_match_rate": exact_rate,
        "generic_advantages": generic_advantages,
        "passed": bool(passed),
        "run_rows": run_rows,
    }


def summarize_detectability(
    *,
    manifest_summary: dict[str, Any],
    observability_summary: dict[str, Any],
    planted_h2_summary: dict[str, Any],
    planted_h1_summary: dict[str, Any],
    real_regression_summary: dict[str, Any],
) -> dict[str, Any]:
    class_rows = list(manifest_summary.get("class_rows") or [])
    observability_by_class = {str(row["object_class"]): row for row in list(observability_summary.get("class_rows") or [])}
    h2_by_class = {str(row["object_class"]): row for row in list(planted_h2_summary.get("class_rows") or [])}
    h1_by_class = {str(row["object_class"]): row for row in list(planted_h1_summary.get("class_rows") or [])}
    failures: list[str] = []
    for class_row in class_rows:
        object_class = str(class_row["object_class"])
        if not bool(class_row.get("anchor_found")):
            failures.append(f"{object_class}:anchor_missing")
            continue
        if not bool(observability_by_class.get(object_class, {}).get("passed")):
            failures.append(f"{object_class}:observability_failed")
        if not bool(h2_by_class.get(object_class, {}).get("passed")):
            failures.append(f"{object_class}:planted_h2_failed")
        if not bool(h1_by_class.get(object_class, {}).get("positive_passed")):
            failures.append(f"{object_class}:planted_h1_failed")
        if bool(h1_by_class.get(object_class, {}).get("null_passed")):
            failures.append(f"{object_class}:null_false_positive")
    if not bool(real_regression_summary.get("preserved_phase18_t_final")):
        failures.append("real_family_regression_failed")
    validated = not failures
    return {
        "detectability_validated": validated,
        "decision": "detectability_validated" if validated else "retain_phase18",
        "decision_status": "final",
        "failure_reasons": failures,
        "upgraded_strongest_claim": validated,
    }


def render_phase19_status_markdown(
    *,
    manifest_summary: dict[str, Any],
    observability_summary: dict[str, Any],
    planted_h2_summary: dict[str, Any],
    planted_h1_summary: dict[str, Any],
    real_regression_summary: dict[str, Any],
    detectability_summary: dict[str, Any],
) -> str:
    lines = [
        "# Phase 19 Status",
        "",
        "## Frozen Framing",
        "- `G1_binding`: bounded hypothesis-final success",
        "- `G2_distributed_geometry`: family-final negative under Phase 18 unless detectability is validated",
        "- `Phase 19` is an epistemic upgrade check, not a repair branch.",
        "",
        "## Manifest",
        f"- `class_count = {int(manifest_summary.get('class_count') or 0)}`",
        f"- `anchor_found_count = {int(manifest_summary.get('anchor_found_count') or 0)}`",
        f"- `anchor_missing_classes = {manifest_summary.get('anchor_missing_classes') or []}`",
        "",
        "## Planted Observability",
    ]
    for row in list(observability_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `passed = {bool(row.get('passed'))}`, `selected_aligner = {row.get('selected_aligner')}`"
        )
    lines.extend(
        [
            "",
            "## Planted H2",
        ]
    )
    for row in list(planted_h2_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `passed = {bool(row.get('passed'))}`, `null_passed_count = {int(row.get('null_passed_count') or 0)}`"
        )
    lines.extend(
        [
            "",
            "## Planted H1",
        ]
    )
    for row in list(planted_h1_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `positive_passed = {bool(row.get('positive_passed'))}`, `null_false_positive_rate = {row.get('null_false_positive_rate')}`"
        )
    lines.extend(
        [
            "",
            "## Real-Family Regression",
            f"- `preserved_phase18_t_final = {bool(real_regression_summary.get('preserved_phase18_t_final'))}`",
            "",
            "## Detectability",
            f"- `detectability_validated = {bool(detectability_summary.get('detectability_validated'))}`",
            f"- `decision = {detectability_summary.get('decision')}`",
            f"- `failure_reasons = {detectability_summary.get('failure_reasons') or []}`",
        ]
    )
    return "\n".join(lines) + "\n"
