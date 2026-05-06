from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from interventions.generation import (
    GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS,
    GEOMETRY_INTERACTION_OBJECT_CLASS,
    GEOMETRY_PLANT_KIND_H1_POSITIVE,
    GEOMETRY_PLANT_KIND_H2_POSITIVE,
    GEOMETRY_PLANT_KIND_NULL_DECOY,
    GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS,
    normalize_geometry_plant_spec,
)

from . import interaction_objects as p17
from .family_closure import BRIDGE_OBJECT_CLASS, QUERY_SOURCE_OBJECT_CLASS, load_json
from .planted_recoverability import (
    GENERIC_GAIN_OVERLAP_MAX,
    H1_RUN_RECOVERY_MIN_RECALL,
    NULL_DIRECT_H1_MAX_FALSE_POSITIVE_RATE,
    evaluate_planted_h1_run,
    evaluate_planted_h2_bundle,
    evaluate_planted_observability,
    select_planted_aligner,
    summarize_planted_h1_class,
)


CLASS_ORDER: tuple[str, ...] = (
    GEOMETRY_INTERACTION_OBJECT_CLASS,
    BRIDGE_OBJECT_CLASS,
    QUERY_SOURCE_OBJECT_CLASS,
)

ANCHORABILITY_MIN_COUNT = 2


def _class_label(object_class: str) -> str:
    return {
        GEOMETRY_INTERACTION_OBJECT_CLASS: "C1",
        BRIDGE_OBJECT_CLASS: "C2",
        QUERY_SOURCE_OBJECT_CLASS: "C3",
    }[object_class]


def _synthetic_anchor_specs() -> dict[str, list[dict[str, Any]]]:
    return {
        GEOMETRY_INTERACTION_OBJECT_CLASS: [
            {
                "object_id": "interaction_predicate_anchor_a",
                "object_class": GEOMETRY_INTERACTION_OBJECT_CLASS,
                "layer_index": 6,
                "source_run": "phase12_geometry_h2_full_seed11",
                "decoder_step_key": "predicate_value",
                "source_span_keys": ["predicate"],
                "head_selector": {"mode": "explicit", "indices": [0]},
                "read_basis": {"mode": "explicit", "basis_ids": ["prelogit_basis_0"]},
                "store_condition": {"mode": "none"},
                "matched_control_type": "wrong_event_same_step",
                "generic_basis_ids": ["prelogit_basis_15"],
                "generic_gain_overlap": 0.0,
                "non_flat_survivor": True,
                "template_family": "anchor_a",
            },
            {
                "object_id": "interaction_agent_anchor_b",
                "object_class": GEOMETRY_INTERACTION_OBJECT_CLASS,
                "layer_index": 6,
                "source_run": "phase12_geometry_h2_full_seed11",
                "decoder_step_key": "agent_value",
                "source_span_keys": ["agent", "filler::agent"],
                "head_selector": {"mode": "explicit", "indices": [0, 1]},
                "read_basis": {"mode": "explicit", "basis_ids": ["prelogit_basis_0", "prelogit_basis_1"]},
                "store_condition": {"mode": "matched_role_store", "selector_type": "semantic_role_bundle"},
                "matched_control_type": "wrong_event_same_step",
                "generic_basis_ids": ["prelogit_basis_15", "prelogit_basis_16"],
                "generic_gain_overlap": 0.0,
                "non_flat_survivor": True,
                "template_family": "anchor_b",
            },
        ],
        BRIDGE_OBJECT_CLASS: [
            {
                "object_id": "cross_step_bridge_predicate_to_agent_anchor_a",
                "object_class": BRIDGE_OBJECT_CLASS,
                "layer_index": 6,
                "source_run": "phase12_geometry_h2_full_seed11",
                "source_step_key": "predicate_value",
                "target_step_key": "agent_value",
                "source_span_keys": ["predicate"],
                "head_selector": {"mode": "explicit", "indices": [0]},
                "bridge_basis": {"mode": "explicit", "basis_ids": ["prelogit_basis_0"]},
                "store_condition": {"mode": "none"},
                "matched_control_type": "wrong_event_same_cardinality",
                "generic_basis_ids": ["prelogit_basis_15"],
                "generic_gain_overlap": 0.0,
                "non_flat_survivor": True,
                "template_family": "anchor_a",
            },
            {
                "object_id": "cross_step_bridge_predicate_to_theme_anchor_b",
                "object_class": BRIDGE_OBJECT_CLASS,
                "layer_index": 6,
                "source_run": "phase12_geometry_h2_full_seed11",
                "source_step_key": "predicate_value",
                "target_step_key": "theme_value",
                "source_span_keys": ["predicate"],
                "head_selector": {"mode": "explicit", "indices": [0, 1]},
                "bridge_basis": {"mode": "explicit", "basis_ids": ["prelogit_basis_0", "prelogit_basis_1"]},
                "store_condition": {"mode": "matched_role_store", "selector_type": "semantic_role_bundle"},
                "matched_control_type": "wrong_event_same_cardinality",
                "generic_basis_ids": ["prelogit_basis_15", "prelogit_basis_16"],
                "generic_gain_overlap": 0.0,
                "non_flat_survivor": True,
                "template_family": "anchor_b",
            },
        ],
        QUERY_SOURCE_OBJECT_CLASS: [
            {
                "object_id": "query_source_relation_predicate_anchor_a",
                "object_class": QUERY_SOURCE_OBJECT_CLASS,
                "layer_index": 6,
                "source_run": "phase12_geometry_h2_full_seed11",
                "query_step_key": "predicate_value",
                "source_span_keys": ["predicate"],
                "head_selector": {"mode": "explicit", "indices": [0]},
                "query_basis": {"mode": "explicit", "basis_ids": ["prelogit_basis_0"]},
                "read_basis": {"mode": "explicit", "basis_ids": ["prelogit_basis_1"]},
                "store_condition": {"mode": "none"},
                "matched_control_type": "wrong_event_same_cardinality",
                "generic_basis_ids": ["prelogit_basis_15"],
                "generic_gain_overlap": 0.0,
                "non_flat_survivor": True,
                "template_family": "anchor_a",
            },
            {
                "object_id": "query_source_relation_agent_anchor_b",
                "object_class": QUERY_SOURCE_OBJECT_CLASS,
                "layer_index": 6,
                "source_run": "phase12_geometry_h2_full_seed11",
                "query_step_key": "agent_value",
                "source_span_keys": ["agent", "filler::agent"],
                "head_selector": {"mode": "explicit", "indices": [0, 1]},
                "query_basis": {"mode": "explicit", "basis_ids": ["prelogit_basis_0"]},
                "read_basis": {"mode": "explicit", "basis_ids": ["prelogit_basis_1", "prelogit_basis_2"]},
                "store_condition": {"mode": "matched_role_store", "selector_type": "semantic_role_bundle"},
                "matched_control_type": "wrong_event_same_cardinality",
                "generic_basis_ids": ["prelogit_basis_15", "prelogit_basis_16"],
                "generic_gain_overlap": 0.0,
                "non_flat_survivor": True,
                "template_family": "anchor_b",
            },
        ],
    }


def get_stage20_synthetic_anchor_specs(
    *,
    object_class: str | None = None,
) -> dict[str, list[dict[str, Any]]] | list[dict[str, Any]]:
    specs = _synthetic_anchor_specs()
    if object_class is None:
        return {key: [dict(row) for row in value] for key, value in specs.items()}
    return [dict(row) for row in specs[str(object_class)]]


def get_stage20_argument_anchor_specs(object_class: str) -> list[dict[str, Any]]:
    anchors = [dict(row) for row in _synthetic_anchor_specs()[str(object_class)]]
    argument_anchors: list[dict[str, Any]] = []
    for anchor in anchors:
        if object_class == GEOMETRY_INTERACTION_OBJECT_CLASS:
            step_key = str(anchor.get("decoder_step_key") or "")
        elif object_class == BRIDGE_OBJECT_CLASS:
            step_key = str(anchor.get("target_step_key") or "")
        else:
            step_key = str(anchor.get("query_step_key") or "")
        if step_key != "predicate_value":
            argument_anchors.append(anchor)
    return argument_anchors


def _family_spec(
    *,
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
                "object_class": object_spec["object_class"],
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
        "anchor_object_id": str(object_spec["object_id"]),
        "geometry_plant_spec": plant_spec,
    }


def build_stage_a_manifest(config: dict[str, Any]) -> dict[str, Any]:
    anchors = _synthetic_anchor_specs()
    h2_bases = list(config.get("h2_base_runs") or [])
    h1_probes = list(config.get("h1_probe_runs") or [])
    classes: list[dict[str, Any]] = []
    for object_class in CLASS_ORDER:
        class_anchors = [dict(row) for row in anchors[object_class]]
        h2_positive = [
            _family_spec(
                object_spec=class_anchors[0],
                plant_kind=GEOMETRY_PLANT_KIND_H2_POSITIVE,
                base_run=str(h2_bases[0]["run_id"]),
                physical_family=str(h2_bases[0]["run_id"]),
                head_scale=1.75,
                basis_scale=1.5,
                store_mode="matched" if str(class_anchors[0]["store_condition"]["mode"]) != "none" else "none",
                source_mode="correct",
                intervention_scope="full_object_scope",
                default_attenuation=0.0,
            ),
            _family_spec(
                object_spec=class_anchors[1],
                plant_kind=GEOMETRY_PLANT_KIND_H2_POSITIVE,
                base_run=str(h2_bases[1]["run_id"]),
                physical_family=str(h2_bases[1]["run_id"]),
                head_scale=1.75,
                basis_scale=1.5,
                store_mode="matched" if str(class_anchors[1]["store_condition"]["mode"]) != "none" else "none",
                source_mode="correct",
                intervention_scope="full_object_scope",
                default_attenuation=0.0,
            ),
        ]
        h1_positive = [
            _family_spec(
                object_spec=class_anchors[0],
                plant_kind=GEOMETRY_PLANT_KIND_H1_POSITIVE,
                base_run=str(h1_probes[0]["run_id"]),
                physical_family=str(h1_probes[0]["physical_family"]),
                head_scale=1.25,
                basis_scale=1.1,
                store_mode="rescue_only" if str(class_anchors[0]["store_condition"]["mode"]) != "none" else "none",
                source_mode="correct",
                intervention_scope="metric_scope_only",
                default_attenuation=0.5,
            ),
            _family_spec(
                object_spec=class_anchors[1],
                plant_kind=GEOMETRY_PLANT_KIND_H1_POSITIVE,
                base_run=str(h1_probes[1]["run_id"]),
                physical_family=str(h1_probes[1]["physical_family"]),
                head_scale=1.25,
                basis_scale=1.1,
                store_mode="rescue_only" if str(class_anchors[1]["store_condition"]["mode"]) != "none" else "none",
                source_mode="correct",
                intervention_scope="metric_scope_only",
                default_attenuation=0.5,
            ),
            _family_spec(
                object_spec=class_anchors[0],
                plant_kind=GEOMETRY_PLANT_KIND_H1_POSITIVE,
                base_run=str(h1_probes[2]["run_id"]),
                physical_family=str(h1_probes[2]["physical_family"]),
                head_scale=1.25,
                basis_scale=1.1,
                store_mode="rescue_only" if str(class_anchors[0]["store_condition"]["mode"]) != "none" else "none",
                source_mode="correct",
                intervention_scope="metric_scope_only",
                default_attenuation=0.5,
            ),
        ]
        null_decoy = []
        for family in h2_positive + h1_positive:
            recipe = dict(family["geometry_plant_spec"])
            null_decoy.append(
                _family_spec(
                    object_spec=next(anchor for anchor in class_anchors if str(anchor["object_id"]) == str(family["anchor_object_id"])),
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
                "usable_anchor_count": len(class_anchors),
                "anchor_rows": class_anchors,
                "planted_h2_families": h2_positive,
                "planted_h1_families": h1_positive,
                "matched_null_families": null_decoy,
            }
        )
    return {"class_count": len(classes), "classes": classes}


def summarize_stage_a_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = list(manifest.get("classes") or [])
    return {
        "class_count": len(rows),
        "total_run_count": sum(
            len(list(row.get("planted_h2_families") or []))
            + len(list(row.get("planted_h1_families") or []))
            + len(list(row.get("matched_null_families") or []))
            for row in rows
        ),
        "class_rows": rows,
    }


def summarize_anchorability(manifest_summary: dict[str, Any]) -> dict[str, Any]:
    class_rows: list[dict[str, Any]] = []
    for row in list(manifest_summary.get("class_rows") or []):
        anchors = list(row.get("anchor_rows") or [])
        usable = [
            anchor
            for anchor in anchors
            if float(anchor.get("generic_gain_overlap") or 0.0) <= GENERIC_GAIN_OVERLAP_MAX
            and bool(anchor.get("object_id"))
        ]
        class_rows.append(
            {
                "object_class": str(row["object_class"]),
                "usable_anchor_count": len(usable),
                "passed": len(usable) >= ANCHORABILITY_MIN_COUNT,
                "anchor_ids": [str(anchor["object_id"]) for anchor in usable],
            }
        )
    return {
        "class_rows": class_rows,
        "passed_count": sum(1 for row in class_rows if bool(row["passed"])),
    }


def _load_bundle(root: Path, entry: dict[str, object]) -> dict[str, object]:
    summary_path = root / str(entry["summary_path"])
    if str(entry["bundle_type"]) == "protocol":
        return p17._load_protocol_bundle(root, summary_path)
    return p17._load_discovery_bundle(root, summary_path)


def _bundle_index(config: dict[str, object], key: str) -> dict[str, dict[str, object]]:
    return {str(row["run_id"]): dict(row) for row in list(config.get(key) or [])}


def evaluate_class_observability(
    *,
    root: Path,
    class_row: dict[str, Any],
    bundle_index: dict[str, dict[str, Any]],
    max_length: int,
) -> dict[str, Any]:
    families = (
        list(class_row.get("planted_h2_families") or [])
        + list(class_row.get("planted_h1_families") or [])
        + list(class_row.get("matched_null_families") or [])
    )
    per_aligner_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for aligner_name in ("A1", "A2", "A3"):
        eval_rows = []
        for family in families:
            bundle = _load_bundle(root, bundle_index[str(family["base_run"])])
            object_spec = dict(family["geometry_plant_spec"]["object_spec"])
            summary = evaluate_planted_observability(
                bundle=bundle,
                object_class=str(class_row["object_class"]),
                object_spec=object_spec,
                plant_spec=dict(family["geometry_plant_spec"]),
                aligner_name=aligner_name,
                max_length=max_length,
            )
            eval_rows.append(
                {
                    "plant_kind": str(family["plant_kind"]),
                    "physical_family": str(family["physical_family"]),
                    "anchor_object_id": str(family["anchor_object_id"]),
                    **summary,
                }
            )
        family_rows.extend(eval_rows)
        per_aligner_rows.append(
            {
                "aligner": aligner_name,
                "intended_step_alignment_rate": min(float(row["intended_step_alignment_rate"]) for row in eval_rows)
                if eval_rows
                else 0.0,
                "unintended_step_alignment_rate": max(float(row["unintended_step_alignment_rate"]) for row in eval_rows)
                if eval_rows
                else 1.0,
                "passed": all(bool(row["passed"]) for row in eval_rows),
            }
        )
    passed_aligners = [row for row in per_aligner_rows if bool(row["passed"])]
    selected = select_planted_aligner(per_aligner_rows=per_aligner_rows)
    return {
        "object_class": str(class_row["object_class"]),
        "passed": len(passed_aligners) >= 2,
        "passed_aligner_count": len(passed_aligners),
        "selected_aligner": selected["selected_aligner"],
        "selected_summary": selected["selected_summary"],
        "aligner_rows": per_aligner_rows,
        "family_rows": family_rows,
    }


def summarize_planted_h2_class(
    *,
    root: Path,
    class_row: dict[str, Any],
    bundle_index: dict[str, dict[str, Any]],
    observability_row: dict[str, Any],
    max_length: int,
) -> dict[str, Any]:
    if not bool(observability_row.get("passed")):
        return {
            "object_class": str(class_row["object_class"]),
            "passed": False,
            "null_passed_count": 0,
            "failure_reason": "observability_failed",
            "positive_rows": [],
            "null_rows": [],
        }
    aligner_name = str(observability_row["selected_aligner"])
    positive_rows = []
    null_rows = []
    h2_families = list(class_row.get("planted_h2_families") or [])
    null_families = [
        dict(row)
        for row in list(class_row.get("matched_null_families") or [])
        if str(row["base_run"]) in {str(family["base_run"]) for family in h2_families}
    ]
    for family in h2_families:
        bundle = _load_bundle(root, bundle_index[str(family["base_run"])])
        positive_rows.append(
            evaluate_planted_h2_bundle(
                bundle=bundle,
                object_class=str(class_row["object_class"]),
                object_spec=dict(family["geometry_plant_spec"]["object_spec"]),
                plant_spec=dict(family["geometry_plant_spec"]),
                aligner_name=aligner_name,
                max_length=max_length,
            )
        )
    for family in null_families:
        bundle = _load_bundle(root, bundle_index[str(family["base_run"])])
        null_rows.append(
            evaluate_planted_h2_bundle(
                bundle=bundle,
                object_class=str(class_row["object_class"]),
                object_spec=dict(family["geometry_plant_spec"]["object_spec"]),
                plant_spec=dict(family["geometry_plant_spec"]),
                aligner_name=aligner_name,
                max_length=max_length,
            )
        )
    return {
        "object_class": str(class_row["object_class"]),
        "passed": bool(positive_rows) and all(bool(row["passed"]) for row in positive_rows) and not any(
            bool(row["passed"]) for row in null_rows
        ),
        "null_passed_count": sum(1 for row in null_rows if bool(row["passed"])),
        "positive_rows": positive_rows,
        "null_rows": null_rows,
    }


def summarize_planted_h1_class_for_stage20(
    *,
    root: Path,
    class_row: dict[str, Any],
    bundle_index: dict[str, dict[str, Any]],
    observability_row: dict[str, Any],
    max_length: int,
) -> dict[str, Any]:
    if not bool(observability_row.get("passed")):
        return {
            "object_class": str(class_row["object_class"]),
            "positive_passed": False,
            "null_passed": False,
            "null_false_positive_rate": 0.0,
            "failure_reason": "observability_failed",
            "positive_rows": [],
            "null_rows": [],
        }
    aligner_name = str(observability_row["selected_aligner"])
    positive_run_rows = []
    for family in list(class_row.get("planted_h1_families") or []):
        bundle = _load_bundle(root, bundle_index[str(family["base_run"])])
        positive_run_rows.append(
            evaluate_planted_h1_run(
                bundle=bundle,
                object_class=str(class_row["object_class"]),
                object_spec=dict(family["geometry_plant_spec"]["object_spec"]),
                plant_spec=dict(family["geometry_plant_spec"]),
                aligner_name=aligner_name,
                physical_family=str(family["physical_family"]),
                max_length=max_length,
            )
        )
    null_run_rows = []
    for family in list(class_row.get("matched_null_families") or []):
        if str(family["physical_family"]) not in {
            str(row["physical_family"]) for row in list(class_row.get("planted_h1_families") or [])
        }:
            continue
        bundle = _load_bundle(root, bundle_index[str(family["base_run"])])
        null_run_rows.append(
            evaluate_planted_h1_run(
                bundle=bundle,
                object_class=str(class_row["object_class"]),
                object_spec=dict(family["geometry_plant_spec"]["object_spec"]),
                plant_spec=dict(family["geometry_plant_spec"]),
                aligner_name=aligner_name,
                physical_family=str(family["physical_family"]),
                max_length=max_length,
            )
        )
    anchor_object_id = str((class_row.get("anchor_rows") or [{}])[0].get("object_id") or "")
    positive_summary = summarize_planted_h1_class(
        object_class=str(class_row["object_class"]),
        plant_kind=GEOMETRY_PLANT_KIND_H1_POSITIVE,
        object_id=anchor_object_id,
        run_rows=positive_run_rows,
    )
    null_false_positive_rate = round(
        mean(int(bool(row.get("run_recovered"))) for row in null_run_rows),
        6,
    ) if null_run_rows else 0.0
    return {
        "object_class": str(class_row["object_class"]),
        "positive_passed": bool(positive_summary["passed"]),
        "positive_summary": positive_summary,
        "null_passed": bool(null_false_positive_rate > NULL_DIRECT_H1_MAX_FALSE_POSITIVE_RATE),
        "null_false_positive_rate": null_false_positive_rate,
        "positive_rows": positive_run_rows,
        "null_rows": null_run_rows,
    }


def _fallback_target(
    *,
    class_label: str,
    object_class: str,
    run_name: str,
    target_id: str,
    phase11_prefix: str,
    panel_regime: str,
    notes: str,
    controls: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "target_id": target_id,
        "run_name": run_name,
        "phase11_prefix": phase11_prefix,
        "panel_regime": panel_regime,
        "notes": notes,
        "tags": ["phase20", "stage_b", class_label, object_class],
        "num_train_epochs": 2,
        "train_batch_size": 4,
        "eval_batch_size": 4,
        "learning_rate": 5e-5,
        "seed": 20,
    }
    payload.update(controls)
    return payload


def build_stage_b_panel_for_class(
    *,
    object_class: str,
    benchmark: str,
    model_name_or_path: str,
    prompt_style: str,
) -> dict[str, Any]:
    class_label = _class_label(object_class).lower()
    if object_class == GEOMETRY_INTERACTION_OBJECT_CLASS:
        h2_controls = {
            "decoder_cross_attention_scale": 1.75,
            "decoder_readout_subspace_keep_ratio": 0.35,
            "decoder_readout_subspace_mode": "keep",
            "decoder_readout_subspace_seed": 20,
        }
        h1_controls = {
            "decoder_cross_attention_scale": 1.25,
            "decoder_readout_subspace_keep_ratio": 0.25,
            "decoder_readout_subspace_mode": "keep",
            "decoder_readout_subspace_seed": 21,
        }
        null_controls = {
            "decoder_cross_attention_scale": 1.25,
            "decoder_readout_keep_ratio": 0.35,
        }
    elif object_class == BRIDGE_OBJECT_CLASS:
        h2_controls = {
            "decoder_cross_attention_scale": 1.75,
            "decoder_cross_attention_layer_keep_ratio": 0.5,
            "decoder_cross_attention_layer_policy": "alternating",
            "decoder_cross_attention_layer_seed": 20,
        }
        h1_controls = {
            "decoder_cross_attention_scale": 1.25,
            "decoder_cross_attention_layer_keep_ratio": 0.5,
            "decoder_cross_attention_layer_policy": "alternating",
            "decoder_cross_attention_layer_seed": 21,
            "decoder_cross_attention_relation_barrier_keep_ratio": 0.35,
            "decoder_cross_attention_relation_barrier_scale": 0.5,
        }
        null_controls = {
            "decoder_cross_attention_scale": 1.25,
            "decoder_cross_attention_layer_keep_ratio": 0.5,
            "decoder_cross_attention_layer_policy": "top",
            "decoder_cross_attention_layer_seed": 22,
        }
    else:
        h2_controls = {
            "decoder_cross_attention_relation_barrier_keep_ratio": 0.35,
            "decoder_cross_attention_relation_barrier_scale": 1.75,
            "decoder_readout_subspace_keep_ratio": 0.35,
            "decoder_readout_subspace_mode": "keep",
            "decoder_readout_subspace_seed": 20,
        }
        h1_controls = {
            "decoder_cross_attention_relation_barrier_keep_ratio": 0.35,
            "decoder_cross_attention_relation_barrier_scale": 1.25,
            "decoder_readout_subspace_keep_ratio": 0.25,
            "decoder_readout_subspace_mode": "keep",
            "decoder_readout_subspace_seed": 21,
        }
        null_controls = {
            "decoder_cross_attention_scale": 1.25,
            "decoder_readout_keep_ratio": 0.35,
        }
    panel_name = f"phase20_stageb_{class_label}"
    return {
        "panel_name": panel_name,
        "benchmark": benchmark,
        "model_name_or_path": model_name_or_path,
        "prompt_style": prompt_style,
        "analysis_family": "seq2seq_lm",
        "metadata": {
            "phase": "phase20",
            "stage": "stage_b",
            "object_class": object_class,
            "operationalization_family": "geometry",
        },
        "targets": [
            _fallback_target(
                class_label=class_label,
                object_class=object_class,
                run_name=f"{panel_name}_planted_h2",
                target_id=f"{panel_name}_planted_h2",
                phase11_prefix=f"{panel_name}_planted_h2",
                panel_regime="planted_h2",
                notes="Phase 20 Stage B class-local constrained fine-tune planted H2 fallback.",
                controls=h2_controls,
            ),
            _fallback_target(
                class_label=class_label,
                object_class=object_class,
                run_name=f"{panel_name}_planted_h1",
                target_id=f"{panel_name}_planted_h1",
                phase11_prefix=f"{panel_name}_planted_h1",
                panel_regime="planted_h1",
                notes="Phase 20 Stage B class-local constrained fine-tune planted H1 fallback.",
                controls=h1_controls,
            ),
            _fallback_target(
                class_label=class_label,
                object_class=object_class,
                run_name=f"{panel_name}_matched_null",
                target_id=f"{panel_name}_matched_null",
                phase11_prefix=f"{panel_name}_matched_null",
                panel_regime="matched_null",
                notes="Phase 20 Stage B class-local constrained fine-tune matched null fallback.",
                controls=null_controls,
            ),
        ],
    }


def summarize_stage_b_preparation(
    *,
    failed_classes: list[str],
    panel_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "triggered": bool(failed_classes),
        "failed_classes": list(failed_classes),
        "panel_paths": dict(panel_paths),
        "stage_b_executed": False,
    }


def summarize_phase20_detectability(
    *,
    anchorability_summary: dict[str, Any],
    observability_summary: dict[str, Any],
    planted_h2_summary: dict[str, Any],
    planted_h1_summary: dict[str, Any],
    real_regression_summary: dict[str, Any],
) -> dict[str, Any]:
    anchor_by_class = {str(row["object_class"]): row for row in list(anchorability_summary.get("class_rows") or [])}
    obs_by_class = {str(row["object_class"]): row for row in list(observability_summary.get("class_rows") or [])}
    h2_by_class = {str(row["object_class"]): row for row in list(planted_h2_summary.get("class_rows") or [])}
    h1_by_class = {str(row["object_class"]): row for row in list(planted_h1_summary.get("class_rows") or [])}
    failures: list[str] = []
    for object_class in CLASS_ORDER:
        if not bool(anchor_by_class.get(object_class, {}).get("passed")):
            failures.append(f"{object_class}:anchorability_failed")
        if not bool(obs_by_class.get(object_class, {}).get("passed")):
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


def render_phase20_status_markdown(
    *,
    manifest_summary: dict[str, Any],
    anchorability_summary: dict[str, Any],
    observability_summary: dict[str, Any],
    planted_h2_summary: dict[str, Any],
    planted_h1_summary: dict[str, Any],
    stage_b_summary: dict[str, Any],
    real_regression_summary: dict[str, Any],
    detectability_summary: dict[str, Any],
) -> str:
    lines = [
        "# Phase 20 Status",
        "",
        "## Frozen Framing",
        "- `G1_binding`: bounded hypothesis-final success",
        "- `G2_distributed_geometry`: `Phase 18` family-final negative unless class-local detectability is validated",
        "- `Phase 20` is a class-local detectability harness, not a new real-family search branch.",
        "",
        "## Stage A Manifest",
        f"- `class_count = {int(manifest_summary.get('class_count') or 0)}`",
        f"- `total_run_count = {int(manifest_summary.get('total_run_count') or 0)}`",
        "",
        "## Anchorability",
    ]
    for row in list(anchorability_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `passed = {bool(row.get('passed'))}`, `usable_anchor_count = {int(row.get('usable_anchor_count') or 0)}`"
        )
    lines.extend(["", "## Observability"])
    for row in list(observability_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `passed = {bool(row.get('passed'))}`, `passed_aligner_count = {int(row.get('passed_aligner_count') or 0)}`, `selected_aligner = {row.get('selected_aligner')}`"
        )
    lines.extend(["", "## Planted H2"])
    for row in list(planted_h2_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `passed = {bool(row.get('passed'))}`, `null_passed_count = {int(row.get('null_passed_count') or 0)}`"
        )
    lines.extend(["", "## Planted H1"])
    for row in list(planted_h1_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `positive_passed = {bool(row.get('positive_passed'))}`, `null_false_positive_rate = {float(row.get('null_false_positive_rate') or 0.0)}`"
        )
    lines.extend(
        [
            "",
            "## Stage B Preparation",
            f"- `triggered = {bool(stage_b_summary.get('triggered'))}`",
            f"- `failed_classes = {stage_b_summary.get('failed_classes') or []}`",
            f"- `stage_b_executed = {bool(stage_b_summary.get('stage_b_executed'))}`",
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


def load_config(path: str | Path) -> dict[str, Any]:
    return load_json(path)
