from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from interventions.generation import (
    GEOMETRY_PLANT_KIND_H1_POSITIVE,
    GEOMETRY_PLANT_KIND_H2_POSITIVE,
    aggregate_control_value,
    normalize_geometry_plant_spec,
)

from . import interaction_objects as p17
from . import planted_recoverability as p19
from . import model_native_detectability as p20
from .family_closure import ALIGNER_NAMES, _align_with_selected, _margin_mean_for_positions, _object_view, load_json


CLASS_ORDER = p20.CLASS_ORDER
ARGUMENT_OBSERVABILITY_MIN_ALIGNERS = 2


def _class_label(object_class: str) -> str:
    return {
        "interaction": "C1",
        "cross_step_bridge": "C2",
        "query_source_relation": "C3",
    }[object_class]


def _bundle_from_prefix(root: Path, phase11_prefix: str) -> dict[str, Any]:
    return p17._load_protocol_bundle(root, root / "reports/protocol_runs" / f"{phase11_prefix}_summary.json")


def _argument_anchor_candidates(object_class: str) -> list[dict[str, Any]]:
    return p20.get_stage20_argument_anchor_specs(object_class)


def build_stage21_manifest(
    *,
    root: Path,
    panel_paths: dict[str, str],
) -> dict[str, Any]:
    class_rows: list[dict[str, Any]] = []
    for object_class in CLASS_ORDER:
        panel_path = root / str(panel_paths[object_class])
        panel = json.loads(panel_path.read_text(encoding="utf-8"))
        argument_anchors = _argument_anchor_candidates(object_class)
        class_rows.append(
            {
                "object_class": object_class,
                "class_label": _class_label(object_class),
                "panel_path": str(panel_path.relative_to(root)),
                "usable_anchor_count": len(argument_anchors),
                "argument_anchor_ids": [str(row["object_id"]) for row in argument_anchors],
                "argument_anchor_specs": argument_anchors,
                "target_rows": [
                    {
                        "target_id": str(target["target_id"]),
                        "run_name": str(target.get("run_name") or target["target_id"]),
                        "phase11_prefix": str(target.get("phase11_prefix") or target.get("run_name") or target["target_id"]),
                        "panel_regime": str(target.get("panel_regime") or "unspecified"),
                    }
                    for target in list(panel.get("targets") or [])
                ],
            }
        )
    return {
        "class_count": len(class_rows),
        "class_rows": class_rows,
    }


def summarize_anchorability(manifest: dict[str, Any]) -> dict[str, Any]:
    class_rows: list[dict[str, Any]] = []
    for row in list(manifest.get("class_rows") or []):
        usable_anchor_count = int(row.get("usable_anchor_count") or 0)
        class_rows.append(
            {
                "object_class": str(row["object_class"]),
                "passed": usable_anchor_count >= 1,
                "usable_anchor_count": usable_anchor_count,
                "anchor_ids": list(row.get("argument_anchor_ids") or []),
            }
        )
    return {
        "class_rows": class_rows,
        "passed_count": sum(1 for row in class_rows if bool(row["passed"])),
    }


def _evaluate_model_native_observability(
    *,
    bundle: dict[str, Any],
    object_class: str,
    object_spec: dict[str, Any],
    aligner_name: str,
    max_length: int,
) -> dict[str, Any]:
    adapter, layer_index = p17.build_adapter_from_bundle(bundle)
    span_index = p17._index_span_records(bundle["span_records"], layer_index=layer_index)
    live_span_cache: dict[str, dict[str, Any]] = {}
    intended_step = p19._intended_step_key(object_class, object_spec)
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
        alignments = _align_with_selected(
            aligner_name=aligner_name,
            tokenizer=adapter.tokenizer,
            target_text=str(item["target_form"]),
            record=item,
            forward=forward,
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
        "passed": intended_rate >= 0.8 and unintended_rate <= 0.2,
    }


def evaluate_argument_class_observability(
    *,
    root: Path,
    class_row: dict[str, Any],
    max_length: int,
) -> dict[str, Any]:
    target_rows = list(class_row.get("target_rows") or [])
    candidate_rows: list[dict[str, Any]] = []
    for anchor in list(class_row.get("argument_anchor_specs") or []):
        per_aligner_rows: list[dict[str, Any]] = []
        family_rows: list[dict[str, Any]] = []
        for aligner_name in ALIGNER_NAMES:
            eval_rows: list[dict[str, Any]] = []
            for target in target_rows:
                bundle = _bundle_from_prefix(root, str(target["phase11_prefix"]))
                summary = _evaluate_model_native_observability(
                    bundle=bundle,
                    object_class=str(class_row["object_class"]),
                    object_spec=dict(anchor),
                    aligner_name=aligner_name,
                    max_length=max_length,
                )
                eval_rows.append(
                    {
                        "panel_regime": str(target["panel_regime"]),
                        "phase11_prefix": str(target["phase11_prefix"]),
                        "run_name": str(target["run_name"]),
                        "anchor_object_id": str(anchor["object_id"]),
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
        passed_aligners = [row for row in per_aligner_rows if bool(row.get("passed"))]
        selected = p19.select_planted_aligner(per_aligner_rows=per_aligner_rows)
        candidate_rows.append(
            {
                "anchor_object_id": str(anchor["object_id"]),
                "object_spec": dict(anchor),
                "passed": len(passed_aligners) >= ARGUMENT_OBSERVABILITY_MIN_ALIGNERS,
                "passed_aligner_count": len(passed_aligners),
                "selected_aligner": str(selected["selected_aligner"]),
                "selected_summary": dict(selected["selected_summary"]),
                "aligner_rows": per_aligner_rows,
                "family_rows": family_rows,
            }
        )

    def _rank_key(row: dict[str, Any]) -> tuple[int, int, float]:
        selected = dict(row.get("selected_summary") or {})
        margin = float(selected.get("intended_step_alignment_rate") or 0.0) - float(
            selected.get("unintended_step_alignment_rate") or 0.0
        )
        return (int(bool(row.get("passed"))), int(row.get("passed_aligner_count") or 0), margin)

    selected_candidate = max(candidate_rows, key=_rank_key) if candidate_rows else None
    return {
        "object_class": str(class_row["object_class"]),
        "passed": bool(selected_candidate and selected_candidate.get("passed")),
        "usable_anchor_count": int(class_row.get("usable_anchor_count") or 0),
        "selected_anchor_object_id": str(selected_candidate.get("anchor_object_id") or "") if selected_candidate else "",
        "selected_aligner": str(selected_candidate.get("selected_aligner") or "A1") if selected_candidate else "A1",
        "passed_aligner_count": int(selected_candidate.get("passed_aligner_count") or 0) if selected_candidate else 0,
        "selected_summary": dict(selected_candidate.get("selected_summary") or {}) if selected_candidate else {},
        "candidate_rows": candidate_rows,
    }


def _model_native_recipe(
    *,
    object_spec: dict[str, Any],
    panel_regime: str,
    base_run: str,
) -> dict[str, Any]:
    if panel_regime == "planted_h2":
        plant_kind = GEOMETRY_PLANT_KIND_H2_POSITIVE
        head_scale = 1.75
        basis_scale = 1.5
        default_attenuation = 0.0
        store_mode = "matched" if p19._supports_store(object_spec) else "none"
        intervention_scope = "full_object_scope"
    else:
        plant_kind = GEOMETRY_PLANT_KIND_H1_POSITIVE
        head_scale = 1.25
        basis_scale = 1.1
        default_attenuation = 0.5
        store_mode = "rescue_only" if p19._supports_store(object_spec) else "none"
        intervention_scope = "metric_scope_only"
    return normalize_geometry_plant_spec(
        plant_kind,
        json.dumps(
            {
                "plant_id": f"{object_spec['object_id']}::{panel_regime}::{base_run}",
                "object_class": object_spec["object_class"],
                "object_spec": dict(object_spec),
                "base_run": base_run,
                "head_scale": head_scale,
                "basis_scale": basis_scale,
                "store_mode": store_mode,
                "source_mode": "correct",
                "intervention_scope": intervention_scope,
                "default_attenuation": default_attenuation,
            }
        ),
    ) or {}


def _evaluate_model_native_h2_bundle(
    *,
    bundle: dict[str, Any],
    object_class: str,
    object_spec: dict[str, Any],
    recipe: dict[str, Any],
    aligner_name: str,
    max_length: int,
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
        baseline = p17._teacher_forced_forward(
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
            forward=baseline,
            item_span_map=item_span_map,
            layer_index=layer_index,
        )
        view = _object_view(object_class=object_class, spec=object_spec, alignments=alignments, item_span_map=item_span_map)
        if view is None:
            continue
        metric_positions = [int(value) for value in list(view["metric_positions"])]
        baseline_margin = _margin_mean_for_positions(baseline, metric_positions, [])
        actual_remove = p19._teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=recipe,
            mode="remove",
        )
        wrong_event = p19._teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=recipe,
            mode="wrong_event",
        )
        shuffled_head = p19._teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=recipe,
            mode="shuffled_head",
        )
        shuffled_basis = p19._teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=recipe,
            mode="shuffled_basis",
        )
        generic_gain = p19._teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=recipe,
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
                "passed_item": bool(gap >= p19.H2_NECESSITY_MIN_GAP),
            }
        )
    recall = round(mean(int(bool(row["passed_item"])) for row in rows), 6) if rows else 0.0
    return {
        "row_count": len(rows),
        "item_recovery_recall": recall,
        "passed": recall >= p19.H2_ITEM_RECOVERY_MIN_RECALL,
        "rows": rows,
    }


def summarize_argument_h2_class(
    *,
    root: Path,
    class_row: dict[str, Any],
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
    selected_anchor_id = str(observability_row["selected_anchor_object_id"])
    selected_anchor = next(
        dict(row)
        for row in list(class_row.get("argument_anchor_specs") or [])
        if str(row["object_id"]) == selected_anchor_id
    )
    aligner_name = str(observability_row["selected_aligner"])
    positive_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    for target in list(class_row.get("target_rows") or []):
        panel_regime = str(target["panel_regime"])
        bundle = _bundle_from_prefix(root, str(target["phase11_prefix"]))
        if panel_regime == "planted_h2":
            positive_rows.append(
                {
                    "panel_regime": panel_regime,
                    "phase11_prefix": str(target["phase11_prefix"]),
                    **_evaluate_model_native_h2_bundle(
                        bundle=bundle,
                        object_class=str(class_row["object_class"]),
                        object_spec=selected_anchor,
                        recipe=_model_native_recipe(
                            object_spec=selected_anchor,
                            panel_regime="planted_h2",
                            base_run=str(target["phase11_prefix"]),
                        ),
                        aligner_name=aligner_name,
                        max_length=max_length,
                    ),
                }
            )
        elif panel_regime == "matched_null":
            null_rows.append(
                {
                    "panel_regime": panel_regime,
                    "phase11_prefix": str(target["phase11_prefix"]),
                    **_evaluate_model_native_h2_bundle(
                        bundle=bundle,
                        object_class=str(class_row["object_class"]),
                        object_spec=selected_anchor,
                        recipe=_model_native_recipe(
                            object_spec=selected_anchor,
                            panel_regime="planted_h2",
                            base_run=str(target["phase11_prefix"]),
                        ),
                        aligner_name=aligner_name,
                        max_length=max_length,
                    ),
                }
            )
    return {
        "object_class": str(class_row["object_class"]),
        "selected_anchor_object_id": selected_anchor_id,
        "passed": bool(positive_rows) and all(bool(row["passed"]) for row in positive_rows) and not any(
            bool(row["passed"]) for row in null_rows
        ),
        "null_passed_count": sum(1 for row in null_rows if bool(row.get("passed"))),
        "positive_rows": positive_rows,
        "null_rows": null_rows,
    }


def _evaluate_model_native_h1_run(
    *,
    bundle: dict[str, Any],
    object_class: str,
    object_spec: dict[str, Any],
    recipe: dict[str, Any],
    aligner_name: str,
    physical_family: str,
    max_length: int,
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
        baseline = p17._teacher_forced_forward(
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
            forward=baseline,
            item_span_map=item_span_map,
            layer_index=layer_index,
        )
        view = _object_view(object_class=object_class, spec=object_spec, alignments=alignments, item_span_map=item_span_map)
        if view is None:
            continue
        counterpart = p17.find_exact_role_counterpart(
            record=item,
            role=p19._counterpart_role_for_object(object_class, object_spec),
            counterpart_index=counterpart_index,
            preferred_strata=("pass", "control"),
            same_split_only=False,
        )
        if counterpart is None:
            continue
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
        actual_only = p19._teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=recipe,
            mode="reinstate_only",
        )
        actual_plus_store = p19._teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=recipe,
            mode="reinstate_plus_store",
        )
        wrong_object = p19._teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=recipe,
            mode="wrong_object",
        )
        path_only = p19._teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=recipe,
            mode="path_only",
        )
        generic_gain = p19._teacher_forced_with_mode(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            view=view,
            object_spec=object_spec,
            plant_spec=recipe,
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
            return float(summary["margin_mean"] or 0.0) - float(baseline_summary["margin_mean"] or 0.0), bool(
                summary["exact_match"]
            )

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
        "run_recovered": bool(rows)
        and rescue_gap >= p17.H1_RESCUE_MIN_GAP
        and exact_rate >= p17.H1_RESCUE_MIN_BINARY_RATE
        and (best_actual - generic_gain) >= p17.H1_GENERIC_ADVANTAGE_MIN,
        "mean_rescue_gap": rescue_gap,
        "aggregated_exact_match_rate": exact_rate,
        "generic_advantage": round(best_actual - generic_gain, 6),
        "rows": rows,
    }


def summarize_argument_h1_class(
    *,
    root: Path,
    class_row: dict[str, Any],
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
    selected_anchor_id = str(observability_row["selected_anchor_object_id"])
    selected_anchor = next(
        dict(row)
        for row in list(class_row.get("argument_anchor_specs") or [])
        if str(row["object_id"]) == selected_anchor_id
    )
    aligner_name = str(observability_row["selected_aligner"])
    positive_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    for target in list(class_row.get("target_rows") or []):
        panel_regime = str(target["panel_regime"])
        bundle = _bundle_from_prefix(root, str(target["phase11_prefix"]))
        if panel_regime == "planted_h1":
            positive_rows.append(
                {
                    "panel_regime": panel_regime,
                    "phase11_prefix": str(target["phase11_prefix"]),
                    **_evaluate_model_native_h1_run(
                        bundle=bundle,
                        object_class=str(class_row["object_class"]),
                        object_spec=selected_anchor,
                        recipe=_model_native_recipe(
                            object_spec=selected_anchor,
                            panel_regime="planted_h1",
                            base_run=str(target["phase11_prefix"]),
                        ),
                        aligner_name=aligner_name,
                        physical_family=str(target["run_name"]),
                        max_length=max_length,
                    ),
                }
            )
        elif panel_regime == "matched_null":
            null_rows.append(
                {
                    "panel_regime": panel_regime,
                    "phase11_prefix": str(target["phase11_prefix"]),
                    **_evaluate_model_native_h1_run(
                        bundle=bundle,
                        object_class=str(class_row["object_class"]),
                        object_spec=selected_anchor,
                        recipe=_model_native_recipe(
                            object_spec=selected_anchor,
                            panel_regime="planted_h1",
                            base_run=str(target["phase11_prefix"]),
                        ),
                        aligner_name=aligner_name,
                        physical_family=str(target["run_name"]),
                        max_length=max_length,
                    ),
                }
            )
    null_false_positive_rate = round(
        mean(int(bool(row.get("run_recovered"))) for row in null_rows),
        6,
    ) if null_rows else 0.0
    return {
        "object_class": str(class_row["object_class"]),
        "selected_anchor_object_id": selected_anchor_id,
        "positive_passed": bool(positive_rows) and all(bool(row["run_recovered"]) for row in positive_rows),
        "null_passed": bool(null_false_positive_rate > p19.NULL_DIRECT_H1_MAX_FALSE_POSITIVE_RATE),
        "null_false_positive_rate": null_false_positive_rate,
        "positive_rows": positive_rows,
        "null_rows": null_rows,
    }


def summarize_phase21_detector(
    *,
    anchorability_summary: dict[str, Any],
    observability_summary: dict[str, Any],
    h2_summary: dict[str, Any],
    h1_summary: dict[str, Any],
    real_regression_summary: dict[str, Any],
) -> dict[str, Any]:
    anchor_by_class = {str(row["object_class"]): row for row in list(anchorability_summary.get("class_rows") or [])}
    obs_by_class = {str(row["object_class"]): row for row in list(observability_summary.get("class_rows") or [])}
    h2_by_class = {str(row["object_class"]): row for row in list(h2_summary.get("class_rows") or [])}
    h1_by_class = {str(row["object_class"]): row for row in list(h1_summary.get("class_rows") or [])}
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
        "ontology_final_supported": validated,
        "decision": "support_ontology_final_upgrade" if validated else "retain_phase18",
        "decision_status": "final",
        "failure_reasons": failures,
        "upgraded_strongest_claim": validated,
    }


def render_phase21_status_markdown(
    *,
    manifest_summary: dict[str, Any],
    anchorability_summary: dict[str, Any],
    observability_summary: dict[str, Any],
    h2_summary: dict[str, Any],
    h1_summary: dict[str, Any],
    real_regression_summary: dict[str, Any],
    detector_summary: dict[str, Any],
) -> str:
    lines = [
        "# Phase 21 Status",
        "",
        "## Frozen Framing",
        "- `G1_binding`: bounded hypothesis-final success",
        "- `G2_distributed_geometry`: `Phase 18` family-final negative unless argument-locus detector validation upgrades the claim.",
        "- `Phase 21` executes the final class-local Stage B detector validation rather than reopening the real `G2` family.",
        "",
        "## Stage B Manifest",
        f"- `class_count = {int(manifest_summary.get('class_count') or 0)}`",
    ]
    for row in list(manifest_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `target_count = {len(list(row.get('target_rows') or []))}`, `argument_anchor_ids = {row.get('argument_anchor_ids') or []}`"
        )
    lines.extend(["", "## Anchorability"])
    for row in list(anchorability_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `passed = {bool(row.get('passed'))}`, `usable_anchor_count = {int(row.get('usable_anchor_count') or 0)}`"
        )
    lines.extend(["", "## Observability"])
    for row in list(observability_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `passed = {bool(row.get('passed'))}`, `selected_anchor = {row.get('selected_anchor_object_id')}`, `passed_aligner_count = {int(row.get('passed_aligner_count') or 0)}`, `selected_aligner = {row.get('selected_aligner')}`"
        )
    lines.extend(["", "## Planted H2"])
    for row in list(h2_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `passed = {bool(row.get('passed'))}`, `selected_anchor = {row.get('selected_anchor_object_id')}`, `null_passed_count = {int(row.get('null_passed_count') or 0)}`"
        )
    lines.extend(["", "## Planted H1"])
    for row in list(h1_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `positive_passed = {bool(row.get('positive_passed'))}`, `selected_anchor = {row.get('selected_anchor_object_id')}`, `null_false_positive_rate = {float(row.get('null_false_positive_rate') or 0.0)}`"
        )
    lines.extend(
        [
            "",
            "## Real-Family Regression",
            f"- `preserved_phase18_t_final = {bool(real_regression_summary.get('preserved_phase18_t_final'))}`",
            "",
            "## Detector Verdict",
            f"- `ontology_final_supported = {bool(detector_summary.get('ontology_final_supported'))}`",
            f"- `decision = {detector_summary.get('decision')}`",
            f"- `failure_reasons = {detector_summary.get('failure_reasons') or []}`",
        ]
    )
    return "\n".join(lines) + "\n"


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
