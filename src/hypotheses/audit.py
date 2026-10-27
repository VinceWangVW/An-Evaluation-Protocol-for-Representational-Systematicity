from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_signature_spec(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def derive_audit_signals(
    phase3_summary: dict[str, Any],
    phase4_summary: dict[str, Any],
    m3_records: list[dict[str, Any]],
    m4_records: list[dict[str, Any]],
    linkage_records: list[dict[str, Any]],
) -> dict[str, Any]:
    layer_index = int(phase4_summary["layer_index"])
    layer_summary = dict(phase3_summary["layers"][str(layer_index)])
    candidate_counts = dict(phase3_summary["candidate_counts"]["by_stratum"])
    fail_candidate_count = int(candidate_counts.get("fail") or 0)
    rescued_fail_items = sorted(
        {
            str(record["target_item_id"])
            for record in m4_records
            if bool(record.get("rescued_from_incorrect"))
        }
    )
    linkage_fail_items_with_rescue = sorted(
        {
            str(record["item_id"])
            for record in linkage_records
            if str(record.get("stratum")) == "fail" and bool(record.get("rescued_from_incorrect"))
        }
    )

    signals = {
        "m0_selected_candidates": int(phase3_summary["candidate_counts"]["selected"]),
        "m0_pass_candidates": int(candidate_counts.get("pass") or 0),
        "m0_fail_candidates": int(candidate_counts.get("fail") or 0),
        "m0_control_candidates": int(candidate_counts.get("control") or 0),
        "m0_matched_set_count": int(phase3_summary.get("matched_set_count") or 0),
        "m0_balanced_strata": all(int(candidate_counts.get(stratum) or 0) > 0 for stratum in ("pass", "fail", "control")),
        "m1_semantic_role_accuracy": _nested_get(layer_summary, "semantic_role", "loo_centroid_accuracy"),
        "m1_semantic_pass_accuracy": _nested_get(layer_summary, "semantic_role_by_stratum", "pass", "loo_centroid_accuracy"),
        "m1_semantic_fail_accuracy": _nested_get(layer_summary, "semantic_role_by_stratum", "fail", "loo_centroid_accuracy"),
        "m1_semantic_control_accuracy": _nested_get(
            layer_summary,
            "semantic_role_by_stratum",
            "control",
            "loo_centroid_accuracy",
        ),
        "m1_surface_position_accuracy": _nested_get(layer_summary, "surface_position", "loo_centroid_accuracy"),
        "m1_surface_control_gap": layer_summary.get("surface_control_gap"),
        "m1_filler_gap": _nested_get(layer_summary, "filler_identity", "within_between_gap"),
        "m1_predicate_gap": _nested_get(layer_summary, "predicate_identity", "within_between_gap"),
        "m2_count": int(_nested_get(phase4_summary, "modules", "M2_structure_preserving_interchange", "count") or 0),
        "m2_joint_success_rate": _nested_get(
            phase4_summary,
            "modules",
            "M2_structure_preserving_interchange",
            "joint_success_rate",
        ),
        "m2_role_preservation_rate": _nested_get(
            phase4_summary,
            "modules",
            "M2_structure_preserving_interchange",
            "role_preservation_rate",
        ),
        "m2_donor_transfer_rate": _nested_get(
            phase4_summary,
            "modules",
            "M2_structure_preserving_interchange",
            "donor_transfer_rate",
        ),
        "m3_count": int(_nested_get(phase4_summary, "modules", "M3_selective_necessity", "count") or 0),
        "m3_semantic_role_drop": _nested_get(
            phase4_summary,
            "modules",
            "M3_selective_necessity",
            "semantic_role_drop",
        ),
        "m3_predicate_control_drop": _nested_get(
            phase4_summary,
            "modules",
            "M3_selective_necessity",
            "predicate_control_drop",
        ),
        "m3_selective_impairment": _nested_get(
            phase4_summary,
            "modules",
            "M3_selective_necessity",
            "selective_impairment",
        ),
        "m3_pass_selective_impairment": _mean_by_stratum(m3_records, "pass", "selective_impairment"),
        "m3_fail_selective_impairment": _mean_by_stratum(m3_records, "fail", "selective_impairment"),
        "m3_control_selective_impairment": _mean_by_stratum(m3_records, "control", "selective_impairment"),
        "m4_count": int(_nested_get(phase4_summary, "modules", "M4_failure_rescue", "count") or 0),
        "m4_rescue_rate": _nested_get(
            phase4_summary,
            "modules",
            "M4_failure_rescue",
            "rescued_from_incorrect_rate",
        ),
        "m4_rescued_fail_item_count": len(rescued_fail_items),
        "m5_count": int(_nested_get(phase4_summary, "modules", "M5_item_level_linkage", "count") or 0),
        "m5_fail_items_with_rescue_signal": int(
            _nested_get(phase4_summary, "modules", "M5_item_level_linkage", "fail_items_with_rescue_signal") or 0
        ),
        "m5_fail_item_rescue_coverage": round(len(linkage_fail_items_with_rescue) / fail_candidate_count, 6)
        if fail_candidate_count
        else None,
    }
    signals["audit_ready"] = bool(
        signals["m0_balanced_strata"]
        and signals["m0_matched_set_count"] >= 1
        and signals["m2_count"] >= 1
        and signals["m3_count"] >= 1
        and signals["m4_count"] >= 1
        and signals["m5_count"] >= 1
        and signals["m1_semantic_role_accuracy"] is not None
    )
    return signals


def derive_generation_audit_signals(
    behavior_summary: dict[str, Any],
    discovery_summary: dict[str, Any],
    protocol_summary: dict[str, Any],
    linkage_records: list[dict[str, Any]],
) -> dict[str, Any]:
    layer_index = int(protocol_summary["layer_index"])
    discovery_layer = dict(discovery_summary["layers"][str(layer_index)])
    behavior_eval = dict(behavior_summary["evaluation"])
    by_split = dict(behavior_eval.get("by_split") or {})
    control_exact_values = [
        float(split_summary["exact_match"])
        for split_name, split_summary in by_split.items()
        if split_name in {"dev", "test"}
    ]
    control_clause_values = [
        float(split_summary["clause_f1"])
        for split_name, split_summary in by_split.items()
        if split_name in {"dev", "test"}
    ]
    candidate_counts = dict(discovery_summary["candidate_counts"]["by_stratum"])
    fail_candidates = int(candidate_counts.get("fail") or 0)
    fail_items_with_advantage = sorted(
        {
            str(record["item_id"])
            for record in linkage_records
            if str(record.get("stratum")) == "fail"
            and float(record.get("rescue_vs_strongest_control_advantage_clause_f1",
                                 record.get("rescue_advantage_clause_f1")) or 0.0) > 0.0
        }
    )
    behavior_anchor_ready = bool(
        by_split.get("gen")
        and control_exact_values
        and int(discovery_summary.get("matched_set_count") or 0) >= 1
        and all(int(candidate_counts.get(stratum) or 0) > 0 for stratum in ("pass", "fail", "control"))
    )
    signals = {
        "b0_overall_exact_match": _nested_get(behavior_eval, "overall", "exact_match"),
        "b0_overall_clause_f1": _nested_get(behavior_eval, "overall", "clause_f1"),
        "b0_gen_exact_match": _nested_get(by_split, "gen", "exact_match"),
        "b0_gen_clause_f1": _nested_get(by_split, "gen", "clause_f1"),
        "b0_control_exact_match": round(mean(control_exact_values), 6) if control_exact_values else None,
        "b0_control_clause_f1": round(mean(control_clause_values), 6) if control_clause_values else None,
        "b0_control_minus_gen_exact_gap": round(
            float(mean(control_exact_values)) - float(_nested_get(by_split, "gen", "exact_match")),
            6,
        )
        if control_exact_values and _nested_get(by_split, "gen", "exact_match") is not None
        else None,
        "b0_control_minus_gen_clause_gap": round(
            float(mean(control_clause_values)) - float(_nested_get(by_split, "gen", "clause_f1")),
            6,
        )
        if control_clause_values and _nested_get(by_split, "gen", "clause_f1") is not None
        else None,
        "m0_selected_candidates": int(discovery_summary["candidate_counts"]["selected"]),
        "m0_pass_candidates": int(candidate_counts.get("pass") or 0),
        "m0_fail_candidates": int(candidate_counts.get("fail") or 0),
        "m0_control_candidates": int(candidate_counts.get("control") or 0),
        "m0_matched_set_count": int(discovery_summary.get("matched_set_count") or 0),
        "m0_balanced_strata": all(int(candidate_counts.get(stratum) or 0) > 0 for stratum in ("pass", "fail", "control")),
        "m1_semantic_role_accuracy": _nested_get(discovery_layer, "semantic_role", "loo_centroid_accuracy"),
        "m1_semantic_pass_accuracy": _nested_get(discovery_layer, "semantic_role_by_stratum", "pass", "loo_centroid_accuracy"),
        "m1_semantic_fail_accuracy": _nested_get(discovery_layer, "semantic_role_by_stratum", "fail", "loo_centroid_accuracy"),
        "m1_surface_position_accuracy": _nested_get(discovery_layer, "surface_position", "loo_centroid_accuracy"),
        "m1_surface_control_gap": discovery_layer.get("surface_control_gap"),
        "m1_surface_control_gap_observed": discovery_layer.get("surface_control_gap") is not None,
        "m1_filler_gap": _nested_get(discovery_layer, "filler_identity", "within_between_gap"),
        "m1_predicate_gap": _nested_get(discovery_layer, "predicate_identity", "within_between_gap"),
        "m1_predicate_gap_observed": _nested_get(discovery_layer, "predicate_identity", "within_between_gap") is not None,
        "m2_count": int(_nested_get(protocol_summary, "modules", "M2_generation_interchange", "count") or 0),
        "m2_actual_mean": _nested_get(protocol_summary, "modules", "M2_generation_interchange", "actual_mean"),
        "m2_wrong_role_mean": _nested_get(protocol_summary, "modules", "M2_generation_interchange", "wrong_role_mean"),
        "m2_non_role_mean": _nested_get(protocol_summary, "modules", "M2_generation_interchange", "non_role_mean"),
        "m2_shuffled_mean": _nested_get(protocol_summary, "modules", "M2_generation_interchange", "shuffled_mean"),
        "m2_strongest_control_mean": _nested_get(protocol_summary, "modules", "M2_generation_interchange", "strongest_control_mean"),
        "m2_actual_binary_rate": _nested_get(protocol_summary, "modules", "M2_generation_interchange", "actual_binary_rate"),
        "m2_strongest_control_binary_rate": _nested_get(protocol_summary, "modules", "M2_generation_interchange", "strongest_control_binary_rate"),
        "m2_actual_vs_wrong_role_advantage_mean": _nested_get(
            protocol_summary,
            "modules",
            "M2_generation_interchange",
            "actual_vs_wrong_role_advantage_mean",
        ),
        "m2_actual_vs_non_role_advantage_mean": _nested_get(
            protocol_summary,
            "modules",
            "M2_generation_interchange",
            "actual_vs_non_role_advantage_mean",
        ),
        "m2_actual_vs_shuffled_advantage_mean": _nested_get(
            protocol_summary,
            "modules",
            "M2_generation_interchange",
            "actual_vs_shuffled_advantage_mean",
        ),
        "m2_actual_vs_strongest_control_advantage_mean": _nested_get(
            protocol_summary,
            "modules",
            "M2_generation_interchange",
            "actual_vs_strongest_control_advantage_mean",
        ),
        "m2_actual_vs_wrong_role_permutation_p_value": _nested_get(
            protocol_summary,
            "modules",
            "M2_generation_interchange",
            "actual_vs_wrong_role_permutation_p_value",
        ),
        "m2_actual_vs_non_role_permutation_p_value": _nested_get(
            protocol_summary,
            "modules",
            "M2_generation_interchange",
            "actual_vs_non_role_permutation_p_value",
        ),
        "m2_actual_vs_shuffled_permutation_p_value": _nested_get(
            protocol_summary,
            "modules",
            "M2_generation_interchange",
            "actual_vs_shuffled_permutation_p_value",
        ),
        "m2_actual_vs_strongest_control_permutation_p_value": _nested_get(
            protocol_summary,
            "modules",
            "M2_generation_interchange",
            "actual_vs_strongest_control_permutation_p_value",
        ),
        "m3_count": int(_nested_get(protocol_summary, "modules", "M3_generation_necessity", "count") or 0),
        "m3_role_ablation_clause_f1_drop": _nested_get(protocol_summary, "modules", "M3_generation_necessity", "role_ablation_clause_f1_drop"),
        "m3_wrong_role_clause_f1_drop": _nested_get(protocol_summary, "modules", "M3_generation_necessity", "wrong_role_clause_f1_drop"),
        "m3_non_role_clause_f1_drop": _nested_get(protocol_summary, "modules", "M3_generation_necessity", "non_role_clause_f1_drop"),
        "m3_strongest_control_clause_f1_drop": _nested_get(
            protocol_summary,
            "modules",
            "M3_generation_necessity",
            "strongest_control_clause_f1_drop",
        ),
        "m3_role_vs_wrong_role_gap_clause_f1": _nested_get(
            protocol_summary,
            "modules",
            "M3_generation_necessity",
            "role_vs_wrong_role_gap_clause_f1",
        ),
        "m3_role_vs_non_role_gap_clause_f1": _nested_get(
            protocol_summary,
            "modules",
            "M3_generation_necessity",
            "role_vs_non_role_gap_clause_f1",
        ),
        "m3_role_vs_strongest_control_gap_clause_f1": _nested_get(
            protocol_summary,
            "modules",
            "M3_generation_necessity",
            "role_vs_strongest_control_gap_clause_f1",
        ),
        "m3_role_vs_wrong_role_permutation_p_value": _nested_get(
            protocol_summary,
            "modules",
            "M3_generation_necessity",
            "role_vs_wrong_role_permutation_p_value",
        ),
        "m3_role_vs_non_role_permutation_p_value": _nested_get(
            protocol_summary,
            "modules",
            "M3_generation_necessity",
            "role_vs_non_role_permutation_p_value",
        ),
        "m3_role_vs_strongest_control_permutation_p_value": _nested_get(
            protocol_summary,
            "modules",
            "M3_generation_necessity",
            "role_vs_strongest_control_permutation_p_value",
        ),
        "m4_count": int(_nested_get(protocol_summary, "modules", "M4_generation_rescue", "count") or 0),
        "m4_actual_mean": _nested_get(protocol_summary, "modules", "M4_generation_rescue", "actual_mean"),
        "m4_wrong_role_mean": _nested_get(protocol_summary, "modules", "M4_generation_rescue", "wrong_role_mean"),
        "m4_non_role_mean": _nested_get(protocol_summary, "modules", "M4_generation_rescue", "non_role_mean"),
        "m4_shuffled_mean": _nested_get(protocol_summary, "modules", "M4_generation_rescue", "shuffled_mean"),
        "m4_alignment_only_mean": _nested_get(protocol_summary, "modules", "M4_generation_rescue", "alignment_only_mean"),
        "m4_shuffled_alignment_mean": _nested_get(protocol_summary, "modules", "M4_generation_rescue", "shuffled_alignment_mean"),
        "m4_wrong_path_mean": _nested_get(protocol_summary, "modules", "M4_generation_rescue", "wrong_path_mean"),
        "m4_generic_decoder_gain_mean": _nested_get(
            protocol_summary,
            "modules",
            "M4_generation_rescue",
            "generic_decoder_gain_mean",
        ),
        "m4_path_only_mean": _nested_get(protocol_summary, "modules", "M4_generation_rescue", "path_only_mean"),
        "m4_strongest_control_mean": _nested_get(protocol_summary, "modules", "M4_generation_rescue", "strongest_control_mean"),
        "m4_actual_binary_rate": _nested_get(protocol_summary, "modules", "M4_generation_rescue", "actual_binary_rate"),
        "m4_strongest_control_binary_rate": _nested_get(protocol_summary, "modules", "M4_generation_rescue", "strongest_control_binary_rate"),
        "m4_actual_vs_alignment_only_advantage_mean": _nested_get(
            protocol_summary,
            "modules",
            "M4_generation_rescue",
            "actual_vs_alignment_only_advantage_mean",
        ),
        "m4_actual_vs_wrong_role_advantage_mean": _nested_get(
            protocol_summary,
            "modules",
            "M4_generation_rescue",
            "actual_vs_wrong_role_advantage_mean",
        ),
        "m4_actual_vs_non_role_advantage_mean": _nested_get(
            protocol_summary,
            "modules",
            "M4_generation_rescue",
            "actual_vs_non_role_advantage_mean",
        ),
        "m4_actual_vs_shuffled_advantage_mean": _nested_get(
            protocol_summary,
            "modules",
            "M4_generation_rescue",
            "actual_vs_shuffled_advantage_mean",
        ),
        "m4_actual_vs_shuffled_alignment_advantage_mean": _nested_get(
            protocol_summary,
            "modules",
            "M4_generation_rescue",
            "actual_vs_shuffled_alignment_advantage_mean",
        ),
        "m4_actual_vs_wrong_path_advantage_mean": _nested_get(
            protocol_summary,
            "modules",
            "M4_generation_rescue",
            "actual_vs_wrong_path_advantage_mean",
        ),
        "m4_actual_vs_generic_decoder_gain_advantage_mean": _nested_get(
            protocol_summary,
            "modules",
            "M4_generation_rescue",
            "actual_vs_generic_decoder_gain_advantage_mean",
        ),
        "m4_actual_vs_path_only_advantage_mean": _nested_get(
            protocol_summary,
            "modules",
            "M4_generation_rescue",
            "actual_vs_path_only_advantage_mean",
        ),
        "m4_actual_vs_strongest_control_advantage_mean": _nested_get(
            protocol_summary,
            "modules",
            "M4_generation_rescue",
            "actual_vs_strongest_control_advantage_mean",
        ),
        "m4_actual_vs_wrong_role_permutation_p_value": _nested_get(
            protocol_summary,
            "modules",
            "M4_generation_rescue",
            "actual_vs_wrong_role_permutation_p_value",
        ),
        "m4_actual_vs_non_role_permutation_p_value": _nested_get(
            protocol_summary,
            "modules",
            "M4_generation_rescue",
            "actual_vs_non_role_permutation_p_value",
        ),
        "m4_actual_vs_shuffled_permutation_p_value": _nested_get(
            protocol_summary,
            "modules",
            "M4_generation_rescue",
            "actual_vs_shuffled_permutation_p_value",
        ),
        "m4_actual_vs_strongest_control_permutation_p_value": _nested_get(
            protocol_summary,
            "modules",
            "M4_generation_rescue",
            "actual_vs_strongest_control_permutation_p_value",
        ),
        "m5_count": int(_nested_get(protocol_summary, "modules", "M5_item_level_linkage", "count") or 0),
        "m5_fail_items_with_rescue_advantage": int(
            _nested_get(protocol_summary, "modules", "M5_item_level_linkage", "fail_items_with_rescue_advantage") or 0
        ),
        "m5_fail_item_rescue_advantage_coverage": round(len(fail_items_with_advantage) / fail_candidates, 6)
        if fail_candidates
        else None,
        "behavior_anchor_ready": behavior_anchor_ready,
    }
    signals["m2_advantage_mean"] = signals["m2_actual_vs_strongest_control_advantage_mean"]
    signals["m2_permutation_p_value"] = signals["m2_actual_vs_strongest_control_permutation_p_value"]
    signals["m3_selective_impairment_clause_f1"] = signals["m3_role_vs_strongest_control_gap_clause_f1"]
    signals["m3_permutation_p_value"] = signals["m3_role_vs_strongest_control_permutation_p_value"]
    signals["m4_advantage_mean"] = signals["m4_actual_vs_strongest_control_advantage_mean"]
    signals["m4_permutation_p_value"] = signals["m4_actual_vs_strongest_control_permutation_p_value"]
    signals["audit_ready"] = bool(
        signals["behavior_anchor_ready"]
        and signals["m2_count"] >= 1
        and signals["m3_count"] >= 1
        and signals["m4_count"] >= 1
        and signals["m1_semantic_role_accuracy"] is not None
        and signals["m1_predicate_gap_observed"]
        and signals["m1_surface_control_gap_observed"]
    )
    return signals


def derive_phase11_binding_audit_signals(
    behavior_summary: dict[str, Any],
    discovery_summary: dict[str, Any],
    protocol_summary: dict[str, Any],
    linkage_records: list[dict[str, Any]],
) -> dict[str, Any]:
    signals = derive_generation_audit_signals(
        behavior_summary=behavior_summary,
        discovery_summary=discovery_summary,
        protocol_summary=protocol_summary,
        linkage_records=linkage_records,
    )
    pass_candidates = int(signals.get("m0_pass_candidates") or 0)
    fail_candidates = int(signals.get("m0_fail_candidates") or 0)
    control_candidates = int(signals.get("m0_control_candidates") or 0)
    available_target_states: list[str] = []
    if pass_candidates:
        available_target_states.append("pass")
    if fail_candidates:
        available_target_states.append("fail")

    signals["state_anchor_ready"] = bool(control_candidates > 0 and available_target_states)
    signals["available_target_strata"] = available_target_states
    signals["m2_observed"] = bool(int(signals.get("m2_count") or 0) > 0)
    signals["m3_observed"] = bool(int(signals.get("m3_count") or 0) > 0)
    signals["m4_observed"] = bool(int(signals.get("m4_count") or 0) > 0)
    signals["binding_dev_audit_ready"] = bool(
        signals["state_anchor_ready"]
        and signals["m3_observed"]
        and (signals["m2_observed"] or signals["m4_observed"])
        and signals["m1_semantic_role_accuracy"] is not None
        and signals["m1_predicate_gap_observed"]
        and signals["m1_surface_control_gap_observed"]
    )
    signals["audit_ready"] = signals["binding_dev_audit_ready"]
    return signals


def derive_phase11_geometry_audit_signals(
    behavior_summary: dict[str, Any],
    discovery_summary: dict[str, Any],
    protocol_summary: dict[str, Any],
    linkage_records: list[dict[str, Any]],
) -> dict[str, Any]:
    signals = derive_generation_audit_signals(
        behavior_summary=behavior_summary,
        discovery_summary=discovery_summary,
        protocol_summary=protocol_summary,
        linkage_records=linkage_records,
    )
    pass_candidates = int(signals.get("m0_pass_candidates") or 0)
    fail_candidates = int(signals.get("m0_fail_candidates") or 0)
    control_candidates = int(signals.get("m0_control_candidates") or 0)
    available_target_states: list[str] = []
    if pass_candidates:
        available_target_states.append("pass")
    if fail_candidates:
        available_target_states.append("fail")

    signals["state_anchor_ready"] = bool(control_candidates > 0 and available_target_states)
    signals["available_target_strata"] = available_target_states
    signals["m2_observed"] = bool(int(signals.get("m2_count") or 0) > 0)
    signals["m3_observed"] = bool(int(signals.get("m3_count") or 0) > 0)
    signals["m4_observed"] = bool(int(signals.get("m4_count") or 0) > 0)
    signals["geometry_signal_ready"] = bool(
        signals["m1_semantic_role_accuracy"] is not None
        and signals["m1_filler_gap"] is not None
        and signals["m1_predicate_gap_observed"]
        and signals["m1_surface_control_gap_observed"]
    )
    signals["geometry_dev_audit_ready"] = bool(
        signals["state_anchor_ready"]
        and signals["geometry_signal_ready"]
        and signals["m2_observed"]
        and signals["m3_observed"]
    )
    signals["audit_ready"] = signals["geometry_dev_audit_ready"]
    return signals


def run_hypothesis_audit(spec: dict[str, Any], signals: dict[str, Any]) -> dict[str, Any]:
    preconditions = evaluate_rule({"all": list(spec.get("audit_preconditions") or [])}, signals)
    hypotheses = [evaluate_hypothesis(hypothesis, signals) for hypothesis in spec.get("hypotheses") or []]
    return {
        "registry_version": spec.get("version"),
        "audit_ready": bool(signals.get("audit_ready")) and preconditions["matched"],
        "preconditions": preconditions,
        "hypotheses": hypotheses,
    }


def build_signal_table(spec: dict[str, Any], signals: dict[str, Any]) -> list[dict[str, Any]]:
    definitions = dict(spec.get("signal_definitions") or {})
    records: list[dict[str, Any]] = []
    for signal_name, value in sorted(signals.items()):
        metadata = dict(definitions.get(signal_name) or {})
        records.append(
            {
                "signal": signal_name,
                "module": metadata.get("module"),
                "description": metadata.get("description"),
                "value": value,
            }
        )
    return records


def build_state_audit_table(audit: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for hypothesis in audit.get("hypotheses") or []:
        for state in hypothesis.get("states") or []:
            records.append(
                {
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "hypothesis_label": hypothesis["hypothesis_label"],
                    "selected_state": hypothesis["selected_state"],
                    "state_id": state["state_id"],
                    "matched": state["matched"],
                    "score": state["score"],
                    "satisfied_conditions": state["satisfied_conditions"],
                    "condition_count": state["condition_count"],
                    "is_selected": state["state_id"] == hypothesis["selected_state"],
                }
            )
    return records


def build_support_matrix(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hypothesis in audit.get("hypotheses") or []:
        row = {
            "hypothesis_id": hypothesis["hypothesis_id"],
            "hypothesis_label": hypothesis["hypothesis_label"],
            "selected_state": hypothesis["selected_state"],
            "selected_score": hypothesis["selected_score"],
            "audit_ready": audit["audit_ready"],
        }
        for state in hypothesis.get("states") or []:
            row[state["state_id"]] = 1 if state["state_id"] == hypothesis["selected_state"] else 0
            row[f"{state['state_id']}_matched"] = state["matched"]
            row[f"{state['state_id']}_score"] = state["score"]
        rows.append(row)
    return rows


def evaluate_hypothesis(hypothesis: dict[str, Any], signals: dict[str, Any]) -> dict[str, Any]:
    state_results: list[dict[str, Any]] = []
    selected_state_id: str | None = None
    selected_score: float | None = None
    for state in hypothesis.get("states") or []:
        if bool(state.get("default")):
            result = {
                "matched": selected_state_id is None,
                "condition_count": 0,
                "satisfied_conditions": 0,
                "score": 1.0 if selected_state_id is None else 0.0,
                "details": [],
                "support_score": 1.0 if selected_state_id is None else 0.0,
                "falsifier_score": 0.0,
                "falsifier_triggered": False,
            }
        else:
            support_result = evaluate_rule(state.get("support") or state.get("rule") or {}, signals)
            falsifier_result = evaluate_rule(state.get("falsifiers") or {}, signals)
            falsifier_triggered = falsifier_result["matched"] if state.get("falsifiers") else False
            result = {
                "matched": support_result["matched"] and not falsifier_triggered,
                "condition_count": support_result["condition_count"] + falsifier_result["condition_count"],
                "satisfied_conditions": support_result["satisfied_conditions"] + (
                    falsifier_result["condition_count"] - falsifier_result["satisfied_conditions"]
                ),
                "score": round(
                    max(0.0, float(support_result["score"]) - (float(falsifier_result["score"]) if falsifier_triggered else 0.0)),
                    6,
                ),
                "details": [
                    *support_result["details"],
                    *[
                        {
                            **detail,
                            "group": "falsifier",
                            "satisfied": not detail["matched"],
                        }
                        for detail in falsifier_result["details"]
                    ],
                ],
                "support_score": support_result["score"],
                "falsifier_score": falsifier_result["score"],
                "falsifier_triggered": falsifier_triggered,
            }
        state_result = {
            "state_id": state["id"],
            "meaning": state.get("meaning"),
            **result,
        }
        state_results.append(state_result)
        if state_result["matched"]:
            if selected_state_id is None or float(state_result["score"]) > float(selected_score or -1.0):
                selected_state_id = state["id"]
                selected_score = state_result["score"]

    if selected_state_id is None:
        selected_state_id = "H0"
        selected_score = 0.0

    return {
        "hypothesis_id": hypothesis["id"],
        "hypothesis_label": hypothesis.get("label"),
        "description": hypothesis.get("description"),
        "selected_state": selected_state_id,
        "selected_score": selected_score,
        "states": state_results,
    }


def evaluate_rule(rule: dict[str, Any], signals: dict[str, Any]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    matched = True
    for group_name in ("all", "any", "none"):
        group_conditions = list(rule.get(group_name) or [])
        if not group_conditions:
            continue
        group_results = [evaluate_condition(condition, signals) for condition in group_conditions]
        if group_name == "all":
            group_matched = all(result["matched"] for result in group_results)
        elif group_name == "any":
            group_matched = any(result["matched"] for result in group_results)
        else:
            group_matched = not any(result["matched"] for result in group_results)
        matched = matched and group_matched
        for result in group_results:
            details.append(
                {
                    "group": group_name,
                    "signal": result["signal"],
                    "description": result["description"],
                    "op": result["op"],
                    "expected": result["expected"],
                    "actual": result["actual"],
                    "matched": result["matched"],
                    "satisfied": result["matched"] if group_name != "none" else not result["matched"],
                }
            )

    condition_count = len(details)
    satisfied_conditions = sum(1 for detail in details if detail["satisfied"])
    score = round(satisfied_conditions / condition_count, 6) if condition_count else 1.0
    return {
        "matched": matched,
        "condition_count": condition_count,
        "satisfied_conditions": satisfied_conditions,
        "score": score,
        "details": details,
    }


def evaluate_condition(condition: dict[str, Any], signals: dict[str, Any]) -> dict[str, Any]:
    signal_name = str(condition["signal"])
    actual = signals.get(signal_name)
    op = str(condition["op"])
    expected = condition.get("value")
    matched = compare_values(actual, op, expected)
    return {
        "signal": signal_name,
        "description": condition.get("description"),
        "op": op,
        "expected": expected,
        "actual": actual,
        "matched": matched,
    }


def compare_values(actual: Any, op: str, expected: Any) -> bool:
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if actual is None or expected is None:
        return False
    if op == ">=":
        return actual >= expected
    if op == ">":
        return actual > expected
    if op == "<=":
        return actual <= expected
    if op == "<":
        return actual < expected
    raise ValueError(f"Unsupported comparison operator: {op}")


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _mean_by_stratum(records: list[dict[str, Any]], stratum: str, key: str) -> float | None:
    values = [
        float(record[key])
        for record in records
        if str(record.get("stratum")) == stratum and record.get(key) is not None
    ]
    if not values:
        return None
    return round(mean(values), 6)
