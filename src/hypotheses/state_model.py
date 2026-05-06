from __future__ import annotations

import copy
import math
from statistics import mean
from typing import Any


STATE_IDS = ("H0", "H1", "H2")
DEFAULT_MIN_POSTERIOR_FOR_DECISION = 0.38

FAMILY_PROFILES: dict[str, dict[str, Any]] = {
    "G1_binding": {
        "version_prefix": "v7_binding_state",
        "presence_features": (
            {
                "feature": "m1_semantic_role_accuracy",
                "direction": "high",
                "boundary_strategy": "h0_vs_present",
                "fallback_boundary": 0.85,
                "scale_floor": 0.04,
            },
            {
                "feature": "m1_surface_control_gap",
                "direction": "high",
                "boundary_strategy": "h0_vs_present",
                "fallback_boundary": 0.02,
                "scale_floor": 0.02,
                "optional": True,
            },
        ),
        "interchange_features": (
            {
                "feature": "m2_advantage_mean",
                "direction": "high",
                "boundary_strategy": "h0_vs_present",
                "fallback_boundary": 0.05,
                "scale_floor": 0.05,
            },
            {
                "feature": "m2_permutation_p_value",
                "direction": "low",
                "boundary_strategy": "fixed",
                "fallback_boundary": 0.05,
                "scale_floor": 0.02,
            },
        ),
        "necessity_features": (
            {
                "feature": "m3_selective_impairment_clause_f1",
                "direction": "high",
                "boundary_strategy": "fixed",
                "fallback_boundary": 0.05,
                "scale_floor": 0.05,
                "optional": True,
            },
            {
                "feature": "m3_permutation_p_value",
                "direction": "low",
                "boundary_strategy": "fixed",
                "fallback_boundary": 0.05,
                "scale_floor": 0.02,
                "optional": True,
            },
        ),
        "rescue_features": (
            {
                "feature": "m4_advantage_mean",
                "direction": "high",
                "boundary_strategy": "h1_vs_nonh1",
                "fallback_boundary": 0.05,
                "scale_floor": 0.05,
                "optional": True,
            },
            {
                "feature": "m4_permutation_p_value",
                "direction": "low",
                "boundary_strategy": "fixed",
                "fallback_boundary": 0.05,
                "scale_floor": 0.02,
                "optional": True,
            },
            {
                "feature": "m5_fail_item_rescue_advantage_coverage",
                "direction": "high",
                "boundary_strategy": "h1_vs_nonh1",
                "fallback_boundary": 0.05,
                "scale_floor": 0.05,
                "optional": True,
            },
        ),
        "realism_features": (
            {
                "feature": "realism_pass_rate",
                "direction": "high",
                "boundary_strategy": "fixed",
                "fallback_boundary": 0.6,
                "scale_floor": 0.1,
                "optional": True,
            },
            {
                "feature": "realism_cosine_margin_mean",
                "direction": "high",
                "boundary_strategy": "fixed",
                "fallback_boundary": 0.05,
                "scale_floor": 0.05,
                "optional": True,
            },
            {
                "feature": "realism_bank_distance_margin_mean",
                "direction": "high",
                "boundary_strategy": "fixed",
                "fallback_boundary": 0.05,
                "scale_floor": 0.05,
                "optional": True,
            },
        ),
        "group_weights": {
            "presence": 1.1,
            "interchange": 1.0,
            "necessity": 1.05,
            "rescue": 1.0,
            "realism": 0.85,
            "behavior_prior": 0.15,
        },
        "minimum_groups_for_decision": 2,
        "minimum_interchange_score_for_h2": 0.55,
        "minimum_necessity_score_for_h2": 0.55,
        "minimum_interchange_score_for_h1": 0.25,
        "maximum_necessity_score_for_h1": 0.45,
        "minimum_presence_score_for_nonh0": 0.65,
        "minimum_realism_score_for_h2": 0.45,
        "minimum_realism_score_for_h1": 0.35,
        "minimum_rescue_score_for_h1": 0.35,
        "allow_h1_without_rescue": True,
        "h1_requires_rescue": False,
        "h1_requires_weak_necessity": False,
        "target_groups": None,
    },
    "G2_distributed_geometry": {
        "version_prefix": "v7_geometry_state",
        "presence_features": (
            {
                "feature": "m1_semantic_role_accuracy",
                "direction": "high",
                "boundary_strategy": "h0_vs_present",
                "fallback_boundary": 0.9,
                "scale_floor": 0.04,
            },
            {
                "feature": "m1_filler_gap",
                "direction": "high",
                "boundary_strategy": "h0_vs_present",
                "fallback_boundary": 0.33,
                "scale_floor": 0.03,
            },
            {
                "feature": "m1_surface_control_gap",
                "direction": "high",
                "boundary_strategy": "h0_vs_present",
                "fallback_boundary": -0.02,
                "scale_floor": 0.02,
                "optional": True,
            },
        ),
        "interchange_features": (
            {
                "feature": "m2_advantage_mean",
                "direction": "high",
                "boundary_strategy": "h0_vs_present",
                "fallback_boundary": 0.05,
                "scale_floor": 0.05,
            },
            {
                "feature": "m2_permutation_p_value",
                "direction": "low",
                "boundary_strategy": "fixed",
                "fallback_boundary": 0.05,
                "scale_floor": 0.02,
            },
        ),
        "necessity_features": (
            {
                "feature": "m3_selective_impairment_clause_f1",
                "direction": "high",
                "boundary_strategy": "fixed",
                "fallback_boundary": 0.05,
                "scale_floor": 0.05,
                "optional": True,
            },
            {
                "feature": "m3_permutation_p_value",
                "direction": "low",
                "boundary_strategy": "fixed",
                "fallback_boundary": 0.05,
                "scale_floor": 0.02,
                "optional": True,
            },
        ),
        "rescue_features": (
            {
                "feature": "m4_advantage_mean",
                "direction": "high",
                "boundary_strategy": "h1_vs_nonh1",
                "fallback_boundary": 0.05,
                "scale_floor": 0.05,
                "optional": True,
            },
            {
                "feature": "m4_permutation_p_value",
                "direction": "low",
                "boundary_strategy": "fixed",
                "fallback_boundary": 0.05,
                "scale_floor": 0.02,
                "optional": True,
            },
            {
                "feature": "m5_fail_item_rescue_advantage_coverage",
                "direction": "high",
                "boundary_strategy": "h1_vs_nonh1",
                "fallback_boundary": 0.05,
                "scale_floor": 0.05,
                "optional": True,
            },
        ),
        "realism_features": (
            {
                "feature": "realism_pass_rate",
                "direction": "high",
                "boundary_strategy": "fixed",
                "fallback_boundary": 0.6,
                "scale_floor": 0.1,
                "optional": True,
            },
            {
                "feature": "realism_cosine_margin_mean",
                "direction": "high",
                "boundary_strategy": "fixed",
                "fallback_boundary": 0.05,
                "scale_floor": 0.05,
                "optional": True,
            },
            {
                "feature": "realism_bank_distance_margin_mean",
                "direction": "high",
                "boundary_strategy": "fixed",
                "fallback_boundary": 0.05,
                "scale_floor": 0.05,
                "optional": True,
            },
        ),
        "group_weights": {
            "presence": 1.1,
            "interchange": 0.95,
            "necessity": 1.1,
            "rescue": 1.35,
            "realism": 1.0,
            "behavior_prior": 0.1,
        },
        "minimum_groups_for_decision": 2,
        "minimum_interchange_score_for_h2": 0.55,
        "minimum_necessity_score_for_h2": 0.55,
        "minimum_interchange_score_for_h1": 0.18,
        "maximum_necessity_score_for_h1": 0.42,
        "minimum_presence_score_for_nonh0": 0.7,
        "minimum_realism_score_for_h2": 0.45,
        "minimum_realism_score_for_h1": 0.4,
        "minimum_rescue_score_for_h1": 0.45,
        "allow_h1_without_rescue": False,
        "h1_requires_rescue": True,
        "h1_requires_weak_necessity": True,
        "target_groups": None,
    },
}


def fit_state_adjudicator(
    rows: list[dict[str, Any]],
    *,
    family: str,
    version: str,
    notes: list[str] | None = None,
    min_posterior_for_decision: float = DEFAULT_MIN_POSTERIOR_FOR_DECISION,
) -> dict[str, Any]:
    profile = _profile_for_family(family)
    thresholds = _build_thresholds(rows, profile)
    behavior_prior = _build_behavior_prior(rows)
    interchange_descriptors = tuple(profile.get("interchange_features") or profile.get("use_features") or ())
    necessity_descriptors = tuple(profile.get("necessity_features") or ())
    feature_sets = {
        "presence": _materialize_feature_set(tuple(profile["presence_features"]), thresholds),
        "interchange": _materialize_feature_set(interchange_descriptors, thresholds),
        "necessity": _materialize_feature_set(necessity_descriptors, thresholds),
        "rescue": _materialize_feature_set(tuple(profile["rescue_features"]), thresholds),
        "realism": _materialize_feature_set(tuple(profile["realism_features"]), thresholds),
    }
    feature_sets["use"] = _merge_feature_sets(feature_sets["interchange"], feature_sets["necessity"])
    provisional_spec = {
        "feature_sets": feature_sets,
        "behavior_prior": behavior_prior,
        "group_weights": dict(profile["group_weights"]),
    }
    group_targets = _build_group_targets(rows, provisional_spec)
    score_thresholds = _build_score_thresholds(rows, provisional_spec)
    group_weights = dict(profile["group_weights"])
    behavior_share = round(
        float(group_weights.get("behavior_prior") or 0.0)
        / max(sum(float(weight) for weight in group_weights.values()), 1e-9),
        6,
    )
    draft_spec = {
        "feature_sets": feature_sets,
        "behavior_prior": behavior_prior,
        "group_weights": group_weights,
        "group_targets": group_targets,
        "minimum_groups_for_decision": int(profile["minimum_groups_for_decision"]),
        "minimum_interchange_score_for_h2": float(
            score_thresholds.get("minimum_interchange_score_for_h2")
            or profile["minimum_interchange_score_for_h2"]
        ),
        "minimum_necessity_score_for_h2": float(
            score_thresholds.get("minimum_necessity_score_for_h2")
            or profile["minimum_necessity_score_for_h2"]
        ),
        "minimum_interchange_score_for_h1": float(
            score_thresholds.get("minimum_interchange_score_for_h1")
            or profile["minimum_interchange_score_for_h1"]
        ),
        "maximum_necessity_score_for_h1": float(
            score_thresholds.get("maximum_necessity_score_for_h1")
            or profile["maximum_necessity_score_for_h1"]
        ),
        "minimum_presence_score_for_nonh0": float(
            score_thresholds.get("minimum_presence_score_for_nonh0") or profile["minimum_presence_score_for_nonh0"]
        ),
        "minimum_realism_score_for_h2": float(
            score_thresholds.get("minimum_realism_score_for_h2") or profile["minimum_realism_score_for_h2"]
        ),
        "minimum_realism_score_for_h1": float(
            score_thresholds.get("minimum_realism_score_for_h1") or profile["minimum_realism_score_for_h1"]
        ),
        "minimum_rescue_score_for_h1": float(
            score_thresholds.get("minimum_rescue_score_for_h1") or profile["minimum_rescue_score_for_h1"]
        ),
        "allow_h1_without_rescue": bool(profile["allow_h1_without_rescue"]),
        "h1_requires_rescue": bool(profile.get("h1_requires_rescue") or False),
        "h1_requires_weak_necessity": bool(profile.get("h1_requires_weak_necessity") or False),
    }
    decision_threshold = _infer_min_posterior_for_decision(
        rows,
        draft_spec,
        fallback=float(min_posterior_for_decision),
    )
    return {
        "version": version,
        "family": family,
        "kind": "state_evidence_adjudicator",
        "min_posterior_for_decision": round(float(decision_threshold), 6),
        "feature_sets": feature_sets,
        "behavior_prior": behavior_prior,
        "group_weights": group_weights,
        "group_targets": group_targets,
        "minimum_groups_for_decision": int(profile["minimum_groups_for_decision"]),
        "minimum_interchange_score_for_h2": float(
            score_thresholds.get("minimum_interchange_score_for_h2")
            or profile["minimum_interchange_score_for_h2"]
        ),
        "minimum_necessity_score_for_h2": float(
            score_thresholds.get("minimum_necessity_score_for_h2")
            or profile["minimum_necessity_score_for_h2"]
        ),
        "minimum_interchange_score_for_h1": float(
            score_thresholds.get("minimum_interchange_score_for_h1")
            or profile["minimum_interchange_score_for_h1"]
        ),
        "maximum_necessity_score_for_h1": float(
            score_thresholds.get("maximum_necessity_score_for_h1")
            or profile["maximum_necessity_score_for_h1"]
        ),
        "minimum_presence_score_for_nonh0": float(
            score_thresholds.get("minimum_presence_score_for_nonh0") or profile["minimum_presence_score_for_nonh0"]
        ),
        "minimum_realism_score_for_h2": float(
            score_thresholds.get("minimum_realism_score_for_h2") or profile["minimum_realism_score_for_h2"]
        ),
        "minimum_realism_score_for_h1": float(
            score_thresholds.get("minimum_realism_score_for_h1") or profile["minimum_realism_score_for_h1"]
        ),
        "minimum_rescue_score_for_h1": float(
            score_thresholds.get("minimum_rescue_score_for_h1") or profile["minimum_rescue_score_for_h1"]
        ),
        "allow_h1_without_rescue": bool(profile["allow_h1_without_rescue"]),
        "h1_requires_rescue": bool(profile.get("h1_requires_rescue") or False),
        "h1_requires_weak_necessity": bool(profile.get("h1_requires_weak_necessity") or False),
        "minimum_use_score_for_h2": round(
            mean(
                [
                    float(score_thresholds.get("minimum_interchange_score_for_h2") or profile["minimum_interchange_score_for_h2"]),
                    float(score_thresholds.get("minimum_necessity_score_for_h2") or profile["minimum_necessity_score_for_h2"]),
                ]
            ),
            6,
        ),
        "minimum_use_score_for_h1_without_rescue": float(
            score_thresholds.get("minimum_interchange_score_for_h1")
            or profile["minimum_interchange_score_for_h1"]
        ),
        "thresholds": thresholds,
        "state_counts": _state_counts(rows),
        "notes": list(notes or []),
        "behavior_feature_share": behavior_share,
    }


def predict_state_evidence_posteriors(
    spec: dict[str, Any],
    signals: dict[str, Any],
) -> dict[str, Any]:
    gate = _evaluate_gate(spec, signals)
    group_scores = _compute_group_scores(spec, signals)
    evidence_flags = _build_evidence_flags(spec, group_scores)

    raw_scores: dict[str, float] = {}
    per_state_distances: dict[str, float | None] = {}
    for state_id in STATE_IDS:
        distance = _state_distance(spec, state_id, group_scores)
        distance = _apply_state_penalties(spec, state_id, distance, evidence_flags)
        per_state_distances[state_id] = distance
        raw_scores[state_id] = float("-inf") if distance is None else -distance

    posteriors = _softmax(raw_scores)
    ranked = sorted(posteriors.items(), key=lambda item: item[1], reverse=True)
    top_state, top_posterior = ranked[0] if ranked else ("H0", 0.0)
    second_posterior = ranked[1][1] if len(ranked) > 1 else 0.0
    selected_state = top_state

    decision_reasons: list[str] = []
    if not gate["audit_ready"]:
        selected_state = "underdetermined"
        decision_reasons.extend(gate["gate_reasons"])
    elif evidence_flags["observed_group_count"] < int(spec.get("minimum_groups_for_decision") or 2):
        selected_state = "underdetermined"
        decision_reasons.append("insufficient_group_evidence")
    elif float(top_posterior) < float(spec.get("min_posterior_for_decision") or DEFAULT_MIN_POSTERIOR_FOR_DECISION):
        selected_state = "underdetermined"
        decision_reasons.append("posterior_below_threshold")
    elif top_state == "H1" and float(group_scores["behavior_prior"]["score"]) >= 0.75 and not evidence_flags["h2_eligible"]:
        selected_state = "underdetermined"
        decision_reasons.append("strong_behavior_without_h2_evidence")
    elif top_state == "H2" and not evidence_flags["h2_eligible"]:
        selected_state = "underdetermined"
        if not evidence_flags.get("h2_interchange_ok", True):
            decision_reasons.append("h2_interchange_evidence_insufficient")
        if not evidence_flags.get("h2_necessity_ok", True):
            decision_reasons.append("h2_necessity_evidence_insufficient")
        if not evidence_flags.get("h2_realism_ok", True):
            decision_reasons.append("h2_realism_evidence_insufficient")
        if not decision_reasons:
            decision_reasons.append("h2_causal_evidence_insufficient")
    elif top_state == "H1" and not evidence_flags["h1_eligible"]:
        selected_state = "underdetermined"
        if float(evidence_flags["presence_score"]) < float(spec.get("minimum_presence_score_for_nonh0") or 0.65):
            decision_reasons.append("h1_presence_evidence_insufficient")
        if not evidence_flags.get("h1_interchange_ok", True):
            decision_reasons.append("h1_interchange_evidence_insufficient")
        if not evidence_flags.get("h1_rescue_ok", True):
            decision_reasons.append("h1_rescue_evidence_insufficient")
        if not evidence_flags.get("h1_necessity_ok", True):
            decision_reasons.append("h1_necessity_too_strong")
        if not evidence_flags.get("h1_realism_ok", True):
            decision_reasons.append("h1_realism_evidence_insufficient")
        if not decision_reasons:
            decision_reasons.append("h1_presence_evidence_insufficient")

    return {
        "selected_state": selected_state,
        "top_state": top_state,
        "top_posterior": round(float(top_posterior), 6),
        "posterior_margin": round(float(top_posterior) - float(second_posterior), 6),
        "posteriors": {state_id: round(float(value), 6) for state_id, value in posteriors.items()},
        "state_distances": {
            state_id: (round(float(distance), 6) if distance is not None else None)
            for state_id, distance in per_state_distances.items()
        },
        "gate": gate,
        "group_scores": group_scores,
        "evidence_flags": evidence_flags,
        "decision_reasons": sorted(set(decision_reasons)),
    }


def evaluate_state_adjudicator(
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    confusion: dict[str, dict[str, int]] = {state_id: {} for state_id in STATE_IDS}
    correct = 0
    underdetermined = 0
    expected_posteriors: list[float] = []
    margins: list[float] = []

    for row in rows:
        prediction = predict_state_evidence_posteriors(spec, row["signals"])
        expected = str(row["expected_state"])
        selected = str(prediction["selected_state"])
        if selected == "underdetermined":
            underdetermined += 1
        if selected == expected:
            correct += 1
        expected_posteriors.append(float(prediction["posteriors"].get(expected) or 0.0))
        margins.append(float(prediction["posterior_margin"]))
        confusion.setdefault(expected, {})
        confusion[expected][selected] = confusion[expected].get(selected, 0) + 1
        predictions.append(
            {
                "phase11_prefix": row["phase11_prefix"],
                "control_id": row["control_id"],
                "expected_state": expected,
                "panel_role": row.get("panel_role"),
                **prediction,
            }
        )

    return {
        "run_count": len(rows),
        "accuracy": round(correct / len(rows), 6) if rows else None,
        "underdetermined_rate": round(underdetermined / len(rows), 6) if rows else None,
        "underdetermined_count": underdetermined,
        "mean_expected_state_posterior": round(mean(expected_posteriors), 6) if expected_posteriors else None,
        "mean_posterior_margin": round(mean(margins), 6) if margins else None,
        "confusion": confusion,
        "per_state_metrics": _compute_per_state_metrics(confusion),
        "macro_f1": _macro_f1(confusion),
        "predictions": predictions,
    }


def without_behavior_prior(spec: dict[str, Any]) -> dict[str, Any]:
    return without_state_groups(spec, "behavior_prior")


def without_state_groups(spec: dict[str, Any], *group_names: str) -> dict[str, Any]:
    cloned = copy.deepcopy(spec)
    group_weights = dict(cloned.get("group_weights") or {})
    for group_name in group_names:
        if group_name in group_weights:
            group_weights[group_name] = 0.0
    cloned["group_weights"] = group_weights
    feature_sets = dict(cloned.get("feature_sets") or {})
    for group_name in group_names:
        if group_name in feature_sets:
            feature_sets[group_name] = []
    if "interchange" in group_names or "necessity" in group_names:
        feature_sets["use"] = _merge_feature_sets(
            list(feature_sets.get("interchange") or []),
            list(feature_sets.get("necessity") or []),
        )
    cloned["feature_sets"] = feature_sets
    behavior_prior = dict(cloned.get("behavior_prior") or {})
    if "behavior_prior" in group_names:
        behavior_prior["weight"] = 0.0
    cloned["behavior_prior"] = behavior_prior
    return cloned


def _profile_for_family(family: str) -> dict[str, Any]:
    if family not in FAMILY_PROFILES:
        raise KeyError(f"Unsupported family: {family}")
    return FAMILY_PROFILES[family]


def _build_thresholds(rows: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    thresholds: dict[str, dict[str, float | None]] = {}
    descriptors = (
        list(profile["presence_features"])
        + list(profile.get("interchange_features") or profile.get("use_features") or ())
        + list(profile.get("necessity_features") or ())
        + list(profile["rescue_features"])
        + list(profile["realism_features"])
    )
    for descriptor in descriptors:
        feature = str(descriptor["feature"])
        values = _signal_values(rows, feature)
        thresholds[feature] = {
            "boundary": _infer_boundary(rows, descriptor),
            "scale": _infer_scale(values, float(descriptor.get("scale_floor") or 0.05)),
            "direction": str(descriptor["direction"]),
        }
    return thresholds


def _build_behavior_prior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature = "b0_gen_clause_f1"
    low_boundary = _threshold_between(
        _state_mean(rows, "H0", feature),
        _state_mean(rows, "H1", feature),
        default=0.05,
    )
    high_boundary = _threshold_between(
        _state_mean(rows, "H1", feature),
        _state_mean(rows, "H2", feature),
        default=max(low_boundary + 0.1, 0.75),
    )
    if high_boundary <= low_boundary:
        high_boundary = round(low_boundary + 0.1, 6)
    return {
        "feature": feature,
        "low_boundary": low_boundary,
        "high_boundary": high_boundary,
        "weight": 0.15,
    }


def _build_group_targets(rows: list[dict[str, Any]], provisional_spec: dict[str, Any]) -> dict[str, dict[str, float]]:
    group_targets: dict[str, dict[str, float]] = {}
    for state_id in STATE_IDS:
        state_rows = [row for row in rows if str(row["expected_state"]) == state_id]
        state_group_scores = [_compute_group_scores(provisional_spec, row["signals"]) for row in state_rows]
        group_targets[state_id] = {
            "presence": _mean_group_score(state_group_scores, "presence", fallback=0.5),
            "interchange": _mean_group_score(state_group_scores, "interchange", fallback=0.5),
            "necessity": _mean_group_score(state_group_scores, "necessity", fallback=0.5),
            "rescue": _mean_group_score(state_group_scores, "rescue", fallback=0.5),
            "realism": _mean_group_score(state_group_scores, "realism", fallback=0.5),
            "behavior_prior": _mean_group_score(state_group_scores, "behavior_prior", fallback=0.5),
        }
        group_targets[state_id]["use"] = round(
            mean([group_targets[state_id]["interchange"], group_targets[state_id]["necessity"]]),
            6,
        )
    return group_targets


def _build_score_thresholds(rows: list[dict[str, Any]], provisional_spec: dict[str, Any]) -> dict[str, float]:
    per_state_group_scores: dict[str, list[dict[str, Any]]] = {
        state_id: [
            _compute_group_scores(provisional_spec, row["signals"])
            for row in rows
            if str(row["expected_state"]) == state_id
        ]
        for state_id in STATE_IDS
    }
    return {
        "minimum_presence_score_for_nonh0": _threshold_between(
            _mean_group_score(per_state_group_scores["H0"], "presence", fallback=0.5),
            _mean_available(
                [
                    _mean_group_score(per_state_group_scores["H1"], "presence", fallback=0.5),
                    _mean_group_score(per_state_group_scores["H2"], "presence", fallback=0.5),
                ]
            ),
            default=0.65,
        ),
        "minimum_interchange_score_for_h2": _threshold_between(
            _mean_group_score(per_state_group_scores["H1"], "interchange", fallback=0.35),
            _mean_group_score(per_state_group_scores["H2"], "interchange", fallback=0.55),
            default=0.55,
        ),
        "minimum_necessity_score_for_h2": _threshold_between(
            _mean_group_score(per_state_group_scores["H1"], "necessity", fallback=0.35),
            _mean_group_score(per_state_group_scores["H2"], "necessity", fallback=0.55),
            default=0.55,
        ),
        "minimum_interchange_score_for_h1": _threshold_between(
            _mean_group_score(per_state_group_scores["H0"], "interchange", fallback=0.15),
            _mean_group_score(per_state_group_scores["H1"], "interchange", fallback=0.3),
            default=0.2,
        ),
        "maximum_necessity_score_for_h1": _threshold_between(
            _mean_group_score(per_state_group_scores["H1"], "necessity", fallback=0.2),
            _mean_group_score(per_state_group_scores["H2"], "necessity", fallback=0.55),
            default=0.42,
        ),
        "minimum_realism_score_for_h2": _threshold_between(
            _mean_available(
                [
                    _mean_group_score(per_state_group_scores["H0"], "realism", fallback=0.5),
                    _mean_group_score(per_state_group_scores["H1"], "realism", fallback=0.5),
                ]
            ),
            _mean_group_score(per_state_group_scores["H2"], "realism", fallback=0.65),
            default=0.45,
        ),
        "minimum_realism_score_for_h1": _threshold_between(
            _mean_group_score(per_state_group_scores["H0"], "realism", fallback=0.3),
            _mean_group_score(per_state_group_scores["H1"], "realism", fallback=0.45),
            default=0.35,
        ),
        "minimum_rescue_score_for_h1": _threshold_between(
            _mean_available(
                [
                    _mean_group_score(per_state_group_scores["H0"], "rescue", fallback=0.2),
                    _mean_group_score(per_state_group_scores["H2"], "rescue", fallback=0.2),
                ]
            ),
            _mean_group_score(per_state_group_scores["H1"], "rescue", fallback=0.35),
            default=0.35,
        ),
    }


def _materialize_feature_set(
    descriptors: tuple[dict[str, Any], ...],
    thresholds: dict[str, dict[str, float | None]],
) -> list[dict[str, Any]]:
    materialized = []
    for descriptor in descriptors:
        feature = str(descriptor["feature"])
        threshold = dict(thresholds.get(feature) or {})
        materialized.append(
            {
                "feature": feature,
                "direction": str(descriptor["direction"]),
                "boundary": threshold.get("boundary"),
                "scale": threshold.get("scale"),
                "optional": bool(descriptor.get("optional") or False),
            }
        )
    return materialized


def _evaluate_gate(spec: dict[str, Any], signals: dict[str, Any]) -> dict[str, Any]:
    gate_reasons: list[str] = []
    if not bool(signals.get("audit_ready")):
        gate_reasons.append("audit_ready_false")
    if not bool(signals.get("state_anchor_ready", True)):
        gate_reasons.append("state_anchor_missing")
    observed_modules = [
        bool(signals.get("m2_observed")),
        bool(signals.get("m3_observed")),
        bool(signals.get("m4_observed")),
    ]
    if not any(observed_modules):
        gate_reasons.append("no_generation_level_module_observed")

    behavior_value = _safe_float(signals.get("b0_gen_clause_f1"))
    target_regime_type = "unknown"
    behavior_prior = dict(spec.get("behavior_prior") or {})
    low_boundary = float(behavior_prior.get("low_boundary") or 0.05)
    high_boundary = float(behavior_prior.get("high_boundary") or max(low_boundary + 0.1, 0.75))
    if behavior_value is not None:
        if behavior_value < low_boundary:
            target_regime_type = "weak"
        elif behavior_value >= high_boundary:
            target_regime_type = "strong"
        else:
            target_regime_type = "intermediate"

    return {
        "audit_ready": not gate_reasons,
        "gate_reasons": gate_reasons,
        "target_regime_type": target_regime_type,
    }


def _compute_group_scores(spec: dict[str, Any], signals: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group_name in ("presence", "interchange", "necessity", "rescue", "realism"):
        feature_set = list((spec.get("feature_sets") or {}).get(group_name) or [])
        component_scores = []
        component_records = []
        for descriptor in feature_set:
            value = _safe_float(signals.get(descriptor["feature"]))
            if value is None:
                continue
            score = _feature_score(
                value=value,
                boundary=_safe_float(descriptor.get("boundary")),
                scale=_safe_float(descriptor.get("scale")) or 0.05,
                direction=str(descriptor.get("direction") or "high"),
            )
            component_scores.append(score)
            component_records.append(
                {
                    "feature": descriptor["feature"],
                    "value": round(value, 6),
                    "score": round(score, 6),
                }
            )
        observed = len(component_scores) > 0
        groups[group_name] = {
            "score": round(mean(component_scores), 6) if component_scores else 0.5,
            "observed": observed,
            "feature_count": len(component_scores),
            "components": component_records,
        }

    interchange_score = float(groups["interchange"]["score"])
    necessity_score = float(groups["necessity"]["score"])
    interchange_observed = bool(groups["interchange"]["observed"])
    necessity_observed = bool(groups["necessity"]["observed"])
    use_components = list(groups["interchange"]["components"]) + list(groups["necessity"]["components"])
    use_scores = []
    if interchange_observed:
        use_scores.append(interchange_score)
    if necessity_observed:
        use_scores.append(necessity_score)
    groups["use"] = {
        "score": round(mean(use_scores), 6) if use_scores else 0.5,
        "observed": bool(use_scores),
        "feature_count": len(use_components),
        "components": use_components,
    }

    behavior_prior = dict(spec.get("behavior_prior") or {})
    behavior_value = _safe_float(signals.get(behavior_prior.get("feature") or "b0_gen_clause_f1"))
    if behavior_value is None:
        groups["behavior_prior"] = {"score": 0.5, "observed": False, "feature_count": 0, "components": []}
    else:
        behavior_score = _scaled_between(
            behavior_value,
            float(behavior_prior.get("low_boundary") or 0.05),
            float(behavior_prior.get("high_boundary") or 0.75),
        )
        groups["behavior_prior"] = {
            "score": round(behavior_score, 6),
            "observed": True,
            "feature_count": 1,
            "components": [{"feature": behavior_prior.get("feature") or "b0_gen_clause_f1", "value": round(behavior_value, 6), "score": round(behavior_score, 6)}],
        }
    return groups


def _mean_group_score(
    state_group_scores: list[dict[str, Any]],
    group_name: str,
    *,
    fallback: float,
) -> float:
    usable = [
        float(value)
        for value in (
            _safe_float(group_scores.get(group_name, {}).get("score"))
            for group_scores in state_group_scores
        )
        if value is not None
    ]
    if not usable:
        return round(float(fallback), 6)
    return round(mean(usable), 6)


def _build_evidence_flags(spec: dict[str, Any], group_scores: dict[str, Any]) -> dict[str, Any]:
    presence_score = float(group_scores["presence"]["score"])
    interchange_score = float(group_scores["interchange"]["score"])
    necessity_score = float(group_scores["necessity"]["score"])
    use_score = float(group_scores["use"]["score"])
    rescue_score = float(group_scores["rescue"]["score"])
    realism_score = float(group_scores["realism"]["score"])
    observed_group_count = sum(
        1
        for group_name in ("presence", "interchange", "necessity", "rescue", "realism")
        if bool(group_scores[group_name]["observed"])
    )
    minimum_presence = float(spec.get("minimum_presence_score_for_nonh0") or 0.65)
    minimum_interchange_h2 = float(
        spec.get("minimum_interchange_score_for_h2") or spec.get("minimum_use_score_for_h2") or 0.55
    )
    minimum_necessity_h2 = float(
        spec.get("minimum_necessity_score_for_h2") or spec.get("minimum_use_score_for_h2") or 0.55
    )
    minimum_interchange_h1 = float(
        spec.get("minimum_interchange_score_for_h1")
        or spec.get("minimum_use_score_for_h1_without_rescue")
        or 0.25
    )
    maximum_necessity_h1 = float(spec.get("maximum_necessity_score_for_h1") or minimum_necessity_h2)
    minimum_rescue_h1 = float(spec.get("minimum_rescue_score_for_h1") or 0.35)
    minimum_realism_h2 = float(spec.get("minimum_realism_score_for_h2") or 0.45)
    minimum_realism_h1 = float(spec.get("minimum_realism_score_for_h1") or minimum_realism_h2)

    h2_realism_ok = not bool(group_scores["realism"]["observed"]) or realism_score >= minimum_realism_h2
    h1_realism_ok = not bool(group_scores["realism"]["observed"]) or realism_score >= minimum_realism_h1
    h2_interchange_ok = bool(group_scores["interchange"]["observed"]) and interchange_score >= minimum_interchange_h2
    h2_necessity_ok = bool(group_scores["necessity"]["observed"]) and necessity_score >= minimum_necessity_h2
    h1_interchange_ok = bool(group_scores["interchange"]["observed"]) and interchange_score >= minimum_interchange_h1
    h1_necessity_ok = not bool(group_scores["necessity"]["observed"]) or necessity_score <= maximum_necessity_h1
    h1_rescue_ok = bool(group_scores["rescue"]["observed"]) and rescue_score >= minimum_rescue_h1

    h2_eligible = bool(
        presence_score >= minimum_presence
        and h2_interchange_ok
        and h2_necessity_ok
        and h2_realism_ok
    )
    h1_eligible = bool(
        presence_score >= minimum_presence
        and h1_interchange_ok
        and h1_realism_ok
        and (not bool(spec.get("h1_requires_weak_necessity")) or h1_necessity_ok)
        and (
            h1_rescue_ok
            or (
                bool(spec.get("allow_h1_without_rescue"))
                and not bool(spec.get("h1_requires_rescue"))
                and use_score >= minimum_interchange_h1
                and necessity_score < minimum_necessity_h2
            )
        )
        and not h2_eligible
    )
    return {
        "presence_score": round(presence_score, 6),
        "interchange_score": round(interchange_score, 6),
        "necessity_score": round(necessity_score, 6),
        "use_score": round(use_score, 6),
        "rescue_score": round(rescue_score, 6),
        "realism_score": round(realism_score, 6),
        "observed_group_count": observed_group_count,
        "h1_interchange_ok": h1_interchange_ok,
        "h1_necessity_ok": h1_necessity_ok,
        "h1_rescue_ok": h1_rescue_ok,
        "h1_realism_ok": h1_realism_ok,
        "h2_interchange_ok": h2_interchange_ok,
        "h2_necessity_ok": h2_necessity_ok,
        "h2_realism_ok": h2_realism_ok,
        "h1_eligible": h1_eligible,
        "h2_eligible": h2_eligible,
    }


def _infer_min_posterior_for_decision(
    rows: list[dict[str, Any]],
    draft_spec: dict[str, Any],
    *,
    fallback: float,
) -> float:
    probe_spec = copy.deepcopy(draft_spec)
    probe_spec["min_posterior_for_decision"] = 0.0
    correct_top_posteriors: list[float] = []
    for row in rows:
        prediction = predict_state_evidence_posteriors(probe_spec, row["signals"])
        if str(prediction["top_state"]) == str(row["expected_state"]):
            correct_top_posteriors.append(float(prediction["top_posterior"]))
    if not correct_top_posteriors:
        return round(float(fallback), 6)
    learned = min(correct_top_posteriors) - 0.005
    return round(max(0.33, min(float(fallback), learned)), 6)


def _state_distance(spec: dict[str, Any], state_id: str, group_scores: dict[str, Any]) -> float | None:
    targets = dict((spec.get("group_targets") or {}).get(state_id) or {})
    group_weights = dict(spec.get("group_weights") or {})
    squared_terms = []
    for group_name in ("presence", "interchange", "necessity", "rescue", "realism", "behavior_prior"):
        weight = float(group_weights.get(group_name) or 0.0)
        if weight <= 0.0:
            continue
        score = _safe_float(group_scores.get(group_name, {}).get("score"))
        target = _safe_float(targets.get(group_name))
        if score is None or target is None:
            continue
        squared_terms.append(weight * ((score - target) ** 2))
    if not squared_terms:
        return None
    return round(mean(squared_terms), 6)


def _state_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {state_id: sum(1 for row in rows if str(row["expected_state"]) == state_id) for state_id in STATE_IDS}


def _apply_state_penalties(
    spec: dict[str, Any],
    state_id: str,
    distance: float | None,
    evidence_flags: dict[str, Any],
) -> float | None:
    if distance is None:
        return None
    adjusted = float(distance)
    if state_id == "H2" and not bool(evidence_flags["h2_eligible"]):
        adjusted += 0.25
    if state_id == "H1" and not bool(evidence_flags["h1_eligible"]):
        adjusted += 0.15
    if state_id == "H0" and float(evidence_flags["presence_score"]) >= float(spec.get("minimum_presence_score_for_nonh0") or 0.65):
        adjusted += 0.05
    return adjusted


def _merge_feature_sets(*feature_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for feature_set in feature_sets:
        for descriptor in feature_set:
            key = (str(descriptor.get("feature") or ""), str(descriptor.get("direction") or ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(descriptor))
    return merged


def _infer_boundary(rows: list[dict[str, Any]], descriptor: dict[str, Any]) -> float:
    strategy = str(descriptor.get("boundary_strategy") or "fixed")
    feature = str(descriptor["feature"])
    default = float(descriptor.get("fallback_boundary") or 0.05)
    if strategy == "fixed":
        return default
    if strategy == "h0_vs_present":
        return _threshold_between(
            _state_mean(rows, "H0", feature),
            _mean_available(
                [
                    _state_mean(rows, "H1", feature),
                    _state_mean(rows, "H2", feature),
                ]
            ),
            default=default,
        )
    if strategy == "h1_vs_nonh1":
        return _threshold_between(
            _mean_available(
                [
                    _state_mean(rows, "H0", feature),
                    _state_mean(rows, "H2", feature),
                ]
            ),
            _state_mean(rows, "H1", feature),
            default=default,
        )
    return default


def _signal_values(rows: list[dict[str, Any]], feature: str) -> list[float]:
    values = []
    for row in rows:
        value = _safe_float(row["signals"].get(feature))
        if value is not None:
            values.append(value)
    return values


def _state_mean(rows: list[dict[str, Any]], state_id: str, feature: str) -> float | None:
    values = []
    for row in rows:
        if str(row["expected_state"]) != state_id:
            continue
        value = _safe_float(row["signals"].get(feature))
        if value is not None:
            values.append(value)
    if not values:
        return None
    return round(mean(values), 6)


def _infer_scale(values: list[float], floor: float) -> float:
    unique_values = sorted({round(value, 6) for value in values})
    if len(unique_values) < 2:
        return round(floor, 6)
    gaps = [right - left for left, right in zip(unique_values, unique_values[1:]) if (right - left) > 0.0]
    if not gaps:
        return round(floor, 6)
    return round(max(floor, min(gaps) / 2.0), 6)


def _feature_score(*, value: float, boundary: float | None, scale: float, direction: str) -> float:
    pivot = 0.0 if boundary is None else float(boundary)
    safe_scale = max(float(scale), 1e-6)
    signed_distance = (float(value) - pivot) / safe_scale
    if direction == "low":
        signed_distance *= -1.0
    return 1.0 / (1.0 + math.exp(-signed_distance))


def _scaled_between(value: float, low_boundary: float, high_boundary: float) -> float:
    if high_boundary <= low_boundary:
        return 0.5
    clipped = min(max(float(value), low_boundary), high_boundary)
    return (clipped - low_boundary) / (high_boundary - low_boundary)


def _threshold_between(left: float | None, right: float | None, *, default: float) -> float:
    if left is None or right is None:
        return round(default, 6)
    return round((float(left) + float(right)) / 2.0, 6)


def _mean_available(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return round(mean(usable), 6)


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    finite_scores = [score for score in scores.values() if math.isfinite(score)]
    if not finite_scores:
        return {state_id: (1.0 / len(scores) if scores else 0.0) for state_id in scores}
    max_score = max(finite_scores)
    exps = {
        state_id: (math.exp(score - max_score) if math.isfinite(score) else 0.0)
        for state_id, score in scores.items()
    }
    total = sum(exps.values())
    if total <= 0.0:
        return {state_id: (1.0 / len(scores) if scores else 0.0) for state_id in scores}
    return {state_id: value / total for state_id, value in exps.items()}


def _compute_per_state_metrics(confusion: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int | None]]:
    metrics: dict[str, dict[str, float | int | None]] = {}
    labels = set(confusion.keys()) | {predicted for bucket in confusion.values() for predicted in bucket.keys()}
    for state_id in sorted(label for label in labels if label in set(STATE_IDS)):
        tp = int(confusion.get(state_id, {}).get(state_id, 0))
        fp = sum(int(confusion.get(expected, {}).get(state_id, 0)) for expected in confusion if expected != state_id)
        fn = sum(int(count) for predicted, count in confusion.get(state_id, {}).items() if predicted != state_id)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0.0:
            f1 = 2.0 * precision * recall / (precision + recall)
        metrics[state_id] = {
            "support": sum(int(count) for count in confusion.get(state_id, {}).values()),
            "precision": round(precision, 6) if precision is not None else None,
            "recall": round(recall, 6) if recall is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
        }
    return metrics


def _macro_f1(confusion: dict[str, dict[str, int]]) -> float | None:
    per_state = _compute_per_state_metrics(confusion)
    f1_values = [float(metrics["f1"]) for metrics in per_state.values() if metrics.get("f1") is not None]
    if not f1_values:
        return None
    return round(mean(f1_values), 6)


__all__ = [
    "evaluate_state_adjudicator",
    "fit_state_adjudicator",
    "predict_state_evidence_posteriors",
    "without_behavior_prior",
    "without_state_groups",
]
