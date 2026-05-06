from __future__ import annotations

import math
from statistics import mean
from typing import Any


DEFAULT_BINDING_FEATURES = (
    "b0_gen_clause_f1",
    "m1_semantic_role_accuracy",
    "m2_advantage_mean",
)

DEFAULT_MIN_POSTERIOR_FOR_DECISION = 0.55
DEFAULT_FEATURE_SCALE_FLOOR = 0.05


def fit_prototype_adjudicator(
    rows: list[dict[str, Any]],
    *,
    features: tuple[str, ...] = DEFAULT_BINDING_FEATURES,
    family: str = "G1_binding",
    version: str = "v5_binding_frozen_v1",
    min_posterior_for_decision: float = DEFAULT_MIN_POSTERIOR_FOR_DECISION,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    grouped = _group_rows_by_expected_state(rows)
    state_prototypes: dict[str, dict[str, float | None]] = {}
    state_counts: dict[str, int] = {}
    for state_id in ("H0", "H1", "H2"):
        state_rows = grouped.get(state_id, [])
        state_counts[state_id] = len(state_rows)
        state_prototypes[state_id] = {
            feature: _mean_signal(state_rows, feature) for feature in features
        }

    feature_scales = {
        feature: _infer_feature_scale(rows, feature)
        for feature in features
    }

    return {
        "version": version,
        "family": family,
        "kind": "prototype_distance_adjudicator",
        "features": list(features),
        "min_posterior_for_decision": round(float(min_posterior_for_decision), 6),
        "state_prototypes": state_prototypes,
        "feature_scales": feature_scales,
        "state_counts": state_counts,
        "notes": list(notes or []),
    }


def predict_state_posteriors(
    spec: dict[str, Any],
    signals: dict[str, Any],
) -> dict[str, Any]:
    features = tuple(str(feature) for feature in spec.get("features") or DEFAULT_BINDING_FEATURES)
    min_posterior = float(spec.get("min_posterior_for_decision") or DEFAULT_MIN_POSTERIOR_FOR_DECISION)
    prototypes = dict(spec.get("state_prototypes") or {})
    scales = {
        str(feature): max(float(scale), DEFAULT_FEATURE_SCALE_FLOOR)
        for feature, scale in (spec.get("feature_scales") or {}).items()
    }

    state_scores: dict[str, float] = {}
    state_distances: dict[str, float | None] = {}
    features_used: dict[str, list[str]] = {}
    for state_id in ("H0", "H1", "H2"):
        prototype = dict(prototypes.get(state_id) or {})
        distance, used = _state_distance(signals, prototype, features, scales)
        state_distances[state_id] = distance
        features_used[state_id] = used
        if distance is None:
            state_scores[state_id] = float("-inf")
        else:
            state_scores[state_id] = -0.5 * distance

    posteriors = _softmax(state_scores)
    ranked = sorted(posteriors.items(), key=lambda item: item[1], reverse=True)
    top_state, top_posterior = ranked[0] if ranked else ("H0", 0.0)
    second_posterior = ranked[1][1] if len(ranked) > 1 else 0.0
    selected_state = top_state if float(top_posterior) >= min_posterior else "underdetermined"

    return {
        "selected_state": selected_state,
        "top_state": top_state,
        "top_posterior": round(float(top_posterior), 6),
        "posterior_margin": round(float(top_posterior) - float(second_posterior), 6),
        "posteriors": {state: round(float(value), 6) for state, value in posteriors.items()},
        "state_distances": {
            state: (round(float(distance), 6) if distance is not None else None)
            for state, distance in state_distances.items()
        },
        "features_used": features_used,
    }


def evaluate_adjudicator(
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    predictions = []
    confusion: dict[str, dict[str, int]] = {state: {} for state in ("H0", "H1", "H2")}
    correct = 0
    underdetermined = 0
    expected_posteriors: list[float] = []
    margins: list[float] = []

    for row in rows:
        prediction = predict_state_posteriors(spec, row["signals"])
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
        "underdetermined_count": underdetermined,
        "mean_expected_state_posterior": round(mean(expected_posteriors), 6) if expected_posteriors else None,
        "mean_posterior_margin": round(mean(margins), 6) if margins else None,
        "confusion": confusion,
        "predictions": predictions,
    }


def _group_rows_by_expected_state(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"H0": [], "H1": [], "H2": []}
    for row in rows:
        grouped.setdefault(str(row["expected_state"]), []).append(row)
    return grouped


def _mean_signal(rows: list[dict[str, Any]], signal_name: str) -> float | None:
    values = []
    for row in rows:
        value = row["signals"].get(signal_name)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return round(mean(values), 6)


def _infer_feature_scale(rows: list[dict[str, Any]], signal_name: str) -> float:
    values = []
    for row in rows:
        value = row["signals"].get(signal_name)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    unique_values = sorted({round(value, 6) for value in values})
    if len(unique_values) < 2:
        return DEFAULT_FEATURE_SCALE_FLOOR
    gaps = [
        right - left
        for left, right in zip(unique_values, unique_values[1:])
        if (right - left) > 0.0
    ]
    if not gaps:
        return DEFAULT_FEATURE_SCALE_FLOOR
    return round(max(DEFAULT_FEATURE_SCALE_FLOOR, min(gaps) / 2.0), 6)


def _state_distance(
    signals: dict[str, Any],
    prototype: dict[str, Any],
    features: tuple[str, ...],
    scales: dict[str, float],
) -> tuple[float | None, list[str]]:
    squared_terms = []
    used_features = []
    for feature in features:
        value = signals.get(feature)
        center = prototype.get(feature)
        if value is None or center is None or isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)) or not isinstance(center, (int, float)):
            continue
        scale = max(float(scales.get(feature) or DEFAULT_FEATURE_SCALE_FLOOR), DEFAULT_FEATURE_SCALE_FLOOR)
        squared_terms.append(((float(value) - float(center)) / scale) ** 2)
        used_features.append(feature)
    if not squared_terms:
        return None, used_features
    return float(mean(squared_terms)), used_features


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    finite = [score for score in scores.values() if math.isfinite(score)]
    if not finite:
        return {state: (1.0 / len(scores) if scores else 0.0) for state in scores}
    max_score = max(finite)
    exps = {
        state: (math.exp(score - max_score) if math.isfinite(score) else 0.0)
        for state, score in scores.items()
    }
    total = sum(exps.values())
    if total <= 0.0:
        return {state: (1.0 / len(scores) if scores else 0.0) for state in scores}
    return {state: value / total for state, value in exps.items()}
