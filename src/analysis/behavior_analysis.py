from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from hypotheses import derive_phase11_binding_audit_signals, derive_phase11_geometry_audit_signals
from utils.io import read_jsonl


BINDING_SIGNAL_DEFINITIONS = {
    "binding_dev_audit_ready": {
        "module": "P0",
        "description": "Step 3 binding-development preconditions are jointly satisfied.",
    },
    "state_anchor_ready": {
        "module": "B0",
        "description": "The current control exposes control items and at least one non-control target stratum.",
    },
    "b0_gen_clause_f1": {
        "module": "B0",
        "description": "Generalization split clause F1.",
    },
    "m1_semantic_role_accuracy": {
        "module": "M1",
        "description": "Leave-one-out centroid accuracy for semantic roles.",
    },
    "m1_surface_control_gap": {
        "module": "M1",
        "description": "Semantic-role accuracy minus surface-position accuracy.",
    },
    "m2_observed": {
        "module": "M2",
        "description": "At least one generation-level interchange record was observed.",
    },
    "m2_advantage_mean": {
        "module": "M2",
        "description": "Actual minus strongest-control interchange advantage.",
    },
    "m3_observed": {
        "module": "M3",
        "description": "At least one generation-level necessity record was observed.",
    },
    "m3_selective_impairment_clause_f1": {
        "module": "M3",
        "description": "Role-ablation minus strongest-control clause-F1 drop.",
    },
    "m4_observed": {
        "module": "M4",
        "description": "At least one generation-level rescue record was observed.",
    },
    "m4_advantage_mean": {
        "module": "M4",
        "description": "Actual minus strongest-control rescue advantage.",
    },
}

GEOMETRY_SIGNAL_DEFINITIONS = {
    "geometry_dev_audit_ready": {
        "module": "P0",
        "description": "Step 6 geometry-development preconditions are jointly satisfied.",
    },
    "state_anchor_ready": {
        "module": "B0",
        "description": "The current run exposes control items and at least one non-control target stratum.",
    },
    "geometry_signal_ready": {
        "module": "M1",
        "description": "Role and filler geometry signals are observable for the current run.",
    },
    "b0_gen_clause_f1": {
        "module": "B0",
        "description": "Generalization split clause F1.",
    },
    "m1_semantic_role_accuracy": {
        "module": "M1",
        "description": "Leave-one-out centroid accuracy for semantic roles.",
    },
    "m1_filler_gap": {
        "module": "M1",
        "description": "Within-versus-between cosine gap for filler identity.",
    },
    "m1_surface_control_gap": {
        "module": "M1",
        "description": "Semantic-role accuracy minus surface-position accuracy.",
    },
    "m2_observed": {
        "module": "M2",
        "description": "At least one generation-level interchange record was observed.",
    },
    "m2_advantage_mean": {
        "module": "M2",
        "description": "Actual minus strongest-control interchange advantage.",
    },
    "m3_observed": {
        "module": "M3",
        "description": "At least one generation-level necessity record was observed.",
    },
    "m3_selective_impairment_clause_f1": {
        "module": "M3",
        "description": "Role-ablation minus strongest-control clause-F1 drop.",
    },
}


def _load_phase11_row(
    project_root: str | Path,
    phase11_prefix: str,
    *,
    control_id: str,
    expected_state: str,
    panel_role: str,
    derive_signals_fn: Any,
) -> dict[str, Any]:
    root = Path(project_root)
    protocol_summary = json.loads(
        (root / "reports/protocol_runs" / f"{phase11_prefix}_summary.json").read_text(encoding="utf-8")
    )
    discovery_prefix = str(protocol_summary.get("discovery_prefix") or phase11_prefix)
    discovery_summary = json.loads(
        (root / "reports/discovery_runs" / f"{discovery_prefix}_summary.json").read_text(encoding="utf-8")
    )
    behavior_run_name = str(discovery_summary["behavior_run_name"])
    behavior_summary = json.loads(
        (root / "reports/behavior_runs" / f"{behavior_run_name}_summary.json").read_text(encoding="utf-8")
    )
    linkage_records = read_jsonl(root / "artifacts/protocol_runs" / f"{phase11_prefix}_m5_generation_item_linkage.jsonl")
    signals = derive_signals_fn(
        behavior_summary=behavior_summary,
        discovery_summary=discovery_summary,
        protocol_summary=protocol_summary,
        linkage_records=linkage_records,
    )
    return {
        "phase11_prefix": phase11_prefix,
        "discovery_prefix": discovery_prefix,
        "behavior_run_name": behavior_run_name,
        "control_id": control_id,
        "expected_state": expected_state,
        "panel_role": panel_role,
        "model_name_or_path": protocol_summary["model_name_or_path"],
        "family": protocol_summary["family"],
        "layer_index": protocol_summary["layer_index"],
        "protocol_config": dict(protocol_summary.get("protocol_config") or {}),
        "signals": signals,
        "behavior_summary": behavior_summary.get("evaluation"),
    }


def load_phase11_binding_row(
    project_root: str | Path,
    phase11_prefix: str,
    *,
    control_id: str,
    expected_state: str,
    panel_role: str = "development",
) -> dict[str, Any]:
    return _load_phase11_row(
        project_root,
        phase11_prefix,
        control_id=control_id,
        expected_state=expected_state,
        panel_role=panel_role,
        derive_signals_fn=derive_phase11_binding_audit_signals,
    )


def load_phase11_geometry_row(
    project_root: str | Path,
    phase11_prefix: str,
    *,
    control_id: str,
    expected_state: str,
    panel_role: str = "development",
) -> dict[str, Any]:
    return _load_phase11_row(
        project_root,
        phase11_prefix,
        control_id=control_id,
        expected_state=expected_state,
        panel_role=panel_role,
        derive_signals_fn=derive_phase11_geometry_audit_signals,
    )


def build_geometry_dev_registry(
    rows: list[dict[str, Any]],
    *,
    version: str = "v6_geometry_dev_v1",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    grouped = _group_rows_by_expected_state(rows)
    thresholds = {
        "gen_h0_h1_boundary": _threshold_between(
            _state_mean(grouped, "H0", "b0_gen_clause_f1"),
            _state_mean(grouped, "H1", "b0_gen_clause_f1"),
            default=0.05,
        ),
        "gen_h1_h2_boundary": _threshold_between(
            _state_mean(grouped, "H1", "b0_gen_clause_f1"),
            _state_mean(grouped, "H2", "b0_gen_clause_f1"),
            default=0.5,
        ),
        "role_presence_boundary": _threshold_between(
            _state_mean(grouped, "H0", "m1_semantic_role_accuracy"),
            max(
                _state_mean(grouped, "H1", "m1_semantic_role_accuracy") or 0.0,
                _state_mean(grouped, "H2", "m1_semantic_role_accuracy") or 0.0,
            ),
            default=0.85,
        ),
        "filler_geometry_boundary": _threshold_between(
            _state_mean(grouped, "H0", "m1_filler_gap"),
            _min_state_mean(grouped, ("H1", "H2"), "m1_filler_gap"),
            default=0.33,
        ),
        "surface_geometry_boundary": _threshold_between(
            _state_mean(grouped, "H0", "m1_surface_control_gap"),
            _min_state_mean(grouped, ("H1", "H2"), "m1_surface_control_gap"),
            default=-0.1,
        ),
        "m2_positive_boundary": _threshold_between(
            _state_mean(grouped, "H0", "m2_advantage_mean"),
            _min_state_mean(grouped, ("H1", "H2"), "m2_advantage_mean"),
            default=0.05,
        ),
    }

    metadata = {
        "family": "G2_distributed_geometry",
        "development_controls": [
            {
                "control_id": row["control_id"],
                "phase11_prefix": row["phase11_prefix"],
                "expected_state": row["expected_state"],
            }
            for row in rows
        ],
        "thresholds": thresholds,
        "notes": list(notes or []),
        "no_forced_h2": True,
        "shared_physical_controls": True,
        "calibration_strategy": (
            "Geometry Step 6 uses generalization behavior to separate H2 from H1, while "
            "role/filler geometry plus a positive M2 floor distinguish {H1,H2} from H0."
        ),
    }

    return {
        "version": version,
        "metadata": metadata,
        "audit_preconditions": [
            {
                "signal": "geometry_dev_audit_ready",
                "op": "==",
                "value": True,
                "description": "Step 6 requires observable role/filler geometry plus generation-level protocol modules.",
            },
        ],
        "signal_definitions": GEOMETRY_SIGNAL_DEFINITIONS,
        "hypotheses": [
            {
                "id": "G2_distributed_geometry",
                "label": "Distributed Geometry Hypothesis",
                "description": "Role/filler information lives in a stable distributed subspace that can be absent, present-but-unused, or present-and-used.",
                "states": [
                    {
                        "id": "H2",
                        "meaning": "A stable distributed geometry is present and causally used for generation.",
                        "support": {
                            "all": [
                                {"signal": "state_anchor_ready", "op": "==", "value": True},
                                {"signal": "geometry_signal_ready", "op": "==", "value": True},
                                {"signal": "m2_observed", "op": "==", "value": True},
                                {"signal": "b0_gen_clause_f1", "op": ">=", "value": thresholds["gen_h1_h2_boundary"]},
                                {
                                    "signal": "m1_semantic_role_accuracy",
                                    "op": ">=",
                                    "value": thresholds["role_presence_boundary"],
                                },
                                {"signal": "m1_filler_gap", "op": ">=", "value": thresholds["filler_geometry_boundary"]},
                                {
                                    "signal": "m1_surface_control_gap",
                                    "op": ">=",
                                    "value": thresholds["surface_geometry_boundary"],
                                },
                                {"signal": "m2_advantage_mean", "op": ">=", "value": thresholds["m2_positive_boundary"]},
                            ]
                        },
                        "falsifiers": {
                            "any": [
                                {"signal": "geometry_signal_ready", "op": "==", "value": False},
                                {"signal": "m2_observed", "op": "==", "value": False},
                                {"signal": "m1_filler_gap", "op": "<", "value": thresholds["filler_geometry_boundary"]},
                                {"signal": "m2_advantage_mean", "op": "<", "value": thresholds["m2_positive_boundary"]},
                            ]
                        },
                    },
                    {
                        "id": "H1",
                        "meaning": "A stable distributed geometry is present but not cleanly used for generation.",
                        "support": {
                            "all": [
                                {"signal": "state_anchor_ready", "op": "==", "value": True},
                                {"signal": "geometry_signal_ready", "op": "==", "value": True},
                                {"signal": "m2_observed", "op": "==", "value": True},
                                {"signal": "b0_gen_clause_f1", "op": "<", "value": thresholds["gen_h1_h2_boundary"]},
                                {
                                    "signal": "m1_semantic_role_accuracy",
                                    "op": ">=",
                                    "value": thresholds["role_presence_boundary"],
                                },
                                {"signal": "m1_filler_gap", "op": ">=", "value": thresholds["filler_geometry_boundary"]},
                                {
                                    "signal": "m1_surface_control_gap",
                                    "op": ">=",
                                    "value": thresholds["surface_geometry_boundary"],
                                },
                                {"signal": "m2_advantage_mean", "op": ">=", "value": thresholds["m2_positive_boundary"]},
                            ]
                        },
                        "falsifiers": {
                            "any": [
                                {"signal": "geometry_signal_ready", "op": "==", "value": False},
                                {"signal": "m2_observed", "op": "==", "value": False},
                                {"signal": "m1_filler_gap", "op": "<", "value": thresholds["filler_geometry_boundary"]},
                                {"signal": "m2_advantage_mean", "op": "<", "value": thresholds["m2_positive_boundary"]},
                            ],
                            "all": [
                                {"signal": "b0_gen_clause_f1", "op": ">=", "value": thresholds["gen_h1_h2_boundary"]},
                                {"signal": "m2_observed", "op": "==", "value": True},
                                {"signal": "m2_advantage_mean", "op": ">=", "value": thresholds["m2_positive_boundary"]},
                            ],
                        },
                    },
                    {
                        "id": "H0",
                        "meaning": "No stable task-relevant distributed geometry is available.",
                        "support": {
                            "all": [
                                {"signal": "state_anchor_ready", "op": "==", "value": True},
                                {"signal": "b0_gen_clause_f1", "op": "<=", "value": thresholds["gen_h0_h1_boundary"]},
                            ],
                            "any": [
                                {"signal": "geometry_signal_ready", "op": "==", "value": False},
                                {
                                    "signal": "m1_semantic_role_accuracy",
                                    "op": "<",
                                    "value": thresholds["role_presence_boundary"],
                                },
                                {"signal": "m1_filler_gap", "op": "<", "value": thresholds["filler_geometry_boundary"]},
                                {
                                    "signal": "m1_surface_control_gap",
                                    "op": "<",
                                    "value": thresholds["surface_geometry_boundary"],
                                },
                                {"signal": "m2_observed", "op": "==", "value": False},
                                {"signal": "m2_advantage_mean", "op": "<", "value": thresholds["m2_positive_boundary"]},
                            ],
                        },
                    },
                ],
            }
        ],
    }


def summarize_geometry_calibration(
    rows: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    registry: dict[str, Any],
) -> dict[str, Any]:
    by_prefix = {row["phase11_prefix"]: row for row in rows}
    grouped = _group_rows_by_expected_state(rows)
    state_profiles = []
    for state_id in ("H0", "H1", "H2"):
        state_profiles.append(
            {
                "state_id": state_id,
                "run_count": len(grouped.get(state_id, [])),
                "mean_b0_gen_clause_f1": _state_mean(grouped, state_id, "b0_gen_clause_f1"),
                "mean_m1_semantic_role_accuracy": _state_mean(grouped, state_id, "m1_semantic_role_accuracy"),
                "mean_m1_filler_gap": _state_mean(grouped, state_id, "m1_filler_gap"),
                "mean_m1_surface_control_gap": _state_mean(grouped, state_id, "m1_surface_control_gap"),
                "mean_m2_advantage_mean": _state_mean(grouped, state_id, "m2_advantage_mean"),
            }
        )

    adjudications = []
    matched = 0
    for audit in audits:
        row = by_prefix[audit["phase11_prefix"]]
        selected = str(audit["support_matrix"][0]["selected_state"]) if audit["support_matrix"] else "H0"
        expected = str(row["expected_state"])
        if selected == expected:
            matched += 1
        adjudications.append(
            {
                "phase11_prefix": row["phase11_prefix"],
                "control_id": row["control_id"],
                "expected_state": expected,
                "selected_state": selected,
                "audit_ready": audit["audit_ready"],
            }
        )

    return {
        "panel_size": len(rows),
        "audit_ready_count": sum(1 for audit in audits if audit["audit_ready"]),
        "expected_state_match_count": matched,
        "expected_state_match_rate": round(matched / len(audits), 6) if audits else None,
        "thresholds": dict(registry.get("metadata", {}).get("thresholds") or {}),
        "state_profiles": state_profiles,
        "adjudications": adjudications,
    }


def build_binding_dev_registry(
    rows: list[dict[str, Any]],
    *,
    version: str = "v5_binding_dev",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    grouped = _group_rows_by_expected_state(rows)
    m2_positive_floor = _threshold_between(
        _state_mean(grouped, "H0", "m2_advantage_mean"),
        _min_state_mean(grouped, ("H1", "H2"), "m2_advantage_mean"),
        default=0.05,
    )
    thresholds = {
        "gen_h0_h1_boundary": _threshold_between(
            _state_mean(grouped, "H0", "b0_gen_clause_f1"),
            _state_mean(grouped, "H1", "b0_gen_clause_f1"),
            default=0.05,
        ),
        "gen_h1_h2_boundary": _threshold_between(
            _state_mean(grouped, "H1", "b0_gen_clause_f1"),
            _state_mean(grouped, "H2", "b0_gen_clause_f1"),
            default=0.5,
        ),
        "role_presence_boundary": _threshold_between(
            _state_mean(grouped, "H0", "m1_semantic_role_accuracy"),
            max(
                _state_mean(grouped, "H1", "m1_semantic_role_accuracy") or 0.0,
                _state_mean(grouped, "H2", "m1_semantic_role_accuracy") or 0.0,
            ),
            default=0.55,
        ),
        "m2_positive_boundary": m2_positive_floor,
        "m3_use_boundary": _threshold_between(
            _max_state_mean(grouped, ("H0", "H1"), "m3_selective_impairment_clause_f1"),
            _state_mean(grouped, "H2", "m3_selective_impairment_clause_f1"),
            default=0.05,
        ),
        "m4_rescue_boundary": _threshold_between(
            _state_mean(grouped, "H0", "m4_advantage_mean"),
            _state_mean(grouped, "H1", "m4_advantage_mean"),
            default=0.03,
        ),
    }

    metadata = {
        "family": "G1_binding",
        "development_controls": [
            {
                "control_id": row["control_id"],
                "phase11_prefix": row["phase11_prefix"],
                "expected_state": row["expected_state"],
            }
            for row in rows
        ],
        "thresholds": thresholds,
        "notes": list(notes or []),
        "no_forced_h2": True,
        "calibration_strategy": (
            "Binding Step 3 uses behavior to separate H2 from H1, while M2 acts as a positive "
            "floor that rules out H0. This avoids assuming that stronger M2 magnitude is "
            "monotonic from H1 to H2."
        ),
    }

    return {
        "version": version,
        "metadata": metadata,
        "audit_preconditions": [
            {
                "signal": "binding_dev_audit_ready",
                "op": "==",
                "value": True,
                "description": "Step 3 requires state-aware anchors plus observable protocol modules.",
            },
        ],
        "signal_definitions": BINDING_SIGNAL_DEFINITIONS,
        "hypotheses": [
            {
                "id": "G1_binding",
                "label": "Binding Hypothesis",
                "description": "Role-filler bindings are present and can be absent, present-but-unused, or present-and-used.",
                "states": [
                    {
                        "id": "H2",
                        "meaning": "Binding structure is present and causally used for generation.",
                        "support": {
                            "all": [
                                {"signal": "state_anchor_ready", "op": "==", "value": True},
                                {
                                    "signal": "b0_gen_clause_f1",
                                    "op": ">=",
                                    "value": thresholds["gen_h1_h2_boundary"],
                                },
                                {
                                    "signal": "m1_semantic_role_accuracy",
                                    "op": ">=",
                                    "value": thresholds["role_presence_boundary"],
                                },
                                {"signal": "m2_observed", "op": "==", "value": True},
                                {
                                    "signal": "m2_advantage_mean",
                                    "op": ">=",
                                    "value": thresholds["m2_positive_boundary"],
                                },
                            ]
                        },
                        "falsifiers": {
                            "any": [
                                {"signal": "state_anchor_ready", "op": "==", "value": False},
                                {"signal": "m2_observed", "op": "==", "value": False},
                                {
                                    "signal": "m2_advantage_mean",
                                    "op": "<",
                                    "value": thresholds["m2_positive_boundary"],
                                },
                            ]
                        },
                    },
                    {
                        "id": "H1",
                        "meaning": "Binding structure is present but not cleanly used for generation.",
                        "support": {
                            "all": [
                                {"signal": "state_anchor_ready", "op": "==", "value": True},
                                {
                                    "signal": "b0_gen_clause_f1",
                                    "op": "<",
                                    "value": thresholds["gen_h1_h2_boundary"],
                                },
                                {
                                    "signal": "m1_semantic_role_accuracy",
                                    "op": ">=",
                                    "value": thresholds["role_presence_boundary"],
                                },
                                {"signal": "m2_observed", "op": "==", "value": True},
                                {
                                    "signal": "m2_advantage_mean",
                                    "op": ">=",
                                    "value": thresholds["m2_positive_boundary"],
                                },
                            ]
                        },
                        "falsifiers": {
                            "any": [
                                {
                                    "signal": "m1_semantic_role_accuracy",
                                    "op": "<",
                                    "value": thresholds["role_presence_boundary"],
                                },
                                {"signal": "m2_observed", "op": "==", "value": False},
                                {
                                    "signal": "m2_advantage_mean",
                                    "op": "<",
                                    "value": thresholds["m2_positive_boundary"],
                                },
                            ],
                            "all": [
                                {
                                    "signal": "b0_gen_clause_f1",
                                    "op": ">=",
                                    "value": thresholds["gen_h1_h2_boundary"],
                                },
                                {"signal": "m2_observed", "op": "==", "value": True},
                                {
                                    "signal": "m2_advantage_mean",
                                    "op": ">=",
                                    "value": thresholds["m2_positive_boundary"],
                                },
                            ],
                        },
                    },
                    {
                        "id": "H0",
                        "meaning": "Binding structure is absent or not behaviorally relevant.",
                        "support": {
                            "all": [
                                {"signal": "state_anchor_ready", "op": "==", "value": True},
                                {
                                    "signal": "b0_gen_clause_f1",
                                    "op": "<=",
                                    "value": thresholds["gen_h0_h1_boundary"],
                                },
                            ],
                            "any": [
                                {
                                    "signal": "m1_semantic_role_accuracy",
                                    "op": "<",
                                    "value": thresholds["role_presence_boundary"],
                                },
                                {"signal": "m2_observed", "op": "==", "value": False},
                                {
                                    "signal": "m2_advantage_mean",
                                    "op": "<",
                                    "value": thresholds["m2_positive_boundary"],
                                },
                            ],
                        },
                    },
                ],
            }
        ],
    }


def summarize_binding_calibration(
    rows: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    registry: dict[str, Any],
) -> dict[str, Any]:
    by_prefix = {row["phase11_prefix"]: row for row in rows}
    state_profiles = []
    for state_id in ("H0", "H1", "H2"):
        state_rows = [row for row in rows if row["expected_state"] == state_id]
        state_profiles.append(
            {
                "state_id": state_id,
                "run_count": len(state_rows),
                "mean_b0_gen_clause_f1": _state_mean(_group_rows_by_expected_state(rows), state_id, "b0_gen_clause_f1"),
                "mean_m1_semantic_role_accuracy": _state_mean(
                    _group_rows_by_expected_state(rows), state_id, "m1_semantic_role_accuracy"
                ),
                "mean_m2_advantage_mean": _state_mean(_group_rows_by_expected_state(rows), state_id, "m2_advantage_mean"),
                "mean_m3_selective_impairment_clause_f1": _state_mean(
                    _group_rows_by_expected_state(rows), state_id, "m3_selective_impairment_clause_f1"
                ),
                "mean_m4_advantage_mean": _state_mean(_group_rows_by_expected_state(rows), state_id, "m4_advantage_mean"),
            }
        )

    adjudications = []
    matched = 0
    for audit in audits:
        row = by_prefix[audit["phase11_prefix"]]
        selected = str(audit["support_matrix"][0]["selected_state"]) if audit["support_matrix"] else "H0"
        expected = str(row["expected_state"])
        if selected == expected:
            matched += 1
        adjudications.append(
            {
                "phase11_prefix": row["phase11_prefix"],
                "control_id": row["control_id"],
                "expected_state": expected,
                "selected_state": selected,
                "audit_ready": audit["audit_ready"],
            }
        )

    return {
        "panel_size": len(rows),
        "audit_ready_count": sum(1 for audit in audits if audit["audit_ready"]),
        "expected_state_match_count": matched,
        "expected_state_match_rate": round(matched / len(audits), 6) if audits else None,
        "thresholds": dict(registry.get("metadata", {}).get("thresholds") or {}),
        "state_profiles": state_profiles,
        "adjudications": adjudications,
    }


def _group_rows_by_expected_state(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"H0": [], "H1": [], "H2": []}
    for row in rows:
        grouped.setdefault(str(row["expected_state"]), []).append(row)
    return grouped


def _state_mean(grouped: dict[str, list[dict[str, Any]]], state_id: str, signal_name: str) -> float | None:
    values = [
        float(row["signals"][signal_name])
        for row in grouped.get(state_id, [])
        if isinstance(row["signals"].get(signal_name), (int, float)) and not isinstance(row["signals"].get(signal_name), bool)
    ]
    if not values:
        return None
    return round(mean(values), 6)


def _max_state_mean(grouped: dict[str, list[dict[str, Any]]], state_ids: tuple[str, ...], signal_name: str) -> float | None:
    values = [_state_mean(grouped, state_id, signal_name) for state_id in state_ids]
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(max(numeric), 6)


def _min_state_mean(grouped: dict[str, list[dict[str, Any]]], state_ids: tuple[str, ...], signal_name: str) -> float | None:
    values = [_state_mean(grouped, state_id, signal_name) for state_id in state_ids]
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(min(numeric), 6)


def _threshold_between(low: float | None, high: float | None, *, default: float) -> float:
    if low is None and high is None:
        return round(default, 6)
    if low is None:
        return round(max(default, float(high) * 0.5), 6)
    if high is None:
        return round(max(default, float(low)), 6)
    return round((float(low) + float(high)) / 2.0, 6)
