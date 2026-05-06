from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from utils.io import read_jsonl


def load_json(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_geometry_s1_failure(
    *,
    dev_summary: dict[str, Any],
    heldout_summary: dict[str, Any],
    target_summary: dict[str, Any],
    dev_predictions: list[dict[str, Any]],
    heldout_predictions: list[dict[str, Any]],
    target_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    dev_group_targets = dict(dev_summary.get("group_targets") or {})
    heldout_expected = summarize_expected_state_groups(heldout_predictions)
    h1_failure_table = summarize_h1_failure_modes(heldout_predictions, target_predictions)

    dev_rescue_gap_h1_h0 = _safe_gap(dev_group_targets, "H1", "H0", "rescue")
    dev_necessity_range = _safe_range(dev_group_targets, "necessity")
    heldout_rescue_gap_h1_h0 = _safe_gap(
        heldout_expected, "H1", "H0", "rescue_score_mean", default=None
    )
    heldout_necessity_range = _safe_range(
        heldout_expected,
        "necessity_score_mean",
        states=("H0", "H1", "H2"),
        default=None,
    )

    flatness_flags = {
        "behavior_deemphasized": (
            float(dev_summary.get("behavior_feature_share") or 0.0) <= 0.05
            and abs(float(dev_summary.get("fit_without_behavior_accuracy_delta") or 0.0)) <= 0.05
        ),
        "dev_rescue_flat_h0_vs_h1": dev_rescue_gap_h1_h0 is not None and abs(dev_rescue_gap_h1_h0) <= 0.05,
        "dev_necessity_flat": dev_necessity_range is not None and dev_necessity_range <= 0.05,
        "heldout_rescue_flat_h0_vs_h1": (
            heldout_rescue_gap_h1_h0 is not None and abs(heldout_rescue_gap_h1_h0) <= 0.05
        ),
        "heldout_necessity_flat": (
            heldout_necessity_range is not None and heldout_necessity_range <= 0.05
        ),
        "heldout_h1_recall_zero": (
            float(((heldout_summary.get("per_state_metrics") or {}).get("H1") or {}).get("recall") or 0.0) == 0.0
        ),
        "target_panel_has_no_h1": int((target_summary.get("selected_state_counts") or {}).get("H1") or 0) == 0,
    }

    primary_bottlenecks = []
    if flatness_flags["dev_rescue_flat_h0_vs_h1"] and flatness_flags["heldout_rescue_flat_h0_vs_h1"]:
        primary_bottlenecks.append("m4_rescue_flat_for_g2")
    if flatness_flags["dev_necessity_flat"] and flatness_flags["heldout_necessity_flat"]:
        primary_bottlenecks.append("m3_necessity_flat_for_g2")
    if flatness_flags["heldout_h1_recall_zero"] and flatness_flags["target_panel_has_no_h1"]:
        primary_bottlenecks.append("no_observable_g2_h1_manifold_under_s1")
    if not primary_bottlenecks:
        primary_bottlenecks.append("mixed_or_unresolved")

    stop_recommendation = {
        "should_block_s2_expansion": True,
        "recommended_next_stage": "diagnose_and_redesign_g2_m3_m4",
        "recommended_actions": [
            "audit strongest-control definitions for rescue swallowing",
            "redesign G2 rescue away from span-style patching toward geometry-specific causal units",
            "redesign G2 necessity metrics away from flat clause-f1 impairment signatures",
        ],
    }

    return {
        "diagnostic_name": "phase14_geometry_s1_failure_review",
        "family": "G2_distributed_geometry",
        "scope": "phase14_s1",
        "dev_panel_name": dev_summary.get("panel_name"),
        "heldout_panel_name": heldout_summary.get("panel_name"),
        "target_panel_name": target_summary.get("panel_name"),
        "headline": (
            "S1 successfully removed behavior dominance, but G2-H1 still fails because "
            "rescue and necessity are effectively flat across H0/H1/H2."
        ),
        "key_metrics": {
            "behavior_feature_share": dev_summary.get("behavior_feature_share"),
            "fit_without_behavior_accuracy_delta": dev_summary.get("fit_without_behavior_accuracy_delta"),
            "heldout_accuracy": heldout_summary.get("accuracy"),
            "heldout_macro_f1": heldout_summary.get("macro_f1"),
            "heldout_h1_recall": ((heldout_summary.get("per_state_metrics") or {}).get("H1") or {}).get("recall"),
            "heldout_underdetermined_rate": heldout_summary.get("underdetermined_rate"),
            "target_h1_count": (target_summary.get("selected_state_counts") or {}).get("H1", 0),
        },
        "group_target_diagnostics": {
            "dev_group_targets": dev_group_targets,
            "heldout_expected_state_group_means": heldout_expected,
            "dev_rescue_gap_h1_minus_h0": dev_rescue_gap_h1_h0,
            "dev_necessity_range": dev_necessity_range,
            "heldout_rescue_gap_h1_minus_h0": heldout_rescue_gap_h1_h0,
            "heldout_necessity_range": heldout_necessity_range,
        },
        "flatness_flags": flatness_flags,
        "primary_bottlenecks": primary_bottlenecks,
        "h1_failure_modes": h1_failure_table,
        "stop_recommendation": stop_recommendation,
    }


def summarize_expected_state_groups(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        expected = row.get("expected_state")
        if not expected:
            continue
        grouped.setdefault(str(expected), []).append(row)

    summary: dict[str, dict[str, Any]] = {}
    for state, rows in grouped.items():
        summary[state] = {
            "count": len(rows),
            "selected_state_counts": dict(Counter(str(row.get("selected_state")) for row in rows)),
            "presence_score_mean": _mean_nested(rows, "evidence_flags", "presence_score"),
            "interchange_score_mean": _mean_nested(rows, "evidence_flags", "interchange_score"),
            "necessity_score_mean": _mean_nested(rows, "evidence_flags", "necessity_score"),
            "rescue_score_mean": _mean_nested(rows, "evidence_flags", "rescue_score"),
            "realism_score_mean": _mean_nested(rows, "evidence_flags", "realism_score"),
            "behavior_prior_mean": _mean_nested(rows, "group_scores", "behavior_prior", "score"),
            "rescue_observed_rate": _rate_nested(rows, "group_scores", "rescue", "observed"),
            "necessity_observed_rate": _rate_nested(rows, "group_scores", "necessity", "observed"),
        }
    return summary


def summarize_h1_failure_modes(
    heldout_predictions: list[dict[str, Any]],
    target_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    heldout_h1 = [row for row in heldout_predictions if str(row.get("expected_state")) == "H1"]
    target_h1 = [
        row
        for row in target_predictions
        if str(row.get("panel_regime") or "").startswith("h1")
    ]
    return {
        "heldout_h1_controls": _summarize_h1_rows(heldout_h1, source="heldout"),
        "target_h1_like_runs": _summarize_h1_rows(target_h1, source="target"),
    }


def build_failure_row_table(
    heldout_predictions: list[dict[str, Any]],
    target_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, items in (
        ("heldout", [row for row in heldout_predictions if str(row.get("expected_state")) == "H1"]),
        ("target", [row for row in target_predictions if str(row.get("panel_regime") or "").startswith("h1")]),
    ):
        for row in items:
            rows.append(
                {
                    "source": source,
                    "id": row.get("control_id") or row.get("target_id"),
                    "phase11_prefix": row.get("phase11_prefix"),
                    "expected_state": row.get("expected_state"),
                    "panel_regime": row.get("panel_regime"),
                    "selected_state": row.get("selected_state"),
                    "decision_reasons": list(row.get("decision_reasons") or []),
                    "presence_score": _nested_value(row, "evidence_flags", "presence_score"),
                    "interchange_score": _nested_value(row, "evidence_flags", "interchange_score"),
                    "necessity_score": _nested_value(row, "evidence_flags", "necessity_score"),
                    "rescue_score": _nested_value(row, "evidence_flags", "rescue_score"),
                    "realism_score": _nested_value(row, "evidence_flags", "realism_score"),
                    "rescue_observed": _nested_value(row, "group_scores", "rescue", "observed"),
                    "m4_advantage_mean": _nested_value(row, "signals", "m4_advantage_mean"),
                    "m3_selective_impairment_clause_f1": _nested_value(
                        row, "signals", "m3_selective_impairment_clause_f1"
                    ),
                }
            )
    return rows


def render_geometry_s1_failure_markdown(summary: dict[str, Any]) -> str:
    flatness = summary["flatness_flags"]
    metrics = summary["key_metrics"]
    groups = summary["group_target_diagnostics"]
    h1_modes = summary["h1_failure_modes"]

    lines = [
        "# Phase 14 Geometry S1 Failure Review",
        "",
        "## Headline",
        "",
        summary["headline"],
        "",
        "## Key Metrics",
        "",
        f"- `behavior_feature_share = {metrics['behavior_feature_share']}`",
        f"- `fit_without_behavior_accuracy_delta = {metrics['fit_without_behavior_accuracy_delta']}`",
        f"- `heldout_accuracy = {metrics['heldout_accuracy']}`",
        f"- `heldout_macro_f1 = {metrics['heldout_macro_f1']}`",
        f"- `heldout_h1_recall = {metrics['heldout_h1_recall']}`",
        f"- `heldout_underdetermined_rate = {metrics['heldout_underdetermined_rate']}`",
        f"- `target_h1_count = {metrics['target_h1_count']}`",
        "",
        "## Flatness Evidence",
        "",
        f"- `dev_rescue_gap_h1_minus_h0 = {groups['dev_rescue_gap_h1_minus_h0']}`",
        f"- `heldout_rescue_gap_h1_minus_h0 = {groups['heldout_rescue_gap_h1_minus_h0']}`",
        f"- `dev_necessity_range = {groups['dev_necessity_range']}`",
        f"- `heldout_necessity_range = {groups['heldout_necessity_range']}`",
        "",
        "## Flags",
        "",
        f"- `behavior_deemphasized = {flatness['behavior_deemphasized']}`",
        f"- `dev_rescue_flat_h0_vs_h1 = {flatness['dev_rescue_flat_h0_vs_h1']}`",
        f"- `heldout_rescue_flat_h0_vs_h1 = {flatness['heldout_rescue_flat_h0_vs_h1']}`",
        f"- `dev_necessity_flat = {flatness['dev_necessity_flat']}`",
        f"- `heldout_necessity_flat = {flatness['heldout_necessity_flat']}`",
        f"- `heldout_h1_recall_zero = {flatness['heldout_h1_recall_zero']}`",
        f"- `target_panel_has_no_h1 = {flatness['target_panel_has_no_h1']}`",
        "",
        "## H1 Failure Modes",
        "",
        f"- Held-out `H1` selected states: `{h1_modes['heldout_h1_controls']['selected_state_counts']}`",
        f"- Held-out `H1` reasons: `{h1_modes['heldout_h1_controls']['decision_reason_counts']}`",
        f"- Target `H1-like` selected states: `{h1_modes['target_h1_like_runs']['selected_state_counts']}`",
        f"- Target `H1-like` reasons: `{h1_modes['target_h1_like_runs']['decision_reason_counts']}`",
        "",
        "## Interpretation",
        "",
        "- `S1` solved the old behavior-dominance issue, but it did not create a rescue-centered `G2-H1` manifold.",
        "- Current `M4` rescue is too flat to separate `H0` from `H1` for geometry.",
        "- Current `M3` necessity is also too flat to separate `H1` from `H2` for geometry.",
        "- Under this evidence, the next move should be a `G2` rescue/necessity redesign diagnostic, not `S2` family expansion.",
        "",
        "## Stop Recommendation",
        "",
        "- Block direct progression to `S2` on top of current `S1` semantics.",
        "- Redesign `G2` rescue around geometry-specific causal units rather than span-style patching.",
        "- Redesign `G2` necessity metrics before any new family-volume expansion.",
        "",
    ]
    return "\n".join(lines)


def _summarize_h1_rows(rows: list[dict[str, Any]], *, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "count": len(rows),
        "selected_state_counts": dict(Counter(str(row.get("selected_state")) for row in rows)),
        "decision_reason_counts": dict(
            Counter(reason for row in rows for reason in (row.get("decision_reasons") or []))
        ),
        "interchange_score_mean": _mean_nested(rows, "evidence_flags", "interchange_score"),
        "necessity_score_mean": _mean_nested(rows, "evidence_flags", "necessity_score"),
        "rescue_score_mean": _mean_nested(rows, "evidence_flags", "rescue_score"),
        "realism_score_mean": _mean_nested(rows, "evidence_flags", "realism_score"),
        "rescue_observed_rate": _rate_nested(rows, "group_scores", "rescue", "observed"),
        "m4_advantage_mean_mean": _mean_nested(rows, "signals", "m4_advantage_mean"),
        "m3_selective_impairment_clause_f1_mean": _mean_nested(
            rows, "signals", "m3_selective_impairment_clause_f1"
        ),
    }


def _mean_nested(rows: list[dict[str, Any]], *path: str) -> float | None:
    values: list[float] = []
    for row in rows:
        cursor = _nested_value(row, *path)
        if isinstance(cursor, bool) or cursor is None:
            continue
        if isinstance(cursor, (int, float)):
            values.append(float(cursor))
    if not values:
        return None
    return round(mean(values), 6)


def _rate_nested(rows: list[dict[str, Any]], *path: str) -> float | None:
    values: list[bool] = []
    for row in rows:
        cursor = _nested_value(row, *path)
        if isinstance(cursor, bool):
            values.append(cursor)
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 6)


def _nested_value(row: dict[str, Any], *path: str) -> Any:
    cursor: Any = row
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def _safe_gap(
    grouped: dict[str, dict[str, Any]],
    upper_state: str,
    lower_state: str,
    key: str,
    *,
    default: float | None = None,
) -> float | None:
    upper = ((grouped.get(upper_state) or {}).get(key))
    lower = ((grouped.get(lower_state) or {}).get(key))
    if upper is None or lower is None:
        return default
    return round(float(upper) - float(lower), 6)


def _safe_range(
    grouped: dict[str, dict[str, Any]],
    key: str,
    *,
    states: tuple[str, ...] = ("H0", "H1", "H2"),
    default: float | None = None,
) -> float | None:
    values = []
    for state in states:
        value = (grouped.get(state) or {}).get(key)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return default
    return round(max(values) - min(values), 6)


__all__ = [
    "build_failure_row_table",
    "load_json",
    "read_jsonl",
    "render_geometry_s1_failure_markdown",
    "summarize_expected_state_groups",
    "summarize_geometry_s1_failure",
    "summarize_h1_failure_modes",
]
