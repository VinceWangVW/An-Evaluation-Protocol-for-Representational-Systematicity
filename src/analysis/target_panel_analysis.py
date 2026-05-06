from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from models.target_panels import TargetPanel, TargetSpec


def infer_surface_backend(
    project_root: str | Path,
    *,
    benchmark_name: str,
    fallback: str | None = None,
) -> str | None:
    summary_path = Path(project_root) / "reports/data_realization" / f"{benchmark_name}_realization_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        backend = payload.get("backend")
        if isinstance(backend, str) and backend:
            return backend
    return fallback


def build_target_design_rows(
    project_root: str | Path,
    panel: TargetPanel,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in panel.targets:
        benchmark_name = str(target.benchmark or panel.benchmark)
        surface_backend = infer_surface_backend(
            project_root,
            benchmark_name=benchmark_name,
            fallback=target.surface_backend or panel.surface_backend,
        )
        rows.append(
            {
                "target_id": target.target_id,
                "run_name": target.run_name,
                "phase11_prefix": target.phase11_prefix or target.run_name,
                "benchmark_name": benchmark_name,
                "prompt_style": str(target.prompt_style or panel.prompt_style),
                "panel_regime": str(target.panel_regime or "unspecified"),
                "surface_backend": surface_backend,
                "include_in_adjudication": bool(target.include_in_adjudication),
                "decoder_cross_attention_scale": target.decoder_cross_attention_scale,
                "train_limit": target.train_limit,
                "train_target_permutation_seed": target.train_target_permutation_seed,
                "encoder_geometry_transform": target.encoder_geometry_transform,
                "encoder_geometry_keep_ratio": target.encoder_geometry_keep_ratio,
                "reuse_checkpoint_from": target.reuse_checkpoint_from,
                "tags": list(target.tags),
                "notes": target.notes,
            }
        )
    return rows


def summarize_target_panel_design(
    panel: TargetPanel,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    adjudicated_rows = [row for row in rows if row["include_in_adjudication"]]
    panel_regime_counts = Counter(str(row["panel_regime"]) for row in adjudicated_rows)
    surface_backend_counts = Counter(str(row.get("surface_backend") or "unknown") for row in adjudicated_rows)
    benchmark_counts = Counter(str(row["benchmark_name"]) for row in adjudicated_rows)

    panel_regimes = set(panel_regime_counts)
    non_default_surface = sum(
        1
        for row in adjudicated_rows
        if (row.get("surface_backend") or panel.surface_backend) not in (panel.surface_backend, None)
    )
    if panel.surface_backend is None:
        non_default_surface = sum(
            1 for row in adjudicated_rows if str(row.get("surface_backend") or "unknown") != "template"
        )

    return {
        "panel_name": panel.panel_name,
        "hypothesis_family": panel.hypothesis_family,
        "row_loader": panel.row_loader,
        "target_count": len(rows),
        "adjudication_target_count": len(adjudicated_rows),
        "panel_regime_counts": dict(panel_regime_counts),
        "surface_backend_counts": dict(surface_backend_counts),
        "benchmark_counts": dict(benchmark_counts),
        "design_checks": {
            "has_strong_regime": "strong" in panel_regimes,
            "has_weak_regime": "weak_audit_ready" in panel_regimes,
            "has_h1_like_regime": any(regime.startswith("h1") for regime in panel_regimes),
            "has_surface_shifted_regime": "surface_shifted" in panel_regimes,
            "has_nondefault_surface_backend": non_default_surface > 0,
        },
        "notes": list(panel.metadata.get("notes") or []),
    }


def summarize_adjudicated_target_panel(
    *,
    panel_name: str,
    hypothesis_family: str | None,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_state_counts = Counter(str(row["selected_state"]) for row in predictions)
    target_regime_type_counts = Counter(str(row["gate"]["target_regime_type"]) for row in predictions)
    panel_regime_counts = Counter(str(row.get("panel_regime") or "unspecified") for row in predictions)
    surface_backend_counts = Counter(str(row.get("surface_backend") or "unknown") for row in predictions)
    decision_reason_counts = Counter(
        reason
        for row in predictions
        for reason in (row.get("decision_reasons") or [])
    )
    audit_ready_count = sum(1 for row in predictions if bool(row.get("gate", {}).get("audit_ready")))
    audit_ready_by_regime = Counter(
        str(row.get("panel_regime") or "unspecified")
        for row in predictions
        if bool(row.get("gate", {}).get("audit_ready"))
    )

    regime_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        regime_rows[str(row.get("panel_regime") or "unspecified")].append(row)

    module_score_means_by_regime: dict[str, dict[str, float | None]] = {}
    selected_state_counts_by_regime: dict[str, dict[str, int]] = {}
    for regime, rows in regime_rows.items():
        selected_state_counts_by_regime[regime] = dict(Counter(str(row["selected_state"]) for row in rows))
        module_score_means_by_regime[regime] = {
            "presence_score_mean": _mean_nested(rows, "evidence_flags", "presence_score"),
            "interchange_score_mean": _mean_nested(rows, "evidence_flags", "interchange_score"),
            "necessity_score_mean": _mean_nested(rows, "evidence_flags", "necessity_score"),
            "use_score_mean": _mean_nested(rows, "evidence_flags", "use_score"),
            "rescue_score_mean": _mean_nested(rows, "evidence_flags", "rescue_score"),
            "realism_score_mean": _mean_nested(rows, "evidence_flags", "realism_score"),
            "behavior_prior_mean": _mean_nested(rows, "group_scores", "behavior_prior", "score"),
            "top_posterior_mean": round(mean(float(row["top_posterior"]) for row in rows), 6) if rows else None,
        }

    distinct_selected_states = sorted(selected_state_counts)
    summary = {
        "panel_name": panel_name,
        "hypothesis_family": hypothesis_family,
        "run_count": len(predictions),
        "selected_state_counts": dict(selected_state_counts),
        "distinct_selected_states": distinct_selected_states,
        "distinct_selected_state_count": len(distinct_selected_states),
        "target_regime_type_counts": dict(target_regime_type_counts),
        "panel_regime_counts": dict(panel_regime_counts),
        "surface_backend_counts": dict(surface_backend_counts),
        "decision_reason_counts": dict(decision_reason_counts),
        "audit_ready_count": audit_ready_count,
        "audit_ready_rate": round(audit_ready_count / len(predictions), 6) if predictions else None,
        "audit_ready_by_panel_regime": dict(audit_ready_by_regime),
        "selected_state_counts_by_panel_regime": selected_state_counts_by_regime,
        "module_score_means_by_panel_regime": module_score_means_by_regime,
        "panel_checks": {
            "has_audit_ready_weak_target": bool(audit_ready_by_regime.get("weak_audit_ready")),
            "has_h1_like_target": any(str(regime).startswith("h1") for regime in panel_regime_counts),
            "has_surface_shifted_target": bool(panel_regime_counts.get("surface_shifted")),
            "has_three_or_more_selected_states": len(distinct_selected_states) >= 3,
        },
    }
    return summary


def _mean_nested(rows: list[dict[str, Any]], *path: str) -> float | None:
    values: list[float] = []
    for row in rows:
        cursor: Any = row
        for key in path:
            if not isinstance(cursor, dict):
                cursor = None
                break
            cursor = cursor.get(key)
        if isinstance(cursor, bool) or cursor is None:
            continue
        if isinstance(cursor, (int, float)):
            values.append(float(cursor))
    if not values:
        return None
    return round(mean(values), 6)


__all__ = [
    "build_target_design_rows",
    "infer_surface_backend",
    "summarize_adjudicated_target_panel",
    "summarize_target_panel_design",
]
