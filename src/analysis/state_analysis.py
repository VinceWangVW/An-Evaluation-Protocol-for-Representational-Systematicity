from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from analysis.phase11 import load_phase11_binding_row, load_phase11_geometry_row
from utils.io import read_jsonl


PATCH_CONTROL_KINDS = ("wrong_role", "non_role", "shuffled")


def load_phase12_binding_row(
    project_root: str | Path,
    phase11_prefix: str,
    *,
    control_id: str,
    expected_state: str,
    panel_role: str = "development",
) -> dict[str, Any]:
    row = load_phase11_binding_row(
        project_root,
        phase11_prefix,
        control_id=control_id,
        expected_state=expected_state,
        panel_role=panel_role,
    )
    return _augment_with_realism(project_root, row)


def load_phase12_geometry_row(
    project_root: str | Path,
    phase11_prefix: str,
    *,
    control_id: str,
    expected_state: str,
    panel_role: str = "development",
) -> dict[str, Any]:
    row = load_phase11_geometry_row(
        project_root,
        phase11_prefix,
        control_id=control_id,
        expected_state=expected_state,
        panel_role=panel_role,
    )
    return _augment_with_realism(project_root, row)


def _augment_with_realism(project_root: str | Path, row: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_root)
    phase11_prefix = str(row["phase11_prefix"])
    m2_path = root / "artifacts/protocol_runs" / f"{phase11_prefix}_m2_generation_interchange.jsonl"
    m4_path = root / "artifacts/protocol_runs" / f"{phase11_prefix}_m4_generation_rescue.jsonl"
    m2_records = read_jsonl(m2_path) if m2_path.exists() else []
    m4_records = read_jsonl(m4_path) if m4_path.exists() else []
    row = dict(row)
    row["signals"] = dict(row["signals"])
    row["signals"].update(summarize_phase11_realism(m2_records=m2_records, m4_records=m4_records))
    return row


def summarize_phase11_realism(
    *,
    m2_records: list[dict[str, Any]],
    m4_records: list[dict[str, Any]],
) -> dict[str, Any]:
    m2_summary = _summarize_realism_records(m2_records)
    m4_summary = _summarize_realism_records(m4_records)
    combined_cosine = _mean_available(
        [m2_summary["cosine_margin_mean"], m4_summary["cosine_margin_mean"]]
    )
    combined_bank = _mean_available(
        [m2_summary["bank_distance_margin_mean"], m4_summary["bank_distance_margin_mean"]]
    )
    combined_pass = _mean_available([m2_summary["pass_rate"], m4_summary["pass_rate"]])

    return {
        "m2_realism_record_count": m2_summary["count"],
        "m2_realism_cosine_margin_mean": m2_summary["cosine_margin_mean"],
        "m2_realism_bank_distance_margin_mean": m2_summary["bank_distance_margin_mean"],
        "m2_realism_pass_rate": m2_summary["pass_rate"],
        "m4_realism_record_count": m4_summary["count"],
        "m4_realism_cosine_margin_mean": m4_summary["cosine_margin_mean"],
        "m4_realism_bank_distance_margin_mean": m4_summary["bank_distance_margin_mean"],
        "m4_realism_pass_rate": m4_summary["pass_rate"],
        "realism_observed": bool((m2_summary["count"] or 0) + (m4_summary["count"] or 0) > 0),
        "realism_record_count": int(m2_summary["count"] or 0) + int(m4_summary["count"] or 0),
        "realism_cosine_margin_mean": combined_cosine,
        "realism_bank_distance_margin_mean": combined_bank,
        "realism_pass_rate": combined_pass,
    }


def summarize_state_signal_axes(signals: dict[str, Any]) -> dict[str, float | None]:
    return {
        "behavior": _as_float(signals.get("b0_gen_clause_f1")),
        "presence_role_accuracy": _as_float(signals.get("m1_semantic_role_accuracy")),
        "presence_filler_gap": _as_float(signals.get("m1_filler_gap")),
        "interchange": _as_float(signals.get("m2_advantage_mean")),
        "necessity": _as_float(signals.get("m3_selective_impairment_clause_f1")),
        "rescue": _as_float(signals.get("m4_advantage_mean")),
        "rescue_coverage": _as_float(signals.get("m5_fail_item_rescue_advantage_coverage")),
        "realism_pass_rate": _as_float(signals.get("realism_pass_rate")),
    }


def _summarize_realism_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    cosine_margins: list[float] = []
    bank_distance_margins: list[float] = []
    pass_flags: list[float] = []

    for record in records:
        actual_cosine = _as_float(record.get("actual_cosine_to_target"))
        actual_bank_distance = _as_float(record.get("actual_min_cosine_distance_to_target_bank"))
        if actual_cosine is None or actual_bank_distance is None:
            continue

        control_cosines = [
            value
            for value in (_as_float(record.get(f"{patch_kind}_cosine_to_target")) for patch_kind in PATCH_CONTROL_KINDS)
            if value is not None
        ]
        control_bank_distances = [
            value
            for value in (
                _as_float(record.get(f"{patch_kind}_min_cosine_distance_to_target_bank"))
                for patch_kind in PATCH_CONTROL_KINDS
            )
            if value is not None
        ]
        if not control_cosines or not control_bank_distances:
            continue

        strongest_control_cosine = max(control_cosines)
        strongest_control_bank_distance = min(control_bank_distances)
        cosine_margin = actual_cosine - strongest_control_cosine
        bank_distance_margin = strongest_control_bank_distance - actual_bank_distance
        cosine_margins.append(cosine_margin)
        bank_distance_margins.append(bank_distance_margin)
        pass_flags.append(1.0 if cosine_margin >= 0.0 and bank_distance_margin >= 0.0 else 0.0)

    return {
        "count": len(cosine_margins),
        "cosine_margin_mean": round(mean(cosine_margins), 6) if cosine_margins else None,
        "bank_distance_margin_mean": round(mean(bank_distance_margins), 6) if bank_distance_margins else None,
        "pass_rate": round(mean(pass_flags), 6) if pass_flags else None,
    }


def _mean_available(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return round(mean(usable), 6)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


__all__ = [
    "load_phase12_binding_row",
    "load_phase12_geometry_row",
    "summarize_state_signal_axes",
    "summarize_phase11_realism",
]
