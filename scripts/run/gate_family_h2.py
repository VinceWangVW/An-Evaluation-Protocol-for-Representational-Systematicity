from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis import interaction_objects as p17
from analysis.family_closure import (
    BRIDGE_OBJECT_CLASS,
    QUERY_SOURCE_OBJECT_CLASS,
    evaluate_object_necessity,
    gate_h2_row,
    load_json,
    summarize_h2_gate,
)
from interventions.generation import GEOMETRY_INTERACTION_OBJECT_CLASS
from utils.io import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run family-closure H2 gates.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/family_closure_controls.json"),
    )
    parser.add_argument(
        "--observability-summary",
        default=str(PROJECT_ROOT / "reports/family_closure/observability_summary.json"),
    )
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def _class_artifact_path(object_class: str) -> Path:
    stem = {
        GEOMETRY_INTERACTION_OBJECT_CLASS: "phase18_c1_interaction_objects.jsonl",
        BRIDGE_OBJECT_CLASS: "phase18_c2_cross_step_bridge_objects.jsonl",
        QUERY_SOURCE_OBJECT_CLASS: "phase18_c3_query_source_relation_objects.jsonl",
    }[object_class]
    return PROJECT_ROOT / "artifacts/phase18" / stem


def _class_summary_path(object_class: str) -> Path:
    stem = {
        GEOMETRY_INTERACTION_OBJECT_CLASS: "phase18_c1_interaction_discovery_summary.json",
        BRIDGE_OBJECT_CLASS: "phase18_c2_cross_step_bridge_discovery_summary.json",
        QUERY_SOURCE_OBJECT_CLASS: "phase18_c3_query_source_relation_discovery_summary.json",
    }[object_class]
    return PROJECT_ROOT / "reports/family_closure" / stem


def _class_h2_summary_path(object_class: str) -> Path:
    stem = {
        GEOMETRY_INTERACTION_OBJECT_CLASS: "phase18_c1_interaction_h2_gate_summary.json",
        BRIDGE_OBJECT_CLASS: "phase18_c2_cross_step_bridge_h2_gate_summary.json",
        QUERY_SOURCE_OBJECT_CLASS: "phase18_c3_query_source_relation_h2_gate_summary.json",
    }[object_class]
    return PROJECT_ROOT / "reports/family_closure" / stem


def _blocked_summary(object_class: str, discovery_summary: dict[str, object]) -> dict[str, object]:
    return {
        "object_class": object_class,
        "gate_ready_candidate_count": 0,
        "passed_count": 0,
        "passed_candidate_ids": [],
        "rows": [],
        "decision": "T-supported",
        "decision_status": "final",
        "blocked": True,
        "blocked_reason": "discovery_rejected",
        "discovery_summary": discovery_summary,
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    observability_summary = load_json(args.observability_summary)
    if not bool(observability_summary.get("sealed")):
        raise SystemExit("Phase 18 observability is not sealed; H2 gates are blocked.")

    selected_aligner = str(observability_summary["selected_aligner"])
    class_order = (
        GEOMETRY_INTERACTION_OBJECT_CLASS,
        BRIDGE_OBJECT_CLASS,
        QUERY_SOURCE_OBJECT_CLASS,
    )
    dev_h2_bundle = p17._load_discovery_bundle(PROJECT_ROOT, PROJECT_ROOT / str(config["dev_controls"]["h2"]["discovery_summary"]))
    dev_h0_bundle = p17._load_discovery_bundle(PROJECT_ROOT, PROJECT_ROOT / str(config["dev_controls"]["h0"]["discovery_summary"]))
    held_h2_bundle = p17._load_discovery_bundle(PROJECT_ROOT, PROJECT_ROOT / str(config["heldout_controls"]["h2"]["discovery_summary"]))
    held_h0_bundle = p17._load_discovery_bundle(PROJECT_ROOT, PROJECT_ROOT / str(config["heldout_controls"]["h0"]["discovery_summary"]))
    dev_h2_adapter, layer_index = p17.build_adapter_from_bundle(dev_h2_bundle)
    dev_h0_adapter, _ = p17.build_adapter_from_bundle(dev_h0_bundle)
    held_h2_adapter, _ = p17.build_adapter_from_bundle(held_h2_bundle)
    held_h0_adapter, _ = p17.build_adapter_from_bundle(held_h0_bundle)

    for object_class in class_order:
        summary_path = _class_summary_path(object_class)
        h2_path = _class_h2_summary_path(object_class)
        discovery_summary = load_json(summary_path)
        catalog = read_jsonl(_class_artifact_path(object_class))
        if not bool(discovery_summary.get("accepted")):
            write_json(h2_path, _blocked_summary(object_class, discovery_summary))
            continue
        rows = []
        for candidate in catalog:
            dev_h2_eval = evaluate_object_necessity(
                object_class=object_class,
                adapter=dev_h2_adapter,
                layer_index=layer_index,
                object_spec=candidate,
                aligner_name=selected_aligner,
                candidate_records=dev_h2_bundle["candidate_records"],
                span_records=dev_h2_bundle["span_records"],
                max_length=args.max_length,
            )
            dev_h0_eval = evaluate_object_necessity(
                object_class=object_class,
                adapter=dev_h0_adapter,
                layer_index=layer_index,
                object_spec=candidate,
                aligner_name=selected_aligner,
                candidate_records=dev_h0_bundle["candidate_records"],
                span_records=dev_h0_bundle["span_records"],
                max_length=args.max_length,
            )
            held_h2_eval = evaluate_object_necessity(
                object_class=object_class,
                adapter=held_h2_adapter,
                layer_index=layer_index,
                object_spec=candidate,
                aligner_name=selected_aligner,
                candidate_records=held_h2_bundle["candidate_records"],
                span_records=held_h2_bundle["span_records"],
                max_length=args.max_length,
            )
            held_h0_eval = evaluate_object_necessity(
                object_class=object_class,
                adapter=held_h0_adapter,
                layer_index=layer_index,
                object_spec=candidate,
                aligner_name=selected_aligner,
                candidate_records=held_h0_bundle["candidate_records"],
                span_records=held_h0_bundle["span_records"],
                max_length=args.max_length,
            )
            rows.append(
                gate_h2_row(
                    object_spec=candidate,
                    dev_summary={
                        "h2_gap": dev_h2_eval["h2_gap"],
                        "h0_gap": dev_h0_eval["h0_gap"],
                        "actual_drop": dev_h2_eval["actual_drop"],
                        "strongest_control_drop": dev_h2_eval["strongest_control_drop"],
                        "realism_pass_rate": min(
                            float(dev_h2_eval["realism_pass_rate"]),
                            float(dev_h0_eval["realism_pass_rate"]),
                        ),
                    },
                    heldout_summary={
                        "h2_gap": held_h2_eval["h2_gap"],
                        "h0_gap": held_h0_eval["h0_gap"],
                        "actual_drop": held_h2_eval["actual_drop"],
                        "strongest_control_drop": held_h2_eval["strongest_control_drop"],
                        "realism_pass_rate": min(
                            float(held_h2_eval["realism_pass_rate"]),
                            float(held_h0_eval["realism_pass_rate"]),
                        ),
                    },
                )
            )
        write_json(
            h2_path,
            summarize_h2_gate(object_class=object_class, rows=rows, discovery_summary=discovery_summary),
        )


if __name__ == "__main__":
    main()
