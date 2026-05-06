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
    build_cross_step_bridge_catalog,
    build_query_source_relation_catalog,
    load_json,
    summarize_family_catalog,
)
from interventions.generation import GEOMETRY_INTERACTION_OBJECT_CLASS
from utils.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build family-closure catalogs.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/family_closure_controls.json"),
    )
    parser.add_argument(
        "--observability-summary",
        default=str(PROJECT_ROOT / "reports/family_closure/observability_summary.json"),
    )
    return parser.parse_args()


def _trace_path(aligner_name: str) -> Path:
    return PROJECT_ROOT / f"artifacts/phase18/phase18_g2_control_traces_{aligner_name.lower()}.jsonl"


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


def _rows_by_control(rows: list[dict[str, object]], control_id: str) -> list[dict[str, object]]:
    return [dict(row) for row in rows if str(row.get("control_id")) == control_id]


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    observability_summary = load_json(args.observability_summary)
    if not bool(observability_summary.get("sealed")):
        raise SystemExit("Phase 18 observability is not sealed; catalog building is blocked.")

    selected_aligner = str(observability_summary["selected_aligner"])
    trace_rows = read_jsonl(_trace_path(selected_aligner))
    dev_h2_rows = _rows_by_control(trace_rows, str(config["dev_controls"]["h2"]["control_id"]))
    dev_h0_rows = _rows_by_control(trace_rows, str(config["dev_controls"]["h0"]["control_id"]))
    held_h2_rows = _rows_by_control(trace_rows, str(config["heldout_controls"]["h2"]["control_id"]))
    held_h0_rows = _rows_by_control(trace_rows, str(config["heldout_controls"]["h0"]["control_id"]))
    layer_index = int(config.get("layer_index") or 0)
    source_run = str(config["dev_controls"]["h2"]["control_id"])

    c1_catalog = p17.build_interaction_catalog(
        dev_h2_rows=dev_h2_rows,
        dev_h0_rows=dev_h0_rows,
        heldout_h2_rows=held_h2_rows,
        heldout_h0_rows=held_h0_rows,
        layer_index=layer_index,
        source_run=source_run,
    )
    c1_summary = dict(p17.summarize_interaction_catalog(c1_catalog))
    c1_summary["object_class"] = GEOMETRY_INTERACTION_OBJECT_CLASS
    c1_summary["aligner_used"] = selected_aligner
    c1_summary["reused_phase17"] = False
    write_jsonl(_class_artifact_path(GEOMETRY_INTERACTION_OBJECT_CLASS), c1_catalog)
    write_json(_class_summary_path(GEOMETRY_INTERACTION_OBJECT_CLASS), c1_summary)

    c2_catalog = build_cross_step_bridge_catalog(
        dev_h2_rows=dev_h2_rows,
        dev_h0_rows=dev_h0_rows,
        heldout_h2_rows=held_h2_rows,
        heldout_h0_rows=held_h0_rows,
        layer_index=layer_index,
        source_run=source_run,
    )
    c2_summary = summarize_family_catalog(catalog=c2_catalog, object_class=BRIDGE_OBJECT_CLASS)
    c2_summary["aligner_used"] = selected_aligner
    write_jsonl(_class_artifact_path(BRIDGE_OBJECT_CLASS), c2_catalog)
    write_json(_class_summary_path(BRIDGE_OBJECT_CLASS), c2_summary)

    c3_catalog = build_query_source_relation_catalog(
        dev_h2_rows=dev_h2_rows,
        dev_h0_rows=dev_h0_rows,
        heldout_h2_rows=held_h2_rows,
        heldout_h0_rows=held_h0_rows,
        layer_index=layer_index,
        source_run=source_run,
    )
    c3_summary = summarize_family_catalog(catalog=c3_catalog, object_class=QUERY_SOURCE_OBJECT_CLASS)
    c3_summary["aligner_used"] = selected_aligner
    write_jsonl(_class_artifact_path(QUERY_SOURCE_OBJECT_CLASS), c3_catalog)
    write_json(_class_summary_path(QUERY_SOURCE_OBJECT_CLASS), c3_summary)

    print(
        {
            "selected_aligner": selected_aligner,
            "c1_accepted": bool(c1_summary.get("accepted")),
            "c2_accepted": bool(c2_summary.get("accepted")),
            "c3_accepted": bool(c3_summary.get("accepted")),
        }
    )


if __name__ == "__main__":
    main()
