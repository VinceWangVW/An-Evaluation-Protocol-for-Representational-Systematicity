from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.family_closure import (
    BRIDGE_OBJECT_CLASS,
    QUERY_SOURCE_OBJECT_CLASS,
    case_condition_direct_h1,
    load_json,
    render_family_closure_status_markdown,
    summarize_family_closure,
)
from interventions.generation import GEOMETRY_INTERACTION_OBJECT_CLASS
from utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize the family-closure branch.")
    parser.add_argument(
        "--observability-summary",
        default=str(PROJECT_ROOT / "reports/family_closure/observability_summary.json"),
    )
    parser.add_argument(
        "--direct-h1-summary",
        default=str(PROJECT_ROOT / "reports/family_closure/direct_h1_summary.json"),
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    observability_summary = load_json(args.observability_summary)
    class_order = (
        GEOMETRY_INTERACTION_OBJECT_CLASS,
        BRIDGE_OBJECT_CLASS,
        QUERY_SOURCE_OBJECT_CLASS,
    )
    class_summaries = [load_json(_class_summary_path(object_class)) for object_class in class_order]
    h2_summaries = [load_json(_class_h2_summary_path(object_class)) for object_class in class_order]
    direct_h1_summary = load_json(args.direct_h1_summary)
    case_summary = case_condition_direct_h1(rescue_rows=list(direct_h1_summary.get("object_rows") or []))
    closure_summary = summarize_family_closure(
        observability_summary=observability_summary,
        class_summaries=class_summaries,
        h2_summaries=h2_summaries,
        direct_h1_summary=direct_h1_summary,
        case_summary=case_summary,
    )
    write_json(PROJECT_ROOT / "reports/family_closure/direct_h1_case_summary.json", case_summary)
    write_json(PROJECT_ROOT / "reports/family_closure/family_closure_summary.json", closure_summary)
    (PROJECT_ROOT / "reports/FAMILY_CLOSURE_STATUS.md").write_text(
        render_family_closure_status_markdown(
            observability_summary=observability_summary,
            class_summaries=class_summaries,
            h2_summaries=h2_summaries,
            direct_h1_summary=direct_h1_summary,
            case_summary=case_summary,
            closure_summary=closure_summary,
        ),
        encoding="utf-8",
    )
    print(closure_summary)


if __name__ == "__main__":
    main()
