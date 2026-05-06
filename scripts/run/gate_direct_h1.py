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
    _top_plausible_objects,
    evaluate_object_direct_h1,
    load_json,
    summarize_direct_h1,
)
from interventions.generation import GEOMETRY_INTERACTION_OBJECT_CLASS
from utils.io import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the family-closure direct H1 branch.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/family_closure_h1like.json"),
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


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    observability_summary = load_json(args.observability_summary)
    if not bool(observability_summary.get("sealed")):
        raise SystemExit("Phase 18 observability is not sealed; direct H1 is blocked.")

    selected_aligner = str(observability_summary["selected_aligner"])
    class_order = (
        GEOMETRY_INTERACTION_OBJECT_CLASS,
        BRIDGE_OBJECT_CLASS,
        QUERY_SOURCE_OBJECT_CLASS,
    )
    top_n = int(config.get("top_objects_per_class") or 2)
    catalog_by_class = {
        object_class: _top_plausible_objects(read_jsonl(_class_artifact_path(object_class)), limit=top_n)
        for object_class in class_order
    }

    object_rows = []
    for probe_run in list(config.get("probe_runs") or []):
        bundle = p17._load_protocol_bundle(PROJECT_ROOT, PROJECT_ROOT / str(probe_run["protocol_summary"]))
        adapter, layer_index = p17.build_adapter_from_bundle(bundle)
        for object_class, candidates in catalog_by_class.items():
            for candidate in candidates:
                object_rows.append(
                    evaluate_object_direct_h1(
                        object_class=object_class,
                        adapter=adapter,
                        layer_index=layer_index,
                        object_spec=candidate,
                        aligner_name=selected_aligner,
                        candidate_records=bundle["candidate_records"],
                        span_records=bundle["span_records"],
                        physical_family=str(probe_run["physical_family"]),
                        max_length=args.max_length,
                    )
                )
    summary = summarize_direct_h1(object_rows=object_rows)
    summary["object_rows"] = object_rows
    summary["selected_aligner"] = selected_aligner
    summary["catalog_object_ids"] = {
        object_class: [str(row["object_id"]) for row in candidates]
        for object_class, candidates in catalog_by_class.items()
    }
    write_json(PROJECT_ROOT / "reports/family_closure/direct_h1_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
