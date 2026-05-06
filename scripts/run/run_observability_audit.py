from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.interaction_objects import _load_discovery_bundle
from analysis.family_closure import ALIGNER_NAMES, load_json, summarize_observability_audit, trace_control_bundle_with_aligner
from utils.io import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the family-closure observability audit.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/family_closure_controls.json"),
    )
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def _trace_path(aligner_name: str) -> Path:
    suffix = aligner_name.lower()
    return PROJECT_ROOT / f"artifacts/phase18/phase18_g2_control_traces_{suffix}.jsonl"


def _alignment_path(aligner_name: str) -> Path:
    suffix = aligner_name.lower()
    return PROJECT_ROOT / f"artifacts/phase18/phase18_g2_control_alignments_{suffix}.jsonl"


def _unaligned_path(aligner_name: str) -> Path:
    suffix = aligner_name.lower()
    return PROJECT_ROOT / f"artifacts/phase18/phase18_g2_control_unaligned_{suffix}.jsonl"


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    controls = []
    for split_key in ("dev_controls", "heldout_controls"):
        for label in ("h2", "h0"):
            control = dict(config[split_key][label])
            control["split_key"] = split_key
            control["label"] = label
            controls.append(control)

    audit_rows = []
    for aligner_name in ALIGNER_NAMES:
        trace_rows = []
        alignment_rows = []
        unaligned_rows = []
        for control in controls:
            bundle = _load_discovery_bundle(PROJECT_ROOT, PROJECT_ROOT / str(control["discovery_summary"]))
            traced = trace_control_bundle_with_aligner(
                bundle=bundle,
                aligner_name=aligner_name,
                max_length=args.max_length,
            )
            trace_rows.extend(list(traced["rows"]))
            alignment_rows.extend(list(traced["alignment_rows"]))
            unaligned_rows.extend(list(traced["unaligned_rows"]))
            audit_rows.append(
                {
                    "aligner": aligner_name,
                    "control_summary": dict(traced["summary"]),
                    "alignment_rows": list(traced["alignment_rows"]),
                }
            )
        write_jsonl(_trace_path(aligner_name), trace_rows)
        write_jsonl(_alignment_path(aligner_name), alignment_rows)
        write_jsonl(_unaligned_path(aligner_name), unaligned_rows)

    summary = summarize_observability_audit(audit_rows=audit_rows)
    summary["trace_paths"] = {
        aligner_name: str(_trace_path(aligner_name).relative_to(PROJECT_ROOT))
        for aligner_name in ALIGNER_NAMES
    }
    write_json(PROJECT_ROOT / "reports/family_closure/observability_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
