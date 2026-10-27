"""Inspect the reported MS-047 trace without inventing unobserved evidence.

The released generator has no historical MS identifiers. Its index 47 is
exported separately, with its original exact IDs, and is never associated
with the published numerical trace. No model download is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from release_analysis import DEFAULT_PROFILE, adjudicate, score_feature


def reported_trace() -> dict:
    return {
        "trace_id": "MS-047",
        "status": "paper_reported",
        "source": "Supplied camera-ready source, Appendix End-to-End Walkthrough; pHtx rebuttal Q3",
        "source_date": "2026-10-27",
        "historical_inputs": None,
        "historical_input_status": "unavailable; chef/poet/praise example is not in released generator vocabulary",
        "model_checkpoint": None,
        "metrics": {
            "exact_match_count": 6, "item_count": 7, "clause_f1": 0.947,
            "role_readout": 0.938, "surface_control_gap": 0.071,
            "filler_control": None, "predicate_control": None,
            "interchange": 0.512, "interchange_strongest_control": 0.181,
            "interchange_remaining_controls": None,
            "necessity": 0.276, "necessity_global_control": 0.058,
            "repair": 0.464, "repair_strongest_control": 0.161,
            "repair_remaining_controls": None, "permutation_p_value": None,
            "realism": 0.714, "linkage": 0.571, "coverage": 0.857,
        },
        "reported_state_scores": {"Causal-use": 0.806, "Present-only": 0.592,
                                  "Unsupported": 0.221, "Undetermined": 0.184},
        "reported_margin": 0.214,
        "reported_state": "Causal-use",
        "raw_L1_status": "unavailable",
        "raw_L2_status": "unavailable; reported rounded state scores are retained above",
        "note": "This transcription is not a recovered experimental record or a newly measured result.",
    }


def reported_signals() -> dict:
    """Arithmetic on reported values; no invented p-value or control values."""
    return {
        "b0_gen_clause_f1": 0.947,
        "m1_semantic_role_accuracy": 0.938,
        "m1_surface_control_gap": 0.071,
        "m2_advantage_mean": 0.512 - 0.181,
        "m2_permutation_p_value": None,
        "m3_selective_impairment_clause_f1": 0.276 - 0.058,
        "m4_advantage_mean": 0.464 - 0.161,
        "realism_pass_rate": 0.714,
        "linkage_pass_rate": 0.571,
        "evidence_coverage": 0.857,
        "audit_ready": True,
        "state_anchor_ready": True,
    }


def build_walkthrough(output: Path, evidence_path: Path | None = None) -> dict:
    canonical = ROOT / "data/processed/phase11_systematicity_v2.jsonl"
    rows = [json.loads(line) for line in canonical.read_text().splitlines() if line]
    families = list(dict.fromkeys(row["metadata"]["counterpart_family_id"] for row in rows))
    family_id = families[47]
    release_rows = [row for row in rows if row["metadata"]["counterpart_family_id"] == family_id]
    assert len(release_rows) == 7
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        "status": "released_generator_index_example",
        "family_index": 47,
        "counterpart_family_id": family_id,
        "historical_MS_047_identity": "not established; do not attach the paper-reported metrics to these inputs",
        "source": canonical.relative_to(ROOT).as_posix(),
        "source_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
        "items": [{"item_id": row["item_id"], "counterpart_type": row["metadata"]["counterpart_type"],
                   "input_text": row["input_text"], "target_form": row["target_form"]} for row in release_rows],
    }
    raw = json.loads(evidence_path.read_text()) if evidence_path else {"signals": reported_signals()}
    signals = raw.get("signals", raw)
    l2 = adjudicate(signals)
    # Preserve every inspectable transformation even when another required
    # feature is absent and the full decision correctly abstains.
    l2["known_feature_transforms"] = {
        key: {"raw": signals[key], **spec, "score": score_feature(signals[key], **spec)}
        for key, spec in DEFAULT_PROFILE["features"].items()
        if isinstance(signals.get(key), (int, float)) and not isinstance(signals[key], bool)
    }
    l2["reported_control_arithmetic"] = {
        "interchange_advantage": 0.512 - 0.181,
        "selective_necessity_vs_global": 0.276 - 0.058,
        "repair_advantage": 0.464 - 0.161,
        "status": "arithmetic_on_rounded_paper_values; not newly measured effects",
    }
    l2["evidence_source"] = str(evidence_path) if evidence_path else "paper-reported rounded features; incomplete"
    l2["result_status"] = "new_decision_profile_recomputation; not a historical rerun"
    outputs = {
        "paper_reported_L1.json": reported_trace(),
        "released_index047_inputs.json": inputs,
        "recomputed_L2.json": l2,
    }
    for name, payload in outputs.items():
        (output / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return {"output_directory": str(output), "released_input_count": len(release_rows),
            "historical_MS_047_inputs": "unavailable", "recomputed_state": l2["selected_state"],
            "missing_features": l2.get("missing_features", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/walkthrough")
    parser.add_argument("--evidence", type=Path, help="Optional newly measured complete L1 signals; see docs/ANALYSIS_GUIDE.md")
    args = parser.parse_args()
    print(json.dumps(build_walkthrough(args.output_dir, args.evidence), indent=2))


if __name__ == "__main__":
    main()
