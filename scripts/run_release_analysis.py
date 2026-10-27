#!/usr/bin/env python3
"""Run implementation-facing analyses; expensive model runs require --execute."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from release_analysis import (
    DEFAULT_PROFILE, adjudicate, cluster_bootstrap, collect_evidence, component_rules,
    engineered_truth_controls, execute_plan, load_json, make_training_plan, read_rows,
    sensitivity, slog_adapt, transfer_compare, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="analysis", required=True)
    p = sub.add_parser("adjudicate", help="Score a complete signals JSON with full transformation trace")
    p.add_argument("--signals", required=True)
    p = sub.add_parser("sensitivity", help="Exhaust six three-level configuration blocks")
    p.add_argument("--panel", required=True)
    p.add_argument("--grid", default="configs/release/sensitivity_grid.json")
    p = sub.add_parser("bootstrap", help="Matched-set cluster bootstrap of measured per-set evidence")
    p.add_argument("--rows", required=True)
    p.add_argument("--replicates", type=int, default=1000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--expected-sets", type=int, default=96)
    p.add_argument("--permutations", type=int, default=199)
    p = sub.add_parser("component-rules", help="Compare three rules on supplied known-truth units")
    p.add_argument("--rows", required=True)
    p = sub.add_parser("truth-controls", help="Execute independent engineered linear fixture")
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    p.add_argument("--observations", type=int, default=128)
    p = sub.add_parser("seed-plan", help="Materialize and optionally execute five-seed model/evidence runs")
    p.add_argument("--model", default="google-t5/t5-small")
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    p.add_argument("--benchmark", default="phase11_systematicity_v2")
    p.add_argument("--plan-dir", default="runs/release/five_seeds")
    p.add_argument("--execute", action="store_true")
    p = sub.add_parser("transfer-plan", help="Materialize T5-base or SLOG-bound full evidence runs")
    p.add_argument("--model", default="google-t5/t5-base")
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    p.add_argument("--benchmark", default="phase11_systematicity_v2")
    p.add_argument("--plan-dir", default="runs/release/t5base")
    p.add_argument("--execute", action="store_true")
    p = sub.add_parser("collect-evidence", help="Export available L1 summaries without filling missing metrics")
    p.add_argument("--panel", required=True)
    p = sub.add_parser("transfer-compare", help="Compare frozen and control-only recalibrated rules")
    p.add_argument("--panel", required=True)
    p = sub.add_parser("slog-adapt", help="Validate explicitly selected original SLOG exact contrasts")
    p.add_argument("--source", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--expected-pairs", type=int, default=96)
    for p in sub.choices.values():
        p.add_argument("--output", required=True)
        p.add_argument("--profile", help="Optional explicit decision-profile JSON")
    args = parser.parse_args()
    if args.profile and args.analysis in ("seed-plan","transfer-plan","slog-adapt"):
        parser.error("--profile is not supported for dataset adaptation or model training plans")
    profile = load_json(args.profile) if args.profile else DEFAULT_PROFILE
    if args.analysis == "adjudicate":
        result = adjudicate(load_json(args.signals), profile)
    elif args.analysis == "sensitivity":
        result = sensitivity(load_json(args.panel), load_json(args.grid), profile)
    elif args.analysis == "bootstrap":
        result = cluster_bootstrap(read_rows(args.rows), replicates=args.replicates, seed=args.seed,
            expected_sets=args.expected_sets, cfg=profile, permutations=args.permutations)
    elif args.analysis == "component-rules":
        result = component_rules(read_rows(args.rows), profile)
    elif args.analysis == "truth-controls":
        if args.profile:
            parser.error("truth-controls freezes its own explicit fixture profile; --profile is unsupported")
        result = engineered_truth_controls(args.seeds, observations=args.observations)
    elif args.analysis in ("seed-plan", "transfer-plan"):
        result = make_training_plan(ROOT, Path(args.plan_dir).resolve(), model=args.model,
            seeds=args.seeds, benchmark=args.benchmark)
        if args.execute:
            result = execute_plan(result, ROOT)
    elif args.analysis == "collect-evidence":
        result = collect_evidence(ROOT, Path(args.panel), profile)
    elif args.analysis == "transfer-compare":
        panel = load_json(args.panel)
        result = transfer_compare(panel["calibration"], panel["targets"], profile)
    else:
        selection = load_json(args.selection)
        actual_hash = hashlib.sha256(Path(args.source).read_bytes()).hexdigest()
        if selection.get("source_sha256") != actual_hash:
            parser.error("SLOG source file does not match the pinned selection SHA256")
        source_rows = read_rows(args.source)
        if not any(r.get("split") == "train" for r in source_rows) or not any(r.get("split") == "dev" for r in source_rows):
            parser.error("Full SLOG adaptation requires original training and development-control rows alongside selected evaluation pairs")
        result = slog_adapt(source_rows, selection, expected_pairs=args.expected_pairs)
        # Retain original rows in the existing model pipeline's canonical schema.
        canonical = {row["item_id"]:dict(row) for row in source_rows if row.get("split") in ("train","dev")}
        for pair in result["pairs"]:
            for side in ("source", "target"):
                row = dict(pair[side])
                row.setdefault("benchmark_name", "slog_selected")
                row.setdefault("raw_split", row["split"])
                row.setdefault("generalization_case", "prespecified_exact_contrast")
                row.setdefault("lexical_signature", [])
                row.setdefault("surface_features", {})
                row.setdefault("metadata", {})
                row["metadata"] = {**row["metadata"], "source_revision": selection["source_revision"],
                                   "selected_pair_id": pair["pair_id"],
                                   "counterpart_family_id": pair["pair_id"], "counterpart_type":side}
                row["role_signature"] = {"frame":"parsed_from_target",**row["role_signature"]}
                canonical[row["item_id"]] = row
        from data.canonicalize import infer_role_signature_from_target_form, infer_surface_features
        for row in canonical.values():
            row.setdefault("benchmark_name","slog_selected")
            row.setdefault("raw_split",row["split"])
            row.setdefault("generalization_case","prespecified_exact_contrast")
            row.setdefault("role_signature",infer_role_signature_from_target_form(row["target_form"]))
            row.setdefault("lexical_signature",[])
            row["surface_features"] = infer_surface_features(row["input_text"],row["target_form"])
            row.setdefault("metadata",{})
        canonical_path = Path(args.output).with_name(Path(args.output).stem+"_canonical.jsonl")
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        canonical_path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in canonical.values()), encoding="utf-8")
        result["canonical_rows"] = str(canonical_path)
    write_json(args.output, result)
    print(f"{args.analysis}: {result.get('status',result.get('selected_state','complete'))}; output={args.output}")


if __name__ == "__main__":
    main()
