from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from discovery import build_matched_audit_sets, join_behavior_predictions, run_candidate_discovery, select_balanced_candidates
from models import build_model_adapter
from utils.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run state-aware discovery for a control run.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--behavior-run", required=True)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--family", default="seq2seq_lm", choices=["causal_lm", "seq2seq_lm"])
    parser.add_argument("--prompt-style", default=None, choices=["raw", "parser_instruction", "sentence_only"])
    parser.add_argument("--layers", nargs="+", type=int, default=[-1])
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--per-stratum", type=int, default=24)
    parser.add_argument("--match-set-limit", type=int, default=24)
    parser.add_argument("--candidate-strata", nargs="+", default=None)
    parser.add_argument("--match-required-strata", nargs="+", default=None)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--precision", default=None, choices=["auto", "fp32", "bf16", "fp16"])
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--decoder-cross-attention-scale", type=float, default=None)
    parser.add_argument("--decoder-cross-attention-keep-ratio", type=float, default=None)
    parser.add_argument("--decoder-cross-attention-layer-keep-ratio", type=float, default=None)
    parser.add_argument("--decoder-cross-attention-layer-seed", type=int, default=None)
    parser.add_argument("--decoder-cross-attention-layer-policy", default=None, choices=["random", "top", "bottom", "alternating", "even", "odd"])
    parser.add_argument("--decoder-cross-attention-bottleneck-keep-ratio", type=float, default=None)
    parser.add_argument("--decoder-cross-attention-bottleneck-seed", type=int, default=None)
    parser.add_argument("--decoder-cross-attention-bottleneck-mode", default=None, choices=["keep", "remove"])
    parser.add_argument("--decoder-cross-attention-relation-barrier-keep-ratio", type=float, default=None)
    parser.add_argument("--decoder-cross-attention-relation-barrier-seed", type=int, default=None)
    parser.add_argument("--decoder-cross-attention-relation-barrier-scale", type=float, default=None)
    parser.add_argument("--decoder-readout-keep-ratio", type=float, default=None)
    parser.add_argument("--decoder-readout-subspace-keep-ratio", type=float, default=None)
    parser.add_argument("--decoder-readout-subspace-seed", type=int, default=None)
    parser.add_argument("--decoder-readout-subspace-mode", default=None, choices=["keep", "remove"])
    parser.add_argument("--encoder-geometry-transform", default=None, choices=["identity", "orthogonal_preserve", "feature_mask", "gaussian_noise"])
    parser.add_argument("--encoder-geometry-seed", type=int, default=None)
    parser.add_argument("--encoder-geometry-keep-ratio", type=float, default=None)
    parser.add_argument("--encoder-geometry-noise-std", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    benchmark_records = read_jsonl(PROJECT_ROOT / "data/processed" / f"{args.benchmark}.jsonl")
    behavior_predictions = read_jsonl(PROJECT_ROOT / "runs/behavior_runs" / f"{args.behavior_run}_predictions.jsonl")
    behavior_summary = json.loads(
        (PROJECT_ROOT / "reports/behavior_runs" / f"{args.behavior_run}_summary.json").read_text(encoding="utf-8")
    )
    eval_examples = [record for record in benchmark_records if str(record["split"]) != "train"]
    joined = join_behavior_predictions(eval_examples, behavior_predictions)

    stratum_counts = Counter(str(record.get("stratum") or "unknown") for record in joined)
    candidate_strata, match_required_strata = infer_state_aware_strata(
        joined,
        candidate_strata=args.candidate_strata,
        match_required_strata=args.match_required_strata,
    )
    selected = select_balanced_candidates(
        joined,
        max_items_per_stratum=args.per_stratum,
        strata=tuple(candidate_strata),
        required_role_keys=("predicate", "agent", "theme"),
    )
    matched_sets = build_matched_audit_sets(
        joined,
        required_strata=tuple(match_required_strata),
        max_sets=args.match_set_limit,
    )
    if not selected:
        raise SystemExit("No state-aware candidates could be selected")

    model_name_or_path = args.model_name_or_path or str(
        behavior_summary.get("trained_model_path") or behavior_summary["model_name_or_path"]
    )
    prompt_style = args.prompt_style or str(behavior_summary.get("prompt_style") or "raw")
    adapter = build_model_adapter(
        model_name_or_path=model_name_or_path,
        family=args.family,
        prompt_style=prompt_style,
        precision=args.precision or str(behavior_summary.get("precision") or "auto"),
        num_beams=args.num_beams or int(behavior_summary.get("num_beams") or 1),
        decoder_cross_attention_scale=(
            args.decoder_cross_attention_scale
            if args.decoder_cross_attention_scale is not None
            else behavior_summary.get("decoder_cross_attention_scale")
        ),
        decoder_cross_attention_keep_ratio=(
            args.decoder_cross_attention_keep_ratio
            if args.decoder_cross_attention_keep_ratio is not None
            else behavior_summary.get("decoder_cross_attention_keep_ratio")
        ),
        decoder_cross_attention_layer_keep_ratio=(
            args.decoder_cross_attention_layer_keep_ratio
            if args.decoder_cross_attention_layer_keep_ratio is not None
            else behavior_summary.get("decoder_cross_attention_layer_keep_ratio")
        ),
        decoder_cross_attention_layer_seed=(
            args.decoder_cross_attention_layer_seed
            if args.decoder_cross_attention_layer_seed is not None
            else int(behavior_summary.get("decoder_cross_attention_layer_seed") or 0)
        ),
        decoder_cross_attention_layer_policy=(
            args.decoder_cross_attention_layer_policy
            if args.decoder_cross_attention_layer_policy is not None
            else str(behavior_summary.get("decoder_cross_attention_layer_policy") or "random")
        ),
        decoder_cross_attention_bottleneck_keep_ratio=(
            args.decoder_cross_attention_bottleneck_keep_ratio
            if args.decoder_cross_attention_bottleneck_keep_ratio is not None
            else behavior_summary.get("decoder_cross_attention_bottleneck_keep_ratio")
        ),
        decoder_cross_attention_bottleneck_seed=(
            args.decoder_cross_attention_bottleneck_seed
            if args.decoder_cross_attention_bottleneck_seed is not None
            else int(behavior_summary.get("decoder_cross_attention_bottleneck_seed") or 0)
        ),
        decoder_cross_attention_bottleneck_mode=(
            args.decoder_cross_attention_bottleneck_mode
            if args.decoder_cross_attention_bottleneck_mode is not None
            else str(behavior_summary.get("decoder_cross_attention_bottleneck_mode") or "keep")
        ),
        decoder_cross_attention_relation_barrier_keep_ratio=(
            args.decoder_cross_attention_relation_barrier_keep_ratio
            if args.decoder_cross_attention_relation_barrier_keep_ratio is not None
            else behavior_summary.get("decoder_cross_attention_relation_barrier_keep_ratio")
        ),
        decoder_cross_attention_relation_barrier_seed=(
            args.decoder_cross_attention_relation_barrier_seed
            if args.decoder_cross_attention_relation_barrier_seed is not None
            else int(behavior_summary.get("decoder_cross_attention_relation_barrier_seed") or 0)
        ),
        decoder_cross_attention_relation_barrier_scale=(
            args.decoder_cross_attention_relation_barrier_scale
            if args.decoder_cross_attention_relation_barrier_scale is not None
            else float(behavior_summary.get("decoder_cross_attention_relation_barrier_scale") or 0.5)
        ),
        decoder_readout_keep_ratio=(
            args.decoder_readout_keep_ratio
            if args.decoder_readout_keep_ratio is not None
            else behavior_summary.get("decoder_readout_keep_ratio")
        ),
        decoder_readout_subspace_keep_ratio=(
            args.decoder_readout_subspace_keep_ratio
            if args.decoder_readout_subspace_keep_ratio is not None
            else behavior_summary.get("decoder_readout_subspace_keep_ratio")
        ),
        decoder_readout_subspace_seed=(
            args.decoder_readout_subspace_seed
            if args.decoder_readout_subspace_seed is not None
            else int(behavior_summary.get("decoder_readout_subspace_seed") or 0)
        ),
        decoder_readout_subspace_mode=(
            args.decoder_readout_subspace_mode
            if args.decoder_readout_subspace_mode is not None
            else str(behavior_summary.get("decoder_readout_subspace_mode") or "remove")
        ),
        encoder_geometry_transform=(
            args.encoder_geometry_transform
            if args.encoder_geometry_transform is not None
            else behavior_summary.get("encoder_geometry_transform")
        ),
        encoder_geometry_seed=(
            args.encoder_geometry_seed
            if args.encoder_geometry_seed is not None
            else int(behavior_summary.get("encoder_geometry_seed") or 0)
        ),
        encoder_geometry_keep_ratio=(
            args.encoder_geometry_keep_ratio
            if args.encoder_geometry_keep_ratio is not None
            else behavior_summary.get("encoder_geometry_keep_ratio")
        ),
        encoder_geometry_noise_std=(
            args.encoder_geometry_noise_std
            if args.encoder_geometry_noise_std is not None
            else behavior_summary.get("encoder_geometry_noise_std")
        ),
    )
    span_records, summary = run_candidate_discovery(
        adapter=adapter,
        candidates=selected,
        layer_indices=list(args.layers),
        max_length=args.max_length,
    )

    run_name = args.output_prefix or build_run_name(args.behavior_run, model_name_or_path, list(args.layers), args.per_stratum)
    write_jsonl(PROJECT_ROOT / "artifacts/discovery_runs" / f"{run_name}_candidate_table.jsonl", selected)
    write_jsonl(PROJECT_ROOT / "artifacts/discovery_runs" / f"{run_name}_matched_sets.jsonl", matched_sets)
    write_jsonl(PROJECT_ROOT / "artifacts/discovery_runs" / f"{run_name}_span_vectors.jsonl", span_records)
    write_json(
        PROJECT_ROOT / "reports/discovery_runs" / f"{run_name}_summary.json",
        {
            "benchmark_name": args.benchmark,
            "behavior_run_name": args.behavior_run,
            "trained_model_path": behavior_summary.get("trained_model_path"),
            "behavior_summary": behavior_summary.get("evaluation"),
            "available_behavior_strata": dict(sorted(stratum_counts.items())),
            "candidate_strata": candidate_strata,
            "match_required_strata": match_required_strata,
            "layers_requested": list(args.layers),
            "matched_set_count": len(matched_sets),
            "precision": args.precision or behavior_summary.get("precision"),
            "num_beams": args.num_beams or behavior_summary.get("num_beams"),
            "decoder_cross_attention_scale": (
                args.decoder_cross_attention_scale
                if args.decoder_cross_attention_scale is not None
                else behavior_summary.get("decoder_cross_attention_scale")
            ),
            "decoder_cross_attention_keep_ratio": (
                args.decoder_cross_attention_keep_ratio
                if args.decoder_cross_attention_keep_ratio is not None
                else behavior_summary.get("decoder_cross_attention_keep_ratio")
            ),
            "decoder_cross_attention_layer_keep_ratio": (
                args.decoder_cross_attention_layer_keep_ratio
                if args.decoder_cross_attention_layer_keep_ratio is not None
                else behavior_summary.get("decoder_cross_attention_layer_keep_ratio")
            ),
            "decoder_cross_attention_layer_seed": (
                args.decoder_cross_attention_layer_seed
                if args.decoder_cross_attention_layer_seed is not None
                else behavior_summary.get("decoder_cross_attention_layer_seed")
            ),
            "decoder_cross_attention_layer_policy": (
                args.decoder_cross_attention_layer_policy
                if args.decoder_cross_attention_layer_policy is not None
                else behavior_summary.get("decoder_cross_attention_layer_policy")
            ),
            "decoder_cross_attention_bottleneck_keep_ratio": (
                args.decoder_cross_attention_bottleneck_keep_ratio
                if args.decoder_cross_attention_bottleneck_keep_ratio is not None
                else behavior_summary.get("decoder_cross_attention_bottleneck_keep_ratio")
            ),
            "decoder_cross_attention_bottleneck_seed": (
                args.decoder_cross_attention_bottleneck_seed
                if args.decoder_cross_attention_bottleneck_seed is not None
                else behavior_summary.get("decoder_cross_attention_bottleneck_seed")
            ),
            "decoder_cross_attention_bottleneck_mode": (
                args.decoder_cross_attention_bottleneck_mode
                if args.decoder_cross_attention_bottleneck_mode is not None
                else behavior_summary.get("decoder_cross_attention_bottleneck_mode")
            ),
            "decoder_cross_attention_relation_barrier_keep_ratio": (
                args.decoder_cross_attention_relation_barrier_keep_ratio
                if args.decoder_cross_attention_relation_barrier_keep_ratio is not None
                else behavior_summary.get("decoder_cross_attention_relation_barrier_keep_ratio")
            ),
            "decoder_cross_attention_relation_barrier_seed": (
                args.decoder_cross_attention_relation_barrier_seed
                if args.decoder_cross_attention_relation_barrier_seed is not None
                else behavior_summary.get("decoder_cross_attention_relation_barrier_seed")
            ),
            "decoder_cross_attention_relation_barrier_scale": (
                args.decoder_cross_attention_relation_barrier_scale
                if args.decoder_cross_attention_relation_barrier_scale is not None
                else behavior_summary.get("decoder_cross_attention_relation_barrier_scale")
            ),
            "decoder_readout_keep_ratio": (
                args.decoder_readout_keep_ratio
                if args.decoder_readout_keep_ratio is not None
                else behavior_summary.get("decoder_readout_keep_ratio")
            ),
            "decoder_readout_subspace_keep_ratio": (
                args.decoder_readout_subspace_keep_ratio
                if args.decoder_readout_subspace_keep_ratio is not None
                else behavior_summary.get("decoder_readout_subspace_keep_ratio")
            ),
            "decoder_readout_subspace_seed": (
                args.decoder_readout_subspace_seed
                if args.decoder_readout_subspace_seed is not None
                else behavior_summary.get("decoder_readout_subspace_seed")
            ),
            "decoder_readout_subspace_mode": (
                args.decoder_readout_subspace_mode
                if args.decoder_readout_subspace_mode is not None
                else behavior_summary.get("decoder_readout_subspace_mode")
            ),
            "encoder_geometry_transform": (
                args.encoder_geometry_transform
                if args.encoder_geometry_transform is not None
                else behavior_summary.get("encoder_geometry_transform")
            ),
            "encoder_geometry_seed": (
                args.encoder_geometry_seed
                if args.encoder_geometry_seed is not None
                else behavior_summary.get("encoder_geometry_seed")
            ),
            "encoder_geometry_keep_ratio": (
                args.encoder_geometry_keep_ratio
                if args.encoder_geometry_keep_ratio is not None
                else behavior_summary.get("encoder_geometry_keep_ratio")
            ),
            "encoder_geometry_noise_std": (
                args.encoder_geometry_noise_std
                if args.encoder_geometry_noise_std is not None
                else behavior_summary.get("encoder_geometry_noise_std")
            ),
            **summary,
        },
    )

    print(
        {
            "run_name": run_name,
            "candidate_strata": candidate_strata,
            "match_required_strata": match_required_strata,
            "selected_candidates": len(selected),
            "matched_set_count": len(matched_sets),
            "span_coverage": summary["span_coverage"],
        }
    )


def infer_state_aware_strata(
    records: list[dict[str, object]],
    *,
    candidate_strata: list[str] | None,
    match_required_strata: list[str] | None,
) -> tuple[list[str], list[str]]:
    if candidate_strata and match_required_strata:
        return list(candidate_strata), list(match_required_strata)

    counts = Counter(str(record.get("stratum") or "unknown") for record in records)
    non_control = [label for label in ("pass", "fail") if counts.get(label, 0) > 0]
    resolved_candidate = list(candidate_strata or (non_control + (["control"] if counts.get("control", 0) > 0 else [])))
    resolved_match = list(match_required_strata or non_control)
    if not resolved_candidate:
        raise SystemExit("No available behavior strata for discovery")
    if not resolved_match:
        raise SystemExit("No non-control strata available for matched-set discovery")
    return resolved_candidate, resolved_match


def build_run_name(
    behavior_run_name: str,
    model_name_or_path: str,
    layers: list[int],
    per_stratum: int,
) -> str:
    model_key = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name_or_path).strip("_")
    layer_key = "_".join(str(layer).replace("-", "m") for layer in layers)
    return f"{behavior_run_name}__{model_key}__layers_{layer_key}__per_{per_stratum}"


if __name__ == "__main__":
    main()
