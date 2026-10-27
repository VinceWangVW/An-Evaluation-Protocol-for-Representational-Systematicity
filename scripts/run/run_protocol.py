from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from discovery import join_behavior_predictions
from interventions import run_generation_phase4_protocol
from models import build_model_adapter
from utils.io import read_jsonl, write_json, write_jsonl


def infer_operationalization_family(
    *,
    explicit: str,
    discovery_prefix: str,
    output_prefix: str,
    model_name_or_path: str,
    discovery_summary: dict[str, object],
) -> str:
    if explicit != "auto":
        return explicit
    haystacks = [
        discovery_prefix,
        output_prefix,
        model_name_or_path,
        str(discovery_summary.get("family_name") or ""),
        str(discovery_summary.get("hypothesis_family") or ""),
    ]
    normalized = " ".join(haystacks).lower()
    if "geometry" in normalized or "distributed_geometry" in normalized:
        return "geometry"
    return "binding"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the generation-level protocol for a control run.")
    parser.add_argument("--phase11-prefix", required=True)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--family", default=None, choices=["causal_lm", "seq2seq_lm"])
    parser.add_argument("--prompt-style", default=None, choices=["raw", "parser_instruction", "sentence_only"])
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--precision", default=None, choices=["auto", "fp32", "bf16", "fp16"])
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
    parser.add_argument("--non-role-policy", default="contentful_first", choices=["contentful_first", "nearest_window", "random_window"])
    parser.add_argument("--strongest-control-aggregation", default="max", choices=["max", "mean", "priority"])
    parser.add_argument("--strongest-control-priority", nargs="+", default=["wrong_role", "non_role", "shuffled"])
    parser.add_argument("--preferred-counterpart-strata", nargs="+", default=["pass", "control"])
    parser.add_argument("--counterpart-same-split-only", action="store_true")
    parser.add_argument("--interchange-target-strata", nargs="*", default=None)
    parser.add_argument("--rescue-target-strata", nargs="*", default=None)
    parser.add_argument("--operationalization-family", default="auto", choices=["auto", "binding", "geometry"])
    parser.add_argument("--geometry-unit-class", default="routing", choices=["routing", "subspace", "trajectory"])
    parser.add_argument(
        "--geometry-object-class",
        default="auto",
        choices=["auto", "store", "route", "read", "interaction", "cross_step_bridge", "query_source_relation"],
    )
    parser.add_argument("--geometry-object-spec", default=None)
    parser.add_argument(
        "--geometry-plant-kind",
        default="off",
        choices=["off", "h2_positive", "h1_positive", "null_decoy"],
    )
    parser.add_argument("--geometry-plant-spec", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    discovery_prefix = args.phase11_prefix
    output_prefix = args.output_prefix or args.phase11_prefix
    summary_path = PROJECT_ROOT / "reports/discovery_runs" / f"{discovery_prefix}_summary.json"
    discovery_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    behavior_run = str(discovery_summary["behavior_run_name"])
    behavior_summary = json.loads(
        (PROJECT_ROOT / "reports/behavior_runs" / f"{behavior_run}_summary.json").read_text(encoding="utf-8")
    )
    model_name_or_path = args.model_name_or_path or str(
        discovery_summary.get("trained_model_path") or behavior_summary.get("trained_model_path") or behavior_summary["model_name_or_path"]
    )
    family = args.family or str(behavior_summary["family"])
    prompt_style = args.prompt_style or str(behavior_summary["prompt_style"])
    layer_index = int(next(iter(discovery_summary["layers"].keys())))
    num_beams = args.num_beams or int(discovery_summary.get("num_beams") or behavior_summary.get("num_beams") or 1)
    precision = args.precision or str(discovery_summary.get("precision") or behavior_summary.get("precision") or "auto")
    decoder_cross_attention_scale = (
        args.decoder_cross_attention_scale
        if args.decoder_cross_attention_scale is not None
        else discovery_summary.get("decoder_cross_attention_scale")
    )
    decoder_cross_attention_keep_ratio = (
        args.decoder_cross_attention_keep_ratio
        if args.decoder_cross_attention_keep_ratio is not None
        else discovery_summary.get("decoder_cross_attention_keep_ratio")
    )
    decoder_cross_attention_layer_keep_ratio = (
        args.decoder_cross_attention_layer_keep_ratio
        if args.decoder_cross_attention_layer_keep_ratio is not None
        else discovery_summary.get("decoder_cross_attention_layer_keep_ratio")
    )
    decoder_readout_keep_ratio = (
        args.decoder_readout_keep_ratio
        if args.decoder_readout_keep_ratio is not None
        else discovery_summary.get("decoder_readout_keep_ratio")
    )
    decoder_readout_subspace_keep_ratio = (
        args.decoder_readout_subspace_keep_ratio
        if args.decoder_readout_subspace_keep_ratio is not None
        else discovery_summary.get("decoder_readout_subspace_keep_ratio")
    )

    interchange_target_strata = resolve_interchange_target_strata(
        explicit=args.interchange_target_strata,
        discovery_summary=discovery_summary,
    )
    rescue_target_strata = resolve_rescue_target_strata(
        explicit=args.rescue_target_strata,
        discovery_summary=discovery_summary,
        interchange_target_strata=interchange_target_strata,
    )
    operationalization_family = infer_operationalization_family(
        explicit=args.operationalization_family,
        discovery_prefix=discovery_prefix,
        output_prefix=output_prefix,
        model_name_or_path=model_name_or_path,
        discovery_summary=discovery_summary,
    )

    adapter = build_model_adapter(
        model_name_or_path=model_name_or_path,
        family=family,
        prompt_style=prompt_style,
        precision=precision,
        num_beams=num_beams,
        decoder_cross_attention_scale=decoder_cross_attention_scale,
        decoder_cross_attention_keep_ratio=decoder_cross_attention_keep_ratio,
        decoder_cross_attention_layer_keep_ratio=decoder_cross_attention_layer_keep_ratio,
        decoder_cross_attention_layer_seed=(
            args.decoder_cross_attention_layer_seed
            if args.decoder_cross_attention_layer_seed is not None
            else int(discovery_summary.get("decoder_cross_attention_layer_seed") or 0)
        ),
        decoder_cross_attention_layer_policy=(
            args.decoder_cross_attention_layer_policy
            if args.decoder_cross_attention_layer_policy is not None
            else str(discovery_summary.get("decoder_cross_attention_layer_policy") or "random")
        ),
        decoder_cross_attention_bottleneck_keep_ratio=(
            args.decoder_cross_attention_bottleneck_keep_ratio
            if args.decoder_cross_attention_bottleneck_keep_ratio is not None
            else discovery_summary.get("decoder_cross_attention_bottleneck_keep_ratio")
        ),
        decoder_cross_attention_bottleneck_seed=(
            args.decoder_cross_attention_bottleneck_seed
            if args.decoder_cross_attention_bottleneck_seed is not None
            else int(discovery_summary.get("decoder_cross_attention_bottleneck_seed") or 0)
        ),
        decoder_cross_attention_bottleneck_mode=(
            args.decoder_cross_attention_bottleneck_mode
            if args.decoder_cross_attention_bottleneck_mode is not None
            else str(discovery_summary.get("decoder_cross_attention_bottleneck_mode") or "keep")
        ),
        decoder_cross_attention_relation_barrier_keep_ratio=(
            args.decoder_cross_attention_relation_barrier_keep_ratio
            if args.decoder_cross_attention_relation_barrier_keep_ratio is not None
            else discovery_summary.get("decoder_cross_attention_relation_barrier_keep_ratio")
        ),
        decoder_cross_attention_relation_barrier_seed=(
            args.decoder_cross_attention_relation_barrier_seed
            if args.decoder_cross_attention_relation_barrier_seed is not None
            else int(discovery_summary.get("decoder_cross_attention_relation_barrier_seed") or 0)
        ),
        decoder_cross_attention_relation_barrier_scale=(
            args.decoder_cross_attention_relation_barrier_scale
            if args.decoder_cross_attention_relation_barrier_scale is not None
            else float(discovery_summary.get("decoder_cross_attention_relation_barrier_scale") or 0.5)
        ),
        decoder_readout_keep_ratio=decoder_readout_keep_ratio,
        decoder_readout_subspace_keep_ratio=decoder_readout_subspace_keep_ratio,
        decoder_readout_subspace_seed=(
            args.decoder_readout_subspace_seed
            if args.decoder_readout_subspace_seed is not None
            else int(discovery_summary.get("decoder_readout_subspace_seed") or 0)
        ),
        decoder_readout_subspace_mode=(
            args.decoder_readout_subspace_mode
            if args.decoder_readout_subspace_mode is not None
            else str(discovery_summary.get("decoder_readout_subspace_mode") or "remove")
        ),
        encoder_geometry_transform=(
            args.encoder_geometry_transform
            if args.encoder_geometry_transform is not None
            else discovery_summary.get("encoder_geometry_transform")
        ),
        encoder_geometry_seed=(
            args.encoder_geometry_seed
            if args.encoder_geometry_seed is not None
            else int(discovery_summary.get("encoder_geometry_seed") or 0)
        ),
        encoder_geometry_keep_ratio=(
            args.encoder_geometry_keep_ratio
            if args.encoder_geometry_keep_ratio is not None
            else discovery_summary.get("encoder_geometry_keep_ratio")
        ),
        encoder_geometry_noise_std=(
            args.encoder_geometry_noise_std
            if args.encoder_geometry_noise_std is not None
            else discovery_summary.get("encoder_geometry_noise_std")
        ),
    )
    candidate_records = read_jsonl(PROJECT_ROOT / "artifacts/discovery_runs" / f"{discovery_prefix}_candidate_table.jsonl")
    matched_sets = read_jsonl(PROJECT_ROOT / "artifacts/discovery_runs" / f"{discovery_prefix}_matched_sets.jsonl")
    span_records = read_jsonl(PROJECT_ROOT / "artifacts/discovery_runs" / f"{discovery_prefix}_span_vectors.jsonl")
    benchmark_records = read_jsonl(PROJECT_ROOT / "data/processed" / f"{discovery_summary['benchmark_name']}.jsonl")
    behavior_predictions = read_jsonl(PROJECT_ROOT / "runs/behavior_runs" / f"{behavior_run}_predictions.jsonl")
    eval_examples = [record for record in benchmark_records if str(record["split"]) != "train"]
    reference_records = join_behavior_predictions(eval_examples, behavior_predictions)

    m2_records, m3_records, m4_records, linkage_records, summary = run_generation_phase4_protocol(
        adapter=adapter,
        candidate_records=candidate_records,
        reference_records=reference_records,
        matched_sets=matched_sets,
        span_records=span_records,
        layer_index=layer_index,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        num_beams=num_beams,
        non_role_policy=args.non_role_policy,
        strongest_control_aggregation=args.strongest_control_aggregation,
        strongest_control_priority=tuple(args.strongest_control_priority),
        preferred_counterpart_strata=tuple(args.preferred_counterpart_strata),
        counterpart_same_split_only=args.counterpart_same_split_only,
        interchange_target_strata=tuple(interchange_target_strata),
        rescue_target_strata=tuple(rescue_target_strata),
        operationalization_family=operationalization_family,
        geometry_unit_class=args.geometry_unit_class,
        geometry_object_class=args.geometry_object_class,
        geometry_object_spec=args.geometry_object_spec,
        geometry_plant_kind=args.geometry_plant_kind,
        geometry_plant_spec=args.geometry_plant_spec,
        seed=args.seed,
    )

    write_jsonl(PROJECT_ROOT / "artifacts/protocol_runs" / f"{output_prefix}_m2_generation_interchange.jsonl", m2_records)
    write_jsonl(PROJECT_ROOT / "artifacts/protocol_runs" / f"{output_prefix}_m3_generation_necessity.jsonl", m3_records)
    write_jsonl(PROJECT_ROOT / "artifacts/protocol_runs" / f"{output_prefix}_m4_generation_rescue.jsonl", m4_records)
    write_jsonl(PROJECT_ROOT / "artifacts/protocol_runs" / f"{output_prefix}_m5_generation_item_linkage.jsonl", linkage_records)
    write_json(
        PROJECT_ROOT / "reports/protocol_runs" / f"{output_prefix}_summary.json",
        {
            "phase11_prefix": output_prefix,
            "discovery_prefix": discovery_prefix,
            "behavior_run_name": behavior_run,
            "model_name_or_path": model_name_or_path,
            "family": family,
            "prompt_style": prompt_style,
            "layer_index": layer_index,
            "hf_endpoint": os.environ.get("HF_ENDPOINT"),
            "num_beams": num_beams,
            "precision": precision,
            "decoder_cross_attention_scale": decoder_cross_attention_scale,
            "decoder_cross_attention_keep_ratio": decoder_cross_attention_keep_ratio,
            "decoder_cross_attention_layer_keep_ratio": decoder_cross_attention_layer_keep_ratio,
            "decoder_cross_attention_layer_seed": (
                args.decoder_cross_attention_layer_seed
                if args.decoder_cross_attention_layer_seed is not None
                else discovery_summary.get("decoder_cross_attention_layer_seed")
            ),
            "decoder_cross_attention_layer_policy": (
                args.decoder_cross_attention_layer_policy
                if args.decoder_cross_attention_layer_policy is not None
                else discovery_summary.get("decoder_cross_attention_layer_policy")
            ),
            "decoder_cross_attention_bottleneck_keep_ratio": (
                args.decoder_cross_attention_bottleneck_keep_ratio
                if args.decoder_cross_attention_bottleneck_keep_ratio is not None
                else discovery_summary.get("decoder_cross_attention_bottleneck_keep_ratio")
            ),
            "decoder_cross_attention_bottleneck_seed": (
                args.decoder_cross_attention_bottleneck_seed
                if args.decoder_cross_attention_bottleneck_seed is not None
                else discovery_summary.get("decoder_cross_attention_bottleneck_seed")
            ),
            "decoder_cross_attention_bottleneck_mode": (
                args.decoder_cross_attention_bottleneck_mode
                if args.decoder_cross_attention_bottleneck_mode is not None
                else discovery_summary.get("decoder_cross_attention_bottleneck_mode")
            ),
            "decoder_cross_attention_relation_barrier_keep_ratio": (
                args.decoder_cross_attention_relation_barrier_keep_ratio
                if args.decoder_cross_attention_relation_barrier_keep_ratio is not None
                else discovery_summary.get("decoder_cross_attention_relation_barrier_keep_ratio")
            ),
            "decoder_cross_attention_relation_barrier_seed": (
                args.decoder_cross_attention_relation_barrier_seed
                if args.decoder_cross_attention_relation_barrier_seed is not None
                else discovery_summary.get("decoder_cross_attention_relation_barrier_seed")
            ),
            "decoder_cross_attention_relation_barrier_scale": (
                args.decoder_cross_attention_relation_barrier_scale
                if args.decoder_cross_attention_relation_barrier_scale is not None
                else discovery_summary.get("decoder_cross_attention_relation_barrier_scale")
            ),
            "decoder_readout_keep_ratio": decoder_readout_keep_ratio,
            "decoder_readout_subspace_keep_ratio": decoder_readout_subspace_keep_ratio,
            "decoder_readout_subspace_seed": (
                args.decoder_readout_subspace_seed
                if args.decoder_readout_subspace_seed is not None
                else discovery_summary.get("decoder_readout_subspace_seed")
            ),
            "decoder_readout_subspace_mode": (
                args.decoder_readout_subspace_mode
                if args.decoder_readout_subspace_mode is not None
                else discovery_summary.get("decoder_readout_subspace_mode")
            ),
            "encoder_geometry_transform": (
                args.encoder_geometry_transform
                if args.encoder_geometry_transform is not None
                else discovery_summary.get("encoder_geometry_transform")
            ),
            "encoder_geometry_seed": (
                args.encoder_geometry_seed
                if args.encoder_geometry_seed is not None
                else discovery_summary.get("encoder_geometry_seed")
            ),
            "encoder_geometry_keep_ratio": (
                args.encoder_geometry_keep_ratio
                if args.encoder_geometry_keep_ratio is not None
                else discovery_summary.get("encoder_geometry_keep_ratio")
            ),
            "encoder_geometry_noise_std": (
                args.encoder_geometry_noise_std
                if args.encoder_geometry_noise_std is not None
                else discovery_summary.get("encoder_geometry_noise_std")
            ),
            "protocol_config": {
                "non_role_policy": args.non_role_policy,
                "strongest_control_aggregation": args.strongest_control_aggregation,
                "strongest_control_priority": list(args.strongest_control_priority),
                "preferred_counterpart_strata": list(args.preferred_counterpart_strata),
                "counterpart_same_split_only": args.counterpart_same_split_only,
                "interchange_target_strata": interchange_target_strata,
                "rescue_target_strata": rescue_target_strata,
                "operationalization_family": operationalization_family,
                "geometry_unit_class": args.geometry_unit_class,
                "geometry_object_class": args.geometry_object_class,
                "geometry_object_spec": args.geometry_object_spec,
                "geometry_plant_kind": args.geometry_plant_kind,
                "geometry_plant_spec": args.geometry_plant_spec,
            },
            **summary,
        },
    )

    print(
        {
            "phase11_prefix": output_prefix,
            "layer_index": layer_index,
            "interchange_target_strata": interchange_target_strata,
            "rescue_target_strata": rescue_target_strata,
            "operationalization_family": operationalization_family,
            "geometry_unit_class": args.geometry_unit_class,
            "geometry_object_class": args.geometry_object_class,
            "geometry_object_spec": args.geometry_object_spec,
            "geometry_plant_kind": args.geometry_plant_kind,
            "geometry_plant_spec": args.geometry_plant_spec,
            "M2": summary["modules"]["M2_generation_interchange"],
            "M3": summary["modules"]["M3_generation_necessity"],
            "M4": summary["modules"]["M4_generation_rescue"],
        }
    )


def resolve_interchange_target_strata(
    *,
    explicit: list[str] | None,
    discovery_summary: dict[str, object],
) -> list[str]:
    if explicit is not None:
        return [str(label) for label in explicit]
    labels = [str(label) for label in discovery_summary.get("match_required_strata") or []]
    return labels or ["fail"]


def resolve_rescue_target_strata(
    *,
    explicit: list[str] | None,
    discovery_summary: dict[str, object],
    interchange_target_strata: list[str],
) -> list[str]:
    if explicit is not None:
        return [str(label) for label in explicit]
    available_strata = {str(label) for label in discovery_summary.get("candidate_strata") or []}
    if "fail" in available_strata and "fail" in interchange_target_strata:
        return ["fail"]
    return []


if __name__ == "__main__":
    main()
