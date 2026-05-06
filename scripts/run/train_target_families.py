from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.harness import evaluate_benchmark
from models.behavior_seq2seq import HfSeq2SeqBehaviorModel, Seq2SeqBehaviorConfig
from models.control_filters import filter_examples_for_control
from models.target_panels import (
    load_target_panel,
    resolve_target_model_name_or_path,
    summarize_target_split,
)
from utils.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a held-out target-model panel.")
    parser.add_argument("--panel-config", required=True)
    parser.add_argument("--benchmark-override", default=None)
    parser.add_argument("--model-name-or-path-override", default=None)
    parser.add_argument("--panel-name-override", default=None)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = load_target_panel(args.panel_config)
    benchmark_name = args.benchmark_override or panel.benchmark
    default_model_name_or_path = args.model_name_or_path_override or panel.model_name_or_path
    panel_name = args.panel_name_override or panel.panel_name
    hf_endpoint = args.hf_endpoint or panel.hf_endpoint
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    targets = panel.targets[: args.max_targets] if args.max_targets is not None else panel.targets
    manifest_rows: list[dict] = []
    trained_targets: dict[str, dict] = {}

    for target in targets:
        target_benchmark = str(target.benchmark or benchmark_name)
        benchmark_records = read_jsonl(PROJECT_ROOT / "data/processed" / f"{target_benchmark}.jsonl")
        train_examples = [record for record in benchmark_records if str(record["split"]) == "train"]
        eval_examples = [record for record in benchmark_records if str(record["split"]) != "train"]
        if not train_examples or not eval_examples:
            raise SystemExit(f"Benchmark {target_benchmark} is missing train/eval splits")
        train_examples = filter_examples_for_control(
            train_examples,
            allowlist=target.train_generalization_case_allowlist,
            denylist=target.train_generalization_case_denylist,
        )
        model_name_or_path = resolve_target_model_name_or_path(
            target_id=target.target_id,
            target_model_name_or_path=target.model_name_or_path,
            default_model_name_or_path=default_model_name_or_path,
            reuse_checkpoint_from=target.reuse_checkpoint_from,
            trained_runs=trained_targets,
        )
        prompt_style = str(target.prompt_style or panel.prompt_style)
        config = Seq2SeqBehaviorConfig(
            model_name_or_path=model_name_or_path,
            prompt_style=prompt_style,
            precision=target.precision,
            num_beams=target.num_beams,
            train_batch_size=target.train_batch_size,
            eval_batch_size=target.eval_batch_size,
            learning_rate=target.learning_rate,
            weight_decay=target.weight_decay,
            num_train_epochs=target.num_train_epochs,
            gradient_accumulation_steps=target.gradient_accumulation_steps,
            train_limit=target.train_limit,
            train_target_permutation_seed=target.train_target_permutation_seed,
            decoder_cross_attention_scale=target.decoder_cross_attention_scale,
            decoder_cross_attention_keep_ratio=target.decoder_cross_attention_keep_ratio,
            decoder_cross_attention_layer_keep_ratio=target.decoder_cross_attention_layer_keep_ratio,
            decoder_cross_attention_layer_seed=target.decoder_cross_attention_layer_seed,
            decoder_cross_attention_layer_policy=target.decoder_cross_attention_layer_policy,
            decoder_cross_attention_bottleneck_keep_ratio=target.decoder_cross_attention_bottleneck_keep_ratio,
            decoder_cross_attention_bottleneck_seed=target.decoder_cross_attention_bottleneck_seed,
            decoder_cross_attention_bottleneck_mode=target.decoder_cross_attention_bottleneck_mode,
            decoder_cross_attention_relation_barrier_keep_ratio=target.decoder_cross_attention_relation_barrier_keep_ratio,
            decoder_cross_attention_relation_barrier_seed=target.decoder_cross_attention_relation_barrier_seed,
            decoder_cross_attention_relation_barrier_scale=target.decoder_cross_attention_relation_barrier_scale,
            decoder_readout_keep_ratio=target.decoder_readout_keep_ratio,
            decoder_readout_subspace_keep_ratio=target.decoder_readout_subspace_keep_ratio,
            decoder_readout_subspace_seed=target.decoder_readout_subspace_seed,
            decoder_readout_subspace_mode=target.decoder_readout_subspace_mode,
            encoder_geometry_transform=target.encoder_geometry_transform,
            encoder_geometry_seed=target.encoder_geometry_seed,
            encoder_geometry_keep_ratio=target.encoder_geometry_keep_ratio,
            encoder_geometry_noise_std=target.encoder_geometry_noise_std,
            seed=target.seed,
            save_directory=str(PROJECT_ROOT / "runs" / "trained_target_families" / target.run_name),
        )
        model = HfSeq2SeqBehaviorModel(config)
        prediction_records, summary = evaluate_benchmark(
            train_examples=train_examples,
            eval_examples=eval_examples,
            model=model,
        )

        target_manifest = summarize_target_split(
            benchmark=target_benchmark,
            target=target,
            train_examples=train_examples,
            eval_examples=eval_examples,
            model_name_or_path=model_name_or_path,
        )
        target_manifest["trained_model_path"] = str(model.training_summary.get("save_directory") or model_name_or_path)
        target_manifest["evaluation"] = summary

        payload = {
            "panel_name": panel_name,
            "benchmark_name": target_benchmark,
            "target_id": target.target_id,
            "run_name": target.run_name,
            "phase11_prefix": target.phase11_prefix or target.run_name,
            "model_name_or_path": model_name_or_path,
            "trained_model_path": target_manifest["trained_model_path"],
            "family": "seq2seq_lm",
            "prompt_style": prompt_style,
            "hf_endpoint": os.environ.get("HF_ENDPOINT"),
            "precision": target.precision,
            "num_beams": target.num_beams,
            "decoder_cross_attention_scale": target.decoder_cross_attention_scale,
            "decoder_cross_attention_keep_ratio": target.decoder_cross_attention_keep_ratio,
            "decoder_cross_attention_layer_keep_ratio": target.decoder_cross_attention_layer_keep_ratio,
            "decoder_cross_attention_layer_seed": target.decoder_cross_attention_layer_seed,
            "decoder_cross_attention_layer_policy": target.decoder_cross_attention_layer_policy,
            "decoder_cross_attention_bottleneck_keep_ratio": target.decoder_cross_attention_bottleneck_keep_ratio,
            "decoder_cross_attention_bottleneck_seed": target.decoder_cross_attention_bottleneck_seed,
            "decoder_cross_attention_bottleneck_mode": target.decoder_cross_attention_bottleneck_mode,
            "decoder_cross_attention_relation_barrier_keep_ratio": target.decoder_cross_attention_relation_barrier_keep_ratio,
            "decoder_cross_attention_relation_barrier_seed": target.decoder_cross_attention_relation_barrier_seed,
            "decoder_cross_attention_relation_barrier_scale": target.decoder_cross_attention_relation_barrier_scale,
            "decoder_readout_keep_ratio": target.decoder_readout_keep_ratio,
            "decoder_readout_subspace_keep_ratio": target.decoder_readout_subspace_keep_ratio,
            "decoder_readout_subspace_seed": target.decoder_readout_subspace_seed,
            "decoder_readout_subspace_mode": target.decoder_readout_subspace_mode,
            "encoder_geometry_transform": target.encoder_geometry_transform,
            "encoder_geometry_seed": target.encoder_geometry_seed,
            "encoder_geometry_keep_ratio": target.encoder_geometry_keep_ratio,
            "encoder_geometry_noise_std": target.encoder_geometry_noise_std,
            "training": model.training_summary,
            "training_filter_summary": target_manifest,
            "evaluation": summary,
            "panel_regime": target.panel_regime,
            "truthfulness_class": target.truthfulness_class,
            "truthfulness_family": target.truthfulness_family,
            "truthfulness_seed": target.truthfulness_seed,
            "detector_family": target.detector_family,
            "detector_seed": target.detector_seed,
            "intended_observable": target.intended_observable,
            "intended_step_key": target.intended_step_key,
            "source_step_key": target.source_step_key,
            "target_step_key": target.target_step_key,
            "paired_null_id": target.paired_null_id,
            "surface_backend": target.surface_backend or panel.surface_backend,
            "include_in_adjudication": target.include_in_adjudication,
            "tags": list(target.tags),
            "notes": target.notes,
        }
        write_jsonl(PROJECT_ROOT / "runs" / "behavior_runs" / f"{target.run_name}_predictions.jsonl", prediction_records)
        write_json(PROJECT_ROOT / "reports" / "behavior_runs" / f"{target.run_name}_summary.json", payload)
        manifest_rows.append(target_manifest)
        trained_targets[target.target_id] = {
            "run_name": target.run_name,
            "trained_model_path": target_manifest["trained_model_path"],
            "phase11_prefix": target.phase11_prefix or target.run_name,
            "panel_regime": target.panel_regime,
            "surface_backend": target.surface_backend or panel.surface_backend,
            "benchmark_name": target_benchmark,
            "include_in_adjudication": target.include_in_adjudication,
        }

    panel_summary = {
        "panel_name": panel_name,
        "benchmark_name": benchmark_name,
        "target_count": len(manifest_rows),
        "target_ids": [row["target_id"] for row in manifest_rows],
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "notes": list(panel.metadata.get("notes") or []),
    }
    write_json(PROJECT_ROOT / "reports" / "trained_target_families" / f"{panel_name}_summary.json", panel_summary)
    write_jsonl(PROJECT_ROOT / "artifacts" / "target_families" / f"{panel_name}_manifest.jsonl", manifest_rows)
    write_jsonl(
        PROJECT_ROOT / "artifacts" / "target_families" / f"{panel_name}_trained_targets.jsonl",
        list(trained_targets.values()),
    )
    print(panel_summary)


if __name__ == "__main__":
    main()
