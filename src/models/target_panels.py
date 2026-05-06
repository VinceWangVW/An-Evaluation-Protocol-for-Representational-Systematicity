from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Phase11TargetSpec:
    target_id: str
    run_name: str
    phase11_prefix: str | None = None
    benchmark: str | None = None
    model_name_or_path: str | None = None
    reuse_checkpoint_from: str | None = None
    prompt_style: str | None = None
    num_train_epochs: int = 8
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    train_limit: int | None = None
    train_target_permutation_seed: int | None = None
    train_generalization_case_allowlist: list[str] = field(default_factory=list)
    train_generalization_case_denylist: list[str] = field(default_factory=list)
    decoder_cross_attention_scale: float | None = None
    decoder_cross_attention_keep_ratio: float | None = None
    decoder_cross_attention_layer_keep_ratio: float | None = None
    decoder_cross_attention_layer_seed: int = 0
    decoder_cross_attention_layer_policy: str = "random"
    decoder_cross_attention_bottleneck_keep_ratio: float | None = None
    decoder_cross_attention_bottleneck_seed: int = 0
    decoder_cross_attention_bottleneck_mode: str = "keep"
    decoder_cross_attention_relation_barrier_keep_ratio: float | None = None
    decoder_cross_attention_relation_barrier_seed: int = 0
    decoder_cross_attention_relation_barrier_scale: float = 0.5
    decoder_readout_keep_ratio: float | None = None
    decoder_readout_subspace_keep_ratio: float | None = None
    decoder_readout_subspace_seed: int = 0
    decoder_readout_subspace_mode: str = "remove"
    encoder_geometry_transform: str | None = None
    encoder_geometry_seed: int = 0
    encoder_geometry_keep_ratio: float | None = None
    encoder_geometry_noise_std: float | None = None
    precision: str = "auto"
    num_beams: int = 1
    seed: int = 0
    analysis_family: str = "seq2seq_lm"
    layers: list[int] = field(default_factory=lambda: [-1])
    per_stratum: int = 24
    match_set_limit: int = 24
    panel_regime: str | None = None
    truthfulness_class: str | None = None
    truthfulness_family: str | None = None
    truthfulness_seed: int | None = None
    detector_family: str | None = None
    detector_seed: int | None = None
    intended_observable: str | None = None
    intended_step_key: str | None = None
    source_step_key: str | None = None
    target_step_key: str | None = None
    paired_null_id: str | None = None
    surface_backend: str | None = None
    include_in_adjudication: bool = True
    notes: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Phase11TargetPanel:
    panel_name: str
    benchmark: str
    model_name_or_path: str
    prompt_style: str = "raw"
    row_loader: str | None = None
    hypothesis_family: str | None = None
    analysis_family: str = "seq2seq_lm"
    layers: list[int] = field(default_factory=lambda: [-1])
    per_stratum: int = 24
    match_set_limit: int = 24
    surface_backend: str | None = None
    hf_endpoint: str | None = None
    targets: list[Phase11TargetSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_phase11_target_panel(path: str | Path) -> Phase11TargetPanel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    targets = [Phase11TargetSpec(**target) for target in payload.pop("targets", [])]
    return Phase11TargetPanel(targets=targets, **payload)


def summarize_target_split(
    *,
    benchmark: str,
    target: Phase11TargetSpec,
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    model_name_or_path: str,
) -> dict[str, Any]:
    train_case_counts: dict[str, int] = {}
    for example in train_examples:
        case_name = str(example.get("generalization_case", "unknown"))
        train_case_counts[case_name] = train_case_counts.get(case_name, 0) + 1
    eval_split_counts: dict[str, int] = {}
    for example in eval_examples:
        split_name = str(example.get("split", "unknown"))
        eval_split_counts[split_name] = eval_split_counts.get(split_name, 0) + 1
    return {
        "panel_benchmark": benchmark,
        "target_id": target.target_id,
        "run_name": target.run_name,
        "phase11_prefix": target.phase11_prefix or target.run_name,
        "model_name_or_path": model_name_or_path,
        "reuse_checkpoint_from": target.reuse_checkpoint_from,
        "train_example_count": len(train_examples),
        "eval_example_count": len(eval_examples),
        "benchmark_override": target.benchmark,
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
        "surface_backend": target.surface_backend,
        "train_case_counts": train_case_counts,
        "eval_split_counts": eval_split_counts,
        "num_train_epochs": target.num_train_epochs,
        "train_limit": target.train_limit,
        "train_target_permutation_seed": target.train_target_permutation_seed,
        "train_generalization_case_allowlist": list(target.train_generalization_case_allowlist),
        "train_generalization_case_denylist": list(target.train_generalization_case_denylist),
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
        "seed": target.seed,
        "analysis_family": target.analysis_family,
        "layers": list(target.layers),
        "per_stratum": target.per_stratum,
        "match_set_limit": target.match_set_limit,
        "include_in_adjudication": target.include_in_adjudication,
        "tags": list(target.tags),
        "notes": target.notes,
    }


def resolve_target_model_name_or_path(
    *,
    target_id: str,
    target_model_name_or_path: str | None,
    default_model_name_or_path: str,
    reuse_checkpoint_from: str | None,
    trained_runs: dict[str, dict[str, Any]],
) -> str:
    if reuse_checkpoint_from:
        if reuse_checkpoint_from not in trained_runs:
            raise KeyError(f"Target {target_id} requested unknown checkpoint source: {reuse_checkpoint_from}")
        return str(trained_runs[reuse_checkpoint_from]["trained_model_path"])
    return str(target_model_name_or_path or default_model_name_or_path)
