from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ControlSpec:
    control_id: str
    run_name: str
    expected_states: dict[str, str]
    model_name_or_path: str | None = None
    reuse_checkpoint_from: str | None = None
    prompt_style: str = "raw"
    num_train_epochs: int = 1
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
    notes: str | None = None


@dataclass(slots=True)
class ControlPanel:
    panel_name: str
    benchmark: str
    model_name_or_path: str
    prompt_style: str = "raw"
    hf_endpoint: str | None = None
    controls: list[ControlSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_control_panel(path: str | Path) -> ControlPanel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    controls = [ControlSpec(**control) for control in payload.pop("controls", [])]
    return ControlPanel(controls=controls, **payload)


def filter_examples_for_control(
    examples: list[dict[str, Any]],
    *,
    allowlist: list[str] | None = None,
    denylist: list[str] | None = None,
) -> list[dict[str, Any]]:
    allow = set(allowlist or [])
    deny = set(denylist or [])
    filtered: list[dict[str, Any]] = []
    for example in examples:
        case_name = str(example.get("generalization_case", "unknown"))
        if allow and case_name not in allow:
            continue
        if deny and case_name in deny:
            continue
        filtered.append(example)
    return filtered


def summarize_control_split(
    *,
    benchmark: str,
    control: ControlSpec,
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
        "control_id": control.control_id,
        "run_name": control.run_name,
        "model_name_or_path": model_name_or_path,
        "expected_states": control.expected_states,
        "train_example_count": len(train_examples),
        "eval_example_count": len(eval_examples),
        "train_generalization_case_allowlist": list(control.train_generalization_case_allowlist),
        "train_generalization_case_denylist": list(control.train_generalization_case_denylist),
        "train_case_counts": train_case_counts,
        "eval_split_counts": eval_split_counts,
        "train_target_permutation_seed": control.train_target_permutation_seed,
        "decoder_cross_attention_scale": control.decoder_cross_attention_scale,
        "decoder_cross_attention_keep_ratio": control.decoder_cross_attention_keep_ratio,
        "decoder_cross_attention_layer_keep_ratio": control.decoder_cross_attention_layer_keep_ratio,
        "decoder_cross_attention_layer_seed": control.decoder_cross_attention_layer_seed,
        "decoder_cross_attention_layer_policy": control.decoder_cross_attention_layer_policy,
        "decoder_cross_attention_bottleneck_keep_ratio": control.decoder_cross_attention_bottleneck_keep_ratio,
        "decoder_cross_attention_bottleneck_seed": control.decoder_cross_attention_bottleneck_seed,
        "decoder_cross_attention_bottleneck_mode": control.decoder_cross_attention_bottleneck_mode,
        "decoder_cross_attention_relation_barrier_keep_ratio": control.decoder_cross_attention_relation_barrier_keep_ratio,
        "decoder_cross_attention_relation_barrier_seed": control.decoder_cross_attention_relation_barrier_seed,
        "decoder_cross_attention_relation_barrier_scale": control.decoder_cross_attention_relation_barrier_scale,
        "decoder_readout_keep_ratio": control.decoder_readout_keep_ratio,
        "decoder_readout_subspace_keep_ratio": control.decoder_readout_subspace_keep_ratio,
        "decoder_readout_subspace_seed": control.decoder_readout_subspace_seed,
        "decoder_readout_subspace_mode": control.decoder_readout_subspace_mode,
        "encoder_geometry_transform": control.encoder_geometry_transform,
        "encoder_geometry_seed": control.encoder_geometry_seed,
        "encoder_geometry_keep_ratio": control.encoder_geometry_keep_ratio,
        "encoder_geometry_noise_std": control.encoder_geometry_noise_std,
        "num_train_epochs": control.num_train_epochs,
        "seed": control.seed,
        "notes": control.notes,
    }


def resolve_model_name_or_path(
    *,
    control_id: str,
    control_model_name_or_path: str | None,
    default_model_name_or_path: str,
    reuse_checkpoint_from: str | None,
    trained_runs: dict[str, dict],
) -> str:
    if reuse_checkpoint_from:
        if reuse_checkpoint_from not in trained_runs:
            raise KeyError(f"Control {control_id} requested unknown checkpoint source: {reuse_checkpoint_from}")
        return str(trained_runs[reuse_checkpoint_from]["trained_model_path"])
    return str(control_model_name_or_path or default_model_name_or_path)


# Backward-compatible aliases for older internal call sites.
Phase11ControlSpec = ControlSpec
Phase11ControlPanel = ControlPanel
load_phase11_control_panel = load_control_panel
