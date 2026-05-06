from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .control_filters import (
    ControlPanel,
    ControlSpec,
    filter_examples_for_control,
    load_control_panel,
    resolve_model_name_or_path,
    summarize_control_split,
)


@dataclass(slots=True)
class StateRunSpec:
    control_id: str
    behavior_run: str
    phase11_prefix: str
    expected_state: str
    panel_split: str
    benchmark: str = "phase11_systematicity_v2"
    family: str = "seq2seq_lm"
    prompt_style: str = "raw"
    layers: list[int] = field(default_factory=lambda: [-1])
    per_stratum: int = 24
    match_set_limit: int = 24
    num_beams: int = 1
    precision: str = "auto"
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


def load_state_control_panel(path: str | Path) -> ControlPanel:
    return load_control_panel(path)


def load_state_run_panel(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["runs"] = [StateRunSpec(**run) for run in payload.get("runs") or []]
    return payload


__all__ = [
    "StateRunSpec",
    "filter_examples_for_control",
    "load_state_control_panel",
    "load_state_run_panel",
    "resolve_model_name_or_path",
    "summarize_control_split",
]


# Backward-compatible aliases for older internal call sites.
Phase12StateRunSpec = StateRunSpec
load_phase12_control_panel = load_state_control_panel
load_phase12_state_run_panel = load_state_run_panel
