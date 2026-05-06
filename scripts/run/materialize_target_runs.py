from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.target_panels import load_target_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize trained target runs through discovery and protocol.")
    parser.add_argument("--panel-config", required=True)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=1)
    return parser.parse_args()


def _build_target_commands(
    *,
    panel,
    target,
    hf_endpoint: str | None,
    operationalization_family: str,
) -> tuple[list[str], list[str]]:
    behavior_run = target.run_name
    output_prefix = str(target.phase11_prefix or target.run_name)
    benchmark_name = str(target.benchmark or panel.benchmark)
    prompt_style = str(target.prompt_style or panel.prompt_style)
    discovery_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run/run_discovery.py"),
        "--benchmark",
        benchmark_name,
        "--behavior-run",
        behavior_run,
        "--output-prefix",
        output_prefix,
        "--family",
        str(target.analysis_family),
        "--prompt-style",
        prompt_style,
        "--layers",
        *[str(layer) for layer in target.layers],
        "--per-stratum",
        str(target.per_stratum),
        "--match-set-limit",
        str(target.match_set_limit),
        "--precision",
        target.precision,
        "--num-beams",
        str(target.num_beams),
    ]
    if hf_endpoint:
        discovery_cmd.extend(["--hf-endpoint", str(hf_endpoint)])
    if target.decoder_cross_attention_scale is not None:
        discovery_cmd.extend(["--decoder-cross-attention-scale", str(target.decoder_cross_attention_scale)])
    if target.decoder_cross_attention_keep_ratio is not None:
        discovery_cmd.extend(["--decoder-cross-attention-keep-ratio", str(target.decoder_cross_attention_keep_ratio)])
    if target.decoder_cross_attention_layer_keep_ratio is not None:
        discovery_cmd.extend(["--decoder-cross-attention-layer-keep-ratio", str(target.decoder_cross_attention_layer_keep_ratio)])
        discovery_cmd.extend(["--decoder-cross-attention-layer-seed", str(target.decoder_cross_attention_layer_seed)])
        discovery_cmd.extend(["--decoder-cross-attention-layer-policy", str(target.decoder_cross_attention_layer_policy)])
    if target.decoder_cross_attention_bottleneck_keep_ratio is not None:
        discovery_cmd.extend(["--decoder-cross-attention-bottleneck-keep-ratio", str(target.decoder_cross_attention_bottleneck_keep_ratio)])
        discovery_cmd.extend(["--decoder-cross-attention-bottleneck-seed", str(target.decoder_cross_attention_bottleneck_seed)])
        discovery_cmd.extend(["--decoder-cross-attention-bottleneck-mode", str(target.decoder_cross_attention_bottleneck_mode)])
    if target.decoder_cross_attention_relation_barrier_keep_ratio is not None:
        discovery_cmd.extend(["--decoder-cross-attention-relation-barrier-keep-ratio", str(target.decoder_cross_attention_relation_barrier_keep_ratio)])
        discovery_cmd.extend(["--decoder-cross-attention-relation-barrier-seed", str(target.decoder_cross_attention_relation_barrier_seed)])
        discovery_cmd.extend(["--decoder-cross-attention-relation-barrier-scale", str(target.decoder_cross_attention_relation_barrier_scale)])
    if target.decoder_readout_keep_ratio is not None:
        discovery_cmd.extend(["--decoder-readout-keep-ratio", str(target.decoder_readout_keep_ratio)])
    if target.decoder_readout_subspace_keep_ratio is not None:
        discovery_cmd.extend(["--decoder-readout-subspace-keep-ratio", str(target.decoder_readout_subspace_keep_ratio)])
        discovery_cmd.extend(["--decoder-readout-subspace-seed", str(target.decoder_readout_subspace_seed)])
        discovery_cmd.extend(["--decoder-readout-subspace-mode", str(target.decoder_readout_subspace_mode)])
    if target.encoder_geometry_transform:
        discovery_cmd.extend(["--encoder-geometry-transform", str(target.encoder_geometry_transform)])
        discovery_cmd.extend(["--encoder-geometry-seed", str(target.encoder_geometry_seed)])
    if target.encoder_geometry_keep_ratio is not None:
        discovery_cmd.extend(["--encoder-geometry-keep-ratio", str(target.encoder_geometry_keep_ratio)])
    if target.encoder_geometry_noise_std is not None:
        discovery_cmd.extend(["--encoder-geometry-noise-std", str(target.encoder_geometry_noise_std)])

    protocol_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run/run_protocol.py"),
        "--phase11-prefix",
        output_prefix,
        "--output-prefix",
        output_prefix,
        "--prompt-style",
        prompt_style,
        "--precision",
        target.precision,
        "--num-beams",
        str(target.num_beams),
    ]
    if hf_endpoint:
        protocol_cmd.extend(["--hf-endpoint", str(hf_endpoint)])
    if target.decoder_cross_attention_scale is not None:
        protocol_cmd.extend(["--decoder-cross-attention-scale", str(target.decoder_cross_attention_scale)])
    if target.decoder_cross_attention_keep_ratio is not None:
        protocol_cmd.extend(["--decoder-cross-attention-keep-ratio", str(target.decoder_cross_attention_keep_ratio)])
    if target.decoder_cross_attention_layer_keep_ratio is not None:
        protocol_cmd.extend(["--decoder-cross-attention-layer-keep-ratio", str(target.decoder_cross_attention_layer_keep_ratio)])
        protocol_cmd.extend(["--decoder-cross-attention-layer-seed", str(target.decoder_cross_attention_layer_seed)])
        protocol_cmd.extend(["--decoder-cross-attention-layer-policy", str(target.decoder_cross_attention_layer_policy)])
    if target.decoder_cross_attention_bottleneck_keep_ratio is not None:
        protocol_cmd.extend(["--decoder-cross-attention-bottleneck-keep-ratio", str(target.decoder_cross_attention_bottleneck_keep_ratio)])
        protocol_cmd.extend(["--decoder-cross-attention-bottleneck-seed", str(target.decoder_cross_attention_bottleneck_seed)])
        protocol_cmd.extend(["--decoder-cross-attention-bottleneck-mode", str(target.decoder_cross_attention_bottleneck_mode)])
    if target.decoder_cross_attention_relation_barrier_keep_ratio is not None:
        protocol_cmd.extend(["--decoder-cross-attention-relation-barrier-keep-ratio", str(target.decoder_cross_attention_relation_barrier_keep_ratio)])
        protocol_cmd.extend(["--decoder-cross-attention-relation-barrier-seed", str(target.decoder_cross_attention_relation_barrier_seed)])
        protocol_cmd.extend(["--decoder-cross-attention-relation-barrier-scale", str(target.decoder_cross_attention_relation_barrier_scale)])
    if target.decoder_readout_keep_ratio is not None:
        protocol_cmd.extend(["--decoder-readout-keep-ratio", str(target.decoder_readout_keep_ratio)])
    if target.decoder_readout_subspace_keep_ratio is not None:
        protocol_cmd.extend(["--decoder-readout-subspace-keep-ratio", str(target.decoder_readout_subspace_keep_ratio)])
        protocol_cmd.extend(["--decoder-readout-subspace-seed", str(target.decoder_readout_subspace_seed)])
        protocol_cmd.extend(["--decoder-readout-subspace-mode", str(target.decoder_readout_subspace_mode)])
    if target.encoder_geometry_transform:
        protocol_cmd.extend(["--encoder-geometry-transform", str(target.encoder_geometry_transform)])
        protocol_cmd.extend(["--encoder-geometry-seed", str(target.encoder_geometry_seed)])
    if target.encoder_geometry_keep_ratio is not None:
        protocol_cmd.extend(["--encoder-geometry-keep-ratio", str(target.encoder_geometry_keep_ratio)])
    if target.encoder_geometry_noise_std is not None:
        protocol_cmd.extend(["--encoder-geometry-noise-std", str(target.encoder_geometry_noise_std)])
    if operationalization_family:
        protocol_cmd.extend(["--operationalization-family", operationalization_family])
    return discovery_cmd, protocol_cmd


def _run_target_materialization(*, env: dict[str, str], discovery_cmd: list[str], protocol_cmd: list[str]) -> None:
    subprocess.run(discovery_cmd, check=True, cwd=PROJECT_ROOT, env=env)
    subprocess.run(protocol_cmd, check=True, cwd=PROJECT_ROOT, env=env)


def main() -> None:
    args = parse_args()
    panel = load_target_panel(args.panel_config)
    env = os.environ.copy()
    hf_endpoint = args.hf_endpoint or panel.hf_endpoint
    operationalization_family = str(panel.metadata.get("operationalization_family") or "").strip()
    if hf_endpoint:
        env["HF_ENDPOINT"] = str(hf_endpoint)

    work_items: list[tuple[list[str], list[str]]] = []
    for target in panel.targets:
        if not target.include_in_adjudication:
            continue
        output_prefix = str(target.phase11_prefix or target.run_name)
        discovery_summary = PROJECT_ROOT / "reports/discovery_runs" / f"{output_prefix}_summary.json"
        protocol_summary = PROJECT_ROOT / "reports/protocol_runs" / f"{output_prefix}_summary.json"
        if not args.force and discovery_summary.exists() and protocol_summary.exists():
            continue
        work_items.append(
            _build_target_commands(
                panel=panel,
                target=target,
                hf_endpoint=hf_endpoint,
                operationalization_family=operationalization_family,
            )
        )

    max_parallel = max(1, int(args.max_parallel))
    if max_parallel == 1 or len(work_items) <= 1:
        for discovery_cmd, protocol_cmd in work_items:
            _run_target_materialization(env=env, discovery_cmd=discovery_cmd, protocol_cmd=protocol_cmd)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_parallel, len(work_items))) as executor:
        futures = [
            executor.submit(
                _run_target_materialization,
                env=env.copy(),
                discovery_cmd=discovery_cmd,
                protocol_cmd=protocol_cmd,
            )
            for discovery_cmd, protocol_cmd in work_items
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
