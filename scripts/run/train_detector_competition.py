from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.detector_ontology_competition import DETECTOR_ORDER, STAGE_A, STAGE_B, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and materialize Phase 23 detector competition families.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/detector_ontology_competition.json"),
    )
    parser.add_argument("--stage", choices=(STAGE_A, STAGE_B), default=STAGE_A)
    parser.add_argument("--detector-family", action="append", default=[])
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-materialize", action="store_true")
    parser.add_argument("--materialize-max-parallel", type=int, default=1)
    return parser.parse_args()


def _run_panel(
    panel_path: Path,
    *,
    hf_endpoint: str | None,
    skip_train: bool,
    force_materialize: bool,
    materialize_max_parallel: int,
) -> None:
    env = os.environ.copy()
    if hf_endpoint:
        env["HF_ENDPOINT"] = hf_endpoint
    if not skip_train:
        train_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run/train_target_families.py"),
            "--panel-config",
            str(panel_path),
        ]
        if hf_endpoint:
            train_cmd.extend(["--hf-endpoint", hf_endpoint])
        subprocess.run(train_cmd, check=True, cwd=PROJECT_ROOT, env=env)
    materialize_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run/materialize_target_runs.py"),
        "--panel-config",
        str(panel_path),
    ]
    if hf_endpoint:
        materialize_cmd.extend(["--hf-endpoint", hf_endpoint])
    if force_materialize:
        materialize_cmd.append("--force")
    materialize_cmd.extend(["--max-parallel", str(max(1, materialize_max_parallel))])
    subprocess.run(materialize_cmd, check=True, cwd=PROJECT_ROOT, env=env)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    panel_paths = dict(config.get("panel_paths") or {})
    hf_endpoint = str(config.get("hf_endpoint") or "").strip() or None
    detectors = args.detector_family or list(DETECTOR_ORDER)
    for detector_family in detectors:
        panel_path = PROJECT_ROOT / str(panel_paths[detector_family][args.stage])
        _run_panel(
            panel_path,
            hf_endpoint=hf_endpoint,
            skip_train=args.skip_train,
            force_materialize=args.force_materialize,
            materialize_max_parallel=args.materialize_max_parallel,
        )


if __name__ == "__main__":
    main()
