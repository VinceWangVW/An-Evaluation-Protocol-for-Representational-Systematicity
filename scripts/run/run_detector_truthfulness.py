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

from analysis.detector_truthfulness import (
    CLASS_ORDER,
    STAGE_A,
    STAGE_B,
    build_truthfulness_manifest,
    load_config,
    render_detector_truthfulness_status_markdown,
    summarize_anchorability,
    summarize_family_validity,
    summarize_h1,
    summarize_h2,
    summarize_observability,
    summarize_stage_b,
    summarize_truthfulness_detector,
)
from utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector-truthfulness validation.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/detector_truthfulness.json"),
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-materialize", action="store_true")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--materialize-max-parallel", type=int, default=None)
    return parser.parse_args()


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


def _train_stage(
    *,
    config_path: str,
    stage: str,
    object_classes: list[str] | None,
    skip_train: bool,
    force_materialize: bool,
    materialize_max_parallel: int | None,
) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run/train_truthfulness_families.py"),
        "--config",
        config_path,
        "--stage",
        stage,
    ]
    if skip_train:
        cmd.append("--skip-train")
    if force_materialize:
        cmd.append("--force-materialize")
    if materialize_max_parallel is not None:
        cmd.extend(["--materialize-max-parallel", str(max(1, materialize_max_parallel))])
    for object_class in object_classes or []:
        cmd.extend(["--object-class", object_class])
    _run(cmd, env=os.environ.copy())


def _real_regression_summary(config: dict[str, object], *, executed: bool) -> dict[str, object]:
    phase18_summary = load_config(
        PROJECT_ROOT / str(dict(config.get("phase18") or {}).get("family_closure_summary"))
    )
    return {
        "executed": executed,
        "preserved_phase18_t_final": (
            str(phase18_summary.get("decision")) == "T-supported"
            and str(phase18_summary.get("decision_status")) == "family-final"
        ),
        "family_closure_summary": str(dict(config.get("phase18") or {}).get("family_closure_summary")),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    materialize_max_parallel = (
        args.materialize_max_parallel
        if args.materialize_max_parallel is not None
        else int(config.get("materialize_max_parallel") or 1)
    )
    if not args.skip_build:
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/run/build_truthfulness_benchmarks.py"),
                "--config",
                args.config,
            ],
            env=os.environ.copy(),
        )

    _train_stage(
        config_path=args.config,
        stage=STAGE_A,
        object_classes=None,
        skip_train=args.skip_train,
        force_materialize=args.force_materialize,
        materialize_max_parallel=materialize_max_parallel,
    )

    manifest = build_truthfulness_manifest(root=PROJECT_ROOT, panel_paths=dict(config["panel_paths"]))
    anchorability = summarize_anchorability(manifest)
    active_stage_by_class = {object_class: STAGE_A for object_class in CLASS_ORDER}
    family_validity = summarize_family_validity(
        root=PROJECT_ROOT,
        manifest=manifest,
        active_stage_by_class=active_stage_by_class,
    )
    observability = summarize_observability(
        root=PROJECT_ROOT,
        manifest=manifest,
        active_stage_by_class=active_stage_by_class,
        max_length=args.max_length,
    )
    h2_summary = summarize_h2(
        root=PROJECT_ROOT,
        manifest=manifest,
        observability_summary=observability,
        active_stage_by_class=active_stage_by_class,
        max_length=args.max_length,
    )
    h1_summary = summarize_h1(
        root=PROJECT_ROOT,
        manifest=manifest,
        observability_summary=observability,
        active_stage_by_class=active_stage_by_class,
        max_length=args.max_length,
    )
    stage_b_summary = summarize_stage_b(
        family_validity_summary=family_validity,
        anchorability_summary=anchorability,
        observability_summary=observability,
        h2_summary=h2_summary,
        h1_summary=h1_summary,
    )

    if bool(stage_b_summary.get("triggered")):
        _train_stage(
            config_path=args.config,
            stage=STAGE_B,
            object_classes=list(stage_b_summary["failed_classes"]),
            skip_train=args.skip_train,
            force_materialize=args.force_materialize,
            materialize_max_parallel=materialize_max_parallel,
        )
        for object_class in list(stage_b_summary["failed_classes"]):
            active_stage_by_class[object_class] = STAGE_B
        stage_b_summary["stage_b_executed"] = True
        family_validity = summarize_family_validity(
            root=PROJECT_ROOT,
            manifest=manifest,
            active_stage_by_class=active_stage_by_class,
        )
        observability = summarize_observability(
            root=PROJECT_ROOT,
            manifest=manifest,
            active_stage_by_class=active_stage_by_class,
            max_length=args.max_length,
        )
        h2_summary = summarize_h2(
            root=PROJECT_ROOT,
            manifest=manifest,
            observability_summary=observability,
            active_stage_by_class=active_stage_by_class,
            max_length=args.max_length,
        )
        h1_summary = summarize_h1(
            root=PROJECT_ROOT,
            manifest=manifest,
            observability_summary=observability,
            active_stage_by_class=active_stage_by_class,
            max_length=args.max_length,
        )

    ready_for_real_regression = (
        family_validity.get("passed_count") == len(CLASS_ORDER)
        and anchorability.get("passed_count") == len(CLASS_ORDER)
        and observability.get("passed_count") == len(CLASS_ORDER)
        and h2_summary.get("passed_count") == len(CLASS_ORDER)
        and h1_summary.get("positive_passed_count") == len(CLASS_ORDER)
    )
    real_regression = _real_regression_summary(config, executed=bool(ready_for_real_regression))
    detector = summarize_truthfulness_detector(
        family_validity_summary=family_validity,
        anchorability_summary=anchorability,
        observability_summary=observability,
        h2_summary=h2_summary,
        h1_summary=h1_summary,
        real_regression_summary=real_regression,
    )

    benchmark_summary = load_config(PROJECT_ROOT / "reports/detector_truthfulness/benchmark_builder_summary.json")
    write_json(PROJECT_ROOT / "reports/detector_truthfulness/manifest_summary.json", manifest)
    write_json(PROJECT_ROOT / "reports/detector_truthfulness/family_validity_summary.json", family_validity)
    write_json(PROJECT_ROOT / "reports/detector_truthfulness/anchorability_summary.json", anchorability)
    write_json(PROJECT_ROOT / "reports/detector_truthfulness/observability_summary.json", observability)
    write_json(PROJECT_ROOT / "reports/detector_truthfulness/h2_summary.json", h2_summary)
    write_json(PROJECT_ROOT / "reports/detector_truthfulness/h1_summary.json", h1_summary)
    write_json(PROJECT_ROOT / "reports/detector_truthfulness/stageb_summary.json", stage_b_summary)
    write_json(PROJECT_ROOT / "reports/detector_truthfulness/real_family_regression_summary.json", real_regression)
    write_json(PROJECT_ROOT / "reports/detector_truthfulness/detector_truthfulness_summary.json", detector)
    (PROJECT_ROOT / "reports/DETECTOR_TRUTHFULNESS_STATUS.md").write_text(
        render_detector_truthfulness_status_markdown(
            benchmark_summary=benchmark_summary,
            manifest_summary=manifest,
            family_validity_summary=family_validity,
            anchorability_summary=anchorability,
            observability_summary=observability,
            h2_summary=h2_summary,
            h1_summary=h1_summary,
            stage_b_summary=stage_b_summary,
            real_regression_summary=real_regression,
            detector_summary=detector,
        ),
        encoding="utf-8",
    )
    print(detector)


if __name__ == "__main__":
    main()
