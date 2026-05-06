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

from analysis.detector_ontology_competition import (  # noqa: E402
    DETECTOR_ORDER,
    STAGE_A,
    STAGE_B,
    build_detector_competition_manifest,
    load_config,
    render_detector_competition_status_markdown,
    summarize_anchorability,
    summarize_detector_competition,
    summarize_family_validity,
    summarize_h1,
    summarize_h2,
    summarize_observability,
    summarize_stage_b,
)
from utils.io import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector ontology competition.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/detector_competition.json"),
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
    detector_families: list[str] | None,
    skip_train: bool,
    force_materialize: bool,
    materialize_max_parallel: int | None,
) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run/train_detector_competition.py"),
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
    for detector_family in detector_families or []:
        cmd.extend(["--detector-family", detector_family])
    _run(cmd, env=os.environ.copy())


def _real_regression_summary(config: dict[str, object], *, validated_detectors: list[str]) -> dict[str, object]:
    if not validated_detectors:
        return {
            "executed": False,
            "validated_detector_count": 0,
            "detector_rows": [],
            "all_validated_detectors_agree_negative": False,
            "family_closure_summary": str(dict(config.get("phase18") or {}).get("family_closure_summary")),
        }
    phase18_summary = load_config(
        PROJECT_ROOT / str(dict(config.get("phase18") or {}).get("family_closure_summary"))
    )
    detector_rows = []
    for detector_family in validated_detectors:
        detector_rows.append(
            {
                "detector_family": detector_family,
                "artifact_reused": True,
                "decision": str(phase18_summary.get("decision")),
                "decision_status": str(phase18_summary.get("decision_status")),
                "agrees_with_negative": (
                    str(phase18_summary.get("decision")) == "T-supported"
                    and str(phase18_summary.get("decision_status")) == "family-final"
                ),
            }
        )
    return {
        "executed": True,
        "validated_detector_count": len(validated_detectors),
        "detector_rows": detector_rows,
        "all_validated_detectors_agree_negative": bool(detector_rows)
        and all(bool(row["agrees_with_negative"]) for row in detector_rows),
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
                str(PROJECT_ROOT / "scripts/run/build_detector_competition_benchmarks.py"),
                "--config",
                args.config,
            ],
            env=os.environ.copy(),
        )

    _train_stage(
        config_path=args.config,
        stage=STAGE_A,
        detector_families=None,
        skip_train=args.skip_train,
        force_materialize=args.force_materialize,
        materialize_max_parallel=materialize_max_parallel,
    )

    manifest = build_detector_competition_manifest(root=PROJECT_ROOT, panel_paths=dict(config["panel_paths"]))
    anchorability = summarize_anchorability(manifest)
    active_stage_by_detector = {detector_family: STAGE_A for detector_family in DETECTOR_ORDER}
    family_validity = summarize_family_validity(
        root=PROJECT_ROOT,
        manifest=manifest,
        active_stage_by_detector=active_stage_by_detector,
    )
    observability = summarize_observability(
        root=PROJECT_ROOT,
        manifest=manifest,
        active_stage_by_detector=active_stage_by_detector,
        max_length=args.max_length,
    )
    h2_summary = summarize_h2(
        root=PROJECT_ROOT,
        manifest=manifest,
        observability_summary=observability,
        active_stage_by_detector=active_stage_by_detector,
        max_length=args.max_length,
    )
    h1_summary = summarize_h1(
        root=PROJECT_ROOT,
        manifest=manifest,
        observability_summary=observability,
        active_stage_by_detector=active_stage_by_detector,
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
            detector_families=list(stage_b_summary["failed_detectors"]),
            skip_train=args.skip_train,
            force_materialize=args.force_materialize,
            materialize_max_parallel=materialize_max_parallel,
        )
        for detector_family in list(stage_b_summary["failed_detectors"]):
            active_stage_by_detector[detector_family] = STAGE_B
        stage_b_summary["stage_b_executed"] = True
        family_validity = summarize_family_validity(
            root=PROJECT_ROOT,
            manifest=manifest,
            active_stage_by_detector=active_stage_by_detector,
        )
        observability = summarize_observability(
            root=PROJECT_ROOT,
            manifest=manifest,
            active_stage_by_detector=active_stage_by_detector,
            max_length=args.max_length,
        )
        h2_summary = summarize_h2(
            root=PROJECT_ROOT,
            manifest=manifest,
            observability_summary=observability,
            active_stage_by_detector=active_stage_by_detector,
            max_length=args.max_length,
        )
        h1_summary = summarize_h1(
            root=PROJECT_ROOT,
            manifest=manifest,
            observability_summary=observability,
            active_stage_by_detector=active_stage_by_detector,
            max_length=args.max_length,
        )

    pre_regression_competition = summarize_detector_competition(
        family_validity_summary=family_validity,
        anchorability_summary=anchorability,
        observability_summary=observability,
        h2_summary=h2_summary,
        h1_summary=h1_summary,
        real_regression_summary={"executed": False, "all_validated_detectors_agree_negative": False},
    )
    real_regression = _real_regression_summary(
        config,
        validated_detectors=list(pre_regression_competition.get("validated_detectors") or []),
    )
    competition_summary = summarize_detector_competition(
        family_validity_summary=family_validity,
        anchorability_summary=anchorability,
        observability_summary=observability,
        h2_summary=h2_summary,
        h1_summary=h1_summary,
        real_regression_summary=real_regression,
    )

    benchmark_summary = load_config(PROJECT_ROOT / "reports/detector_competition/benchmark_builder_summary.json")
    write_json(PROJECT_ROOT / "reports/detector_competition/manifest_summary.json", manifest)
    write_json(PROJECT_ROOT / "reports/detector_competition/family_validity_summary.json", family_validity)
    write_json(PROJECT_ROOT / "reports/detector_competition/anchorability_summary.json", anchorability)
    write_json(PROJECT_ROOT / "reports/detector_competition/observability_summary.json", observability)
    write_json(PROJECT_ROOT / "reports/detector_competition/h2_summary.json", h2_summary)
    write_json(PROJECT_ROOT / "reports/detector_competition/h1_summary.json", h1_summary)
    write_json(PROJECT_ROOT / "reports/detector_competition/stageb_summary.json", stage_b_summary)
    write_json(PROJECT_ROOT / "reports/detector_competition/real_family_regression_summary.json", real_regression)
    write_json(PROJECT_ROOT / "reports/detector_competition/detector_ontology_competition_summary.json", competition_summary)
    (PROJECT_ROOT / "reports/DETECTOR_COMPETITION_STATUS.md").write_text(
        render_detector_competition_status_markdown(
            benchmark_summary=benchmark_summary,
            manifest_summary=manifest,
            family_validity_summary=family_validity,
            anchorability_summary=anchorability,
            observability_summary=observability,
            h2_summary=h2_summary,
            h1_summary=h1_summary,
            stage_b_summary=stage_b_summary,
            real_regression_summary=real_regression,
            competition_summary=competition_summary,
        ),
        encoding="utf-8",
    )
    print(competition_summary)


if __name__ == "__main__":
    main()
