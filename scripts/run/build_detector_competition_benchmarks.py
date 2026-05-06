from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.detector_ontology_competition import (  # noqa: E402
    build_detector_competition_benchmarks,
    build_detector_competition_panels,
    load_config,
)
from utils.io import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build detector-competition benchmarks and panels.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/detector_competition.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    benchmark_summary = build_detector_competition_benchmarks(config)
    panel_summary = build_detector_competition_panels(config)
    write_json(PROJECT_ROOT / "reports/detector_competition/benchmark_builder_summary.json", benchmark_summary)
    write_json(PROJECT_ROOT / "reports/detector_competition/panel_builder_summary.json", panel_summary)
    print({"benchmarks": benchmark_summary["benchmark_count"], "panels": panel_summary["panel_count"]})


if __name__ == "__main__":
    main()
