from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.detector_truthfulness import build_truthfulness_benchmarks, build_truthfulness_panels, load_config
from utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build detector-truthfulness benchmarks and panel configs.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/detector_truthfulness.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["project_root"] = str(PROJECT_ROOT)
    benchmark_summary = build_truthfulness_benchmarks(config)
    panel_summary = build_truthfulness_panels(config)
    write_json(PROJECT_ROOT / "reports/detector_truthfulness/benchmark_builder_summary.json", benchmark_summary)
    write_json(PROJECT_ROOT / "reports/detector_truthfulness/panel_builder_summary.json", panel_summary)
    print(
        json.dumps(
            {
                "benchmark_count": benchmark_summary["benchmark_count"],
                "panel_count": panel_summary["panel_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
