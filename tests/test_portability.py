from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_release_configs_leave_hub_endpoint_to_the_user() -> None:
    for name in ("detector_truthfulness", "detector_competition"):
        config = json.loads((ROOT / "configs/experiments" / f"{name}.json").read_text())
        assert config["hf_endpoint"] is None


def test_released_analysis_imports_resolve() -> None:
    state_analysis = importlib.import_module("analysis.state_analysis")
    behavior_analysis = importlib.import_module("analysis.behavior_analysis")
    assert state_analysis.load_phase11_binding_row is behavior_analysis.load_phase11_binding_row
    assert state_analysis.load_phase11_geometry_row is behavior_analysis.load_phase11_geometry_row


def test_shipped_reports_and_operational_files_have_no_machine_dependencies() -> None:
    forbidden = ("/root/" + "autodl-tmp/", "https://hf-" + "mirror.com")
    failures = []
    for directory in ("src", "scripts", "configs", "reports", "artifacts"):
        for path in (ROOT / directory).rglob("*"):
            if path.suffix not in {".py", ".json", ".jsonl", ".sh"}:
                continue
            if any(value in path.read_text(encoding="utf-8") for value in forbidden):
                failures.append(str(path.relative_to(ROOT)))
    assert not failures, failures


def test_model_runner_help_works_from_another_directory(tmp_path: Path) -> None:
    # Exercises the public entry-point imports without requesting model weights.
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run/run_protocol.py"), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--phase11-prefix" in result.stdout
