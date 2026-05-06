from __future__ import annotations

import json
from pathlib import Path

import analysis.detector_truthfulness as p22
from analysis.detector_truthfulness import (
    CLASS_ORDER,
    FAMILY_ARGUMENT_H1,
    FAMILY_ARGUMENT_H2,
    FAMILY_ARGUMENT_NULL,
    FAMILY_PREDICATE_SANITY,
    STAGE_A,
    STAGE_B,
    _evaluation_target_rows,
    build_truthfulness_benchmarks,
    build_truthfulness_manifest,
    build_truthfulness_panels,
    summarize_family_validity,
    summarize_observability,
    summarize_truthfulness_detector,
)


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "project_root": str(tmp_path),
        "seed": 22,
        "family_counts": {"train": 1, "dev": 1, "gen": 1},
        "panel_paths": {
            "interaction": {
                "stage_a": "configs/experiments/detector_truthfulness_c1.json",
                "stage_b": "configs/experiments/detector_truthfulness_c1_stage_b.json",
            },
            "cross_step_bridge": {
                "stage_a": "configs/experiments/detector_truthfulness_c2.json",
                "stage_b": "configs/experiments/detector_truthfulness_c2_stage_b.json",
            },
            "query_source_relation": {
                "stage_a": "configs/experiments/detector_truthfulness_c3.json",
                "stage_b": "configs/experiments/detector_truthfulness_c3_stage_b.json",
            },
        },
        "phase18": {"family_closure_summary": "reports/family_closure/family_closure_summary.json"},
    }


def test_build_truthfulness_benchmarks_writes_processed_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = build_truthfulness_benchmarks(config)
    assert summary["benchmark_count"] == 3
    for row in summary["benchmark_rows"]:
        processed_path = tmp_path / row["processed_path"]
        assert processed_path.exists()
        first_row = json.loads(processed_path.read_text(encoding="utf-8").splitlines()[0])
        assert first_row["metadata"]["truthfulness_benchmark"] is True
        assert row["canonical_example_count"] > 0


def test_build_truthfulness_panels_emit_expected_target_counts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = build_truthfulness_panels(config)
    assert summary["panel_count"] == 3
    for row in summary["panel_rows"]:
        assert row["stage_a_target_count"] == 8
        assert row["stage_b_target_count"] == 6
        stage_a_panel = json.loads((tmp_path / row["stage_a_panel_path"]).read_text(encoding="utf-8"))
        assert len(stage_a_panel["targets"]) == 8
        assert all("truthfulness_class" in target for target in stage_a_panel["targets"])
        assert all("paired_null_id" in target for target in stage_a_panel["targets"])


def test_build_truthfulness_manifest_reads_stage_panels(tmp_path: Path) -> None:
    config = _config(tmp_path)
    build_truthfulness_panels(config)
    manifest = build_truthfulness_manifest(root=tmp_path, panel_paths=config["panel_paths"])
    assert manifest["class_count"] == 3
    by_class = {row["object_class"]: row for row in manifest["class_rows"]}
    for object_class in CLASS_ORDER:
        assert len(by_class[object_class]["stage_rows"][STAGE_A]["target_rows"]) == 8
        assert len(by_class[object_class]["stage_rows"][STAGE_B]["target_rows"]) == 6


def test_truthfulness_detector_requires_all_components() -> None:
    detector = summarize_truthfulness_detector(
        family_validity_summary={
            "class_rows": [
                {"object_class": "interaction", "passed": True},
                {"object_class": "cross_step_bridge", "passed": True},
                {"object_class": "query_source_relation", "passed": True},
            ]
        },
        anchorability_summary={
            "class_rows": [
                {"object_class": "interaction", "passed": True},
                {"object_class": "cross_step_bridge", "passed": True},
                {"object_class": "query_source_relation", "passed": True},
            ]
        },
        observability_summary={
            "class_rows": [
                {"object_class": "interaction", "passed": True, "argument_specific_observability_failure": False},
                {"object_class": "cross_step_bridge", "passed": False, "argument_specific_observability_failure": True},
                {"object_class": "query_source_relation", "passed": True, "argument_specific_observability_failure": False},
            ]
        },
        h2_summary={
            "class_rows": [
                {"object_class": "interaction", "passed": True},
                {"object_class": "cross_step_bridge", "passed": False},
                {"object_class": "query_source_relation", "passed": True},
            ]
        },
        h1_summary={
            "class_rows": [
                {"object_class": "interaction", "positive_passed": True, "null_passed": False},
                {"object_class": "cross_step_bridge", "positive_passed": False, "null_passed": False},
                {"object_class": "query_source_relation", "positive_passed": True, "null_passed": False},
            ]
        },
        real_regression_summary={"preserved_phase18_t_final": True},
    )
    assert detector["detectability_validated"] is False
    assert detector["decision"] == "retain_phase18"
    assert "cross_step_bridge:argument_specific_observability_failure" in detector["failure_reasons"]


def test_evaluation_target_rows_preserve_stage_a_predicate_for_stage_b() -> None:
    class_row = {
        "stage_rows": {
            STAGE_A: {
                "target_rows": [
                    {"run_name": "pred_seed1", "panel_regime": FAMILY_PREDICATE_SANITY},
                    {"run_name": "pred_seed2", "panel_regime": FAMILY_PREDICATE_SANITY},
                    {"run_name": "arg_h2_seed1", "panel_regime": FAMILY_ARGUMENT_H2},
                ]
            },
            STAGE_B: {
                "target_rows": [
                    {"run_name": "arg_h2_stageb_seed1", "panel_regime": FAMILY_ARGUMENT_H2},
                    {"run_name": "arg_h1_stageb_seed1", "panel_regime": FAMILY_ARGUMENT_H1},
                    {"run_name": "arg_null_stageb_seed1", "panel_regime": FAMILY_ARGUMENT_NULL},
                ]
            },
        }
    }
    target_rows = _evaluation_target_rows(class_row=class_row, stage_name=STAGE_B)
    assert [row["run_name"] for row in target_rows] == [
        "pred_seed1",
        "pred_seed2",
        "arg_h2_stageb_seed1",
        "arg_h1_stageb_seed1",
        "arg_null_stageb_seed1",
    ]


def test_stage_b_family_validity_reuses_stage_a_predicate_rows(monkeypatch) -> None:
    manifest = {
        "class_rows": [
            {
                "object_class": "interaction",
                "benchmark_name": "detector_truthfulness_c1",
                "stage_rows": {
                    STAGE_A: {
                        "target_rows": [
                            {"run_name": "pred_seed1", "panel_regime": FAMILY_PREDICATE_SANITY, "truthfulness_seed": 1, "truthfulness_family": FAMILY_PREDICATE_SANITY},
                            {"run_name": "pred_seed2", "panel_regime": FAMILY_PREDICATE_SANITY, "truthfulness_seed": 2, "truthfulness_family": FAMILY_PREDICATE_SANITY},
                        ]
                    },
                    STAGE_B: {
                        "target_rows": [
                            {"run_name": "arg_h2_seed1", "panel_regime": FAMILY_ARGUMENT_H2, "truthfulness_seed": 1, "truthfulness_family": FAMILY_ARGUMENT_H2},
                            {"run_name": "arg_h1_seed1", "panel_regime": FAMILY_ARGUMENT_H1, "truthfulness_seed": 1, "truthfulness_family": FAMILY_ARGUMENT_H1},
                            {"run_name": "arg_null_seed1", "panel_regime": FAMILY_ARGUMENT_NULL, "truthfulness_seed": 1, "truthfulness_family": FAMILY_ARGUMENT_NULL},
                            {"run_name": "arg_h2_seed2", "panel_regime": FAMILY_ARGUMENT_H2, "truthfulness_seed": 2, "truthfulness_family": FAMILY_ARGUMENT_H2},
                            {"run_name": "arg_h1_seed2", "panel_regime": FAMILY_ARGUMENT_H1, "truthfulness_seed": 2, "truthfulness_family": FAMILY_ARGUMENT_H1},
                            {"run_name": "arg_null_seed2", "panel_regime": FAMILY_ARGUMENT_NULL, "truthfulness_seed": 2, "truthfulness_family": FAMILY_ARGUMENT_NULL},
                        ]
                    },
                },
            }
        ]
    }

    metric_by_case = {
        FAMILY_PREDICATE_SANITY: {"count": 1, "exact_match": 1.0, "clause_f1": 1.0},
        FAMILY_ARGUMENT_H2: {"count": 1, "exact_match": 0.9, "clause_f1": 0.9},
        FAMILY_ARGUMENT_H1: {"count": 1, "exact_match": 0.7, "clause_f1": 0.7},
        FAMILY_ARGUMENT_NULL: {"count": 1, "exact_match": 0.1, "clause_f1": 0.1},
    }

    monkeypatch.setattr(p22, "_benchmark_index", lambda root, benchmark_name: {})
    monkeypatch.setattr(
        p22,
        "_prediction_records",
        lambda root, run_name: [{"generalization_case": run_name}],
    )
    monkeypatch.setattr(
        p22,
        "_filter_records_for_family",
        lambda records, family_case: [{"generalization_case": family_case}],
    )
    monkeypatch.setattr(
        p22,
        "_family_metric",
        lambda records: metric_by_case[str(records[0]["generalization_case"])],
    )

    summary = summarize_family_validity(root=Path("/tmp"), manifest=manifest, active_stage_by_class={"interaction": STAGE_B})
    row = summary["class_rows"][0]
    assert row["passed"] is True
    assert row["stage"] == STAGE_B


def test_stage_b_observability_reuses_stage_a_predicate_rows(monkeypatch) -> None:
    manifest = {
        "class_rows": [
            {
                "object_class": "interaction",
                "stage_rows": {
                    STAGE_A: {
                        "target_rows": [
                            {"run_name": "pred_seed1", "panel_regime": FAMILY_PREDICATE_SANITY, "truthfulness_seed": 1, "phase11_prefix": "pred_seed1", "intended_step_key": "predicate_value"},
                            {"run_name": "pred_seed2", "panel_regime": FAMILY_PREDICATE_SANITY, "truthfulness_seed": 2, "phase11_prefix": "pred_seed2", "intended_step_key": "predicate_value"},
                        ]
                    },
                    STAGE_B: {
                        "target_rows": [
                            {"run_name": "arg_h2_seed1", "panel_regime": FAMILY_ARGUMENT_H2, "truthfulness_seed": 1, "phase11_prefix": "arg_h2_seed1", "intended_step_key": "agent_value"},
                            {"run_name": "arg_h1_seed1", "panel_regime": FAMILY_ARGUMENT_H1, "truthfulness_seed": 1, "phase11_prefix": "arg_h1_seed1", "intended_step_key": "agent_value"},
                            {"run_name": "arg_null_seed1", "panel_regime": FAMILY_ARGUMENT_NULL, "truthfulness_seed": 1, "phase11_prefix": "arg_null_seed1", "intended_step_key": "agent_value"},
                            {"run_name": "arg_h2_seed2", "panel_regime": FAMILY_ARGUMENT_H2, "truthfulness_seed": 2, "phase11_prefix": "arg_h2_seed2", "intended_step_key": "agent_value"},
                            {"run_name": "arg_h1_seed2", "panel_regime": FAMILY_ARGUMENT_H1, "truthfulness_seed": 2, "phase11_prefix": "arg_h1_seed2", "intended_step_key": "agent_value"},
                            {"run_name": "arg_null_seed2", "panel_regime": FAMILY_ARGUMENT_NULL, "truthfulness_seed": 2, "phase11_prefix": "arg_null_seed2", "intended_step_key": "agent_value"},
                        ]
                    },
                },
            }
        ]
    }

    monkeypatch.setattr(p22, "_bundle_from_prefix", lambda root, phase11_prefix: {"phase11_prefix": phase11_prefix})
    monkeypatch.setattr(
        p22,
        "_evaluate_run_observability",
        lambda **kwargs: {
            "aligner": kwargs["aligner_name"],
            "row_count": 1,
            "intended_step_alignment_rate": 1.0,
            "unintended_step_alignment_rate": 0.0,
            "passed": True,
        },
    )

    summary = summarize_observability(
        root=Path("/tmp"),
        manifest=manifest,
        active_stage_by_class={"interaction": STAGE_B},
        max_length=32,
    )
    row = summary["class_rows"][0]
    assert row["passed"] is True
    assert row["predicate_sanity_passed"] is True
    assert row["argument_families_passed"] is True
