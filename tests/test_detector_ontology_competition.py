from __future__ import annotations

import json
from pathlib import Path

import analysis.detector_ontology_competition as p23
from analysis.detector_ontology_competition import (
    D1_ARGUMENT_LOCUS,
    D2_PREDICATE_CONDITIONED_RELATION,
    D3_CROSS_STEP_BRIDGE,
    DETECTOR_ORDER,
    FAMILY_DETECTOR_H1,
    FAMILY_DETECTOR_H2,
    FAMILY_DETECTOR_NULL,
    FAMILY_PREDICATE_SANITY,
    STAGE_A,
    STAGE_B,
    _evaluation_target_rows,
    _score_bridge_observability,
    _score_step_observability,
    build_detector_competition_benchmarks,
    build_detector_competition_manifest,
    build_detector_competition_panels,
    detector_registry,
    load_config,
    summarize_detector_competition,
)


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "project_root": str(tmp_path),
        "seed": 23,
        "family_counts": {"train": 1, "dev": 1, "gen": 1},
        "panel_paths": {
            D1_ARGUMENT_LOCUS: {
                "stage_a": "configs/experiments/detector_competition_d1.json",
                "stage_b": "configs/experiments/detector_competition_d1_stage_b.json",
            },
            D2_PREDICATE_CONDITIONED_RELATION: {
                "stage_a": "configs/experiments/detector_competition_d2.json",
                "stage_b": "configs/experiments/detector_competition_d2_stage_b.json",
            },
            D3_CROSS_STEP_BRIDGE: {
                "stage_a": "configs/experiments/detector_competition_d3.json",
                "stage_b": "configs/experiments/detector_competition_d3_stage_b.json",
            },
        },
        "phase18": {"family_closure_summary": "reports/family_closure/family_closure_summary.json"},
    }


def test_detector_registry_has_fixed_family_mapping() -> None:
    registry = detector_registry()
    assert list(registry) == list(DETECTOR_ORDER)
    assert registry[D1_ARGUMENT_LOCUS]["object_class"] == "interaction"
    assert registry[D2_PREDICATE_CONDITIONED_RELATION]["object_class"] == "query_source_relation"
    assert registry[D3_CROSS_STEP_BRIDGE]["object_class"] == "cross_step_bridge"


def test_load_config_infers_project_root_from_config_path(tmp_path: Path) -> None:
    config_path = tmp_path / "configs/experiments/detector_ontology_competition.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"seed": 23}), encoding="utf-8")
    loaded = load_config(config_path)
    assert loaded["project_root"] == str(tmp_path)


def test_build_detector_competition_benchmarks_writes_processed_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = build_detector_competition_benchmarks(config)
    assert summary["benchmark_count"] == 3
    for row in summary["benchmark_rows"]:
        processed_path = tmp_path / row["processed_path"]
        assert processed_path.exists()
        first_row = json.loads(processed_path.read_text(encoding="utf-8").splitlines()[0])
        assert first_row["metadata"]["detector_competition_benchmark"] is True
        assert first_row["metadata"]["detector_family"] in DETECTOR_ORDER
        assert row["canonical_example_count"] > 0


def test_build_detector_competition_panels_emit_expected_target_counts_and_metadata(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = build_detector_competition_panels(config)
    assert summary["panel_count"] == 3
    for row in summary["panel_rows"]:
        assert row["stage_a_target_count"] == 8
        assert row["stage_b_target_count"] == 6
        stage_a_panel = json.loads((tmp_path / row["stage_a_panel_path"]).read_text(encoding="utf-8"))
        assert len(stage_a_panel["targets"]) == 8
        for target in stage_a_panel["targets"]:
            assert "detector_family" in target
            assert "detector_seed" in target
            assert "truthfulness_family" in target
            assert "intended_observable" in target
            assert "source_step_key" in target
            assert "target_step_key" in target
            assert "paired_null_id" in target


def test_build_detector_competition_manifest_reads_stage_panels(tmp_path: Path) -> None:
    config = _config(tmp_path)
    build_detector_competition_panels(config)
    manifest = build_detector_competition_manifest(root=tmp_path, panel_paths=config["panel_paths"])
    assert manifest["detector_count"] == 3
    by_detector = {row["detector_family"]: row for row in manifest["detector_rows"]}
    for detector_family in DETECTOR_ORDER:
        assert len(by_detector[detector_family]["stage_rows"][STAGE_A]["target_rows"]) == 8
        assert len(by_detector[detector_family]["stage_rows"][STAGE_B]["target_rows"]) == 6


def test_step_local_observability_passes_for_clean_intended_alignment() -> None:
    rows = [
        {"aligned_steps": ["agent_value"]},
        {"aligned_steps": ["agent_value"]},
        {"aligned_steps": ["agent_value"]},
    ]
    summary = _score_step_observability(rows, intended_step_key="agent_value")
    assert summary["passed"] is True
    assert summary["intended_step_alignment_rate"] == 1.0
    assert summary["unintended_step_alignment_rate"] == 0.0


def test_bridge_observability_requires_source_target_and_coalignment() -> None:
    rows = [
        {"aligned_steps": ["predicate_value", "agent_value"]},
        {"aligned_steps": ["predicate_value", "agent_value"]},
        {"aligned_steps": ["predicate_value", "agent_value"]},
    ]
    summary = _score_bridge_observability(rows, source_step_key="predicate_value", target_step_key="agent_value")
    assert summary["passed"] is True
    assert summary["source_step_alignment_rate"] == 1.0
    assert summary["target_step_alignment_rate"] == 1.0
    assert summary["bridge_coalignment_rate"] == 1.0
    assert summary["unintended_extra_step_alignment_rate"] == 0.0


def test_evaluation_target_rows_reuse_stage_a_predicate_in_stage_b() -> None:
    detector_row = {
        "stage_rows": {
            STAGE_A: {
                "target_rows": [
                    {"run_name": "pred_seed1", "panel_regime": FAMILY_PREDICATE_SANITY},
                    {"run_name": "pred_seed2", "panel_regime": FAMILY_PREDICATE_SANITY},
                ]
            },
            STAGE_B: {
                "target_rows": [
                    {"run_name": "h2_stageb_seed1", "panel_regime": FAMILY_DETECTOR_H2},
                    {"run_name": "h1_stageb_seed1", "panel_regime": FAMILY_DETECTOR_H1},
                    {"run_name": "null_stageb_seed1", "panel_regime": FAMILY_DETECTOR_NULL},
                ]
            },
        }
    }
    rows = _evaluation_target_rows(detector_row=detector_row, stage_name=STAGE_B)
    assert [row["run_name"] for row in rows] == [
        "pred_seed1",
        "pred_seed2",
        "h2_stageb_seed1",
        "h1_stageb_seed1",
        "null_stageb_seed1",
    ]


def test_detector_competition_decision_retain_when_no_validated_detector() -> None:
    summary = summarize_detector_competition(
        family_validity_summary={"detector_rows": [{"detector_family": detector, "passed": False} for detector in DETECTOR_ORDER]},
        anchorability_summary={"detector_rows": [{"detector_family": detector, "passed": True} for detector in DETECTOR_ORDER]},
        observability_summary={"detector_rows": [{"detector_family": detector, "passed": False, "detector_specific_observability_failure": False} for detector in DETECTOR_ORDER]},
        h2_summary={"detector_rows": [{"detector_family": detector, "passed": False} for detector in DETECTOR_ORDER]},
        h1_summary={"detector_rows": [{"detector_family": detector, "positive_passed": False, "null_passed": False} for detector in DETECTOR_ORDER]},
        real_regression_summary={"executed": False, "all_validated_detectors_agree_negative": False},
    )
    assert summary["decision"] == "retain_phase18"
    assert summary["strongest_claim_decision"] == "retain_phase18"


def test_detector_competition_decision_inconclusive_when_validated_detectors_disagree() -> None:
    detector_rows = [
        {"detector_family": D1_ARGUMENT_LOCUS, "passed": True},
        {"detector_family": D2_PREDICATE_CONDITIONED_RELATION, "passed": True},
        {"detector_family": D3_CROSS_STEP_BRIDGE, "passed": False},
    ]
    h1_rows = [
        {"detector_family": D1_ARGUMENT_LOCUS, "positive_passed": True, "null_passed": False},
        {"detector_family": D2_PREDICATE_CONDITIONED_RELATION, "positive_passed": True, "null_passed": False},
        {"detector_family": D3_CROSS_STEP_BRIDGE, "positive_passed": False, "null_passed": False},
    ]
    summary = summarize_detector_competition(
        family_validity_summary={"detector_rows": detector_rows},
        anchorability_summary={"detector_rows": detector_rows},
        observability_summary={
            "detector_rows": [
                {"detector_family": D1_ARGUMENT_LOCUS, "passed": True, "detector_specific_observability_failure": False},
                {"detector_family": D2_PREDICATE_CONDITIONED_RELATION, "passed": True, "detector_specific_observability_failure": False},
                {"detector_family": D3_CROSS_STEP_BRIDGE, "passed": False, "detector_specific_observability_failure": True},
            ]
        },
        h2_summary={"detector_rows": detector_rows},
        h1_summary={"detector_rows": h1_rows},
        real_regression_summary={"executed": True, "all_validated_detectors_agree_negative": False},
    )
    assert summary["decision"] == "detector_competition_inconclusive"
    assert summary["strongest_claim_decision"] == "retain_phase18"


def test_detector_competition_supports_upgrade_when_all_validated_detectors_agree_negative() -> None:
    detector_rows = [{"detector_family": detector, "passed": True} for detector in DETECTOR_ORDER]
    h1_rows = [{"detector_family": detector, "positive_passed": True, "null_passed": False} for detector in DETECTOR_ORDER]
    summary = summarize_detector_competition(
        family_validity_summary={"detector_rows": detector_rows},
        anchorability_summary={"detector_rows": detector_rows},
        observability_summary={"detector_rows": [{"detector_family": detector, "passed": True, "detector_specific_observability_failure": False} for detector in DETECTOR_ORDER]},
        h2_summary={"detector_rows": detector_rows},
        h1_summary={"detector_rows": h1_rows},
        real_regression_summary={"executed": True, "all_validated_detectors_agree_negative": True},
    )
    assert summary["decision"] == "support_ontology_final_upgrade"
    assert summary["strongest_claim_decision"] == "support_ontology_final_upgrade"
