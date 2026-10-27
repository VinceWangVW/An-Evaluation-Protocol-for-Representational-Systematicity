import copy
import json
import math
from pathlib import Path

import pytest

from release_analysis import (
    DEFAULT_PROFILE, adjudicate, cluster_bootstrap, closure_decision, component_rules,
    engineered_truth_controls, make_training_plan, score_feature, sensitivity,
    slog_adapt, transfer_compare, component_failure_panel, make_suite_plan, collect_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def panel():
    return json.loads((ROOT/"tests/fixtures/analysis/panel.json").read_text())


def test_sigmoid_and_strict_required_gates(panel):
    assert score_feature(.5,.5,.1) == .5
    assert score_feature(1000,.5,.01) == 1
    signals = panel["targets"][0]["signals"]
    assert adjudicate(signals)["selected_state"] == "Causal-use"
    for key,value in (("realism_pass_rate",.6),("linkage_pass_rate",.5),("evidence_coverage",.8)):
        result = adjudicate({**signals,key:value})
        assert result["selected_state"] == "Undetermined"
    missing = {**signals,"m2_permutation_p_value":None}
    assert adjudicate(missing)["selected_state"] == "Undetermined"
    assert "m2_permutation_p_value" in adjudicate(missing)["missing_features"]


def test_target_independent_grid_and_disjointness(panel):
    grid = json.loads((ROOT/"configs/release/sensitivity_grid.json").read_text())
    result = sensitivity(panel,grid)
    changed = copy.deepcopy(panel)
    changed["targets"][0]["signals"]["m2_advantage_mean"] = 0
    other = sensitivity(changed,grid)
    assert result["configurations"] == 729
    assert result["compatible_count"] == other["compatible_count"]
    changed["targets"][0]["id"] = changed["calibration"][0]["id"]
    with pytest.raises(ValueError,match="disjoint"):
        sensitivity(changed,grid)


def test_bootstrap_preserves_clusters_recomputes_p_and_is_reproducible(panel):
    signal = panel["targets"][0]["signals"]
    rows = [{"matched_set_id":str(i),"signals":{**signal,"m2_advantage_mean":v,"m2_permutation_p_value":.99}}
            for i,v in enumerate((.2,.3,.4,.5))]
    a = cluster_bootstrap(rows,replicates=12,seed=3,expected_sets=4,permutations=19)
    b = cluster_bootstrap(rows,replicates=12,seed=3,expected_sets=4,permutations=19)
    assert a == b
    assert a["point_estimate"]["m2_permutation_p_value"] != .99
    assert all(len(r["indices"]) == 4 for r in a["samples"])
    with pytest.raises(ValueError,match="unique"):
        cluster_bootstrap(rows+rows[:1],expected_sets=5)


def test_closure_requires_exact_frozen_basis_and_full_causal_chain(panel):
    objects = [{"object_id":str(i),"object_class":"bridge","step":i%2,"template":i%3,
                "discovery_gap":.3,"necessity_gap":.3,"null_effect":0.,"realism_pass_rate":.9}
               for i in range(12)]
    context = dict(sealed=True,complete=True,present_route_complete=True,present_route_found=False)
    assert closure_decision(objects,DEFAULT_PROFILE,**context)["selected_state"] == "Undetermined"
    basis = [{k:o[k] for k in ("object_id","object_class","step","template")} for o in objects]
    assert closure_decision(objects[:-1],DEFAULT_PROFILE,frozen_basis=basis,**context)["selected_state"] == "Undetermined"
    assert closure_decision(objects,DEFAULT_PROFILE,frozen_basis=basis,**context)["selected_state"] == "Undetermined"
    objects[0]["signals"] = panel["targets"][0]["signals"]
    result = closure_decision(objects,DEFAULT_PROFILE,frozen_basis=basis,**context)
    assert result["selected_state"] == "Causal-use"
    objects[0]["null_effect"] = math.nan
    with pytest.raises(ValueError,match="finite"):
        closure_decision(objects,DEFAULT_PROFILE,frozen_basis=basis,**context)


def test_component_rules_fail_on_missing_truth_or_metrics(panel):
    row = {"id":"a","signals":panel["targets"][0]["signals"],"truth":"causal","truth_provenance":"fixture"}
    result = component_rules([row])
    assert result["statistics"]["full"]["causal_retained"] == 1
    with pytest.raises(ValueError,match="truth"):
        component_rules([{**row,"truth_provenance":None}])


def test_truth_fixture_has_disjoint_confirmation_and_explicit_status():
    result = engineered_truth_controls([0],observations=32)
    assert result["experiment_context"] == "engineered_structural_model"
    assert len(result["rows"]) == 6
    assert all(r["independent_confirmation_n"] == 32 for r in result["rows"])
    assert all(r["signals"]["m2_permutation_p_value"] > 0 for r in result["rows"])


def test_seed_plan_executes_training_and_both_family_evidence(tmp_path):
    plan = make_training_plan(ROOT,tmp_path,seeds=[13])
    assert len(plan["commands"]) == 8
    assert "train_target_families.py" in plan["commands"][0][1]
    panel = json.loads((tmp_path/"training_panel.json").read_text())
    assert len(panel["targets"]) == 8
    assert {r["seed"] for r in panel["targets"] if "target" in r["tags"]} == {13}
    assert all(r["seed"] == 1013 for r in panel["targets"] if "calibration" in r["tags"])
    complete_audits = [argv for argv in plan["commands"] if "run_matched_audit.py" in argv[1]]
    assert len(complete_audits) == 4
    assert all(argv[argv.index("--family")+1] == "binding" for argv in complete_audits)
    scaled = next(argv for argv in complete_audits if "--decoder-cross-attention-scale" in argv)
    assert scaled[scaled.index("--checkpoint")+1].endswith("calibration_causal")
    assert scaled[scaled.index("--decoder-cross-attention-scale")+1] == "0.25"
    with pytest.raises(ValueError,match="disjoint"):
        make_training_plan(ROOT,tmp_path,seeds=[0,1000])


def test_component_panel_measures_declared_41_cases_from_raw_predictions():
    result = component_failure_panel(seed=31,observations=24)
    assert result["experiment_context"] == "structural_causal_model"
    assert len(result["rows"]) == 41
    assert sum(row["truth"] == "causal" for row in result["rows"]) == 7
    assert sum(row["truth"] == "noncausal" for row in result["rows"]) == 34
    assert result["comparison"]["units"] == 41
    row = result["rows"][0]
    observations = row["raw_observations"]
    measured = sum((r["interchange_prediction"] == r["counterfactual_label"])
                   -(r["interchange_control_prediction"] == r["counterfactual_label"]) for r in observations)/len(observations)
    assert row["signals"]["m2_advantage_mean"] == measured
    off = next(row for row in result["rows"] if row["mechanism"] == "off_manifold")
    assert off["signals"]["realism_pass_rate"] == 0
    assert all(r["intervention_state_norm"] > r["natural_state_norm"] for r in off["raw_observations"])


def test_suite_connects_measured_sets_to_statistical_analyses(tmp_path):
    config = {"seeds":[0],"transfer_seeds":[1],"benchmark":"phase11_systematicity_v2",
              "bootstrap_replicates":1000,"bootstrap_seed":13,"grid":"configs/release/sensitivity_grid.json"}
    result = make_suite_plan(ROOT,tmp_path,config)
    bootstraps = [argv for argv in result["commands"] if "bootstrap" in argv]
    assert len(bootstraps) == 1
    assert bootstraps[0][bootstraps[0].index("--rows")+1].endswith("L1_per_set.jsonl")
    assert any("component-panel" in argv for argv in result["commands"])
    assert any("sensitivity" in argv for argv in result["commands"])
    assert any("transfer-compare" in argv for argv in result["commands"])


def test_collector_reads_matched_audit_schema_and_common_scorer(tmp_path,panel):
    audit_dir = tmp_path/"audit"
    audit_dir.mkdir()
    signals = panel["targets"][0]["signals"]
    (audit_dir/"summary.json").write_text(json.dumps({"signals":signals}))
    control = {"target_id":"control","run_name":"control","seed":1,"tags":["binding","calibration"],"panel_regime":"causal"}
    payload = {"targets":[control],"metadata":{"audit_outputs":{"control":str(audit_dir)}}}
    path = tmp_path/"panel.json"
    path.write_text(json.dumps(payload))
    result = collect_evidence(tmp_path,path)
    assert result["calibration"][0]["decision"] == adjudicate(signals)
    assert result["calibration"][0]["signals"] == signals


def test_transfer_recalibration_does_not_look_at_target(panel):
    a = transfer_compare(panel["calibration"],panel["targets"])
    targets = copy.deepcopy(panel["targets"])
    targets[0]["signals"]["m2_advantage_mean"] = -.5
    b = transfer_compare(panel["calibration"],targets)
    assert a["recalibrated_profile"] == b["recalibrated_profile"]


def test_slog_adapter_checks_full_lf_and_exact_single_role():
    left = {"item_id":"a","input_text":"The chef praised the poet.","target_form":"praise(agent=chef,theme=poet)",
            "split":"gen","role_signature":{"predicate":"praise","agent":"chef","theme":"poet"}}
    right = {"item_id":"b","input_text":"The artist praised the poet.","target_form":"praise(agent=artist,theme=poet)",
             "split":"gen","role_signature":{"predicate":"praise","agent":"artist","theme":"poet"}}
    selection = {"source_revision":"test-fixture","source_sha256":"abc","selection_rationale":"test fixture",
                 "pairs":[{"pair_id":"p","source_id":"a","target_id":"b","changed_roles":["agent"]}]}
    assert len(slog_adapt([left,right],selection,expected_pairs=1)["pairs"]) == 1
    bad = {**right,"target_form":"praise(agent=artist,theme=poet,agent_mod=brisk)"}
    with pytest.raises(ValueError,match="additional"):
        slog_adapt([left,bad],selection,expected_pairs=1)
    duplicate = {**selection,"pairs":selection["pairs"]+[{**selection["pairs"][0],"pair_id":"second-id"}]}
    with pytest.raises(ValueError,match="unique endpoints"):
        slog_adapt([left,right],duplicate,expected_pairs=2)
