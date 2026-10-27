import copy
import json
import math
from pathlib import Path

import pytest

from release_analysis import (
    DEFAULT_PROFILE, adjudicate, cluster_bootstrap, closure_decision, component_rules,
    engineered_truth_controls, make_training_plan, score_feature, sensitivity,
    slog_adapt, transfer_compare,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def panel():
    return json.loads((ROOT/"configs/release/implementation_fixture_panel.json").read_text())


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
    assert result["historical_result"] is False
    assert len(result["rows"]) == 6
    assert all(r["independent_confirmation_n"] == 32 for r in result["rows"])
    assert all(r["signals"]["m2_permutation_p_value"] > 0 for r in result["rows"])


def test_seed_plan_executes_training_and_both_family_evidence(tmp_path):
    plan = make_training_plan(ROOT,tmp_path,seeds=[13])
    assert len(plan["commands"]) == 5
    assert "train_target_families.py" in plan["commands"][0][1]
    panel = json.loads((tmp_path/"training_panel.json").read_text())
    assert len(panel["targets"]) == 8
    assert {r["seed"] for r in panel["targets"] if "target" in r["tags"]} == {13}
    assert all(r["seed"] == 1013 for r in panel["targets"] if "calibration" in r["tags"])
    with pytest.raises(ValueError,match="disjoint"):
        make_training_plan(ROOT,tmp_path,seeds=[0,1000])


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
