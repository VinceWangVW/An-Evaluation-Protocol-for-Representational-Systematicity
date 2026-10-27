import importlib.util
import json
from pathlib import Path


def test_walkthrough_keeps_unknown_historical_inputs_and_abstains(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_walkthrough.py"
    spec = importlib.util.spec_from_file_location("walkthrough", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.build_walkthrough(tmp_path)
    assert result["released_input_count"] == 7
    assert result["recomputed_state"] == "Undetermined"
    historical = json.loads((tmp_path / "paper_reported_L1.json").read_text())
    assert historical["historical_inputs"] is None
    assert historical["metrics"]["filler_control"] is None
    assert historical["reported_state"] == "Causal-use"
    l2 = json.loads((tmp_path / "recomputed_L2.json").read_text())
    behavior_score = l2["known_feature_transforms"]["b0_gen_clause_f1"]["score"]
    assert abs(behavior_score - 0.977) < 0.001
    released = json.loads((tmp_path / "released_index047_inputs.json").read_text())
    assert "family_00047" in released["counterpart_family_id"]
    assert len({item["counterpart_type"] for item in released["items"]}) == 7
