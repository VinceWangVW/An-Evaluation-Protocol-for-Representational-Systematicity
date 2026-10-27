import importlib.util
import json
from pathlib import Path

import pytest

from data.systematicity_schema import REQUIRED_COUNTERPART_TYPES
from release_analysis import adjudicate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def walkthrough():
    path = ROOT / "scripts/run_walkthrough.py"
    spec = importlib.util.spec_from_file_location("walkthrough", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_panel_has_exact_seven_semantic_contrasts_and_repair_pair(walkthrough):
    panel = walkthrough.build_panel()
    rows = {row["metadata"]["counterpart_type"]: row for row in panel["rows"]}
    designs = {row["counterpart_type"]: row for row in panel["schema"]}
    assert set(rows) == set(REQUIRED_COUNTERPART_TYPES)
    assert len({row["input_text"] for row in rows.values()}) == 7
    assert rows["rescue_donor"]["target_form"] == rows["rescue_target"]["target_form"]
    assert rows["rescue_target"]["role_signature"]["predicate"] == "praise"
    assert rows["rescue_target"]["role_signature"]["agent"] == "chef"
    assert rows["rescue_target"]["role_signature"]["theme"] == "poet"
    assert rows["rescue_donor"]["metadata"]["voice"] == "active"
    assert rows["rescue_target"]["metadata"]["voice"] == "passive"
    assert rows["rescue_target"]["metadata"]["embedding_depth"] == 1
    for kind, changed in (("agent_swap", "agent"), ("theme_swap", "theme")):
        before, after = rows["rescue_target"]["role_signature"], rows[kind]["role_signature"]
        assert {key for key in ("predicate", "agent", "theme") if before[key] != after[key]} == {changed}
    assert designs["modifier_swap"]["agent_modifier"] == designs["rescue_target"]["theme_modifier"]
    assert designs["modifier_swap"]["theme_modifier"] == designs["rescue_target"]["agent_modifier"]
    assert len(panel["pairs"]["m2_pairs"]) == 2
    assert panel["pairs"]["m4_pair"] == {
        "target_item_id": "MS-047:rescue_target", "donor_item_id": "MS-047:rescue_donor"}
    assert panel["validation"]["valid_candidate_count"] == 21


def test_preparation_is_deterministic_and_has_no_invented_measurements(walkthrough, tmp_path):
    result = walkthrough.build_walkthrough(tmp_path)
    assert result["status"] == "prepared"
    assert result["item_count"] == 7
    assert "selected_state" not in result
    assert not list(tmp_path.glob("L1*"))
    assert not list(tmp_path.glob("L2*"))
    for name in result["files"]:
        assert (tmp_path / name).read_bytes() == (ROOT / "data/walkthrough" / name).read_bytes()
    manifest = json.loads((tmp_path / "MS-047_manifest.json").read_text())
    assert manifest["provenance"] == "programmatic semantic design"
    assert "state_scores" not in manifest
    assert "metrics" not in manifest


def test_measured_evidence_uses_shared_adjudicator(walkthrough, tmp_path):
    signals = {
        "b0_gen_clause_f1": 0.9, "m1_semantic_role_accuracy": 0.99,
        "m1_surface_control_gap": 0.2, "m2_advantage_mean": 0.5,
        "m2_permutation_p_value": 0.001, "m3_selective_impairment_clause_f1": 0.3,
        "m4_advantage_mean": 0.5, "realism_pass_rate": 0.9,
        "linkage_pass_rate": 0.9, "evidence_coverage": 1.0,
        "audit_ready": True, "state_anchor_ready": True,
    }
    evidence_path = tmp_path / "measured.json"
    evidence_path.write_text(json.dumps({"signals": signals}))
    output = tmp_path / "output"
    result = walkthrough.build_walkthrough(output, evidence_path)
    assert result["status"] == "scored"
    assert json.loads((output / "L2_decision.json").read_text()) == adjudicate(signals)
    assert json.loads((output / "L1_signals.json").read_text())["signals"] == signals
    with pytest.raises(ValueError, match="not both"):
        walkthrough.build_walkthrough(output, evidence_path, checkpoint="local-model")


def test_design_rejects_degenerate_role_or_modifier_contrasts(walkthrough, tmp_path):
    config = json.loads(walkthrough.DEFAULT_CONFIG.read_text())
    config["semantic_design"]["alternate_agent"] = "chef"
    path = tmp_path / "design.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="distinct"):
        walkthrough.build_panel(path)


def test_checkpoint_walkthrough_executes_real_tiny_t5_offline(walkthrough, tmp_path, monkeypatch):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    rows = walkthrough.build_panel()["rows"]
    words = sorted({word for row in rows for field in ("input_text", "target_form") for word in row[field].split()})
    vocabulary = {word: i for i, word in enumerate(["<pad>", "</s>", "<unk>"] + words)}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>",
        model_input_names=["input_ids", "attention_mask"], model_max_length=96,
    )
    checkpoint = tmp_path / "checkpoint"
    tokenizer.save_pretrained(checkpoint)
    torch.manual_seed(7)
    model = T5ForConditionalGeneration(T5Config(
        vocab_size=len(vocabulary), d_model=16, d_kv=8, d_ff=32, num_layers=2,
        num_decoder_layers=2, num_heads=2, dropout_rate=0.0,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    ))
    model.save_pretrained(checkpoint)
    output = tmp_path / "audit"
    result = walkthrough.build_walkthrough(output, checkpoint=str(checkpoint), device="cpu",
                                            layer_index=1, max_new_tokens=2, permutations=19)
    assert result["status"] == "executed"
    assert result["summary"]["counts"]["repair"] == 1
    assert result["summary"]["counts"]["interchange"] == 2
    assert result["summary"]["counts"]["necessity"] == 14
    operations = [json.loads(line) for line in (output / "raw_operations.jsonl").read_text().splitlines()]
    assert {row["operation"] for row in operations} == {"interchange", "necessity", "repair"}
    assert (output / "L1_per_set.jsonl").exists() and (output / "L2_decisions.jsonl").exists()
