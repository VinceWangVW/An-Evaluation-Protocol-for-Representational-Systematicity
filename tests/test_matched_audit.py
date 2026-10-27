import copy
import json
from pathlib import Path
import random

import pytest

from experiments.matched_audit import audit_matched_sets,write_audit
from release_analysis import REQUIRED,aggregate_clusters

ROOT=Path(__file__).resolve().parents[1]


class StableParser:
    """An intervention-insensitive model tests real zero-effect bookkeeping."""
    model_name_or_path="test:stable-parser"

    def __init__(self,rows):
        self.targets={r["input_text"]:r["target_form"] for r in rows}
        self.calls=[]

    def trace_text(self,text,probes,layer_indices):
        vectors=[]
        for probe in probes:
            vector=[1.,0.,0.] if probe["label"]=="agent" else [0.,1.,0.] if probe["label"]=="theme" else [0.,0.,1.]
            vectors.append({**probe,"vector":vector,"token_indices":[probe["start_char"]],"layer_index":1})
        return {"span_vectors":vectors,"tokens":list(text)}

    def generate_text(self,text,**kwargs):
        return self.targets[text]

    def generate_with_intervention(self,text,**kwargs):
        self.calls.append({"text":text,**kwargs})
        return {"generated_text":self.targets[text]}


@pytest.fixture
def rows():
    all_rows=[json.loads(line) for line in (ROOT/"data/processed/phase11_systematicity_v2.jsonl").read_text().splitlines()]
    # Second family includes a real passive hard target and active easy source.
    return all_rows[7:14]


def test_complete_audit_records_zero_effects_without_skipping_repair(rows,tmp_path):
    adapter=StableParser(rows)
    payload=audit_matched_sets(rows,adapter,permutations=19)
    counts=payload["summary"]["counts"]
    assert counts=={"items":7,"interchange":2,"necessity":14,"repair":1,"eligible_repair_errors":0}
    assert payload["summary"]["repair_error_conditional_rate"] is None
    signals=payload["per_set"][0]["signals"]
    assert signals["m2_advantage_mean"]==0
    assert signals["m3_selective_impairment_clause_f1"]==0
    assert signals["m4_advantage_mean"]==0
    assert payload["summary"]["baseline_exact_match"]==1
    assert payload["summary"]["behavioral_anchor"]["passed"] is True
    repair=[r for r in payload["operations"] if r["operation"]=="repair"][0]
    assert set(repair["outcomes"])=={"actual","wrong_role","non_role","shuffled","path_only","generic_decoder_gain"}
    assert repair["eligible_error"] is False
    assert repair["advantage"]==repair["gains"]["actual"]-max(v for k,v in repair["gains"].items() if k!="actual")
    assert any(call["control_overrides"]=={"decoder_cross_attention_scale":1.75} for call in adapter.calls)
    files=write_audit(tmp_path,payload)
    assert set(files)=={"L1_per_set.jsonl","L2_decisions.jsonl","raw_operations.jsonl","baseline_predictions.jsonl","span_vectors.jsonl","summary.json"}
    summary=json.loads(Path(files["summary.json"]).read_text())
    assert summary["signals"]==summary["summary"]["signals"]
    assert len(Path(files["raw_operations.jsonl"]).read_text().splitlines())==17


def test_matched_audit_rejects_incomplete_or_nonexact_pairs(rows):
    with pytest.raises(ValueError,match="seven"):
        audit_matched_sets(rows[:-1],StableParser(rows))
    changed=copy.deepcopy(rows)
    next(row for row in changed if row["metadata"]["counterpart_type"]=="agent_swap")["target_form"]="contact(agent=unexpected,theme=unrelated)"
    with pytest.raises(ValueError,match="signature"):
        audit_matched_sets(changed,StableParser(changed))


def test_necessity_records_all_strongest_controls_and_global_diagnostic(rows):
    payload=audit_matched_sets(rows,StableParser(rows),permutations=19)
    necessity=[r for r in payload["operations"] if r["operation"]=="necessity"]
    assert len(necessity)==14
    for row in necessity:
        assert set(row["outcomes"])=={"actual","wrong_role","non_role","global"}
        assert row["strongest_control"]==max(row["drops"]["wrong_role"],row["drops"]["non_role"])
        assert row["advantage"]==row["drops"]["actual"]-row["strongest_control"]
        assert {len(row["ablation_token_indices"][name]) for name in ("actual","wrong_role","non_role")}=={row["matched_token_count"]}


def test_shuffle_excludes_counterfactual_filler_and_failed_anchor_abstains(rows):
    payload=audit_matched_sets(rows,StableParser(rows),permutations=19)
    by_id={row["item_id"]:row for row in rows}
    for record in payload["operations"]:
        if record["operation"]=="interchange":
            donor=by_id[record["donor_item_id"]]
            assert record["shuffled_donor_filler"] != donor["role_signature"][record["role"]]
            assert record["shuffled_donor_item_id"] not in {record["target_item_id"],record["donor_item_id"]}
    adapter=StableParser(rows)
    adapter.targets={text:"" for text in adapter.targets}
    failed=audit_matched_sets(rows,adapter,permutations=19)
    assert failed["summary"]["behavioral_anchor"]["passed"] is False
    assert failed["signals"]["state_anchor_ready"] is False
    assert all(not row["signals"]["state_anchor_ready"] and row["decision"]["selected_state"]=="Undetermined"
               for row in failed["per_set"])


def test_bootstrap_point_uses_same_matched_set_estimator_as_panel():
    all_rows=[json.loads(line) for line in (ROOT/"data/processed/phase11_systematicity_v2.jsonl").read_text().splitlines()][:21]
    # Give the third set the held-out anchor split, while the behavioral prior
    # still estimates the mean of all three declared sets.
    for row in all_rows[14:]:
        row["split"]="gen"

    class HeterogeneousParser(StableParser):
        def __init__(self,records):
            super().__init__(records)
            self.rng=random.Random(30)

        def trace_text(self,text,probes,layer_indices):
            output=super().trace_text(text,probes,layer_indices)
            for span in output["span_vectors"]:
                span["vector"]=[self.rng.uniform(-1,1) for _ in range(5)]
            return output

    adapter=HeterogeneousParser(all_rows)
    # Force nonidentical set-level behavioral scores without changing any gold
    # labels or artificially filling evidence values.
    for row in all_rows[:7]:
        adapter.targets[row["input_text"]]=""
    payload=audit_matched_sets(all_rows,adapter,seed=13,permutations=19)
    point=aggregate_clusters(payload["per_set"],random.Random(13),permutations=19)
    for key in REQUIRED:
        if isinstance(point[key],bool):
            assert payload["signals"][key] is point[key]
        else:
            assert payload["signals"][key]==pytest.approx(point[key]),key
    assert payload["summary"]["behavioral_anchor"]["clause_f1"] != payload["signals"]["b0_gen_clause_f1"]
    # The pooled readout is preserved for inspection, not substituted for the
    # matched-set estimator used by the decision and its bootstrap.
    assert payload["summary"]["readouts"]["role"]["accuracy"] != pytest.approx(payload["signals"]["m1_semantic_role_accuracy"])
