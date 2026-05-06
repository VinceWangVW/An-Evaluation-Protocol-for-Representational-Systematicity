from __future__ import annotations

from analysis.interaction_objects import align_decoder_steps
from analysis.family_closure import (
    ALIGNER_BASELINE,
    ALIGNER_SLOT,
    BRIDGE_OBJECT_CLASS,
    QUERY_SOURCE_OBJECT_CLASS,
    align_decoder_steps_slot_aware,
    build_cross_step_bridge_catalog,
    build_query_source_relation_catalog,
    gate_h2_row,
    summarize_family_closure,
)
from interventions.generation import (
    GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS,
    GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS,
    normalize_geometry_object_spec,
)


class _SlotTokenizer:
    eos_token_id = 1

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def __call__(self, text: str, add_special_tokens: bool = False):
        pieces = (
            text.replace("(", " ( ")
            .replace(")", " ) ")
            .replace(",", " , ")
            .split()
        )
        ids = []
        for piece in pieces:
            if piece not in self._vocab:
                self._vocab[piece] = len(self._vocab) + 10
            ids.append(self._vocab[piece])
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids):
        return [str(value) for value in ids]


def _make_trace_row(control_id: str, step_key: str, head_index: int, score: float, overlap: float = 0.05):
    return {
        "item_id": f"{control_id}:{step_key}:{head_index}",
        "control_id": control_id,
        "generalization_case": "active_depth1",
        "stratum": "pass" if "h2" in control_id else "control",
        "split": "gen",
        "decoder_step_key": step_key,
        "decoder_step_index": 0,
        "generated_token_text": "x",
        "layer_index": 6,
        "head_index": head_index,
        "source_span_keys": [step_key],
        "source_token_indices": [1, 2],
        "cross_attn_mass": score,
        "counterpart_margin_delta": score,
        "step_exact_match": True,
        "step_exact_match_rate": 1.0,
        "prelogit_basis_scores": {
            "prelogit_basis_0": 0.8 if head_index == 0 else 0.2,
            "prelogit_basis_1": 0.2 if head_index == 0 else 0.8,
            "prelogit_basis_2": 0.1,
        },
        "generic_decoder_gain_overlap": overlap,
        "generic_basis_ids": ["prelogit_basis_15", "prelogit_basis_16"],
    }


def test_slot_aware_alignment_recovers_agent_theme_when_exact_alignment_fails() -> None:
    tokenizer = _SlotTokenizer()
    record = {
        "target_form": "admire(agent=jonah,theme=gavin)",
        "role_signature": {"predicate": "admire", "agent": "jonah", "theme": "gavin"},
    }
    baseline = align_decoder_steps(tokenizer, record["target_form"], record)
    slot_aware = align_decoder_steps_slot_aware(tokenizer, record["target_form"], record)
    assert "predicate_value" in baseline["alignments"]
    assert "agent_value" not in baseline["alignments"]
    assert "theme_value" not in baseline["alignments"]
    assert "agent_value" in slot_aware["alignments"]
    assert "theme_value" in slot_aware["alignments"]


def test_bridge_and_query_specs_roundtrip() -> None:
    bridge = normalize_geometry_object_spec(
        "cross_step_bridge",
        """{"object_id":"cross_step_bridge_predicate_to_agent_top1","object_class":"cross_step_bridge","layer_index":6,"source_run":"phase12_geometry_h2_full_seed11","source_step_key":"predicate_value","target_step_key":"agent_value","source_span_keys":["predicate"],"head_selector":{"mode":"explicit","indices":[1]},"bridge_basis":{"mode":"explicit","basis_ids":["prelogit_basis_0"]},"store_condition":{"mode":"none"},"matched_control_type":"wrong_event_same_cardinality"}""",
    )
    query = normalize_geometry_object_spec(
        "query_source_relation",
        """{"object_id":"query_source_relation_agent_top1","object_class":"query_source_relation","layer_index":6,"source_run":"phase12_geometry_h2_full_seed11","query_step_key":"agent_value","source_span_keys":["agent","filler::agent"],"head_selector":{"mode":"explicit","indices":[2]},"query_basis":{"mode":"explicit","basis_ids":["prelogit_basis_0"]},"read_basis":{"mode":"explicit","basis_ids":["prelogit_basis_1"]},"store_condition":{"mode":"none"},"matched_control_type":"wrong_event_same_cardinality"}""",
    )
    assert bridge is not None and bridge["object_class"] == GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS
    assert query is not None and query["object_class"] == GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS


def test_c2_and_c3_catalogs_emit_twelve_rows() -> None:
    dev_h2 = []
    dev_h0 = []
    held_h2 = []
    held_h0 = []
    for step_key in ("predicate_value", "agent_value", "theme_value"):
        for head_index in range(4):
            dev_h2.append(_make_trace_row("dev_h2", step_key, head_index, 0.9 - 0.1 * head_index))
            dev_h0.append(_make_trace_row("dev_h0", step_key, head_index, 0.1 + 0.01 * head_index, overlap=0.01))
            held_h2.append(_make_trace_row("held_h2", step_key, head_index, 0.85 - 0.1 * head_index))
            held_h0.append(_make_trace_row("held_h0", step_key, head_index, 0.1 + 0.01 * head_index, overlap=0.01))
    c2 = build_cross_step_bridge_catalog(
        dev_h2_rows=dev_h2,
        dev_h0_rows=dev_h0,
        heldout_h2_rows=held_h2,
        heldout_h0_rows=held_h0,
        layer_index=6,
        source_run="phase12_geometry_h2_full_seed11",
    )
    c3 = build_query_source_relation_catalog(
        dev_h2_rows=dev_h2,
        dev_h0_rows=dev_h0,
        heldout_h2_rows=held_h2,
        heldout_h0_rows=held_h0,
        layer_index=6,
        source_run="phase12_geometry_h2_full_seed11",
    )
    assert len(c2) == 12
    assert len(c3) == 12
    assert all(row["object_class"] == BRIDGE_OBJECT_CLASS for row in c2)
    assert all(row["object_class"] == QUERY_SOURCE_OBJECT_CLASS for row in c3)


def test_gate_and_branch_logic_require_full_family_closure() -> None:
    row = gate_h2_row(
        object_spec={"object_id": "bridge_1", "object_class": BRIDGE_OBJECT_CLASS, "generic_gain_overlap": 0.1},
        dev_summary={"h2_gap": 0.0, "h0_gap": 0.0, "realism_pass_rate": 1.0},
        heldout_summary={"h2_gap": 0.0, "h0_gap": 0.0, "realism_pass_rate": 1.0},
    )
    assert row["passed"] is False
    closure = summarize_family_closure(
        observability_summary={"sealed": True, "decision": "predicate_dominant", "selected_aligner": ALIGNER_BASELINE},
        class_summaries=[
            {"object_class": "interaction", "accepted": False},
            {"object_class": "cross_step_bridge", "accepted": False},
            {"object_class": "query_source_relation", "accepted": False},
        ],
        h2_summaries=[
            {"object_class": "interaction", "passed_count": 0},
            {"object_class": "cross_step_bridge", "passed_count": 0},
            {"object_class": "query_source_relation", "passed_count": 0},
        ],
        direct_h1_summary={"passed_count": 0},
        case_summary={"accepted_slice_count": 0},
    )
    assert closure["decision"] == "T-supported"
    assert closure["decision_status"] == "family-final"
