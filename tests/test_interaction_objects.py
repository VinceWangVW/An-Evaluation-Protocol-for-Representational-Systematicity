from __future__ import annotations

from analysis.interaction_objects import (
    GEOMETRY_INTERACTION_OBJECT_CLASS,
    STEP_TEMPLATE_FAMILIES,
    STEP_SOURCE_KEYS,
    align_decoder_steps,
    build_interaction_catalog,
    case_condition_interaction_objects,
    summarize_interaction_catalog,
)
from interventions.generation import normalize_geometry_object_spec


class _ToyTokenizer:
    eos_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = False):
        vocab = {
            "admire": 10,
            "jonah": 11,
            "gavin": 12,
            "agent": 13,
            "=": 14,
            "theme": 15,
            "(": 16,
            ")": 17,
            ",": 18,
        }
        pieces = []
        for token in (
            text.replace("(", " ( ")
            .replace(")", " ) ")
            .replace(",", " , ")
            .replace("=", " = ")
            .split()
        ):
            pieces.append(vocab.setdefault(token, len(vocab) + 20))
        return {"input_ids": pieces}

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
        "source_span_keys": list(STEP_SOURCE_KEYS[step_key]),
        "source_token_indices": [1, 2],
        "cross_attn_mass": score,
        "counterpart_margin_delta": score,
        "step_exact_match": True,
        "step_exact_match_rate": 1.0,
        "prelogit_basis_scores": {
            "prelogit_basis_0": 0.7 if head_index == 0 else 0.2,
            "prelogit_basis_1": 0.2 if head_index == 0 else 0.7,
            "prelogit_basis_2": 0.1,
        },
        "generic_decoder_gain_overlap": overlap,
    }


def test_interaction_spec_roundtrip() -> None:
    raw = """{"object_id":"interaction_agent_step_headtop1_basis1","object_class":"interaction","layer_index":6,"source_run":"phase12_geometry_h2_full_seed11","decoder_step_key":"agent_value","source_span_keys":["agent","filler::agent"],"head_selector":{"mode":"explicit","indices":[3]},"read_basis":{"mode":"explicit","basis_ids":["prelogit_basis_0"]},"store_condition":{"mode":"none"},"matched_control_type":"wrong_event_same_step"}"""
    spec = normalize_geometry_object_spec("interaction", raw)
    assert spec is not None
    assert spec["object_class"] == GEOMETRY_INTERACTION_OBJECT_CLASS
    assert spec["decoder_step_key"] == "agent_value"
    assert spec["head_selector"]["indices"] == [3]


def test_align_decoder_steps_finds_predicate_agent_theme() -> None:
    tokenizer = _ToyTokenizer()
    record = {
        "target_form": "admire(agent=jonah,theme=gavin)",
        "role_signature": {"predicate": "admire", "agent": "jonah", "theme": "gavin"},
    }
    aligned = align_decoder_steps(tokenizer, record["target_form"], record)
    assert "predicate_value" in aligned["alignments"]
    assert "agent_value" in aligned["alignments"]
    assert "theme_value" in aligned["alignments"]


def test_build_interaction_catalog_outputs_twelve_rows() -> None:
    dev_h2 = []
    dev_h0 = []
    held_h2 = []
    held_h0 = []
    for step_key in STEP_SOURCE_KEYS:
        for head_index in range(4):
            dev_h2.append(_make_trace_row("dev_h2", step_key, head_index, 0.8 - (0.1 * head_index)))
            dev_h0.append(_make_trace_row("dev_h0", step_key, head_index, 0.1 + (0.01 * head_index), overlap=0.01))
            held_h2.append(_make_trace_row("held_h2", step_key, head_index, 0.7 - (0.1 * head_index)))
            held_h0.append(_make_trace_row("held_h0", step_key, head_index, 0.1 + (0.01 * head_index), overlap=0.01))
    catalog = build_interaction_catalog(
        dev_h2_rows=dev_h2,
        dev_h0_rows=dev_h0,
        heldout_h2_rows=held_h2,
        heldout_h0_rows=held_h0,
        layer_index=6,
        source_run="phase12_geometry_h2_full_seed11",
    )
    assert len(catalog) == 12
    assert all(row["object_class"] == GEOMETRY_INTERACTION_OBJECT_CLASS for row in catalog)
    assert {row["template_family"] for row in catalog} == {name for name, _ in STEP_TEMPLATE_FAMILIES}
    summary = summarize_interaction_catalog(catalog)
    assert summary["object_count"] == 12


def test_case_condition_returns_t_supported_when_h2_blocked() -> None:
    case_summary = case_condition_interaction_objects(
        h2_summary={"passed_count": 0},
        rescue_rows=[],
    )
    assert case_summary["branch_decision"]["decision"] == "T-supported"
