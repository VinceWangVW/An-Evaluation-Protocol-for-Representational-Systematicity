from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from interventions.generation import (
    GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS,
    GEOMETRY_INTERACTION_OBJECT_CLASS,
    GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS,
    aggregate_control_value,
    stable_seed,
)
from utils.io import read_jsonl

from . import interaction_objects as p17


ALIGNER_BASELINE = "A1"
ALIGNER_SLOT = "A2"
ALIGNER_BRIDGE = "A3"
ALIGNER_NAMES: tuple[str, ...] = (ALIGNER_BASELINE, ALIGNER_SLOT, ALIGNER_BRIDGE)

H2_NECESSITY_MIN_GAP = p17.H2_NECESSITY_MIN_GAP
H0_NECESSITY_MAX_ABS_GAP = p17.H0_NECESSITY_MAX_ABS_GAP
REALISM_MIN_PASS_RATE = p17.REALISM_MIN_PASS_RATE
GENERIC_GAIN_MAX_OVERLAP = p17.GENERIC_GAIN_MAX_OVERLAP
H1_RESCUE_MIN_GAP = p17.H1_RESCUE_MIN_GAP
H1_GENERIC_ADVANTAGE_MIN = p17.H1_GENERIC_ADVANTAGE_MIN
H1_RESCUE_MIN_BINARY_RATE = p17.H1_RESCUE_MIN_BINARY_RATE
CASE_MIN_SLICE_RESCUE_GAP = p17.CASE_MIN_SLICE_RESCUE_GAP
CASE_MIN_SLICE_BINARY_RATE = p17.CASE_MIN_SLICE_BINARY_RATE
HEAD_BOOST_SCALE = p17.HEAD_BOOST_SCALE
READ_BOOST_SCALE = p17.READ_BOOST_SCALE
DISCOVERY_MIN_CONTRAST_GAP = p17.DISCOVERY_MIN_CONTRAST_GAP

STEP_SOURCE_KEYS = dict(p17.STEP_SOURCE_KEYS)
STEP_STORE_SELECTOR = dict(p17.STEP_STORE_SELECTOR)
STEP_TEMPLATE_FAMILIES = tuple(p17.STEP_TEMPLATE_FAMILIES)

BRIDGE_OBJECT_CLASS = GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS
QUERY_SOURCE_OBJECT_CLASS = GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS

BRIDGE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("predicate_value", "agent_value", "predicate_to_agent"),
    ("predicate_value", "theme_value", "predicate_to_theme"),
    ("agent_value", "theme_value", "shared_argument"),
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _field_value(step_key: str, target_text: str, record: dict[str, Any]) -> str:
    fields = p17._parse_target_fields(record)
    return str(fields.get(step_key) or "").strip()


def _slot_texts(step_key: str, target_text: str, record: dict[str, Any]) -> list[str]:
    value = _field_value(step_key, target_text, record)
    if not value:
        return []
    if step_key == "predicate_value":
        predicate_match = re.match(r"^\s*([a-z_]+)\(", str(target_text))
        predicate_text = predicate_match.group(1) if predicate_match else value
        return [f"predicate={predicate_text}", predicate_text, value]
    slot_name = "agent" if step_key == "agent_value" else "theme"
    return [f"{slot_name}={value}", value]


def _all_subsequence_windows(sequence: list[int], subsequence: list[int]) -> list[list[int]]:
    if not sequence or not subsequence or len(subsequence) > len(sequence):
        return []
    windows: list[list[int]] = []
    for start in range(0, len(sequence) - len(subsequence) + 1):
        if sequence[start : start + len(subsequence)] == subsequence:
            windows.append(list(range(start, start + len(subsequence))))
    return windows


def align_decoder_steps_slot_aware(tokenizer: Any, target_text: str, record: dict[str, Any]) -> dict[str, Any]:
    target_ids = p17._non_special_target_ids(tokenizer, target_text)
    target_tokens = p17._non_special_tokens(tokenizer, target_ids)
    alignments: dict[str, list[int]] = {}
    unaligned: list[str] = []
    for step_key in ("predicate_value", "agent_value", "theme_value"):
        windows: list[list[int]] = []
        for text in _slot_texts(step_key, target_text, record):
            field_ids = p17._non_special_target_ids(tokenizer, text)
            windows.extend(_all_subsequence_windows(target_ids, field_ids))
        if windows:
            alignments[step_key] = list(windows[0])
        else:
            unaligned.append(step_key)
    return {
        "target_ids": target_ids,
        "target_tokens": target_tokens,
        "alignments": alignments,
        "unaligned_steps": unaligned,
    }


def _window_mass(
    *,
    cross_attentions: list[Any],
    layer_index: int,
    step_positions: list[int],
    source_token_indices: list[int],
) -> float:
    if not cross_attentions or not step_positions or not source_token_indices:
        return 0.0
    cross_layer_index = p17._layer_module_index(layer_index, len(cross_attentions))
    cross_attn = cross_attentions[cross_layer_index]
    num_heads = int(cross_attn.shape[0])
    scores: list[float] = []
    for head_index in range(num_heads):
        for position in step_positions:
            scores.append(float(cross_attn[head_index, position, source_token_indices].sum().item()))
    return float(mean(scores)) if scores else 0.0


def align_decoder_steps_bridge(
    *,
    tokenizer: Any,
    target_text: str,
    record: dict[str, Any],
    forward: dict[str, Any],
    item_span_map: dict[str, Any],
    layer_index: int,
) -> dict[str, Any]:
    slot_alignments = align_decoder_steps_slot_aware(tokenizer, target_text, record)
    exact_alignments = p17.align_decoder_steps(tokenizer, target_text, record)
    target_ids = list(slot_alignments["target_ids"])
    target_tokens = list(slot_alignments["target_tokens"])
    alignments: dict[str, list[int]] = {}
    unaligned: list[str] = []
    for step_key in ("predicate_value", "agent_value", "theme_value"):
        candidate_windows: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for source in (slot_alignments, exact_alignments):
            positions = list(source["alignments"].get(step_key) or [])
            if positions:
                key = tuple(positions)
                if key not in seen:
                    candidate_windows.append(positions)
                    seen.add(key)
        for text in _slot_texts(step_key, target_text, record):
            token_ids = p17._non_special_target_ids(tokenizer, text)
            for positions in _all_subsequence_windows(target_ids, token_ids):
                key = tuple(positions)
                if key not in seen:
                    candidate_windows.append(positions)
                    seen.add(key)
        if not candidate_windows:
            unaligned.append(step_key)
            continue
        source_token_indices = p17._source_token_indices(item_span_map, STEP_SOURCE_KEYS[step_key])
        ranked = sorted(
            (
                (
                    _window_mass(
                        cross_attentions=list(forward["cross_attentions"]),
                        layer_index=layer_index,
                        step_positions=window,
                        source_token_indices=source_token_indices,
                    ),
                    window,
                )
                for window in candidate_windows
            ),
            reverse=True,
        )
        alignments[step_key] = list(ranked[0][1])
    return {
        "target_ids": target_ids,
        "target_tokens": target_tokens,
        "alignments": alignments,
        "unaligned_steps": unaligned,
    }


def resolve_decoder_steps(
    *,
    aligner_name: str,
    tokenizer: Any,
    target_text: str,
    record: dict[str, Any],
    forward: dict[str, Any] | None = None,
    item_span_map: dict[str, Any] | None = None,
    layer_index: int | None = None,
) -> dict[str, Any]:
    if aligner_name == ALIGNER_BASELINE:
        return p17.align_decoder_steps(tokenizer, target_text, record)
    if aligner_name == ALIGNER_SLOT:
        return align_decoder_steps_slot_aware(tokenizer, target_text, record)
    if aligner_name == ALIGNER_BRIDGE:
        if forward is None or item_span_map is None or layer_index is None:
            raise ValueError("A3 bridge aligner requires forward outputs, item spans, and layer index")
        return align_decoder_steps_bridge(
            tokenizer=tokenizer,
            target_text=target_text,
            record=record,
            forward=forward,
            item_span_map=item_span_map,
            layer_index=layer_index,
        )
    raise ValueError(f"Unknown aligner: {aligner_name}")


def trace_control_bundle_with_aligner(
    *,
    bundle: dict[str, Any],
    aligner_name: str,
    max_length: int = 256,
) -> dict[str, Any]:
    adapter, layer_index = p17.build_adapter_from_bundle(bundle)
    span_index = p17._index_span_records(bundle["span_records"], layer_index=layer_index)
    live_span_cache: dict[str, dict[str, Any]] = {}
    trace_rows: list[dict[str, Any]] = []
    unaligned_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    for item in bundle["candidate_records"]:
        item_span_map = p17._ensure_item_span_map(
            adapter=adapter,
            item=item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        forward = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
        )
        if not forward["cross_attentions"]:
            continue
        alignments = resolve_decoder_steps(
            aligner_name=aligner_name,
            tokenizer=adapter.tokenizer,
            target_text=str(item["target_form"]),
            record=item,
            forward=forward,
            item_span_map=item_span_map,
            layer_index=layer_index,
        )
        for step_key in ("predicate_value", "agent_value", "theme_value"):
            positions = list(alignments["alignments"].get(step_key) or [])
            alignment_rows.append(
                {
                    "item_id": str(item["item_id"]),
                    "control_id": str(bundle["summary"]["behavior_run_name"]),
                    "aligner": aligner_name,
                    "decoder_step_key": step_key,
                    "positions": positions,
                    "aligned": bool(positions),
                }
            )
        if alignments["unaligned_steps"]:
            for step_key in list(alignments["unaligned_steps"]):
                unaligned_rows.append(
                    {
                        "item_id": str(item["item_id"]),
                        "control_id": str(bundle["summary"]["behavior_run_name"]),
                        "aligner": aligner_name,
                        "decoder_step_key": step_key,
                        "status": "unaligned_step",
                    }
                )
        source_indices_by_step = {
            step_key: p17._source_token_indices(item_span_map, STEP_SOURCE_KEYS[step_key])
            for step_key in ("predicate_value", "agent_value", "theme_value")
        }
        cross_layer_index = p17._layer_module_index(layer_index, len(forward["cross_attentions"]))
        cross_attn = forward["cross_attentions"][cross_layer_index]
        num_heads = int(cross_attn.shape[0])
        for step_key, step_positions in dict(alignments["alignments"]).items():
            if not step_positions:
                continue
            source_indices = list(source_indices_by_step.get(step_key) or [])
            step_summary = p17._token_margin_summary(
                logits=forward["logits"],
                target_ids=forward["target_ids"],
                step_positions=step_positions,
                compare_ids=[],
            )
            basis_scores = p17._basis_scores_for_steps(forward["decoder_hidden"], step_positions)
            for head_index in range(num_heads):
                head_mass = []
                for position in step_positions:
                    if not source_indices:
                        head_mass.append(0.0)
                    else:
                        head_mass.append(float(cross_attn[head_index, position, source_indices].sum().item()))
                trace_rows.append(
                    {
                        "item_id": str(item["item_id"]),
                        "control_id": str(bundle["summary"]["behavior_run_name"]),
                        "aligner": aligner_name,
                        "generalization_case": str(item["generalization_case"]),
                        "stratum": str(item["stratum"]),
                        "split": str(item["split"]),
                        "decoder_step_key": step_key,
                        "decoder_step_index": int(step_positions[0]),
                        "generated_token_text": str(forward["target_tokens"][step_positions[0]]),
                        "layer_index": int(layer_index),
                        "head_index": int(head_index),
                        "source_span_keys": list(STEP_SOURCE_KEYS[step_key]),
                        "source_token_indices": list(source_indices),
                        "cross_attn_mass": round(mean(head_mass), 6) if head_mass else 0.0,
                        "counterpart_margin_delta": step_summary["margin_mean"],
                        "step_exact_match": bool(step_summary["exact_match"]),
                        "step_exact_match_rate": step_summary["exact_match_rate"],
                        "prelogit_basis_scores": basis_scores,
                    }
                )
    annotated_rows = p17._trace_rows_with_generic_overlap(trace_rows)
    aligned_counts = Counter(
        str(row["decoder_step_key"]) for row in alignment_rows if bool(row["aligned"])
    )
    total_counts = Counter(str(row["decoder_step_key"]) for row in alignment_rows)
    return {
        "rows": annotated_rows,
        "unaligned_rows": unaligned_rows,
        "alignment_rows": alignment_rows,
        "summary": {
            "control_id": str(bundle["summary"]["behavior_run_name"]),
            "aligner": aligner_name,
            "row_count": len(annotated_rows),
            "unaligned_count": len(unaligned_rows),
            "step_counts": dict(Counter(str(row["decoder_step_key"]) for row in annotated_rows)),
            "aligned_step_rates": {
                step_key: round(float(aligned_counts.get(step_key, 0)) / max(int(total_counts.get(step_key, 0)), 1), 6)
                for step_key in ("predicate_value", "agent_value", "theme_value")
            },
        },
    }


def _disagreement_matrix(alignment_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[tuple[str, str], tuple[int, ...]]] = defaultdict(dict)
    for row in alignment_rows:
        grouped[str(row["aligner"])][(str(row["control_id"]), f"{row['item_id']}::{row['decoder_step_key']}")] = tuple(int(v) for v in list(row["positions"]))
    matrix: dict[str, dict[str, float]] = {}
    for lhs in ALIGNER_NAMES:
        matrix[lhs] = {}
        for rhs in ALIGNER_NAMES:
            keys = sorted(set(grouped[lhs].keys()) | set(grouped[rhs].keys()))
            if not keys:
                matrix[lhs][rhs] = 1.0
                continue
            same = sum(1 for key in keys if grouped[lhs].get(key, tuple()) == grouped[rhs].get(key, tuple()))
            matrix[lhs][rhs] = round(float(same) / float(len(keys)), 6)
    return matrix


def summarize_observability_audit(
    *,
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    per_aligner: dict[str, dict[str, Any]] = {}
    grouped_controls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_alignments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        grouped_controls[str(row["aligner"])].append(dict(row["control_summary"]))
        grouped_alignments[str(row["aligner"])].extend(list(row["alignment_rows"]))
    disagreement_matrix = _disagreement_matrix(
        [row for rows in grouped_alignments.values() for row in rows]
    )
    for aligner_name in ALIGNER_NAMES:
        control_summaries = grouped_controls.get(aligner_name, [])
        step_rates: dict[str, float] = {}
        per_control_coverage: dict[str, dict[str, float]] = {}
        for step_key in ("predicate_value", "agent_value", "theme_value"):
            values = [float(summary["aligned_step_rates"][step_key]) for summary in control_summaries]
            step_rates[step_key] = round(mean(values), 6) if values else 0.0
        for summary in control_summaries:
            per_control_coverage[str(summary["control_id"])] = dict(summary["aligned_step_rates"])
        per_aligner[aligner_name] = {
            "aligned_step_rates": step_rates,
            "per_control_coverage": per_control_coverage,
            "row_count": sum(int(summary["row_count"]) for summary in control_summaries),
            "unaligned_count": sum(int(summary["unaligned_count"]) for summary in control_summaries),
            "disagreement_matrix": disagreement_matrix,
        }
    baseline_rates = dict(per_aligner.get(ALIGNER_BASELINE, {}).get("aligned_step_rates") or {})
    selected_aligner = ALIGNER_BASELINE
    winning_nonbaseline = None
    winning_score = -1.0
    for aligner_name in (ALIGNER_SLOT, ALIGNER_BRIDGE):
        rates = dict(per_aligner.get(aligner_name, {}).get("aligned_step_rates") or {})
        score = max(float(rates.get("agent_value", 0.0)), float(rates.get("theme_value", 0.0)))
        if score > winning_score:
            winning_score = score
            winning_nonbaseline = aligner_name
    decision = "ambiguous"
    sealed = False
    if winning_nonbaseline is not None:
        rates = dict(per_aligner.get(winning_nonbaseline, {}).get("aligned_step_rates") or {})
        if (
            (
                float(rates.get("agent_value", 0.0)) >= 0.8
                and float(baseline_rates.get("agent_value", 0.0)) < 0.2
            )
            or (
                float(rates.get("theme_value", 0.0)) >= 0.8
                and float(baseline_rates.get("theme_value", 0.0)) < 0.2
            )
        ):
            decision = "rerun_c1_with_best_aligner"
            selected_aligner = winning_nonbaseline
            sealed = True
    if decision == "ambiguous":
        all_low = True
        for aligner_name in ALIGNER_NAMES:
            rates = dict(per_aligner.get(aligner_name, {}).get("aligned_step_rates") or {})
            if float(rates.get("agent_value", 0.0)) >= 0.2 or float(rates.get("theme_value", 0.0)) >= 0.2:
                all_low = False
                break
        if all_low and float(baseline_rates.get("predicate_value", 0.0)) >= 0.8:
            decision = "predicate_dominant"
            selected_aligner = ALIGNER_BASELINE
            sealed = True
    return {
        "aligners": per_aligner,
        "selected_aligner": selected_aligner,
        "decision": decision,
        "sealed": sealed,
    }


def _basis_gap_for_step(dev_h2_rows: list[dict[str, Any]], dev_h0_rows: list[dict[str, Any]], step_key: str, basis_ids: list[str]) -> float:
    h2_counter = p17._aggregate_basis_scores([row for row in dev_h2_rows if str(row["decoder_step_key"]) == step_key]).get(step_key, Counter())
    h0_counter = p17._aggregate_basis_scores([row for row in dev_h0_rows if str(row["decoder_step_key"]) == step_key]).get(step_key, Counter())
    if not basis_ids:
        return 0.0
    return round(
        mean(float(h2_counter.get(basis_id, 0.0)) - float(h0_counter.get(basis_id, 0.0)) for basis_id in basis_ids),
        6,
    )


def _generic_basis_ids(rows: list[dict[str, Any]], fallback_basis_ids: list[str]) -> list[str]:
    if rows:
        first = dict(rows[0])
        basis_ids = [str(value) for value in list(first.get("generic_basis_ids") or [])]
        if basis_ids:
            return basis_ids
    return list(fallback_basis_ids)


def build_cross_step_bridge_catalog(
    *,
    dev_h2_rows: list[dict[str, Any]],
    dev_h0_rows: list[dict[str, Any]],
    heldout_h2_rows: list[dict[str, Any]],
    heldout_h0_rows: list[dict[str, Any]],
    layer_index: int,
    source_run: str,
) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for source_step_key, target_step_key, pair_name in BRIDGE_PAIRS:
        ranked_heads = p17._top_heads_for_step(dev_h2_rows, dev_h0_rows, target_step_key)
        ranked_source_bases = p17._top_bases_for_step(dev_h2_rows, dev_h0_rows, source_step_key)
        ranked_target_bases = p17._top_bases_for_step(dev_h2_rows, dev_h0_rows, target_step_key)
        target_h2_scores = p17._aggregate_trace_scores([row for row in dev_h2_rows if str(row["decoder_step_key"]) == target_step_key])
        target_h0_scores = p17._aggregate_trace_scores([row for row in dev_h0_rows if str(row["decoder_step_key"]) == target_step_key])
        held_target_h2_scores = p17._aggregate_trace_scores([row for row in heldout_h2_rows if str(row["decoder_step_key"]) == target_step_key])
        held_target_h0_scores = p17._aggregate_trace_scores([row for row in heldout_h0_rows if str(row["decoder_step_key"]) == target_step_key])
        for template_name, template in STEP_TEMPLATE_FAMILIES:
            head_count = int(template["head_count"])
            basis_count = int(template["basis_count"])
            selected_heads = list(ranked_heads[:head_count])
            selected_source_bases = list(ranked_source_bases[:basis_count]) or list(ranked_target_bases[:basis_count])
            selected_target_bases = list(ranked_target_bases[:basis_count]) or list(ranked_source_bases[:basis_count])
            if len(selected_heads) < head_count:
                selected_heads.extend([value for value in range(8) if value not in selected_heads][: head_count - len(selected_heads)])
            if len(selected_source_bases) < basis_count:
                selected_source_bases.extend(
                    [f"prelogit_basis_{index}" for index in range(32) if f"prelogit_basis_{index}" not in selected_source_bases][
                        : basis_count - len(selected_source_bases)
                    ]
                )
            if len(selected_target_bases) < basis_count:
                selected_target_bases.extend(
                    [f"prelogit_basis_{index}" for index in range(32) if f"prelogit_basis_{index}" not in selected_target_bases][
                        : basis_count - len(selected_target_bases)
                    ]
                )
            dev_head_gap = mean(
                float(target_h2_scores.get((target_step_key, head), {}).get("score", 0.0))
                - float(target_h0_scores.get((target_step_key, head), {}).get("score", 0.0))
                for head in selected_heads
            )
            held_head_gap = mean(
                float(held_target_h2_scores.get((target_step_key, head), {}).get("score", 0.0))
                - float(held_target_h0_scores.get((target_step_key, head), {}).get("score", 0.0))
                for head in selected_heads
            )
            source_basis_gap = _basis_gap_for_step(dev_h2_rows, dev_h0_rows, source_step_key, selected_source_bases)
            target_basis_gap = _basis_gap_for_step(dev_h2_rows, dev_h0_rows, target_step_key, selected_target_bases)
            held_source_basis_gap = _basis_gap_for_step(heldout_h2_rows, heldout_h0_rows, source_step_key, selected_source_bases)
            held_target_basis_gap = _basis_gap_for_step(heldout_h2_rows, heldout_h0_rows, target_step_key, selected_target_bases)
            overlap_rows = [
                row
                for row in dev_h2_rows
                if str(row["decoder_step_key"]) == target_step_key and int(row["head_index"]) in selected_heads
            ]
            generic_overlap = round(mean(float(row.get("generic_decoder_gain_overlap") or 0.0) for row in overlap_rows), 6) if overlap_rows else 0.0
            store_condition = {"mode": "none"}
            if bool(template["use_store"]):
                store_condition = {"mode": "matched_role_store", "selector_type": STEP_STORE_SELECTOR[target_step_key]}
            bridge_basis_ids = list(dict.fromkeys(selected_source_bases + selected_target_bases))
            dev_contrast = round(float(dev_head_gap) + float(source_basis_gap) + float(target_basis_gap) - float(generic_overlap), 6)
            held_contrast = round(float(held_head_gap) + float(held_source_basis_gap) + float(held_target_basis_gap) - float(generic_overlap), 6)
            row = {
                "object_id": f"cross_step_bridge_{pair_name}_{template_name}",
                "object_class": BRIDGE_OBJECT_CLASS,
                "layer_index": int(layer_index),
                "source_run": str(source_run),
                "source_step_key": source_step_key,
                "target_step_key": target_step_key,
                "source_span_keys": list(STEP_SOURCE_KEYS[source_step_key]),
                "head_selector": {"mode": "explicit", "indices": selected_heads},
                "bridge_basis": {"mode": "explicit", "basis_ids": bridge_basis_ids},
                "store_condition": store_condition,
                "matched_control_type": "wrong_event_same_cardinality",
                "template_family": template_name,
                "pair_family": pair_name,
                "generic_basis_ids": _generic_basis_ids(overlap_rows, bridge_basis_ids),
                "dev_h2_contrast": dev_contrast,
                "dev_h0_contrast": 0.0,
                "heldout_h2_contrast": held_contrast,
                "heldout_h0_contrast": 0.0,
                "generic_gain_overlap": generic_overlap,
            }
            row["non_flat_survivor"] = bool(dev_contrast >= DISCOVERY_MIN_CONTRAST_GAP)
            catalog.append(row)
    return catalog


def build_query_source_relation_catalog(
    *,
    dev_h2_rows: list[dict[str, Any]],
    dev_h0_rows: list[dict[str, Any]],
    heldout_h2_rows: list[dict[str, Any]],
    heldout_h0_rows: list[dict[str, Any]],
    layer_index: int,
    source_run: str,
) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for step_key in ("predicate_value", "agent_value", "theme_value"):
        ranked_heads = p17._top_heads_for_step(dev_h2_rows, dev_h0_rows, step_key)
        ranked_bases = p17._top_bases_for_step(dev_h2_rows, dev_h0_rows, step_key)
        h2_scores = p17._aggregate_trace_scores([row for row in dev_h2_rows if str(row["decoder_step_key"]) == step_key])
        h0_scores = p17._aggregate_trace_scores([row for row in dev_h0_rows if str(row["decoder_step_key"]) == step_key])
        held_h2_scores = p17._aggregate_trace_scores([row for row in heldout_h2_rows if str(row["decoder_step_key"]) == step_key])
        held_h0_scores = p17._aggregate_trace_scores([row for row in heldout_h0_rows if str(row["decoder_step_key"]) == step_key])
        for template_name, template in STEP_TEMPLATE_FAMILIES:
            head_count = int(template["head_count"])
            basis_count = int(template["basis_count"])
            selected_heads = list(ranked_heads[:head_count])
            query_basis_ids = list(ranked_bases[:basis_count])
            read_basis_ids = list(ranked_bases[basis_count : basis_count * 2]) or list(ranked_bases[:basis_count])
            if len(selected_heads) < head_count:
                selected_heads.extend([value for value in range(8) if value not in selected_heads][: head_count - len(selected_heads)])
            if len(query_basis_ids) < basis_count:
                query_basis_ids.extend(
                    [f"prelogit_basis_{index}" for index in range(32) if f"prelogit_basis_{index}" not in query_basis_ids][
                        : basis_count - len(query_basis_ids)
                    ]
                )
            if len(read_basis_ids) < basis_count:
                read_basis_ids.extend(
                    [f"prelogit_basis_{index}" for index in range(32) if f"prelogit_basis_{index}" not in read_basis_ids][
                        : basis_count - len(read_basis_ids)
                    ]
                )
            dev_head_gap = mean(
                float(h2_scores.get((step_key, head), {}).get("score", 0.0))
                - float(h0_scores.get((step_key, head), {}).get("score", 0.0))
                for head in selected_heads
            )
            held_head_gap = mean(
                float(held_h2_scores.get((step_key, head), {}).get("score", 0.0))
                - float(held_h0_scores.get((step_key, head), {}).get("score", 0.0))
                for head in selected_heads
            )
            query_basis_gap = _basis_gap_for_step(dev_h2_rows, dev_h0_rows, step_key, query_basis_ids)
            read_basis_gap = _basis_gap_for_step(dev_h2_rows, dev_h0_rows, step_key, read_basis_ids)
            held_query_basis_gap = _basis_gap_for_step(heldout_h2_rows, heldout_h0_rows, step_key, query_basis_ids)
            held_read_basis_gap = _basis_gap_for_step(heldout_h2_rows, heldout_h0_rows, step_key, read_basis_ids)
            overlap_rows = [
                row for row in dev_h2_rows if str(row["decoder_step_key"]) == step_key and int(row["head_index"]) in selected_heads
            ]
            generic_overlap = round(mean(float(row.get("generic_decoder_gain_overlap") or 0.0) for row in overlap_rows), 6) if overlap_rows else 0.0
            store_condition = {"mode": "none"}
            if bool(template["use_store"]):
                store_condition = {"mode": "matched_role_store", "selector_type": STEP_STORE_SELECTOR[step_key]}
            dev_contrast = round(float(dev_head_gap) + float(query_basis_gap) + float(read_basis_gap) - float(generic_overlap), 6)
            held_contrast = round(float(held_head_gap) + float(held_query_basis_gap) + float(held_read_basis_gap) - float(generic_overlap), 6)
            row = {
                "object_id": f"query_source_relation_{step_key}_{template_name}",
                "object_class": QUERY_SOURCE_OBJECT_CLASS,
                "layer_index": int(layer_index),
                "source_run": str(source_run),
                "query_step_key": step_key,
                "source_span_keys": list(STEP_SOURCE_KEYS[step_key]),
                "head_selector": {"mode": "explicit", "indices": selected_heads},
                "query_basis": {"mode": "explicit", "basis_ids": query_basis_ids},
                "read_basis": {"mode": "explicit", "basis_ids": read_basis_ids},
                "store_condition": store_condition,
                "matched_control_type": "wrong_event_same_cardinality",
                "template_family": template_name,
                "step_family": step_key,
                "generic_basis_ids": _generic_basis_ids(overlap_rows, read_basis_ids),
                "dev_h2_contrast": dev_contrast,
                "dev_h0_contrast": 0.0,
                "heldout_h2_contrast": held_contrast,
                "heldout_h0_contrast": 0.0,
                "generic_gain_overlap": generic_overlap,
            }
            row["non_flat_survivor"] = bool(dev_contrast >= DISCOVERY_MIN_CONTRAST_GAP)
            catalog.append(row)
    return catalog


def summarize_family_catalog(
    *,
    catalog: list[dict[str, Any]],
    object_class: str,
) -> dict[str, Any]:
    survivors = [row for row in catalog if bool(row.get("non_flat_survivor"))]
    template_families = sorted({str(row["template_family"]) for row in survivors})
    if object_class == BRIDGE_OBJECT_CLASS:
        coverage = sorted({str(row["pair_family"]) for row in survivors})
    elif object_class == QUERY_SOURCE_OBJECT_CLASS:
        coverage = sorted({str(row["step_family"]) for row in survivors})
    else:
        coverage = sorted({str(row.get("decoder_step_key") or "") for row in survivors if str(row.get("decoder_step_key") or "")})
    accepted = len(catalog) == 12 and len(survivors) >= 8 and len(coverage) >= 2 and len(template_families) >= 2
    return {
        "object_class": object_class,
        "object_count": len(catalog),
        "surviving_candidate_count": len(survivors),
        "surviving_candidate_ids": [str(row["object_id"]) for row in survivors],
        "surviving_coverages": coverage,
        "surviving_template_families": template_families,
        "accepted": accepted,
        "decision": "continue_to_h2_gate" if accepted else "T-supported",
        "decision_status": "preliminary" if accepted else "final",
    }


def _wrong_bridge_source_step(source_step_key: str, target_step_key: str) -> str:
    if target_step_key == "agent_value":
        return "theme_value"
    if target_step_key == "theme_value":
        return "agent_value"
    return "agent_value" if source_step_key != "agent_value" else "theme_value"


def _align_with_selected(
    *,
    aligner_name: str,
    tokenizer: Any,
    target_text: str,
    record: dict[str, Any],
    forward: dict[str, Any],
    item_span_map: dict[str, Any],
    layer_index: int,
) -> dict[str, Any]:
    return resolve_decoder_steps(
        aligner_name=aligner_name,
        tokenizer=tokenizer,
        target_text=target_text,
        record=record,
        forward=forward,
        item_span_map=item_span_map,
        layer_index=layer_index,
    )


def _margin_mean_for_positions(forward: dict[str, Any], step_positions: list[int], compare_ids: list[int]) -> float:
    return float(
        p17._token_margin_summary(
            logits=forward["logits"],
            target_ids=forward["target_ids"],
            step_positions=step_positions,
            compare_ids=compare_ids,
        )["margin_mean"]
        or 0.0
    )


def _object_view(
    *,
    object_class: str,
    spec: dict[str, Any],
    alignments: dict[str, Any],
    item_span_map: dict[str, Any],
) -> dict[str, Any] | None:
    if object_class == GEOMETRY_INTERACTION_OBJECT_CLASS:
        step_key = str(spec["decoder_step_key"])
        step_positions = list(alignments["alignments"].get(step_key) or [])
        source_span_keys = [str(value) for value in list(spec["source_span_keys"])]
        if not step_positions:
            return None
        return {
            "metric_step_key": step_key,
            "metric_positions": step_positions,
            "intervention_positions": step_positions,
            "source_token_indices": p17._source_token_indices(item_span_map, source_span_keys),
            "wrong_source_token_indices": p17._source_token_indices(item_span_map, p17._wrong_source_span_keys(step_key)),
            "head_indices": [int(value) for value in list(dict(spec["head_selector"]).get("indices") or [])],
            "basis_indices": p17._basis_ids_to_indices(list(dict(spec["read_basis"]).get("basis_ids") or [])),
            "generic_basis_indices": p17._basis_ids_to_indices(list(spec.get("generic_basis_ids") or [])),
            "store_condition": dict(spec.get("store_condition") or {"mode": "none"}),
        }
    if object_class == BRIDGE_OBJECT_CLASS:
        source_step_key = str(spec["source_step_key"])
        target_step_key = str(spec["target_step_key"])
        source_positions = list(alignments["alignments"].get(source_step_key) or [])
        target_positions = list(alignments["alignments"].get(target_step_key) or [])
        if not source_positions or not target_positions:
            return None
        wrong_source_step = _wrong_bridge_source_step(source_step_key, target_step_key)
        wrong_source_positions = list(alignments["alignments"].get(wrong_source_step) or [])
        source_span_keys = [str(value) for value in list(spec["source_span_keys"])]
        wrong_span_keys = STEP_SOURCE_KEYS.get(wrong_source_step, STEP_SOURCE_KEYS["agent_value"])
        return {
            "metric_step_key": target_step_key,
            "metric_positions": target_positions,
            "intervention_positions": sorted(set(source_positions + target_positions)),
            "wrong_intervention_positions": sorted(set(wrong_source_positions + target_positions)) if wrong_source_positions else list(target_positions),
            "source_token_indices": p17._source_token_indices(item_span_map, source_span_keys),
            "wrong_source_token_indices": p17._source_token_indices(item_span_map, wrong_span_keys),
            "head_indices": [int(value) for value in list(dict(spec["head_selector"]).get("indices") or [])],
            "basis_indices": p17._basis_ids_to_indices(list(dict(spec["bridge_basis"]).get("basis_ids") or [])),
            "generic_basis_indices": p17._basis_ids_to_indices(list(spec.get("generic_basis_ids") or [])),
            "store_condition": dict(spec.get("store_condition") or {"mode": "none"}),
        }
    step_key = str(spec["query_step_key"])
    step_positions = list(alignments["alignments"].get(step_key) or [])
    source_span_keys = [str(value) for value in list(spec["source_span_keys"])]
    if not step_positions:
        return None
    return {
        "metric_step_key": step_key,
        "metric_positions": step_positions,
        "intervention_positions": step_positions,
        "source_token_indices": p17._source_token_indices(item_span_map, source_span_keys),
        "wrong_source_token_indices": p17._source_token_indices(item_span_map, p17._wrong_source_span_keys(step_key)),
        "head_indices": [int(value) for value in list(dict(spec["head_selector"]).get("indices") or [])],
        "basis_indices": list(
            dict.fromkeys(
                p17._basis_ids_to_indices(list(dict(spec["query_basis"]).get("basis_ids") or []))
                + p17._basis_ids_to_indices(list(dict(spec["read_basis"]).get("basis_ids") or []))
            )
        ),
        "generic_basis_indices": p17._basis_ids_to_indices(list(spec.get("generic_basis_ids") or [])),
        "store_condition": dict(spec.get("store_condition") or {"mode": "none"}),
    }


def evaluate_object_necessity(
    *,
    object_class: str,
    adapter: Any,
    layer_index: int,
    object_spec: dict[str, Any],
    aligner_name: str,
    candidate_records: list[dict[str, Any]],
    span_records: list[dict[str, Any]],
    max_length: int = 256,
) -> dict[str, Any]:
    span_index = p17._index_span_records(span_records, layer_index=layer_index)
    live_span_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    eligible = 0
    for item in candidate_records:
        item_span_map = p17._ensure_item_span_map(
            adapter=adapter,
            item=item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        baseline = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
        )
        alignments = _align_with_selected(
            aligner_name=aligner_name,
            tokenizer=adapter.tokenizer,
            target_text=str(item["target_form"]),
            record=item,
            forward=baseline,
            item_span_map=item_span_map,
            layer_index=layer_index,
        )
        view = _object_view(object_class=object_class, spec=object_spec, alignments=alignments, item_span_map=item_span_map)
        if view is None or not view["source_token_indices"]:
            continue
        eligible += 1
        metric_positions = list(view["metric_positions"])
        baseline_margin = _margin_mean_for_positions(baseline, metric_positions, [])
        head_mask = p17._build_cross_attn_head_mask(
            adapter.model,
            layer_index=layer_index,
            head_indices=list(view["head_indices"]),
            selected_scale=0.0,
        )
        actual = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=head_mask,
            encoder_token_indices=list(view["source_token_indices"]) if str(view["store_condition"].get("mode")) != "none" else None,
            encoder_scale=0.0 if str(view["store_condition"].get("mode")) != "none" else None,
            step_indices=list(view["intervention_positions"]),
            basis_indices=list(view["basis_indices"]),
            basis_mode="remove",
        )
        wrong_event = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=head_mask,
            encoder_token_indices=list(view["wrong_source_token_indices"]) if str(view["store_condition"].get("mode")) != "none" else None,
            encoder_scale=0.0 if str(view["store_condition"].get("mode")) != "none" else None,
            step_indices=list(view.get("wrong_intervention_positions") or view["intervention_positions"]),
            basis_indices=list(view["basis_indices"]),
            basis_mode="remove",
        )
        shuffled_heads = p17._shuffled_indices(
            int(getattr(adapter.model.config, "num_heads", 8) or 8),
            list(view["head_indices"]),
            len(list(view["head_indices"])),
            stable_seed(item["item_id"], object_spec["object_id"], "heads"),
        )
        shuffled_head_output = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=p17._build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=shuffled_heads, selected_scale=0.0),
            step_indices=list(view["intervention_positions"]),
            basis_indices=list(view["basis_indices"]),
            basis_mode="remove",
        )
        hidden_size = int(baseline["decoder_hidden"].shape[-1]) if getattr(baseline["decoder_hidden"], "ndim", 0) == 2 else 512
        shuffled_basis = p17._shuffled_indices(
            hidden_size,
            list(view["basis_indices"]),
            len(list(view["basis_indices"])),
            stable_seed(item["item_id"], object_spec["object_id"], "basis"),
        )
        shuffled_basis_output = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=head_mask,
            step_indices=list(view["intervention_positions"]),
            basis_indices=shuffled_basis,
            basis_mode="remove",
        )
        generic_output = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            step_indices=list(view["metric_positions"]),
            basis_indices=list(view["generic_basis_indices"] or view["basis_indices"]),
            basis_mode="remove",
        )

        actual_drop = baseline_margin - _margin_mean_for_positions(actual, metric_positions, [])
        wrong_event_drop = baseline_margin - _margin_mean_for_positions(wrong_event, metric_positions, [])
        shuffled_head_drop = baseline_margin - _margin_mean_for_positions(shuffled_head_output, metric_positions, [])
        shuffled_basis_drop = baseline_margin - _margin_mean_for_positions(shuffled_basis_output, metric_positions, [])
        generic_drop = baseline_margin - _margin_mean_for_positions(generic_output, metric_positions, [])
        strongest_control = aggregate_control_value(
            {
                "wrong_event_same_cardinality": wrong_event_drop,
                "shuffled_head_same_cardinality": shuffled_head_drop,
                "shuffled_basis_same_cardinality": shuffled_basis_drop,
                "generic_decoder_gain": generic_drop,
            },
            method="max",
            priority=(
                "wrong_event_same_cardinality",
                "shuffled_head_same_cardinality",
                "shuffled_basis_same_cardinality",
                "generic_decoder_gain",
            ),
        )
        rows.append(
            {
                "item_id": str(item["item_id"]),
                "stratum": str(item["stratum"]),
                "generalization_case": str(item["generalization_case"]),
                "metric_step_key": str(view["metric_step_key"]),
                "baseline_margin": round(baseline_margin, 6),
                "actual_drop": round(actual_drop, 6),
                "wrong_event_same_cardinality_drop": round(wrong_event_drop, 6),
                "shuffled_head_same_cardinality_drop": round(shuffled_head_drop, 6),
                "shuffled_basis_same_cardinality_drop": round(shuffled_basis_drop, 6),
                "generic_decoder_gain_drop": round(generic_drop, 6),
                "strongest_control_drop": round(float(strongest_control), 6),
                "necessity_gap": round(actual_drop - float(strongest_control), 6),
            }
        )
    h2_rows = [row for row in rows if str(row["stratum"]) == "pass"]
    h0_rows = [row for row in rows if str(row["stratum"]) == "control"]
    return {
        "object_id": str(object_spec["object_id"]),
        "object_class": object_class,
        "row_count": len(rows),
        "eligible_item_count": eligible,
        "realism_pass_rate": round(len(rows) / max(eligible, 1), 6),
        "actual_drop": round(mean(float(row["actual_drop"]) for row in rows), 6) if rows else None,
        "strongest_control_drop": round(mean(float(row["strongest_control_drop"]) for row in rows), 6) if rows else None,
        "h2_gap": round(mean(float(row["necessity_gap"]) for row in h2_rows), 6) if h2_rows else None,
        "h0_gap": round(mean(float(row["necessity_gap"]) for row in h0_rows), 6) if h0_rows else None,
        "rows": rows,
    }


def gate_h2_row(
    *,
    object_spec: dict[str, Any],
    dev_summary: dict[str, Any],
    heldout_summary: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    dev_h2_gap = dev_summary.get("h2_gap")
    dev_h0_gap = dev_summary.get("h0_gap")
    held_h2_gap = heldout_summary.get("h2_gap")
    held_h0_gap = heldout_summary.get("h0_gap")
    realism_pass_rate = min(float(dev_summary.get("realism_pass_rate") or 0.0), float(heldout_summary.get("realism_pass_rate") or 0.0))
    generic_overlap = float(object_spec.get("generic_gain_overlap") or 0.0)
    if dev_h2_gap is None or float(dev_h2_gap) < H2_NECESSITY_MIN_GAP:
        reasons.append("dev_h2_gap_below_threshold")
    if held_h2_gap is None or float(held_h2_gap) < H2_NECESSITY_MIN_GAP:
        reasons.append("heldout_h2_gap_below_threshold")
    if dev_h0_gap is None or abs(float(dev_h0_gap)) > H0_NECESSITY_MAX_ABS_GAP:
        reasons.append("dev_h0_gap_above_null_band")
    if held_h0_gap is None or abs(float(held_h0_gap)) > H0_NECESSITY_MAX_ABS_GAP:
        reasons.append("heldout_h0_gap_above_null_band")
    if realism_pass_rate < REALISM_MIN_PASS_RATE:
        reasons.append("realism_below_threshold")
    if generic_overlap > GENERIC_GAIN_MAX_OVERLAP:
        reasons.append("generic_gain_overlap_too_high")
    return {
        "object_id": str(object_spec["object_id"]),
        "object_class": str(object_spec["object_class"]),
        "passed": not reasons,
        "failure_reasons": reasons,
        "dev_h2_gap": dev_h2_gap,
        "dev_h0_gap": dev_h0_gap,
        "heldout_h2_gap": held_h2_gap,
        "heldout_h0_gap": held_h0_gap,
        "realism_pass_rate": round(realism_pass_rate, 6),
        "generic_gain_overlap": generic_overlap,
    }


def summarize_h2_gate(
    *,
    object_class: str,
    rows: list[dict[str, Any]],
    discovery_summary: dict[str, Any],
) -> dict[str, Any]:
    passed = [row for row in rows if bool(row.get("passed"))]
    return {
        "object_class": object_class,
        "gate_ready_candidate_count": len(rows),
        "passed_count": len(passed),
        "passed_candidate_ids": [str(row["object_id"]) for row in passed],
        "rows": rows,
        "decision": "continue" if passed else "T-supported",
        "decision_status": "preliminary" if passed else "final",
        "discovery_summary": discovery_summary,
    }


def _top_plausible_objects(catalog: list[dict[str, Any]], limit: int = 2) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in catalog
        if bool(row.get("non_flat_survivor"))
        and float(row.get("generic_gain_overlap") or 0.0) <= GENERIC_GAIN_MAX_OVERLAP
    ]
    ranked = sorted(
        filtered,
        key=lambda row: float(row.get("dev_h2_contrast") or 0.0) - float(row.get("dev_h0_contrast") or 0.0),
        reverse=True,
    )
    return [dict(row) for row in ranked[:limit]]


def _counterpart_role(metric_step_key: str) -> str:
    if metric_step_key == "theme_value":
        return "theme"
    return "agent"


def evaluate_object_direct_h1(
    *,
    object_class: str,
    adapter: Any,
    layer_index: int,
    object_spec: dict[str, Any],
    aligner_name: str,
    candidate_records: list[dict[str, Any]],
    span_records: list[dict[str, Any]],
    physical_family: str,
    max_length: int = 256,
) -> dict[str, Any]:
    span_index = p17._index_span_records(span_records, layer_index=layer_index)
    live_span_cache: dict[str, dict[str, Any]] = {}
    counterpart_index = p17.build_exact_counterpart_index(candidate_records)
    rows: list[dict[str, Any]] = []
    for item in candidate_records:
        if str(item.get("stratum")) != "fail":
            continue
        item_span_map = p17._ensure_item_span_map(
            adapter=adapter,
            item=item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        baseline = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
        )
        alignments = _align_with_selected(
            aligner_name=aligner_name,
            tokenizer=adapter.tokenizer,
            target_text=str(item["target_form"]),
            record=item,
            forward=baseline,
            item_span_map=item_span_map,
            layer_index=layer_index,
        )
        view = _object_view(object_class=object_class, spec=object_spec, alignments=alignments, item_span_map=item_span_map)
        if view is None or not view["source_token_indices"]:
            continue
        counterpart = p17.find_exact_role_counterpart(
            record=item,
            role=_counterpart_role(str(view["metric_step_key"])),
            counterpart_index=counterpart_index,
            preferred_strata=("pass", "control"),
            same_split_only=False,
        )
        if counterpart is None:
            continue
        counterpart_span_map = p17._ensure_item_span_map(
            adapter=adapter,
            item=counterpart,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        metric_positions = list(view["metric_positions"])
        baseline_summary = p17._token_margin_summary(
            logits=baseline["logits"],
            target_ids=baseline["target_ids"],
            step_positions=metric_positions,
            compare_ids=p17._counterpart_compare_ids(
                tokenizer=adapter.tokenizer,
                step_key=str(view["metric_step_key"]),
                target_positions=metric_positions,
                counterpart_record=counterpart,
            ),
        )
        if bool(baseline_summary["exact_match"]):
            continue
        source_keys = [str(value) for value in list(object_spec.get("source_span_keys") or STEP_SOURCE_KEYS.get(str(view["metric_step_key"]), []))]
        store_patch_indices, store_patch_vectors = p17._store_patch_vectors(
            target_span_map=item_span_map,
            donor_span_map=counterpart_span_map,
            source_span_keys=source_keys,
        )
        wrong_patch_indices, wrong_patch_vectors = p17._store_patch_vectors(
            target_span_map=item_span_map,
            donor_span_map=counterpart_span_map,
            source_span_keys=source_keys,
            donor_source_span_keys=p17._wrong_source_span_keys(str(view["metric_step_key"])),
        )
        head_mask = p17._build_cross_attn_head_mask(
            adapter.model,
            layer_index=layer_index,
            head_indices=list(view["head_indices"]),
            selected_scale=HEAD_BOOST_SCALE,
        )
        actual_only = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=head_mask,
            step_indices=list(view["intervention_positions"]),
            basis_indices=list(view["basis_indices"]),
            basis_mode="boost",
            basis_scale=READ_BOOST_SCALE,
        )
        actual_plus_store = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=head_mask,
            encoder_token_indices=store_patch_indices if str(view["store_condition"].get("mode")) != "none" else None,
            encoder_replacement_vectors=store_patch_vectors if str(view["store_condition"].get("mode")) != "none" else None,
            step_indices=list(view["intervention_positions"]),
            basis_indices=list(view["basis_indices"]),
            basis_mode="boost",
            basis_scale=READ_BOOST_SCALE,
        )
        wrong_reinstate = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=head_mask,
            encoder_token_indices=wrong_patch_indices if str(view["store_condition"].get("mode")) != "none" else None,
            encoder_replacement_vectors=wrong_patch_vectors if str(view["store_condition"].get("mode")) != "none" else None,
            step_indices=list(view["intervention_positions"]),
            basis_indices=list(view["basis_indices"]),
            basis_mode="boost",
            basis_scale=READ_BOOST_SCALE,
        )
        path_only = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=head_mask,
        )
        generic_gain = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            step_indices=list(metric_positions),
            basis_indices=list(view["generic_basis_indices"] or view["basis_indices"]),
            basis_mode="boost",
            basis_scale=READ_BOOST_SCALE,
        )

        def margin_gain(forward: dict[str, Any]) -> tuple[float, bool]:
            summary = p17._token_margin_summary(
                logits=forward["logits"],
                target_ids=forward["target_ids"],
                step_positions=metric_positions,
                compare_ids=p17._counterpart_compare_ids(
                    tokenizer=adapter.tokenizer,
                    step_key=str(view["metric_step_key"]),
                    target_positions=metric_positions,
                    counterpart_record=counterpart,
                ),
            )
            return float(summary["margin_mean"] or 0.0) - float(baseline_summary["margin_mean"] or 0.0), bool(summary["exact_match"])

        actual_only_gain, actual_only_exact = margin_gain(actual_only)
        actual_plus_store_gain, actual_plus_store_exact = margin_gain(actual_plus_store)
        wrong_gain, _ = margin_gain(wrong_reinstate)
        path_gain, _ = margin_gain(path_only)
        generic_gain_value, _ = margin_gain(generic_gain)
        strongest_control = aggregate_control_value(
            {
                "wrong_object_reinstate": wrong_gain,
                "path_only": path_gain,
                "generic_decoder_gain": generic_gain_value,
            },
            method="max",
            priority=("wrong_object_reinstate", "path_only", "generic_decoder_gain"),
        )
        best_actual = max(actual_only_gain, actual_plus_store_gain)
        rows.append(
            {
                "item_id": str(item["item_id"]),
                "generalization_case": str(item["generalization_case"]),
                "physical_family": str(physical_family),
                "object_class": object_class,
                "baseline_margin": round(float(baseline_summary["margin_mean"] or 0.0), 6),
                "object_reinstate_only_gain": round(actual_only_gain, 6),
                "object_reinstate_plus_store_gain": round(actual_plus_store_gain, 6),
                "wrong_object_reinstate_gain": round(wrong_gain, 6),
                "path_only_gain": round(path_gain, 6),
                "generic_decoder_gain": round(generic_gain_value, 6),
                "best_object_actual_gain": round(best_actual, 6),
                "strongest_control_gain": round(float(strongest_control), 6),
                "rescue_gap": round(best_actual - float(strongest_control), 6),
                "object_exact_match": bool(actual_only_exact or actual_plus_store_exact),
            }
        )
    return {
        "object_id": str(object_spec["object_id"]),
        "object_class": object_class,
        "physical_family": physical_family,
        "row_count": len(rows),
        "rows": rows,
    }


def summarize_direct_h1(
    *,
    object_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped_object_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in object_rows:
        grouped_object_rows[(str(row["object_class"]), str(row["object_id"]))].extend(list(row.get("rows") or []))
    summaries: list[dict[str, Any]] = []
    for (object_class, object_id), flattened in grouped_object_rows.items():
        grouped_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item_row in flattened:
            grouped_by_family[str(item_row["physical_family"])].append(item_row)
        passing_families: list[str] = []
        generic_advantages: list[float] = []
        exact_hits: list[int] = []
        for family_id, family_rows in grouped_by_family.items():
            rescue_gap = mean(float(item["rescue_gap"]) for item in family_rows)
            best_actual = mean(float(item["best_object_actual_gain"]) for item in family_rows)
            generic_gain = mean(float(item["generic_decoder_gain"]) for item in family_rows)
            if rescue_gap >= H1_RESCUE_MIN_GAP:
                passing_families.append(family_id)
            generic_advantages.append(round(best_actual - generic_gain, 6))
            exact_hits.extend(int(bool(item["object_exact_match"])) for item in family_rows)
        exact_rate = round(mean(exact_hits), 6) if exact_hits else 0.0
        passed = (
            len(set(passing_families)) >= 2
            and max(generic_advantages or [0.0]) >= H1_GENERIC_ADVANTAGE_MIN
            and min(generic_advantages or [0.0]) >= 0.0
            and exact_rate >= H1_RESCUE_MIN_BINARY_RATE
        )
        summaries.append(
            {
                "object_class": object_class,
                "object_id": object_id,
                "passed": passed,
                "passing_families": sorted(set(passing_families)),
                "aggregated_exact_match_rate": exact_rate,
                "generic_advantages": generic_advantages,
                "row_count": len(flattened),
            }
        )
    passed_rows = [row for row in summaries if bool(row["passed"])]
    return {
        "passed_count": len(passed_rows),
        "rows": summaries,
        "decision": "global_h1_supported" if passed_rows else "global_h1_failed",
    }


def case_condition_direct_h1(
    *,
    rescue_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    slices: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for object_row in rescue_rows:
        for row in list(object_row.get("rows") or []):
            key = (str(row["generalization_case"]), str(row["object_class"]))
            slices[key].append(row)
    accepted: list[dict[str, Any]] = []
    for (generalization_case, object_class), rows in slices.items():
        families = sorted({str(row["physical_family"]) for row in rows})
        rescue_gap = round(mean(float(row["rescue_gap"]) for row in rows), 6)
        exact_rate = round(mean(int(bool(row["object_exact_match"])) for row in rows), 6)
        if rescue_gap >= CASE_MIN_SLICE_RESCUE_GAP and len(families) > 1 and exact_rate >= CASE_MIN_SLICE_BINARY_RATE:
            accepted.append(
                {
                    "generalization_case": generalization_case,
                    "object_class": object_class,
                    "physical_families": families,
                    "slice_rescue_gap": rescue_gap,
                    "slice_exact_match_rescue_rate": exact_rate,
                }
            )
    return {
        "accepted_slice_count": len(accepted),
        "accepted_slices": accepted,
    }


def summarize_family_closure(
    *,
    observability_summary: dict[str, Any],
    class_summaries: list[dict[str, Any]],
    h2_summaries: list[dict[str, Any]],
    direct_h1_summary: dict[str, Any],
    case_summary: dict[str, Any],
) -> dict[str, Any]:
    if not bool(observability_summary.get("sealed")):
        return {
            "decision": "halted",
            "decision_status": "blocked",
            "explanation": "Observability audit remained ambiguous.",
        }
    h2_passed = any(int(summary.get("passed_count") or 0) > 0 for summary in h2_summaries)
    direct_h1_passed = int(direct_h1_summary.get("passed_count") or 0) > 0
    local_h1_passed = int(case_summary.get("accepted_slice_count") or 0) >= 2
    all_classes_done = len(class_summaries) == 3 and all("accepted" in summary for summary in class_summaries)
    if h2_passed and direct_h1_passed:
        return {
            "decision": "U-supported",
            "decision_status": "final",
            "explanation": "At least one object class passed H2 and direct H1 rescue also survived globally.",
        }
    if h2_passed and local_h1_passed and not direct_h1_passed:
        return {
            "decision": "L-supported",
            "decision_status": "final",
            "explanation": "No global H1 survived, but at least two local slices support local-only H1.",
        }
    if all_classes_done and not h2_passed and not direct_h1_passed and not local_h1_passed:
        return {
            "decision": "T-supported",
            "decision_status": "family-final",
            "explanation": "Observability was sealed, all three object classes failed H2, and direct H1 failed globally and slice-locally.",
        }
    return {
        "decision": "halted",
        "decision_status": "blocked",
        "explanation": "Phase 18 did not reach a valid branch decision.",
    }


def render_family_closure_status_markdown(
    *,
    observability_summary: dict[str, Any],
    class_summaries: list[dict[str, Any]],
    h2_summaries: list[dict[str, Any]],
    direct_h1_summary: dict[str, Any],
    case_summary: dict[str, Any],
    closure_summary: dict[str, Any],
) -> str:
    lines = [
        "# Phase 18 Status",
        "",
        "## Frozen Framing",
        "- `G1_binding`: bounded hypothesis-final success",
        "- `G2_distributed_geometry`: partial adjudication unless Phase 18 reaches family-level closure",
        "- Repository strongest framing stays `bounded protocol + partial adjudication` until Phase 18 reaches a final family-level branch.",
        "",
        "## Observability",
        f"- `decision = {observability_summary.get('decision')}`",
        f"- `selected_aligner = {observability_summary.get('selected_aligner')}`",
        f"- `sealed = {bool(observability_summary.get('sealed'))}`",
        "",
        "## Class Discovery",
    ]
    for summary in class_summaries:
        lines.append(
            f"- `{summary.get('object_class')}`: `object_count = {int(summary.get('object_count') or 0)}`, "
            f"`surviving_candidate_count = {int(summary.get('surviving_candidate_count') or 0)}`, "
            f"`accepted = {bool(summary.get('accepted'))}`"
        )
    lines.extend(
        [
            "",
            "## H2 Gate",
        ]
    )
    for summary in h2_summaries:
        lines.append(
            f"- `{summary.get('object_class')}`: `passed_count = {int(summary.get('passed_count') or 0)}`, "
            f"`decision = {summary.get('decision')}`"
        )
    lines.extend(
        [
            "",
            "## Direct H1",
            f"- `passed_count = {int(direct_h1_summary.get('passed_count') or 0)}`",
            "",
            "## Slice-Local H1",
            f"- `accepted_slice_count = {int(case_summary.get('accepted_slice_count') or 0)}`",
            "",
            "## Closure",
            f"- `decision = {closure_summary.get('decision')}`",
            f"- `decision_status = {closure_summary.get('decision_status')}`",
            "",
            "## Interpretation",
            f"- {closure_summary.get('explanation')}",
        ]
    )
    return "\n".join(lines) + "\n"
