from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .canonicalize import (
    infer_generalization_case,
    infer_lexical_signature,
    infer_role_signature,
    infer_surface_features,
    normalize_whitespace,
)
from .systematicity_schema import SystematicitySchemaRecord, REQUIRED_COUNTERPART_TYPES, build_counterpart_families
from .schema import CanonicalExample


def validate_surface_candidate(record: SystematicitySchemaRecord, candidate: dict[str, Any]) -> dict[str, Any]:
    text = normalize_whitespace(str(candidate.get("text", "")))
    lowered = text.lower()
    predicate_surface = record.predicate_past if record.voice == "active" else record.predicate_participle
    checks = {
        "has_agent": record.agent.lower() in lowered,
        "has_theme": record.theme.lower() in lowered,
        "has_agent_modifier": record.agent_modifier.lower() in lowered,
        "has_theme_modifier": record.theme_modifier.lower() in lowered,
        "has_predicate_surface": predicate_surface.lower() in lowered,
        "has_no_pronouns": not any(pronoun in lowered.split() for pronoun in ("he", "she", "they", "them", "his", "her", "their")),
        "voice_order_valid": _validate_voice_order(text, record),
        "context_requirement_valid": _validate_context_requirement(lowered, record),
    }
    valid = all(checks.values())
    return {
        **candidate,
        "text": text,
        "validation": checks,
        "valid": valid,
    }


def select_primary_candidates(
    *,
    schema_records: Iterable[SystematicitySchemaRecord],
    validated_candidates: Iterable[dict[str, Any]],
    selection_strategy: str = "shortest",
) -> tuple[list[dict[str, Any]], list[CanonicalExample], dict[str, Any]]:
    schema_by_id = {record.item_id: record for record in schema_records}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in validated_candidates:
        grouped[str(candidate["item_id"])].append(candidate)

    selected_candidates: list[dict[str, Any]] = []
    canonical_examples: list[CanonicalExample] = []
    valid_count = 0
    invalid_count = 0

    for item_id, record in schema_by_id.items():
        candidates = grouped.get(item_id, [])
        valid_candidates = [candidate for candidate in candidates if candidate.get("valid")]
        valid_count += len(valid_candidates)
        invalid_count += len(candidates) - len(valid_candidates)
        if not valid_candidates:
            continue
        if selection_strategy == "longest":
            selected = sorted(valid_candidates, key=lambda candidate: len(candidate["text"]), reverse=True)[0]
        else:
            selected = sorted(valid_candidates, key=lambda candidate: len(candidate["text"]))[0]
        selected["selected_as_primary"] = True
        selected_candidates.append(selected)
        canonical_examples.append(_build_canonical_example(record, selected["text"]))

    family_summary = summarize_counterpart_integrity(
        schema_records=schema_by_id.values(),
        selected_item_ids={candidate["item_id"] for candidate in selected_candidates},
    )
    summary = {
        "schema_record_count": len(schema_by_id),
        "surface_candidate_count": sum(len(candidates) for candidates in grouped.values()),
        "valid_candidate_count": valid_count,
        "invalid_candidate_count": invalid_count,
        "primary_selected_count": len(selected_candidates),
        "canonical_example_count": len(canonical_examples),
        "selection_strategy": selection_strategy,
        **family_summary,
    }
    return selected_candidates, canonical_examples, summary


def summarize_counterpart_integrity(
    *,
    schema_records: Iterable[SystematicitySchemaRecord],
    selected_item_ids: set[str],
) -> dict[str, Any]:
    families = build_counterpart_families(schema_records)
    total_families = len(families)
    valid_families = 0
    usable_m2 = 0
    usable_m4 = 0
    by_construction = Counter()

    for family in families:
        member_ids = set(family["members"].values())
        if all(member_id in selected_item_ids for member_id in member_ids):
            valid_families += 1
            by_construction[family["construction_family"]] += 1
        selected_members = member_ids & selected_item_ids
        if family["members"]["rescue_target"] in selected_members:
            for relation in family["m2_pairs"]:
                if relation["counterpart_item_id"] in selected_members:
                    usable_m2 += 1
        if family["members"]["rescue_target"] in selected_members and family["members"]["rescue_donor"] in selected_members:
            usable_m4 += 1

    return {
        "family_count": total_families,
        "fully_validated_family_count": valid_families,
        "exact_counterpart_coverage_rate": (valid_families / total_families) if total_families else 0.0,
        "usable_m2_pair_count": usable_m2,
        "usable_m4_pair_count": usable_m4,
        "validated_family_counts_by_construction": dict(by_construction),
    }


def summarize_validated_candidates(validated_candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    validated_candidates = list(validated_candidates)
    check_pass_counts = Counter()
    valid_count = 0
    for candidate in validated_candidates:
        if candidate.get("valid"):
            valid_count += 1
        for check_name, matched in candidate.get("validation", {}).items():
            if matched:
                check_pass_counts[check_name] += 1
    total = len(validated_candidates)
    return {
        "candidate_count": total,
        "valid_count": valid_count,
        "valid_rate": (valid_count / total) if total else 0.0,
        "check_pass_rates": {
            key: (value / total) if total else 0.0 for key, value in sorted(check_pass_counts.items())
        },
    }


def _build_canonical_example(record: SystematicitySchemaRecord, text: str) -> CanonicalExample:
    role_signature = infer_role_signature(text, record.target_form)
    metadata = {
        **record.metadata,
        "schema_id": record.schema_id,
        "counterpart_family_id": record.counterpart_family_id,
        "counterpart_type": record.counterpart_type,
        "construction_family": record.construction_family,
        "voice": record.voice,
        "embedding_depth": record.embedding_depth,
        "predicate_lemma": record.predicate_lemma,
        "context_predicate_lemma": record.context_predicate_lemma,
        "context_filler": record.context_filler,
    }
    return CanonicalExample(
        item_id=record.item_id,
        split=record.split,
        raw_split=record.raw_split,
        benchmark_name=record.benchmark_name,
        input_text=text,
        target_form=record.target_form,
        generalization_case=infer_generalization_case(record.construction_family, metadata),
        lexical_signature=infer_lexical_signature(text, role_signature),
        role_signature=role_signature,
        surface_features=infer_surface_features(text, record.target_form),
        metadata=metadata,
    )


def _validate_voice_order(text: str, record: SystematicitySchemaRecord) -> bool:
    lowered = text.lower()
    agent_index = lowered.find(record.agent.lower())
    theme_index = lowered.find(record.theme.lower())
    predicate_index = lowered.find(
        (record.predicate_past if record.voice == "active" else record.predicate_participle).lower()
    )
    if min(agent_index, theme_index, predicate_index) == -1:
        return False
    if record.voice == "active":
        return agent_index < predicate_index < theme_index
    by_index = lowered.find(" by ")
    return theme_index < predicate_index and by_index != -1 and by_index < agent_index


def _validate_context_requirement(lowered: str, record: SystematicitySchemaRecord) -> bool:
    if record.embedding_depth <= 0:
        return True
    return (
        record.context_filler.lower() in lowered
        and record.context_predicate_past.lower() in lowered
        and (" who " in f" {lowered} " or " that " in f" {lowered} ")
    )
