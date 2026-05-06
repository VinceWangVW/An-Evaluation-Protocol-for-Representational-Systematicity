from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable

from data.canonicalize import infer_role_signature, infer_surface_features


def join_behavior_predictions(
    benchmark_examples: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_item_id = {example["item_id"]: example for example in benchmark_examples}
    enriched: list[dict[str, Any]] = []

    for record in prediction_records:
        item_id = str(record["item_id"])
        example = by_item_id.get(item_id)
        if example is None:
            continue

        target_form = str(example.get("target_form") or "")
        input_text = str(example.get("input_text") or "")
        roles = dict(example.get("role_signature") or infer_role_signature(input_text, target_form))
        surface = dict(example.get("surface_features") or infer_surface_features(input_text, target_form))
        metadata = dict(example.get("metadata") or {})
        enriched.append(
            {
                **record,
                "input_text": input_text,
                "target_form": target_form,
                "lexical_signature": list(example.get("lexical_signature") or []),
                "role_signature": roles,
                "surface_features": surface,
                "metadata": metadata,
                "candidate_key": build_candidate_key(
                    benchmark_name=str(example["benchmark_name"]),
                    generalization_case=str(record["generalization_case"]),
                    role_signature=roles,
                    surface_features=surface,
                ),
            }
        )

    return enriched


def join_phase2_predictions(
    benchmark_examples: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return join_behavior_predictions(benchmark_examples, prediction_records)


def build_candidate_key(
    benchmark_name: str,
    generalization_case: str,
    role_signature: dict[str, Any],
    surface_features: dict[str, Any],
) -> str:
    frame = str(role_signature.get("frame") or "unknown")
    predicate = str(role_signature.get("predicate") or "unknown")
    input_length = int(surface_features.get("input_length_tokens") or -1)
    return "|".join([benchmark_name, generalization_case, frame, predicate, str(input_length)])


def strict_match_key(record: dict[str, Any]) -> tuple[str, str, str, str, int]:
    roles = record.get("role_signature") or {}
    surface = record.get("surface_features") or {}
    return (
        str(record["benchmark_name"]),
        str(record["generalization_case"]),
        str(roles.get("frame") or "unknown"),
        str(roles.get("predicate") or "unknown"),
        int(surface.get("input_length_tokens") or -1),
    )


def relaxed_match_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    roles = record.get("role_signature") or {}
    surface = record.get("surface_features") or {}
    return (
        str(record["benchmark_name"]),
        str(record["generalization_case"]),
        str(roles.get("frame") or "unknown"),
        int(surface.get("input_length_tokens") or -1),
    )


def select_balanced_candidates(
    records: list[dict[str, Any]],
    max_items_per_stratum: int = 16,
    strata: tuple[str, ...] = ("pass", "fail", "control"),
    required_role_keys: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    for stratum in strata:
        grouped: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        stratum_records = [
            record
            for record in records
            if str(record.get("stratum")) == stratum
            and bool(record.get("role_signature"))
            and has_required_roles(record, required_role_keys)
        ]
        for record in sorted(
            stratum_records,
            key=lambda item: (
                str(item.get("generalization_case")),
                str(item.get("item_id")),
            ),
        ):
            grouped[str(record["generalization_case"])].append(record)

        case_names = sorted(grouped)
        while len([record for record in selected if str(record["stratum"]) == stratum]) < max_items_per_stratum:
            advanced = False
            for case_name in case_names:
                bucket = grouped[case_name]
                if not bucket:
                    continue
                selected.append(bucket.popleft())
                advanced = True
                if len([record for record in selected if str(record["stratum"]) == stratum]) >= max_items_per_stratum:
                    break
            if not advanced:
                break

    return selected


def has_required_roles(
    record: dict[str, Any],
    required_role_keys: tuple[str, ...] | None,
) -> bool:
    if not required_role_keys:
        return True
    roles = dict(record.get("role_signature") or {})
    return all(bool(roles.get(role_key)) for role_key in required_role_keys)


def build_matched_audit_sets(
    records: list[dict[str, Any]],
    required_strata: tuple[str, ...] = ("pass", "fail"),
    max_sets: int = 24,
) -> list[dict[str, Any]]:
    matches = _build_matches(records, required_strata=required_strata, key_fn=strict_match_key, level="strict")
    if len(matches) < max_sets:
        used_item_ids = {item_id for match in matches for item_id in match["members"].values()}
        remaining = [record for record in records if str(record["item_id"]) not in used_item_ids]
        matches.extend(
            _build_matches(
                remaining,
                required_strata=required_strata,
                key_fn=relaxed_match_key,
                level="relaxed",
            )
        )
    return matches[:max_sets]


def _build_matches(
    records: list[dict[str, Any]],
    required_strata: tuple[str, ...],
    key_fn: Any,
    level: str,
) -> list[dict[str, Any]]:
    grouped: dict[Any, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[key_fn(record)][str(record["stratum"])].append(record)

    matches: list[dict[str, Any]] = []
    used_item_ids: set[str] = set()

    for key in sorted(grouped, key=lambda item: tuple(str(piece) for piece in item)):
        strata_bucket = grouped[key]
        if any(not strata_bucket.get(stratum) for stratum in required_strata):
            continue

        members: dict[str, str] = {}
        member_cases: dict[str, str] = {}
        failed = False
        for stratum in required_strata:
            candidate = _first_unused(strata_bucket[stratum], used_item_ids)
            if candidate is None:
                failed = True
                break
            members[stratum] = str(candidate["item_id"])
            member_cases[stratum] = str(candidate["generalization_case"])

        if failed:
            continue

        used_item_ids.update(members.values())
        matches.append(
            {
                "match_level": level,
                "match_key": [str(piece) for piece in key] if isinstance(key, tuple) else [str(key)],
                "members": members,
                "member_generalization_cases": member_cases,
            }
        )

    return matches


def _first_unused(records: Iterable[dict[str, Any]], used_item_ids: set[str]) -> dict[str, Any] | None:
    for record in records:
        item_id = str(record["item_id"])
        if item_id not in used_item_ids:
            return record
    return None
