from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median

from .schema import CanonicalExample, SemanticQuartet


def build_phase1_report(
    examples: list[CanonicalExample], quartets: list[SemanticQuartet]
) -> dict[str, object]:
    return {
        "example_count": len(examples),
        "split_stats": split_stats(examples),
        "duplicate_leakage": duplicate_leakage(examples),
        "surface_balance": surface_balance(examples),
        "quartet_validation": quartet_validation(examples, quartets),
    }


def split_stats(examples: list[CanonicalExample]) -> dict[str, object]:
    by_split: dict[str, list[CanonicalExample]] = defaultdict(list)
    for example in examples:
        by_split[example.split].append(example)

    report: dict[str, object] = {}
    for split, split_examples in sorted(by_split.items()):
        input_lengths = [int(example.surface_features["input_length_tokens"]) for example in split_examples]
        target_lengths = [int(example.surface_features["target_length_tokens"]) for example in split_examples]
        cases = Counter(example.generalization_case for example in split_examples)
        report[split] = {
            "count": len(split_examples),
            "avg_input_length": round(mean(input_lengths), 3),
            "median_input_length": median(input_lengths),
            "avg_target_length": round(mean(target_lengths), 3),
            "median_target_length": median(target_lengths),
            "generalization_cases_top10": cases.most_common(10),
        }
    return report


def duplicate_leakage(examples: list[CanonicalExample]) -> dict[str, object]:
    pair_to_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    input_to_splits: dict[str, set[str]] = defaultdict(set)

    for example in examples:
        pair_to_splits[(example.input_text, example.target_form)].add(example.split)
        input_to_splits[example.input_text].add(example.split)

    cross_split_pairs = [
        {"input_text": key[0], "target_form": key[1], "splits": sorted(splits)}
        for key, splits in pair_to_splits.items()
        if len(splits) > 1
    ]
    repeated_inputs = [
        {"input_text": input_text, "splits": sorted(splits)}
        for input_text, splits in input_to_splits.items()
        if len(splits) > 1
    ]

    return {
        "cross_split_pair_count": len(cross_split_pairs),
        "cross_split_pair_examples": cross_split_pairs[:20],
        "cross_split_input_count": len(repeated_inputs),
        "cross_split_input_examples": repeated_inputs[:20],
    }


def surface_balance(examples: list[CanonicalExample]) -> dict[str, object]:
    by_split: dict[str, list[CanonicalExample]] = defaultdict(list)
    for example in examples:
        by_split[example.split].append(example)

    balance: dict[str, object] = {}
    for split, split_examples in sorted(by_split.items()):
        input_lengths = [int(example.surface_features["input_length_tokens"]) for example in split_examples]
        target_lengths = [int(example.surface_features["target_length_tokens"]) for example in split_examples]
        balance[split] = {
            "min_input_length": min(input_lengths) if input_lengths else 0,
            "max_input_length": max(input_lengths) if input_lengths else 0,
            "min_target_length": min(target_lengths) if target_lengths else 0,
            "max_target_length": max(target_lengths) if target_lengths else 0,
        }
    return balance


def quartet_validation(
    examples: list[CanonicalExample], quartets: list[SemanticQuartet]
) -> dict[str, object]:
    known_ids = {example.item_id for example in examples}
    invalid = []
    for quartet in quartets:
        missing_ids = [item_id for item_id in quartet.item_ids if item_id not in known_ids]
        if missing_ids:
            invalid.append({"quartet_id": quartet.quartet_id, "missing_ids": missing_ids})

    return {
        "quartet_count": len(quartets),
        "invalid_quartet_count": len(invalid),
        "invalid_quartets": invalid[:20],
    }
