from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from .metrics import score_prediction
from .stratify import assign_behavior_stratum


def evaluate_benchmark(
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    model: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.fit(train_examples)
    predictions = model.predict_many(eval_examples)

    if len(predictions) != len(eval_examples):
        raise ValueError("Prediction count does not match eval example count")

    records: list[dict[str, Any]] = []
    for example, prediction in zip(eval_examples, predictions):
        metrics = score_prediction(prediction=prediction, gold=str(example["target_form"]))
        stratum = assign_behavior_stratum(example, metrics)
        record = {
            "item_id": example["item_id"],
            "benchmark_name": example["benchmark_name"],
            "split": example["split"],
            "generalization_case": example["generalization_case"],
            "prediction": prediction,
            "gold": example["target_form"],
            "stratum": stratum,
            **metrics,
        }
        records.append(record)

    summary = summarize_records(records)
    return records, summary


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    overall = aggregate_metrics(records)
    by_split = group_summary(records, key="split")
    by_case = group_summary(records, key="generalization_case", top_k=25)
    by_stratum = group_summary(records, key="stratum")
    failure_categories = Counter(record["failure_category"] for record in records)

    return {
        "overall": overall,
        "by_split": by_split,
        "by_generalization_case_top25": by_case,
        "by_stratum": by_stratum,
        "failure_categories": failure_categories.most_common(),
    }


def group_summary(records: list[dict[str, Any]], key: str, top_k: int | None = None) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)

    items = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    if top_k is not None:
        items = items[:top_k]
    return {group_key: aggregate_metrics(group_records) for group_key, group_records in items}


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "exact_match": 0.0,
            "normalized_exact_match": 0.0,
            "clause_exact_match": 0.0,
            "clause_f1": 0.0,
        }

    count = len(records)
    return {
        "count": count,
        "exact_match": round(sum(bool(record["exact_match"]) for record in records) / count, 6),
        "normalized_exact_match": round(
            sum(bool(record["normalized_exact_match"]) for record in records) / count, 6
        ),
        "clause_exact_match": round(
            sum(bool(record["clause_exact_match"]) for record in records) / count, 6
        ),
        "clause_f1": round(mean(float(record["clause_f1"]) for record in records), 6),
    }
