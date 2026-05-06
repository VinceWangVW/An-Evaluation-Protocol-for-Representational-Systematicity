from __future__ import annotations

from collections import Counter
from math import sqrt
from statistics import mean
from typing import Any


def dot(left: list[float], right: list[float]) -> float:
    return sum(left_item * right_item for left_item, right_item in zip(left, right))


def l2_norm(vector: list[float]) -> float:
    return sqrt(dot(vector, vector))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = l2_norm(left)
    right_norm = l2_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot(left, right) / (left_norm * right_norm)


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("Cannot compute mean vector for an empty list")
    width = len(vectors[0])
    totals = [0.0] * width
    for vector in vectors:
        if len(vector) != width:
            raise ValueError("Vector width mismatch")
        for idx, value in enumerate(vector):
            totals[idx] += value
    return [value / len(vectors) for value in totals]


def summarize_label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record["label"]) for record in records).items()))


def leave_one_out_centroid_accuracy(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = {str(record["label"]) for record in records}
    if len(records) < 4 or len(labels) < 2:
        return {
            "count": len(records),
            "label_counts": summarize_label_counts(records),
            "loo_centroid_accuracy": None,
        }

    correct = 0
    usable = 0
    for index, record in enumerate(records):
        remaining = records[:index] + records[index + 1 :]
        grouped: dict[str, list[list[float]]] = {}
        for other in remaining:
            grouped.setdefault(str(other["label"]), []).append(list(other["vector"]))

        if any(not vectors for vectors in grouped.values()) or len(grouped) < 2:
            continue

        centroids = {label: mean_vector(vectors) for label, vectors in grouped.items()}
        predicted = max(
            centroids.items(),
            key=lambda item: cosine_similarity(list(record["vector"]), item[1]),
        )[0]
        usable += 1
        if predicted == str(record["label"]):
            correct += 1

    accuracy = correct / usable if usable else None
    return {
        "count": len(records),
        "label_counts": summarize_label_counts(records),
        "loo_centroid_accuracy": round(accuracy, 6) if accuracy is not None else None,
    }


def identity_stability(records: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(str(record["label"]) for record in records)
    eligible = [record for record in records if label_counts[str(record["label"])] >= 2]
    if len(eligible) < 2:
        return {
            "count": len(records),
            "eligible_count": len(eligible),
            "label_counts": dict(sorted(label_counts.items())),
            "avg_within_cosine": None,
            "avg_between_cosine": None,
            "within_between_gap": None,
        }

    within: list[float] = []
    between: list[float] = []
    for left_index, left_record in enumerate(eligible):
        for right_record in eligible[left_index + 1 :]:
            similarity = cosine_similarity(list(left_record["vector"]), list(right_record["vector"]))
            if str(left_record["label"]) == str(right_record["label"]):
                within.append(similarity)
            else:
                between.append(similarity)

    within_mean = mean(within) if within else None
    between_mean = mean(between) if between else None
    gap = None
    if within_mean is not None and between_mean is not None:
        gap = within_mean - between_mean

    return {
        "count": len(records),
        "eligible_count": len(eligible),
        "label_counts": dict(sorted(label_counts.items())),
        "avg_within_cosine": round(within_mean, 6) if within_mean is not None else None,
        "avg_between_cosine": round(between_mean, 6) if between_mean is not None else None,
        "within_between_gap": round(gap, 6) if gap is not None else None,
    }
