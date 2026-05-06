from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from discovery.metrics import cosine_similarity, mean_vector


class CentroidProbe:
    def __init__(self, namespace: str, layer_index: int, centroids: dict[str, list[float]]) -> None:
        self.namespace = namespace
        self.layer_index = layer_index
        self.centroids = centroids

    def predict(self, vector: list[float]) -> dict[str, Any]:
        scored = [
            (label, cosine_similarity(vector, centroid))
            for label, centroid in self.centroids.items()
        ]
        label, score = max(scored, key=lambda item: item[1])
        return {
            "label": label,
            "score": round(score, 6),
        }


def build_centroid_probes(span_records: list[dict[str, Any]], layer_index: int) -> dict[str, CentroidProbe]:
    grouped: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for record in span_records:
        if int(record["layer_index"]) != layer_index:
            continue
        grouped[str(record["namespace"])][str(record["label"])].append(list(record["vector"]))

    probes = {}
    for namespace, labels in grouped.items():
        centroids = {label: mean_vector(vectors) for label, vectors in labels.items()}
        probes[namespace] = CentroidProbe(namespace=namespace, layer_index=layer_index, centroids=centroids)
    return probes


def accuracy(records: list[dict[str, Any]], probe: CentroidProbe) -> float | None:
    if not records:
        return None
    correct = 0
    for record in records:
        prediction = probe.predict(list(record["vector"]))
        if prediction["label"] == str(record["label"]):
            correct += 1
    return round(correct / len(records), 6)


def label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record["label"]) for record in records).items()))
