from __future__ import annotations

from typing import Mapping


def assign_behavior_stratum(example: Mapping[str, object], metrics: Mapping[str, object]) -> str:
    split = str(example.get("split", ""))
    generalization_case = str(example.get("generalization_case", ""))

    if split in {"dev", "test"} or generalization_case == "in_distribution":
        return "control"
    is_systematic_pass = bool(metrics.get("clause_exact_match")) or bool(metrics.get("exact_match"))
    return "pass" if is_systematic_pass else "fail"
