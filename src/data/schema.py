from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CanonicalExample:
    item_id: str
    split: str
    raw_split: str
    benchmark_name: str
    input_text: str
    target_form: str
    generalization_case: str
    lexical_signature: tuple[str, ...] = field(default_factory=tuple)
    role_signature: dict[str, str] = field(default_factory=dict)
    surface_features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lexical_signature"] = list(self.lexical_signature)
        return payload


@dataclass(slots=True)
class SemanticQuartet:
    quartet_id: str
    benchmark_name: str
    predicate: str
    agent_a: str
    agent_b: str
    theme_a: str
    theme_b: str
    item_ids: tuple[str, str, str, str]
    split_pattern: tuple[str, str, str, str]
    role_scheme: str = "agent-theme"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["item_ids"] = list(self.item_ids)
        payload["split_pattern"] = list(self.split_pattern)
        return payload
