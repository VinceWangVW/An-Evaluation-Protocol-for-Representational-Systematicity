from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import CanonicalExample, SemanticQuartet


def write_examples_jsonl(path: str | Path, examples: Iterable[CanonicalExample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=True) + "\n")


def write_quartets_jsonl(path: str | Path, quartets: Iterable[SemanticQuartet]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for quartet in quartets:
            handle.write(json.dumps(quartet.to_dict(), ensure_ascii=True) + "\n")


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
