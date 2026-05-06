from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonicalize import (
    infer_generalization_case,
    infer_lexical_signature,
    infer_role_signature,
    infer_surface_features,
    normalize_split_name,
    normalize_whitespace,
)
from .schema import CanonicalExample


@dataclass(slots=True)
class DatasetSpec:
    benchmark_name: str
    loader_type: str
    required: bool = True
    dataset_name: str | None = None
    dataset_config: str | None = None
    data_dir: str | None = None
    data_file: str | None = None
    split_files: dict[str, str] = field(default_factory=dict)
    has_header: bool = False
    columns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    payload = json.loads(Path(path).read_text())
    return DatasetSpec(**payload)


def load_benchmark(spec: DatasetSpec, project_root: str | Path) -> list[CanonicalExample]:
    project_root = Path(project_root)
    if spec.loader_type == "huggingface":
        return _load_huggingface_dataset(spec)
    if spec.loader_type == "local_tsv_dir":
        return _load_local_tsv_dir(spec, project_root)
    if spec.loader_type == "local_jsonl":
        return _load_local_jsonl(spec, project_root)
    raise ValueError(f"Unsupported loader_type: {spec.loader_type}")


def _load_huggingface_dataset(spec: DatasetSpec) -> list[CanonicalExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("datasets package is required for Hugging Face loaders") from exc

    if not spec.dataset_name:
        raise ValueError("dataset_name is required for Hugging Face loaders")

    dataset = load_dataset(spec.dataset_name, spec.dataset_config)
    examples: list[CanonicalExample] = []

    for raw_split, split_dataset in dataset.items():
        input_key = _select_column(
            split_dataset.column_names,
            preferred=("input_text", "source", "src", "sentence", "text", "question"),
        )
        target_key = _select_column(
            split_dataset.column_names,
            preferred=(
                "target_form",
                "target",
                "tgt",
                "logical_form",
                "parse",
                "output",
                "label",
            ),
            forbidden={input_key},
        )
        case_key = _select_column(
            split_dataset.column_names,
            preferred=(
                "generalization_case",
                "distribution",
                "distribution_type",
                "gen_type",
                "type",
            ),
            allow_missing=True,
            forbidden={input_key, target_key},
        )

        for row_index, row in enumerate(split_dataset):
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {input_key, target_key, case_key}
            }
            examples.append(
                _canonicalize_record(
                    benchmark_name=spec.benchmark_name,
                    raw_split=raw_split,
                    row_index=row_index,
                    input_text=str(row[input_key]),
                    target_form=str(row[target_key]),
                    generalization_case=(
                        str(row[case_key]) if case_key and row.get(case_key) is not None else None
                    ),
                    metadata=metadata,
                )
            )
    return examples


def _load_local_tsv_dir(spec: DatasetSpec, project_root: Path) -> list[CanonicalExample]:
    if not spec.data_dir:
        raise ValueError("data_dir is required for local_tsv_dir loaders")

    data_dir = project_root / spec.data_dir
    if not data_dir.exists():
        if spec.required:
            raise FileNotFoundError(f"Required data_dir not found: {data_dir}")
        return []

    columns = spec.columns or ["input_text", "target_form", "generalization_case"]
    examples: list[CanonicalExample] = []

    for raw_split, filename in spec.split_files.items():
        split_path = data_dir / filename
        if not split_path.exists():
            if spec.required:
                raise FileNotFoundError(f"Required split file not found: {split_path}")
            continue

        with split_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            if spec.has_header:
                header = next(reader, None)
                if header is None:
                    continue
                effective_columns = [normalize_whitespace(col) for col in header]
            else:
                effective_columns = columns

            for row_index, row in enumerate(reader):
                if not row:
                    continue
                record = {}
                for index, column_name in enumerate(effective_columns):
                    record[column_name] = row[index] if index < len(row) else ""

                metadata = {
                    key: value
                    for key, value in record.items()
                    if key not in {"input_text", "target_form", "generalization_case"}
                }
                examples.append(
                    _canonicalize_record(
                        benchmark_name=spec.benchmark_name,
                        raw_split=raw_split,
                        row_index=row_index,
                        input_text=str(record.get("input_text", "")),
                        target_form=str(record.get("target_form", "")),
                        generalization_case=str(record.get("generalization_case", "")),
                        metadata=metadata,
                    )
                )

    return examples


def _load_local_jsonl(spec: DatasetSpec, project_root: Path) -> list[CanonicalExample]:
    if not spec.data_file:
        raise ValueError("data_file is required for local_jsonl loaders")

    data_path = project_root / spec.data_file
    if not data_path.exists():
        if spec.required:
            raise FileNotFoundError(f"Required data_file not found: {data_path}")
        return []

    records = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    examples: list[CanonicalExample] = []
    for row_index, record in enumerate(records):
        metadata = dict(record.get("metadata", {}))
        if spec.metadata:
            metadata = {**spec.metadata, **metadata}
        examples.append(
            CanonicalExample(
                item_id=str(record.get("item_id", f"{spec.benchmark_name}:unknown:{row_index:06d}")),
                split=str(record.get("split", "train")),
                raw_split=str(record.get("raw_split", record.get("split", "train"))),
                benchmark_name=str(record.get("benchmark_name", spec.benchmark_name)),
                input_text=str(record.get("input_text", "")),
                target_form=str(record.get("target_form", "")),
                generalization_case=str(record.get("generalization_case", "unknown")),
                lexical_signature=tuple(record.get("lexical_signature", [])),
                role_signature=dict(record.get("role_signature", {})),
                surface_features=dict(record.get("surface_features", {})),
                metadata=metadata,
            )
        )
    return examples


def _canonicalize_record(
    benchmark_name: str,
    raw_split: str,
    row_index: int,
    input_text: str,
    target_form: str,
    generalization_case: str | None,
    metadata: dict[str, Any] | None = None,
) -> CanonicalExample:
    input_text = normalize_whitespace(input_text)
    target_form = normalize_whitespace(target_form)
    metadata = metadata or {}
    role_signature = infer_role_signature(input_text, target_form)

    return CanonicalExample(
        item_id=f"{benchmark_name}:{normalize_split_name(raw_split)}:{row_index:06d}",
        split=normalize_split_name(raw_split),
        raw_split=raw_split,
        benchmark_name=benchmark_name,
        input_text=input_text,
        target_form=target_form,
        generalization_case=infer_generalization_case(generalization_case, metadata),
        lexical_signature=infer_lexical_signature(input_text, role_signature),
        role_signature=role_signature,
        surface_features=infer_surface_features(input_text, target_form),
        metadata=metadata,
    )


def _select_column(
    columns: list[str],
    preferred: tuple[str, ...],
    allow_missing: bool = False,
    forbidden: set[str] | None = None,
) -> str | None:
    forbidden = forbidden or set()
    lowered = {column.lower(): column for column in columns}
    for candidate in preferred:
        if candidate.lower() in lowered and lowered[candidate.lower()] not in forbidden:
            return lowered[candidate.lower()]

    available = [column for column in columns if column not in forbidden]
    if not available and not allow_missing:
        raise ValueError(f"Could not find usable column in {columns}")
    if not available and allow_missing:
        return None
    if allow_missing:
        return None
    return available[0]
