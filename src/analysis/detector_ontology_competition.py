from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from data.systematicity_schema import SystematicitySchemaRecord, summarize_schema_records
from data.surface_templates import render_template_candidates
from data.surface_validation import (
    select_primary_candidates,
    summarize_validated_candidates,
    validate_surface_candidate,
)
from interventions.generation import (
    GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS,
    GEOMETRY_INTERACTION_OBJECT_CLASS,
    GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS,
)
from utils.io import write_json

from . import interaction_objects as p17
from . import model_native_detectability as p20
from . import argument_detector_validation as p21
from . import detector_truthfulness as p22
from .family_closure import ALIGNER_NAMES, _align_with_selected, load_json


D1_ARGUMENT_LOCUS = "argument_locus"
D2_PREDICATE_CONDITIONED_RELATION = "predicate_conditioned_relation"
D3_CROSS_STEP_BRIDGE = "cross_step_bridge"

DETECTOR_ORDER: tuple[str, ...] = (
    D1_ARGUMENT_LOCUS,
    D2_PREDICATE_CONDITIONED_RELATION,
    D3_CROSS_STEP_BRIDGE,
)

STAGE_A = "stage_a"
STAGE_B = "stage_b"

FAMILY_PREDICATE_SANITY = "predicate_sanity_h2"
FAMILY_DETECTOR_H2 = "detector_h2"
FAMILY_DETECTOR_H1 = "detector_h1"
FAMILY_DETECTOR_NULL = "detector_null"

FAMILY_ORDER: tuple[str, ...] = (
    FAMILY_PREDICATE_SANITY,
    FAMILY_DETECTOR_H2,
    FAMILY_DETECTOR_H1,
    FAMILY_DETECTOR_NULL,
)

STAGE_B_FAMILIES: tuple[str, ...] = (
    FAMILY_DETECTOR_H2,
    FAMILY_DETECTOR_H1,
    FAMILY_DETECTOR_NULL,
)

FAMILY_THRESHOLDS = {
    FAMILY_PREDICATE_SANITY: 0.9,
    FAMILY_DETECTOR_H2: 0.85,
    FAMILY_DETECTOR_H1: 0.55,
}

DEFAULT_COUNTS = {
    "train": 8,
    "dev": 2,
    "gen": 2,
}

BASE_FAMILY_BY_DETECTOR_FAMILY = {
    FAMILY_PREDICATE_SANITY: p22.FAMILY_PREDICATE_SANITY,
    FAMILY_DETECTOR_H2: p22.FAMILY_ARGUMENT_H2,
    FAMILY_DETECTOR_H1: p22.FAMILY_ARGUMENT_H1,
    FAMILY_DETECTOR_NULL: p22.FAMILY_ARGUMENT_NULL,
}


def _detector_registry() -> dict[str, dict[str, Any]]:
    return {
        D1_ARGUMENT_LOCUS: {
            "label": "D1",
            "case_prefix": "d1",
            "object_class": GEOMETRY_INTERACTION_OBJECT_CLASS,
            "benchmark_name": "detector_competition_d1",
            "panel_name": "detector_competition_d1",
            "intended_observable": "argument_locus",
            "source_step_key_seed_map": {1: "agent_value", 2: "theme_value"},
            "target_step_key_seed_map": {1: "agent_value", 2: "theme_value"},
        },
        D2_PREDICATE_CONDITIONED_RELATION: {
            "label": "D2",
            "case_prefix": "d2",
            "object_class": GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS,
            "benchmark_name": "detector_competition_d2",
            "panel_name": "detector_competition_d2",
            "intended_observable": "predicate_value",
            "source_step_key_seed_map": {1: "predicate_value", 2: "predicate_value"},
            "target_step_key_seed_map": {1: "predicate_value", 2: "predicate_value"},
        },
        D3_CROSS_STEP_BRIDGE: {
            "label": "D3",
            "case_prefix": "d3",
            "object_class": GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS,
            "benchmark_name": "detector_competition_d3",
            "panel_name": "detector_competition_d3",
            "intended_observable": "bridge",
            "source_step_key_seed_map": {1: "predicate_value", 2: "predicate_value"},
            "target_step_key_seed_map": {1: "agent_value", 2: "theme_value"},
        },
    }


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.setdefault("project_root", str(config_path.parents[2]))
    return payload


def detector_registry() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in _detector_registry().items()}


def _benchmark_name(detector_family: str) -> str:
    return str(_detector_registry()[detector_family]["benchmark_name"])


def _panel_name(detector_family: str) -> str:
    return str(_detector_registry()[detector_family]["panel_name"])


def _detector_family_case(detector_family: str, family_name: str) -> str:
    return f"{_detector_registry()[detector_family]['case_prefix']}_{family_name}"


def _family_record_count(config: dict[str, Any], split_name: str) -> int:
    counts = dict(config.get("family_counts") or {})
    if split_name in counts:
        return int(counts[split_name])
    return int(DEFAULT_COUNTS[split_name])


def _source_step_key(detector_family: str, family_name: str, seed: int) -> str:
    if family_name == FAMILY_PREDICATE_SANITY:
        return "predicate_value"
    return str(_detector_registry()[detector_family]["source_step_key_seed_map"][seed])


def _target_step_key(detector_family: str, family_name: str, seed: int) -> str:
    if family_name == FAMILY_PREDICATE_SANITY:
        return "predicate_value"
    return str(_detector_registry()[detector_family]["target_step_key_seed_map"][seed])


def _intended_observable(detector_family: str, family_name: str, seed: int) -> str:
    if family_name == FAMILY_PREDICATE_SANITY:
        return "predicate_value"
    if detector_family == D3_CROSS_STEP_BRIDGE:
        return f"{_source_step_key(detector_family, family_name, seed)}->{_target_step_key(detector_family, family_name, seed)}"
    return str(_target_step_key(detector_family, family_name, seed))


def _stage_family_name(family_name: str) -> str:
    return BASE_FAMILY_BY_DETECTOR_FAMILY[family_name]


def _record_metadata(
    *,
    detector_family: str,
    family_name: str,
) -> dict[str, Any]:
    return {
        "truthfulness_benchmark": True,
        "detector_competition_benchmark": True,
        "detector_family": detector_family,
        "detector_label": str(_detector_registry()[detector_family]["label"]),
        "truthfulness_family": family_name,
        "truthfulness_family_case": _detector_family_case(detector_family, family_name),
    }


def _retag_records(
    *,
    base_records: list[SystematicitySchemaRecord],
    detector_family: str,
    family_name: str,
    benchmark_name: str,
) -> list[SystematicitySchemaRecord]:
    family_case = _detector_family_case(detector_family, family_name)
    records: list[SystematicitySchemaRecord] = []
    for base in base_records:
        metadata = dict(base.metadata)
        metadata.update(_record_metadata(detector_family=detector_family, family_name=family_name))
        records.append(
            SystematicitySchemaRecord(
                item_id=base.item_id,
                benchmark_name=benchmark_name,
                split=base.split,
                raw_split=base.raw_split,
                schema_id=base.schema_id,
                counterpart_family_id=base.counterpart_family_id,
                counterpart_type=base.counterpart_type,
                construction_family=family_case,
                voice=base.voice,
                embedding_depth=base.embedding_depth,
                predicate_lemma=base.predicate_lemma,
                predicate_past=base.predicate_past,
                predicate_participle=base.predicate_participle,
                context_predicate_lemma=base.context_predicate_lemma,
                context_predicate_past=base.context_predicate_past,
                agent=base.agent,
                theme=base.theme,
                agent_modifier=base.agent_modifier,
                theme_modifier=base.theme_modifier,
                context_filler=base.context_filler,
                target_form=base.target_form,
                metadata=metadata,
            )
        )
    return records


def _build_detector_family_records(
    *,
    detector_family: str,
    family_name: str,
    family_index: int,
    split_name: str,
    rng: random.Random,
) -> list[SystematicitySchemaRecord]:
    registry_row = _detector_registry()[detector_family]
    benchmark_name = str(registry_row["benchmark_name"])
    object_class = str(registry_row["object_class"])
    base_family_name = _stage_family_name(family_name)
    base_records = p22._build_truthfulness_family_records(
        benchmark_name=benchmark_name,
        object_class=object_class,
        family_name=base_family_name,
        family_index=family_index,
        split_name=split_name,
        rng=rng,
    )
    return _retag_records(
        base_records=base_records,
        detector_family=detector_family,
        family_name=family_name,
        benchmark_name=benchmark_name,
    )


def build_detector_competition_benchmarks(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["project_root"])
    processed_dir = root / "data/processed"
    raw_dir = root / "data/raw/phase11"
    derived_dir = root / "data/derived"
    reports_dir = root / "reports/detector_competition"
    benchmark_rows: list[dict[str, Any]] = []
    rng = random.Random(int(config.get("seed") or 23))
    for detector_family in DETECTOR_ORDER:
        benchmark_name = _benchmark_name(detector_family)
        object_class = str(_detector_registry()[detector_family]["object_class"])
        schema_records: list[SystematicitySchemaRecord] = []
        for family_name in FAMILY_ORDER:
            for split_name in ("train", "dev", "gen"):
                for family_index in range(_family_record_count(config, split_name)):
                    schema_records.extend(
                        _build_detector_family_records(
                            detector_family=detector_family,
                            family_name=family_name,
                            family_index=family_index,
                            split_name=split_name,
                            rng=rng,
                        )
                    )
        candidates = [
            validate_surface_candidate(record, render_template_candidates(record, num_candidates=1)[0])
            for record in schema_records
        ]
        selected_candidates, canonical_examples, selection_summary = select_primary_candidates(
            schema_records=schema_records,
            validated_candidates=candidates,
            selection_strategy="shortest",
        )
        for example in canonical_examples:
            example.metadata["truthfulness_benchmark"] = True
            example.metadata["detector_competition_benchmark"] = True
            example.metadata["detector_family"] = detector_family
        processed_path = processed_dir / f"{benchmark_name}.jsonl"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        with processed_path.open("w", encoding="utf-8") as handle:
            for row in canonical_examples:
                handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        raw_dir.mkdir(parents=True, exist_ok=True)
        derived_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        with (raw_dir / f"{benchmark_name}_schema.jsonl").open("w", encoding="utf-8") as handle:
            for row in schema_records:
                handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        with (derived_dir / f"{benchmark_name}_selected_candidates.jsonl").open("w", encoding="utf-8") as handle:
            for row in selected_candidates:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        benchmark_summary = summarize_schema_records(schema_records)
        benchmark_summary.update(
            {
                "detector_family": detector_family,
                "object_class": object_class,
                "benchmark_name": benchmark_name,
                "canonical_example_count": len(canonical_examples),
                "selection": selection_summary,
                "candidate_validation": summarize_validated_candidates(candidates),
                "processed_path": str(processed_path.relative_to(root)),
            }
        )
        (reports_dir / f"{benchmark_name}_summary.json").write_text(
            json.dumps(benchmark_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        benchmark_rows.append(benchmark_summary)
    return {
        "benchmark_count": len(benchmark_rows),
        "benchmark_rows": benchmark_rows,
    }


def _detector_anchor_specs(detector_family: str) -> list[dict[str, Any]]:
    if detector_family == D1_ARGUMENT_LOCUS:
        base = list(p20.get_stage20_synthetic_anchor_specs(object_class=GEOMETRY_INTERACTION_OBJECT_CLASS))
        agent_anchor = next(dict(row) for row in base if str(row.get("decoder_step_key")) == "agent_value")
        theme_anchor = dict(agent_anchor)
        agent_anchor["object_id"] = "d1_interaction_agent_anchor_a"
        agent_anchor["template_family"] = "anchor_a"
        theme_anchor["object_id"] = "d1_interaction_theme_anchor_b"
        theme_anchor["decoder_step_key"] = "theme_value"
        theme_anchor["source_span_keys"] = ["theme", "filler::theme"]
        theme_anchor["template_family"] = "anchor_b"
        return [agent_anchor, theme_anchor]
    if detector_family == D2_PREDICATE_CONDITIONED_RELATION:
        base = list(p20.get_stage20_synthetic_anchor_specs(object_class=GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS))
        predicate_anchor = next(dict(row) for row in base if str(row.get("query_step_key")) == "predicate_value")
        support_anchor = dict(predicate_anchor)
        predicate_anchor["object_id"] = "d2_predicate_relation_anchor_a"
        predicate_anchor["template_family"] = "anchor_a"
        support_anchor["object_id"] = "d2_predicate_relation_anchor_b"
        support_anchor["head_selector"] = {"mode": "explicit", "indices": [0, 1]}
        support_anchor["query_basis"] = {"mode": "explicit", "basis_ids": ["prelogit_basis_0", "prelogit_basis_1"]}
        support_anchor["read_basis"] = {"mode": "explicit", "basis_ids": ["prelogit_basis_1", "prelogit_basis_2"]}
        support_anchor["store_condition"] = {"mode": "matched_role_store", "selector_type": "semantic_role_bundle"}
        support_anchor["template_family"] = "anchor_b"
        return [predicate_anchor, support_anchor]
    anchors = [dict(row) for row in p20.get_stage20_synthetic_anchor_specs(object_class=GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS)]
    anchors[0]["object_id"] = "d3_bridge_predicate_to_agent_anchor_a"
    anchors[0]["template_family"] = "anchor_a"
    anchors[1]["object_id"] = "d3_bridge_predicate_to_theme_anchor_b"
    anchors[1]["template_family"] = "anchor_b"
    return anchors


def _panel_target(
    *,
    detector_family: str,
    family_name: str,
    seed: int,
    phase11_prefix: str,
    paired_null_id: str | None,
    recipe: dict[str, Any],
) -> dict[str, Any]:
    detector_row = _detector_registry()[detector_family]
    source_step_key = _source_step_key(detector_family, family_name, seed)
    target_step_key = _target_step_key(detector_family, family_name, seed)
    return {
        "target_id": phase11_prefix,
        "run_name": phase11_prefix,
        "phase11_prefix": phase11_prefix,
        "benchmark": str(detector_row["benchmark_name"]),
        "panel_regime": family_name,
        "truthfulness_class": str(detector_row["object_class"]),
        "truthfulness_family": _detector_family_case(detector_family, family_name),
        "truthfulness_seed": seed,
        "detector_family": detector_family,
        "detector_seed": seed,
        "intended_observable": _intended_observable(detector_family, family_name, seed),
        "intended_step_key": target_step_key,
        "source_step_key": source_step_key,
        "target_step_key": target_step_key,
        "paired_null_id": paired_null_id,
        "train_generalization_case_allowlist": [_detector_family_case(detector_family, family_name)],
        "num_train_epochs": 2,
        "train_batch_size": 4,
        "eval_batch_size": 4,
        "learning_rate": 5e-5,
        "seed": seed,
        **recipe,
    }


def _stage_a_recipe(detector_family: str, family_name: str) -> dict[str, Any]:
    if detector_family == D1_ARGUMENT_LOCUS:
        if family_name == FAMILY_PREDICATE_SANITY:
            return {
                "decoder_cross_attention_scale": 1.5,
                "decoder_readout_subspace_keep_ratio": 0.35,
                "decoder_readout_subspace_mode": "keep",
                "decoder_readout_subspace_seed": 22,
            }
        if family_name == FAMILY_DETECTOR_H2:
            return {
                "decoder_cross_attention_scale": 1.75,
                "decoder_readout_subspace_keep_ratio": 0.45,
                "decoder_readout_subspace_mode": "keep",
                "decoder_readout_subspace_seed": 23,
            }
        if family_name == FAMILY_DETECTOR_H1:
            return {
                "decoder_cross_attention_scale": 1.25,
                "decoder_readout_subspace_keep_ratio": 0.25,
                "decoder_readout_subspace_mode": "keep",
                "decoder_readout_subspace_seed": 24,
                "decoder_cross_attention_relation_barrier_keep_ratio": 0.25,
                "decoder_cross_attention_relation_barrier_scale": 0.5,
            }
        return {
            "decoder_cross_attention_scale": 1.25,
            "decoder_readout_keep_ratio": 0.35,
        }
    if detector_family == D2_PREDICATE_CONDITIONED_RELATION:
        if family_name == FAMILY_PREDICATE_SANITY:
            return {
                "decoder_readout_subspace_keep_ratio": 0.35,
                "decoder_readout_subspace_mode": "keep",
                "decoder_readout_subspace_seed": 22,
                "decoder_cross_attention_relation_barrier_keep_ratio": 0.2,
                "decoder_cross_attention_relation_barrier_scale": 0.5,
            }
        if family_name == FAMILY_DETECTOR_H2:
            return {
                "decoder_cross_attention_relation_barrier_keep_ratio": 0.35,
                "decoder_cross_attention_relation_barrier_scale": 1.75,
                "decoder_readout_subspace_keep_ratio": 0.45,
                "decoder_readout_subspace_mode": "keep",
                "decoder_readout_subspace_seed": 23,
            }
        if family_name == FAMILY_DETECTOR_H1:
            return {
                "decoder_cross_attention_relation_barrier_keep_ratio": 0.35,
                "decoder_cross_attention_relation_barrier_scale": 1.25,
                "decoder_readout_subspace_keep_ratio": 0.25,
                "decoder_readout_subspace_mode": "keep",
                "decoder_readout_subspace_seed": 24,
            }
        return {
            "decoder_cross_attention_scale": 1.25,
            "decoder_readout_keep_ratio": 0.35,
        }
    if family_name == FAMILY_PREDICATE_SANITY:
        return {
            "decoder_cross_attention_scale": 1.5,
            "decoder_cross_attention_layer_keep_ratio": 0.5,
            "decoder_cross_attention_layer_policy": "alternating",
            "decoder_cross_attention_layer_seed": 22,
        }
    if family_name == FAMILY_DETECTOR_H2:
        return {
            "decoder_cross_attention_scale": 1.75,
            "decoder_cross_attention_layer_keep_ratio": 0.5,
            "decoder_cross_attention_layer_policy": "alternating",
            "decoder_cross_attention_layer_seed": 23,
        }
    if family_name == FAMILY_DETECTOR_H1:
        return {
            "decoder_cross_attention_scale": 1.25,
            "decoder_cross_attention_layer_keep_ratio": 0.5,
            "decoder_cross_attention_layer_policy": "alternating",
            "decoder_cross_attention_layer_seed": 24,
            "decoder_cross_attention_relation_barrier_keep_ratio": 0.35,
            "decoder_cross_attention_relation_barrier_scale": 0.5,
        }
    return {
        "decoder_cross_attention_scale": 1.25,
        "decoder_cross_attention_layer_keep_ratio": 0.5,
        "decoder_cross_attention_layer_policy": "top",
        "decoder_cross_attention_layer_seed": 25,
    }


def _stage_b_recipe(detector_family: str, family_name: str) -> dict[str, Any]:
    if detector_family == D1_ARGUMENT_LOCUS:
        if family_name == FAMILY_DETECTOR_H2:
            return {
                "decoder_cross_attention_scale": 2.0,
                "decoder_readout_subspace_keep_ratio": 0.55,
                "decoder_readout_subspace_mode": "keep",
                "decoder_readout_subspace_seed": 32,
            }
        if family_name == FAMILY_DETECTOR_H1:
            return {
                "decoder_cross_attention_scale": 1.5,
                "decoder_readout_subspace_keep_ratio": 0.35,
                "decoder_readout_subspace_mode": "keep",
                "decoder_readout_subspace_seed": 33,
                "decoder_cross_attention_relation_barrier_keep_ratio": 0.3,
                "decoder_cross_attention_relation_barrier_scale": 0.65,
            }
        return {
            "decoder_cross_attention_scale": 1.5,
            "decoder_readout_keep_ratio": 0.45,
        }
    if detector_family == D2_PREDICATE_CONDITIONED_RELATION:
        if family_name == FAMILY_DETECTOR_H2:
            return {
                "decoder_cross_attention_relation_barrier_keep_ratio": 0.5,
                "decoder_cross_attention_relation_barrier_scale": 2.0,
                "decoder_readout_subspace_keep_ratio": 0.55,
                "decoder_readout_subspace_mode": "keep",
                "decoder_readout_subspace_seed": 32,
            }
        if family_name == FAMILY_DETECTOR_H1:
            return {
                "decoder_cross_attention_relation_barrier_keep_ratio": 0.5,
                "decoder_cross_attention_relation_barrier_scale": 1.5,
                "decoder_readout_subspace_keep_ratio": 0.35,
                "decoder_readout_subspace_mode": "keep",
                "decoder_readout_subspace_seed": 33,
            }
        return {
            "decoder_cross_attention_scale": 1.5,
            "decoder_readout_keep_ratio": 0.45,
        }
    if family_name == FAMILY_DETECTOR_H2:
        return {
            "decoder_cross_attention_scale": 2.0,
            "decoder_cross_attention_layer_keep_ratio": 0.65,
            "decoder_cross_attention_layer_policy": "alternating",
            "decoder_cross_attention_layer_seed": 32,
        }
    if family_name == FAMILY_DETECTOR_H1:
        return {
            "decoder_cross_attention_scale": 1.5,
            "decoder_cross_attention_layer_keep_ratio": 0.65,
            "decoder_cross_attention_layer_policy": "alternating",
            "decoder_cross_attention_layer_seed": 33,
            "decoder_cross_attention_relation_barrier_keep_ratio": 0.3,
            "decoder_cross_attention_relation_barrier_scale": 0.65,
        }
    return {
        "decoder_cross_attention_scale": 1.5,
        "decoder_cross_attention_layer_keep_ratio": 0.65,
        "decoder_cross_attention_layer_policy": "top",
        "decoder_cross_attention_layer_seed": 34,
    }


def build_detector_competition_panels(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["project_root"])
    configs_dir = root / "configs/experiments"
    panel_rows: list[dict[str, Any]] = []
    for detector_family in DETECTOR_ORDER:
        panel_name = _panel_name(detector_family)
        benchmark_name = _benchmark_name(detector_family)
        stage_a_targets: list[dict[str, Any]] = []
        stage_b_targets: list[dict[str, Any]] = []
        for seed in (1, 2):
            stage_a_targets.append(
                _panel_target(
                    detector_family=detector_family,
                    family_name=FAMILY_PREDICATE_SANITY,
                    seed=seed,
                    phase11_prefix=f"{panel_name}_{FAMILY_PREDICATE_SANITY}_seed{seed}",
                    paired_null_id=None,
                    recipe=_stage_a_recipe(detector_family, FAMILY_PREDICATE_SANITY),
                )
            )
            paired_null = f"{panel_name}_{FAMILY_DETECTOR_NULL}_seed{seed}"
            stage_a_targets.extend(
                [
                    _panel_target(
                        detector_family=detector_family,
                        family_name=FAMILY_DETECTOR_H2,
                        seed=seed,
                        phase11_prefix=f"{panel_name}_{FAMILY_DETECTOR_H2}_seed{seed}",
                        paired_null_id=paired_null,
                        recipe=_stage_a_recipe(detector_family, FAMILY_DETECTOR_H2),
                    ),
                    _panel_target(
                        detector_family=detector_family,
                        family_name=FAMILY_DETECTOR_H1,
                        seed=seed,
                        phase11_prefix=f"{panel_name}_{FAMILY_DETECTOR_H1}_seed{seed}",
                        paired_null_id=paired_null,
                        recipe=_stage_a_recipe(detector_family, FAMILY_DETECTOR_H1),
                    ),
                    _panel_target(
                        detector_family=detector_family,
                        family_name=FAMILY_DETECTOR_NULL,
                        seed=seed,
                        phase11_prefix=paired_null,
                        paired_null_id=None,
                        recipe=_stage_a_recipe(detector_family, FAMILY_DETECTOR_NULL),
                    ),
                ]
            )
            stage_b_targets.extend(
                [
                    _panel_target(
                        detector_family=detector_family,
                        family_name=FAMILY_DETECTOR_H2,
                        seed=seed,
                        phase11_prefix=f"{panel_name}_{FAMILY_DETECTOR_H2}_stageb_seed{seed}",
                        paired_null_id=f"{panel_name}_{FAMILY_DETECTOR_NULL}_stageb_seed{seed}",
                        recipe=_stage_b_recipe(detector_family, FAMILY_DETECTOR_H2),
                    ),
                    _panel_target(
                        detector_family=detector_family,
                        family_name=FAMILY_DETECTOR_H1,
                        seed=seed,
                        phase11_prefix=f"{panel_name}_{FAMILY_DETECTOR_H1}_stageb_seed{seed}",
                        paired_null_id=f"{panel_name}_{FAMILY_DETECTOR_NULL}_stageb_seed{seed}",
                        recipe=_stage_b_recipe(detector_family, FAMILY_DETECTOR_H1),
                    ),
                    _panel_target(
                        detector_family=detector_family,
                        family_name=FAMILY_DETECTOR_NULL,
                        seed=seed,
                        phase11_prefix=f"{panel_name}_{FAMILY_DETECTOR_NULL}_stageb_seed{seed}",
                        paired_null_id=None,
                        recipe=_stage_b_recipe(detector_family, FAMILY_DETECTOR_NULL),
                    ),
                ]
            )
        stage_a_panel = {
            "panel_name": panel_name,
            "benchmark": benchmark_name,
            "model_name_or_path": "google-t5/t5-small",
            "prompt_style": "parser_instruction",
            "analysis_family": "seq2seq_lm",
            "metadata": {
                "phase": "phase23",
                "stage": STAGE_A,
                "detector_family": detector_family,
                "object_class": str(_detector_registry()[detector_family]["object_class"]),
                "operationalization_family": "geometry",
            },
            "targets": stage_a_targets,
        }
        stage_b_panel = {
            "panel_name": f"{panel_name}_{STAGE_B}",
            "benchmark": benchmark_name,
            "model_name_or_path": "google-t5/t5-small",
            "prompt_style": "parser_instruction",
            "analysis_family": "seq2seq_lm",
            "metadata": {
                "phase": "phase23",
                "stage": STAGE_B,
                "detector_family": detector_family,
                "object_class": str(_detector_registry()[detector_family]["object_class"]),
                "operationalization_family": "geometry",
            },
            "targets": stage_b_targets,
        }
        stage_a_path = configs_dir / f"{panel_name}.json"
        stage_b_path = configs_dir / f"{panel_name}_{STAGE_B}.json"
        stage_a_path.parent.mkdir(parents=True, exist_ok=True)
        stage_a_path.write_text(json.dumps(stage_a_panel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stage_b_path.write_text(json.dumps(stage_b_panel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        panel_rows.append(
            {
                "detector_family": detector_family,
                "stage_a_panel_path": str(stage_a_path.relative_to(root)),
                "stage_b_panel_path": str(stage_b_path.relative_to(root)),
                "stage_a_target_count": len(stage_a_targets),
                "stage_b_target_count": len(stage_b_targets),
            }
        )
    return {"panel_rows": panel_rows, "panel_count": len(panel_rows)}


def build_detector_competition_manifest(*, root: Path, panel_paths: dict[str, dict[str, str]]) -> dict[str, Any]:
    detector_rows: list[dict[str, Any]] = []
    for detector_family in DETECTOR_ORDER:
        stage_rows: dict[str, Any] = {}
        for stage_name in (STAGE_A, STAGE_B):
            panel_path = root / str(panel_paths[detector_family][stage_name])
            panel = json.loads(panel_path.read_text(encoding="utf-8"))
            stage_rows[stage_name] = {
                "panel_path": str(panel_path.relative_to(root)),
                "target_rows": [
                    {
                        "target_id": str(target["target_id"]),
                        "run_name": str(target.get("run_name") or target["target_id"]),
                        "phase11_prefix": str(target.get("phase11_prefix") or target.get("run_name") or target["target_id"]),
                        "panel_regime": str(target["panel_regime"]),
                        "truthfulness_family": str(target["truthfulness_family"]),
                        "truthfulness_seed": int(target["truthfulness_seed"]),
                        "detector_family": str(target["detector_family"]),
                        "detector_seed": int(target["detector_seed"]),
                        "intended_observable": str(target["intended_observable"]),
                        "intended_step_key": str(target["intended_step_key"]),
                        "source_step_key": str(target["source_step_key"]),
                        "target_step_key": str(target["target_step_key"]),
                        "paired_null_id": target.get("paired_null_id"),
                    }
                    for target in list(panel.get("targets") or [])
                ],
            }
        detector_rows.append(
            {
                "detector_family": detector_family,
                "object_class": str(_detector_registry()[detector_family]["object_class"]),
                "benchmark_name": _benchmark_name(detector_family),
                "anchor_specs": _detector_anchor_specs(detector_family),
                "stage_rows": stage_rows,
            }
        )
    return {"detector_count": len(detector_rows), "detector_rows": detector_rows}


def summarize_anchorability(manifest: dict[str, Any]) -> dict[str, Any]:
    detector_rows = []
    for row in list(manifest.get("detector_rows") or []):
        usable = len(list(row.get("anchor_specs") or []))
        detector_rows.append(
            {
                "detector_family": str(row["detector_family"]),
                "object_class": str(row["object_class"]),
                "passed": usable >= 2,
                "usable_anchor_count": usable,
                "anchor_ids": [str(anchor["object_id"]) for anchor in list(row.get("anchor_specs") or [])],
            }
        )
    return {"detector_rows": detector_rows, "passed_count": sum(1 for row in detector_rows if bool(row["passed"]))}


def _evaluation_target_rows(*, detector_row: dict[str, Any], stage_name: str) -> list[dict[str, Any]]:
    if stage_name == STAGE_A:
        return list(detector_row["stage_rows"][STAGE_A]["target_rows"])
    stage_a_rows = list(detector_row["stage_rows"][STAGE_A]["target_rows"])
    stage_b_rows = list(detector_row["stage_rows"][stage_name]["target_rows"])
    predicate_rows = [row for row in stage_a_rows if str(row["panel_regime"]) == FAMILY_PREDICATE_SANITY]
    detector_rows = [row for row in stage_b_rows if str(row["panel_regime"]) != FAMILY_PREDICATE_SANITY]
    return predicate_rows + detector_rows


def summarize_family_validity(
    *,
    root: Path,
    manifest: dict[str, Any],
    active_stage_by_detector: dict[str, str],
) -> dict[str, Any]:
    detector_rows: list[dict[str, Any]] = []
    for detector_row in list(manifest.get("detector_rows") or []):
        detector_family = str(detector_row["detector_family"])
        stage_name = active_stage_by_detector.get(detector_family, STAGE_A)
        target_rows = _evaluation_target_rows(detector_row=detector_row, stage_name=stage_name)
        family_rows: list[dict[str, Any]] = []
        metrics_by_family: dict[tuple[str, int], dict[str, Any]] = {}
        for target in target_rows:
            predictions = p22._prediction_records(root, str(target["run_name"]))
            family_case = str(target["truthfulness_family"])
            subset = p22._filter_records_for_family(predictions, family_case)
            metric = p22._family_metric(subset)
            metric.update(
                {
                    "panel_regime": str(target["panel_regime"]),
                    "run_name": str(target["run_name"]),
                    "truthfulness_seed": int(target["truthfulness_seed"]),
                    "truthfulness_family": family_case,
                    "detector_family": detector_family,
                }
            )
            metrics_by_family[(str(target["panel_regime"]), int(target["truthfulness_seed"]))] = metric
            family_rows.append(metric)
        passed = True
        failure_reasons: list[str] = []
        for seed in (1, 2):
            pred = metrics_by_family[(FAMILY_PREDICATE_SANITY, seed)]
            if max(pred["exact_match"], pred["clause_f1"]) < FAMILY_THRESHOLDS[FAMILY_PREDICATE_SANITY]:
                passed = False
                failure_reasons.append(f"{FAMILY_PREDICATE_SANITY}:seed{seed}")
            h2 = metrics_by_family[(FAMILY_DETECTOR_H2, seed)]
            if max(h2["exact_match"], h2["clause_f1"]) < FAMILY_THRESHOLDS[FAMILY_DETECTOR_H2]:
                passed = False
                failure_reasons.append(f"{FAMILY_DETECTOR_H2}:seed{seed}")
            h1 = metrics_by_family[(FAMILY_DETECTOR_H1, seed)]
            if max(h1["exact_match"], h1["clause_f1"]) < FAMILY_THRESHOLDS[FAMILY_DETECTOR_H1]:
                passed = False
                failure_reasons.append(f"{FAMILY_DETECTOR_H1}:seed{seed}")
            if h1["clause_f1"] >= h2["clause_f1"]:
                passed = False
                failure_reasons.append(f"{FAMILY_DETECTOR_H1}:not_weaker_than_h2:seed{seed}")
            null_row = metrics_by_family[(FAMILY_DETECTOR_NULL, seed)]
            if null_row["clause_f1"] > max(0.4, h1["clause_f1"] - 0.1):
                passed = False
                failure_reasons.append(f"{FAMILY_DETECTOR_NULL}:too_strong:seed{seed}")
        detector_rows.append(
            {
                "detector_family": detector_family,
                "object_class": str(detector_row["object_class"]),
                "stage": stage_name,
                "passed": passed,
                "failure_reasons": failure_reasons,
                "family_rows": family_rows,
            }
        )
    return {"detector_rows": detector_rows, "passed_count": sum(1 for row in detector_rows if bool(row["passed"]))}


def _bundle_from_prefix(root: Path, phase11_prefix: str) -> dict[str, Any]:
    return p17._load_protocol_bundle(root, root / "reports/protocol_runs" / f"{phase11_prefix}_summary.json")


def _alignment_rows(
    *,
    bundle: dict[str, Any],
    aligner_name: str,
    max_length: int,
) -> list[dict[str, Any]]:
    adapter, layer_index = p17.build_adapter_from_bundle(bundle)
    span_index = p17._index_span_records(bundle["span_records"], layer_index=layer_index)
    live_span_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for item in bundle["candidate_records"]:
        item_span_map = p17._ensure_item_span_map(
            adapter=adapter,
            item=item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        forward = p17._teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
        )
        alignments = _align_with_selected(
            aligner_name=aligner_name,
            tokenizer=adapter.tokenizer,
            target_text=str(item["target_form"]),
            record=item,
            forward=forward,
            item_span_map=item_span_map,
            layer_index=layer_index,
        )
        rows.append(
            {
                "item_id": str(item["item_id"]),
                "aligned_steps": sorted(str(key) for key in dict(alignments.get("alignments") or {}).keys()),
            }
        )
    return rows


def _score_step_observability(rows: list[dict[str, Any]], *, intended_step_key: str) -> dict[str, Any]:
    intended_rate = round(
        mean(int(intended_step_key in set(row["aligned_steps"])) for row in rows),
        6,
    ) if rows else 0.0
    unintended_rate = round(
        mean(int(any(step != intended_step_key for step in set(row["aligned_steps"]))) for row in rows),
        6,
    ) if rows else 1.0
    return {
        "row_count": len(rows),
        "intended_step_alignment_rate": intended_rate,
        "unintended_step_alignment_rate": unintended_rate,
        "passed": intended_rate >= 0.8 and unintended_rate <= 0.2,
    }


def _score_bridge_observability(
    rows: list[dict[str, Any]],
    *,
    source_step_key: str,
    target_step_key: str,
) -> dict[str, Any]:
    source_rate = round(mean(int(source_step_key in set(row["aligned_steps"])) for row in rows), 6) if rows else 0.0
    target_rate = round(mean(int(target_step_key in set(row["aligned_steps"])) for row in rows), 6) if rows else 0.0
    coalignment_rate = (
        round(
            mean(
                int(
                    source_step_key in set(row["aligned_steps"]) and target_step_key in set(row["aligned_steps"])
                )
                for row in rows
            ),
            6,
        )
        if rows
        else 0.0
    )
    unintended_rate = round(
        mean(
            int(any(step not in {source_step_key, target_step_key} for step in set(row["aligned_steps"])))
            for row in rows
        ),
        6,
    ) if rows else 1.0
    return {
        "row_count": len(rows),
        "source_step_alignment_rate": source_rate,
        "target_step_alignment_rate": target_rate,
        "bridge_coalignment_rate": coalignment_rate,
        "unintended_extra_step_alignment_rate": unintended_rate,
        "passed": source_rate >= 0.8 and target_rate >= 0.8 and coalignment_rate >= 0.8 and unintended_rate <= 0.2,
    }


def _evaluate_target_observability(
    *,
    detector_family: str,
    panel_regime: str,
    rows: list[dict[str, Any]],
    source_step_key: str,
    target_step_key: str,
) -> dict[str, Any]:
    if panel_regime == FAMILY_PREDICATE_SANITY:
        summary = _score_step_observability(rows, intended_step_key="predicate_value")
        summary["observable_locus"] = "predicate_value"
        summary["observability_mode"] = "predicate_sanity"
        return summary
    if detector_family == D1_ARGUMENT_LOCUS:
        summary = _score_step_observability(rows, intended_step_key=target_step_key)
        summary["observable_locus"] = target_step_key
        summary["observability_mode"] = "step_local"
        return summary
    if detector_family == D2_PREDICATE_CONDITIONED_RELATION:
        summary = _score_step_observability(rows, intended_step_key="predicate_value")
        summary["observable_locus"] = "predicate_value"
        summary["observability_mode"] = "predicate_conditioned"
        summary["predicate_value_alignment_rate"] = float(summary["intended_step_alignment_rate"])
        return summary
    summary = _score_bridge_observability(rows, source_step_key=source_step_key, target_step_key=target_step_key)
    summary["observable_locus"] = f"{source_step_key}->{target_step_key}"
    summary["observability_mode"] = "bridge"
    return summary


def _observability_margin(summary: dict[str, Any]) -> float:
    if str(summary.get("observability_mode")) == "bridge":
        return float(summary.get("bridge_coalignment_rate") or 0.0) - float(
            summary.get("unintended_extra_step_alignment_rate") or 0.0
        )
    return float(summary.get("intended_step_alignment_rate") or 0.0) - float(
        summary.get("unintended_step_alignment_rate") or 0.0
    )


def summarize_observability(
    *,
    root: Path,
    manifest: dict[str, Any],
    active_stage_by_detector: dict[str, str],
    max_length: int,
) -> dict[str, Any]:
    detector_rows: list[dict[str, Any]] = []
    for detector_row in list(manifest.get("detector_rows") or []):
        detector_family = str(detector_row["detector_family"])
        stage_name = active_stage_by_detector.get(detector_family, STAGE_A)
        target_rows = _evaluation_target_rows(detector_row=detector_row, stage_name=stage_name)
        target_summaries: list[dict[str, Any]] = []
        predicate_passes = 0
        detector_passes = 0
        predicate_total = 0
        detector_total = 0
        for target in target_rows:
            aligner_rows: list[dict[str, Any]] = []
            bundle = _bundle_from_prefix(root, str(target["phase11_prefix"]))
            for aligner_name in ALIGNER_NAMES:
                rows = _alignment_rows(bundle=bundle, aligner_name=aligner_name, max_length=max_length)
                summary = _evaluate_target_observability(
                    detector_family=detector_family,
                    panel_regime=str(target["panel_regime"]),
                    rows=rows,
                    source_step_key=str(target["source_step_key"]),
                    target_step_key=str(target["target_step_key"]),
                )
                aligner_rows.append({"aligner": aligner_name, **summary})
            passed_aligner_count = sum(1 for row in aligner_rows if bool(row["passed"]))
            selected = max(aligner_rows, key=_observability_margin)
            passed = passed_aligner_count >= 2
            summary_row = {
                "run_name": str(target["run_name"]),
                "panel_regime": str(target["panel_regime"]),
                "truthfulness_seed": int(target["truthfulness_seed"]),
                "detector_family": detector_family,
                "intended_observable": str(target["intended_observable"]),
                "source_step_key": str(target["source_step_key"]),
                "target_step_key": str(target["target_step_key"]),
                "passed": passed,
                "passed_aligner_count": passed_aligner_count,
                "selected_aligner": str(selected["aligner"]),
                "selected_summary": selected,
                "aligner_rows": aligner_rows,
            }
            target_summaries.append(summary_row)
            if str(target["panel_regime"]) == FAMILY_PREDICATE_SANITY:
                predicate_total += 1
                predicate_passes += int(passed)
            else:
                detector_total += 1
                detector_passes += int(passed)
        predicate_ok = predicate_total > 0 and predicate_passes == predicate_total
        detector_ok = detector_total > 0 and detector_passes == detector_total
        detector_rows.append(
            {
                "detector_family": detector_family,
                "object_class": str(detector_row["object_class"]),
                "stage": stage_name,
                "passed": predicate_ok and detector_ok,
                "predicate_sanity_passed": predicate_ok,
                "detector_families_passed": detector_ok,
                "detector_specific_observability_failure": predicate_ok and not detector_ok,
                "target_rows": target_summaries,
            }
        )
    return {"detector_rows": detector_rows, "passed_count": sum(1 for row in detector_rows if bool(row["passed"]))}


def _select_anchor(detector_family: str, target: dict[str, Any]) -> dict[str, Any]:
    anchors = [dict(row) for row in _detector_anchor_specs(detector_family)]
    if detector_family == D1_ARGUMENT_LOCUS:
        return dict(next(row for row in anchors if str(row.get("decoder_step_key")) == str(target["target_step_key"])))
    if detector_family == D2_PREDICATE_CONDITIONED_RELATION:
        index = max(0, min(len(anchors) - 1, int(target["truthfulness_seed"]) - 1))
        return dict(anchors[index])
    return dict(next(row for row in anchors if str(row.get("target_step_key")) == str(target["target_step_key"])))


def summarize_h2(
    *,
    root: Path,
    manifest: dict[str, Any],
    observability_summary: dict[str, Any],
    active_stage_by_detector: dict[str, str],
    max_length: int,
) -> dict[str, Any]:
    obs_by_detector = {str(row["detector_family"]): row for row in list(observability_summary.get("detector_rows") or [])}
    detector_rows: list[dict[str, Any]] = []
    for detector_row in list(manifest.get("detector_rows") or []):
        detector_family = str(detector_row["detector_family"])
        obs = obs_by_detector[detector_family]
        stage_name = active_stage_by_detector.get(detector_family, STAGE_A)
        target_rows = list(detector_row["stage_rows"][stage_name]["target_rows"])
        if not bool(obs.get("passed")):
            detector_rows.append(
                {
                    "detector_family": detector_family,
                    "object_class": str(detector_row["object_class"]),
                    "passed": False,
                    "stage": stage_name,
                    "null_passed_count": 0,
                    "failure_reason": "observability_failed",
                    "positive_rows": [],
                    "null_rows": [],
                }
            )
            continue
        aligner_by_run = {
            str(row["run_name"]): str(row["selected_aligner"])
            for row in list(obs.get("target_rows") or [])
        }
        positive_rows: list[dict[str, Any]] = []
        null_rows: list[dict[str, Any]] = []
        for target in target_rows:
            family = str(target["panel_regime"])
            if family not in {FAMILY_DETECTOR_H2, FAMILY_DETECTOR_NULL}:
                continue
            bundle = _bundle_from_prefix(root, str(target["phase11_prefix"]))
            anchor = _select_anchor(detector_family, target)
            row = {
                "run_name": str(target["run_name"]),
                "truthfulness_seed": int(target["truthfulness_seed"]),
                **p21._evaluate_model_native_h2_bundle(
                    bundle=bundle,
                    object_class=str(detector_row["object_class"]),
                    object_spec=anchor,
                    recipe=p21._model_native_recipe(
                        object_spec=anchor,
                        panel_regime="planted_h2",
                        base_run=str(target["phase11_prefix"]),
                    ),
                    aligner_name=aligner_by_run[str(target["run_name"])],
                    max_length=max_length,
                ),
            }
            if family == FAMILY_DETECTOR_H2:
                positive_rows.append(row)
            else:
                null_rows.append(row)
        detector_rows.append(
            {
                "detector_family": detector_family,
                "object_class": str(detector_row["object_class"]),
                "stage": stage_name,
                "passed": bool(positive_rows) and all(bool(row["passed"]) for row in positive_rows) and not any(
                    bool(row["passed"]) for row in null_rows
                ),
                "null_passed_count": sum(1 for row in null_rows if bool(row.get("passed"))),
                "positive_rows": positive_rows,
                "null_rows": null_rows,
            }
        )
    return {"detector_rows": detector_rows, "passed_count": sum(1 for row in detector_rows if bool(row["passed"]))}


def summarize_h1(
    *,
    root: Path,
    manifest: dict[str, Any],
    observability_summary: dict[str, Any],
    active_stage_by_detector: dict[str, str],
    max_length: int,
) -> dict[str, Any]:
    obs_by_detector = {str(row["detector_family"]): row for row in list(observability_summary.get("detector_rows") or [])}
    detector_rows: list[dict[str, Any]] = []
    for detector_row in list(manifest.get("detector_rows") or []):
        detector_family = str(detector_row["detector_family"])
        obs = obs_by_detector[detector_family]
        stage_name = active_stage_by_detector.get(detector_family, STAGE_A)
        target_rows = list(detector_row["stage_rows"][stage_name]["target_rows"])
        if not bool(obs.get("passed")):
            detector_rows.append(
                {
                    "detector_family": detector_family,
                    "object_class": str(detector_row["object_class"]),
                    "positive_passed": False,
                    "stage": stage_name,
                    "null_false_positive_rate": 0.0,
                    "failure_reason": "observability_failed",
                    "positive_rows": [],
                    "null_rows": [],
                }
            )
            continue
        aligner_by_run = {
            str(row["run_name"]): str(row["selected_aligner"])
            for row in list(obs.get("target_rows") or [])
        }
        positive_rows: list[dict[str, Any]] = []
        null_rows: list[dict[str, Any]] = []
        for target in target_rows:
            family = str(target["panel_regime"])
            if family not in {FAMILY_DETECTOR_H1, FAMILY_DETECTOR_NULL}:
                continue
            bundle = _bundle_from_prefix(root, str(target["phase11_prefix"]))
            anchor = _select_anchor(detector_family, target)
            row = {
                "run_name": str(target["run_name"]),
                "truthfulness_seed": int(target["truthfulness_seed"]),
                **p21._evaluate_model_native_h1_run(
                    bundle=bundle,
                    object_class=str(detector_row["object_class"]),
                    object_spec=anchor,
                    recipe=p21._model_native_recipe(
                        object_spec=anchor,
                        panel_regime="planted_h1",
                        base_run=str(target["phase11_prefix"]),
                    ),
                    aligner_name=aligner_by_run[str(target["run_name"])],
                    physical_family=str(target["run_name"]),
                    max_length=max_length,
                ),
            }
            if family == FAMILY_DETECTOR_H1:
                positive_rows.append(row)
            else:
                null_rows.append(row)
        null_rate = round(mean(int(bool(row.get("run_recovered"))) for row in null_rows), 6) if null_rows else 0.0
        detector_rows.append(
            {
                "detector_family": detector_family,
                "object_class": str(detector_row["object_class"]),
                "stage": stage_name,
                "positive_passed": bool(positive_rows) and all(bool(row["run_recovered"]) for row in positive_rows),
                "null_passed": bool(null_rate > 0.2),
                "null_false_positive_rate": null_rate,
                "positive_rows": positive_rows,
                "null_rows": null_rows,
            }
        )
    return {"detector_rows": detector_rows, "positive_passed_count": sum(1 for row in detector_rows if bool(row["positive_passed"]))}


def summarize_stage_b(
    *,
    family_validity_summary: dict[str, Any],
    anchorability_summary: dict[str, Any],
    observability_summary: dict[str, Any],
    h2_summary: dict[str, Any],
    h1_summary: dict[str, Any],
) -> dict[str, Any]:
    failed_detectors: list[str] = []
    by_validity = {str(row["detector_family"]): row for row in list(family_validity_summary.get("detector_rows") or [])}
    by_anchor = {str(row["detector_family"]): row for row in list(anchorability_summary.get("detector_rows") or [])}
    by_obs = {str(row["detector_family"]): row for row in list(observability_summary.get("detector_rows") or [])}
    by_h2 = {str(row["detector_family"]): row for row in list(h2_summary.get("detector_rows") or [])}
    by_h1 = {str(row["detector_family"]): row for row in list(h1_summary.get("detector_rows") or [])}
    for detector_family in DETECTOR_ORDER:
        if not bool(by_validity.get(detector_family, {}).get("passed")):
            failed_detectors.append(detector_family)
            continue
        if not bool(by_anchor.get(detector_family, {}).get("passed")):
            failed_detectors.append(detector_family)
            continue
        if not bool(by_obs.get(detector_family, {}).get("passed")):
            failed_detectors.append(detector_family)
            continue
        if not bool(by_h2.get(detector_family, {}).get("passed")):
            failed_detectors.append(detector_family)
            continue
        if not bool(by_h1.get(detector_family, {}).get("positive_passed")) or bool(
            by_h1.get(detector_family, {}).get("null_passed")
        ):
            failed_detectors.append(detector_family)
    return {
        "triggered": bool(failed_detectors),
        "failed_detectors": failed_detectors,
        "stage_b_executed": False,
    }


def summarize_detector_competition(
    *,
    family_validity_summary: dict[str, Any],
    anchorability_summary: dict[str, Any],
    observability_summary: dict[str, Any],
    h2_summary: dict[str, Any],
    h1_summary: dict[str, Any],
    real_regression_summary: dict[str, Any],
) -> dict[str, Any]:
    by_validity = {str(row["detector_family"]): row for row in list(family_validity_summary.get("detector_rows") or [])}
    by_anchor = {str(row["detector_family"]): row for row in list(anchorability_summary.get("detector_rows") or [])}
    by_obs = {str(row["detector_family"]): row for row in list(observability_summary.get("detector_rows") or [])}
    by_h2 = {str(row["detector_family"]): row for row in list(h2_summary.get("detector_rows") or [])}
    by_h1 = {str(row["detector_family"]): row for row in list(h1_summary.get("detector_rows") or [])}
    detector_rows: list[dict[str, Any]] = []
    validated_detectors: list[str] = []
    for detector_family in DETECTOR_ORDER:
        failures: list[str] = []
        if not bool(by_validity.get(detector_family, {}).get("passed")):
            failures.append("family_construction_failed")
        if not bool(by_anchor.get(detector_family, {}).get("passed")):
            failures.append("anchorability_failed")
        if not bool(by_obs.get(detector_family, {}).get("passed")):
            failures.append("observability_failed")
        if bool(by_obs.get(detector_family, {}).get("detector_specific_observability_failure")):
            failures.append("detector_specific_observability_failure")
        if not bool(by_h2.get(detector_family, {}).get("passed")):
            failures.append("detector_h2_failed")
        if not bool(by_h1.get(detector_family, {}).get("positive_passed")):
            failures.append("detector_h1_failed")
        if bool(by_h1.get(detector_family, {}).get("null_passed")):
            failures.append("null_false_positive")
        validated = not failures
        if validated:
            validated_detectors.append(detector_family)
        detector_rows.append(
            {
                "detector_family": detector_family,
                "truthfulness_validated": validated,
                "failure_reasons": failures,
            }
        )
    if not validated_detectors:
        competition_decision = "retain_phase18"
        strongest_claim_decision = "retain_phase18"
    elif bool(real_regression_summary.get("all_validated_detectors_agree_negative")):
        competition_decision = "support_ontology_final_upgrade"
        strongest_claim_decision = "support_ontology_final_upgrade"
    else:
        competition_decision = "detector_competition_inconclusive"
        strongest_claim_decision = "retain_phase18"
    return {
        "detector_rows": detector_rows,
        "validated_detector_count": len(validated_detectors),
        "validated_detectors": validated_detectors,
        "decision": competition_decision,
        "strongest_claim_decision": strongest_claim_decision,
        "decision_status": "final",
        "supports_ontology_final_upgrade": strongest_claim_decision == "support_ontology_final_upgrade",
    }


def render_detector_competition_status_markdown(
    *,
    benchmark_summary: dict[str, Any],
    manifest_summary: dict[str, Any],
    family_validity_summary: dict[str, Any],
    anchorability_summary: dict[str, Any],
    observability_summary: dict[str, Any],
    h2_summary: dict[str, Any],
    h1_summary: dict[str, Any],
    stage_b_summary: dict[str, Any],
    real_regression_summary: dict[str, Any],
    competition_summary: dict[str, Any],
) -> str:
    lines = [
        "# Phase 23 Status",
        "",
        "## Frozen Framing",
        "- `G1_binding`: bounded hypothesis-final success",
        "- `G2_distributed_geometry`: `Phase 18` family-final negative under the frozen closure rule.",
        "- `Phase 23` is a separate detector-ontology competition branch.",
        "",
        "## Benchmarks",
        f"- `benchmark_count = {int(benchmark_summary.get('benchmark_count') or 0)}`",
    ]
    for row in list(benchmark_summary.get("benchmark_rows") or []):
        lines.append(
            f"- `{row['detector_family']}`: `canonical_example_count = {int(row.get('canonical_example_count') or 0)}`, `coverage = {float(dict(row.get('selection') or {}).get('exact_counterpart_coverage_rate') or 0.0)}`"
        )
    lines.extend(["", "## Panels"])
    for row in list(manifest_summary.get("detector_rows") or []):
        lines.append(
            f"- `{row['detector_family']}`: `stage_a_target_count = {len(list(row['stage_rows']['stage_a']['target_rows']))}`, `stage_b_target_count = {len(list(row['stage_rows']['stage_b']['target_rows']))}`"
        )
    lines.extend(["", "## Family Validity"])
    for row in list(family_validity_summary.get("detector_rows") or []):
        lines.append(
            f"- `{row['detector_family']}`: `passed = {bool(row.get('passed'))}`, `stage = {row.get('stage')}`, `failure_reasons = {row.get('failure_reasons') or []}`"
        )
    lines.extend(["", "## Anchorability"])
    for row in list(anchorability_summary.get("detector_rows") or []):
        lines.append(
            f"- `{row['detector_family']}`: `passed = {bool(row.get('passed'))}`, `usable_anchor_count = {int(row.get('usable_anchor_count') or 0)}`"
        )
    lines.extend(["", "## Observability"])
    for row in list(observability_summary.get("detector_rows") or []):
        lines.append(
            f"- `{row['detector_family']}`: `passed = {bool(row.get('passed'))}`, `predicate_sanity_passed = {bool(row.get('predicate_sanity_passed'))}`, `detector_families_passed = {bool(row.get('detector_families_passed'))}`, `detector_specific_failure = {bool(row.get('detector_specific_observability_failure'))}`"
        )
    lines.extend(["", "## Detector H2"])
    for row in list(h2_summary.get("detector_rows") or []):
        lines.append(
            f"- `{row['detector_family']}`: `passed = {bool(row.get('passed'))}`, `stage = {row.get('stage')}`, `null_passed_count = {int(row.get('null_passed_count') or 0)}`"
        )
    lines.extend(["", "## Detector H1"])
    for row in list(h1_summary.get("detector_rows") or []):
        lines.append(
            f"- `{row['detector_family']}`: `positive_passed = {bool(row.get('positive_passed'))}`, `stage = {row.get('stage')}`, `null_false_positive_rate = {float(row.get('null_false_positive_rate') or 0.0)}`"
        )
    lines.extend(
        [
            "",
            "## Stage B",
            f"- `triggered = {bool(stage_b_summary.get('triggered'))}`",
            f"- `failed_detectors = {stage_b_summary.get('failed_detectors') or []}`",
            f"- `stage_b_executed = {bool(stage_b_summary.get('stage_b_executed'))}`",
            "",
            "## Real-Family Regression",
            f"- `executed = {bool(real_regression_summary.get('executed'))}`",
            f"- `all_validated_detectors_agree_negative = {bool(real_regression_summary.get('all_validated_detectors_agree_negative'))}`",
            "",
            "## Competition Verdict",
            f"- `decision = {competition_summary.get('decision')}`",
            f"- `strongest_claim_decision = {competition_summary.get('strongest_claim_decision')}`",
            f"- `validated_detectors = {competition_summary.get('validated_detectors') or []}`",
        ]
    )
    return "\n".join(lines) + "\n"
