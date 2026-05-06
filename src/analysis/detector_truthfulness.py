from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from data.systematicity_schema import (
    CONTEXT_FILLERS,
    CONTEXT_PREDICATES,
    FILLERS,
    MODIFIERS,
    PREDICATES,
    SystematicitySchemaRecord,
    build_counterpart_families,
    build_target_form,
    summarize_schema_records,
)
from data.surface_templates import render_template_candidates
from data.surface_validation import select_primary_candidates, summarize_validated_candidates, validate_surface_candidate
from eval.harness import aggregate_metrics
from interventions.generation import (
    GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS,
    GEOMETRY_INTERACTION_OBJECT_CLASS,
    GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS,
)
from utils.io import read_jsonl

from . import interaction_objects as p17
from . import model_native_detectability as p20
from . import argument_detector_validation as p21
from .family_closure import ALIGNER_NAMES, _align_with_selected, load_json


CLASS_ORDER: tuple[str, ...] = (
    GEOMETRY_INTERACTION_OBJECT_CLASS,
    GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS,
    GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS,
)
STAGE_A = "stage_a"
STAGE_B = "stage_b"

FAMILY_PREDICATE_SANITY = "predicate_sanity_h2"
FAMILY_ARGUMENT_H2 = "argument_h2"
FAMILY_ARGUMENT_H1 = "argument_h1"
FAMILY_ARGUMENT_NULL = "argument_null"

FAMILY_ORDER: tuple[str, ...] = (
    FAMILY_PREDICATE_SANITY,
    FAMILY_ARGUMENT_H2,
    FAMILY_ARGUMENT_H1,
    FAMILY_ARGUMENT_NULL,
)

FAMILY_THRESHOLDS = {
    FAMILY_PREDICATE_SANITY: 0.9,
    FAMILY_ARGUMENT_H2: 0.85,
    FAMILY_ARGUMENT_H1: 0.55,
}

DEFAULT_COUNTS = {
    "train": 8,
    "dev": 2,
    "gen": 2,
}

CLASS_LABELS = {
    GEOMETRY_INTERACTION_OBJECT_CLASS: "c1",
    GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS: "c2",
    GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS: "c3",
}

CLASS_PANEL_NAMES = {
    GEOMETRY_INTERACTION_OBJECT_CLASS: "detector_truthfulness_c1",
    GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS: "detector_truthfulness_c2",
    GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS: "detector_truthfulness_c3",
}


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _class_family_case(object_class: str, family_name: str) -> str:
    return f"{CLASS_LABELS[object_class]}_{family_name}"


def _benchmark_name(object_class: str) -> str:
    return CLASS_PANEL_NAMES[object_class]


def _class_blueprint(object_class: str) -> dict[str, Any]:
    if object_class == GEOMETRY_INTERACTION_OBJECT_CLASS:
        return {
            "voice": "active",
            "depth": 0,
            "argument_step_seed_map": {1: "agent_value", 2: "agent_value"},
            "predicate_pool": ["admire", "follow", "help"],
            "argument_predicate": "help",
        }
    if object_class == GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS:
        return {
            "voice": "active",
            "depth": 1,
            "argument_step_seed_map": {1: "agent_value", 2: "theme_value"},
            "predicate_pool": ["support", "contact", "help"],
            "argument_predicate": "support",
        }
    return {
        "voice": "active",
        "depth": 0,
        "argument_step_seed_map": {1: "agent_value", 2: "agent_value"},
        "predicate_pool": ["contact", "support", "help"],
        "argument_predicate": "contact",
    }


def _predicate_record(family_name: str, family_index: int) -> dict[str, str]:
    pool = {row["lemma"]: row for row in PREDICATES}
    if family_name == FAMILY_PREDICATE_SANITY:
        ordered = [pool["admire"], pool["follow"], pool["help"], pool["support"], pool["contact"]]
        return ordered[family_index % len(ordered)]
    if family_name in {FAMILY_ARGUMENT_H2, FAMILY_ARGUMENT_H1, FAMILY_ARGUMENT_NULL}:
        lemma = {
            GEOMETRY_INTERACTION_OBJECT_CLASS: "help",
            GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS: "support",
            GEOMETRY_QUERY_SOURCE_RELATION_OBJECT_CLASS: "contact",
        }
    return pool["help"]


def _family_record_count(config: dict[str, Any], split_name: str) -> int:
    counts = dict(config.get("family_counts") or {})
    if split_name in counts:
        return int(counts[split_name])
    return int(DEFAULT_COUNTS[split_name])


def _build_truthfulness_family_records(
    *,
    benchmark_name: str,
    object_class: str,
    family_name: str,
    family_index: int,
    split_name: str,
    rng: random.Random,
) -> list[SystematicitySchemaRecord]:
    predicate_pool = {row["lemma"]: row for row in PREDICATES}
    blueprint = _class_blueprint(object_class)
    if family_name == FAMILY_PREDICATE_SANITY:
        predicate = predicate_pool[blueprint["predicate_pool"][family_index % len(blueprint["predicate_pool"])]]
    else:
        predicate = predicate_pool[blueprint["argument_predicate"]]
    context_predicate = CONTEXT_PREDICATES[family_index % len(CONTEXT_PREDICATES)]
    filler_pool = rng.sample(list(FILLERS), 4)
    modifier_pool = rng.sample(list(MODIFIERS), 2)
    context_filler = CONTEXT_FILLERS[family_index % len(CONTEXT_FILLERS)]
    base_agent, base_theme, alt_agent, alt_theme = filler_pool
    agent_modifier, theme_modifier = modifier_pool
    family_id = f"{benchmark_name}:{split_name}:{family_name}:{family_index:04d}"
    voice = str(blueprint["voice"])
    depth = int(blueprint["depth"])
    item_specs = (
        {
            "counterpart_type": "rescue_donor",
            "split": "train" if split_name == "train" else "dev",
            "voice": "active",
            "depth": 0,
            "agent": base_agent,
            "theme": base_theme,
            "agent_modifier": agent_modifier,
            "theme_modifier": theme_modifier,
            "allowed_delta": "surface_only",
        },
        {
            "counterpart_type": "rescue_target",
            "split": split_name,
            "voice": voice,
            "depth": depth,
            "agent": base_agent,
            "theme": base_theme,
            "agent_modifier": agent_modifier,
            "theme_modifier": theme_modifier,
            "allowed_delta": "hard_target",
        },
        {
            "counterpart_type": "role_swap",
            "split": split_name,
            "voice": voice,
            "depth": depth,
            "agent": base_theme,
            "theme": base_agent,
            "agent_modifier": theme_modifier,
            "theme_modifier": agent_modifier,
            "allowed_delta": "role_swap",
        },
        {
            "counterpart_type": "agent_swap",
            "split": split_name,
            "voice": voice,
            "depth": depth,
            "agent": alt_agent,
            "theme": base_theme,
            "agent_modifier": agent_modifier,
            "theme_modifier": theme_modifier,
            "allowed_delta": "agent_swap",
        },
        {
            "counterpart_type": "theme_swap",
            "split": split_name,
            "voice": voice,
            "depth": depth,
            "agent": base_agent,
            "theme": alt_theme,
            "agent_modifier": agent_modifier,
            "theme_modifier": theme_modifier,
            "allowed_delta": "theme_swap",
        },
        {
            "counterpart_type": "filler_swap",
            "split": split_name,
            "voice": voice,
            "depth": depth,
            "agent": alt_agent,
            "theme": alt_theme,
            "agent_modifier": agent_modifier,
            "theme_modifier": theme_modifier,
            "allowed_delta": "filler_swap",
        },
        {
            "counterpart_type": "modifier_swap",
            "split": split_name,
            "voice": voice,
            "depth": depth,
            "agent": base_agent,
            "theme": base_theme,
            "agent_modifier": theme_modifier,
            "theme_modifier": agent_modifier,
            "allowed_delta": "modifier_swap",
        },
    )
    records: list[SystematicitySchemaRecord] = []
    for spec in item_specs:
        target_agent = str(spec["agent"])
        target_theme = str(spec["theme"])
        if family_name == FAMILY_ARGUMENT_NULL:
            target_agent, target_theme = target_theme, target_agent
        schema_id = f"{family_id}:{spec['counterpart_type']}"
        target_form = build_target_form(
            predicate_lemma=predicate["lemma"],
            agent=target_agent,
            theme=target_theme,
            agent_modifier=str(spec["agent_modifier"]),
            theme_modifier=str(spec["theme_modifier"]),
        )
        records.append(
            SystematicitySchemaRecord(
                item_id=schema_id,
                benchmark_name=benchmark_name,
                split=str(spec["split"]),
                raw_split=str(spec["split"]),
                schema_id=schema_id,
                counterpart_family_id=family_id,
                counterpart_type=str(spec["counterpart_type"]),
                construction_family=_class_family_case(object_class, family_name),
                voice=str(spec["voice"]),
                embedding_depth=int(spec["depth"]),
                predicate_lemma=predicate["lemma"],
                predicate_past=predicate["past"],
                predicate_participle=predicate["participle"],
                context_predicate_lemma=context_predicate["lemma"],
                context_predicate_past=context_predicate["past"],
                agent=str(spec["agent"]),
                theme=str(spec["theme"]),
                agent_modifier=str(spec["agent_modifier"]),
                theme_modifier=str(spec["theme_modifier"]),
                context_filler=context_filler,
                target_form=target_form,
                metadata={
                    "family_split": split_name,
                    "allowed_delta": spec["allowed_delta"],
                    "truthfulness_class": object_class,
                    "truthfulness_family": family_name,
                    "truthfulness_family_case": _class_family_case(object_class, family_name),
                    "truthfulness_counterpart_index": family_index,
                    "no_forced_h2": True,
                },
            )
        )
    return records


def build_truthfulness_benchmarks(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["project_root"])
    processed_dir = root / "data/processed"
    raw_dir = root / "data/raw/phase11"
    derived_dir = root / "data/derived"
    reports_dir = root / "reports/detector_truthfulness"
    benchmark_rows: list[dict[str, Any]] = []
    rng = random.Random(int(config.get("seed") or 22))
    for object_class in CLASS_ORDER:
        benchmark_name = _benchmark_name(object_class)
        schema_records: list[SystematicitySchemaRecord] = []
        for family_name in FAMILY_ORDER:
            for split_name in ("train", "dev", "gen"):
                for family_index in range(_family_record_count(config, split_name)):
                    schema_records.extend(
                        _build_truthfulness_family_records(
                            benchmark_name=benchmark_name,
                            object_class=object_class,
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
        processed_payload = [example.to_dict() for example in canonical_examples]
        processed_path = processed_dir / f"{benchmark_name}.jsonl"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        with processed_path.open("w", encoding="utf-8") as handle:
            for row in processed_payload:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
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
                "benchmark_name": benchmark_name,
                "object_class": object_class,
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


def _panel_target(
    *,
    benchmark_name: str,
    object_class: str,
    family_name: str,
    seed: int,
    phase11_prefix: str,
    panel_regime: str,
    intended_step_key: str,
    paired_null_id: str | None,
    recipe: dict[str, Any],
) -> dict[str, Any]:
    return {
        "target_id": phase11_prefix,
        "run_name": phase11_prefix,
        "phase11_prefix": phase11_prefix,
        "benchmark": benchmark_name,
        "panel_regime": panel_regime,
        "truthfulness_class": object_class,
        "truthfulness_family": _class_family_case(object_class, family_name),
        "truthfulness_seed": seed,
        "intended_step_key": intended_step_key,
        "paired_null_id": paired_null_id,
        "train_generalization_case_allowlist": [_class_family_case(object_class, family_name)],
        "num_train_epochs": 2,
        "train_batch_size": 4,
        "eval_batch_size": 4,
        "learning_rate": 5e-5,
        "seed": seed,
        **recipe,
    }


def _stage_a_recipe(object_class: str, family_name: str) -> dict[str, Any]:
    if object_class == GEOMETRY_INTERACTION_OBJECT_CLASS:
        if family_name == FAMILY_PREDICATE_SANITY:
            return {
                "decoder_cross_attention_scale": 1.5,
                "decoder_readout_subspace_keep_ratio": 0.35,
                "decoder_readout_subspace_mode": "keep",
                "decoder_readout_subspace_seed": 22,
            }
        if family_name == FAMILY_ARGUMENT_H2:
            return {
                "decoder_cross_attention_scale": 1.75,
                "decoder_readout_subspace_keep_ratio": 0.45,
                "decoder_readout_subspace_mode": "keep",
                "decoder_readout_subspace_seed": 23,
            }
        if family_name == FAMILY_ARGUMENT_H1:
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
    if object_class == GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS:
        if family_name == FAMILY_PREDICATE_SANITY:
            return {
                "decoder_cross_attention_scale": 1.5,
                "decoder_cross_attention_layer_keep_ratio": 0.5,
                "decoder_cross_attention_layer_policy": "alternating",
                "decoder_cross_attention_layer_seed": 22,
            }
        if family_name == FAMILY_ARGUMENT_H2:
            return {
                "decoder_cross_attention_scale": 1.75,
                "decoder_cross_attention_layer_keep_ratio": 0.5,
                "decoder_cross_attention_layer_policy": "alternating",
                "decoder_cross_attention_layer_seed": 23,
            }
        if family_name == FAMILY_ARGUMENT_H1:
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
    if family_name == FAMILY_PREDICATE_SANITY:
        return {
            "decoder_readout_subspace_keep_ratio": 0.35,
            "decoder_readout_subspace_mode": "keep",
            "decoder_readout_subspace_seed": 22,
            "decoder_cross_attention_relation_barrier_keep_ratio": 0.2,
            "decoder_cross_attention_relation_barrier_scale": 0.5,
        }
    if family_name == FAMILY_ARGUMENT_H2:
        return {
            "decoder_cross_attention_relation_barrier_keep_ratio": 0.35,
            "decoder_cross_attention_relation_barrier_scale": 1.75,
            "decoder_readout_subspace_keep_ratio": 0.45,
            "decoder_readout_subspace_mode": "keep",
            "decoder_readout_subspace_seed": 23,
        }
    if family_name == FAMILY_ARGUMENT_H1:
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


def _stage_b_recipe(object_class: str, family_name: str) -> dict[str, Any]:
    if object_class == GEOMETRY_INTERACTION_OBJECT_CLASS:
        if family_name == FAMILY_ARGUMENT_H2:
            return {
                "decoder_cross_attention_scale": 2.0,
                "decoder_readout_subspace_keep_ratio": 0.55,
                "decoder_readout_subspace_mode": "keep",
                "decoder_readout_subspace_seed": 32,
            }
        if family_name == FAMILY_ARGUMENT_H1:
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
    if object_class == GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS:
        if family_name == FAMILY_ARGUMENT_H2:
            return {
                "decoder_cross_attention_scale": 2.0,
                "decoder_cross_attention_layer_keep_ratio": 0.7,
                "decoder_cross_attention_layer_policy": "alternating",
                "decoder_cross_attention_layer_seed": 32,
            }
        if family_name == FAMILY_ARGUMENT_H1:
            return {
                "decoder_cross_attention_scale": 1.5,
                "decoder_cross_attention_layer_keep_ratio": 0.7,
                "decoder_cross_attention_layer_policy": "alternating",
                "decoder_cross_attention_layer_seed": 33,
                "decoder_cross_attention_relation_barrier_keep_ratio": 0.45,
                "decoder_cross_attention_relation_barrier_scale": 0.65,
            }
        return {
            "decoder_cross_attention_scale": 1.5,
            "decoder_cross_attention_layer_keep_ratio": 0.7,
            "decoder_cross_attention_layer_policy": "top",
            "decoder_cross_attention_layer_seed": 34,
        }
    if family_name == FAMILY_ARGUMENT_H2:
        return {
            "decoder_cross_attention_relation_barrier_keep_ratio": 0.45,
            "decoder_cross_attention_relation_barrier_scale": 2.0,
            "decoder_readout_subspace_keep_ratio": 0.55,
            "decoder_readout_subspace_mode": "keep",
            "decoder_readout_subspace_seed": 32,
        }
    if family_name == FAMILY_ARGUMENT_H1:
        return {
            "decoder_cross_attention_relation_barrier_keep_ratio": 0.45,
            "decoder_cross_attention_relation_barrier_scale": 1.5,
            "decoder_readout_subspace_keep_ratio": 0.35,
            "decoder_readout_subspace_mode": "keep",
            "decoder_readout_subspace_seed": 33,
        }
    return {
        "decoder_cross_attention_scale": 1.5,
        "decoder_readout_keep_ratio": 0.45,
    }


def build_truthfulness_panels(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["project_root"])
    configs_dir = root / "configs/experiments"
    panel_rows: list[dict[str, Any]] = []
    for object_class in CLASS_ORDER:
        benchmark_name = _benchmark_name(object_class)
        panel_name = benchmark_name
        stage_a_targets: list[dict[str, Any]] = []
        stage_b_targets: list[dict[str, Any]] = []
        blueprint = _class_blueprint(object_class)
        for seed in (1, 2):
            stage_a_targets.append(
                _panel_target(
                    benchmark_name=benchmark_name,
                    object_class=object_class,
                    family_name=FAMILY_PREDICATE_SANITY,
                    seed=seed,
                    phase11_prefix=f"{panel_name}_{FAMILY_PREDICATE_SANITY}_seed{seed}",
                    panel_regime=FAMILY_PREDICATE_SANITY,
                    intended_step_key="predicate_value",
                    paired_null_id=None,
                    recipe=_stage_a_recipe(object_class, FAMILY_PREDICATE_SANITY),
                )
            )
            paired_null = f"{panel_name}_{FAMILY_ARGUMENT_NULL}_seed{seed}"
            intended_step = str(blueprint["argument_step_seed_map"][seed])
            stage_a_targets.append(
                _panel_target(
                    benchmark_name=benchmark_name,
                    object_class=object_class,
                    family_name=FAMILY_ARGUMENT_H2,
                    seed=seed,
                    phase11_prefix=f"{panel_name}_{FAMILY_ARGUMENT_H2}_seed{seed}",
                    panel_regime=FAMILY_ARGUMENT_H2,
                    intended_step_key=intended_step,
                    paired_null_id=paired_null,
                    recipe=_stage_a_recipe(object_class, FAMILY_ARGUMENT_H2),
                )
            )
            stage_a_targets.append(
                _panel_target(
                    benchmark_name=benchmark_name,
                    object_class=object_class,
                    family_name=FAMILY_ARGUMENT_H1,
                    seed=seed,
                    phase11_prefix=f"{panel_name}_{FAMILY_ARGUMENT_H1}_seed{seed}",
                    panel_regime=FAMILY_ARGUMENT_H1,
                    intended_step_key=intended_step,
                    paired_null_id=paired_null,
                    recipe=_stage_a_recipe(object_class, FAMILY_ARGUMENT_H1),
                )
            )
            stage_a_targets.append(
                _panel_target(
                    benchmark_name=benchmark_name,
                    object_class=object_class,
                    family_name=FAMILY_ARGUMENT_NULL,
                    seed=seed,
                    phase11_prefix=paired_null,
                    panel_regime=FAMILY_ARGUMENT_NULL,
                    intended_step_key=intended_step,
                    paired_null_id=None,
                    recipe=_stage_a_recipe(object_class, FAMILY_ARGUMENT_NULL),
                )
            )
            stage_b_targets.extend(
                [
                    _panel_target(
                        benchmark_name=benchmark_name,
                        object_class=object_class,
                        family_name=FAMILY_ARGUMENT_H2,
                        seed=seed,
                        phase11_prefix=f"{panel_name}_{FAMILY_ARGUMENT_H2}_stageb_seed{seed}",
                        panel_regime=FAMILY_ARGUMENT_H2,
                        intended_step_key=intended_step,
                        paired_null_id=f"{panel_name}_{FAMILY_ARGUMENT_NULL}_stageb_seed{seed}",
                        recipe=_stage_b_recipe(object_class, FAMILY_ARGUMENT_H2),
                    ),
                    _panel_target(
                        benchmark_name=benchmark_name,
                        object_class=object_class,
                        family_name=FAMILY_ARGUMENT_H1,
                        seed=seed,
                        phase11_prefix=f"{panel_name}_{FAMILY_ARGUMENT_H1}_stageb_seed{seed}",
                        panel_regime=FAMILY_ARGUMENT_H1,
                        intended_step_key=intended_step,
                        paired_null_id=f"{panel_name}_{FAMILY_ARGUMENT_NULL}_stageb_seed{seed}",
                        recipe=_stage_b_recipe(object_class, FAMILY_ARGUMENT_H1),
                    ),
                    _panel_target(
                        benchmark_name=benchmark_name,
                        object_class=object_class,
                        family_name=FAMILY_ARGUMENT_NULL,
                        seed=seed,
                        phase11_prefix=f"{panel_name}_{FAMILY_ARGUMENT_NULL}_stageb_seed{seed}",
                        panel_regime=FAMILY_ARGUMENT_NULL,
                        intended_step_key=intended_step,
                        paired_null_id=None,
                        recipe=_stage_b_recipe(object_class, FAMILY_ARGUMENT_NULL),
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
                "phase": "phase22",
                "stage": STAGE_A,
                "object_class": object_class,
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
                "phase": "phase22",
                "stage": STAGE_B,
                "object_class": object_class,
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
                "object_class": object_class,
                "stage_a_panel_path": str(stage_a_path.relative_to(root)),
                "stage_b_panel_path": str(stage_b_path.relative_to(root)),
                "stage_a_target_count": len(stage_a_targets),
                "stage_b_target_count": len(stage_b_targets),
            }
        )
    return {"panel_rows": panel_rows, "panel_count": len(panel_rows)}


def build_truthfulness_manifest(*, root: Path, panel_paths: dict[str, dict[str, str]]) -> dict[str, Any]:
    class_rows: list[dict[str, Any]] = []
    for object_class in CLASS_ORDER:
        stage_rows: dict[str, Any] = {}
        for stage_name in (STAGE_A, STAGE_B):
            panel_path = root / str(panel_paths[object_class][stage_name])
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
                        "intended_step_key": str(target["intended_step_key"]),
                        "paired_null_id": target.get("paired_null_id"),
                    }
                    for target in list(panel.get("targets") or [])
                ],
            }
        class_rows.append(
            {
                "object_class": object_class,
                "benchmark_name": _benchmark_name(object_class),
                "anchor_specs": p20.get_stage20_synthetic_anchor_specs(object_class=object_class),
                "stage_rows": stage_rows,
            }
        )
    return {"class_count": len(class_rows), "class_rows": class_rows}


def summarize_anchorability(manifest: dict[str, Any]) -> dict[str, Any]:
    class_rows = []
    for row in list(manifest.get("class_rows") or []):
        usable = len(list(row.get("anchor_specs") or []))
        class_rows.append(
            {
                "object_class": str(row["object_class"]),
                "passed": usable >= 2,
                "usable_anchor_count": usable,
                "anchor_ids": [str(anchor["object_id"]) for anchor in list(row.get("anchor_specs") or [])],
            }
        )
    return {"class_rows": class_rows, "passed_count": sum(1 for row in class_rows if bool(row["passed"]))}


def _benchmark_index(root: Path, benchmark_name: str) -> dict[str, dict[str, Any]]:
    return {str(row["item_id"]): row for row in read_jsonl(root / "data/processed" / f"{benchmark_name}.jsonl")}


def _prediction_records(root: Path, run_name: str) -> list[dict[str, Any]]:
    return read_jsonl(root / "runs/behavior_runs" / f"{run_name}_predictions.jsonl")


def _filter_records_for_family(records: list[dict[str, Any]], family_case: str) -> list[dict[str, Any]]:
    return [record for record in records if str(record.get("generalization_case")) == family_case]


def _family_metric(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = aggregate_metrics(records)
    return {
        "count": int(metrics["count"]),
        "exact_match": float(metrics["exact_match"]),
        "clause_f1": float(metrics["clause_f1"]),
    }


def summarize_family_validity(
    *,
    root: Path,
    manifest: dict[str, Any],
    active_stage_by_class: dict[str, str],
) -> dict[str, Any]:
    class_rows: list[dict[str, Any]] = []
    for class_row in list(manifest.get("class_rows") or []):
        object_class = str(class_row["object_class"])
        stage_name = active_stage_by_class.get(object_class, STAGE_A)
        target_rows = _evaluation_target_rows(class_row=class_row, stage_name=stage_name)
        benchmark_index = _benchmark_index(root, str(class_row["benchmark_name"]))
        family_rows: list[dict[str, Any]] = []
        metrics_by_family: dict[tuple[str, int], dict[str, Any]] = {}
        for target in target_rows:
            predictions = _prediction_records(root, str(target["run_name"]))
            family_case = str(target["truthfulness_family"])
            subset = _filter_records_for_family(predictions, family_case)
            metric = _family_metric(subset)
            metric.update(
                {
                    "panel_regime": str(target["panel_regime"]),
                    "run_name": str(target["run_name"]),
                    "truthfulness_seed": int(target["truthfulness_seed"]),
                    "truthfulness_family": family_case,
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
            h2 = metrics_by_family[(FAMILY_ARGUMENT_H2, seed)]
            if max(h2["exact_match"], h2["clause_f1"]) < FAMILY_THRESHOLDS[FAMILY_ARGUMENT_H2]:
                passed = False
                failure_reasons.append(f"{FAMILY_ARGUMENT_H2}:seed{seed}")
            h1 = metrics_by_family[(FAMILY_ARGUMENT_H1, seed)]
            if max(h1["exact_match"], h1["clause_f1"]) < FAMILY_THRESHOLDS[FAMILY_ARGUMENT_H1]:
                passed = False
                failure_reasons.append(f"{FAMILY_ARGUMENT_H1}:seed{seed}")
            if h1["clause_f1"] >= h2["clause_f1"]:
                passed = False
                failure_reasons.append(f"{FAMILY_ARGUMENT_H1}:not_weaker_than_h2:seed{seed}")
            null_row = metrics_by_family[(FAMILY_ARGUMENT_NULL, seed)]
            if null_row["clause_f1"] > max(0.4, h1["clause_f1"] - 0.1):
                passed = False
                failure_reasons.append(f"{FAMILY_ARGUMENT_NULL}:too_strong:seed{seed}")
        class_rows.append(
            {
                "object_class": object_class,
                "stage": stage_name,
                "passed": passed,
                "failure_reasons": failure_reasons,
                "family_rows": family_rows,
            }
        )
    return {"class_rows": class_rows, "passed_count": sum(1 for row in class_rows if bool(row["passed"]))}


def _bundle_from_prefix(root: Path, phase11_prefix: str) -> dict[str, Any]:
    return p17._load_protocol_bundle(root, root / "reports/protocol_runs" / f"{phase11_prefix}_summary.json")


def _evaluate_run_observability(
    *,
    bundle: dict[str, Any],
    intended_step_key: str,
    aligner_name: str,
    max_length: int,
) -> dict[str, Any]:
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
        aligned_steps = {str(key) for key in dict(alignments.get("alignments") or {}).keys()}
        rows.append(
            {
                "item_id": str(item["item_id"]),
                "intended_step_aligned": intended_step_key in aligned_steps,
                "unintended_step_aligned": any(step != intended_step_key for step in aligned_steps),
            }
        )
    intended_rate = round(mean(int(bool(row["intended_step_aligned"])) for row in rows), 6) if rows else 0.0
    unintended_rate = round(mean(int(bool(row["unintended_step_aligned"])) for row in rows), 6) if rows else 1.0
    return {
        "aligner": aligner_name,
        "row_count": len(rows),
        "intended_step_alignment_rate": intended_rate,
        "unintended_step_alignment_rate": unintended_rate,
        "passed": intended_rate >= 0.8 and unintended_rate <= 0.2,
    }


def summarize_observability(
    *,
    root: Path,
    manifest: dict[str, Any],
    active_stage_by_class: dict[str, str],
    max_length: int,
) -> dict[str, Any]:
    class_rows: list[dict[str, Any]] = []
    for class_row in list(manifest.get("class_rows") or []):
        object_class = str(class_row["object_class"])
        stage_name = active_stage_by_class.get(object_class, STAGE_A)
        target_rows = _evaluation_target_rows(class_row=class_row, stage_name=stage_name)
        target_summaries: list[dict[str, Any]] = []
        predicate_passes = 0
        argument_passes = 0
        predicate_total = 0
        argument_total = 0
        for target in target_rows:
            bundle = _bundle_from_prefix(root, str(target["phase11_prefix"]))
            aligner_rows = [
                _evaluate_run_observability(
                    bundle=bundle,
                    intended_step_key=str(target["intended_step_key"]),
                    aligner_name=aligner,
                    max_length=max_length,
                )
                for aligner in ALIGNER_NAMES
            ]
            passed_aligner_count = sum(1 for row in aligner_rows if bool(row["passed"]))
            selected = max(
                aligner_rows,
                key=lambda row: (
                    int(bool(row["passed"])),
                    float(row["intended_step_alignment_rate"]) - float(row["unintended_step_alignment_rate"]),
                ),
            )
            passed = passed_aligner_count >= 2
            family = str(target["panel_regime"])
            summary = {
                "run_name": str(target["run_name"]),
                "panel_regime": family,
                "truthfulness_seed": int(target["truthfulness_seed"]),
                "intended_step_key": str(target["intended_step_key"]),
                "passed": passed,
                "passed_aligner_count": passed_aligner_count,
                "selected_aligner": str(selected["aligner"]),
                "selected_summary": selected,
                "aligner_rows": aligner_rows,
            }
            target_summaries.append(summary)
            if family == FAMILY_PREDICATE_SANITY:
                predicate_total += 1
                predicate_passes += int(passed)
            else:
                argument_total += 1
                argument_passes += int(passed)
        predicate_ok = predicate_total > 0 and predicate_passes == predicate_total
        argument_ok = argument_total > 0 and argument_passes == argument_total
        class_rows.append(
            {
                "object_class": object_class,
                "stage": stage_name,
                "passed": predicate_ok and argument_ok,
                "predicate_sanity_passed": predicate_ok,
                "argument_families_passed": argument_ok,
                "argument_specific_observability_failure": predicate_ok and not argument_ok,
                "target_rows": target_summaries,
            }
        )
    return {"class_rows": class_rows, "passed_count": sum(1 for row in class_rows if bool(row["passed"]))}


def _evaluation_target_rows(*, class_row: dict[str, Any], stage_name: str) -> list[dict[str, Any]]:
    if stage_name == STAGE_A:
        return list(class_row["stage_rows"][STAGE_A]["target_rows"])
    stage_a_rows = list(class_row["stage_rows"][STAGE_A]["target_rows"])
    stage_b_rows = list(class_row["stage_rows"][stage_name]["target_rows"])
    predicate_rows = [row for row in stage_a_rows if str(row["panel_regime"]) == FAMILY_PREDICATE_SANITY]
    argument_rows = [row for row in stage_b_rows if str(row["panel_regime"]) != FAMILY_PREDICATE_SANITY]
    return predicate_rows + argument_rows


def _select_anchor(object_class: str, intended_step_key: str) -> dict[str, Any]:
    anchors = list(p20.get_stage20_synthetic_anchor_specs(object_class=object_class))
    if object_class == GEOMETRY_INTERACTION_OBJECT_CLASS:
        return dict(next(anchor for anchor in anchors if str(anchor.get("decoder_step_key")) == ("predicate_value" if intended_step_key == "predicate_value" else "agent_value")))
    if object_class == GEOMETRY_CROSS_STEP_BRIDGE_OBJECT_CLASS:
        target_key = "theme_value" if intended_step_key == "theme_value" else "agent_value"
        return dict(next(anchor for anchor in anchors if str(anchor.get("target_step_key")) == target_key))
    return dict(next(anchor for anchor in anchors if str(anchor.get("query_step_key")) == ("predicate_value" if intended_step_key == "predicate_value" else "agent_value")))


def summarize_h2(
    *,
    root: Path,
    manifest: dict[str, Any],
    observability_summary: dict[str, Any],
    active_stage_by_class: dict[str, str],
    max_length: int,
) -> dict[str, Any]:
    obs_by_class = {str(row["object_class"]): row for row in list(observability_summary.get("class_rows") or [])}
    class_rows: list[dict[str, Any]] = []
    for class_row in list(manifest.get("class_rows") or []):
        object_class = str(class_row["object_class"])
        obs = obs_by_class[object_class]
        stage_name = active_stage_by_class.get(object_class, STAGE_A)
        target_rows = list(class_row["stage_rows"][stage_name]["target_rows"])
        if not bool(obs.get("passed")):
            class_rows.append(
                {
                    "object_class": object_class,
                    "passed": False,
                    "stage": stage_name,
                    "null_passed_count": 0,
                    "failure_reason": "observability_failed",
                    "positive_rows": [],
                    "null_rows": [],
                }
            )
            continue
        positive_rows = []
        null_rows = []
        for target in target_rows:
            family = str(target["panel_regime"])
            if family not in {FAMILY_ARGUMENT_H2, FAMILY_ARGUMENT_NULL}:
                continue
            bundle = _bundle_from_prefix(root, str(target["phase11_prefix"]))
            anchor = _select_anchor(object_class, str(target["intended_step_key"]))
            row = {
                "run_name": str(target["run_name"]),
                "truthfulness_seed": int(target["truthfulness_seed"]),
                **p21._evaluate_model_native_h2_bundle(
                    bundle=bundle,
                    object_class=object_class,
                    object_spec=anchor,
                    recipe=p21._model_native_recipe(
                        object_spec=anchor,
                        panel_regime="planted_h2",
                        base_run=str(target["phase11_prefix"]),
                    ),
                    aligner_name="A1",
                    max_length=max_length,
                ),
            }
            if family == FAMILY_ARGUMENT_H2:
                positive_rows.append(row)
            else:
                null_rows.append(row)
        class_rows.append(
            {
                "object_class": object_class,
                "stage": stage_name,
                "passed": bool(positive_rows) and all(bool(row["passed"]) for row in positive_rows) and not any(bool(row["passed"]) for row in null_rows),
                "null_passed_count": sum(1 for row in null_rows if bool(row.get("passed"))),
                "positive_rows": positive_rows,
                "null_rows": null_rows,
            }
        )
    return {"class_rows": class_rows, "passed_count": sum(1 for row in class_rows if bool(row["passed"]))}


def summarize_h1(
    *,
    root: Path,
    manifest: dict[str, Any],
    observability_summary: dict[str, Any],
    active_stage_by_class: dict[str, str],
    max_length: int,
) -> dict[str, Any]:
    obs_by_class = {str(row["object_class"]): row for row in list(observability_summary.get("class_rows") or [])}
    class_rows: list[dict[str, Any]] = []
    for class_row in list(manifest.get("class_rows") or []):
        object_class = str(class_row["object_class"])
        obs = obs_by_class[object_class]
        stage_name = active_stage_by_class.get(object_class, STAGE_A)
        target_rows = list(class_row["stage_rows"][stage_name]["target_rows"])
        if not bool(obs.get("passed")):
            class_rows.append(
                {
                    "object_class": object_class,
                    "positive_passed": False,
                    "stage": stage_name,
                    "null_false_positive_rate": 0.0,
                    "failure_reason": "observability_failed",
                    "positive_rows": [],
                    "null_rows": [],
                }
            )
            continue
        positive_rows = []
        null_rows = []
        for target in target_rows:
            family = str(target["panel_regime"])
            if family not in {FAMILY_ARGUMENT_H1, FAMILY_ARGUMENT_NULL}:
                continue
            bundle = _bundle_from_prefix(root, str(target["phase11_prefix"]))
            anchor = _select_anchor(object_class, str(target["intended_step_key"]))
            row = {
                "run_name": str(target["run_name"]),
                "truthfulness_seed": int(target["truthfulness_seed"]),
                **p21._evaluate_model_native_h1_run(
                    bundle=bundle,
                    object_class=object_class,
                    object_spec=anchor,
                    recipe=p21._model_native_recipe(
                        object_spec=anchor,
                        panel_regime="planted_h1",
                        base_run=str(target["phase11_prefix"]),
                    ),
                    aligner_name="A1",
                    physical_family=str(target["run_name"]),
                    max_length=max_length,
                ),
            }
            if family == FAMILY_ARGUMENT_H1:
                positive_rows.append(row)
            else:
                null_rows.append(row)
        null_rate = round(mean(int(bool(row.get("run_recovered"))) for row in null_rows), 6) if null_rows else 0.0
        class_rows.append(
            {
                "object_class": object_class,
                "stage": stage_name,
                "positive_passed": bool(positive_rows) and all(bool(row["run_recovered"]) for row in positive_rows),
                "null_passed": bool(null_rate > 0.2),
                "null_false_positive_rate": null_rate,
                "positive_rows": positive_rows,
                "null_rows": null_rows,
            }
        )
    return {"class_rows": class_rows, "positive_passed_count": sum(1 for row in class_rows if bool(row["positive_passed"]))}


def summarize_stage_b(
    *,
    family_validity_summary: dict[str, Any],
    anchorability_summary: dict[str, Any],
    observability_summary: dict[str, Any],
    h2_summary: dict[str, Any],
    h1_summary: dict[str, Any],
) -> dict[str, Any]:
    failed_classes: list[str] = []
    by_validity = {str(row["object_class"]): row for row in list(family_validity_summary.get("class_rows") or [])}
    by_anchor = {str(row["object_class"]): row for row in list(anchorability_summary.get("class_rows") or [])}
    by_obs = {str(row["object_class"]): row for row in list(observability_summary.get("class_rows") or [])}
    by_h2 = {str(row["object_class"]): row for row in list(h2_summary.get("class_rows") or [])}
    by_h1 = {str(row["object_class"]): row for row in list(h1_summary.get("class_rows") or [])}
    for object_class in CLASS_ORDER:
        if not bool(by_validity.get(object_class, {}).get("passed")):
            failed_classes.append(object_class)
            continue
        if not bool(by_anchor.get(object_class, {}).get("passed")):
            failed_classes.append(object_class)
            continue
        if not bool(by_obs.get(object_class, {}).get("passed")):
            failed_classes.append(object_class)
            continue
        if not bool(by_h2.get(object_class, {}).get("passed")):
            failed_classes.append(object_class)
            continue
        if not bool(by_h1.get(object_class, {}).get("positive_passed")) or bool(by_h1.get(object_class, {}).get("null_passed")):
            failed_classes.append(object_class)
    return {
        "triggered": bool(failed_classes),
        "failed_classes": failed_classes,
        "stage_b_executed": False,
    }


def summarize_truthfulness_detector(
    *,
    family_validity_summary: dict[str, Any],
    anchorability_summary: dict[str, Any],
    observability_summary: dict[str, Any],
    h2_summary: dict[str, Any],
    h1_summary: dict[str, Any],
    real_regression_summary: dict[str, Any],
) -> dict[str, Any]:
    by_validity = {str(row["object_class"]): row for row in list(family_validity_summary.get("class_rows") or [])}
    by_anchor = {str(row["object_class"]): row for row in list(anchorability_summary.get("class_rows") or [])}
    by_obs = {str(row["object_class"]): row for row in list(observability_summary.get("class_rows") or [])}
    by_h2 = {str(row["object_class"]): row for row in list(h2_summary.get("class_rows") or [])}
    by_h1 = {str(row["object_class"]): row for row in list(h1_summary.get("class_rows") or [])}
    failures: list[str] = []
    for object_class in CLASS_ORDER:
        if not bool(by_validity.get(object_class, {}).get("passed")):
            failures.append(f"{object_class}:family_construction_failed")
        if not bool(by_anchor.get(object_class, {}).get("passed")):
            failures.append(f"{object_class}:anchorability_failed")
        if not bool(by_obs.get(object_class, {}).get("passed")):
            failures.append(f"{object_class}:observability_failed")
        if bool(by_obs.get(object_class, {}).get("argument_specific_observability_failure")):
            failures.append(f"{object_class}:argument_specific_observability_failure")
        if not bool(by_h2.get(object_class, {}).get("passed")):
            failures.append(f"{object_class}:argument_h2_failed")
        if not bool(by_h1.get(object_class, {}).get("positive_passed")):
            failures.append(f"{object_class}:argument_h1_failed")
        if bool(by_h1.get(object_class, {}).get("null_passed")):
            failures.append(f"{object_class}:null_false_positive")
    if not bool(real_regression_summary.get("preserved_phase18_t_final", True)):
        failures.append("real_family_regression_failed")
    validated = not failures
    return {
        "detectability_validated": validated,
        "decision": "support_ontology_final_upgrade" if validated else "retain_phase18",
        "decision_status": "final",
        "failure_reasons": failures,
        "upgraded_strongest_claim": validated,
    }


def render_detector_truthfulness_status_markdown(
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
    detector_summary: dict[str, Any],
) -> str:
    lines = [
        "# Phase 22 Status",
        "",
        "## Frozen Framing",
        "- `G1_binding`: bounded hypothesis-final success",
        "- `G2_distributed_geometry`: `Phase 18` family-final negative unless detector truthfulness is validated.",
        "- `Phase 22` validates the detector rather than reopening the real `G2` family.",
        "",
        "## Benchmarks",
        f"- `benchmark_count = {int(benchmark_summary.get('benchmark_count') or 0)}`",
    ]
    for row in list(benchmark_summary.get("benchmark_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `canonical_example_count = {int(row.get('canonical_example_count') or 0)}`, `coverage = {float(dict(row.get('selection') or {}).get('exact_counterpart_coverage_rate') or 0.0)}`"
        )
    lines.extend(["", "## Panels"])
    for row in list(manifest_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `stage_a_target_count = {len(list(row['stage_rows']['stage_a']['target_rows']))}`, `stage_b_target_count = {len(list(row['stage_rows']['stage_b']['target_rows']))}`"
        )
    lines.extend(["", "## Family Validity"])
    for row in list(family_validity_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `passed = {bool(row.get('passed'))}`, `stage = {row.get('stage')}`, `failure_reasons = {row.get('failure_reasons') or []}`"
        )
    lines.extend(["", "## Anchorability"])
    for row in list(anchorability_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `passed = {bool(row.get('passed'))}`, `usable_anchor_count = {int(row.get('usable_anchor_count') or 0)}`"
        )
    lines.extend(["", "## Observability"])
    for row in list(observability_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `passed = {bool(row.get('passed'))}`, `predicate_sanity_passed = {bool(row.get('predicate_sanity_passed'))}`, `argument_families_passed = {bool(row.get('argument_families_passed'))}`, `argument_specific_failure = {bool(row.get('argument_specific_observability_failure'))}`"
        )
    lines.extend(["", "## Argument H2"])
    for row in list(h2_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `passed = {bool(row.get('passed'))}`, `stage = {row.get('stage')}`, `null_passed_count = {int(row.get('null_passed_count') or 0)}`"
        )
    lines.extend(["", "## Argument H1"])
    for row in list(h1_summary.get("class_rows") or []):
        lines.append(
            f"- `{row['object_class']}`: `positive_passed = {bool(row.get('positive_passed'))}`, `stage = {row.get('stage')}`, `null_false_positive_rate = {float(row.get('null_false_positive_rate') or 0.0)}`"
        )
    lines.extend(
        [
            "",
            "## Stage B",
            f"- `triggered = {bool(stage_b_summary.get('triggered'))}`",
            f"- `failed_classes = {stage_b_summary.get('failed_classes') or []}`",
            f"- `stage_b_executed = {bool(stage_b_summary.get('stage_b_executed'))}`",
            "",
            "## Real-Family Regression",
            f"- `preserved_phase18_t_final = {bool(real_regression_summary.get('preserved_phase18_t_final'))}`",
            f"- `executed = {bool(real_regression_summary.get('executed'))}`",
            "",
            "## Detector Verdict",
            f"- `detectability_validated = {bool(detector_summary.get('detectability_validated'))}`",
            f"- `decision = {detector_summary.get('decision')}`",
            f"- `failure_reasons = {detector_summary.get('failure_reasons') or []}`",
        ]
    )
    return "\n".join(lines) + "\n"
