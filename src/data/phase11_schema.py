from __future__ import annotations

from dataclasses import asdict, dataclass, field
import random
from typing import Any, Iterable


BENCHMARK_NAME = "phase11_systematicity_v1"
REQUIRED_COUNTERPART_TYPES = (
    "rescue_donor",
    "rescue_target",
    "agent_swap",
    "theme_swap",
    "role_swap",
    "filler_swap",
    "modifier_swap",
)

PREDICATES = (
    {"lemma": "admire", "past": "admired", "participle": "admired"},
    {"lemma": "follow", "past": "followed", "participle": "followed"},
    {"lemma": "help", "past": "helped", "participle": "helped"},
    {"lemma": "support", "past": "supported", "participle": "supported"},
    {"lemma": "contact", "past": "contacted", "participle": "contacted"},
)
CONTEXT_PREDICATES = (
    {"lemma": "notice", "past": "noticed", "participle": "noticed"},
    {"lemma": "visit", "past": "visited", "participle": "visited"},
    {"lemma": "consult", "past": "consulted", "participle": "consulted"},
)
FILLERS = (
    "Aiden",
    "Bianca",
    "Caleb",
    "Diana",
    "Ethan",
    "Freya",
    "Gavin",
    "Hazel",
    "Iris",
    "Jonah",
    "Kira",
    "Luca",
    "Maya",
    "Nolan",
    "Opal",
    "Piper",
)
MODIFIERS = (
    "curious",
    "patient",
    "brisk",
    "careful",
    "steady",
    "vivid",
    "calm",
    "focused",
    "gentle",
    "nimble",
)
CONTEXT_FILLERS = (
    "witness",
    "neighbor",
    "guide",
    "pilot",
    "scholar",
    "artist",
)

CONSTRUCTION_PLANS = {
    "train": (
        {"family": "active_depth0", "voice": "active", "depth": 0},
        {"family": "passive_depth0", "voice": "passive", "depth": 0},
        {"family": "active_depth1", "voice": "active", "depth": 1},
    ),
    "dev": (
        {"family": "active_depth0", "voice": "active", "depth": 0},
        {"family": "passive_depth0", "voice": "passive", "depth": 0},
        {"family": "active_depth1", "voice": "active", "depth": 1},
    ),
    "gen": (
        {"family": "passive_depth1", "voice": "passive", "depth": 1},
        {"family": "active_depth1", "voice": "active", "depth": 1},
        {"family": "passive_depth0", "voice": "passive", "depth": 0},
    ),
}


@dataclass(slots=True)
class Phase11SchemaRecord:
    item_id: str
    benchmark_name: str
    split: str
    raw_split: str
    schema_id: str
    counterpart_family_id: str
    counterpart_type: str
    construction_family: str
    voice: str
    embedding_depth: int
    predicate_lemma: str
    predicate_past: str
    predicate_participle: str
    context_predicate_lemma: str
    context_predicate_past: str
    agent: str
    theme: str
    agent_modifier: str
    theme_modifier: str
    context_filler: str
    target_form: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_target_form(
    *,
    predicate_lemma: str,
    agent: str,
    theme: str,
    agent_modifier: str,
    theme_modifier: str,
) -> str:
    return (
        f"{predicate_lemma}("
        f"agent={agent.lower()},"
        f"theme={theme.lower()},"
        f"agent_mod={agent_modifier},"
        f"theme_mod={theme_modifier}"
        f")"
    )


def generate_schema_records(
    *,
    train_families: int,
    dev_families: int,
    gen_families: int,
    seed: int = 13,
    benchmark_name: str = BENCHMARK_NAME,
) -> list[Phase11SchemaRecord]:
    rng = random.Random(seed)
    records: list[Phase11SchemaRecord] = []
    for split_name, family_count in (("train", train_families), ("dev", dev_families), ("gen", gen_families)):
        plans = CONSTRUCTION_PLANS[split_name]
        for family_index in range(family_count):
            plan = plans[family_index % len(plans)]
            family_id = f"{benchmark_name}:{split_name}:family_{family_index:05d}"
            records.extend(
                _build_family_records(
                    family_id=family_id,
                    benchmark_name=benchmark_name,
                    family_split=split_name,
                    family_plan=plan,
                    rng=rng,
                )
            )
    return records


def build_counterpart_families(records: Iterable[Phase11SchemaRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Phase11SchemaRecord]] = {}
    for record in records:
        grouped.setdefault(record.counterpart_family_id, []).append(record)

    families: list[dict[str, Any]] = []
    for family_id, members in sorted(grouped.items()):
        by_type = {member.counterpart_type: member for member in members}
        families.append(
            {
                "counterpart_family_id": family_id,
                "benchmark_name": members[0].benchmark_name,
                "construction_family": members[0].construction_family,
                "required_counterpart_coverage": all(
                    counterpart_type in by_type for counterpart_type in REQUIRED_COUNTERPART_TYPES
                ),
                "members": {key: value.item_id for key, value in by_type.items()},
                "m2_pairs": [
                    {
                        "source_item_id": by_type["rescue_target"].item_id,
                        "counterpart_item_id": by_type["agent_swap"].item_id,
                        "relation": "agent_swap",
                    },
                    {
                        "source_item_id": by_type["rescue_target"].item_id,
                        "counterpart_item_id": by_type["theme_swap"].item_id,
                        "relation": "theme_swap",
                    },
                ],
                "m4_pair": {
                    "target_item_id": by_type["rescue_target"].item_id,
                    "donor_item_id": by_type["rescue_donor"].item_id,
                },
                "donor_target_lf_match": (
                    by_type["rescue_donor"].target_form == by_type["rescue_target"].target_form
                ),
            }
        )
    return families


def summarize_schema_records(records: Iterable[Phase11SchemaRecord]) -> dict[str, Any]:
    records = list(records)
    families = build_counterpart_families(records)
    by_split: dict[str, int] = {}
    by_counterpart_type: dict[str, int] = {}
    by_construction_family: dict[str, int] = {}
    for record in records:
        by_split[record.split] = by_split.get(record.split, 0) + 1
        by_counterpart_type[record.counterpart_type] = by_counterpart_type.get(record.counterpart_type, 0) + 1
        by_construction_family[record.construction_family] = (
            by_construction_family.get(record.construction_family, 0) + 1
        )
    return {
        "benchmark_name": records[0].benchmark_name if records else BENCHMARK_NAME,
        "record_count": len(records),
        "family_count": len(families),
        "required_counterpart_types": list(REQUIRED_COUNTERPART_TYPES),
        "split_counts": by_split,
        "counterpart_type_counts": by_counterpart_type,
        "construction_family_counts": by_construction_family,
        "exact_counterpart_coverage_rate": (
            sum(1 for family in families if family["required_counterpart_coverage"]) / len(families)
            if families
            else 0.0
        ),
        "donor_target_lf_match_rate": (
            sum(1 for family in families if family["donor_target_lf_match"]) / len(families)
            if families
            else 0.0
        ),
    }


def _build_family_records(
    *,
    family_id: str,
    benchmark_name: str,
    family_split: str,
    family_plan: dict[str, Any],
    rng: random.Random,
) -> list[Phase11SchemaRecord]:
    predicate = rng.choice(PREDICATES)
    context_predicate = rng.choice(CONTEXT_PREDICATES)
    filler_pool = rng.sample(FILLERS, 4)
    modifier_pool = rng.sample(MODIFIERS, 2)
    context_filler = rng.choice(CONTEXT_FILLERS)

    base_agent, base_theme, alt_agent, alt_theme = filler_pool
    agent_modifier, theme_modifier = modifier_pool
    target_voice = str(family_plan["voice"])
    target_depth = int(family_plan["depth"])
    construction_family = str(family_plan["family"])

    item_specs = (
        {
            "counterpart_type": "rescue_donor",
            "split": "train" if family_split == "train" else "dev",
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
            "split": family_split,
            "voice": target_voice,
            "depth": target_depth,
            "agent": base_agent,
            "theme": base_theme,
            "agent_modifier": agent_modifier,
            "theme_modifier": theme_modifier,
            "allowed_delta": "hard_target",
        },
        {
            "counterpart_type": "role_swap",
            "split": family_split,
            "voice": target_voice,
            "depth": target_depth,
            "agent": base_theme,
            "theme": base_agent,
            "agent_modifier": theme_modifier,
            "theme_modifier": agent_modifier,
            "allowed_delta": "role_swap",
        },
        {
            "counterpart_type": "agent_swap",
            "split": family_split,
            "voice": target_voice,
            "depth": target_depth,
            "agent": alt_agent,
            "theme": base_theme,
            "agent_modifier": agent_modifier,
            "theme_modifier": theme_modifier,
            "allowed_delta": "agent_swap",
        },
        {
            "counterpart_type": "theme_swap",
            "split": family_split,
            "voice": target_voice,
            "depth": target_depth,
            "agent": base_agent,
            "theme": alt_theme,
            "agent_modifier": agent_modifier,
            "theme_modifier": theme_modifier,
            "allowed_delta": "theme_swap",
        },
        {
            "counterpart_type": "filler_swap",
            "split": family_split,
            "voice": target_voice,
            "depth": target_depth,
            "agent": alt_agent,
            "theme": alt_theme,
            "agent_modifier": agent_modifier,
            "theme_modifier": theme_modifier,
            "allowed_delta": "filler_swap",
        },
        {
            "counterpart_type": "modifier_swap",
            "split": family_split,
            "voice": target_voice,
            "depth": target_depth,
            "agent": base_agent,
            "theme": base_theme,
            "agent_modifier": theme_modifier,
            "theme_modifier": agent_modifier,
            "allowed_delta": "modifier_swap",
        },
    )

    records: list[Phase11SchemaRecord] = []
    for spec in item_specs:
        schema_id = f"{family_id}:{spec['counterpart_type']}"
        target_form = build_target_form(
            predicate_lemma=predicate["lemma"],
            agent=str(spec["agent"]),
            theme=str(spec["theme"]),
            agent_modifier=str(spec["agent_modifier"]),
            theme_modifier=str(spec["theme_modifier"]),
        )
        records.append(
            Phase11SchemaRecord(
                item_id=schema_id,
                benchmark_name=benchmark_name,
                split=str(spec["split"]),
                raw_split=str(spec["split"]),
                schema_id=schema_id,
                counterpart_family_id=family_id,
                counterpart_type=str(spec["counterpart_type"]),
                construction_family=construction_family,
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
                    "family_split": family_split,
                    "allowed_delta": spec["allowed_delta"],
                    "no_forced_h2": True,
                },
            )
        )
    return records
