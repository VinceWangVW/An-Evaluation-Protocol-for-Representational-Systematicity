from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from .schema import CanonicalExample, SemanticQuartet


def build_semantic_quartets(examples: list[CanonicalExample]) -> list[SemanticQuartet]:
    index: dict[tuple[str, str, str, str], CanonicalExample] = {}
    grouped: dict[tuple[str, str], list[CanonicalExample]] = defaultdict(list)

    for example in examples:
        roles = example.role_signature
        if not {"agent", "theme", "predicate"} <= set(roles):
            continue
        key = (
            example.benchmark_name,
            example.split,
            roles["predicate"].lower(),
            roles["agent"].lower(),
            roles["theme"].lower(),
        )
        index[key] = example
        grouped[(example.benchmark_name, example.split)].append(example)

    quartets: list[SemanticQuartet] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()

    for (benchmark_name, split), split_examples in grouped.items():
        by_predicate: dict[str, list[CanonicalExample]] = defaultdict(list)
        for example in split_examples:
            by_predicate[example.role_signature["predicate"].lower()].append(example)

        for predicate, predicate_examples in by_predicate.items():
            agents = sorted({example.role_signature["agent"].lower() for example in predicate_examples})
            themes = sorted({example.role_signature["theme"].lower() for example in predicate_examples})

            for agent_a, agent_b in combinations(agents, 2):
                for theme_a, theme_b in combinations(themes, 2):
                    lookup_keys = [
                        (benchmark_name, split, predicate, agent_a, theme_a),
                        (benchmark_name, split, predicate, agent_a, theme_b),
                        (benchmark_name, split, predicate, agent_b, theme_a),
                        (benchmark_name, split, predicate, agent_b, theme_b),
                    ]
                    if not all(key in index for key in lookup_keys):
                        continue

                    quartet_key = (benchmark_name, split, predicate, agent_a, agent_b, theme_a, theme_b)
                    if quartet_key in seen:
                        continue
                    seen.add(quartet_key)

                    quartet_examples = [index[key] for key in lookup_keys]
                    quartets.append(
                        SemanticQuartet(
                            quartet_id=(
                                f"{benchmark_name}:{split}:{predicate}:{agent_a}:{agent_b}:{theme_a}:{theme_b}"
                            ),
                            benchmark_name=benchmark_name,
                            predicate=predicate,
                            agent_a=agent_a,
                            agent_b=agent_b,
                            theme_a=theme_a,
                            theme_b=theme_b,
                            item_ids=tuple(example.item_id for example in quartet_examples),
                            split_pattern=tuple(example.split for example in quartet_examples),
                        )
                    )

    return quartets
