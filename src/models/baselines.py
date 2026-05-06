from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


def tokenize(text: str) -> list[str]:
    return [token.strip(".,;:!?").lower() for token in text.split() if token.strip(".,;:!?")]


class BaseBehaviorModel:
    family: str = "base"

    def fit(self, train_examples: list[dict[str, Any]]) -> None:
        self.train_examples = train_examples

    def predict_many(self, eval_examples: list[dict[str, Any]]) -> list[str]:
        return [self.predict_one(example) for example in eval_examples]

    def predict_one(self, example: dict[str, Any]) -> str:
        raise NotImplementedError


class ExactLookupBaseline(BaseBehaviorModel):
    family = "memorization_lookup"

    def fit(self, train_examples: list[dict[str, Any]]) -> None:
        super().fit(train_examples)
        self.index = {example["input_text"]: example["target_form"] for example in train_examples}

    def predict_one(self, example: dict[str, Any]) -> str:
        return self.index.get(example["input_text"], "")


class TokenOverlapRetrievalBaseline(BaseBehaviorModel):
    family = "lexical_retrieval"

    def fit(self, train_examples: list[dict[str, Any]]) -> None:
        super().fit(train_examples)
        self.index = []
        self.postings: dict[str, list[int]] = {}
        for example in train_examples:
            example_tokens = Counter(tokenize(example["input_text"]))
            idx = len(self.index)
            self.index.append(
                {
                    "input_text": example["input_text"],
                    "target_form": example["target_form"],
                    "tokens": example_tokens,
                    "length": sum(example_tokens.values()),
                }
            )
            for token in example_tokens:
                self.postings.setdefault(token, []).append(idx)

    def predict_one(self, example: dict[str, Any]) -> str:
        eval_tokens = Counter(tokenize(example["input_text"]))
        eval_length = sum(eval_tokens.values())
        candidate_indices = self._candidate_indices(eval_tokens)

        best_score = None
        best_target = ""
        for candidate_index in candidate_indices:
            candidate = self.index[candidate_index]
            overlap = sum((eval_tokens & candidate["tokens"]).values())
            union = sum((eval_tokens | candidate["tokens"]).values()) or 1
            jaccard = overlap / union
            length_gap = abs(eval_length - candidate["length"])
            score = (jaccard, overlap, -length_gap)
            if best_score is None or score > best_score:
                best_score = score
                best_target = candidate["target_form"]
        return best_target

    def _candidate_indices(self, eval_tokens: Counter[str]) -> list[int]:
        ranked_tokens = sorted(
            eval_tokens,
            key=lambda token: (len(self.postings.get(token, [])), token),
        )

        candidate_set: set[int] = set()
        for token in ranked_tokens[:3]:
            candidate_set.update(self.postings.get(token, []))

        if candidate_set:
            return list(candidate_set)
        return list(range(len(self.index)))


@dataclass(slots=True)
class SimpleTemplateBaseline(BaseBehaviorModel):
    benchmark_name: str
    family: str = "symbolic_template"

    def predict_one(self, example: dict[str, Any]) -> str:
        roles = example.get("role_signature") or {}
        if not {"agent", "theme", "predicate", "frame"} <= set(roles):
            return ""
        if self.benchmark_name != "cogs_official":
            return ""

        predicate_surface = str(roles["predicate"])
        predicate = lemmatize_predicate(predicate_surface)
        agent = str(roles["agent"])
        theme = str(roles["theme"])
        frame = str(roles["frame"])

        if frame == "simple_active":
            return render_cogs_simple(predicate=predicate, agent=agent, theme=theme, star_subject=True)
        if frame == "simple_passive":
            return render_cogs_simple(predicate=predicate, agent=agent, theme=theme, star_subject=False)
        return ""


def lemmatize_predicate(surface: str) -> str:
    token = surface.lower()
    if token.endswith("ied") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("ed") and len(token) > 2:
        stem = token[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    return token


def is_proper(token: str) -> bool:
    return token[:1].isupper()


def render_entity_clause(token: str, variable: str) -> str:
    if is_proper(token):
        return ""
    return f"{token.lower()} ( {variable} )"


def render_argument(token: str, variable: str) -> str:
    return token if is_proper(token) else variable


def render_cogs_simple(predicate: str, agent: str, theme: str, star_subject: bool) -> str:
    agent_var = "x _ 1"
    event_var = "x _ 2"
    theme_var = "x _ 4"

    subject_clause = render_entity_clause(agent, agent_var)
    theme_clause = render_entity_clause(theme, theme_var)
    prefix = "* " if star_subject and subject_clause else ""
    left = f"{prefix}{subject_clause}".strip()
    clauses = []
    if left:
        clauses.append(left)

    clauses.append(f"{predicate} . agent ( {event_var} , {render_argument(agent, agent_var)} )")
    clauses.append(f"{predicate} . theme ( {event_var} , {render_argument(theme, theme_var)} )")

    if theme_clause:
        clauses.append(theme_clause)

    if not clauses:
        return ""

    head = clauses[0]
    tail = clauses[1:]
    if not tail:
        return head
    return f"{head} ; " + " AND ".join(tail)


def build_baseline(name: str, benchmark_name: str) -> BaseBehaviorModel:
    if name == "exact_lookup":
        return ExactLookupBaseline()
    if name == "token_overlap":
        return TokenOverlapRetrievalBaseline()
    if name == "simple_template":
        return SimpleTemplateBaseline(benchmark_name=benchmark_name)
    raise ValueError(f"Unknown baseline: {name}")
