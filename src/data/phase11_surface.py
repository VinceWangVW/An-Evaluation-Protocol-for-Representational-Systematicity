from __future__ import annotations

import json
from typing import Any

from .phase11_schema import Phase11SchemaRecord


def build_realization_prompt(record: Phase11SchemaRecord, num_candidates: int) -> str:
    voice_instruction = (
        "Use active voice so the agent appears before the predicate and the theme appears after it."
        if record.voice == "active"
        else "Use passive voice so the theme appears before the predicate and the agent appears in a by-phrase."
    )
    depth_instruction = (
        f"Include a relative clause using the context filler '{record.context_filler}' and the context predicate "
        f"'{record.context_predicate_lemma}'."
        if record.embedding_depth > 0
        else "Do not add a relative clause."
    )
    return (
        "Return strict JSON with a top-level key 'candidates' containing a list of objects. "
        "Each object must contain a 'text' field only.\n"
        f"Generate {num_candidates} natural-language realizations.\n"
        f"Predicate lemma: {record.predicate_lemma}\n"
        f"Agent filler: {record.agent}\n"
        f"Theme filler: {record.theme}\n"
        f"Agent modifier: {record.agent_modifier}\n"
        f"Theme modifier: {record.theme_modifier}\n"
        f"{voice_instruction}\n"
        f"{depth_instruction}\n"
        "Do not change fillers. Do not use pronouns. Preserve the semantic roles exactly."
    )


def render_template_candidates(record: Phase11SchemaRecord, num_candidates: int = 3) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for variant_index in range(num_candidates):
        candidates.append(
            {
                "candidate_id": f"{record.item_id}::template_{variant_index}",
                "item_id": record.item_id,
                "backend": "template",
                "text": _render_template(record, variant_index),
                "template_variant": variant_index,
            }
        )
    return candidates


def realize_with_nuwa(
    *,
    record: Phase11SchemaRecord,
    client: Any,
    model: str,
    num_candidates: int = 3,
    timeout: float | None = 60.0,
) -> list[dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": build_realization_prompt(record, num_candidates=num_candidates),
            }
        ],
        timeout=timeout,
    )
    content = response.choices[0].message.content if response.choices else ""
    payload = _parse_candidate_payload(content)
    candidates = payload.get("candidates", [])[:num_candidates]
    realized: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        text = str(candidate.get("text", "")).strip()
        if not text:
            continue
        realized.append(
            {
                "candidate_id": f"{record.item_id}::nuwa_{candidate_index}",
                "item_id": record.item_id,
                "backend": "nuwa",
                "text": text,
                "model_name": model,
            }
        )
    return realized


def _render_template(record: Phase11SchemaRecord, variant_index: int) -> str:
    if record.voice == "active":
        base_variants = (
            f"The {record.agent_modifier} {record.agent} {record.predicate_past} the {record.theme_modifier} {record.theme}.",
            f"{record.agent}, the {record.agent_modifier} participant, {record.predicate_past} {record.theme}, the {record.theme_modifier} participant.",
            f"The {record.agent_modifier} {record.agent} clearly {record.predicate_past} the {record.theme_modifier} {record.theme}.",
        )
        relative_variants = (
            f"The {record.agent_modifier} {record.agent} who {record.context_predicate_past} the {record.context_filler} {record.predicate_past} the {record.theme_modifier} {record.theme}.",
            f"{record.agent}, the {record.agent_modifier} participant who {record.context_predicate_past} the {record.context_filler}, {record.predicate_past} {record.theme}, the {record.theme_modifier} participant.",
            f"The {record.agent_modifier} {record.agent} who quietly {record.context_predicate_past} the {record.context_filler} clearly {record.predicate_past} the {record.theme_modifier} {record.theme}.",
        )
    else:
        base_variants = (
            f"The {record.theme_modifier} {record.theme} was {record.predicate_participle} by the {record.agent_modifier} {record.agent}.",
            f"{record.theme}, the {record.theme_modifier} participant, was {record.predicate_participle} by {record.agent}, the {record.agent_modifier} participant.",
            f"The {record.theme_modifier} {record.theme} was clearly {record.predicate_participle} by the {record.agent_modifier} {record.agent}.",
        )
        relative_variants = (
            f"The {record.theme_modifier} {record.theme} was {record.predicate_participle} by the {record.agent_modifier} {record.agent} who {record.context_predicate_past} the {record.context_filler}.",
            f"{record.theme}, the {record.theme_modifier} participant, was {record.predicate_participle} by {record.agent}, the {record.agent_modifier} participant who {record.context_predicate_past} the {record.context_filler}.",
            f"The {record.theme_modifier} {record.theme} was clearly {record.predicate_participle} by the {record.agent_modifier} {record.agent} who quietly {record.context_predicate_past} the {record.context_filler}.",
        )
    variants = relative_variants if record.embedding_depth > 0 else base_variants
    return variants[variant_index % len(variants)]


def _parse_candidate_payload(content: str) -> dict[str, Any]:
    text = content.strip()
    if not text:
        return {"candidates": []}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    lines = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
    return {"candidates": [{"text": line} for line in lines]}
