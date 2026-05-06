from __future__ import annotations

import re
from collections.abc import Mapping


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
DETERMINERS = {"a", "an", "the"}
AUXILIARIES = {"was", "were", "is", "are"}
PREPOSITIONS = {"by", "to", "from", "with", "of", "for", "in", "on", "at"}
ATOM_SPLIT_RE = re.compile(r"\s+(?:AND|;)\s+")
UNARY_ATOM_RE = re.compile(r"^\*?\s*(.+?)\s*\(\s*(.+?)\s*\)\s*$")
ROLE_ATOM_RE = re.compile(r"^\*?\s*(.+?)\s*\.\s*(agent|theme|recipient)\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)\s*$")
VAR_RE = re.compile(r"^x\s*_*\s*\d+$", flags=re.IGNORECASE)
FUNCTIONAL_TARGET_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*\((.*)\)\s*$")


def normalize_split_name(raw_split: str) -> str:
    key = raw_split.strip().lower()
    mapping = {
        "train": "train",
        "training": "train",
        "dev": "dev",
        "valid": "dev",
        "validation": "dev",
        "test": "test",
        "gen": "gen",
        "generalization": "gen",
    }
    return mapping.get(key, key or "unknown")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def normalize_token(token: str) -> str:
    return token.lower()


def infer_generalization_case(
    value: str | None, metadata: Mapping[str, object] | None = None
) -> str:
    if value:
        cleaned = normalize_whitespace(str(value))
        if cleaned:
            return cleaned

    if metadata:
        for key in (
            "generalization_case",
            "distribution",
            "distribution_type",
            "gen_type",
            "type",
        ):
            if key in metadata and metadata[key] is not None:
                cleaned = normalize_whitespace(str(metadata[key]))
                if cleaned:
                    return cleaned

    return "unspecified"


def infer_role_signature(input_text: str, target_form: str = "") -> dict[str, str]:
    parsed_from_target = infer_role_signature_from_target_form(target_form)
    if parsed_from_target:
        heuristic = infer_surface_role_signature(input_text)
        return {
            "frame": parsed_from_target.get("frame", heuristic.get("frame", "parsed_from_target")),
            "surface_subject": heuristic.get("surface_subject", ""),
            "surface_object": heuristic.get("surface_object", ""),
            **parsed_from_target,
        }
    return infer_surface_role_signature(input_text)


def infer_surface_role_signature(input_text: str) -> dict[str, str]:
    tokens = tokenize_words(input_text)
    lowered = [normalize_token(token) for token in tokens]

    # Pattern: "Emma rolled a teacher ."
    if len(tokens) == 4 and lowered[2] in DETERMINERS:
        return {
            "frame": "simple_active",
            "surface_subject": tokens[0],
            "surface_object": tokens[3],
            "predicate": tokens[1],
            "agent": tokens[0],
            "theme": tokens[3],
        }

    # Pattern: "Emma rolled John ."
    if len(tokens) == 3:
        return {
            "frame": "simple_active",
            "surface_subject": tokens[0],
            "surface_object": tokens[2],
            "predicate": tokens[1],
            "agent": tokens[0],
            "theme": tokens[2],
        }

    # Pattern: "The sailor dusted a boy ."
    if len(tokens) == 5 and lowered[0] in DETERMINERS and lowered[3] in DETERMINERS:
        return {
            "frame": "simple_active",
            "surface_subject": tokens[1],
            "surface_object": tokens[4],
            "predicate": tokens[2],
            "agent": tokens[1],
            "theme": tokens[4],
        }

    # Pattern: "A rose was helped by a dog ."
    if (
        len(tokens) == 7
        and lowered[0] in DETERMINERS
        and lowered[2] in AUXILIARIES
        and lowered[4] == "by"
        and lowered[5] in DETERMINERS
    ):
        return {
            "frame": "simple_passive",
            "surface_subject": tokens[1],
            "surface_object": tokens[6],
            "predicate": tokens[3],
            "agent": tokens[6],
            "theme": tokens[1],
        }

    return {}


def infer_role_signature_from_target_form(target_form: str) -> dict[str, str]:
    functional_signature = infer_role_signature_from_functional_target(target_form)
    if functional_signature:
        return functional_signature

    atoms = [piece.strip() for piece in ATOM_SPLIT_RE.split(normalize_whitespace(target_form)) if piece.strip()]
    if not atoms:
        return {}

    entity_by_var: dict[str, str] = {}
    role_atoms: list[dict[str, str]] = []

    for atom in atoms:
        role_match = ROLE_ATOM_RE.match(atom)
        if role_match:
            predicate, role_name, event_var, filler = role_match.groups()
            role_atoms.append(
                {
                    "predicate": normalize_whitespace(predicate),
                    "role": normalize_whitespace(role_name).lower(),
                    "event_var": normalize_var(event_var),
                    "filler": normalize_whitespace(filler),
                }
            )
            continue

        unary_match = UNARY_ATOM_RE.match(atom)
        if not unary_match:
            continue
        predicate, argument = unary_match.groups()
        normalized_argument = normalize_var(argument)
        if is_variable(normalized_argument):
            entity_by_var.setdefault(normalized_argument, normalize_whitespace(predicate))

    if not role_atoms:
        return {}

    grouped: dict[tuple[str, str], dict[str, str]] = {}
    ordered_keys: list[tuple[str, str]] = []
    for atom in role_atoms:
        key = (atom["predicate"], atom["event_var"])
        if key not in grouped:
            grouped[key] = {}
            ordered_keys.append(key)
        grouped[key][atom["role"]] = resolve_filler(atom["filler"], entity_by_var)

    chosen_key: tuple[str, str] | None = None
    for key in ordered_keys:
        roles = grouped[key]
        if {"agent", "theme"} <= set(roles):
            chosen_key = key
            break
    if chosen_key is None:
        for key in ordered_keys:
            roles = grouped[key]
            if {"agent", "theme", "recipient"} & set(roles):
                chosen_key = key
                break
    if chosen_key is None:
        return {}

    predicate, _ = chosen_key
    roles = grouped[chosen_key]
    signature = {
        "frame": "parsed_from_target",
        "predicate": normalize_whitespace(predicate),
    }
    for role_name in ("agent", "theme", "recipient"):
        value = roles.get(role_name)
        if value:
            signature[role_name] = value
    return signature


def infer_role_signature_from_functional_target(target_form: str) -> dict[str, str]:
    normalized = normalize_whitespace(target_form)
    match = FUNCTIONAL_TARGET_RE.match(normalized)
    if not match:
        return {}

    predicate, payload = match.groups()
    if "=" not in payload:
        return {}

    role_map: dict[str, str] = {}
    for piece in payload.split(","):
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        cleaned_key = normalize_whitespace(key).lower()
        cleaned_value = normalize_whitespace(value)
        if cleaned_key in {"agent", "theme", "recipient"} and cleaned_value:
            role_map[cleaned_key] = cleaned_value

    if not {"agent", "theme"} & set(role_map):
        return {}

    signature = {
        "frame": "parsed_from_functional_target",
        "predicate": normalize_whitespace(predicate),
    }
    signature.update(role_map)
    return signature


def normalize_var(value: str) -> str:
    compact = normalize_whitespace(value).replace(" ", "")
    return compact.lower()


def is_variable(value: str) -> bool:
    return bool(VAR_RE.fullmatch(value))


def resolve_filler(value: str, entity_by_var: Mapping[str, str]) -> str:
    normalized_value = normalize_var(value)
    if is_variable(normalized_value):
        return entity_by_var.get(normalized_value, normalize_whitespace(value))
    return normalize_whitespace(value)


def infer_lexical_signature(input_text: str, role_signature: Mapping[str, str]) -> tuple[str, ...]:
    if role_signature:
        values = [
            normalize_token(value)
            for key, value in role_signature.items()
            if key in {"agent", "theme", "predicate"}
        ]
        return tuple(sorted(values))

    content_tokens = []
    for token in tokenize_words(input_text):
        lowered = normalize_token(token)
        if lowered in DETERMINERS or lowered in AUXILIARIES or lowered in PREPOSITIONS:
            continue
        content_tokens.append(lowered)
    return tuple(sorted(content_tokens))


def infer_surface_features(input_text: str, target_form: str) -> dict[str, object]:
    input_tokens = tokenize_words(input_text)
    target_tokens = tokenize_words(target_form)
    unique_input = {normalize_token(token) for token in input_tokens}
    unique_target = {normalize_token(token) for token in target_tokens}

    return {
        "input_length_tokens": len(input_tokens),
        "target_length_tokens": len(target_tokens),
        "input_length_chars": len(input_text),
        "target_length_chars": len(target_form),
        "shared_token_count": len(unique_input & unique_target),
    }
