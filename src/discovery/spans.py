from __future__ import annotations

import re
from typing import Any


ROLE_KEYS = ("agent", "theme", "recipient")
_IRREGULAR_PREDICATES = {
    "be": {"am", "is", "are", "was", "were", "been", "being"},
    "buy": {"bought", "buying", "buys"},
    "catch": {"caught", "catching", "catches"},
    "choose": {"chose", "chosen", "choosing", "chooses"},
    "collapse": {"collapsed", "collapsing", "collapses"},
    "cook": {"cooked", "cooking", "cooks"},
    "draw": {"drew", "drawn", "drawing", "draws"},
    "eat": {"ate", "eaten", "eating", "eats"},
    "find": {"found", "finding", "finds"},
    "freeze": {"froze", "frozen", "freezing", "freezes"},
    "give": {"gave", "given", "giving", "gives"},
    "go": {"went", "gone", "going", "goes"},
    "hear": {"heard", "hearing", "hears"},
    "help": {"helped", "helping", "helps"},
    "hunt": {"hunted", "hunting", "hunts"},
    "know": {"knew", "known", "knowing", "knows"},
    "lend": {"lent", "lending", "lends"},
    "make": {"made", "making", "makes"},
    "miss": {"missed", "missing", "misses"},
    "notice": {"noticed", "noticing", "notices"},
    "offer": {"offered", "offering", "offers"},
    "pack": {"packed", "packing", "packs"},
    "paint": {"painted", "painting", "paints"},
    "read": {"read", "reading", "reads"},
    "roll": {"rolled", "rolling", "rolls"},
    "say": {"said", "saying", "says"},
    "see": {"saw", "seen", "seeing", "sees"},
    "sell": {"sold", "selling", "sells"},
    "serve": {"served", "serving", "serves"},
    "take": {"took", "taken", "taking", "takes"},
    "teach": {"taught", "teaching", "teaches"},
    "tell": {"told", "telling", "tells"},
    "think": {"thought", "thinking", "thinks"},
    "wire": {"wired", "wiring", "wires"},
    "write": {"wrote", "written", "writing", "writes"},
}
_REVERSE_IRREGULAR = {
    inflected: lemma
    for lemma, forms in _IRREGULAR_PREDICATES.items()
    for inflected in forms
}


def find_phrase_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    if not phrase.strip():
        return []
    pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", flags=re.IGNORECASE)
    return [match.span() for match in pattern.finditer(text)]


def build_probe_targets(record: dict[str, Any]) -> list[dict[str, Any]]:
    input_text = str(record["input_text"])
    roles = dict(record.get("role_signature") or {})
    predicate = str(roles.get("predicate") or "")
    mention_spans = {
        role_key: _resolve_surface_span(input_text, str(roles.get(role_key) or ""), occurrence=0)
        for role_key in ROLE_KEYS
    }
    subject, subject_span, obj, object_span = _infer_surface_mentions(
        roles=roles,
        mention_spans=mention_spans,
        text=input_text,
    )
    predicate_span = _resolve_predicate_span(input_text, predicate)

    probes: list[dict[str, Any]] = []
    probes.extend(_make_surface_probe(record, "surface_subject", subject, subject_span))
    probes.extend(_make_surface_probe(record, "surface_object", obj, object_span))
    probes.extend(_make_role_probe(record, "agent", roles.get("agent"), subject, subject_span, obj, object_span))
    probes.extend(_make_role_probe(record, "theme", roles.get("theme"), subject, subject_span, obj, object_span))
    probes.extend(_make_predicate_probe(record, predicate, predicate_span))
    return probes


def _resolve_surface_span(text: str, phrase: str, occurrence: int) -> tuple[int, int] | None:
    spans = find_phrase_spans(text, phrase)
    if not spans:
        return None
    index = min(occurrence, len(spans) - 1)
    return spans[index]


def _infer_surface_mentions(
    roles: dict[str, Any],
    mention_spans: dict[str, tuple[int, int] | None],
    text: str,
) -> tuple[str, tuple[int, int] | None, str, tuple[int, int] | None]:
    subject_text = str(roles.get("surface_subject") or "")
    object_text = str(roles.get("surface_object") or "")
    subject_span = _resolve_surface_span(text, subject_text, occurrence=0)
    object_occurrence = 1 if subject_text and object_text and subject_text.lower() == object_text.lower() else 0
    object_span = _resolve_surface_span(text, object_text, occurrence=object_occurrence)

    ranked_mentions = sorted(
        (
            (role_key, span, str(roles.get(role_key) or ""))
            for role_key, span in mention_spans.items()
            if span is not None and roles.get(role_key)
        ),
        key=lambda item: (item[1][0], item[1][1], item[0]),
    )

    if subject_span is None and ranked_mentions:
        _, subject_span, subject_text = ranked_mentions[0]
    if object_span is None:
        for role_key, span, filler in ranked_mentions:
            if span == subject_span and filler.lower() == subject_text.lower():
                continue
            object_text = filler
            object_span = span
            break

    return subject_text, subject_span, object_text, object_span


def _resolve_predicate_span(text: str, predicate: str) -> tuple[int, int] | None:
    if not predicate:
        return None
    exact = _resolve_surface_span(text, predicate, occurrence=0)
    if exact is not None:
        return exact

    candidates = _predicate_variants(predicate)
    for match in re.finditer(r"[A-Za-z']+", text):
        token = match.group(0)
        normalized = _normalize_predicate_form(token)
        if token.lower() in candidates or normalized == predicate.lower():
            return match.span()
    return None


def _predicate_variants(predicate: str) -> set[str]:
    base = predicate.lower()
    variants = {
        base,
        f"{base}s",
        f"{base}es",
        f"{base}ing",
        f"{base}ed",
        f"{base}d",
    }
    if base.endswith("e"):
        variants.add(f"{base[:-1]}ing")
        variants.add(f"{base}d")
    if base.endswith("y") and len(base) > 1:
        variants.add(f"{base[:-1]}ied")
    variants.update(_IRREGULAR_PREDICATES.get(base, set()))
    return variants


def _normalize_predicate_form(token: str) -> str:
    lowered = token.lower().strip(".,;:?!")
    if lowered in _REVERSE_IRREGULAR:
        return _REVERSE_IRREGULAR[lowered]
    for suffix in ("ing", "ied", "ed", "es", "s"):
        if lowered.endswith(suffix) and len(lowered) > len(suffix) + 1:
            if suffix == "ied":
                return f"{lowered[:-3]}y"
            stem = lowered[: -len(suffix)]
            if suffix == "ing" and stem.endswith("z"):
                return f"{stem}e"
            if suffix in {"ed", "es", "s"} and stem.endswith("e"):
                return stem
            if suffix == "ed":
                return stem
            return stem
    return lowered


def _make_surface_probe(
    record: dict[str, Any],
    label: str,
    surface_text: str,
    span: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    if span is None or not surface_text:
        return []
    return [
        {
            "probe_id": f"{record['item_id']}::{label}",
            "namespace": "surface_position",
            "label": label,
            "surface_text": surface_text,
            "start_char": span[0],
            "end_char": span[1],
        }
    ]


def _make_role_probe(
    record: dict[str, Any],
    role_label: str,
    filler_value: Any,
    subject_text: str,
    subject_span: tuple[int, int] | None,
    object_text: str,
    object_span: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    filler = str(filler_value or "")
    if not filler:
        return []

    span = None
    if filler.lower() == subject_text.lower():
        span = subject_span
    elif filler.lower() == object_text.lower():
        span = object_span
    else:
        span = _resolve_surface_span(str(record["input_text"]), filler, occurrence=0)
    if span is None:
        return []

    return [
        {
            "probe_id": f"{record['item_id']}::{role_label}",
            "namespace": "semantic_role",
            "label": role_label,
            "surface_text": filler,
            "start_char": span[0],
            "end_char": span[1],
        },
        {
            "probe_id": f"{record['item_id']}::filler::{role_label}",
            "namespace": "filler_identity",
            "label": filler.lower(),
            "surface_text": filler,
            "start_char": span[0],
            "end_char": span[1],
        },
    ]


def _make_predicate_probe(
    record: dict[str, Any],
    predicate: str,
    span: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    if span is None or not predicate:
        return []
    return [
        {
            "probe_id": f"{record['item_id']}::predicate",
            "namespace": "predicate_identity",
            "label": predicate.lower(),
            "surface_text": predicate,
            "start_char": span[0],
            "end_char": span[1],
        }
    ]
