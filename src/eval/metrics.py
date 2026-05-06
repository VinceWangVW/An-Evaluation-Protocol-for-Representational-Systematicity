from __future__ import annotations

import re
from collections import Counter


ATOM_SPLIT_RE = re.compile(r"\s+(?:AND|;)\s+")
ATOM_RE = re.compile(r"^\*?\s*(.+?)\s*\(\s*(.+?)\s*\)\s*$")
VAR_RE = re.compile(r"^(?:x_*\d+|\d+)$")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def normalize_predicate(predicate: str) -> str:
    collapsed = normalize_text(predicate).lower()
    return collapsed.replace(" . ", ".").replace(" .", ".").replace(". ", ".")


def normalize_argument(argument: str, var_map: dict[str, str]) -> str:
    token = normalize_text(argument)
    compact = token.replace(" ", "")
    if VAR_RE.fullmatch(compact):
        if compact not in var_map:
            var_map[compact] = f"v{len(var_map) + 1}"
        return var_map[compact]
    return token.lower()


def split_arguments(argument_blob: str) -> list[str]:
    return [piece.strip() for piece in argument_blob.split(",") if piece.strip()]


def extract_normalized_atoms(text: str) -> list[str]:
    pieces = [piece.strip() for piece in ATOM_SPLIT_RE.split(normalize_text(text)) if piece.strip()]
    parsed_atoms: list[tuple[str, list[str]]] = []

    for piece in pieces:
        match = ATOM_RE.match(piece)
        if not match:
            parsed_atoms.append((piece.lower(), []))
            continue
        predicate, args_blob = match.groups()
        parsed_atoms.append((normalize_predicate(predicate), split_arguments(args_blob)))

    parsed_atoms.sort(key=lambda item: (item[0], tuple(normalize_text(arg) for arg in item[1])))

    var_map: dict[str, str] = {}
    atoms: list[str] = []

    for predicate, raw_args in parsed_atoms:
        args = [normalize_argument(arg, var_map) for arg in raw_args]
        atoms.append(f"{predicate}({','.join(args)})" if raw_args else predicate)

    return atoms


def multiset_f1(pred_atoms: list[str], gold_atoms: list[str]) -> tuple[float, float, float]:
    pred_counter = Counter(pred_atoms)
    gold_counter = Counter(gold_atoms)
    overlap = sum((pred_counter & gold_counter).values())

    pred_total = sum(pred_counter.values())
    gold_total = sum(gold_counter.values())

    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / gold_total if gold_total else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def categorize_failure(exact_match: bool, normalized_match: bool, clause_exact: bool, clause_f1: float, prediction: str) -> str:
    if exact_match:
        return "exact_match"
    if normalized_match:
        return "normalized_only"
    if clause_exact:
        return "clause_exact_only"
    if not prediction.strip():
        return "empty_prediction"
    if clause_f1 > 0.0:
        return "partial_clause_overlap"
    return "no_clause_overlap"


def score_prediction(prediction: str, gold: str) -> dict[str, object]:
    prediction = prediction or ""
    gold = gold or ""

    raw_pred = prediction.strip()
    raw_gold = gold.strip()
    norm_pred = normalize_text(raw_pred)
    norm_gold = normalize_text(raw_gold)

    pred_atoms = extract_normalized_atoms(prediction)
    gold_atoms = extract_normalized_atoms(gold)
    precision, recall, f1 = multiset_f1(pred_atoms, gold_atoms)
    clause_exact = pred_atoms == gold_atoms

    exact_match = raw_pred == raw_gold
    normalized_match = norm_pred == norm_gold

    return {
        "exact_match": exact_match,
        "normalized_exact_match": normalized_match,
        "clause_exact_match": clause_exact,
        "clause_precision": round(precision, 6),
        "clause_recall": round(recall, 6),
        "clause_f1": round(f1, 6),
        "pred_atom_count": len(pred_atoms),
        "gold_atom_count": len(gold_atoms),
        "failure_category": categorize_failure(
            exact_match=exact_match,
            normalized_match=normalized_match,
            clause_exact=clause_exact,
            clause_f1=f1,
            prediction=prediction,
        ),
    }
