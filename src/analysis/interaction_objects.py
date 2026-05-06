from __future__ import annotations

import inspect
import json
import random
import re
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local lightweight environments may not have torch
    torch = None  # type: ignore[assignment]

from interventions.generation import (
    GEOMETRY_INTERACTION_OBJECT_CLASS,
    aggregate_control_value,
    build_exact_counterpart_index,
    find_exact_role_counterpart,
    infer_geometry_object_class,
    normalize_geometry_object_spec,
    opposite_role,
    stable_seed,
)
from utils.io import read_jsonl


DISCOVERY_MIN_CONTRAST_GAP = 0.05
H2_NECESSITY_MIN_GAP = 0.05
H0_NECESSITY_MAX_ABS_GAP = 0.02
REALISM_MIN_PASS_RATE = 0.8
GENERIC_GAIN_MAX_OVERLAP = 0.2
H1_RESCUE_MIN_GAP = 0.05
H1_GENERIC_ADVANTAGE_MIN = 0.02
H1_RESCUE_MIN_BINARY_RATE = 0.2
CASE_MIN_SLICE_RESCUE_GAP = 0.05
CASE_MIN_SLICE_BINARY_RATE = 0.2
TOP_BASIS_COUNT = 8
GENERIC_BASIS_COUNT = 16
READ_BOOST_SCALE = 1.5
HEAD_BOOST_SCALE = 1.5

STEP_SOURCE_KEYS: dict[str, list[str]] = {
    "predicate_value": ["predicate"],
    "agent_value": ["agent", "filler::agent"],
    "theme_value": ["theme", "filler::theme"],
}

STEP_STORE_SELECTOR: dict[str, str] = {
    "predicate_value": "predicate_bundle",
    "agent_value": "semantic_role_bundle",
    "theme_value": "semantic_role_bundle",
}

STEP_TEMPLATE_FAMILIES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("top1_head_top1_basis_no_store", {"head_count": 1, "basis_count": 1, "use_store": False}),
    ("top2_head_bundle_top1_basis_no_store", {"head_count": 2, "basis_count": 1, "use_store": False}),
    ("top1_head_top1_basis_matched_store", {"head_count": 1, "basis_count": 1, "use_store": True}),
    ("top2_head_bundle_top2_basis_bundle_matched_store", {"head_count": 2, "basis_count": 2, "use_store": True}),
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - exercised only in environments without torch
        raise ModuleNotFoundError("Phase 17 interaction execution requires torch")


def _ensure_item_span_map(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from interventions.analyze import ensure_item_span_map as _runtime_ensure_item_span_map

    return _runtime_ensure_item_span_map(*args, **kwargs)


def _index_span_records(span_records: list[dict[str, Any]], layer_index: int) -> dict[tuple[str, str], dict[str, Any]]:
    from interventions.analyze import index_span_records as _runtime_index_span_records

    return _runtime_index_span_records(span_records, layer_index)


def _build_model_adapter(*args: Any, **kwargs: Any) -> Any:
    from models import build_model_adapter as _runtime_build_model_adapter

    return _runtime_build_model_adapter(*args, **kwargs)


def _summary_prefix(summary_path: str | Path) -> str:
    stem = Path(summary_path).stem
    return stem[: -len("_summary")] if stem.endswith("_summary") else stem


def _discovery_artifact_path(project_root: str | Path, prefix: str, suffix: str) -> Path:
    return Path(project_root) / "artifacts/discovery_runs" / f"{prefix}_{suffix}.jsonl"


def _load_discovery_bundle(project_root: str | Path, summary_path: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    summary = load_json(summary_path)
    prefix = _summary_prefix(summary_path)
    behavior_summary = load_json(root / "reports/behavior_runs" / f"{summary['behavior_run_name']}_summary.json")
    return {
        "summary_path": str(summary_path),
        "prefix": prefix,
        "summary": summary,
        "behavior_summary": behavior_summary,
        "candidate_records": read_jsonl(_discovery_artifact_path(root, prefix, "candidate_table")),
        "matched_sets": read_jsonl(_discovery_artifact_path(root, prefix, "matched_sets")),
        "span_records": read_jsonl(_discovery_artifact_path(root, prefix, "span_vectors")),
    }


def _load_protocol_bundle(project_root: str | Path, protocol_summary_path: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    protocol_summary = load_json(protocol_summary_path)
    discovery_prefix = str(protocol_summary["discovery_prefix"])
    discovery_summary_path = root / "reports/discovery_runs" / f"{discovery_prefix}_summary.json"
    bundle = _load_discovery_bundle(root, discovery_summary_path)
    bundle["protocol_summary"] = protocol_summary
    return bundle


def build_adapter_from_bundle(bundle: dict[str, Any]) -> Any:
    summary = dict(bundle["summary"])
    behavior_summary = dict(bundle["behavior_summary"])
    layer_index = int(next(iter(summary["layers"].keys())))
    return _build_model_adapter(
        model_name_or_path=str(summary.get("trained_model_path") or behavior_summary.get("trained_model_path") or behavior_summary["model_name_or_path"]),
        family=str(behavior_summary["family"]),
        prompt_style=str(behavior_summary["prompt_style"]),
        precision=str(summary.get("precision") or behavior_summary.get("precision") or "auto"),
        num_beams=int(summary.get("num_beams") or behavior_summary.get("num_beams") or 1),
        decoder_cross_attention_scale=summary.get("decoder_cross_attention_scale"),
        decoder_cross_attention_keep_ratio=summary.get("decoder_cross_attention_keep_ratio"),
        decoder_cross_attention_layer_keep_ratio=summary.get("decoder_cross_attention_layer_keep_ratio"),
        decoder_cross_attention_layer_seed=int(summary.get("decoder_cross_attention_layer_seed") or 0),
        decoder_cross_attention_layer_policy=str(summary.get("decoder_cross_attention_layer_policy") or "random"),
        decoder_cross_attention_bottleneck_keep_ratio=summary.get("decoder_cross_attention_bottleneck_keep_ratio"),
        decoder_cross_attention_bottleneck_seed=int(summary.get("decoder_cross_attention_bottleneck_seed") or 0),
        decoder_cross_attention_bottleneck_mode=str(summary.get("decoder_cross_attention_bottleneck_mode") or "keep"),
        decoder_cross_attention_relation_barrier_keep_ratio=summary.get("decoder_cross_attention_relation_barrier_keep_ratio"),
        decoder_cross_attention_relation_barrier_seed=int(summary.get("decoder_cross_attention_relation_barrier_seed") or 0),
        decoder_cross_attention_relation_barrier_scale=float(summary.get("decoder_cross_attention_relation_barrier_scale") or 0.5),
        decoder_readout_keep_ratio=summary.get("decoder_readout_keep_ratio"),
        decoder_readout_subspace_keep_ratio=summary.get("decoder_readout_subspace_keep_ratio"),
        decoder_readout_subspace_seed=int(summary.get("decoder_readout_subspace_seed") or 0),
        decoder_readout_subspace_mode=str(summary.get("decoder_readout_subspace_mode") or "remove"),
        encoder_geometry_transform=summary.get("encoder_geometry_transform"),
        encoder_geometry_seed=int(summary.get("encoder_geometry_seed") or 0),
        encoder_geometry_keep_ratio=summary.get("encoder_geometry_keep_ratio"),
        encoder_geometry_noise_std=summary.get("encoder_geometry_noise_std"),
    ), layer_index


def _parse_target_fields(record: dict[str, Any]) -> dict[str, str]:
    roles = dict(record.get("role_signature") or {})
    metadata = dict(record.get("metadata") or {})
    target_form = str(record.get("target_form") or "")
    predicate = str(roles.get("predicate") or metadata.get("predicate_lemma") or "")
    if not predicate:
        match = re.match(r"^\s*([a-z_]+)\(", target_form)
        predicate = match.group(1) if match else ""
    fields = {
        "predicate_value": predicate,
        "agent_value": str(roles.get("agent") or _extract_target_field(target_form, "agent")),
        "theme_value": str(roles.get("theme") or _extract_target_field(target_form, "theme")),
    }
    return {key: value for key, value in fields.items() if value}


def _extract_target_field(target_form: str, field_name: str) -> str:
    match = re.search(rf"{re.escape(field_name)}=([a-z_]+)", str(target_form))
    return match.group(1) if match else ""


def _non_special_target_ids(tokenizer: Any, target_text: str) -> list[int]:
    encoded = tokenizer(target_text, add_special_tokens=False)
    return [int(value) for value in list(encoded["input_ids"])]


def _non_special_tokens(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [str(token) for token in tokenizer.convert_ids_to_tokens(token_ids)]


def _leftmost_subsequence_positions(sequence: list[int], subsequence: list[int]) -> list[int]:
    if not sequence or not subsequence or len(subsequence) > len(sequence):
        return []
    for start in range(0, len(sequence) - len(subsequence) + 1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return list(range(start, start + len(subsequence)))
    return []


def align_decoder_steps(tokenizer: Any, target_text: str, record: dict[str, Any]) -> dict[str, Any]:
    target_ids = _non_special_target_ids(tokenizer, target_text)
    target_tokens = _non_special_tokens(tokenizer, target_ids)
    fields = _parse_target_fields(record)
    alignments: dict[str, list[int]] = {}
    unaligned: list[str] = []
    for step_key in ("predicate_value", "agent_value", "theme_value"):
        field_text = str(fields.get(step_key) or "").strip()
        if not field_text:
            unaligned.append(step_key)
            continue
        field_ids = _non_special_target_ids(tokenizer, field_text)
        positions = _leftmost_subsequence_positions(target_ids, field_ids)
        if positions:
            alignments[step_key] = positions
        else:
            unaligned.append(step_key)
    return {
        "target_ids": target_ids,
        "target_tokens": target_tokens,
        "alignments": alignments,
        "unaligned_steps": unaligned,
    }


def _source_token_indices(item_span_map: dict[str, Any], source_span_keys: list[str]) -> list[int]:
    values: list[int] = []
    for key in source_span_keys:
        record = item_span_map.get(key)
        if record is None:
            continue
        values.extend(int(index) for index in list(record.get("token_indices") or []))
    return sorted({value for value in values})


def _layer_module_index(layer_index: int, total_layers: int) -> int:
    if total_layers <= 0:
        return 0
    return max(0, min(total_layers - 1, int(layer_index) - 1))


def _decoder_start_token_id(model: Any, tokenizer: Any) -> int:
    if getattr(model.config, "decoder_start_token_id", None) is not None:
        return int(model.config.decoder_start_token_id)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return int(tokenizer.pad_token_id)
    return int(tokenizer.eos_token_id)


def _eos_token_id(model: Any, tokenizer: Any) -> int:
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return int(tokenizer.eos_token_id)
    return int(model.config.eos_token_id)


def _supports_kwarg(callable_obj: Any, name: str) -> bool:
    try:
        return name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


@contextmanager
def _temporary_encoder_token_edit(
    model: Any,
    *,
    layer_index: int,
    token_indices: list[int],
    replacement_vectors: dict[int, list[float]] | None = None,
    scale: float | None = None,
):
    _require_torch()
    if not token_indices:
        yield
        return
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "block"):
        yield
        return
    blocks = list(model.encoder.block)
    module_index = _layer_module_index(layer_index, len(blocks))
    block = blocks[module_index]

    def hook_fn(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
        for token_index in token_indices:
            token_index = int(token_index)
            if replacement_vectors is not None and token_index in replacement_vectors:
                vector = torch.tensor(replacement_vectors[token_index], device=hidden.device, dtype=hidden.dtype)
                hidden[:, token_index, :] = vector
            elif scale is not None:
                hidden[:, token_index, :].mul_(float(scale))
            else:
                hidden[:, token_index, :].zero_()
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = block.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def _temporary_lm_head_basis_edit(
    model: Any,
    *,
    step_indices: list[int],
    basis_indices: list[int],
    mode: str = "remove",
    scale: float = 1.0,
):
    _require_torch()
    if not step_indices or not basis_indices or not hasattr(model, "lm_head"):
        yield
        return

    def pre_hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args:
            return args
        hidden = args[0]
        if not isinstance(hidden, torch.Tensor):
            return args
        edited = hidden.clone()
        for step_index in step_indices:
            for basis_index in basis_indices:
                if mode == "remove":
                    edited[:, int(step_index), int(basis_index)] = 0.0
                elif mode == "boost":
                    edited[:, int(step_index), int(basis_index)] *= float(scale)
                else:
                    raise ValueError(f"Unsupported decoder basis edit mode: {mode}")
        if len(args) == 1:
            return (edited,)
        return (edited, *args[1:])

    handle = model.lm_head.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


def _shift_right_decoder_inputs(model: Any, tokenizer: Any, target_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    eos_token_id = _eos_token_id(model, tokenizer)
    start_token_id = _decoder_start_token_id(model, tokenizer)
    label_ids = list(target_ids) + [eos_token_id]
    decoder_input_ids = [start_token_id] + list(target_ids)
    labels = torch.tensor([label_ids], dtype=torch.long)
    decoder_inputs = torch.tensor([decoder_input_ids], dtype=torch.long)
    return labels, decoder_inputs


def _adapter_context_stack(adapter: Any) -> ExitStack:
    from models.geometry_access_controls import (
        temporary_seq2seq_cross_attention_bottleneck_projection,
        temporary_seq2seq_cross_attention_layer_keep_ratio,
        temporary_seq2seq_cross_attention_relation_barrier,
        temporary_seq2seq_decoder_readout_subspace_projection,
    )
    from models.routing_controls import (
        autocast_context,
        temporary_seq2seq_cross_attention_feature_mask,
        temporary_seq2seq_cross_attention_scale,
        temporary_seq2seq_decoder_readout_feature_mask,
    )

    stack = ExitStack()
    stack.enter_context(
        temporary_seq2seq_cross_attention_scale(adapter.model, adapter.decoder_cross_attention_scale)
    )
    stack.enter_context(
        temporary_seq2seq_cross_attention_feature_mask(adapter.model, adapter.decoder_cross_attention_keep_ratio)
    )
    stack.enter_context(
        temporary_seq2seq_cross_attention_layer_keep_ratio(
            adapter.model if adapter.family == "seq2seq_lm" else None,
            adapter.decoder_cross_attention_layer_keep_ratio if adapter.family == "seq2seq_lm" else None,
            seed=adapter.decoder_cross_attention_layer_seed,
            policy=adapter.decoder_cross_attention_layer_policy,
        )
    )
    stack.enter_context(
        temporary_seq2seq_cross_attention_bottleneck_projection(
            adapter.model if adapter.family == "seq2seq_lm" else None,
            adapter.decoder_cross_attention_bottleneck_keep_ratio if adapter.family == "seq2seq_lm" else None,
            seed=adapter.decoder_cross_attention_bottleneck_seed,
            mode=adapter.decoder_cross_attention_bottleneck_mode,
        )
    )
    stack.enter_context(
        temporary_seq2seq_cross_attention_relation_barrier(
            adapter.model if adapter.family == "seq2seq_lm" else None,
            adapter.decoder_cross_attention_relation_barrier_keep_ratio if adapter.family == "seq2seq_lm" else None,
            seed=adapter.decoder_cross_attention_relation_barrier_seed,
            scale=adapter.decoder_cross_attention_relation_barrier_scale,
        )
    )
    stack.enter_context(
        temporary_seq2seq_decoder_readout_feature_mask(
            adapter.model if adapter.family == "seq2seq_lm" else None,
            adapter.decoder_readout_keep_ratio if adapter.family == "seq2seq_lm" else None,
        )
    )
    stack.enter_context(
        temporary_seq2seq_decoder_readout_subspace_projection(
            adapter.model if adapter.family == "seq2seq_lm" else None,
            adapter.decoder_readout_subspace_keep_ratio if adapter.family == "seq2seq_lm" else None,
            seed=adapter.decoder_readout_subspace_seed,
            mode=adapter.decoder_readout_subspace_mode,
        )
    )
    return stack


def _build_cross_attn_head_mask(model: Any, *, layer_index: int, head_indices: list[int], selected_scale: float) -> torch.Tensor | None:
    _require_torch()
    if not hasattr(model, "decoder") or not hasattr(model.decoder, "block"):
        return None
    total_layers = len(list(model.decoder.block))
    if total_layers <= 0:
        return None
    num_heads = int(getattr(model.config, "num_heads", 0) or 0)
    if num_heads <= 0:
        return None
    mask = torch.ones((total_layers, num_heads), dtype=torch.float32)
    layer_slot = _layer_module_index(layer_index, total_layers)
    for head_index in list(head_indices):
        if 0 <= int(head_index) < num_heads:
            mask[layer_slot, int(head_index)] = float(selected_scale)
    return mask


def _teacher_forced_forward(
    *,
    adapter: Any,
    input_text: str,
    target_text: str,
    layer_index: int,
    max_length: int,
    head_mask: torch.Tensor | None = None,
    encoder_token_indices: list[int] | None = None,
    encoder_replacement_vectors: dict[int, list[float]] | None = None,
    encoder_scale: float | None = None,
    step_indices: list[int] | None = None,
    basis_indices: list[int] | None = None,
    basis_mode: str | None = None,
    basis_scale: float = 1.0,
) -> dict[str, Any]:
    _require_torch()
    from models.routing_controls import autocast_context

    if adapter.family != "seq2seq_lm":
        raise NotImplementedError("Phase 17 interaction tracing currently supports seq2seq models only")
    prompt_text = adapter.build_prompt(input_text)
    encoded = adapter.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {key: value.to(adapter.device) for key, value in encoded.items()}
    target_ids = _non_special_target_ids(adapter.tokenizer, target_text)
    labels, decoder_input_ids = _shift_right_decoder_inputs(adapter.model, adapter.tokenizer, target_ids)
    labels = labels.to(adapter.device)
    decoder_input_ids = decoder_input_ids.to(adapter.device)

    kwargs: dict[str, Any] = {
        **inputs,
        "decoder_input_ids": decoder_input_ids,
        "output_attentions": True,
        "output_hidden_states": True,
        "return_dict": True,
        "use_cache": False,
    }
    if _supports_kwarg(adapter.model.forward, "cross_attn_head_mask") and head_mask is not None:
        kwargs["cross_attn_head_mask"] = head_mask.to(adapter.device)

    with _adapter_context_stack(adapter) as stack:
        stack.enter_context(
            _temporary_encoder_token_edit(
                adapter.model,
                layer_index=layer_index,
                token_indices=list(encoder_token_indices or []),
                replacement_vectors=encoder_replacement_vectors,
                scale=encoder_scale,
            )
            if encoder_token_indices
            else nullcontext()
        )
        stack.enter_context(
            _temporary_lm_head_basis_edit(
                adapter.model,
                step_indices=list(step_indices or []),
                basis_indices=list(basis_indices or []),
                mode=str(basis_mode or "remove"),
                scale=float(basis_scale),
            )
            if basis_mode is not None and step_indices and basis_indices
            else nullcontext()
        )
        with autocast_context(adapter.device, adapter.precision):
            with torch.inference_mode():
                outputs = adapter.model(**kwargs)

    decoder_hidden_states = list(outputs.decoder_hidden_states or [])
    cross_attentions = list(outputs.cross_attentions or [])
    final_decoder = decoder_hidden_states[-1][0].detach().float().cpu() if decoder_hidden_states else torch.empty(0)
    logits = outputs.logits[0].detach().float().cpu()
    predicted_ids = logits.argmax(dim=-1).tolist()
    predicted_tokens = _non_special_tokens(adapter.tokenizer, predicted_ids[: len(target_ids)])
    return {
        "prompt_text": prompt_text,
        "target_ids": target_ids,
        "target_tokens": _non_special_tokens(adapter.tokenizer, target_ids),
        "predicted_ids": predicted_ids,
        "predicted_tokens": predicted_tokens,
        "decoder_hidden": final_decoder,
        "cross_attentions": [tensor[0].detach().float().cpu() for tensor in cross_attentions],
        "logits": logits,
    }


def _basis_scores_for_steps(decoder_hidden: torch.Tensor, step_positions: list[int]) -> dict[str, float]:
    _require_torch()
    if decoder_hidden.numel() == 0 or not step_positions:
        return {}
    pooled = decoder_hidden[step_positions, :].abs().mean(dim=0)
    if pooled.numel() == 0:
        return {}
    topk = min(TOP_BASIS_COUNT, int(pooled.numel()))
    scores, indices = torch.topk(pooled, k=topk)
    total = float(scores.sum().item()) or 1.0
    return {
        f"prelogit_basis_{int(index)}": round(float(score.item()) / total, 6)
        for score, index in zip(scores, indices)
    }


def _fallback_compare_token_id(logits_row: torch.Tensor, correct_token_id: int) -> int:
    _require_torch()
    sorted_indices = torch.argsort(logits_row, descending=True).tolist()
    for index in sorted_indices:
        if int(index) != int(correct_token_id):
            return int(index)
    return int(correct_token_id)


def _counterpart_compare_ids(
    *,
    tokenizer: Any,
    step_key: str,
    target_positions: list[int],
    counterpart_record: dict[str, Any] | None,
) -> list[int]:
    if counterpart_record is None:
        return []
    counterpart_alignments = align_decoder_steps(tokenizer, str(counterpart_record["target_form"]), counterpart_record)
    positions = list(counterpart_alignments["alignments"].get(step_key) or [])
    ids = list(counterpart_alignments["target_ids"])
    if not positions:
        return []
    usable = min(len(target_positions), len(positions))
    return [int(ids[positions[index]]) for index in range(usable)]


def _token_margin_summary(
    *,
    logits: torch.Tensor,
    target_ids: list[int],
    step_positions: list[int],
    compare_ids: list[int],
) -> dict[str, Any]:
    if not step_positions:
        return {
            "margin_mean": None,
            "exact_match_rate": None,
            "exact_match": False,
        }
    margins: list[float] = []
    exact_hits: list[int] = []
    for relative_index, position in enumerate(step_positions):
        correct_id = int(target_ids[position])
        compare_id = (
            int(compare_ids[relative_index])
            if relative_index < len(compare_ids) and int(compare_ids[relative_index]) != correct_id
            else _fallback_compare_token_id(logits[position], correct_id)
        )
        margin = float(logits[position, correct_id]) - float(logits[position, compare_id])
        margins.append(margin)
        exact_hits.append(int(int(torch.argmax(logits[position]).item()) == correct_id))
    return {
        "margin_mean": round(mean(margins), 6) if margins else None,
        "exact_match_rate": round(mean(exact_hits), 6) if exact_hits else None,
        "exact_match": bool(all(exact_hits)),
    }


def _trace_rows_with_generic_overlap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    basis_mass: Counter[str] = Counter()
    for row in rows:
        for basis_id, score in dict(row.get("prelogit_basis_scores") or {}).items():
            basis_mass[str(basis_id)] += float(score)
    generic_basis_ids = [basis_id for basis_id, _ in basis_mass.most_common(GENERIC_BASIS_COUNT)]
    annotated: list[dict[str, Any]] = []
    for row in rows:
        scores = dict(row.get("prelogit_basis_scores") or {})
        overlap = sum(float(scores.get(basis_id, 0.0)) for basis_id in generic_basis_ids)
        enriched = dict(row)
        enriched["generic_basis_ids"] = list(generic_basis_ids)
        enriched["generic_decoder_gain_overlap"] = round(float(overlap), 6)
        annotated.append(enriched)
    return annotated


def trace_control_bundle(
    *,
    bundle: dict[str, Any],
    project_root: str | Path,
    max_length: int = 256,
) -> dict[str, Any]:
    adapter, layer_index = build_adapter_from_bundle(bundle)
    span_index = _index_span_records(bundle["span_records"], layer_index=layer_index)
    live_span_cache: dict[str, dict[str, Any]] = {}
    counterpart_index = build_exact_counterpart_index(bundle["candidate_records"])
    trace_rows: list[dict[str, Any]] = []
    unaligned_rows: list[dict[str, Any]] = []
    for item in bundle["candidate_records"]:
        item_span_map = _ensure_item_span_map(
            adapter=adapter,
            item=item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        alignments = align_decoder_steps(adapter.tokenizer, str(item["target_form"]), item)
        if alignments["unaligned_steps"]:
            for step_key in list(alignments["unaligned_steps"]):
                unaligned_rows.append(
                    {
                        "item_id": str(item["item_id"]),
                        "control_id": str(bundle["summary"]["behavior_run_name"]),
                        "decoder_step_key": step_key,
                        "status": "unaligned_step",
                    }
                )
        source_indices_by_step = {
            step_key: _source_token_indices(item_span_map, STEP_SOURCE_KEYS[step_key])
            for step_key in ("predicate_value", "agent_value", "theme_value")
        }
        forward = _teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
        )
        if not forward["cross_attentions"]:
            continue
        cross_layer_index = _layer_module_index(layer_index, len(forward["cross_attentions"]))
        cross_attn = forward["cross_attentions"][cross_layer_index]
        num_heads = int(cross_attn.shape[0])
        for step_key, step_positions in dict(alignments["alignments"]).items():
            if not step_positions:
                continue
            source_indices = list(source_indices_by_step.get(step_key) or [])
            counterpart = None
            if step_key in {"agent_value", "theme_value"}:
                role = "agent" if step_key == "agent_value" else "theme"
                counterpart = find_exact_role_counterpart(
                    record=item,
                    role=role,
                    counterpart_index=counterpart_index,
                    preferred_strata=("pass", "control"),
                    same_split_only=False,
                )
            compare_ids = _counterpart_compare_ids(
                tokenizer=adapter.tokenizer,
                step_key=step_key,
                target_positions=step_positions,
                counterpart_record=counterpart,
            )
            step_summary = _token_margin_summary(
                logits=forward["logits"],
                target_ids=forward["target_ids"],
                step_positions=step_positions,
                compare_ids=compare_ids,
            )
            basis_scores = _basis_scores_for_steps(forward["decoder_hidden"], step_positions)
            for head_index in range(num_heads):
                head_mass = []
                for position in step_positions:
                    if not source_indices:
                        head_mass.append(0.0)
                    else:
                        head_mass.append(float(cross_attn[head_index, position, source_indices].sum().item()))
                trace_rows.append(
                    {
                        "item_id": str(item["item_id"]),
                        "control_id": str(bundle["summary"]["behavior_run_name"]),
                        "generalization_case": str(item["generalization_case"]),
                        "stratum": str(item["stratum"]),
                        "split": str(item["split"]),
                        "decoder_step_key": step_key,
                        "decoder_step_index": int(step_positions[0]),
                        "generated_token_text": str(forward["target_tokens"][step_positions[0]]),
                        "layer_index": int(layer_index),
                        "head_index": int(head_index),
                        "source_span_keys": list(STEP_SOURCE_KEYS[step_key]),
                        "source_token_indices": list(source_indices),
                        "cross_attn_mass": round(mean(head_mass), 6) if head_mass else 0.0,
                        "counterpart_margin_delta": step_summary["margin_mean"],
                        "step_exact_match": bool(step_summary["exact_match"]),
                        "step_exact_match_rate": step_summary["exact_match_rate"],
                        "prelogit_basis_scores": basis_scores,
                    }
                )
    annotated_rows = _trace_rows_with_generic_overlap(trace_rows)
    return {
        "rows": annotated_rows,
        "unaligned_rows": unaligned_rows,
        "summary": {
            "control_id": str(bundle["summary"]["behavior_run_name"]),
            "row_count": len(annotated_rows),
            "unaligned_count": len(unaligned_rows),
            "step_counts": dict(Counter(str(row["decoder_step_key"]) for row in annotated_rows)),
        },
    }


def _aggregate_trace_scores(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, float]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["decoder_step_key"]), int(row["head_index"]))].append(row)
    aggregated: dict[tuple[str, int], dict[str, float]] = {}
    for key, bucket in grouped.items():
        mass = mean(float(row.get("cross_attn_mass") or 0.0) for row in bucket)
        margin = mean(float(row.get("counterpart_margin_delta") or 0.0) for row in bucket)
        overlap = mean(float(row.get("generic_decoder_gain_overlap") or 0.0) for row in bucket)
        aggregated[key] = {
            "score": round(mass + margin - overlap, 6),
            "mass": round(mass, 6),
            "margin": round(margin, 6),
            "overlap": round(overlap, 6),
        }
    return aggregated


def _aggregate_basis_scores(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    by_step: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        step_key = str(row["decoder_step_key"])
        for basis_id, score in dict(row.get("prelogit_basis_scores") or {}).items():
            by_step[step_key][str(basis_id)] += float(score)
    return by_step


def _top_heads_for_step(dev_h2_rows: list[dict[str, Any]], dev_h0_rows: list[dict[str, Any]], step_key: str) -> list[int]:
    h2_scores = _aggregate_trace_scores([row for row in dev_h2_rows if str(row["decoder_step_key"]) == step_key])
    h0_scores = _aggregate_trace_scores([row for row in dev_h0_rows if str(row["decoder_step_key"]) == step_key])
    combined: list[tuple[float, int]] = []
    seen_heads = sorted({head for _, head in h2_scores.keys()} | {head for _, head in h0_scores.keys()})
    for head_index in seen_heads:
        score = float(h2_scores.get((step_key, head_index), {}).get("score", 0.0)) - float(
            h0_scores.get((step_key, head_index), {}).get("score", 0.0)
        )
        combined.append((score, int(head_index)))
    combined.sort(reverse=True)
    return [head for _, head in combined]


def _top_bases_for_step(dev_h2_rows: list[dict[str, Any]], dev_h0_rows: list[dict[str, Any]], step_key: str) -> list[str]:
    h2_counter = _aggregate_basis_scores([row for row in dev_h2_rows if str(row["decoder_step_key"]) == step_key]).get(step_key, Counter())
    h0_counter = _aggregate_basis_scores([row for row in dev_h0_rows if str(row["decoder_step_key"]) == step_key]).get(step_key, Counter())
    basis_ids = sorted(set(h2_counter.keys()) | set(h0_counter.keys()))
    ranked = sorted(
        ((float(h2_counter.get(basis_id, 0.0)) - float(h0_counter.get(basis_id, 0.0)), basis_id) for basis_id in basis_ids),
        reverse=True,
    )
    return [basis_id for _, basis_id in ranked]


def build_interaction_catalog(
    *,
    dev_h2_rows: list[dict[str, Any]],
    dev_h0_rows: list[dict[str, Any]],
    heldout_h2_rows: list[dict[str, Any]],
    heldout_h0_rows: list[dict[str, Any]],
    layer_index: int,
    source_run: str,
) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    generic_basis_ids = list(_trace_rows_with_generic_overlap(dev_h2_rows)[:1][0].get("generic_basis_ids", [])) if dev_h2_rows else []
    for step_key in ("predicate_value", "agent_value", "theme_value"):
        ranked_heads = _top_heads_for_step(dev_h2_rows, dev_h0_rows, step_key)
        ranked_bases = _top_bases_for_step(dev_h2_rows, dev_h0_rows, step_key)
        h2_scores = _aggregate_trace_scores([row for row in dev_h2_rows if str(row["decoder_step_key"]) == step_key])
        h0_scores = _aggregate_trace_scores([row for row in dev_h0_rows if str(row["decoder_step_key"]) == step_key])
        held_h2_scores = _aggregate_trace_scores([row for row in heldout_h2_rows if str(row["decoder_step_key"]) == step_key])
        held_h0_scores = _aggregate_trace_scores([row for row in heldout_h0_rows if str(row["decoder_step_key"]) == step_key])
        for template_name, template in STEP_TEMPLATE_FAMILIES:
            head_count = int(template["head_count"])
            basis_count = int(template["basis_count"])
            selected_heads = list(ranked_heads[:head_count])
            selected_bases = list(ranked_bases[:basis_count])
            if len(selected_heads) < head_count:
                selected_heads = selected_heads + [value for value in range(8) if value not in selected_heads][: head_count - len(selected_heads)]
            if len(selected_bases) < basis_count:
                selected_bases = selected_bases + [f"prelogit_basis_{index}" for index in range(32) if f"prelogit_basis_{index}" not in selected_bases][: basis_count - len(selected_bases)]
            dev_h2_contrast = mean(float(h2_scores.get((step_key, head), {}).get("score", 0.0)) for head in selected_heads)
            dev_h0_contrast = mean(float(h0_scores.get((step_key, head), {}).get("score", 0.0)) for head in selected_heads)
            heldout_h2_contrast = mean(float(held_h2_scores.get((step_key, head), {}).get("score", 0.0)) for head in selected_heads)
            heldout_h0_contrast = mean(float(held_h0_scores.get((step_key, head), {}).get("score", 0.0)) for head in selected_heads)
            overlap_scores = [
                float(row.get("generic_decoder_gain_overlap") or 0.0)
                for row in dev_h2_rows
                if str(row["decoder_step_key"]) == step_key and int(row["head_index"]) in selected_heads
            ]
            generic_overlap = round(mean(overlap_scores), 6) if overlap_scores else 0.0
            store_condition = {"mode": "none"}
            if bool(template["use_store"]):
                store_condition = {"mode": "matched_role_store", "selector_type": STEP_STORE_SELECTOR[step_key]}
            object_id = f"interaction_{step_key}_{template_name}"
            row = {
                "object_id": object_id,
                "object_class": GEOMETRY_INTERACTION_OBJECT_CLASS,
                "layer_index": int(layer_index),
                "source_run": str(source_run),
                "decoder_step_key": step_key,
                "source_span_keys": list(STEP_SOURCE_KEYS[step_key]),
                "head_selector": {"mode": "explicit", "indices": selected_heads},
                "read_basis": {"mode": "explicit", "basis_ids": selected_bases},
                "store_condition": store_condition,
                "matched_control_type": "wrong_event_same_step",
                "template_family": template_name,
                "generic_basis_ids": list(generic_basis_ids[: max(1, basis_count)] or selected_bases),
                "dev_h2_contrast": round(float(dev_h2_contrast), 6),
                "dev_h0_contrast": round(float(dev_h0_contrast), 6),
                "heldout_h2_contrast": round(float(heldout_h2_contrast), 6),
                "heldout_h0_contrast": round(float(heldout_h0_contrast), 6),
                "generic_gain_overlap": generic_overlap,
            }
            row["non_flat_survivor"] = bool((row["dev_h2_contrast"] - row["dev_h0_contrast"]) >= DISCOVERY_MIN_CONTRAST_GAP)
            catalog.append(row)
    return catalog


def summarize_interaction_catalog(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    survivors = [row for row in catalog if bool(row.get("non_flat_survivor"))]
    survivor_step_keys = sorted({str(row["decoder_step_key"]) for row in survivors})
    survivor_templates = sorted({str(row["template_family"]) for row in survivors})
    accepted = (
        len(catalog) == 12
        and all(str(row.get("object_class")) == GEOMETRY_INTERACTION_OBJECT_CLASS for row in catalog)
        and len(survivors) >= 8
        and len(survivor_step_keys) >= 2
        and len(survivor_templates) >= 2
    )
    decision = "continue_to_h2_gate" if accepted else "T-supported"
    decision_status = "preliminary" if accepted else "final"
    return {
        "object_count": len(catalog),
        "interaction_step_counts": dict(Counter(str(row["decoder_step_key"]) for row in catalog)),
        "surviving_candidate_count": len(survivors),
        "surviving_candidate_ids": [str(row["object_id"]) for row in survivors],
        "surviving_step_keys": survivor_step_keys,
        "surviving_template_families": survivor_templates,
        "accepted": accepted,
        "decision": decision,
        "decision_status": decision_status,
    }


def _basis_ids_to_indices(basis_ids: list[str]) -> list[int]:
    indices: list[int] = []
    for basis_id in list(basis_ids):
        match = re.search(r"(\d+)$", str(basis_id))
        if match:
            indices.append(int(match.group(1)))
    return indices


def _wrong_source_span_keys(step_key: str) -> list[str]:
    if step_key == "predicate_value":
        return list(STEP_SOURCE_KEYS["agent_value"])
    if step_key == "agent_value":
        return list(STEP_SOURCE_KEYS["theme_value"])
    return list(STEP_SOURCE_KEYS["agent_value"])


def _shuffled_indices(total: int, excluded: list[int], count: int, seed: int) -> list[int]:
    pool = [index for index in range(total) if index not in {int(value) for value in excluded}]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return list(pool[:count])


def _store_patch_vectors(
    *,
    target_span_map: dict[str, Any],
    donor_span_map: dict[str, Any],
    source_span_keys: list[str],
    donor_source_span_keys: list[str] | None = None,
) -> tuple[list[int], dict[int, list[float]]]:
    donor_keys = list(donor_source_span_keys or source_span_keys)
    target_token_indices: list[int] = []
    replacement_vectors: dict[int, list[float]] = {}
    for target_key, donor_key in zip(source_span_keys, donor_keys):
        target_record = target_span_map.get(target_key)
        donor_record = donor_span_map.get(donor_key)
        if target_record is None or donor_record is None:
            continue
        for token_index in list(target_record.get("token_indices") or []):
            target_token_indices.append(int(token_index))
            replacement_vectors[int(token_index)] = list(donor_record["vector"])
    return sorted({int(value) for value in target_token_indices}), replacement_vectors


def evaluate_interaction_necessity_object(
    *,
    adapter: Any,
    layer_index: int,
    interaction_spec: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    span_records: list[dict[str, Any]],
    max_length: int = 256,
) -> dict[str, Any]:
    span_index = _index_span_records(span_records, layer_index=layer_index)
    live_span_cache: dict[str, dict[str, Any]] = {}
    step_key = str(interaction_spec["decoder_step_key"])
    source_span_keys = [str(value) for value in list(interaction_spec["source_span_keys"])]
    head_indices = [int(value) for value in list(dict(interaction_spec["head_selector"]).get("indices") or [])]
    basis_indices = _basis_ids_to_indices(list(dict(interaction_spec["read_basis"]).get("basis_ids") or []))
    store_condition = dict(interaction_spec.get("store_condition") or {"mode": "none"})
    generic_basis_indices = _basis_ids_to_indices(list(interaction_spec.get("generic_basis_ids") or []))

    rows: list[dict[str, Any]] = []
    eligible = 0
    for item in candidate_records:
        item_span_map = _ensure_item_span_map(
            adapter=adapter,
            item=item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        alignments = align_decoder_steps(adapter.tokenizer, str(item["target_form"]), item)
        step_positions = list(alignments["alignments"].get(step_key) or [])
        if not step_positions:
            continue
        source_token_indices = _source_token_indices(item_span_map, source_span_keys)
        if not source_token_indices:
            continue
        eligible += 1
        baseline = _teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
        )
        compare_ids: list[int] = []
        baseline_summary = _token_margin_summary(
            logits=baseline["logits"],
            target_ids=baseline["target_ids"],
            step_positions=step_positions,
            compare_ids=compare_ids,
        )
        actual = _teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=_build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=head_indices, selected_scale=0.0),
            encoder_token_indices=source_token_indices if str(store_condition.get("mode")) != "none" else None,
            encoder_scale=0.0 if str(store_condition.get("mode")) != "none" else None,
            step_indices=step_positions,
            basis_indices=basis_indices,
            basis_mode="remove",
        )
        wrong_event = _teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=_build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=head_indices, selected_scale=0.0),
            encoder_token_indices=_source_token_indices(item_span_map, _wrong_source_span_keys(step_key)) if str(store_condition.get("mode")) != "none" else None,
            encoder_scale=0.0 if str(store_condition.get("mode")) != "none" else None,
            step_indices=step_positions,
            basis_indices=basis_indices,
            basis_mode="remove",
        )
        shuffled_heads = _shuffled_indices(int(getattr(adapter.model.config, "num_heads", 8) or 8), head_indices, len(head_indices), stable_seed(item["item_id"], interaction_spec["object_id"], "heads"))
        shuffled_head_output = _teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=_build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=shuffled_heads, selected_scale=0.0),
            step_indices=step_positions,
            basis_indices=basis_indices,
            basis_mode="remove",
        )
        hidden_size = int(baseline["decoder_hidden"].shape[-1]) if baseline["decoder_hidden"].ndim == 2 else 512
        shuffled_basis = _shuffled_indices(hidden_size, basis_indices, len(basis_indices), stable_seed(item["item_id"], interaction_spec["object_id"], "basis"))
        shuffled_basis_output = _teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=_build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=head_indices, selected_scale=0.0),
            step_indices=step_positions,
            basis_indices=shuffled_basis,
            basis_mode="remove",
        )
        generic_output = _teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            step_indices=step_positions,
            basis_indices=generic_basis_indices or basis_indices,
            basis_mode="remove",
        )

        def margin_mean(forward: dict[str, Any]) -> float:
            return float(
                _token_margin_summary(
                    logits=forward["logits"],
                    target_ids=forward["target_ids"],
                    step_positions=step_positions,
                    compare_ids=[],
                )["margin_mean"]
                or 0.0
            )

        baseline_margin = float(baseline_summary["margin_mean"] or 0.0)
        actual_drop = baseline_margin - margin_mean(actual)
        wrong_event_drop = baseline_margin - margin_mean(wrong_event)
        shuffled_head_drop = baseline_margin - margin_mean(shuffled_head_output)
        shuffled_basis_drop = baseline_margin - margin_mean(shuffled_basis_output)
        generic_drop = baseline_margin - margin_mean(generic_output)
        strongest_control = aggregate_control_value(
            {
                "wrong_event_same_step": actual_drop if False else wrong_event_drop,
                "shuffled_head_same_cardinality": shuffled_head_drop,
                "shuffled_basis_same_cardinality": shuffled_basis_drop,
                "generic_decoder_gain": generic_drop,
            },
            method="max",
            priority=(
                "wrong_event_same_step",
                "shuffled_head_same_cardinality",
                "shuffled_basis_same_cardinality",
                "generic_decoder_gain",
            ),
        )
        rows.append(
            {
                "item_id": str(item["item_id"]),
                "stratum": str(item["stratum"]),
                "generalization_case": str(item["generalization_case"]),
                "decoder_step_key": step_key,
                "baseline_margin": round(baseline_margin, 6),
                "actual_drop": round(actual_drop, 6),
                "wrong_event_same_step_drop": round(wrong_event_drop, 6),
                "shuffled_head_same_cardinality_drop": round(shuffled_head_drop, 6),
                "shuffled_basis_same_cardinality_drop": round(shuffled_basis_drop, 6),
                "generic_decoder_gain_drop": round(generic_drop, 6),
                "strongest_control_drop": round(float(strongest_control), 6),
                "necessity_gap": round(actual_drop - float(strongest_control), 6),
            }
        )

    h2_rows = [row for row in rows if str(row["stratum"]) == "pass"]
    h0_rows = [row for row in rows if str(row["stratum"]) == "control"]
    summary = {
        "object_id": str(interaction_spec["object_id"]),
        "row_count": len(rows),
        "eligible_item_count": eligible,
        "realism_pass_rate": round(len(rows) / max(eligible, 1), 6),
        "actual_drop": round(mean(float(row["actual_drop"]) for row in rows), 6) if rows else None,
        "strongest_control_drop": round(mean(float(row["strongest_control_drop"]) for row in rows), 6) if rows else None,
        "h2_gap": round(mean(float(row["necessity_gap"]) for row in h2_rows), 6) if h2_rows else None,
        "h0_gap": round(mean(float(row["necessity_gap"]) for row in h0_rows), 6) if h0_rows else None,
        "rows": rows,
    }
    return summary


def _gate_h2_row(
    *,
    interaction_object: dict[str, Any],
    dev_summary: dict[str, Any],
    heldout_summary: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    dev_h2_gap = dev_summary.get("h2_gap")
    dev_h0_gap = dev_summary.get("h0_gap")
    held_h2_gap = heldout_summary.get("h2_gap")
    held_h0_gap = heldout_summary.get("h0_gap")
    realism_pass_rate = min(float(dev_summary.get("realism_pass_rate") or 0.0), float(heldout_summary.get("realism_pass_rate") or 0.0))
    generic_overlap = float(interaction_object.get("generic_gain_overlap") or 0.0)
    if dev_h2_gap is None or float(dev_h2_gap) < H2_NECESSITY_MIN_GAP:
        reasons.append("dev_h2_gap_below_threshold")
    if held_h2_gap is None or float(held_h2_gap) < H2_NECESSITY_MIN_GAP:
        reasons.append("heldout_h2_gap_below_threshold")
    if dev_h0_gap is None or abs(float(dev_h0_gap)) > H0_NECESSITY_MAX_ABS_GAP:
        reasons.append("dev_h0_gap_above_null_band")
    if held_h0_gap is None or abs(float(held_h0_gap)) > H0_NECESSITY_MAX_ABS_GAP:
        reasons.append("heldout_h0_gap_above_null_band")
    if realism_pass_rate < REALISM_MIN_PASS_RATE:
        reasons.append("realism_below_threshold")
    if generic_overlap > GENERIC_GAIN_MAX_OVERLAP:
        reasons.append("generic_gain_overlap_too_high")
    return {
        "object_id": str(interaction_object["object_id"]),
        "object_class": GEOMETRY_INTERACTION_OBJECT_CLASS,
        "passed": not reasons,
        "gate_status": "pass" if not reasons else "fail_threshold",
        "failure_reasons": reasons,
        "dev_h2_gap": dev_h2_gap,
        "dev_h0_gap": dev_h0_gap,
        "heldout_h2_gap": held_h2_gap,
        "heldout_h0_gap": held_h0_gap,
        "dev_actual_drop": dev_summary.get("actual_drop"),
        "dev_strongest_control_drop": dev_summary.get("strongest_control_drop"),
        "heldout_actual_drop": heldout_summary.get("actual_drop"),
        "heldout_strongest_control_drop": heldout_summary.get("strongest_control_drop"),
        "realism_pass_rate": round(realism_pass_rate, 6),
        "generic_gain_overlap": generic_overlap,
    }


def summarize_interaction_h2_gate(rows: list[dict[str, Any]], discovery_summary: dict[str, Any]) -> dict[str, Any]:
    passed = [row for row in rows if bool(row.get("passed"))]
    decision = "continue_to_h1_gate" if passed else "T-supported"
    decision_status = "preliminary" if passed else "final"
    return {
        "gate_ready_candidate_count": len(rows),
        "passed_count": len(passed),
        "passed_candidate_ids": [str(row["object_id"]) for row in passed],
        "rows": rows,
        "decision": decision,
        "decision_status": decision_status,
        "discovery_summary": discovery_summary,
    }


def evaluate_interaction_rescue_object(
    *,
    adapter: Any,
    layer_index: int,
    interaction_spec: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    matched_sets: list[dict[str, Any]],
    span_records: list[dict[str, Any]],
    physical_family: str,
    max_length: int = 256,
) -> dict[str, Any]:
    span_index = _index_span_records(span_records, layer_index=layer_index)
    live_span_cache: dict[str, dict[str, Any]] = {}
    counterpart_index = build_exact_counterpart_index(candidate_records)
    step_key = str(interaction_spec["decoder_step_key"])
    source_span_keys = [str(value) for value in list(interaction_spec["source_span_keys"])]
    head_indices = [int(value) for value in list(dict(interaction_spec["head_selector"]).get("indices") or [])]
    basis_indices = _basis_ids_to_indices(list(dict(interaction_spec["read_basis"]).get("basis_ids") or []))
    store_condition = dict(interaction_spec.get("store_condition") or {"mode": "none"})
    generic_basis_indices = _basis_ids_to_indices(list(interaction_spec.get("generic_basis_ids") or []))

    rows: list[dict[str, Any]] = []
    for item in candidate_records:
        if str(item.get("stratum")) != "fail":
            continue
        item_span_map = _ensure_item_span_map(
            adapter=adapter,
            item=item,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        alignments = align_decoder_steps(adapter.tokenizer, str(item["target_form"]), item)
        step_positions = list(alignments["alignments"].get(step_key) or [])
        if not step_positions:
            continue
        role = "agent" if step_key == "agent_value" else "theme" if step_key == "theme_value" else "agent"
        counterpart = find_exact_role_counterpart(
            record=item,
            role=role,
            counterpart_index=counterpart_index,
            preferred_strata=("pass", "control"),
            same_split_only=False,
        )
        if counterpart is None:
            continue
        counterpart_span_map = _ensure_item_span_map(
            adapter=adapter,
            item=counterpart,
            span_index=span_index,
            live_span_cache=live_span_cache,
            layer_index=layer_index,
            max_length=max_length,
        )
        source_token_indices = _source_token_indices(item_span_map, source_span_keys)
        if not source_token_indices:
            continue
        baseline = _teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
        )
        baseline_summary = _token_margin_summary(
            logits=baseline["logits"],
            target_ids=baseline["target_ids"],
            step_positions=step_positions,
            compare_ids=_counterpart_compare_ids(
                tokenizer=adapter.tokenizer,
                step_key=step_key,
                target_positions=step_positions,
                counterpart_record=counterpart,
            ),
        )
        if bool(baseline_summary["exact_match"]):
            continue

        store_patch_indices, store_patch_vectors = _store_patch_vectors(
            target_span_map=item_span_map,
            donor_span_map=counterpart_span_map,
            source_span_keys=source_span_keys,
        )
        wrong_patch_indices, wrong_patch_vectors = _store_patch_vectors(
            target_span_map=item_span_map,
            donor_span_map=counterpart_span_map,
            source_span_keys=source_span_keys,
            donor_source_span_keys=_wrong_source_span_keys(step_key),
        )
        actual_only = _teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=_build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=head_indices, selected_scale=HEAD_BOOST_SCALE),
            step_indices=step_positions,
            basis_indices=basis_indices,
            basis_mode="boost",
            basis_scale=READ_BOOST_SCALE,
        )
        actual_plus_store = _teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=_build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=head_indices, selected_scale=HEAD_BOOST_SCALE),
            encoder_token_indices=store_patch_indices if str(store_condition.get("mode")) != "none" else None,
            encoder_replacement_vectors=store_patch_vectors if str(store_condition.get("mode")) != "none" else None,
            step_indices=step_positions,
            basis_indices=basis_indices,
            basis_mode="boost",
            basis_scale=READ_BOOST_SCALE,
        )
        wrong_reinstate = _teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=_build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=head_indices, selected_scale=HEAD_BOOST_SCALE),
            encoder_token_indices=wrong_patch_indices if str(store_condition.get("mode")) != "none" else None,
            encoder_replacement_vectors=wrong_patch_vectors if str(store_condition.get("mode")) != "none" else None,
            step_indices=step_positions,
            basis_indices=basis_indices,
            basis_mode="boost",
            basis_scale=READ_BOOST_SCALE,
        )
        path_only = _teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            head_mask=_build_cross_attn_head_mask(adapter.model, layer_index=layer_index, head_indices=head_indices, selected_scale=HEAD_BOOST_SCALE),
        )
        generic_gain = _teacher_forced_forward(
            adapter=adapter,
            input_text=str(item["input_text"]),
            target_text=str(item["target_form"]),
            layer_index=layer_index,
            max_length=max_length,
            step_indices=step_positions,
            basis_indices=generic_basis_indices or basis_indices,
            basis_mode="boost",
            basis_scale=READ_BOOST_SCALE,
        )

        def margin_gain(forward: dict[str, Any]) -> tuple[float, bool]:
            summary = _token_margin_summary(
                logits=forward["logits"],
                target_ids=forward["target_ids"],
                step_positions=step_positions,
                compare_ids=_counterpart_compare_ids(
                    tokenizer=adapter.tokenizer,
                    step_key=step_key,
                    target_positions=step_positions,
                    counterpart_record=counterpart,
                ),
            )
            return float(summary["margin_mean"] or 0.0) - float(baseline_summary["margin_mean"] or 0.0), bool(summary["exact_match"])

        actual_only_gain, actual_only_exact = margin_gain(actual_only)
        actual_plus_store_gain, actual_plus_store_exact = margin_gain(actual_plus_store)
        wrong_gain, wrong_exact = margin_gain(wrong_reinstate)
        path_gain, path_exact = margin_gain(path_only)
        generic_gain_value, generic_exact = margin_gain(generic_gain)
        strongest_control = aggregate_control_value(
            {
                "wrong_interaction_reinstate": wrong_gain,
                "path_only": path_gain,
                "generic_decoder_gain": generic_gain_value,
            },
            method="max",
            priority=("wrong_interaction_reinstate", "path_only", "generic_decoder_gain"),
        )
        best_actual = max(actual_only_gain, actual_plus_store_gain)
        rows.append(
            {
                "item_id": str(item["item_id"]),
                "generalization_case": str(item["generalization_case"]),
                "physical_family": str(physical_family),
                "decoder_step_key": step_key,
                "baseline_margin": round(float(baseline_summary["margin_mean"] or 0.0), 6),
                "object_reinstate_only_gain": round(actual_only_gain, 6),
                "interaction_reinstate_plus_store_gain": round(actual_plus_store_gain, 6),
                "wrong_interaction_reinstate_gain": round(wrong_gain, 6),
                "path_only_gain": round(path_gain, 6),
                "generic_decoder_gain": round(generic_gain_value, 6),
                "best_object_actual_gain": round(best_actual, 6),
                "strongest_control_gain": round(float(strongest_control), 6),
                "rescue_gap": round(best_actual - float(strongest_control), 6),
                "object_exact_match": bool(actual_only_exact or actual_plus_store_exact),
                "wrong_exact_match": bool(wrong_exact),
                "path_exact_match": bool(path_exact),
                "generic_exact_match": bool(generic_exact),
            }
        )
    return {
        "object_id": str(interaction_spec["object_id"]),
        "physical_family": physical_family,
        "row_count": len(rows),
        "rows": rows,
    }


def summarize_interaction_h1_gate(
    *,
    object_rows: list[dict[str, Any]],
    h2_summary: dict[str, Any],
) -> dict[str, Any]:
    if int(h2_summary.get("passed_count") or 0) <= 0:
        return {
            "blocked": True,
            "blocked_reason": "no_h2_passed_object",
            "passed_count": 0,
            "rows": [],
        }
    gate_rows: list[dict[str, Any]] = []
    grouped_object_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in object_rows:
        grouped_object_rows[str(row["object_id"])].extend(list(row.get("rows") or []))
    for object_id, flattened in grouped_object_rows.items():
        grouped_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item_row in flattened:
            grouped_by_family[str(item_row["physical_family"])].append(item_row)
        passing_family_ids: list[str] = []
        exact_hits = []
        generic_advantages = []
        for family_id, family_rows in grouped_by_family.items():
            family_rescue_gap = mean(float(item["rescue_gap"]) for item in family_rows)
            family_best_actual = mean(float(item["best_object_actual_gain"]) for item in family_rows)
            family_generic = mean(float(item["generic_decoder_gain"]) for item in family_rows)
            if family_rescue_gap >= H1_RESCUE_MIN_GAP:
                passing_family_ids.append(family_id)
            generic_advantages.append(round(family_best_actual - family_generic, 6))
            exact_hits.extend(int(bool(item["object_exact_match"])) for item in family_rows)
        exact_rate = round(mean(exact_hits), 6) if exact_hits else 0.0
        passed = (
            len(set(passing_family_ids)) >= 2
            and max(generic_advantages or [0.0]) >= H1_GENERIC_ADVANTAGE_MIN
            and min(generic_advantages or [0.0]) >= 0.0
            and exact_rate >= H1_RESCUE_MIN_BINARY_RATE
        )
        gate_rows.append(
            {
                "object_id": object_id,
                "passed": passed,
                "passing_families": sorted(set(passing_family_ids)),
                "aggregated_exact_match_rate": exact_rate,
                "generic_advantages": generic_advantages,
                "row_count": len(flattened),
            }
        )
    return {
        "blocked": False,
        "passed_count": sum(1 for row in gate_rows if bool(row["passed"])),
        "rows": gate_rows,
    }


def case_condition_interaction_objects(
    *,
    h2_summary: dict[str, Any],
    rescue_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if int(h2_summary.get("passed_count") or 0) <= 0:
        return {
            "blocked": True,
            "blocked_reason": "no_h2_passed_object",
            "accepted_slice_count": 0,
            "accepted_slices": [],
            "branch_decision": {
                "decision": "T-supported",
                "decision_status": "final",
                "explanation": "No interaction object passed the H2 necessity gate.",
            },
        }
    slices: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for object_row in rescue_rows:
        for row in list(object_row.get("rows") or []):
            key = (str(row["generalization_case"]), str(row["decoder_step_key"]))
            slices[key].append(row)
    accepted: list[dict[str, Any]] = []
    for (generalization_case, decoder_step_key), rows in slices.items():
        families = sorted({str(row["physical_family"]) for row in rows})
        rescue_gap = round(mean(float(row["rescue_gap"]) for row in rows), 6)
        exact_rate = round(mean(int(bool(row["object_exact_match"])) for row in rows), 6)
        if rescue_gap >= CASE_MIN_SLICE_RESCUE_GAP and len(families) > 1 and exact_rate >= CASE_MIN_SLICE_BINARY_RATE:
            accepted.append(
                {
                    "generalization_case": generalization_case,
                    "decoder_step_key": decoder_step_key,
                    "physical_families": families,
                    "slice_rescue_gap": rescue_gap,
                    "slice_exact_match_rescue_rate": exact_rate,
                }
            )
    return {
        "blocked": False,
        "accepted_slice_count": len(accepted),
        "accepted_slices": accepted,
        "branch_decision": {
            "decision": "L-supported" if len(accepted) >= 2 else "T-supported",
            "decision_status": "final",
            "explanation": (
                "At least two local slices support local-only H1."
                if len(accepted) >= 2
                else "No global or slice-local H1 rescue survived after the interaction-object gates."
            ),
        },
    }


def render_phase17_status_markdown(
    *,
    discovery_summary: dict[str, Any],
    h2_summary: dict[str, Any],
    h1_summary: dict[str, Any],
    case_summary: dict[str, Any],
) -> str:
    branch = dict(case_summary.get("branch_decision") or {})
    return "\n".join(
        [
            "# Phase 17 Status",
            "",
            "## Frozen Framing",
            "- `G1_binding`: bounded hypothesis-final success",
            "- `G2_distributed_geometry`: partial adjudication unless Phase 17 interaction objects pass the gated branch",
            "- Repository strongest framing stays `bounded protocol + partial adjudication` until Phase 17 reaches `U-supported`.",
            "",
            "## Discovery",
            f"- `object_count = {int(discovery_summary.get('object_count') or 0)}`",
            f"- `surviving_candidate_count = {int(discovery_summary.get('surviving_candidate_count') or 0)}`",
            f"- `accepted = {bool(discovery_summary.get('accepted'))}`",
            "",
            "## H2 Gate",
            f"- `passed_count = {int(h2_summary.get('passed_count') or 0)}`",
            f"- `decision = {h2_summary.get('decision')}`",
            f"- `decision_status = {h2_summary.get('decision_status')}`",
            "",
            "## H1 Gate",
            f"- `blocked = {bool(h1_summary.get('blocked'))}`",
            f"- `passed_count = {int(h1_summary.get('passed_count') or 0)}`",
            "",
            "## Case-Conditioned",
            f"- `accepted_slice_count = {int(case_summary.get('accepted_slice_count') or 0)}`",
            f"- `decision = {branch.get('decision')}`",
            f"- `decision_status = {branch.get('decision_status')}`",
            "",
            "## Interpretation",
            f"- {branch.get('explanation') or 'Phase 17 has not yet produced a branch decision.'}",
        ]
    ) + "\n"
