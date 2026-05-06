from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any

import torch

_FEATURE_MASK_CACHE: dict[tuple[int, float, str, str], torch.Tensor] = {}


def first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    raise TypeError(f"Unsupported layer output type: {type(output)!r}")


def replace_first_tensor(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple):
        return (replacement,) + output[1:]
    raise TypeError(f"Unsupported layer output type: {type(output)!r}")


@contextmanager
def temporary_seq2seq_cross_attention_scale(model: Any, scale: float | None):
    if scale is None or abs(float(scale) - 1.0) < 1e-8:
        yield
        return
    if not hasattr(model, "decoder") or not hasattr(model.decoder, "block"):
        raise NotImplementedError("Cross-attention scaling currently supports encoder-decoder models with decoder blocks")

    handles = []
    for block in list(model.decoder.block):
        if not hasattr(block, "layer") or len(block.layer) < 2:
            continue
        layer = block.layer[1]
        if not hasattr(layer, "EncDecAttention"):
            continue

        def hook_fn(_module: Any, _args: tuple[Any, ...], output: Any, *, scale_value: float = float(scale)) -> Any:
            hidden = first_tensor(output).clone()
            hidden.mul_(scale_value)
            return replace_first_tensor(output, hidden)

        handles.append(layer.EncDecAttention.register_forward_hook(hook_fn))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def temporary_seq2seq_cross_attention_feature_mask(model: Any, keep_ratio: float | None):
    if keep_ratio is None or abs(float(keep_ratio) - 1.0) < 1e-8:
        yield
        return
    if not hasattr(model, "decoder") or not hasattr(model.decoder, "block"):
        raise NotImplementedError("Cross-attention feature masking currently supports encoder-decoder models with decoder blocks")

    handles = []
    for block in list(model.decoder.block):
        if not hasattr(block, "layer") or len(block.layer) < 2:
            continue
        layer = block.layer[1]
        if not hasattr(layer, "EncDecAttention"):
            continue

        def hook_fn(_module: Any, _args: tuple[Any, ...], output: Any, *, ratio: float = float(keep_ratio)) -> Any:
            hidden = first_tensor(output).clone()
            mask = _feature_mask(
                hidden_size=int(hidden.shape[-1]),
                keep_ratio=ratio,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            hidden.mul_(mask)
            return replace_first_tensor(output, hidden)

        handles.append(layer.EncDecAttention.register_forward_hook(hook_fn))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def temporary_seq2seq_decoder_readout_feature_mask(model: Any, keep_ratio: float | None):
    if keep_ratio is None or abs(float(keep_ratio) - 1.0) < 1e-8:
        yield
        return
    if not hasattr(model, "lm_head"):
        raise NotImplementedError("Decoder readout masking currently supports seq2seq models with lm_head")

    def pre_hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args:
            return args
        hidden = args[0]
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(f"Unsupported lm_head input type: {type(hidden)!r}")
        masked = hidden.clone()
        mask = _feature_mask(
            hidden_size=int(masked.shape[-1]),
            keep_ratio=float(keep_ratio),
            device=masked.device,
            dtype=masked.dtype,
        )
        masked.mul_(mask)
        if len(args) == 1:
            return (masked,)
        return (masked, *args[1:])

    handle = model.lm_head.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


def autocast_context(device: str, precision: str):
    normalized = normalize_precision(precision)
    if not device.startswith("cuda") or normalized == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if normalized == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def normalize_precision(precision: str | None) -> str:
    if precision is None or precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp32"
    return str(precision)


def _feature_mask(
    *,
    hidden_size: int,
    keep_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    clipped = max(0.0, min(1.0, float(keep_ratio)))
    key = (hidden_size, round(clipped, 6), str(device), str(dtype))
    cached = _FEATURE_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    keep_dim = max(1, min(hidden_size, int(round(hidden_size * clipped))))
    mask = torch.zeros((hidden_size,), device=device, dtype=dtype)
    mask[:keep_dim] = 1
    _FEATURE_MASK_CACHE[key] = mask
    return mask
