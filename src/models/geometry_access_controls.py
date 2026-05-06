from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

from .routing_controls import first_tensor, replace_first_tensor


_SUBSPACE_CACHE: dict[tuple[int, float, int, str, str], torch.Tensor] = {}


def apply_random_subspace_projection(
    hidden: torch.Tensor,
    *,
    keep_ratio: float,
    seed: int = 0,
    mode: str = "remove",
) -> torch.Tensor:
    if hidden.ndim < 2:
        raise ValueError("Subspace projection expects at least a 2D hidden-state tensor")
    projector = _random_subspace_projector(
        hidden_size=int(hidden.shape[-1]),
        keep_ratio=float(keep_ratio),
        seed=int(seed),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    projected = torch.matmul(hidden, projector)
    if mode == "keep":
        return projected
    if mode == "remove":
        return hidden - projected
    raise ValueError(f"Unsupported subspace projection mode: {mode}")


@contextmanager
def temporary_seq2seq_cross_attention_layer_keep_ratio(
    model: Any,
    keep_ratio: float | None,
    *,
    seed: int = 0,
    policy: str = "random",
):
    if keep_ratio is None or abs(float(keep_ratio) - 1.0) < 1e-8:
        yield
        return
    if not hasattr(model, "decoder") or not hasattr(model.decoder, "block"):
        raise NotImplementedError("Cross-attention layer blocking currently supports encoder-decoder models with decoder blocks")

    blocks = list(model.decoder.block)
    if not blocks:
        yield
        return

    kept_indices = _choose_kept_block_indices(
        total=len(blocks),
        keep_ratio=float(keep_ratio),
        seed=int(seed),
        policy=policy,
    )
    handles = []
    for block_index, block in enumerate(blocks):
        if block_index in kept_indices:
            continue
        if not hasattr(block, "layer") or len(block.layer) < 2:
            continue
        layer = block.layer[1]
        if not hasattr(layer, "EncDecAttention"):
            continue

        def hook_fn(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            hidden = first_tensor(output).clone()
            hidden.zero_()
            return replace_first_tensor(output, hidden)

        handles.append(layer.EncDecAttention.register_forward_hook(hook_fn))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def temporary_seq2seq_cross_attention_bottleneck_projection(
    model: Any,
    keep_ratio: float | None,
    *,
    seed: int = 0,
    mode: str = "keep",
):
    if keep_ratio is None or abs(float(keep_ratio) - 1.0) < 1e-8:
        yield
        return
    if not hasattr(model, "decoder") or not hasattr(model.decoder, "block"):
        raise NotImplementedError("Cross-attention bottleneck projection currently supports encoder-decoder models with decoder blocks")

    handles = []
    for block in list(model.decoder.block):
        if not hasattr(block, "layer") or len(block.layer) < 2:
            continue
        layer = block.layer[1]
        if not hasattr(layer, "EncDecAttention"):
            continue

        def hook_fn(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            ratio: float = float(keep_ratio),
            seed_value: int = int(seed),
            mode_value: str = str(mode),
        ) -> Any:
            hidden = first_tensor(output).clone()
            transformed = apply_random_subspace_projection(
                hidden,
                keep_ratio=ratio,
                seed=seed_value,
                mode=mode_value,
            )
            return replace_first_tensor(output, transformed)

        handles.append(layer.EncDecAttention.register_forward_hook(hook_fn))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def temporary_seq2seq_cross_attention_relation_barrier(
    model: Any,
    keep_ratio: float | None,
    *,
    seed: int = 0,
    scale: float = 0.5,
):
    if keep_ratio is None or abs(float(keep_ratio) - 1.0) < 1e-8:
        yield
        return
    if not hasattr(model, "decoder") or not hasattr(model.decoder, "block"):
        raise NotImplementedError("Cross-attention relation barrier currently supports encoder-decoder models with decoder blocks")

    handles = []
    for block in list(model.decoder.block):
        if not hasattr(block, "layer") or len(block.layer) < 2:
            continue
        layer = block.layer[1]
        if not hasattr(layer, "EncDecAttention"):
            continue

        def hook_fn(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            ratio: float = float(keep_ratio),
            seed_value: int = int(seed),
            scale_value: float = float(scale),
        ) -> Any:
            hidden = first_tensor(output).clone()
            relation_subspace = apply_random_subspace_projection(
                hidden,
                keep_ratio=ratio,
                seed=seed_value,
                mode="keep",
            )
            residual = hidden - relation_subspace
            transformed = residual + (relation_subspace * scale_value)
            return replace_first_tensor(output, transformed)

        handles.append(layer.EncDecAttention.register_forward_hook(hook_fn))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def temporary_seq2seq_decoder_readout_subspace_projection(
    model: Any,
    keep_ratio: float | None,
    *,
    seed: int = 0,
    mode: str = "remove",
):
    if keep_ratio is None or abs(float(keep_ratio) - 1.0) < 1e-8:
        yield
        return
    if not hasattr(model, "lm_head"):
        raise NotImplementedError("Decoder readout subspace projection currently supports seq2seq models with lm_head")

    def pre_hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args:
            return args
        hidden = args[0]
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(f"Unsupported lm_head input type: {type(hidden)!r}")
        transformed = apply_random_subspace_projection(
            hidden,
            keep_ratio=float(keep_ratio),
            seed=int(seed),
            mode=mode,
        )
        if len(args) == 1:
            return (transformed,)
        return (transformed, *args[1:])

    handle = model.lm_head.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


def _choose_kept_block_indices(*, total: int, keep_ratio: float, seed: int, policy: str = "random") -> set[int]:
    clipped = max(0.0, min(1.0, float(keep_ratio)))
    if clipped <= 0.0:
        return set()
    keep_count = min(total, max(1, int(round(total * clipped))))
    normalized_policy = str(policy or "random")
    if normalized_policy == "top":
        return set(range(keep_count))
    if normalized_policy == "bottom":
        return set(range(total - keep_count, total))
    if normalized_policy in {"alternating", "even"}:
        preferred = [index for index in range(total) if index % 2 == 0]
        fallback = [index for index in range(total) if index % 2 == 1]
        return set((preferred + fallback)[:keep_count])
    if normalized_policy == "odd":
        preferred = [index for index in range(total) if index % 2 == 1]
        fallback = [index for index in range(total) if index % 2 == 0]
        return set((preferred + fallback)[:keep_count])
    if normalized_policy != "random":
        raise ValueError(f"Unsupported cross-attention layer policy: {policy}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(total, generator=generator).tolist()
    return set(sorted(order[:keep_count]))


def _random_subspace_projector(
    *,
    hidden_size: int,
    keep_ratio: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    clipped = max(0.0, min(1.0, float(keep_ratio)))
    key = (hidden_size, round(clipped, 6), int(seed), str(device), str(dtype))
    cached = _SUBSPACE_CACHE.get(key)
    if cached is not None:
        return cached

    rank = max(1, min(hidden_size, int(round(hidden_size * clipped))))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    basis = torch.randn((hidden_size, rank), generator=generator, device=device, dtype=torch.float32)
    q, _ = torch.linalg.qr(basis, mode="reduced")
    projector = torch.matmul(q, q.transpose(0, 1)).to(dtype=dtype)
    _SUBSPACE_CACHE[key] = projector
    return projector


__all__ = [
    "apply_random_subspace_projection",
    "temporary_seq2seq_cross_attention_bottleneck_projection",
    "temporary_seq2seq_cross_attention_layer_keep_ratio",
    "temporary_seq2seq_cross_attention_relation_barrier",
    "temporary_seq2seq_decoder_readout_subspace_projection",
]
