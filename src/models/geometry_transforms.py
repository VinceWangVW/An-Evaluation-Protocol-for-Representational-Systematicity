from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

from .routing_controls import first_tensor, replace_first_tensor


_MATRIX_CACHE: dict[tuple[str, int, int, str, str], torch.Tensor] = {}
_MASK_CACHE: dict[tuple[int, float, str, str], torch.Tensor] = {}


def apply_geometry_transform_tensor(
    hidden: torch.Tensor,
    *,
    transform: str | None,
    seed: int = 0,
    keep_ratio: float | None = None,
    noise_std: float | None = None,
) -> torch.Tensor:
    if transform is None or transform in {"", "identity"}:
        return hidden

    if hidden.ndim < 2:
        raise ValueError("Geometry transforms expect at least a 2D hidden-state tensor")

    if transform == "orthogonal_preserve":
        matrix = _orthogonal_matrix(
            hidden_size=int(hidden.shape[-1]),
            seed=seed,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return torch.matmul(hidden, matrix)

    if transform == "feature_mask":
        ratio = 0.25 if keep_ratio is None else float(keep_ratio)
        mask = _feature_mask(
            hidden_size=int(hidden.shape[-1]),
            keep_ratio=ratio,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return hidden * mask

    if transform == "gaussian_noise":
        std = 0.35 if noise_std is None else float(noise_std)
        generator = torch.Generator(device=hidden.device)
        generator.manual_seed(int(seed))
        noise = torch.randn(
            hidden.shape,
            generator=generator,
            device=hidden.device,
            dtype=torch.float32,
        ).to(hidden.dtype)
        return hidden + noise * std

    raise ValueError(f"Unknown encoder geometry transform: {transform}")


@contextmanager
def temporary_seq2seq_encoder_geometry_transform(
    model: Any,
    transform: str | None,
    *,
    seed: int = 0,
    keep_ratio: float | None = None,
    noise_std: float | None = None,
):
    if transform is None or transform in {"", "identity"}:
        yield
        return
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "block"):
        raise NotImplementedError("Encoder geometry transforms currently support encoder-decoder models with encoder blocks")

    blocks = list(model.encoder.block)
    if not blocks:
        raise NotImplementedError("Encoder geometry transforms require at least one encoder block")

    last_block = blocks[-1]

    def hook_fn(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        hidden = first_tensor(output).clone()
        transformed = apply_geometry_transform_tensor(
            hidden,
            transform=transform,
            seed=seed,
            keep_ratio=keep_ratio,
            noise_std=noise_std,
        )
        return replace_first_tensor(output, transformed)

    handle = last_block.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def _orthogonal_matrix(
    *,
    hidden_size: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = ("orthogonal_preserve", hidden_size, int(seed), str(device), str(dtype))
    cached = _MATRIX_CACHE.get(key)
    if cached is not None:
        return cached

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    basis = torch.randn((hidden_size, hidden_size), generator=generator, device=device, dtype=torch.float32)
    q, _r = torch.linalg.qr(basis, mode="reduced")
    matrix = q.to(dtype=dtype)
    _MATRIX_CACHE[key] = matrix
    return matrix


def _feature_mask(
    *,
    hidden_size: int,
    keep_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    clipped = max(0.0, min(1.0, float(keep_ratio)))
    key = (hidden_size, round(clipped, 6), str(device), str(dtype))
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached

    keep_dim = max(1, min(hidden_size, int(round(hidden_size * clipped))))
    mask = torch.zeros((hidden_size,), device=device, dtype=dtype)
    mask[:keep_dim] = 1
    _MASK_CACHE[key] = mask
    return mask


__all__ = [
    "apply_geometry_transform_tensor",
    "temporary_seq2seq_encoder_geometry_transform",
]
