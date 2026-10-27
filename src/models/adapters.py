from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from .geometry_access_controls import (
    temporary_seq2seq_cross_attention_bottleneck_projection,
    temporary_seq2seq_cross_attention_layer_keep_ratio,
    temporary_seq2seq_cross_attention_relation_barrier,
    temporary_seq2seq_decoder_readout_subspace_projection,
)
from .geometry_transforms import temporary_seq2seq_encoder_geometry_transform
from .routing_controls import (
    autocast_context,
    normalize_precision,
    temporary_seq2seq_cross_attention_feature_mask,
    temporary_seq2seq_cross_attention_scale,
    temporary_seq2seq_decoder_readout_feature_mask,
)


def build_prompt_text(prompt_style: str, input_text: str) -> str:
    if prompt_style == "raw":
        return input_text
    if prompt_style == "parser_instruction":
        return f"Parse the sentence into a logical form.\nSentence: {input_text}\nLogical form:"
    if prompt_style == "sentence_only":
        return f"Sentence: {input_text}"
    raise ValueError(f"Unknown prompt style: {prompt_style}")


class BaseModelAdapter:
    family: str = "base"

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Any,
        model: Any,
        device: str,
        prompt_style: str,
        precision: str = "auto",
        num_beams: int = 1,
        decoder_cross_attention_scale: float | None = None,
        decoder_cross_attention_keep_ratio: float | None = None,
        decoder_cross_attention_layer_keep_ratio: float | None = None,
        decoder_cross_attention_layer_seed: int = 0,
        decoder_cross_attention_layer_policy: str = "random",
        decoder_cross_attention_bottleneck_keep_ratio: float | None = None,
        decoder_cross_attention_bottleneck_seed: int = 0,
        decoder_cross_attention_bottleneck_mode: str = "keep",
        decoder_cross_attention_relation_barrier_keep_ratio: float | None = None,
        decoder_cross_attention_relation_barrier_seed: int = 0,
        decoder_cross_attention_relation_barrier_scale: float = 0.5,
        decoder_readout_keep_ratio: float | None = None,
        decoder_readout_subspace_keep_ratio: float | None = None,
        decoder_readout_subspace_seed: int = 0,
        decoder_readout_subspace_mode: str = "remove",
        encoder_geometry_transform: str | None = None,
        encoder_geometry_seed: int = 0,
        encoder_geometry_keep_ratio: float | None = None,
        encoder_geometry_noise_std: float | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.prompt_style = prompt_style
        self.precision = normalize_precision(precision)
        self.num_beams = max(1, int(num_beams))
        self.decoder_cross_attention_scale = decoder_cross_attention_scale
        self.decoder_cross_attention_keep_ratio = decoder_cross_attention_keep_ratio
        self.decoder_cross_attention_layer_keep_ratio = decoder_cross_attention_layer_keep_ratio
        self.decoder_cross_attention_layer_seed = int(decoder_cross_attention_layer_seed)
        self.decoder_cross_attention_layer_policy = decoder_cross_attention_layer_policy
        self.decoder_cross_attention_bottleneck_keep_ratio = decoder_cross_attention_bottleneck_keep_ratio
        self.decoder_cross_attention_bottleneck_seed = int(decoder_cross_attention_bottleneck_seed)
        self.decoder_cross_attention_bottleneck_mode = decoder_cross_attention_bottleneck_mode
        self.decoder_cross_attention_relation_barrier_keep_ratio = decoder_cross_attention_relation_barrier_keep_ratio
        self.decoder_cross_attention_relation_barrier_seed = int(decoder_cross_attention_relation_barrier_seed)
        self.decoder_cross_attention_relation_barrier_scale = float(decoder_cross_attention_relation_barrier_scale)
        self.decoder_readout_keep_ratio = decoder_readout_keep_ratio
        self.decoder_readout_subspace_keep_ratio = decoder_readout_subspace_keep_ratio
        self.decoder_readout_subspace_seed = int(decoder_readout_subspace_seed)
        self.decoder_readout_subspace_mode = decoder_readout_subspace_mode
        self.encoder_geometry_transform = encoder_geometry_transform
        self.encoder_geometry_seed = int(encoder_geometry_seed)
        self.encoder_geometry_keep_ratio = encoder_geometry_keep_ratio
        self.encoder_geometry_noise_std = encoder_geometry_noise_std

    def build_prompt(self, input_text: str) -> str:
        return build_prompt_text(self.prompt_style, input_text)

    def prompt_token_metadata(
        self,
        input_text: str,
        max_length: int = 256,
    ) -> dict[str, Any]:
        prompt_text = self.build_prompt(input_text)
        offsets, tokens, _inputs = self._encode_prompt(prompt_text=prompt_text, max_length=max_length)
        return {
            "prompt_text": prompt_text,
            "tokens": tokens,
            "offsets": offsets,
        }

    def trace_text(
        self,
        input_text: str,
        probes: list[dict[str, Any]],
        layer_indices: list[int],
        max_length: int = 256,
    ) -> dict[str, Any]:
        prompt_text = self.build_prompt(input_text)
        offsets, tokens, inputs = self._encode_prompt(prompt_text=prompt_text, max_length=max_length)

        hidden_states = self._forward_hidden_states(inputs)
        resolved_layers = resolve_layer_indices(layer_indices, len(hidden_states))
        span_vectors: list[dict[str, Any]] = []
        prompt_offset = input_offset_in_prompt(prompt_text=prompt_text, input_text=input_text)

        for probe in probes:
            prompt_start = prompt_offset + int(probe["start_char"])
            prompt_end = prompt_offset + int(probe["end_char"])
            token_indices = token_indices_for_span(
                offsets=offsets,
                start_char=prompt_start,
                end_char=prompt_end,
            )
            if not token_indices:
                continue

            probe_tokens = [tokens[index] for index in token_indices]
            for layer_index in resolved_layers:
                pooled = (
                    hidden_states[layer_index][0, token_indices, :].mean(dim=0).detach().float().cpu().tolist()
                )
                span_vectors.append(
                    {
                        "probe_id": str(probe["probe_id"]),
                        "namespace": str(probe["namespace"]),
                        "label": str(probe["label"]),
                        "surface_text": str(probe["surface_text"]),
                        "start_char": prompt_start,
                        "end_char": prompt_end,
                        "token_indices": token_indices,
                        "token_texts": probe_tokens,
                        "layer_index": layer_index,
                        "vector": pooled,
                    }
                )

        return {
            "prompt_text": prompt_text,
            "tokens": tokens,
            "span_vectors": span_vectors,
        }

    def intervene_text(
        self,
        input_text: str,
        probes: list[dict[str, Any]],
        layer_index: int,
        patch_specs: list[dict[str, Any]] | None = None,
        ablation_specs: list[dict[str, Any]] | None = None,
        max_length: int = 256,
    ) -> dict[str, Any]:
        prompt_text = self.build_prompt(input_text)
        offsets, tokens, inputs = self._encode_prompt(prompt_text=prompt_text, max_length=max_length)
        resolved_layer = resolve_layer_indices([layer_index], self.hidden_state_count())[0]
        prompt_offset = input_offset_in_prompt(prompt_text=prompt_text, input_text=input_text)
        layer_module = self.layer_module_for_hidden_index(resolved_layer)

        def hook_fn(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            hidden = first_tensor(output).clone()
            for spec in patch_specs or []:
                replacement = torch.tensor(spec["replacement_vector"], device=hidden.device, dtype=hidden.dtype)
                for token_index in list(spec["token_indices"]):
                    hidden[:, int(token_index), :] = replacement
            for spec in ablation_specs or []:
                replacement = torch.zeros(hidden.size(-1), device=hidden.device, dtype=hidden.dtype)
                for token_index in list(spec["token_indices"]):
                    hidden[:, int(token_index), :] = replacement
            return replace_first_tensor(output, hidden)

        handle = layer_module.register_forward_hook(hook_fn)
        try:
            hidden_states = self._forward_hidden_states(inputs)
        finally:
            handle.remove()

        span_vectors: list[dict[str, Any]] = []
        for probe in probes:
            prompt_start = prompt_offset + int(probe["start_char"])
            prompt_end = prompt_offset + int(probe["end_char"])
            token_indices = token_indices_for_span(
                offsets=offsets,
                start_char=prompt_start,
                end_char=prompt_end,
            )
            if not token_indices:
                continue
            pooled = (
                hidden_states[resolved_layer][0, token_indices, :].mean(dim=0).detach().float().cpu().tolist()
            )
            span_vectors.append(
                {
                    "probe_id": str(probe["probe_id"]),
                    "namespace": str(probe["namespace"]),
                    "label": str(probe["label"]),
                    "surface_text": str(probe["surface_text"]),
                    "start_char": prompt_start,
                    "end_char": prompt_end,
                    "token_indices": token_indices,
                    "token_texts": [tokens[index] for index in token_indices],
                    "layer_index": resolved_layer,
                    "vector": pooled,
                }
            )

        return {
            "prompt_text": prompt_text,
            "tokens": tokens,
            "span_vectors": span_vectors,
        }

    def generate_text(self, input_text: str, max_new_tokens: int = 64, num_beams: int | None = None) -> str:
        prompt_text = self.build_prompt(input_text)
        encoded = self.tokenizer(prompt_text, return_tensors="pt", truncation=True)
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with temporary_seq2seq_encoder_geometry_transform(
            self.model,
            self.encoder_geometry_transform if self.family == "seq2seq_lm" else None,
            seed=self.encoder_geometry_seed,
            keep_ratio=self.encoder_geometry_keep_ratio,
            noise_std=self.encoder_geometry_noise_std,
        ):
            with temporary_seq2seq_cross_attention_scale(self.model, self.decoder_cross_attention_scale):
                with temporary_seq2seq_cross_attention_feature_mask(
                    self.model,
                    self.decoder_cross_attention_keep_ratio,
                ):
                    with temporary_seq2seq_cross_attention_layer_keep_ratio(
                        self.model if self.family == "seq2seq_lm" else None,
                        self.decoder_cross_attention_layer_keep_ratio if self.family == "seq2seq_lm" else None,
                        seed=self.decoder_cross_attention_layer_seed,
                        policy=self.decoder_cross_attention_layer_policy,
                    ):
                        with temporary_seq2seq_cross_attention_bottleneck_projection(
                            self.model if self.family == "seq2seq_lm" else None,
                            self.decoder_cross_attention_bottleneck_keep_ratio if self.family == "seq2seq_lm" else None,
                            seed=self.decoder_cross_attention_bottleneck_seed,
                            mode=self.decoder_cross_attention_bottleneck_mode,
                        ):
                            with temporary_seq2seq_cross_attention_relation_barrier(
                                self.model if self.family == "seq2seq_lm" else None,
                                self.decoder_cross_attention_relation_barrier_keep_ratio if self.family == "seq2seq_lm" else None,
                                seed=self.decoder_cross_attention_relation_barrier_seed,
                                scale=self.decoder_cross_attention_relation_barrier_scale,
                            ):
                                with temporary_seq2seq_decoder_readout_feature_mask(
                                    self.model if self.family == "seq2seq_lm" else None,
                                    self.decoder_readout_keep_ratio if self.family == "seq2seq_lm" else None,
                                ):
                                    with temporary_seq2seq_decoder_readout_subspace_projection(
                                        self.model if self.family == "seq2seq_lm" else None,
                                        self.decoder_readout_subspace_keep_ratio if self.family == "seq2seq_lm" else None,
                                        seed=self.decoder_readout_subspace_seed,
                                        mode=self.decoder_readout_subspace_mode,
                                    ):
                                        with autocast_context(self.device, self.precision):
                                            generated = self.model.generate(
                                                **inputs,
                                                max_new_tokens=max_new_tokens,
                                                num_beams=max(1, int(num_beams or self.num_beams)),
                                                pad_token_id=self.tokenizer.pad_token_id,
                                            )
        return self._decode_generated(prompt_text, generated)

    def generate_with_intervention(
        self,
        input_text: str,
        layer_index: int,
        patch_specs: list[dict[str, Any]] | None = None,
        ablation_specs: list[dict[str, Any]] | None = None,
        control_overrides: dict[str, Any] | None = None,
        max_length: int = 256,
        max_new_tokens: int = 64,
        num_beams: int | None = None,
    ) -> dict[str, Any]:
        prompt_text = self.build_prompt(input_text)
        _offsets, tokens, inputs = self._encode_prompt(prompt_text=prompt_text, max_length=max_length)
        resolved_layer = resolve_layer_indices([layer_index], self.hidden_state_count())[0]
        layer_module = self.layer_module_for_hidden_index(resolved_layer)
        overrides = dict(control_overrides or {})

        def control_value(name: str, default: Any) -> Any:
            return overrides.get(name, default)

        def hook_fn(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            hidden = first_tensor(output).clone()
            for spec in patch_specs or []:
                replacement = torch.tensor(spec["replacement_vector"], device=hidden.device, dtype=hidden.dtype)
                for token_index in list(spec["token_indices"]):
                    hidden[:, int(token_index), :] = replacement
            for spec in ablation_specs or []:
                replacement = torch.zeros(hidden.size(-1), device=hidden.device, dtype=hidden.dtype)
                for token_index in list(spec["token_indices"]):
                    hidden[:, int(token_index), :] = replacement
            return replace_first_tensor(output, hidden)

        handle = layer_module.register_forward_hook(hook_fn)
        try:
            with temporary_seq2seq_encoder_geometry_transform(
                self.model,
                control_value("encoder_geometry_transform", self.encoder_geometry_transform)
                if self.family == "seq2seq_lm"
                else None,
                seed=int(control_value("encoder_geometry_seed", self.encoder_geometry_seed)),
                keep_ratio=control_value("encoder_geometry_keep_ratio", self.encoder_geometry_keep_ratio),
                noise_std=control_value("encoder_geometry_noise_std", self.encoder_geometry_noise_std),
            ):
                with temporary_seq2seq_cross_attention_scale(
                    self.model,
                    control_value("decoder_cross_attention_scale", self.decoder_cross_attention_scale),
                ):
                    with temporary_seq2seq_cross_attention_feature_mask(
                        self.model,
                        control_value("decoder_cross_attention_keep_ratio", self.decoder_cross_attention_keep_ratio),
                    ):
                        with temporary_seq2seq_cross_attention_layer_keep_ratio(
                            self.model if self.family == "seq2seq_lm" else None,
                            control_value(
                                "decoder_cross_attention_layer_keep_ratio",
                                self.decoder_cross_attention_layer_keep_ratio,
                            )
                            if self.family == "seq2seq_lm"
                            else None,
                            seed=int(
                                control_value(
                                    "decoder_cross_attention_layer_seed",
                                    self.decoder_cross_attention_layer_seed,
                                )
                            ),
                            policy=str(
                                control_value(
                                    "decoder_cross_attention_layer_policy",
                                    self.decoder_cross_attention_layer_policy,
                                )
                            ),
                        ):
                            with temporary_seq2seq_cross_attention_bottleneck_projection(
                                self.model if self.family == "seq2seq_lm" else None,
                                control_value(
                                    "decoder_cross_attention_bottleneck_keep_ratio",
                                    self.decoder_cross_attention_bottleneck_keep_ratio,
                                )
                                if self.family == "seq2seq_lm"
                                else None,
                                seed=int(
                                    control_value(
                                        "decoder_cross_attention_bottleneck_seed",
                                        self.decoder_cross_attention_bottleneck_seed,
                                    )
                                ),
                                mode=str(
                                    control_value(
                                        "decoder_cross_attention_bottleneck_mode",
                                        self.decoder_cross_attention_bottleneck_mode,
                                    )
                                ),
                            ):
                                with temporary_seq2seq_cross_attention_relation_barrier(
                                    self.model if self.family == "seq2seq_lm" else None,
                                    control_value(
                                        "decoder_cross_attention_relation_barrier_keep_ratio",
                                        self.decoder_cross_attention_relation_barrier_keep_ratio,
                                    )
                                    if self.family == "seq2seq_lm"
                                    else None,
                                    seed=int(
                                        control_value(
                                            "decoder_cross_attention_relation_barrier_seed",
                                            self.decoder_cross_attention_relation_barrier_seed,
                                        )
                                    ),
                                    scale=float(
                                        control_value(
                                            "decoder_cross_attention_relation_barrier_scale",
                                            self.decoder_cross_attention_relation_barrier_scale,
                                        )
                                    ),
                                ):
                                    with temporary_seq2seq_decoder_readout_feature_mask(
                                        self.model if self.family == "seq2seq_lm" else None,
                                        control_value("decoder_readout_keep_ratio", self.decoder_readout_keep_ratio)
                                        if self.family == "seq2seq_lm"
                                        else None,
                                    ):
                                        with temporary_seq2seq_decoder_readout_subspace_projection(
                                            self.model if self.family == "seq2seq_lm" else None,
                                            control_value(
                                                "decoder_readout_subspace_keep_ratio",
                                                self.decoder_readout_subspace_keep_ratio,
                                            )
                                            if self.family == "seq2seq_lm"
                                            else None,
                                            seed=int(
                                                control_value(
                                                    "decoder_readout_subspace_seed",
                                                    self.decoder_readout_subspace_seed,
                                                )
                                            ),
                                            mode=str(
                                                control_value(
                                                    "decoder_readout_subspace_mode",
                                                    self.decoder_readout_subspace_mode,
                                                )
                                            ),
                                        ):
                                            with autocast_context(self.device, self.precision):
                                                generated = self.model.generate(
                                                    **inputs,
                                                    max_new_tokens=max_new_tokens,
                                                    num_beams=max(1, int(num_beams or self.num_beams)),
                                                    pad_token_id=self.tokenizer.pad_token_id,
                                                )
        finally:
            handle.remove()

        return {
            "prompt_text": prompt_text,
            "tokens": tokens,
            "generated_text": self._decode_generated(prompt_text, generated),
            "layer_index": resolved_layer,
        }

    def _decode_generated(self, prompt_text: str, generated: Any) -> str:
        decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        if self.family == "causal_lm" and decoded.startswith(prompt_text):
            return decoded[len(prompt_text) :].strip()
        return decoded.strip()

    def _forward_hidden_states(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        raise NotImplementedError

    def _layer_modules(self) -> list[Any]:
        raise NotImplementedError

    def hidden_state_count(self) -> int:
        return len(self._layer_modules()) + 1

    def layer_module_for_hidden_index(self, hidden_index: int) -> Any:
        if hidden_index == 0:
            raise ValueError("Embedding layer interventions are not supported")
        modules = self._layer_modules()
        module_index = hidden_index - 1
        if module_index < 0 or module_index >= len(modules):
            raise IndexError(f"Hidden-state index {hidden_index} is out of range")
        return modules[module_index]

    def _encode_prompt(
        self,
        prompt_text: str,
        max_length: int,
    ) -> tuple[list[list[int]], list[str], dict[str, torch.Tensor]]:
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        return offsets, tokens, inputs


class CausalLMAdapter(BaseModelAdapter):
    family = "causal_lm"

    def _forward_hidden_states(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        return list(outputs.hidden_states)

    def _layer_modules(self) -> list[Any]:
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return list(self.model.gpt_neox.layers)
        if hasattr(self.model, "layers"):
            return list(self.model.layers)
        raise NotImplementedError("Unsupported causal LM architecture for interventions")


class Seq2SeqLMAdapter(BaseModelAdapter):
    family = "seq2seq_lm"

    def _forward_hidden_states(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        decoder_start_token_id = self.model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
        decoder_input_ids = torch.tensor([[decoder_start_token_id]], device=self.device)
        with temporary_seq2seq_encoder_geometry_transform(
            self.model,
            self.encoder_geometry_transform,
            seed=self.encoder_geometry_seed,
            keep_ratio=self.encoder_geometry_keep_ratio,
            noise_std=self.encoder_geometry_noise_std,
        ):
            with temporary_seq2seq_cross_attention_scale(self.model, self.decoder_cross_attention_scale):
                with temporary_seq2seq_cross_attention_feature_mask(
                    self.model,
                    self.decoder_cross_attention_keep_ratio,
                ):
                    with temporary_seq2seq_cross_attention_layer_keep_ratio(
                        self.model,
                        self.decoder_cross_attention_layer_keep_ratio,
                        seed=self.decoder_cross_attention_layer_seed,
                        policy=self.decoder_cross_attention_layer_policy,
                    ):
                        with temporary_seq2seq_cross_attention_bottleneck_projection(
                            self.model,
                            self.decoder_cross_attention_bottleneck_keep_ratio,
                            seed=self.decoder_cross_attention_bottleneck_seed,
                            mode=self.decoder_cross_attention_bottleneck_mode,
                        ):
                            with temporary_seq2seq_cross_attention_relation_barrier(
                                self.model,
                                self.decoder_cross_attention_relation_barrier_keep_ratio,
                                seed=self.decoder_cross_attention_relation_barrier_seed,
                                scale=self.decoder_cross_attention_relation_barrier_scale,
                            ):
                                with torch.inference_mode():
                                    outputs = self.model(
                                        **inputs,
                                        decoder_input_ids=decoder_input_ids,
                                        output_hidden_states=True,
                                        return_dict=True,
                                    )
        return list(outputs.encoder_hidden_states)

    def _layer_modules(self) -> list[Any]:
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "block"):
            return list(self.model.encoder.block)
        raise NotImplementedError("Unsupported seq2seq architecture for interventions")


def build_model_adapter(
    model_name_or_path: str,
    family: str = "auto",
    device: str | None = None,
    prompt_style: str | None = None,
    precision: str = "auto",
    num_beams: int = 1,
    decoder_cross_attention_scale: float | None = None,
    decoder_cross_attention_keep_ratio: float | None = None,
    decoder_cross_attention_layer_keep_ratio: float | None = None,
    decoder_cross_attention_layer_seed: int = 0,
    decoder_cross_attention_layer_policy: str = "random",
    decoder_cross_attention_bottleneck_keep_ratio: float | None = None,
    decoder_cross_attention_bottleneck_seed: int = 0,
    decoder_cross_attention_bottleneck_mode: str = "keep",
    decoder_cross_attention_relation_barrier_keep_ratio: float | None = None,
    decoder_cross_attention_relation_barrier_seed: int = 0,
    decoder_cross_attention_relation_barrier_scale: float = 0.5,
    decoder_readout_keep_ratio: float | None = None,
    decoder_readout_subspace_keep_ratio: float | None = None,
    decoder_readout_subspace_seed: int = 0,
    decoder_readout_subspace_mode: str = "remove",
    encoder_geometry_transform: str | None = None,
    encoder_geometry_seed: int = 0,
    encoder_geometry_keep_ratio: float | None = None,
    encoder_geometry_noise_std: float | None = None,
) -> BaseModelAdapter:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model_name_or_path.startswith("toy://"):
        tokenizer = WhitespaceTokenizer()
        model = ToyCausalLM(vocab_size=tokenizer.vocab_size)
        model.to(resolved_device)
        model.eval()
        return CausalLMAdapter(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            model=model,
            device=resolved_device,
            prompt_style=prompt_style or "parser_instruction",
            precision=precision,
            num_beams=num_beams,
            decoder_cross_attention_scale=decoder_cross_attention_scale,
            decoder_cross_attention_keep_ratio=decoder_cross_attention_keep_ratio,
            decoder_cross_attention_layer_keep_ratio=decoder_cross_attention_layer_keep_ratio,
            decoder_cross_attention_layer_seed=decoder_cross_attention_layer_seed,
            decoder_cross_attention_layer_policy=decoder_cross_attention_layer_policy,
            decoder_cross_attention_bottleneck_keep_ratio=decoder_cross_attention_bottleneck_keep_ratio,
            decoder_cross_attention_bottleneck_seed=decoder_cross_attention_bottleneck_seed,
            decoder_cross_attention_bottleneck_mode=decoder_cross_attention_bottleneck_mode,
            decoder_cross_attention_relation_barrier_keep_ratio=decoder_cross_attention_relation_barrier_keep_ratio,
            decoder_cross_attention_relation_barrier_seed=decoder_cross_attention_relation_barrier_seed,
            decoder_cross_attention_relation_barrier_scale=decoder_cross_attention_relation_barrier_scale,
            decoder_readout_keep_ratio=decoder_readout_keep_ratio,
            decoder_readout_subspace_keep_ratio=decoder_readout_subspace_keep_ratio,
            decoder_readout_subspace_seed=decoder_readout_subspace_seed,
            decoder_readout_subspace_mode=decoder_readout_subspace_mode,
            encoder_geometry_transform=encoder_geometry_transform,
            encoder_geometry_seed=encoder_geometry_seed,
            encoder_geometry_keep_ratio=encoder_geometry_keep_ratio,
            encoder_geometry_noise_std=encoder_geometry_noise_std,
        )

    resolved_family = family
    if resolved_family == "auto":
        config = AutoConfig.from_pretrained(model_name_or_path)
        resolved_family = "seq2seq_lm" if bool(getattr(config, "is_encoder_decoder", False)) else "causal_lm"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_prompt_style = prompt_style or (
        "raw" if resolved_family == "seq2seq_lm" else "parser_instruction"
    )
    if resolved_family == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model.to(resolved_device)
        model.eval()
        return CausalLMAdapter(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            model=model,
            device=resolved_device,
            prompt_style=resolved_prompt_style,
            precision=precision,
            num_beams=num_beams,
            decoder_cross_attention_scale=decoder_cross_attention_scale,
            decoder_cross_attention_keep_ratio=decoder_cross_attention_keep_ratio,
            decoder_cross_attention_layer_keep_ratio=decoder_cross_attention_layer_keep_ratio,
            decoder_cross_attention_layer_seed=decoder_cross_attention_layer_seed,
            decoder_cross_attention_layer_policy=decoder_cross_attention_layer_policy,
        decoder_cross_attention_bottleneck_keep_ratio=decoder_cross_attention_bottleneck_keep_ratio,
        decoder_cross_attention_bottleneck_seed=decoder_cross_attention_bottleneck_seed,
        decoder_cross_attention_bottleneck_mode=decoder_cross_attention_bottleneck_mode,
        decoder_cross_attention_relation_barrier_keep_ratio=decoder_cross_attention_relation_barrier_keep_ratio,
        decoder_cross_attention_relation_barrier_seed=decoder_cross_attention_relation_barrier_seed,
        decoder_cross_attention_relation_barrier_scale=decoder_cross_attention_relation_barrier_scale,
        decoder_readout_keep_ratio=decoder_readout_keep_ratio,
            decoder_readout_subspace_keep_ratio=decoder_readout_subspace_keep_ratio,
            decoder_readout_subspace_seed=decoder_readout_subspace_seed,
            decoder_readout_subspace_mode=decoder_readout_subspace_mode,
            encoder_geometry_transform=encoder_geometry_transform,
            encoder_geometry_seed=encoder_geometry_seed,
            encoder_geometry_keep_ratio=encoder_geometry_keep_ratio,
            encoder_geometry_noise_std=encoder_geometry_noise_std,
        )
    if resolved_family == "seq2seq_lm":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        model.to(resolved_device)
        model.eval()
        return Seq2SeqLMAdapter(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            model=model,
            device=resolved_device,
            prompt_style=resolved_prompt_style,
            precision=precision,
            num_beams=num_beams,
            decoder_cross_attention_scale=decoder_cross_attention_scale,
            decoder_cross_attention_keep_ratio=decoder_cross_attention_keep_ratio,
            decoder_cross_attention_layer_keep_ratio=decoder_cross_attention_layer_keep_ratio,
            decoder_cross_attention_layer_seed=decoder_cross_attention_layer_seed,
            decoder_cross_attention_layer_policy=decoder_cross_attention_layer_policy,
            decoder_cross_attention_bottleneck_keep_ratio=decoder_cross_attention_bottleneck_keep_ratio,
            decoder_cross_attention_bottleneck_seed=decoder_cross_attention_bottleneck_seed,
            decoder_cross_attention_bottleneck_mode=decoder_cross_attention_bottleneck_mode,
            decoder_cross_attention_relation_barrier_keep_ratio=decoder_cross_attention_relation_barrier_keep_ratio,
            decoder_cross_attention_relation_barrier_seed=decoder_cross_attention_relation_barrier_seed,
            decoder_cross_attention_relation_barrier_scale=decoder_cross_attention_relation_barrier_scale,
            decoder_readout_keep_ratio=decoder_readout_keep_ratio,
            decoder_readout_subspace_keep_ratio=decoder_readout_subspace_keep_ratio,
            decoder_readout_subspace_seed=decoder_readout_subspace_seed,
            decoder_readout_subspace_mode=decoder_readout_subspace_mode,
            encoder_geometry_transform=encoder_geometry_transform,
            encoder_geometry_seed=encoder_geometry_seed,
            encoder_geometry_keep_ratio=encoder_geometry_keep_ratio,
            encoder_geometry_noise_std=encoder_geometry_noise_std,
        )
    raise ValueError(f"Unknown model family: {resolved_family}")


def resolve_layer_indices(layer_indices: list[int], total_layers: int) -> list[int]:
    resolved = []
    for layer_index in layer_indices:
        resolved_index = layer_index if layer_index >= 0 else total_layers + layer_index
        if resolved_index < 0 or resolved_index >= total_layers:
            raise IndexError(f"Layer index {layer_index} is out of range for {total_layers} layers")
        resolved.append(resolved_index)
    return resolved


def token_indices_for_span(
    offsets: list[list[int]],
    start_char: int,
    end_char: int,
) -> list[int]:
    indices = []
    for index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        if token_end <= start_char or token_start >= end_char:
            continue
        indices.append(index)
    return indices


def input_offset_in_prompt(prompt_text: str, input_text: str) -> int:
    index = prompt_text.find(input_text)
    return index if index >= 0 else 0


class WhitespaceTokenizer:
    def __init__(self, vocab_size: int = 4096) -> None:
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.token_to_id = {
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
        }
        self.id_to_token = {
            self.pad_token_id: self.pad_token,
            self.eos_token_id: self.eos_token,
        }

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        return_offsets_mapping: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        del return_tensors
        token_matches = list(re.finditer(r"\S+", text))
        tokens = [match.group(0) for match in token_matches]
        offsets = [list(match.span()) for match in token_matches]

        if truncation and max_length is not None:
            tokens = tokens[:max_length]
            offsets = offsets[:max_length]
        if not tokens:
            tokens = [self.eos_token]
            offsets = [[0, 0]]

        input_ids = [self._token_id(token) for token in tokens]
        encoded = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        }
        if return_offsets_mapping:
            encoded["offset_mapping"] = torch.tensor([offsets], dtype=torch.long)
        return encoded

    def convert_ids_to_tokens(self, input_ids: Any) -> list[str]:
        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.detach().cpu().tolist()
        else:
            ids = list(input_ids)
        return [self.id_to_token.get(int(token_id), "<unk>") for token_id in ids]

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        tokens = self.convert_ids_to_tokens(token_ids)
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in {self.pad_token, self.eos_token}]
        return " ".join(tokens)

    def _token_id(self, token: str) -> int:
        lowered = token.lower()
        if lowered in self.token_to_id:
            return int(self.token_to_id[lowered])

        if len(self.token_to_id) < self.vocab_size:
            token_id = len(self.token_to_id)
        else:
            token_id = 2 + (abs(hash(lowered)) % max(1, self.vocab_size - 2))
        self.token_to_id[lowered] = token_id
        self.id_to_token.setdefault(token_id, token)
        return token_id


class ToyCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 96,
        num_layers: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        hidden = self.embedding(input_ids)
        hidden_states = [hidden]
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = attention_mask.eq(0) if attention_mask is not None else None

        for layer in self.layers:
            hidden = layer(hidden, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
            hidden_states.append(hidden)

        logits = self.lm_head(hidden)
        if not return_dict:
            return logits, tuple(hidden_states)
        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        pad_token_id: int | None = None,
        **_: Any,
    ) -> torch.Tensor:
        del attention_mask, max_new_tokens, pad_token_id
        return input_ids


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
