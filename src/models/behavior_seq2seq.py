from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .adapters import build_prompt_text
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


@dataclass(slots=True)
class Seq2SeqBehaviorConfig:
    model_name_or_path: str
    prompt_style: str = "raw"
    device: str | None = None
    precision: str = "auto"
    max_source_length: int = 128
    max_target_length: int = 192
    max_new_tokens: int = 192
    num_beams: int = 1
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 1
    train_limit: int | None = None
    train_target_permutation_seed: int | None = None
    decoder_cross_attention_scale: float | None = None
    decoder_cross_attention_keep_ratio: float | None = None
    decoder_cross_attention_layer_keep_ratio: float | None = None
    decoder_cross_attention_layer_seed: int = 0
    decoder_cross_attention_layer_policy: str = "random"
    decoder_cross_attention_bottleneck_keep_ratio: float | None = None
    decoder_cross_attention_bottleneck_seed: int = 0
    decoder_cross_attention_bottleneck_mode: str = "keep"
    decoder_cross_attention_relation_barrier_keep_ratio: float | None = None
    decoder_cross_attention_relation_barrier_seed: int = 0
    decoder_cross_attention_relation_barrier_scale: float = 0.5
    decoder_readout_keep_ratio: float | None = None
    decoder_readout_subspace_keep_ratio: float | None = None
    decoder_readout_subspace_seed: int = 0
    decoder_readout_subspace_mode: str = "remove"
    encoder_geometry_transform: str | None = None
    encoder_geometry_seed: int = 0
    encoder_geometry_keep_ratio: float | None = None
    encoder_geometry_noise_std: float | None = None
    seed: int = 0
    save_directory: str | None = None


class _Seq2SeqDataset(Dataset):
    def __init__(self, prompts: list[str], targets: list[str]) -> None:
        self.prompts = prompts
        self.targets = targets

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {
            "prompt": self.prompts[index],
            "target": self.targets[index],
        }


class _PromptDataset(Dataset):
    def __init__(self, prompts: list[str]) -> None:
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> str:
        return self.prompts[index]


class HfSeq2SeqBehaviorModel:
    def __init__(self, config: Seq2SeqBehaviorConfig) -> None:
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = normalize_precision(config.precision)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path)
        self.model.to(self.device)
        self.training_summary: dict[str, Any] = {
            "trained": False,
            "epochs": config.num_train_epochs,
            "learning_rate": config.learning_rate,
        }
        seed_everything(config.seed)

    def fit(self, train_examples: list[dict[str, Any]]) -> None:
        if self.config.num_train_epochs <= 0:
            self.training_summary["trained"] = False
            self.training_summary["train_example_count"] = 0
            self.training_summary["train_limit"] = self.config.train_limit
            self.training_summary["train_target_permutation_seed"] = self.config.train_target_permutation_seed
            self.training_summary["precision"] = self.precision
            self.training_summary["num_beams"] = self.config.num_beams
            self.training_summary["decoder_cross_attention_scale"] = self.config.decoder_cross_attention_scale
            self.training_summary["decoder_cross_attention_keep_ratio"] = self.config.decoder_cross_attention_keep_ratio
            self.training_summary["decoder_cross_attention_layer_keep_ratio"] = self.config.decoder_cross_attention_layer_keep_ratio
            self.training_summary["decoder_cross_attention_layer_seed"] = self.config.decoder_cross_attention_layer_seed
            self.training_summary["decoder_cross_attention_layer_policy"] = self.config.decoder_cross_attention_layer_policy
            self.training_summary["decoder_cross_attention_bottleneck_keep_ratio"] = self.config.decoder_cross_attention_bottleneck_keep_ratio
            self.training_summary["decoder_cross_attention_bottleneck_seed"] = self.config.decoder_cross_attention_bottleneck_seed
            self.training_summary["decoder_cross_attention_bottleneck_mode"] = self.config.decoder_cross_attention_bottleneck_mode
            self.training_summary["decoder_cross_attention_relation_barrier_keep_ratio"] = self.config.decoder_cross_attention_relation_barrier_keep_ratio
            self.training_summary["decoder_cross_attention_relation_barrier_seed"] = self.config.decoder_cross_attention_relation_barrier_seed
            self.training_summary["decoder_cross_attention_relation_barrier_scale"] = self.config.decoder_cross_attention_relation_barrier_scale
            self.training_summary["decoder_readout_keep_ratio"] = self.config.decoder_readout_keep_ratio
            self.training_summary["decoder_readout_subspace_keep_ratio"] = self.config.decoder_readout_subspace_keep_ratio
            self.training_summary["decoder_readout_subspace_seed"] = self.config.decoder_readout_subspace_seed
            self.training_summary["decoder_readout_subspace_mode"] = self.config.decoder_readout_subspace_mode
            self.training_summary["encoder_geometry_transform"] = self.config.encoder_geometry_transform
            self.training_summary["encoder_geometry_seed"] = self.config.encoder_geometry_seed
            self.training_summary["encoder_geometry_keep_ratio"] = self.config.encoder_geometry_keep_ratio
            self.training_summary["encoder_geometry_noise_std"] = self.config.encoder_geometry_noise_std
            return

        effective_examples = list(train_examples)
        if self.config.train_limit is not None and self.config.train_limit < len(effective_examples):
            sampler = random.Random(self.config.seed)
            sampler.shuffle(effective_examples)
            effective_examples = effective_examples[: self.config.train_limit]

        if self.config.train_target_permutation_seed is not None and effective_examples:
            permuted_targets = [str(example["target_form"]) for example in effective_examples]
            shuffler = random.Random(self.config.train_target_permutation_seed)
            shuffler.shuffle(permuted_targets)
            effective_examples = [
                {
                    **example,
                    "target_form": permuted_targets[index],
                }
                for index, example in enumerate(effective_examples)
            ]

        prompts = [build_prompt_text(self.config.prompt_style, str(example["input_text"])) for example in effective_examples]
        targets = [str(example["target_form"]) for example in effective_examples]
        dataset = _Seq2SeqDataset(prompts=prompts, targets=targets)
        loader = DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=self._collate_train,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.model.train()
        epoch_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)

        with temporary_seq2seq_encoder_geometry_transform(
            self.model,
            self.config.encoder_geometry_transform,
            seed=self.config.encoder_geometry_seed,
            keep_ratio=self.config.encoder_geometry_keep_ratio,
            noise_std=self.config.encoder_geometry_noise_std,
        ):
            with temporary_seq2seq_cross_attention_scale(self.model, self.config.decoder_cross_attention_scale):
                with temporary_seq2seq_cross_attention_feature_mask(
                    self.model,
                    self.config.decoder_cross_attention_keep_ratio,
                ):
                    with temporary_seq2seq_cross_attention_layer_keep_ratio(
                        self.model,
                        self.config.decoder_cross_attention_layer_keep_ratio,
                        seed=self.config.decoder_cross_attention_layer_seed,
                        policy=self.config.decoder_cross_attention_layer_policy,
                    ):
                        with temporary_seq2seq_cross_attention_bottleneck_projection(
                            self.model,
                            self.config.decoder_cross_attention_bottleneck_keep_ratio,
                            seed=self.config.decoder_cross_attention_bottleneck_seed,
                            mode=self.config.decoder_cross_attention_bottleneck_mode,
                        ):
                            with temporary_seq2seq_cross_attention_relation_barrier(
                                self.model,
                                self.config.decoder_cross_attention_relation_barrier_keep_ratio,
                                seed=self.config.decoder_cross_attention_relation_barrier_seed,
                                scale=self.config.decoder_cross_attention_relation_barrier_scale,
                            ):
                                with temporary_seq2seq_decoder_readout_feature_mask(
                                    self.model,
                                    self.config.decoder_readout_keep_ratio,
                                ):
                                    with temporary_seq2seq_decoder_readout_subspace_projection(
                                        self.model,
                                        self.config.decoder_readout_subspace_keep_ratio,
                                        seed=self.config.decoder_readout_subspace_seed,
                                        mode=self.config.decoder_readout_subspace_mode,
                                    ):
                                        for _epoch in range(self.config.num_train_epochs):
                                            batch_losses: list[float] = []
                                            for step, batch in enumerate(loader, start=1):
                                                outputs = self.model(**batch)
                                                loss = outputs.loss / self.config.gradient_accumulation_steps
                                                loss.backward()
                                                batch_losses.append(float(loss.detach().cpu()) * self.config.gradient_accumulation_steps)

                                                if step % self.config.gradient_accumulation_steps == 0:
                                                    optimizer.step()
                                                    optimizer.zero_grad(set_to_none=True)

                                            if len(loader) % self.config.gradient_accumulation_steps != 0:
                                                optimizer.step()
                                                optimizer.zero_grad(set_to_none=True)

                                            if batch_losses:
                                                epoch_losses.append(mean(batch_losses))

        self.model.eval()
        self.training_summary = {
            "trained": True,
            "epochs": self.config.num_train_epochs,
            "learning_rate": self.config.learning_rate,
            "train_example_count": len(effective_examples),
            "train_limit": self.config.train_limit,
            "train_target_permutation_seed": self.config.train_target_permutation_seed,
            "steps_per_epoch": math.ceil(len(effective_examples) / max(1, self.config.train_batch_size)),
            "mean_epoch_loss": round(mean(epoch_losses), 6) if epoch_losses else None,
            "epoch_losses": [round(loss, 6) for loss in epoch_losses],
            "precision": self.precision,
            "num_beams": self.config.num_beams,
            "decoder_cross_attention_scale": self.config.decoder_cross_attention_scale,
            "decoder_cross_attention_keep_ratio": self.config.decoder_cross_attention_keep_ratio,
            "decoder_cross_attention_layer_keep_ratio": self.config.decoder_cross_attention_layer_keep_ratio,
            "decoder_cross_attention_layer_seed": self.config.decoder_cross_attention_layer_seed,
            "decoder_cross_attention_layer_policy": self.config.decoder_cross_attention_layer_policy,
            "decoder_cross_attention_bottleneck_keep_ratio": self.config.decoder_cross_attention_bottleneck_keep_ratio,
            "decoder_cross_attention_bottleneck_seed": self.config.decoder_cross_attention_bottleneck_seed,
            "decoder_cross_attention_bottleneck_mode": self.config.decoder_cross_attention_bottleneck_mode,
            "decoder_cross_attention_relation_barrier_keep_ratio": self.config.decoder_cross_attention_relation_barrier_keep_ratio,
            "decoder_cross_attention_relation_barrier_seed": self.config.decoder_cross_attention_relation_barrier_seed,
            "decoder_cross_attention_relation_barrier_scale": self.config.decoder_cross_attention_relation_barrier_scale,
            "decoder_readout_keep_ratio": self.config.decoder_readout_keep_ratio,
            "decoder_readout_subspace_keep_ratio": self.config.decoder_readout_subspace_keep_ratio,
            "decoder_readout_subspace_seed": self.config.decoder_readout_subspace_seed,
            "decoder_readout_subspace_mode": self.config.decoder_readout_subspace_mode,
            "encoder_geometry_transform": self.config.encoder_geometry_transform,
            "encoder_geometry_seed": self.config.encoder_geometry_seed,
            "encoder_geometry_keep_ratio": self.config.encoder_geometry_keep_ratio,
            "encoder_geometry_noise_std": self.config.encoder_geometry_noise_std,
        }
        if self.config.save_directory:
            save_path = Path(self.config.save_directory)
            save_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.training_summary["save_directory"] = str(save_path)

    def predict_many(self, eval_examples: list[dict[str, Any]]) -> list[str]:
        prompts = [build_prompt_text(self.config.prompt_style, str(example["input_text"])) for example in eval_examples]
        dataset = _PromptDataset(prompts)
        loader = DataLoader(
            dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=self._collate_infer,
        )

        predictions: list[str] = []
        self.model.eval()
        with torch.inference_mode():
            with temporary_seq2seq_encoder_geometry_transform(
                self.model,
                self.config.encoder_geometry_transform,
                seed=self.config.encoder_geometry_seed,
                keep_ratio=self.config.encoder_geometry_keep_ratio,
                noise_std=self.config.encoder_geometry_noise_std,
            ):
                with temporary_seq2seq_cross_attention_scale(self.model, self.config.decoder_cross_attention_scale):
                    with temporary_seq2seq_cross_attention_feature_mask(
                        self.model,
                        self.config.decoder_cross_attention_keep_ratio,
                    ):
                        with temporary_seq2seq_cross_attention_layer_keep_ratio(
                            self.model,
                            self.config.decoder_cross_attention_layer_keep_ratio,
                            seed=self.config.decoder_cross_attention_layer_seed,
                            policy=self.config.decoder_cross_attention_layer_policy,
                        ):
                            with temporary_seq2seq_cross_attention_bottleneck_projection(
                                self.model,
                                self.config.decoder_cross_attention_bottleneck_keep_ratio,
                                seed=self.config.decoder_cross_attention_bottleneck_seed,
                                mode=self.config.decoder_cross_attention_bottleneck_mode,
                            ):
                                with temporary_seq2seq_cross_attention_relation_barrier(
                                    self.model,
                                    self.config.decoder_cross_attention_relation_barrier_keep_ratio,
                                    seed=self.config.decoder_cross_attention_relation_barrier_seed,
                                    scale=self.config.decoder_cross_attention_relation_barrier_scale,
                                ):
                                    with temporary_seq2seq_decoder_readout_feature_mask(
                                        self.model,
                                        self.config.decoder_readout_keep_ratio,
                                    ):
                                        with temporary_seq2seq_decoder_readout_subspace_projection(
                                            self.model,
                                            self.config.decoder_readout_subspace_keep_ratio,
                                            seed=self.config.decoder_readout_subspace_seed,
                                            mode=self.config.decoder_readout_subspace_mode,
                                        ):
                                            for batch in loader:
                                                with autocast_context(self.device, self.precision):
                                                    generated = self.model.generate(
                                                        **batch,
                                                        max_new_tokens=self.config.max_new_tokens,
                                                        num_beams=max(1, int(self.config.num_beams)),
                                                        pad_token_id=self.tokenizer.pad_token_id,
                                                    )
                                                predictions.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return [prediction.strip() for prediction in predictions]

    def _collate_train(self, batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        targets = [item["target"] for item in batch]
        encoded_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_source_length,
        )
        encoded_targets = self.tokenizer(
            text_target=targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_target_length,
        )
        labels = encoded_targets["input_ids"]
        labels = labels.masked_fill(labels.eq(self.tokenizer.pad_token_id), -100)
        return {
            "input_ids": encoded_inputs["input_ids"].to(self.device),
            "attention_mask": encoded_inputs["attention_mask"].to(self.device),
            "labels": labels.to(self.device),
        }

    def _collate_infer(self, batch: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            list(batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_source_length,
        )
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
