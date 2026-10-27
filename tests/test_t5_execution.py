"""Execute real tiny-T5 training and interventions offline, without paper-result claims."""
import json
import math
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

from models.adapters import build_model_adapter
from models.behavior_seq2seq import HfSeq2SeqBehaviorModel, Seq2SeqBehaviorConfig


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_t5(tmp_path, monkeypatch):
    # Local fixtures use the same public HF interfaces as pretrained T5. Network
    # access is explicitly disabled so this check is useful in an offline checkout.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    examples = [json.loads(line) for line in (ROOT / "data/processed/phase11_systematicity_v2.jsonl").read_text().splitlines()][:7]
    special = ["<pad>", "</s>", "<unk>"]
    words = sorted({token for row in examples for field in ("input_text", "target_form") for token in row[field].split()})
    vocabulary = {token: index for index, token in enumerate(special + words)}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>",
        model_input_names=["input_ids", "attention_mask"], model_max_length=64,
    )
    tokenizer.save_pretrained(tmp_path / "initial")
    torch.manual_seed(17)
    model = T5ForConditionalGeneration(T5Config(
        vocab_size=len(vocabulary), d_model=16, d_kv=8, d_ff=32,
        num_layers=2, num_decoder_layers=2, num_heads=2, dropout_rate=0.0,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    ))
    model.save_pretrained(tmp_path / "initial")
    return tmp_path, examples


def test_tiny_t5_training_generation_and_targeted_intervention(tiny_t5):
    path, examples = tiny_t5
    behavior = HfSeq2SeqBehaviorModel(Seq2SeqBehaviorConfig(
        model_name_or_path=str(path / "initial"), device="cpu", precision="float32",
        train_batch_size=2, eval_batch_size=2, num_train_epochs=1, learning_rate=1e-3,
        max_source_length=32, max_target_length=16, max_new_tokens=4,
        seed=17, save_directory=str(path / "trained"),
    ))
    initial_weight = behavior.model.shared.weight.detach().clone()
    behavior.fit(examples[:2])
    assert behavior.training_summary["trained"] is True
    assert math.isfinite(behavior.training_summary["mean_epoch_loss"])
    assert not torch.equal(initial_weight, behavior.model.shared.weight)
    predictions = behavior.predict_many(examples[:2])
    assert len(predictions) == 2 and all(isinstance(value, str) for value in predictions)

    adapter = build_model_adapter(str(path / "trained"), family="seq2seq_lm", device="cpu", precision="float32")
    text = examples[0]["input_text"]
    filler = examples[0]["role_signature"]["agent"]
    start = text.lower().index(filler)
    probe = {
        "probe_id": "agent", "namespace": "semantic_role", "label": "agent",
        "surface_text": text[start:start + len(filler)], "start_char": start,
        "end_char": start + len(filler),
    }
    trace = adapter.trace_text(text, probes=[probe], layer_indices=[1])
    assert len(trace["span_vectors"]) == 1
    original = trace["span_vectors"][0]
    assert len(original["vector"]) == 16 and original["token_indices"]
    patched = adapter.intervene_text(text, probes=[probe], layer_index=1, ablation_specs=[{
        "token_indices": original["token_indices"],
    }])
    assert patched["span_vectors"][0]["vector"] == [0.0] * 16
    assert original["vector"] != [0.0] * 16
    # The intervention hook must not leak into later runs.
    restored = adapter.trace_text(text, probes=[probe], layer_indices=[1])
    assert restored["span_vectors"][0]["vector"] == pytest.approx(original["vector"])

    encoder_block = adapter.layer_module_for_hidden_index(1)
    hooks_before = len(encoder_block._forward_hooks)
    seen_encoder_states = []
    observer = adapter.layer_module_for_hidden_index(2).register_forward_pre_hook(
        lambda _module, args: seen_encoder_states.append(args[0].detach().clone())
    )
    try:
        generated = adapter.generate_with_intervention(
            text, layer_index=1, ablation_specs=[{"token_indices": original["token_indices"]}],
            max_new_tokens=4,
        )
    finally:
        observer.remove()
    assert isinstance(generated["generated_text"], str)
    assert seen_encoder_states, "Generation must actually execute the encoder intervention path"
    for hidden in seen_encoder_states:
        assert torch.count_nonzero(hidden[:, original["token_indices"], :]) == 0
    assert len(encoder_block._forward_hooks) == hooks_before


def test_t5_adapter_preserves_relation_barrier_settings(tiny_t5):
    path, _examples = tiny_t5
    adapter = build_model_adapter(
        str(path / "initial"), family="seq2seq_lm", device="cpu",
        decoder_cross_attention_relation_barrier_keep_ratio=0.375,
        decoder_cross_attention_relation_barrier_seed=19,
        decoder_cross_attention_relation_barrier_scale=0.2,
    )
    assert adapter.decoder_cross_attention_relation_barrier_keep_ratio == 0.375
    assert adapter.decoder_cross_attention_relation_barrier_seed == 19
    assert adapter.decoder_cross_attention_relation_barrier_scale == 0.2
