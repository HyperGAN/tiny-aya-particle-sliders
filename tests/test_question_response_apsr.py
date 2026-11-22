from contextlib import ExitStack
from threading import RLock

import pytest
import torch
from transformers import Cohere2Config, Cohere2ForCausalLM

from evidence_dial import question_response_apsr as profile
from particle_sliders.prompt_steering import PromptSteering


ROW = {"context": "One source calls it blue.", "question": "What color?", "options": None}


class Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        content = messages[0]["content"]
        if content.startswith(profile.ATTRIBUTE_INSTRUCTION):
            return {"input_ids": torch.tensor([[1, 8, 4, 5, 6]])}
        assert not content.startswith("Use only the evidence")
        return {"input_ids": torch.tensor([[1, 7, 4, 5, 6]])}

    def decode(self, ids, **kwargs):
        if kwargs.get("clean_up_tokenization_spaces") is False:
            assert ids == [4, 5, 6]
            return f"{ROW['context']}\n{ROW['question']}\nAnswer:"
        return ",".join(str(token) for token in ids)


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(profile.protocol, "MAX_NEW_TOKENS", 4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = Cohere2ForCausalLM(Cohere2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, pad_token_id=0,
            bos_token_id=1, eos_token_id=3, sliding_window=16,
        )).eval().requires_grad_(False)
        model.generation_config.min_new_tokens = 2
        probes = [PromptSteering(16).eval().requires_grad_(False) for _ in model.model.layers]
        with torch.no_grad():
            for index, probe in enumerate(probes):
                probe.probe.bias.fill_(index + 1)
    return model, Tokenizer(), probes


def test_exact_question_response_scope_matches_archived_hook(native):
    model, tokenizer, probes = native
    traces = [[] for _ in probes]
    inputs = torch.tensor([[1, 7, 4, 5, 6]])
    start = 2

    def archived_hook(index):
        def edit(_layer, args):
            hidden = args[0]
            prefill = not traces[index]
            active = hidden[:, start:] if prefill else hidden
            steered, coefficient = probes[index](active, 1)
            traces[index].append(coefficient.flatten())
            return (torch.cat((hidden[:, :start], steered), dim=1), *args[1:]) if prefill else (steered, *args[1:])

        return edit

    state = {(name, id(value), value._version) for name, value in model.named_parameters()}
    with torch.inference_mode(), ExitStack() as handles:
        for index, layer in enumerate(model.model.layers):
            handles.callback(layer.register_forward_pre_hook(archived_hook(index)).remove)
        original = model.generate(input_ids=inputs, do_sample=False, use_cache=True, max_new_tokens=4,
                                  pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=model.generation_config.eos_token_id)[0, inputs.shape[1]:].tolist()
    result = profile._generate(model, tokenizer, ROW, probes)
    assert result["output_ids"] == original
    assert result["prefill_start"] == start and result["aligned_prefill_positions"] == 3
    assert result["eligible_positions"] == 3 + len(original) - 1
    assert "prefill_only" not in result
    assert result["coefficients"] == [torch.cat(layer).tolist() for layer in traces]
    assert result["weights"] == "raw" and result["ema_decay"] is None
    assert {(name, id(value), value._version) for name, value in model.named_parameters()} == state
    assert not any(layer._forward_pre_hooks for layer in model.model.layers)


def test_prefill_only_scope_matches_native_prefill_with_unedited_cached_inputs(native):
    model, tokenizer, probes = native
    inputs = torch.tensor([[1, 7, 4, 5, 6]])
    start = 2
    coefficients = []

    def register_prefill(layer, probe):
        def edit(_layer, args):
            handle.remove()
            hidden = args[0]
            steered, coefficient = probe(hidden[:, start:], 1)
            coefficients.append(coefficient.flatten().tolist())
            return (torch.cat((hidden[:, :start], steered), dim=1), *args[1:])

        handle = layer.register_forward_pre_hook(edit)
        return handle

    with torch.inference_mode(), ExitStack() as handles:
        for layer, probe in zip(model.model.layers, probes, strict=True):
            handles.callback(register_prefill(layer, probe).remove)
        expected = model.generate(input_ids=inputs, do_sample=False, use_cache=True, max_new_tokens=4,
                                  pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=model.generation_config.eos_token_id)[0, inputs.shape[1]:].tolist()

    engine = profile.QuestionResponseAPSR.__new__(profile.QuestionResponseAPSR)
    engine._model, engine._tokenizer, engine._probes = native
    engine._lock = RLock()
    with ExitStack() as handles:
        for probe in probes:
            def prefill_only(_probe, args):
                assert args[0].shape == (1, 3, model.config.hidden_size)

            handles.callback(probe.register_forward_pre_hook(prefill_only).remove)
        result = engine.generate(**ROW, prefill_only=True)
    assert len(expected) > 1
    assert result["output_ids"] == expected
    assert result["prefill_only"] is True
    assert result["eligible_positions"] == result["aligned_prefill_positions"] == 3
    assert result["coefficients"] == coefficients
    assert result["weights"] == "raw" and result["ema_decay"] is None
    assert not any(layer._forward_pre_hooks for layer in model.model.layers)


def test_question_response_hooks_release_after_failure(native):
    model, tokenizer, probes = native

    def stop(_layer, _args, _result):
        raise RuntimeError("fixture failed during native forward")

    with model.lm_head.register_forward_hook(stop):
        with pytest.raises(RuntimeError, match="fixture failed"):
            profile._generate(model, tokenizer, ROW, probes)
    assert not any(layer._forward_pre_hooks for layer in model.model.layers)


def test_question_response_rejects_another_raw_fit(tmp_path):
    dummy = tmp_path / "other-fit.pt"
    dummy.write_bytes(b"not the measured probe")
    with pytest.raises(ValueError, match="source changed"):
        profile._load_raw(dummy)


@pytest.mark.parametrize("extra", [
    "adapter_config.json", "model.safetensors", "chat_template.jinja", "added_tokens.json",
    "additional_chat_templates/default.jinja",
])
def test_question_response_rejects_extra_loader_inputs(tmp_path, extra):
    (tmp_path / "notes.txt").touch()
    profile._check_loader_inputs(tmp_path)
    extra_path = tmp_path / extra
    extra_path.parent.mkdir(parents=True, exist_ok=True)
    extra_path.touch()
    with pytest.raises(ValueError, match="additional model or tokenizer loader inputs"):
        profile._check_loader_inputs(tmp_path)
