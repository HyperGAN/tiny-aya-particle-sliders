"""Replay the measured Tiny Aya question-and-response A-PSR profile."""

from contextlib import ExitStack
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
from threading import RLock

import torch
from particle_sliders import prompt_steering

from . import protocol


RAW_PROBE_SHA256 = "2d352c55b9c770d89c58c08b2b46f453717e3fe8c98cdcf353caa65f8fd02bca"
LL_RAW_PROBE_SHA256 = "0cf389d5952a4fa373b912b08649ac4557393214524ebc953c5b3af8d55a9e38"
LL_RAW_PLAN_SHA256 = "4ff20a68cb0283bd10f22c2dc819d9c28f88995b67637a2901424e4cd8a1bc7a"
LL_RAW_CAPTURE_SHA256 = "b4b8414ba4faa5cc690b5d8ba00a30a8ef2bc901d14d9fb50dd9722f2e30c185"
NATIVE_IDENTITY_SHA256 = "8cefba4ad05ab68e98b768df34c77e0831f93644ac53bd93bca9ebd7732ad61a"
PROTOCOL_SHA256 = "30785c733ee1177cf30282fbd6929c6b1f3d83aacfebee06a7c665ccbb3df59a"
SHARED_PSR_SHA256 = "67ee03066e255fe105735820b0edba53757d943b93239b1279de707d23dbae59"
RUNTIME = {"torch": "2.14.1+cu130", "transformers": "4.57.6", "accelerate": "1.12.0"}
ATTRIBUTE_INSTRUCTION = (
    "Use only the evidence, overriding everyday knowledge. "
    "Documents have equal authority regardless of order or date. "
)
BASE_FILES = {
    "README.md": "640219fd6a7727b13a9fc23db5286acd0018b43d75902c29c6ac29abec0be8f1",
    "config.json": "458a9265b6a96e5ad4bbe1857716bd69e434f062f8b32fae48f5f06197d26384",
    "generation_config.json": "0efa696e1ebc0484f8e16fcd1141dba1e897c4699ff971052e43ce5d6d4bf4d0",
    "model-00001-of-00002.safetensors": "0590022d0efe06f12e08e1e94302f069dff404341b426f7435c6303725635fb1",
    "model-00002-of-00002.safetensors": "2ac3809931d113785058809f27775620e0b8aec6fbc7e163c9c09c08b5f90dab",
    "model.safetensors.index.json": "5e87be717ff88ed7fd1e03aa1de1399daef2f2d2e3ee539c3ad1b0ed9d1ae74e",
    "signatures/verification-instructions.txt": "ba7533fe9f4da42ca86463262de866b8b8d295633c65f1cde86b0d918ab4f321",
    "special_tokens_map.json": "2b7ce39630f8a18705331a9b32c3b2236e96e7410791f98a9f8b6494e3ab2ea6",
    "tokenizer.json": "2227ea9c52e8afb3f98bfed2679008b275f2664de69dfde174b374389eb0225d",
    "tokenizer_config.json": "fb56a3ef7659346aae196a2ab877355f1e7fd871c36843f709bb2e23c606d972",
}
ALPHA = 1
LAYERS = 36
WIDTH = 2048
ALLOCATOR_GIB = 11


def _check_file(path, expected):
    with Path(path).open("rb") as stream:
        actual = file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Question-and-response A-PSR source changed: {path}")


def _check_loader_inputs(snapshot):
    overrides = ("adapter_config.json", "model.safetensors", "chat_template.jinja", "added_tokens.json")
    if (any((snapshot / name).is_file() for name in overrides)
            or any((snapshot / "additional_chat_templates").glob("*.jinja"))):
        raise ValueError("Tiny Aya snapshot contains additional model or tokenizer loader inputs")


def _load_raw(path):
    with Path(path).open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
        if digest not in (RAW_PROBE_SHA256, LL_RAW_PROBE_SHA256):
            raise ValueError(f"Question-and-response A-PSR source changed: {path}")
        stream.seek(0)
        packet = torch.load(stream, map_location="cpu", weights_only=True)

    is_ll = digest == LL_RAW_PROBE_SHA256
    metadata = ("tiny-aya-a-psr-question-and-response-loglikelihood", 210) if is_ll else (
        "tiny-aya-a-psr-question-and-response", 630,
    )
    base = packet.get("base")
    if ((packet.get("type"), packet.get("completed")) != metadata or packet.get("weights") != "raw"
            or "ema_decay" not in packet or packet["ema_decay"] is not None
            or not isinstance(base, dict) or base.get("sha256") != NATIVE_IDENTITY_SHA256):
        raise ValueError("Expected the exact question-and-response RAW probe fit")
    if is_ll:
        plan, cohort = packet.get("plan"), packet.get("cohort")
        if (packet.get("objective") != "response_negative_log_likelihood_with_position_penalty"
                or not isinstance(plan, dict) or plan.get("sha256") != LL_RAW_PLAN_SHA256
                or not isinstance(cohort, dict) or cohort.get("sha256") != LL_RAW_CAPTURE_SHA256):
            raise ValueError("Unexpected provenance for the converged LL RAW probe fit")
    probes = packet["probes"]
    expected = {
        f"{layer}.{name}": shape for layer in range(LAYERS)
        for name, shape in (("probe.weight", (1, WIDTH)), ("probe.bias", (1,)), ("direction.weight", (1, WIDTH)))
    }
    if set(probes) != set(expected) or any(
        not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name] or tensor.dtype != torch.float32
        or tensor.device.type != "cpu" or not torch.isfinite(tensor).all() for name, tensor in probes.items()
    ):
        raise ValueError("The pinned RAW fit must contain 36 distinct finite FP32 probe states")
    return probes, digest


def _request(tokenizer, row, device):
    attributed = protocol.prompt(row, teacher=False)
    if not attributed.startswith(ATTRIBUTE_INSTRUCTION):
        raise ValueError("The original grounding prompt changed")

    def encode(text):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True,
            tokenize=True, return_tensors="pt", return_dict=True,
        )["input_ids"].to(device)

    neutral = encode(attributed.removeprefix(ATTRIBUTE_INSTRUCTION))
    neutral_ids = neutral[0].tolist()
    reference_ids = encode(attributed)[0].tolist()
    width = next((index for index, (left, right) in enumerate(zip(reversed(neutral_ids), reversed(reference_ids)))
                  if left != right), min(len(neutral_ids), len(reference_ids)))
    start = len(neutral_ids) - width
    suffix = tokenizer.decode(neutral_ids[start:], clean_up_tokenization_spaces=False)
    if width <= 1 or any(value not in suffix for value in (row["question"], row["context"], "Answer:")):
        raise ValueError("The question, evidence and answer marker must lie in the shared native token suffix")
    return neutral, start


def _generate(model, tokenizer, row, probes, *, prefill_only=False, raw_probe_sha256=RAW_PROBE_SHA256):
    if not isinstance(prefill_only, bool):
        raise TypeError("prefill_only must be a boolean")
    layers = model.model.layers
    device = next(model.parameters()).device
    inputs, start = _request(tokenizer, row, device)
    prefix = inputs.shape[1]
    traces = [[] for _ in layers]
    calls = [0 for _ in layers]

    def edit(index, probe):
        def before(_layer, args):
            hidden = args[0]
            prefill = calls[index] == 0
            width = prefix if prefill else 1
            if hidden.shape != (1, width, model.config.hidden_size):
                raise RuntimeError("Unexpected native question-and-response cache position")
            calls[index] += 1
            if prefill_only and not prefill:
                return args
            active = hidden[:, start:] if prefill else hidden
            steered, coefficient = probe(active, ALPHA)
            if prefill_only and (coefficient.shape != (1, prefix - start) or coefficient.dtype != torch.float32):
                raise RuntimeError("Expected one native FP32 coefficient per eligible prefill position")
            traces[index].append(coefficient.flatten())
            edited = torch.cat((hidden[:, :start], steered), dim=1) if prefill else steered
            return (edited, *args[1:])

        return before

    with torch.inference_mode(), ExitStack() as handles:
        for index, (layer, probe) in enumerate(zip(layers, probes, strict=True)):
            handles.callback(layer.register_forward_pre_hook(edit(index, probe)).remove)
        result = model.generate(
            input_ids=inputs, do_sample=False, use_cache=True, max_new_tokens=protocol.MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=model.generation_config.eos_token_id,
        )[0, prefix:]
        ids = result.tolist()
        if not 0 < len(ids) <= protocol.MAX_NEW_TOKENS or calls != [len(ids)] * len(layers):
            raise RuntimeError("Tiny Aya did not generate one native decoder call per output token")
        eligible = prefix - start if prefill_only else prefix - start + len(ids) - 1
        coefficients = torch.stack([torch.cat(values) for values in traces]).cpu()
        if coefficients.shape != (len(layers), eligible) or not torch.isfinite(coefficients).all():
            raise RuntimeError("Unexpected question-and-response A-PSR coefficient trace")
    trace = dict(output=tokenizer.decode(ids, skip_special_tokens=True), output_ids=ids,
                 input_tokens=prefix, generated_tokens=len(ids), prefill_start=start,
                 aligned_prefill_positions=prefix - start, eligible_positions=eligible,
                 coefficients=coefficients.tolist(), alpha=ALPHA, weights="raw", ema_decay=None,
                 base_revision=protocol.REVISION, raw_probe_sha256=raw_probe_sha256)
    if prefill_only:
        trace["prefill_only"] = True
    return trace


class QuestionResponseAPSR:
    """Opt in to the measured native profile using a local base snapshot and RAW probe."""

    def __init__(self, snapshot, raw_probe):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if {package: version(package) for package in RUNTIME} != RUNTIME:
            raise ValueError("Use the measured Torch, Transformers and Accelerate runtime")
        _check_file(protocol.__file__, PROTOCOL_SHA256)
        _check_file(prompt_steering.__file__, SHARED_PSR_SHA256)
        fitted, self._raw_probe_sha256 = _load_raw(raw_probe)
        snapshot = Path(snapshot)
        _check_loader_inputs(snapshot)
        for name, expected in BASE_FILES.items():
            _check_file(snapshot / name, expected)
        if torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one CUDA device before opening this native profile")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(ALLOCATOR_GIB * 2**30 / properties.total_memory, 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, loading = AutoModelForCausalLM.from_pretrained(
            snapshot, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
            trust_remote_code=False, device_map={"": "cuda:0"}, low_cpu_mem_usage=True, output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError("Original Tiny Aya pretrained weights did not load exactly")
        if (model.config.model_type != "cohere2" or model.config.hidden_size != WIDTH
                or len(model.model.layers) != LAYERS or not all(
                    value.is_cuda and value.dtype == torch.bfloat16 for value in model.parameters())
                or not all(value.is_cuda for value in model.buffers())):
            raise RuntimeError("The measured Tiny Aya CUDA base geometry or dtype changed")
        self._model = model.eval().requires_grad_(False)
        self._tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        self._probes = torch.nn.ModuleList([
            prompt_steering.PromptSteering(WIDTH, device="cuda:0") for _ in model.model.layers
        ]).eval().requires_grad_(False)
        self._probes.load_state_dict(fitted, strict=True)
        self._lock = RLock()

    def generate(self, context, question, *, options=None, prefill_only=False):
        """Use the fitted probes on the shared suffix, and on cached steps by default."""
        with self._lock:
            return _generate(self._model, self._tokenizer, dict(context=context, question=question, options=options),
                             self._probes, prefill_only=prefill_only, raw_probe_sha256=self._raw_probe_sha256)
