"""Command-R acceptance parity on a pinned public tiny checkpoint.

Applies Adapters/acceptance_test_notes.md to TransformerBridge: legacy
HookedTransformer has no Cohere loader. This is adapter acceptance, not a
production-checkpoint or generation-quality certification.
"""

from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "trl-internal-testing/tiny-CohereForCausalLM"
REVISION = "90bc56bd12c642f6bf5e710ebd7c626dd7c0201c"
PROMPT = "The capital of France is"
ATOL = 1e-4
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cohere_acceptance_reference():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokens = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"]
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    residuals = {}

    def capture_final_residual(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        residuals["final"] = inputs[0].detach().clone()

    handle = hf.model.norm.register_forward_pre_hook(capture_final_residual)
    try:
        with torch.inference_mode():
            output = hf(tokens, output_hidden_states=True, output_attentions=True, use_cache=False)
    finally:
        handle.remove()
    assert output.hidden_states is not None
    assert output.attentions is not None
    return tokens, {
        "logits": output.logits,
        "blocks.0.hook_in": output.hidden_states[0],
        "blocks.0.attn.hook_pattern": output.attentions[0],
        f"blocks.{hf.config.num_hidden_layers - 1}.hook_out": residuals["final"],
    }


@pytest.mark.parametrize("processed", [False, True], ids=["raw", "processed"])
def test_cohere_weight_conversion(cohere_acceptance_reference, processed: bool) -> None:
    tokens, expected = cohere_acceptance_reference
    bridge = TransformerBridge.boot_transformers(
        MODEL, revision=REVISION, device="cpu", dtype=torch.float32
    ).eval()
    if processed:
        # Centering changes absolute logits; isolate preparation/scale folding.
        bridge.process_weights(
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
            refactor_factored_attn_matrices=False,
        )
    with torch.inference_mode():
        logits, cache = bridge.run_with_cache(tokens)
    actual = {"logits": logits, **{name: cache[name] for name in expected if name != "logits"}}
    print(f"checkpoint={MODEL}@{REVISION} box=local device=cpu processed={processed}")
    print(f"prompt={PROMPT!r} tokens={tokens.tolist()}")
    for name, reference in expected.items():
        candidate = actual[name]
        assert candidate.dtype == reference.dtype == torch.float32, name
        assert candidate.shape == reference.shape, name
        assert torch.isfinite(candidate).all() and torch.isfinite(reference).all(), name
        delta = (candidate - reference).abs().max().item()
        print(f"{name}: max_absolute_delta={delta:.9g} tolerance={ATOL}")
        assert delta <= ATOL, f"{name}: max absolute delta {delta:.9g} exceeds {ATOL}"
