"""Offline numerical regression coverage for Command-R's architecture variants."""

import copy

import pytest
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    CohereConfig,
    CohereForCausalLM,
)

from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)


@pytest.mark.parametrize("use_qk_norm", [False, True])
@pytest.mark.parametrize("mode", ["raw", "processed", "compatibility", "no_processing", "default"])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("scale", [0.0625, 0.125, 1.0])
def test_cohere_logits_match_hf(use_qk_norm: bool, mode: str, tied: bool, scale: float) -> None:
    torch.manual_seed(42)
    config = CohereConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        use_qk_norm=use_qk_norm,
        logit_scale=scale,
        tie_word_embeddings=tied,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    hf = CohereForCausalLM(config).eval()
    if use_qk_norm:
        with torch.no_grad():
            for layer in hf.model.layers:
                layer.self_attn.q_norm.weight.uniform_(0.5, 1.5)
                layer.self_attn.k_norm.weight.uniform_(0.5, 1.5)
    bridge = build_bridge_from_module(
        copy.deepcopy(hf), "CohereForCausalLM", hf_config=config
    ).eval()
    # Preparing a wrapped model again must not rescale the already-live weights.
    bridge.adapter.prepare_model(bridge.original_model)
    torch.testing.assert_close(bridge.embed.W_E, hf.model.embed_tokens.weight, rtol=0, atol=0)
    assert bridge.W_U.data_ptr() == bridge.unembed.original_component.weight.data_ptr()
    options = dict(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    if mode == "processed":
        bridge.process_weights(**options)
    elif mode == "compatibility":
        bridge.enable_compatibility_mode(disable_warnings=True, **options)
    elif mode == "no_processing":
        bridge.enable_compatibility_mode(disable_warnings=True, no_processing=True)
    elif mode == "default":
        bridge.enable_compatibility_mode(disable_warnings=True)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11]])
    with torch.no_grad():
        expected = hf(tokens).logits
        actual, cache = bridge.run_with_cache(tokens)
        reconstructed = cache["ln_final.hook_out"] @ bridge.W_U + bridge.b_U
    torch.testing.assert_close(reconstructed, actual, atol=1e-6, rtol=1e-5)
    if mode == "default":
        # Unembed centering changes logits by a vocabulary-independent constant.
        actual = actual.log_softmax(-1)
        expected = expected.log_softmax(-1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        expected_loss = torch.nn.functional.cross_entropy(
            expected[:, :-1].reshape(-1, config.vocab_size), tokens[:, 1:].reshape(-1)
        )
        actual_loss = bridge(tokens, return_type="loss")
    torch.testing.assert_close(actual_loss, expected_loss, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("model_type", ["llama", "qwen3", "olmo2"])
def test_standard_rope_and_qk_norm_paths_match_hf(model_type: str) -> None:
    """Protect shared attention's no-norm, per-head RMS, and flattened RMS paths."""
    torch.manual_seed(42)
    config = AutoConfig.for_model(
        model_type,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    hf = AutoModelForCausalLM.from_config(config).eval()
    bridge = build_bridge_from_module(copy.deepcopy(hf), type(hf).__name__, hf_config=config).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        expected = hf(tokens).logits
        actual = bridge(tokens)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
