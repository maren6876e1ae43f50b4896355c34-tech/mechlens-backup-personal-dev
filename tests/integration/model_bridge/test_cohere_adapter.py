"""Integration tests for Cohere architecture adapter (CohereForCausalLM).

Model: trl-internal-testing/tiny-CohereForCausalLM
  - 2 layers, ~8M params, CPU-safe, no gating required
  - tie_word_embeddings=True by default
  - logit_scale=0.125 (canonical Command-R is 0.0625; tiny diverges so
    regression tests catch silent-fallback bugs in the passthrough)

QK-normalized variants and processed logit parity are covered offline in
``test_cohere_numerical_regression.py`` using real HF models from tiny configs.
"""

from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import NormalizationBridge
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)

MODEL = "trl-internal-testing/tiny-CohereForCausalLM"


@pytest.fixture(scope="module")
def cohere_bridge():
    """Load tiny Cohere bridge once per module (no weight processing)."""
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def cohere_bridge_processed():
    """Bridge with standard transforms disabled; preparation already folded the scale."""
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)
    bridge.process_weights(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    return bridge


@pytest.fixture(scope="module")
def cohere_hf() -> Any:
    """Load the raw HF model for side-by-side comparisons."""
    return AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager"
    ).eval()


# ---------------------------------------------------------------------------
# 1. Bridge creation — exercises all 4 registration points + component_mapping
# ---------------------------------------------------------------------------


class TestCohereBridgeCreation:
    """Verify the bridge loads cleanly and exposes the expected structure."""

    def test_boot_transformers_succeeds(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge is not None

    def test_block_count(self, cohere_bridge: TransformerBridge) -> None:
        # tiny model has 2 layers
        assert len(cohere_bridge.blocks) == 2

    def test_has_core_components(self, cohere_bridge: TransformerBridge) -> None:
        assert hasattr(cohere_bridge, "embed")
        assert hasattr(cohere_bridge, "unembed")
        assert hasattr(cohere_bridge, "ln_final")
        assert hasattr(cohere_bridge, "rotary_emb")

    def test_no_ln2_in_blocks(self, cohere_bridge: TransformerBridge) -> None:
        # Parallel block: no post_attention_layernorm
        for block in cohere_bridge.blocks:
            assert not hasattr(block, "ln2"), "Parallel block must not have ln2"

    def test_ln1_subtracts_mean_like_hf_layernorm(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """uses_rms_norm=False must make ln1 mean-subtract (true LayerNorm), not RMS.

        Drives blocks[0].ln1 on a deliberately non-zero-mean input and compares
        against the raw HF CohereLayerNorm (which subtracts the mean). If
        uses_rms_norm regressed to True, the bridge would skip mean subtraction
        and diverge from HF by O(1), tripping this assertion.
        """
        ln1 = cohere_bridge.blocks[0].ln1
        hf_ln1 = cohere_hf.model.layers[0].input_layernorm
        d_model = cohere_bridge.cfg.d_model
        torch.manual_seed(0)
        # +5.0 shift guarantees a large mean so LN and RMS outputs differ markedly.
        x = torch.randn(1, 4, d_model) + 5.0
        with torch.no_grad():
            bridge_out = ln1(x)
            hf_out = hf_ln1(x)
        ln_diff = (bridge_out - hf_out).abs().max().item()
        assert ln_diff < 1e-4, f"ln1 does not match HF CohereLayerNorm: max_diff={ln_diff:.6f}"
        # And it must NOT match an RMS-only (no mean subtraction) normalization.
        variance = x.float().pow(2).mean(-1, keepdim=True)
        rms_only = (
            x.float()
            * torch.rsqrt(variance + ln1.original_component.variance_epsilon)
            * ln1.original_component.weight.float()
        ).to(bridge_out.dtype)
        rms_diff = (bridge_out - rms_only).abs().max().item()
        assert rms_diff > 1e-2, (
            "ln1 output matches RMS-only normalization; uses_rms_norm must be False "
            f"so the mean is subtracted (rms_diff={rms_diff:.6f})"
        )

    def test_cfg_logit_scale_matches_hf(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Regression: logit_scale must propagate from HF (not silently fall back to 0.0625)."""
        bridge_scale = getattr(cohere_bridge.cfg, "logit_scale")
        assert bridge_scale == cohere_hf.config.logit_scale
        # Anchor 0.125 so a passthrough regression that defaults to 0.0625 also trips here.
        assert bridge_scale == pytest.approx(0.125)

    def test_cfg_rope_parameters_matches_hf(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Regression: rope_parameters must propagate from HF (same passthrough trap as logit_scale)."""
        assert getattr(cohere_bridge.cfg, "rope_parameters") == cohere_hf.config.rope_parameters


# ---------------------------------------------------------------------------
# 2. Forward equivalence — HF logits ≈ bridge logits
# ---------------------------------------------------------------------------


class TestCohereForwardEquivalence:
    """Verify bridge and HF produce identical logits for the same input."""

    def test_forward_returns_logits(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            output = cohere_bridge(tokens)
        assert output.shape[0] == 1
        assert output.shape[1] == 4
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_matches_hf(self, cohere_bridge: TransformerBridge, cohere_hf: Any) -> None:
        """Bridge delegates to HF native forward — logits should be identical."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_out = cohere_bridge(tokens)
            hf_out = cohere_hf(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff:.6f}"


# ---------------------------------------------------------------------------
# 3. Logit scale preserved end-to-end
# ---------------------------------------------------------------------------


class TestCohereLogitScaleEndToEnd:
    """Weight processing must preserve the scale folded during preparation."""

    def test_unembed_weight_is_scaled_once(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        torch.testing.assert_close(
            cohere_bridge_processed.unembed.original_component.weight,
            cohere_hf.lm_head.weight * cohere_hf.config.logit_scale,
            rtol=0,
            atol=0,
        )

    def test_processed_logits_match_hf(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.no_grad():
            expected = cohere_hf(tokens).logits
            actual = cohere_bridge_processed(tokens)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)


# ---------------------------------------------------------------------------
# 4. Tied embedding preserved — embed.W_E must NOT be scaled
# ---------------------------------------------------------------------------


class TestCohereTiedEmbedding:
    """Processing must preserve the input embeddings of tied models."""

    def test_embed_weight_equals_hf_embed_tokens(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        # Processing must not change the embedding shared by HF's input/output layers.
        hf_embed = cohere_hf.model.embed_tokens.weight.detach()  # [d_vocab, d_model]
        tl_embed = cohere_bridge_processed.embed.W_E  # [d_vocab, d_model]
        max_diff = (tl_embed - hf_embed).abs().max().item()
        assert (
            max_diff < 1e-6
        ), f"embed.W_E was corrupted (during processing): max_diff={max_diff:.6f}"


# ---------------------------------------------------------------------------
# 5. HF component references — bridges retain the wrapped modules
# ---------------------------------------------------------------------------


class TestCohereHFDelegation:
    """Spot-check that bridge submodules delegate to actual HF modules.

    This confirms setup_component_testing wired rotary_emb correctly and that
    NormalizationBridge and PositionEmbeddingsAttentionBridge hold live HF objects.
    """

    def test_ln1_original_component_is_hf_norm(self, cohere_bridge: TransformerBridge) -> None:
        # bridge.blocks[0].ln1.original_component should be the live HF CohereLayerNorm
        ln1 = cohere_bridge.blocks[0].ln1
        assert isinstance(ln1, NormalizationBridge)
        assert ln1.original_component is not None
        # CohereLayerNorm has variance_epsilon (not eps)
        assert hasattr(
            ln1.original_component, "variance_epsilon"
        ), "original_component is not CohereLayerNorm (missing variance_epsilon)"

    def test_attn_original_component_is_hf_attention(
        self, cohere_bridge: TransformerBridge
    ) -> None:
        attn = cohere_bridge.blocks[0].attn
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.original_component is not None
        # CohereAttention has q_proj
        assert hasattr(
            attn.original_component, "q_proj"
        ), "original_component is not CohereAttention (missing q_proj)"

    def test_ln_final_original_component_is_hf_norm(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge.ln_final.original_component is not None
        assert hasattr(cohere_bridge.ln_final.original_component, "variance_epsilon")


# ---------------------------------------------------------------------------
# 6. Parallel-attn hooks fire correctly
# ---------------------------------------------------------------------------


class TestCohereParallelHooks:
    """Verify hook placement for the parallel attention+MLP block."""

    def test_no_hook_resid_mid(self, cohere_bridge: TransformerBridge) -> None:
        # Parallel block has no intermediate residual stream
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        assert not any("hook_resid_mid" in k for k in cache.keys())

    def test_attn_and_mlp_hooks_fire(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.attn.hook_in" in cache
            assert f"blocks.{i}.attn.hook_out" in cache
            assert f"blocks.{i}.mlp.hook_in" in cache
            assert f"blocks.{i}.mlp.hook_out" in cache

    def test_residual_hooks_fire(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.hook_resid_pre" in cache
            assert f"blocks.{i}.hook_resid_post" in cache
