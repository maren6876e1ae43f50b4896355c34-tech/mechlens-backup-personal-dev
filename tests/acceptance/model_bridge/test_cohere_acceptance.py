"""Float32 acceptance parity for the Command-R architecture adapter."""

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
def cohere_inputs() -> torch.Tensor:
    """Tokenize a fixed prompt with the checkpoint's pinned tokenizer revision."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    return tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"]


@pytest.fixture(scope="module")
def cohere_hf() -> torch.nn.Module:
    """Load the Hugging Face reference model in the same numerical configuration."""
    return AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()


@pytest.fixture(scope="module")
def cohere_bridge() -> TransformerBridge:
    """Load the Command-R bridge in float32 with eager attention."""
    return TransformerBridge.boot_transformers(
        MODEL,
        revision=REVISION,
        device="cpu",
        dtype=torch.float32,
    ).eval()


def test_cohere_logits_match_huggingface(
    cohere_inputs: torch.Tensor,
    cohere_hf: torch.nn.Module,
    cohere_bridge: TransformerBridge,
) -> None:
    """Require Command-R bridge logits to match Hugging Face within 1e-4."""
    with torch.inference_mode():
        expected = cohere_hf(cohere_inputs).logits
        actual = cohere_bridge(cohere_inputs)

    assert expected.dtype == torch.float32
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=0)
