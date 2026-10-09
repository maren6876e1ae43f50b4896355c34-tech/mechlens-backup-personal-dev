---
title: Model Tables
---
# Model Tables

```{warning}
`HookedTransformer` is deprecated as of TransformerLens 3.0 and will be removed in the next major version. New code should use [`TransformerBridge`](migrating_to_v3.md) instead. Existing `HookedTransformer` code continues to work through the 3.x branch via a compatibility layer. See the [migration guide](migrating_to_v3.md) for conversion recipes.
```

TransformerLens 3.0 provides two model loading paths, each with its own set of supported models.

- **HookedTransformer** -- The original TransformerLens models with full hook-point access and mechanistic interpretability support.
- **TransformerBridge Models** -- Automatic compatibility layer for thousands of HuggingFace models across supported architectures.

```{toctree}
/generated/model_properties_table
/generated/transformer_bridge_models
```

## Command-R

Command-R is supported through `TransformerBridge`. Models with the HuggingFace
architecture `CohereForCausalLM` use `CohereArchitectureAdapter`, selected
automatically by the standard `boot_transformers` loader. The legacy
`HookedTransformer.from_pretrained` loader does not support this architecture.

The following example loads the public tiny Cohere checkpoint used by the
Command-R acceptance test:

```python
import torch
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers(
    "trl-internal-testing/tiny-CohereForCausalLM",
    revision="90bc56bd12c642f6bf5e710ebd7c626dd7c0201c",
    device="cpu",
    dtype=torch.float32,
).eval()
tokens = bridge.tokenizer(
    "The capital of France is", add_special_tokens=False, return_tensors="pt"
)["input_ids"]
with torch.inference_mode():
    logits, cache = bridge.run_with_cache(tokens)
resid_pre = cache["blocks.0.hook_in"]
attention_pattern = cache["blocks.0.attn.hook_pattern"]
```

To load a full Command-R checkpoint, use its model ID, for example
`CohereForAI/c4ai-command-r-v01`, in the same loader. Omit the tiny checkpoint's
`revision` or replace it with a revision of the selected model. Choose a device
and dtype appropriate for that checkpoint's memory requirements, and configure
[`HF_TOKEN`](getting_started.md#hf-token) if access is gated.

The adapter handles adjacent-pair RoPE, parallel attention/MLP, optional QK
normalization, and logit scaling. The tiny-checkpoint acceptance test compares
float32 logits and intermediate activations with HuggingFace at a maximum
absolute tolerance of `1e-4`. This validates the tiny checkpoint, not every
production Command-R checkpoint. Its processed case disables optional weight
transforms; it does not certify default compatibility-mode processing. See
[compatibility mode](compatibility_mode.md) before enabling those transforms.

## DeepSeek

DeepSeek support is also available through `TransformerBridge.boot_transformers`.
The loader selects the adapter from the model's HuggingFace architecture:

| Model family | HuggingFace architecture | Adapter class |
| --- | --- | --- |
| DeepSeek-V2, V2-Lite, Coder-V2 | `DeepseekV2ForCausalLM` | `DeepSeekV2ArchitectureAdapter` |
| DeepSeek-V3, full DeepSeek-R1 | `DeepseekV3ForCausalLM` | `DeepSeekV3ArchitectureAdapter` |

**Full DeepSeek-R1 uses `DeepSeekV3ArchitectureAdapter`**; it does not require a
separate R1 adapter. R1 distilled checkpoints use their underlying architecture:
the R1-Distill-Llama models use `LlamaArchitectureAdapter`, R1-Distill-Qwen models
use `Qwen2ArchitectureAdapter`, and R1-0528-Qwen3-8B uses
`Qwen3ArchitectureAdapter`.

Use a DeepSeek checkpoint ID with the same `boot_transformers` API shown above;
adapter classes do not need to be instantiated manually. Architecture support
does not imply that every checkpoint has passed end-to-end verification. Consult
the [model verification table](../generated/transformer_bridge_models.md) for
checkpoint-level results. Full R1 verification is recorded as skipped for
resource limits in the current registry. DeepSeek MLA blocks also have
[hook compatibility differences](compatibility_mode.md) from standard split-QKV
attention.
