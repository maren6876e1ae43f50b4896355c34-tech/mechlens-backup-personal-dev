---
title: Command-R and DeepSeek
---

# Command-R and DeepSeek

## Command-R

TransformerLens supports the Command-R family through the Hugging Face
`CohereForCausalLM` architecture. Load a Command-R checkpoint with
`TransformerBridge`:

```python
import torch

from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers(
    "CohereLabs/c4ai-command-r-v01",
    device="cuda",
    dtype=torch.bfloat16,
)
```

For a small CPU smoke test, use the same architecture with the pinned test
checkpoint:

```python
bridge = TransformerBridge.boot_transformers(
    "trl-internal-testing/tiny-CohereForCausalLM",
    device="cpu",
    dtype=torch.float32,
)
```

The adapter preserves Command-R's grouped-query attention, parallel
attention/MLP block layout, true LayerNorm, logit scaling, tied embeddings,
and interleaved rotary embeddings.

## DeepSeek

DeepSeek support is provided through two bridge adapters:

- `DeepSeekV2ArchitectureAdapter` handles `DeepseekV2ForCausalLM`: DeepSeek-V2,
  DeepSeek-V2-Lite, and DeepSeek-Coder-V2.
- `DeepSeekV3ArchitectureAdapter` handles `DeepseekV3ForCausalLM`: DeepSeek-V3
  and DeepSeek-R1.

Both adapters support DeepSeek's Multi-Head Latent Attention (MLA), mixture of
experts with shared experts, initial dense layers, RMSNorm, and rotary
position embeddings. V2-Lite's direct query-projection variant is handled by
the V2 adapter.

Load a supported DeepSeek checkpoint through the same interface:

```python
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers(
    "deepseek-ai/DeepSeek-V2-Lite",
    device="cuda",
)
```

For DeepSeek-V3 and DeepSeek-R1, retain Hugging Face's default SDPA attention
implementation. The adapter does not support eager attention for their
asymmetric MLA query/key dimensions.
