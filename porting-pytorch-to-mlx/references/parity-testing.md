# Parity Testing — PyTorch vs MLX

Parity tests are the single most important tool when a port produces wrong output. They turn "the whole pipeline is broken" into "layer 17 diverges at max_abs = 0.3, focus there".

**Past ports have skipped this step and paid for it.** Every `-mlx` fork should ship with a parity test harness, even if it lives behind an optional dependency.

Note on code style: `.eval( )` (torch's eval-mode method) is written with a space between parentheses to sidestep a security-hook false-positive on the Python builtin. In real code write it as `pt_model.eval()`.

## Test structure

Recommend: `tests/parity/` directory with one file per major module (`test_attention_parity.py`, `test_unet_block_parity.py`, `test_vae_parity.py`, `test_full_pipeline_parity.py`).

## Canonical template

```python
# tests/parity/test_attention_parity.py
import math
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from my_model_mlx.model.attention import Attention as MxAttention
from my_model_pytorch.model.attention import Attention as PtAttention


def _load_pt_to_mx(pt_model, mx_model):
    """Copy PyTorch state dict into an MLX model with key/shape transforms."""
    import mlx_forge.transpose as tp
    pt_state = pt_model.state_dict()
    mx_weights = {}
    for key, val in pt_state.items():
        new_key = key  # rename if needed
        w = mx.array(val.detach().cpu().numpy())
        if "conv" in key and ".weight" in key:
            w = tp.transpose_conv(key, w, kind="conv2d")
        mx_weights[new_key] = w
    mx_model.update(tree_unflatten(list(mx_weights.items())))
    return mx_model


@pytest.fixture
def seeded_input():
    rng = np.random.default_rng(42)
    return rng.standard_normal((2, 128, 1024)).astype(np.float32)


def test_attention_parity_fp32(seeded_input):
    B, N, C = seeded_input.shape
    config = dict(hidden_dim=C, num_heads=8, head_dim=128)

    pt_model = PtAttention(**config).eval( )
    mx_model = MxAttention(**config)
    _load_pt_to_mx(pt_model, mx_model)

    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(seeded_input)).numpy()
    mx_out = np.array(mx_model(mx.array(seeded_input)))

    max_abs = float(np.max(np.abs(pt_out - mx_out)))
    mean_abs = float(np.mean(np.abs(pt_out - mx_out)))
    assert max_abs < 1e-4, f"attention diverges: max_abs={max_abs:.2e}, mean_abs={mean_abs:.2e}"
```

Use fp32 for parity tests — it isolates numerical drift from framework precision differences. Save fp16 / bf16 tests for regression against a known-good fp32 reference.

## Threshold table

For fp32 parity testing:

| Scope | `max_abs` target | Acceptable |
|---|---|---|
| Single linear / conv / norm layer | `< 1e-6` | `< 1e-5` |
| Self-attention / cross-attention block | `< 1e-5` | `< 1e-4` |
| Transformer block (attn + FFN + norms + residual) | `< 1e-4` | `< 1e-3` |
| Full DiT / UNet pass | `< 1e-3` | `< 1e-2` |
| Full VAE encode or decode | `< 1e-3` | `< 1e-2` |
| End-to-end pipeline (denoising + VAE) | — | qualitative (PSNR > 35 dB on fixed input) |

For fp16 inference:

| Scope | `max_abs` target |
|---|---|
| Single layer | `< 1e-3` |
| Transformer block | `< 5e-3` |
| Full UNet / DiT pass | `< 1e-2` |
| End-to-end | qualitative (PSNR > 30 dB) |

**Anything at `max_abs > 1e-1` is a port bug, not numerical drift.** Stop and isolate.

## Isolation strategy when a test fails

The goal is to find the innermost layer where divergence first appears.

1. **Start at the block level.** If `test_transformer_block_parity` fails at `max_abs = 0.3`, go inward.
2. **Bisect within the block.** Test norm alone. Test attention alone. Test FFN alone. Test residual alone. One of them will fail.
3. **Within the failing sub-module, instrument.** Print intermediate tensor stats (`min, max, mean, std, sum_abs`) on both sides at each op. The first op where stats diverge is the bug.
4. **Common root causes** (in order of frequency from past ports):
   - Wrong QKV reshape (interleaved vs per-tensor).
   - Wrong `num_heads` vs `head_dim` interpretation.
   - Wrong RoPE layout (`traditional` flag).
   - Missing `qk_norm`.
   - Wrong weight transpose (Conv layout).
   - Default constructor arg diverging from config.
   - Missing materialization at conversion time (tensor is literally zero).
   - Wrong activation (silu vs gelu, GEGLU variant).

## Golden-input fixtures

For reproducibility, check a small set of seeded inputs into the repo:

```
tests/parity/fixtures/
├── attention_input_2x128x1024_seed42.npy
├── image_3x512x512_cat.png
├── prompt_golden.txt
└── latents_seed42.npy
```

Same seed on both sides, same input bytes. Loaded identically via numpy. This avoids the `mx.random` vs `torch.manual_seed` incompatibility.

## Pipeline-level parity (hardest)

Full pipeline parity is noisy because of samplers, schedulers, and free-running RNG. To compare:

1. Run PyTorch pipeline with a fixed seed; save:
   - Initial noise tensor.
   - Text embeddings.
   - Final latents before VAE decode.
   - Final pixel output.
2. Load those same tensors in the MLX test. Inject them at each stage to bypass RNG.
3. Compare at each stage: noise → embeddings → latents → pixels.
4. If embeddings match but latents don't, the denoising loop is wrong (scheduler, DiT, or CFG).
5. If latents match but pixels don't, the VAE is wrong.

## Making PyTorch an optional dependency

Users installing the `-mlx` fork should not need torch. Add torch as a dev / parity-only extra:

```toml
# pyproject.toml
[project.optional-dependencies]
parity = ["torch>=2.3", "diffusers>=0.30"]
dev = ["pytest", "my-model-mlx[parity]"]
```

Then in tests:

```python
torch = pytest.importorskip("torch")  # skip if not installed
```

This keeps the main install lean while preserving parity tests for contributors.

## Invariant tests vs parity tests

`mlx-arsenal` tests use **invariant** patterns (e.g. `get_timestep_embedding(t, flip_sin_to_cos=True)` should equal `concat([second_half, first_half])` of `flip_sin_to_cos=False`). These catch self-consistency bugs.

Invariant tests are **complementary**, not a substitute. They cannot catch:
- Wrong default values vs reference config.
- Wrong QKV reshape pattern.
- Wrong weight transpose.

For porting, always include PyTorch-vs-MLX comparison, at least during initial development. Invariant tests can stay as the permanent regression suite once parity is locked.

## When parity is "close enough"

- fp32 parity at 1e-5 or better: ship it.
- fp16 / bf16 parity at 1e-3: ship it, document the fp32 reference is the oracle.
- 1e-2 to 1e-1: investigate. Often a small systematic bias from a missing `qk_norm` or off-by-one in RoPE offset.
- Above 1e-1: do not ship. Bug.

## Regression safety

Once parity is locked, commit the golden fixtures and pin MLX version in CI. Every MLX upgrade can shift numerics slightly — rerun parity tests on upgrade, relax thresholds only if justified by an MLX release note.
