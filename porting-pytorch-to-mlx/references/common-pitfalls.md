# Common PyTorch → MLX Port Pitfalls

Six recurring traps that have broken past ports. Each comes with a concrete failure mode — read these *before* writing MLX code, not after debugging wrong output.

## 1. Constructor defaults silently diverging from model config

**What goes wrong:** A PyTorch module's `__init__` has a default that differs from the checkpoint's `config.json`. If the config doesn't store that parameter explicitly, the MLX port inherits the *default* instead of the trained value.

**Past failure (Hunyuan3D VAE, FourierEmbedder):** Default was `include_pi=True`, config specified `include_pi=False`. Not stored in config.json, so the default propagated silently into MLX. Frequencies got multiplied by π → garbage SDF values → mesh rendered as grid noise. Correlation with PyTorch output: 0.09.

**Rule:** Cross-check **every** constructor default against the source model's training config. Pay special attention to boolean flags that affect numerical computation: `include_pi`, `use_bias`, `qk_norm`, `flip_sin_to_cos`, `downscale_freq_shift`, `pre_norm`. If a parameter isn't in `config.json`, set the MLX default to match the training config value, not the framework default.

## 2. `attention_head_dim` is secretly `num_heads`

**What goes wrong:** In `diffusers.UNet2DConditionModel`, config field `attention_head_dim` is misnamed — it is actually `num_attention_heads`. The real per-head dimension is `channels // attention_head_dim`.

**Past failure (Hunyuan3D-2.1 paint UNet, SD 2.1):** Config `attention_head_dim=[5, 10, 20, 20]` with `block_out_channels=[320, 640, 1280, 1280]`. Correct interpretation:
- Block 0: **5 heads of dim 64** (not 64 heads of dim 5)
- Block 1: 10 heads of dim 64
- Block 2–3: 20 heads of dim 64

Weights (Linear 320→320) have identical shape either way, so the bug loads silently. But the reshape `(B, L, heads, dim)` puts different dims in different axes, changing the softmax pattern completely. Error per UNet pass: 0.21 × 15 denoising steps → cyan / neutral textures. Fixing to `num_heads = attention_head_dim[i]` brought parity to 1e-5.

**Rule:** For diffusers UNets: `num_heads = config.attention_head_dim[i]` directly — **don't divide**. Let `head_dim = channels // num_heads`. Verify by comparing `model.attn1.heads` to the raw config value. If building both `diffusers.UNet2DConditionModel.from_config(cfg)` and the MLX port side-by-side, the PyTorch `tb.attn1.heads` is authoritative.

## 3. Interleaved QKV vs independent reshape

**What goes wrong:** Some attention implementations concatenate Q, K, V *before* reshaping into multi-head format. The result is that heads are interleaved across the Q/K/V dimension. Standard per-tensor reshape produces totally wrong attention.

**Past failure (Hunyuan3D DiT attention):** Source did `qkv = cat([q, k, v], -1); qkv = qkv.view(B, N, heads, 3*hd); q, k, v = qkv.chunk(3, -1)`. Porting naively with separate reshapes (`q.view(B,N,heads,hd)`) produces max-diff 2.5 vs 6.7e-4 with correct interleaving. Model is trained with this interleaving so there is no choice.

**Rule:** Before writing attention, read the EXACT PyTorch reshape / view / split sequence. Any `cat(...) → view(... , 3*hd) → split` is an interleaved pattern and must be replicated as-is. Don't assume the "natural" per-tensor reshape.

See `attention-patterns.md` for the translated MLX code for both patterns.

## 4. Weight layout differences

**What goes wrong:** Conv weights have different memory layout across frameworks.

- PyTorch Conv2d: `(O, I, H, W)` — channels-first, out-first.
- MLX Conv2d: `(O, H, W, I)` — channels-last, out-first.
- PyTorch ConvTranspose2d: `(I, O, H, W)` — in-first.
- MLX ConvTranspose2d: `(O, H, W, I)` — same layout as Conv2d but with swapped channel semantics during op.
- Linear: `(O, I)` identical in both.
- Embedding: `(vocab, dim)` identical in both.

If the conversion recipe forgets to transpose, the weights load without errors but the convolution computes nonsense.

**Rule:** Use `mlx_forge.transpose.transpose_conv(key, weight, kind)` in the recipe — it handles all the conv variants generically. For hand-rolled conversion: when you see a key containing `.conv.weight` or ending in a spatial pattern, transpose.

## 5. Normalization semantic drift

**What goes wrong:** Different frameworks have different defaults and subtle behavior differences for normalization layers.

- **Default epsilon:** PyTorch `LayerNorm` is 1e-5, MLX `mlx.nn.LayerNorm` is 1e-5 — match. But diffusers `GroupNorm` often uses 1e-6, and some transformers set 1e-12. Always read the source.
- **RMSNorm weight semantics:** some implementations fuse gain into sqrt (`x / sqrt(var + eps) * (1 + gain)`), some don't (`* gain`). Check the reference.
- **AdaLN variants:** additive-only `x + shift` is NOT the same as classic `x * (1 + scale) + shift`. Hunyuan3D uses additive-only; LTX uses classic with 9 packed params; Matrix-Game has a `condition_type="token_replace"` branch.
- **GroupNorm num_groups:** if the reference uses `num_groups=32` unconditionally, don't compute it from channels — match exactly.

**Rule:** For every norm layer: read the exact PyTorch forward pass. Don't assume MLX defaults match.

## 6. Non-obvious flags and activation choices

**What goes wrong:** A small config flag flips a computation subtly and silently.

Common culprits:
- `qk_norm` — if true, Q and K are normalized before the dot product. Skipping this changes attention scale significantly.
- `use_bias` on Linear / Conv / QKV projections — differs across block types within the same model.
- `cross_attention_dim` — when cross-attn is present but dim differs from self-attn, projection shapes differ.
- Activation: GEGLU (`gate * gelu(up)`), SwiGLU (`gate * silu(up)`), GELU (no gate), ReLU². Each has a different parameter count and math.
- `sandwich_norm`, `pre_norm` vs `post_norm` — position of norm relative to attention / FFN changes residual math.
- `flip_sin_to_cos`, `downscale_freq_shift` in timestep embedding — affects which half of the sinusoidal encoding goes where.

**Rule:** Before translating a block, list every flag the PyTorch class checks and write a mini-table mapping config value → code branch. Then translate each branch.

## Other subtler pitfalls

- **RNG is not seed-compatible.** `mx.random.normal(key=…)` and `torch.randn(generator=…)` use different algorithms. For parity tests, generate once in numpy and inject into both sides.
- **In-place semantics.** `x[idx] = y` in MLX is a rebind (copy-on-write-style), not true mutation. Don't port PyTorch code that relies on in-place performance.
- **`tensor.contiguous()`** has no exact MLX analog — arrays are logically contiguous. Ignore contiguous calls unless downstream expects a specific stride (rare).
- **`torch.einsum`** exists as `mx.einsum` but subscript semantics are identical. Prefer keeping einsum expressions as-is rather than rewriting to matmul — fewer bugs.
- **`F.scaled_dot_product_attention`** with `is_causal=True` → use `mx.fast.scaled_dot_product_attention(q, k, v, mask="causal")`. For GQA (fewer KV heads), pass unequal-head Q and K — MLX handles it natively.
- **`torch.cumsum` on bf16** differs numerically from fp32 cumulative sums. If parity is just-above-threshold, try casting to fp32 for the cumsum step.
- **`F.silu` vs `F.hardswish`** — easy typo when reading quickly. Triple-check activation names.

## Reading strategy

When reading PyTorch source before porting, open three files side-by-side:
1. The module file (`model.py`).
2. The config (`config.json` in the checkpoint).
3. The base class from the parent framework (diffusers / transformers).

The module defaults often mislead; the config is the oracle; the parent class hides additional defaults one level deeper.
