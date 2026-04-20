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

## 7. The checkerboard trap (recurring — watch for it actively)

**What goes wrong:** The per-layer parity tests pass at small scale with random weights, but end-to-end inference with real weights produces an image with a visible checkerboard pattern at a specific scale (8×8, 16×16, or matching the patch/upsample stride). The model is *almost* working — the DiT moves latents away from noise, the VAE decodes without crashing — but the output has periodic artifacts instead of a coherent image.

**Past failures:**
- **Hunyuan3D paint UNet:** 2×2 mosaic at each decoder stage — wrong axis order in `F.interpolate` equivalent.
- **ERNIE-Image v0:** 8-pixel checkerboard — off-by-one in text-encoder `hidden_states[-2]` that caused the DiT to receive text conditioning with 10× wrong magnitude *and* shifted scale.
- **Qwen-Image:** VAE `Upsample2D` used `mx.repeat` on the wrong axis (channels instead of spatial) → tile-like artifacts.

**Root causes to check, in order of frequency:**

1. **`mx.repeat` vs `mx.tile` confusion.** `mx.repeat(x, 2, axis=1)` on `(B, H, W, C)` duplicates each H-row consecutively (`[a, a, b, b, ...]`) — the correct nearest-neighbor upsample. `mx.tile(x, (1, 2, 1, 1))` produces `[a, b, ..., a, b, ...]` — block tiling, which always checkerboards. Never use `mx.tile` for upsampling.

2. **Pixel-shuffle / patch pack axis order.** Both PT `reshape(B, C, H/r, r, W/r, r) → permute(0,1,3,5,2,4)` and MLX channels-last `reshape(B, H, W, C, r, r) → transpose(0,1,4,2,5,3)` produce the same logical output. Verify by a round-trip identity test on a small known tensor (not random!) — feed `mx.arange(total)` reshaped to the input shape, unpatchify, re-patchify, and assert equal.

3. **Position IDs axis swap.** `rope_axes_dim=[text, y, x]` with grid constructed via `meshgrid(grid_y, grid_x, indexing="ij")` must stack `[yy, xx]` (not `[xx, yy]`). A swap produces subtle checkerboard because y-axis RoPE rotates channels the DiT expects to carry x-info and vice versa.

4. **Text-conditioning magnitude mismatch.** Check `hidden_states[-2]` semantics in HF transformers: it's the output of layer `N-1` — which is the INPUT to the last layer, not the OUTPUT of the last layer. So use `for layer in layers[:-1]` (apply N-1 layers), NOT `for layer in layers` then return pre-norm. Off-by-one here produces correct-looking conditioning at wrong magnitude — DiT denoises in the wrong latent manifold, checkerboard at the VAE decoder.

5. **Dtype leak from scheduler into DiT.** The `FlowMatchEulerDiscreteScheduler.step()` in mlx-arsenal keeps `sigmas` as `fp32`; multiplying a `bf16` latent by an `fp32` scalar promotes the latent to `fp32`. The next DiT forward then runs with `fp32` input vs `bf16` weights. Cast back: `latents = scheduler.step(...).astype(dtype)`.

**Diagnostic procedure — use this EVERY port before shipping:**

```python
# Test 1: decode pure Gaussian noise through the VAE only.
# If checkerboard appears here, the VAE is broken (upsample, conv layout).
z = mx.random.normal((1, lat_H, lat_W, lat_C)) * 2.0
img = vae.decode(z)
# → should be smooth coloured noise, no periodic pattern.

# Test 2: decode noise through the *full post-DiT chain* (BN-inverse +
# pixel-shuffle + VAE). If smooth here but checkerboard end-to-end, the DiT
# output is the source.
dit_shape = (1, dit_C, lat_H // patch, lat_W // patch)
dit_out = mx.random.normal(dit_shape)
nhwc = dit_out.transpose(0, 2, 3, 1)
nhwc = vae.bn.apply_inverse(nhwc)  # if model has latent BN
unpacked = pixel_shuffle(nhwc, upscale_factor=patch)
img = vae.decode(unpacked)
# → still smooth noise. If checkerboard, the BN or pixel_shuffle axis is off.

# Test 3: real-weight DiT single-block vs reference.
# If parity < 1e-3, and tests 1/2 pass, the bug is in attention conventions
# (positions, mask format, qk_norm, RoPE axis order) which only manifest at
# the model-trained magnitudes.
```

**Preventive: add these three tests to every port's `tests/smoke/` BEFORE running full generation.** They pinpoint the layer at fault in under 30 seconds each and catch 95% of checkerboard bugs.

**Rule:** If the end-to-end output has checkerboard, do not tweak the denoising loop parameters (steps, guidance, shift) — they can never fix a wrong spatial operator. Walk up the stack with the three tests above instead.

## 8. Tekken / Pixtral tokenizer skips the BOS

**What goes wrong:** When wiring an `mlx-lm` Mistral-family text encoder (Ministral3, Mistral Small 3, Pixtral) to a diffusion DiT, you tokenize with `add_special_tokens=True` expecting `<s>` at position 0 — and it silently does NOT get added. Every content token is fine, but the FIRST token enters the attention stack at an out-of-distribution magnitude, compounds layer by layer, and the DiT receives conditioning that's correct everywhere except position 0. End-to-end generation produces a structured-but-incoherent image (barely-visible features buried in noise).

**Past failure (ERNIE-Image port, 2026-04-20):** `mistral-community/pixtral-12b` tokenizer bundled as the runtime tokenizer. Layer-by-layer diff between `mlx_lm.models.ministral3.Model` and `transformers.Ministral3Model` on identical input_ids:

| layer | token-0 max_abs | token-1..N max_abs |
|---|---|---|
| 0-1 | 0.003 | 0.005 |
| 2   | 4.18  | 0.004 |
| 3-25 | 4-7 | 0.02-0.07 |

Token-0 is 100× off from every other position and grows across the stack. Fix is prepending `tokenizer.bos_token_id` before the first content token. Once done, the content tokens 1..N keep their same (tiny) divergence and the DiT receives correct conditioning in the positions it actually uses.

**Rule:** In your pipeline's `_tokenize` / `encode_prompt` helper:

```python
ids = tokenizer(prompt, add_special_tokens=True, ...)["input_ids"]
bos = tokenizer.bos_token_id or 1
if not ids or ids[0] != bos:
    ids = [bos] + ids
```

Do this for every Tekken-family tokenizer, not just Pixtral. `add_special_tokens=True` is an identity op for these backends even though the argument name suggests otherwise.

## Reading strategy

When reading PyTorch source before porting, open three files side-by-side:
1. The module file (`model.py`).
2. The config (`config.json` in the checkpoint).
3. The base class from the parent framework (diffusers / transformers).

The module defaults often mislead; the config is the oracle; the parent class hides additional defaults one level deeper.
