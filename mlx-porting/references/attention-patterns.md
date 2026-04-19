# Attention Translation Patterns

Attention is the densest source of port bugs because shape-compatible code can be semantically wrong. This file lists the patterns seen across past ports with direct translations.

## Pattern A — Standard per-tensor reshape (SAFE path)

**PyTorch:**

```python
q = q_proj(x).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
k = k_proj(x).view(B, N, H, D).transpose(1, 2)
v = v_proj(x).view(B, N, H, D).transpose(1, 2)
out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
out = out.transpose(1, 2).reshape(B, N, H * D)
out = o_proj(out)
```

**MLX:**

```python
q = q_proj(x).reshape(B, N, H, D).transpose(0, 2, 1, 3)  # (B, H, N, D)
k = k_proj(x).reshape(B, N, H, D).transpose(0, 2, 1, 3)
v = v_proj(x).reshape(B, N, H, D).transpose(0, 2, 1, 3)
out = mx.fast.scaled_dot_product_attention(
    q, k, v, scale=1.0 / math.sqrt(D),
    mask="causal" if causal else None,
)
out = out.transpose(0, 2, 1, 3).reshape(B, N, H * D)
out = o_proj(out)
```

Notes:
- `mx.fast.scaled_dot_product_attention` **requires** an explicit `scale` argument. Don't forget.
- `mask="causal"` is the clean idiom; otherwise pass an additive mask tensor (0 for keep, -inf for mask).
- MLX transpose takes an axis tuple, not pair-swap semantics like PyTorch.

## Pattern B — Interleaved QKV (the trap)

**PyTorch (Hunyuan3D DiT, some SD variants):**

```python
qkv = torch.cat([q, k, v], dim=-1)            # (B, N, 3*C)
qkv = qkv.view(B, N, H, 3 * D)                # interleave heads
q, k, v = qkv.chunk(3, dim=-1)                # (B, N, H, D) each
q = q.transpose(1, 2)                         # (B, H, N, D)
k = k.transpose(1, 2)
v = v.transpose(1, 2)
```

If you naively do `q.view(B, N, H, D)` **per tensor** in the MLX port instead of replicating the interleaving, attention produces garbage — max diff 2.5 instead of 6.7e-4.

**MLX (matching the interleaving):**

```python
qkv = mx.concatenate([q, k, v], axis=-1)      # (B, N, 3*C)
qkv = qkv.reshape(B, N, H, 3 * D)             # same interleave
q, k, v = mx.split(qkv, 3, axis=-1)           # (B, N, H, D) each
q = q.transpose(0, 2, 1, 3)                   # (B, H, N, D)
k = k.transpose(0, 2, 1, 3)
v = v.transpose(0, 2, 1, 3)
```

## Pattern C — Fused QKV projection

**PyTorch:**

```python
qkv = self.qkv_proj(x)                        # Linear(C -> 3*C)
qkv = qkv.view(B, N, 3, H, D).permute(2, 0, 3, 1, 4)   # (3, B, H, N, D)
q, k, v = qkv.unbind(0)
```

**MLX:**

```python
qkv = self.qkv_proj(x)
qkv = qkv.reshape(B, N, 3, H, D).transpose(2, 0, 3, 1, 4)
q, k, v = qkv[0], qkv[1], qkv[2]
```

When loading weights, a fused QKV projection can be concatenated from three separate PyTorch `q_proj`, `k_proj`, `v_proj` tensors in the conversion recipe (`cat` along dim 0 for `(out, in)` layout).

## Pattern D — GQA (Grouped Query Attention)

**PyTorch (Llama-like):**

```python
q = q_proj(x).view(B, N, H_Q, D).transpose(1, 2)   # H_Q heads
k = k_proj(x).view(B, N, H_KV, D).transpose(1, 2)  # fewer heads
v = v_proj(x).view(B, N, H_KV, D).transpose(1, 2)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**MLX:** `mx.fast.scaled_dot_product_attention` handles mismatched Q / KV head counts natively — don't manually repeat K/V heads.

```python
q = q_proj(x).reshape(B, N, H_Q, D).transpose(0, 2, 1, 3)
k = k_proj(x).reshape(B, N, H_KV, D).transpose(0, 2, 1, 3)
v = v_proj(x).reshape(B, N, H_KV, D).transpose(0, 2, 1, 3)
out = mx.fast.scaled_dot_product_attention(
    q, k, v, scale=1.0 / math.sqrt(D), mask="causal",
)
```

MLX 0.30+ has a fused vector GQA kernel for decode — it kicks in automatically when `N == 1`.

## Pattern E — Rotary embeddings (RoPE)

**PyTorch (hand-rolled):**

```python
def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
```

**MLX (prefer fast path):**

```python
q = mx.fast.rope(q, D, traditional=False, base=10000.0, scale=1.0, offset=0)
k = mx.fast.rope(k, D, traditional=False, base=10000.0, scale=1.0, offset=0)
```

Use `traditional=True` for the original LLaMA RoPE layout (chunk-swap), `False` for the HuggingFace / GPT-NeoX layout (interleaved). **Check the reference before choosing** — getting this wrong produces token-position scrambling.

For RoPE with NTK scaling or YaRN, compute `cos / sin` manually and multiply — `mx.fast.rope` doesn't cover all scaling schemes yet.

## Pattern F — `qk_norm`

**PyTorch:**

```python
if self.qk_norm:
    q = self.q_norm(q)
    k = self.k_norm(k)
```

Applied in head-split shape `(B, H, N, D)`. Usually RMSNorm over D. Easy to forget this if it's gated behind a config flag — check for `q_norm` / `k_norm` attributes and config flags like `qk_norm`, `qk_layernorm`, `use_qk_norm`.

## Pattern G — Cross attention

Same as self-attention but K and V come from a different input. Extra pitfall: `cross_attention_dim` may differ from `hidden_dim`, making `k_proj` and `v_proj` shape `(cross_dim, head_dim * num_heads)` instead of `(hidden_dim, head_dim * num_heads)`. Read the config carefully.

## Mask conventions

- PyTorch additive mask: 0 for keep, -inf for masked.
- MLX `mx.fast.scaled_dot_product_attention` accepts:
  - `mask="causal"` (built-in lower-triangular).
  - A float tensor with 0 / -inf semantics matching PyTorch.
  - A bool tensor: True = keep, False = mask (opposite of PyTorch key_padding_mask — watch out).
- Sliding-window / block-causal masks: use `mlx_arsenal.attention.sliding_window_mask` if available, else build manually and pass as float.

## Quick sanity checklist when translating attention

- [ ] Did I match the reshape pattern exactly (A vs B)?
- [ ] Is `scale` explicitly passed to `mx.fast.scaled_dot_product_attention`?
- [ ] Does my transpose use an axis tuple (MLX) and not `.transpose(1, 2)` (PyTorch)?
- [ ] Did I account for `qk_norm`?
- [ ] Did I pick the right RoPE layout (`traditional=True/False`)?
- [ ] Is mask polarity correct (0=keep, not 1=keep)?
- [ ] For GQA: Q head count is H_Q, K/V is H_KV — no manual repeat needed?
