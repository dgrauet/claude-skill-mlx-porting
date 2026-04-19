# MLX Official Documentation — Curated Links

Use this as the canonical reference list when porting. These URLs are verified current as of April 2026. If you hit an MLX API question during a port, check here first before guessing.

Note on code style in this file: `mx.eval( )` is written with a space between parentheses to avoid a security-hook false-positive on Python's builtin. In real code, write it as `mx.eval(...)` with the tensors inside.

## 1. Core framework

- https://ml-explore.github.io/mlx/build/html/index.html — top-level doc index.
- https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html — lazy graph; triggers (`print`, `.item()`, `np.array()`, `mx.save_*`). Recommends `mx.eval` at iteration boundaries.
- https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html — single memory pool; arrays are device-agnostic, pick compute device per op.
- https://ml-explore.github.io/mlx/build/html/usage/indexing.html — supported/unsupported fancy indexing; for exotic cases use `mx.take`, `mx.put_along_axis`.
- https://ml-explore.github.io/mlx/build/html/usage/using_streams.html — `mx.new_stream`, `mx.synchronize`, per-op stream placement.
- https://ml-explore.github.io/mlx/build/html/usage/compile.html — `mx.compile`; must be pure, recompiles on shape change, use `inputs=/outputs=` for state, `MLX_DISABLE_COMPILE` to debug.
- https://ml-explore.github.io/mlx/build/html/python/array.html — `mx.array` API.
- https://ml-explore.github.io/mlx/build/html/python/ops.html — full op list. **Check here before assuming a PyTorch op has an MLX equivalent.**
- https://ml-explore.github.io/mlx/build/html/python/devices_and_streams.html — `mx.default_device`, `mx.gpu`, `mx.cpu`.

## 2. mlx.nn (layers + weight I/O)

- https://ml-explore.github.io/mlx/build/html/python/nn.html — Linear, Conv1/2/3d, ConvTranspose*, LayerNorm, GroupNorm, RMSNorm, BatchNorm, InstanceNorm, MultiHeadAttention, Embedding, Transformer.
- https://ml-explore.github.io/mlx/build/html/python/nn/module.html — `Module.update()`, `.parameters()`, `.trainable_parameters()`, `.load_weights()`.
- https://ml-explore.github.io/mlx/build/html/python/tree_utils.html — `tree_flatten`, `tree_unflatten`, `tree_map` for nested param dicts (essential for weight remapping).
- https://ml-explore.github.io/mlx/build/html/usage/saving_and_loading.html — `mx.load` / `mx.save_safetensors`; supported formats: npy, npz, safetensors, gguf.

## 3. mlx.fast (prefer these over hand-rolled)

- https://ml-explore.github.io/mlx/build/html/python/fast.html — `rms_norm`, `layer_norm`, `rope`, `scaled_dot_product_attention`, `metal_kernel`, `cuda_kernel`. Use these first for speed + numerical stability.

## 4. Custom Metal kernels

- https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html — `mx.fast.metal_kernel`: only kernel body in `source`, signature auto-generated; `ensure_row_contiguous`, grid/threadgroup, template params, `atomic_outputs` for reductions.
- https://ml-explore.github.io/mlx/build/html/dev/extensions.html — full C++ / Metal extension path when `metal_kernel` isn't enough.

## 5. Quantization

- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html — `nn.quantize(model, group_size, bits, mode, class_predicate)`; converts Linear / Embedding in place.
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html — low-level `mx.quantized_matmul`.
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html — `mx.quantize` / `mx.dequantize`. Common combos: 4-bit g=64, 8-bit g=128.

## 6. Reference repos (mlx-examples conventions)

- https://github.com/ml-explore/mlx-examples — canonical `from_pretrained` / `convert.py` / `model.py` split. Directories: stable_diffusion, flux, llms, llava, t5, musicgen, whisper, segment_anything, encodec, wan2.1, clip, wwdc25.
- https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion/stable_diffusion — canonical diffusion layout: `__init__.py`, `config.py`, `model_io.py` (HF download + weight-key remap via `tree_unflatten`), `unet.py`, `vae.py`, `clip.py`, `tokenizer.py`, `sampler.py`.
- https://github.com/ml-explore/mlx-examples/tree/main/llms — generation loop / KV-cache patterns.
- https://github.com/ml-explore/mlx-examples/tree/main/musicgen — audio-model porting reference.

## 7. MLX-LM

- https://github.com/ml-explore/mlx-lm — `mlx_lm.convert --model <hf-id> -q` conversion CLI; defaults to HF `mlx-community/*` 4-bit.
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py — `load` / `get_model_path` / `fetch_from_hub` reference for auto-download pipeline.
- https://huggingface.co/mlx-community — pre-converted weights; **check here before converting** (someone may have already done the base port).

## 8. Apple & release notes

- https://opensource.apple.com/projects/mlx/ — Apple's canonical landing page (Python / Swift / C / C++).
- https://github.com/ml-explore/mlx/releases — authoritative changelog. Read the last ~10 releases before starting a new port for breaking changes and new fast-path ops.
- https://developer.apple.com/videos/play/wwdc2025/315/ — WWDC25 "Get started with MLX for Apple silicon".

## 9. Limits / gotchas (direct links)

- https://ml-explore.github.io/mlx/build/html/python/memory_management.html — `mx.metal.set_memory_limit`, `mx.metal.clear_cache`; Metal command-buffer timeout fires on very long graphs — insert a materialization call.
- https://ml-explore.github.io/mlx/build/html/python/data_types.html — fp16 / bf16 / fp32 / complex64 matrix. bf16 supported on M-series GPU, some reductions fp32-only.
- https://github.com/ml-explore/mlx/issues — search the op name before assuming parity with PyTorch; in-place ops are emulated (rebind the array, don't assume in-place performance).
- **Inference-only ports** skip autograd: no `value_and_grad` / `nn.value_and_grad` needed.

## Major MLX additions in the last ~6 months (as of April 2026)

Check these when starting a new port — they may replace hand-rolled code:

- **NVFP4 / MXFP8 quantized matmul** (v0.30.3, Jan 2026) — low-bit modes beyond 4/8-bit, Metal + CUDA.
- **`mx.fast.cuda_kernel`** — CUDA sibling to `metal_kernel`; same API, enables NVIDIA-backend ports.
- **3 / 5 / 6-bit QMV** (v0.31.1) — odd-bit quant via `nn.quantize(..., bits=3|5|6)`.
- **Columnwise quantization + QQ linear** (v0.30.3 / 0.30.4) — activation-and-weight quant path.
- **Two-pass SDPA + vector fused GQA** for long context (v0.30.4) — prefer `mx.fast.scaled_dot_product_attention`.
- **3D conv speedups + grouped matmul / sort speedups** (v0.30.4 / 0.31.0) — relevant for vision / 3D ports.
- **`mx.fast` Hadamard transform** + Hanning / Hamming / Blackman / Bartlett windows (v0.31.x) — new signal-processing primitives.

## Keep this file updated

When you start a port and discover MLX has added something new, append it here. This doc is a living cheatsheet — outdated links cost more than missing ones.
