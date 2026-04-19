"""Reusable helpers for PyTorch vs MLX parity tests.

Copy this file into your `-mlx` fork's tests directory (typically
`tests/parity/_helpers.py`). Import the functions you need in each
parity test file.

Both torch and MLX are expected to be installed in the dev environment.
Gate the parity-test directory behind a `[parity]` optional dependency in
pyproject.toml so end users don't need torch.

Style note: `_materialize(...)` wraps `mx.eval` on the tensors passed in,
forcing lazy graphs to compute. Use it before any numpy conversion or
`save_safetensors` call — otherwise you read zeros from lazy tensors.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _materialize(*tensors) -> None:
    """Force MLX lazy tensors to compute. Wraps mx.eval."""
    import mlx.core as mx
    mx.eval(*tensors)


def make_seeded_input(shape: tuple[int, ...], seed: int = 42, dtype=np.float32) -> np.ndarray:
    """Produce a reproducible numpy array usable as input to both PT and MLX.

    MLX's `mx.random` and PyTorch's `torch.manual_seed` are NOT cross-compatible,
    so generate once in numpy and inject into both sides.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(dtype)


def pt_to_mx(val):
    """Convert a PyTorch tensor to an MLX array via numpy (safe, always works)."""
    import mlx.core as mx
    return mx.array(val.detach().cpu().float().numpy())


def mx_to_np(val) -> np.ndarray:
    """Convert an MLX array to numpy after forcing materialization."""
    _materialize(val)
    return np.array(val)


def transpose_pt_conv(weight_np: np.ndarray, ndim: int) -> np.ndarray:
    """Transpose a PyTorch conv weight (O,I,*K) to MLX layout (O,*K,I).

    ndim = 1 for Conv1d (O,I,K -> O,K,I)
    ndim = 2 for Conv2d (O,I,H,W -> O,H,W,I)
    ndim = 3 for Conv3d (O,I,D,H,W -> O,D,H,W,I)
    """
    if ndim == 1:
        return weight_np.transpose(0, 2, 1)
    if ndim == 2:
        return weight_np.transpose(0, 2, 3, 1)
    if ndim == 3:
        return weight_np.transpose(0, 2, 3, 4, 1)
    raise ValueError(f"unsupported conv ndim: {ndim}")


def load_pt_state_into_mx(
    mx_model,
    pt_state_dict: dict,
    rename: Callable[[str], str] | None = None,
    conv_keys: set[str] | None = None,
    conv_ndim_by_key: dict[str, int] | None = None,
) -> None:
    """Copy a PyTorch state_dict into an MLX model.

    Args:
        mx_model: MLX Module instance.
        pt_state_dict: PyTorch state_dict (keys -> tensors).
        rename: optional callable to remap PT key -> MLX key.
        conv_keys: keys that correspond to conv weights needing transpose.
        conv_ndim_by_key: map of conv-weight key -> conv dimensionality (1/2/3).

    Keys not present in either side are silently skipped - validate separately
    in a debug harness.
    """
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    rename = rename or (lambda k: k)
    conv_keys = conv_keys or set()
    conv_ndim_by_key = conv_ndim_by_key or {}

    mx_flat: list[tuple[str, mx.array]] = []
    for pt_key, pt_val in pt_state_dict.items():
        new_key = rename(pt_key)
        arr = np.ascontiguousarray(pt_val.detach().cpu().float().numpy())
        if pt_key in conv_keys:
            ndim = conv_ndim_by_key.get(pt_key, 2)
            arr = transpose_pt_conv(arr, ndim)
        mx_flat.append((new_key, mx.array(arr)))

    mx_model.update(tree_unflatten(mx_flat))
    _materialize(mx_model.parameters())


def assert_parity(
    pt_out,
    mx_out,
    threshold: float,
    name: str = "output",
) -> None:
    """Assert MLX output matches PyTorch reference within `threshold` max_abs.

    Raises AssertionError with diagnostic stats on failure.
    """
    import torch

    pt_np = pt_out.detach().cpu().float().numpy() if isinstance(pt_out, torch.Tensor) else np.asarray(pt_out)
    mx_np = mx_to_np(mx_out) if not isinstance(mx_out, np.ndarray) else mx_out

    if pt_np.shape != mx_np.shape:
        raise AssertionError(
            f"[{name}] shape mismatch: pt={pt_np.shape} vs mx={mx_np.shape}"
        )

    diff = np.abs(pt_np - mx_np)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    if max_abs >= threshold:
        pt_range = (float(pt_np.min()), float(pt_np.max()))
        mx_range = (float(mx_np.min()), float(mx_np.max()))
        rel_err = float((diff / (np.abs(pt_np) + 1e-8)).mean())
        raise AssertionError(
            f"[{name}] parity FAIL: max_abs={max_abs:.3e} "
            f"(threshold={threshold:.1e}), mean_abs={mean_abs:.3e}, "
            f"rel_err={rel_err:.3e}, pt_range={pt_range}, mx_range={mx_range}"
        )


def tensor_stats(name: str, t) -> dict:
    """Compute summary stats on a tensor for bisection instrumentation.

    Works for both MLX arrays and torch tensors. Returns a dict you can log
    from both sides; when stats diverge between PT and MLX at a specific op,
    you've found the divergent layer.
    """
    import torch

    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().float().numpy()
    else:
        arr = mx_to_np(t)
    return {
        "name": name,
        "shape": tuple(arr.shape),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "abs_sum": float(np.abs(arr).sum()),
    }


# ---------------------------------------------------------------------------
# Checkerboard diagnostic helpers — for post-port sanity checks.
# See common-pitfalls.md #7.
# ---------------------------------------------------------------------------


def detect_checkerboard(image_np: np.ndarray, max_period: int = 32) -> dict:
    """Return periodicity statistics for an image (H, W, C) float array.

    If the output contains a spatial checkerboard at stride `p`, the
    autocorrelation at offset `p` will be unusually high relative to offsets
    `p-1` and `p+1`. Computes a cheap 1D proxy (average autocorrelation over
    rows and channels) and returns the strongest period under `max_period`.

    Use in smoke tests: `assert detect_checkerboard(img)["strongest_period"] is None`.
    """
    if image_np.ndim == 4:
        image_np = image_np[0]  # drop batch
    if image_np.ndim == 3 and image_np.shape[-1] in (1, 3, 4):
        gray = image_np.mean(axis=-1)
    else:
        gray = image_np
    H, W = gray.shape[-2:]
    row_mean = gray - gray.mean(axis=-1, keepdims=True)
    acf = np.zeros(min(max_period, W // 2) + 1)
    for tau in range(len(acf)):
        if tau == 0:
            acf[0] = 1.0
        else:
            a = row_mean[..., :-tau]
            b = row_mean[..., tau:]
            num = (a * b).mean()
            den = (row_mean**2).mean() + 1e-12
            acf[tau] = float(num / den)
    # Any offset with autocorrelation > 0.3 and a clear local max vs neighbours
    # is suspicious.
    suspicious = []
    for tau in range(2, len(acf) - 1):
        if acf[tau] > 0.3 and acf[tau] > acf[tau - 1] and acf[tau] > acf[tau + 1]:
            suspicious.append((tau, float(acf[tau])))
    suspicious.sort(key=lambda x: -x[1])
    return {
        "autocorrelation": acf.tolist(),
        "suspicious_periods": suspicious,
        "strongest_period": suspicious[0][0] if suspicious else None,
    }


def noise_decode_check(vae_decode_fn, latent_shape: tuple, seed: int = 0) -> "np.ndarray":
    """Decode a standard-normal latent through ``vae_decode_fn`` and return the
    image as a numpy array. Pair with ``detect_checkerboard`` to assert a
    smooth output on noise — any periodicity on noise input means the spatial
    operators in the decoder are broken, independent of model weights.
    """
    import mlx.core as mx

    mx.random.seed(seed)
    z = mx.random.normal(latent_shape) * 2.0
    img = vae_decode_fn(z)
    return mx_to_np(img)
