# Standard `-mlx` Fork Repo Layout

Convention developed across LTX-2, Hunyuan3D-2.1, CogVideoX-Fun, Matrix-Game and others. Following this layout makes every port look the same — contributors can orient quickly, tests find the right paths, and `mlx-forge` recipes plug in predictably.

## Naming

- GitHub fork / repo: `<OriginalModelName>-mlx`. Example: `Hunyuan3D-2.1-mlx`, `LTX-2-mlx`, `CogVideoX-Fun-mlx`.
- Python package: `<original_model_name>_mlx` (snake_case). Example: `hunyuan3d_mlx`, `ltx_core_mlx`.
- HF weights repo: `<your-hf-user>/<original-model-name>-mlx`.

## Single-package layout (smaller models)

For models with a single main component + VAE + text encoder. Example: Qwen-Image, Mistral-Small.

```
<model>-mlx/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── <model>_mlx/
│   ├── __init__.py
│   ├── pipeline_mlx.py              # high-level from_pretrained entry
│   ├── model/
│   │   ├── __init__.py
│   │   ├── transformer.py           # main module
│   │   ├── attention.py             # attention blocks
│   │   ├── vae.py
│   │   └── text_encoder.py
│   ├── config.py                    # dataclasses mirroring source config.json
│   ├── scheduler.py                 # if applicable
│   └── utils/
│       ├── __init__.py
│       ├── weights.py               # load_split_safetensors, HF download
│       └── memory.py                # aggressive_cleanup, peak_memory helpers
├── tests/
│   ├── parity/                      # PT vs MLX (torch = optional dep)
│   ├── smoke/                       # shapes / config / e2e no-numeric
│   └── fixtures/                    # golden npy / png
└── examples/
    ├── minimal.py
    └── README.md
```

## Monorepo layout (complex pipelines)

For pipelines with multiple independent packages (core model, pipelines, optional trainer). Example: LTX-2-mlx.

```
<model>-mlx/
├── README.md
├── pyproject.toml                   # workspace / root
├── packages/
│   ├── <model>-core-mlx/
│   │   ├── pyproject.toml
│   │   └── src/<model>_core_mlx/
│   │       ├── model/
│   │       │   ├── transformer/
│   │       │   ├── video_vae/
│   │       │   ├── audio_vae/
│   │       │   └── upsampler/
│   │       ├── conditioning/
│   │       ├── loader/
│   │       │   ├── sd_ops.py        # safetensors ops
│   │       │   └── fuse_loras.py
│   │       ├── text_encoders/
│   │       │   └── gemma/
│   │       └── utils/
│   │           ├── weights.py
│   │           └── memory.py
│   ├── <model>-pipelines-mlx/
│   │   └── src/<model>_pipelines_mlx/
│   │       ├── ti2vid_t2v.py
│   │       ├── ti2vid_i2v.py
│   │       ├── scheduler.py
│   │       └── cli.py               # `<model>-mlx` command
│   └── <model>-trainer/             # optional
│       └── src/<model>_trainer_mlx/
│           ├── trainer.py
│           └── training_strategies/
└── tests/
    ├── <model>-core-mlx/
    ├── <model>-pipelines-mlx/
    └── <model>-trainer/
```

## HF auto-download pattern

Standard single entrypoint:

```python
# <model>_mlx/pipeline_mlx.py
import os
from huggingface_hub import snapshot_download

class Pipeline:
    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs):
        local_dir = os.environ.get(
            f"{cls.ENV_NAME}_MLX_WEIGHTS_DIR",  # e.g. HUNYUAN3D_MLX_WEIGHTS_DIR
            snapshot_download(repo_id, allow_patterns=["*.safetensors", "*.json", "*.txt"]),
        )
        return cls._load_from_dir(local_dir, **kwargs)
```

- `<MODEL>_MLX_WEIGHTS_DIR` env var overrides the HF download — for local dev / offline use. Standardize this across ports.
- Use `snapshot_download` with `allow_patterns` to skip PyTorch-format weights if they exist in the repo.
- Lazy-load safetensors via `mx.load` — don't read into host memory first.

**Existing inconsistency:** LTX-2-mlx uses a conftest-based lookup instead of an env var. New ports should standardize on the env-var pattern above.

## pyproject.toml template

```toml
[project]
name = "<model>-mlx"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mlx>=0.30",
    "mlx-arsenal>=0.1",
    "huggingface-hub>=0.20",
    "safetensors>=0.4",
    "numpy",
    "Pillow",       # if image I/O
]

[project.optional-dependencies]
parity = ["torch>=2.3", "diffusers>=0.30"]
dev = [
    "pytest",
    "pytest-xdist",
    "<model>-mlx[parity]",
    "ruff",
]
forge = ["mlx-forge"]  # only if users need to re-convert from PT

[project.scripts]
"<model>-mlx" = "<model>_mlx.cli:main"   # if a CLI exists
```

## README.md sections (in order)

1. **Header** — one-line description, badges (Python, MLX, HF repo link).
2. **Features** — what works, what doesn't (be honest — "Stage 1 bit-exact, Stage 2 within PSNR 32 dB").
3. **Requirements** — Apple Silicon M-series, Python ≥ 3.11, macOS version, expected RAM.
4. **Installation** — `pip install <model>-mlx`, optional `[parity]` / `[forge]` extras.
5. **Quick Start** — shortest possible working example (CLI + Python code).
6. **Model Card / Architecture** — brief description, link to original paper / repo.
7. **Converting Weights Yourself** — link to mlx-forge + recipe name (if applicable).
8. **Performance** — table of peak memory / wallclock on M2, M3, M4 for representative inputs.
9. **Known Limitations** — things that don't match reference (e.g. `texture_size=2048` due to Metal command-buffer budget).
10. **Citation** — original paper BibTeX.
11. **License** — match upstream.

## .gitignore essentials

```
__pycache__/
*.egg-info/
.pytest_cache/
.ruff_cache/
.venv/
build/
dist/
# generated
*.safetensors
*.mlx
!tests/fixtures/*.safetensors  # keep small golden fixtures
# large media
*.mp4
*.png
!tests/fixtures/*.png
# user dirs
weights/
outputs/
```

## CI (GitHub Actions)

Minimum pipeline:

- **Lint** (ruff).
- **Type check** (mypy or pyright, optional).
- **Smoke tests** (shape + config, no numerics) — run on every push.
- **Parity tests** (needs torch) — run on PR to main only, gated behind `[parity]` extra install.
- **E2E** (downloads weights, runs small golden input) — nightly, not per-push.

Self-hosted macOS runner required — GitHub's Linux runners can't exercise MLX.

## Release convention

- Tag semver: `v0.1.0`, `v0.2.0`, etc.
- Pin MLX minor version in `pyproject.toml` — MLX numerics shift between minor releases; don't let users surprise themselves on upgrade.
- HF weights repo versioned via branches: `main` (latest), `v0.1.x` (compat branch).
- Release notes should always include: MLX version tested, peak memory measured, any breaking API changes.
