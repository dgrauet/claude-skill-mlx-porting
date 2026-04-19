# claude-skill-mlx-porting

[![CI](https://github.com/dgrauet/claude-skill-mlx-porting/actions/workflows/ci.yml/badge.svg)](https://github.com/dgrauet/claude-skill-mlx-porting/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/dgrauet/claude-skill-mlx-porting)](./LICENSE)
[![Release](https://img.shields.io/github/v/release/dgrauet/claude-skill-mlx-porting)](https://github.com/dgrauet/claude-skill-mlx-porting/releases)
[![Last commit](https://img.shields.io/github/last-commit/dgrauet/claude-skill-mlx-porting)](https://github.com/dgrauet/claude-skill-mlx-porting/commits/main)
[![Triggering accuracy](https://img.shields.io/badge/triggering-100%25-brightgreen)](#measured-performance)

A [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills) that captures the end-to-end workflow for **porting PyTorch / CUDA models to Apple MLX** for inference on Apple Silicon. It distills the pitfalls, conventions, and reusable helpers accumulated across ~10 production MLX ports (LTX-2, Hunyuan3D-2.1, CogVideoX-Fun, Matrix-Game, Mistral-Small, Qwen Image, and others).

When installed, Claude automatically invokes the skill whenever you start working on an MLX port — scaffolding a new `-mlx` fork, translating attention / RoPE / VAE layers, setting up PyTorch↔MLX parity tests, or diagnosing wrong numerics.

## Why this skill

MLX ports fail in subtle, silent ways. The same line of code that works in PyTorch can load without error in MLX and produce garbage output because of:

- `diffusers` `attention_head_dim` misnomer (it's actually `num_attention_heads`)
- Fused QKV with `cat → view → chunk` interleaving that breaks under a standard per-tensor reshape
- Constructor defaults diverging from the trained model's config (boolean flags like `include_pi`, `qk_norm`, `flip_sin_to_cos`)
- Lazy MLX tensors serialized as zeros before materialization
- PyTorch Conv `(O, I, *K)` vs MLX `(O, *K, I)` weight layout

Each of these has burned a real port. The skill turns this hard-won knowledge into a checklist-driven workflow Claude can follow on your behalf.

## What's inside

```
porting-pytorch-to-mlx/
├── SKILL.md                        # 7-step workflow + six reading-time traps
├── references/
│   ├── mlx-docs.md                 # Curated official MLX URLs (core, nn, fast, quant, WWDC25…)
│   ├── common-pitfalls.md          # The six traps with concrete past-failure examples
│   ├── attention-patterns.md       # Standard MHA / GQA / RoPE / interleaved QKV / SDPA fast path
│   ├── weight-conversion.md        # mlx-forge recipe pattern, materialization rule, transpose
│   ├── parity-testing.md           # PT↔MLX test templates, threshold table, bisection strategy
│   └── repo-layout.md              # `-mlx` fork standard layout, pyproject, HF auto-download
├── scripts/
│   └── parity_helpers.py           # Reusable PT↔MLX helpers (assert_parity, load_pt_state_into_mx…)
└── evals/
    └── evals.json                  # Five representative test cases used to validate the skill
```

## Installation

### Option 1 — source copy (recommended for development)

```bash
git clone https://github.com/dgrauet/claude-skill-mlx-porting.git
cp -r claude-skill-mlx-porting/porting-pytorch-to-mlx ~/.claude/skills/
```

Claude Code will pick up the skill on next session.

### Option 2 — `.skill` package

Drop `porting-pytorch-to-mlx.skill` into Claude Code:
- In Claude Code, use `/skill install <path>` (or the UI skill install flow)
- Or unpack into `~/.claude/skills/` — `.skill` is a tarball with a manifest

## Usage

Once installed, simply describe your porting task normally. The skill auto-triggers on phrasings like:

- *"Port OmniGen2 to MLX"*
- *"Scaffold a new -mlx fork for a diffusion model"*
- *"My MLX UNet port is producing cyan textures, what's wrong?"*
- *"Set up PT vs MLX parity tests for my VAE"*
- *"How do I use `mx.fast.scaled_dot_product_attention` with GQA?"*
- *"Write an mlx-forge recipe for my Llama fine-tune"*

Claude reads `SKILL.md`, consults the relevant reference file(s), and applies the 7-step workflow. For conversion-recipe authoring specifically, it delegates to a companion [`mlx-recipe`](https://github.com/dgrauet/claude-skill-mlx-recipe) skill if installed.

## Companion tools

The skill references two complementary open-source libraries. They are optional but recommended:

- **[mlx-forge](https://github.com/dgrauet/mlx-forge)** — CLI tool for PyTorch → MLX weight conversion with per-model recipes. The skill explicitly delegates recipe authoring to an `mlx-recipe` skill; `mlx-forge` is its runtime.
- **[mlx-arsenal](https://github.com/dgrauet/mlx-arsenal)** — Reusable MLX ops (flow-matching primitives, spatial ops, norms, attention masks, tiling, encoding) extracted from production ports. Check here before hand-rolling an op.

## Measured performance

Validated on 3 evaluation iterations against a baseline of Claude Opus 4.7 with the same prompt but no skill context:

| Metric | Value |
|---|---|
| Triggering accuracy (20 queries, precision + recall) | **100%** |
| Pass-rate lift vs baseline, workflow-intensive tasks | +10 to +25 percentage points |
| Strongest lift on | parity-harness layout, scaffolding, inter-skill delegation, bundled helpers |
| Saturated (baseline already handles well) | diagnostic Q/A about known MLX APIs |

The skill's durable value sits in **workflow discipline** (conventions, parity testing, delegation boundaries) — not in raw MLX knowledge, which modern frontier models already possess.

## Contributing

This skill grows organically as new MLX ports surface new pitfalls. If you hit a bug class not in `common-pitfalls.md` or use a pattern worth sharing, please open an issue or PR.

Particularly valuable contributions:

- **New entries in `common-pitfalls.md`** with a concrete past-failure example.
- **New reference files** if a whole class of port (audio, 3D mesh, speech, sparse) has distinct conventions worth documenting.
- **Additions to `scripts/parity_helpers.py`** for commonly duplicated helper code.
- **Additional trigger-eval queries** in `evals/evals.json` to improve the test surface.

## License

MIT — see [`LICENSE`](./LICENSE).
