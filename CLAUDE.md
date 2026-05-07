# CLAUDE.md

Project context for the `mlx-porting` Claude Code skill repository.

## What this repo is

The source-of-truth for the **`mlx-porting`** Claude Code skill — a workflow guide for porting PyTorch / CUDA models to Apple MLX. Released as a `.skill` tarball on GitHub releases.

## Layout

```
mlx-porting/           # The skill itself (SKILL.md + references/ + scripts/ + evals/)
.github/workflows/     # CI (validate + intendant audit + commitlint) and Release (tarball on tag)
.github/scripts/       # validate.py — schema check for SKILL.md frontmatter
docs/                  # Project docs (not shipped in the skill tarball)
.intendant.toml        # Governance config (stack=auto, advisory mode)
release-please-config.json + .release-please-manifest.json  # Release-please scaffolding
```

## Two copies, one source

The skill is **also deployed** at `~/.claude/skills/mlx-porting/` so Claude Code can load it. That deployed copy is a **separate filesystem location**, not a symlink. When editing skill content:

- Edit in this repo (`mlx-porting/`), commit, push.
- The deployed copy at `~/.claude/skills/mlx-porting/` is updated separately (manual sync or whatever deployment mechanism the user runs).
- If the user edits the deployed copy directly during a session, sync back into the repo before committing.

## Commands

```bash
python3 .github/scripts/validate.py          # Validate SKILL.md frontmatter and structure
tar czf mlx-porting.skill mlx-porting        # Build the release tarball locally
uvx intendant audit . --severity=required    # Run governance audit (matches CI)
pre-commit run --all-files                   # Run hooks (trailing whitespace, gitleaks, etc.)
```

## Release workflow

- Tag-driven: pushing `vX.Y.Z` triggers `.github/workflows/release.yml`, which validates, builds `mlx-porting.skill`, and attaches it to the GitHub release.
- Versions are tracked in `.release-please-manifest.json`. Use release-please conventions for changelog generation.

## Conventions

- **Conventional commits required** (enforced by commitlint in CI). Common prefixes: `feat(pitfalls):`, `feat(skill):`, `fix:`, `chore:`, `ci:`, `docs:`.
- **Pitfalls are numbered and append-only** in `mlx-porting/references/common-pitfalls.md`. New pitfalls get the next integer (currently #9) and a corresponding `feat(pitfalls): add #N — <topic>` commit.
- **Don't refactor pitfall numbering** — referenced from `SKILL.md` checklists; renumbering breaks cross-references.
