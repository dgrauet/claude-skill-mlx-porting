#!/usr/bin/env python3
"""Validate the `mlx-porting` skill structure.

Ran by the CI workflow on every push and PR. Fails with a non-zero exit code
and a readable message when anything is off. Also runnable locally:

    python3 .github/scripts/validate.py
"""

from __future__ import annotations

import json
import py_compile
import re
import sys
from pathlib import Path

SKILL_ROOT = Path("mlx-porting")
REQUIRED_FRONTMATTER = ("name", "description")
REQUIRED_EVAL_KEYS = ("id", "prompt", "expected_output")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"OK   {msg}")


def validate_frontmatter(skill_md: Path) -> None:
    text = skill_md.read_text()
    if not text.startswith("---\n"):
        fail(f"{skill_md}: file does not start with `---` frontmatter")
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        fail(f"{skill_md}: malformed frontmatter (expected closing `---`)")
    fm = parts[1]
    for key in REQUIRED_FRONTMATTER:
        if not re.search(rf"^{re.escape(key)}\s*:", fm, re.M):
            fail(f"{skill_md}: frontmatter missing required key `{key}`")
    ok(f"{skill_md}: frontmatter has {', '.join(REQUIRED_FRONTMATTER)}")


def validate_evals(evals_path: Path) -> None:
    try:
        data = json.loads(evals_path.read_text())
    except json.JSONDecodeError as exc:
        fail(f"{evals_path}: invalid JSON — {exc}")
    if not isinstance(data, dict):
        fail(f"{evals_path}: top-level must be a JSON object")
    for top_key in ("skill_name", "evals"):
        if top_key not in data:
            fail(f"{evals_path}: missing top-level key `{top_key}`")
    if not isinstance(data["evals"], list) or not data["evals"]:
        fail(f"{evals_path}: `evals` must be a non-empty array")
    seen_ids = set()
    for i, eval_entry in enumerate(data["evals"]):
        if not isinstance(eval_entry, dict):
            fail(f"{evals_path}: eval #{i} is not an object")
        for key in REQUIRED_EVAL_KEYS:
            if key not in eval_entry:
                fail(f"{evals_path}: eval #{i} missing key `{key}`")
        eid = eval_entry["id"]
        if eid in seen_ids:
            fail(f"{evals_path}: duplicate eval id `{eid}`")
        seen_ids.add(eid)
    ok(f"{evals_path}: {len(data['evals'])} evals, all required keys present")


def validate_references(skill_md: Path) -> None:
    text = skill_md.read_text()
    # Matches references like `references/foo.md` or `scripts/foo.py` inside backticks.
    pattern = re.compile(r"`((?:references|scripts)/[^`\s]+)`")
    missing: list[str] = []
    for match in pattern.finditer(text):
        ref_path = SKILL_ROOT / match.group(1)
        if not ref_path.exists():
            missing.append(str(ref_path))
    if missing:
        fail(f"{skill_md} references missing files:\n  " + "\n  ".join(sorted(set(missing))))
    ok(f"{skill_md}: all backticked references/… and scripts/… paths resolve")


def validate_python_syntax(scripts_dir: Path) -> None:
    py_files = sorted(scripts_dir.glob("*.py"))
    if not py_files:
        ok(f"{scripts_dir}: no python scripts to compile")
        return
    for p in py_files:
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as exc:
            fail(f"{p}: syntax error — {exc.msg}")
        ok(f"{p}: syntax valid")


def main() -> None:
    if not SKILL_ROOT.is_dir():
        fail(f"{SKILL_ROOT} not found (run from repo root)")

    validate_frontmatter(SKILL_ROOT / "SKILL.md")
    validate_evals(SKILL_ROOT / "evals" / "evals.json")
    validate_references(SKILL_ROOT / "SKILL.md")
    validate_python_syntax(SKILL_ROOT / "scripts")

    print("\nAll validations passed.")


if __name__ == "__main__":
    main()
