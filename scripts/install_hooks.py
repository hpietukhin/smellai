#!/usr/bin/env python3
"""Install git hooks from scripts/hooks/ into .git/hooks/.

Usage:
    uv run python scripts/install_hooks.py
"""
import shutil
import stat
import sys
from pathlib import Path


def install_hooks() -> None:
    repo_root = Path(__file__).parent.parent
    hooks_src = repo_root / "scripts" / "hooks"
    hooks_dst = repo_root / ".git" / "hooks"

    if not hooks_dst.exists():
        print(f"Error: {hooks_dst} not found — are you in a git repository?", file=sys.stderr)
        sys.exit(1)

    installed = []
    for src in hooks_src.iterdir():
        if src.name.startswith(".") or not src.is_file():
            continue
        dst = hooks_dst / src.name
        shutil.copy2(src, dst)
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        installed.append(src.name)
        print(f"Installed: .git/hooks/{src.name}")

    if not installed:
        print("No hooks found in scripts/hooks/")
    else:
        print(f"\n{len(installed)} hook(s) installed successfully.")


if __name__ == "__main__":
    install_hooks()
