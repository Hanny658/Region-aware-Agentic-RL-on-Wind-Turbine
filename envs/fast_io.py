"""Tiny helpers to edit OpenFAST / ROSCO input files in place by parameter name."""
from __future__ import annotations

import re
from pathlib import Path


def set_param(path: str | Path, name: str, value: str) -> None:
    """Replace the value column of the line whose parameter name is `name`.
    Works for '   12.1   RotSpeed  - ...' and 'True  Flag - ...' and ROSCO '0  ! Name  - ...' styles.
    For array-valued lines (e.g. BlPitch(1)) pass the exact name incl. parentheses."""
    p = Path(path)
    lines = p.read_text().splitlines()
    pat = re.compile(r"^(\s*)(\S+)(\s+)(!?\s*)" + re.escape(name) + r"(\s|$)")
    hit = False
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if m:
            lines[i] = f"{m.group(1)}{value}{m.group(3)}{m.group(4)}{name}{ln[m.end(5) - 1:]}" \
                if m.group(5) else f"{m.group(1)}{value}{m.group(3)}{m.group(4)}{name}"
            hit = True
            break
    if not hit:
        raise KeyError(f"{name} not found in {p}")
    p.write_text("\n".join(lines) + "\n")


def get_param(path: str | Path, name: str) -> str:
    pat = re.compile(r"^\s*(\S+)\s+!?\s*" + re.escape(name) + r"(\s|$)")
    for ln in Path(path).read_text().splitlines():
        m = pat.match(ln)
        if m:
            return m.group(1)
    raise KeyError(f"{name} not found in {path}")
