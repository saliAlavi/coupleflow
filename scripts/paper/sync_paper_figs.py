#!/usr/bin/env python
"""Copy every figure the paper includes into paper/figures/.

Keeps paper/ self-contained (uploadable to Overleaf as is). Figure
scripts keep writing to figures/; run this after them, before LaTeX.

    python scripts/paper/sync_paper_figs.py
"""
import re
import shutil
from pathlib import Path

PAPER = Path("paper")
SRC = Path("figures")
DST = PAPER / "figures"

if __name__ == "__main__":
    DST.mkdir(exist_ok=True)
    tex = "".join(p.read_text() for p in PAPER.glob("*.tex"))
    names = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", tex)))
    for name in names:
        shutil.copy2(SRC / name, DST / name)
        print("copied", DST / name)
