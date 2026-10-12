#!/usr/bin/env python
"""Write paper/abstract.md (plain text for OpenReview) from main.tex.

Expands the number macros of tables/numbers.tex and strips LaTeX markup.

    python scripts/paper/make_abstract.py   # run after make_numbers.py
"""
import re
from pathlib import Path

PAPER = Path("paper")


def plain(tex: str, macros: dict[str, str]) -> str:
    """Convert a LaTeX snippet to readable plain text."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)             # LaTeX comments (keeps escaped \%)
    tex = re.sub(r"\\([A-Za-z]+)\{\}", r"\\\1", tex)
    tex = tex.replace("\\\\", " ")
    for name in sorted(macros, key=len, reverse=True):
        tex = re.sub(rf"\\{name}(?![A-Za-z])", lambda m, v=macros[name]: v, tex)
    for a, b in [(r"\emph{", "{"), (r"\textbf{", "{"), (r"\mathrm{", "{"), ("\\,", " "),
                 ("\\!", ""), ("--", "–"), (r"\%", "%"), (r"\approx", "≈"), (r"\Phi", "Φ"),
                 (r"\kappa", "κ"), (r"\sqrt", "√"), (r"\log", "log"), ("~", " ")]:
        tex = tex.replace(a, b)
    tex = re.sub(r"\$([^$]*)\$", r"\1", tex)
    tex = tex.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", tex).strip()


if __name__ == "__main__":
    src = (PAPER / "main.tex").read_text()
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}",
                             (PAPER / "tables/numbers.tex").read_text()))
    title = re.search(r"\\aistatstitle\{(.*?)\}\n\\aistatsauthor", src, re.S).group(1)
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1)
    out = ("# AISTATS 2027 — abstract (registration due Tue 29 Sep 2026, 23:59 AoE)\n\n"
           "Generated from paper/main.tex by scripts/paper/make_abstract.py "
           "(numbers from tables/numbers.tex); do not edit by hand.\n\n"
           f"**Title:** {plain(title, macros)}\n\n**Abstract:**\n\n{plain(abstract, macros)}\n")
    (PAPER / "abstract.md").write_text(out)
    print(out)
