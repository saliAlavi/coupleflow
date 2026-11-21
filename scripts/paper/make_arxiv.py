#!/usr/bin/env python
"""Build the arXiv version of the paper from paper/ without modifying paper/.

    python scripts/paper/make_arxiv.py
Outputs: arxiv/src/ (exactly the upload), arxiv/coupleflow_arxiv.zip, and arxiv/main.pdf
compiled from arxiv/src/ the way arXiv does it (pdflatex only, no BibTeX).

AISTATS allows preprints during review only in "a neutral, non-AISTATS format" that does not
identify the paper as an AISTATS submission (AISTATS 2027 Submission FAQ 26). So the arXiv copy
  - uses the plain two-column article class with the AISTATS text block (6.75 x 9.25 in,
    0.25 in column gap), so floats keep their sizes, instead of aistats2027.sty;
  - names the authors, drops the AISTATS checklist, and title-cases the section headings;
  - fixes the unfinished edits in the abstract and introduction;
  - strips every LaTeX comment (arXiv publishes the source) and ships main.bbl;
  - contains only the files pdflatex reads (from a -recorder build).
Every substitution must match exactly once, so a changed paper/main.tex fails here, loudly.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

PAPER = Path("paper")
OUT = Path("arxiv")
ZIP = OUT / "coupleflow_arxiv.zip"

TITLE = ("CoupleFlow: Null-Calibrated Auditory Attention Decoding With Time-Locked "
         "Brain-Speech Coupling and Flow-Matching Likelihood Ratios")
AUTHORS = [("Ali Alavi", "alavibajestan.1@osu.edu"),
           ("Donald S. Williamson", "williamson.413@osu.edu")]
AFFILIATION = "Department of Computer Science and Engineering, The Ohio State University"

PREAMBLE = r"""\pdfoutput=1
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[letterpaper,textwidth=6.75in,textheight=9.25in,columnsep=0.25in,centering,footskip=0.4in]{geometry}
"""
AUTHOR_BLOCK = (r"\author{"
                + r"\hspace{3em}".join(
                    rf"\begin{{tabular}}[t]{{c}}{name}\\ \normalsize\texttt{{{mail}}}\end{{tabular}}"
                    for name, mail in AUTHORS)
                + rf"\\[0.5em]" "\n" rf"\normalsize {AFFILIATION}}}" "\n")

# (old, new) pairs applied to main.tex; each old string must occur exactly once
EDITS = [
    ("\\documentclass[twoside]{article}\n\\usepackage{aistats2027}\n", PREAMBLE),
    (r"\renewcommand{\bibsection}{\subsubsection*{\bibname}}",
     r"\renewcommand{\bibsection}{\section*{\bibname}}"),
    ("\\usepackage[hidelinks]{hyperref}\n",
     "\\usepackage[hidelinks]{hyperref}\n\\hypersetup{pdftitle={%s},pdfauthor={%s}}\n"
     % (TITLE, ", ".join(n for n, _ in AUTHORS))),
    ("\n\\raggedbottom\n", "\n\\raggedbottom\n\\sloppy\n"),            # aistats2027.sty set \sloppy
    # unfinished edits in the submitted abstract and introduction
    ("likelihood-ratio decision rule,  a conditional flow-matching head adds a likelihood-ratio "
     "decision rule: it selects", "likelihood-ratio decision rule: it selects"),
    ("removes any preference for speech model finds likely a priori",
     "removes any preference for speech the model finds likely a priori"),
    ("\nA Using a few seconds of multichannel", "\nUsing a few seconds of multichannel"),
]
# regex substitutions on main.tex; each must match exactly once
REGEX_EDITS = [
    (r"\\twocolumn\[\n\\aistatstitle\{(?P<t>.*?)\}\n\\aistatsauthor\{Anonymous Authors\}\n"
     r"\\aistatsaddress\{Anonymous Institution\}\]",
     lambda m: "\\title{%s}\n%s\\date{}\n\\maketitle" % (m["t"], AUTHOR_BLOCK)),
    (r"\\section\*\{Checklist\}.*?(?=\\clearpage\s*\\appendix)", lambda m: ""),
]
SECTIONS = {  # AISTATS enforces all-caps headings; the source types them in caps
    "INTRODUCTION": "Introduction", "COUPLEFLOW": "CoupleFlow", "THEORY": "Theory",
    "EXPERIMENTS": "Experiments", "DISCUSSION AND LIMITATIONS": "Discussion and Limitations",
    "EXTENDED RELATED WORK": "Extended Related Work", "PROOFS": "Proofs",
    "DATASETS AND PREPROCESSING": "Datasets and Preprocessing",
    "NETWORK DETAILS": "Network Details", "TRAINING OBJECTIVE": "Training Objective",
    "ADDITIONAL RESULTS": "Additional Results",
    "CONFIDENCE AND SEQUENTIAL DECISIONS": "Confidence and Sequential Decisions",
}
# must not appear anywhere in the upload (case-insensitive)
FORBIDDEN = ["aistats", "anonymous author", "anonymous institution", "do not distribute"]


def fail(msg):
    sys.exit(f"make_arxiv: {msg}")


def replace_once(text, old, new):
    n = text.count(old)
    if n != 1:
        fail(f"expected 1 occurrence, found {n}: {old[:70]!r}")
    return text.replace(old, new)


def sub_once(text, pattern, repl):
    text, n = re.subn(pattern, repl, text, flags=re.S)
    if n != 1:
        fail(f"expected 1 match, found {n}: {pattern[:70]!r}")
    return text


def strip_comments(text):
    """Drop whole-line comments; cut trailing comments down to a bare % (keeps spacing)."""
    out = []
    for line in text.splitlines(keepends=True):
        m = re.search(r"(?:^|[^\\])(?:\\\\)*%", line)       # first unescaped %
        if not m:
            out.append(line)
        elif line[:m.end() - 1].strip():
            out.append(line[:m.end() - 1] + "%\n")
    return re.sub(r"\n{3,}", "\n\n", "".join(out))


def pdflatex(d, recorder=False):
    env = dict(os.environ, SOURCE_DATE_EPOCH=os.environ.get("SOURCE_DATE_EPOCH", "1791331200"))
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    if recorder:
        cmd.insert(1, "-recorder")
    for _ in range(3):
        r = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True,
                           encoding="latin-1")
        if r.returncode:
            print("\n".join(r.stdout.splitlines()[-30:]))
            fail(f"pdflatex failed in {d}")
    return (Path(d) / "main.log").read_text(encoding="latin-1")


def transform(stage):
    main = stage / "main.tex"
    text = main.read_text()
    for old, new in EDITS:
        text = replace_once(text, old, new)
    for pattern, repl in REGEX_EDITS:
        text = sub_once(text, pattern, repl)
    main.write_text(text)
    for tex in stage.rglob("*.tex"):
        text = tex.read_text()
        for caps, title in SECTIONS.items():
            text = text.replace(rf"\section{{{caps}}}", rf"\section{{{title}}}")
        if left := re.findall(r"\\section\{([A-Z][A-Z ]{3,})\}", text):
            fail(f"all-caps section heading not in SECTIONS: {left}")
        tex.write_text(strip_comments(text))


def inputs(stage):
    """Files under stage/ that pdflatex read, minus its own build products."""
    found = set()
    for line in (stage / "main.fls").read_text().splitlines():
        if not line.startswith("INPUT "):
            continue
        p = Path(line[6:])
        p = (stage / p) if not p.is_absolute() else p
        try:
            rel = p.resolve().relative_to(stage.resolve())
        except ValueError:
            continue                                         # TeX distribution file
        if not (rel.parent == Path(".") and rel.stem == "main"
                and rel.suffix not in (".tex", ".bbl")):         # skip main.aux, main.out, ...
            found.add(rel)
    if Path("main.bbl") not in found:
        fail("pdflatex did not read main.bbl")
    return sorted(found)


def check_log(log):
    bad = sorted(set(re.findall(r"Warning: (?:Citation|Reference) `([^']+)'", log)))
    if bad:
        fail("undefined citations/references:\n  " + "\n  ".join(bad))
    if "may have changed" in log:
        fail("labels did not settle after 3 passes")
    over = [float(x) for x in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)]
    print(f"overfull hboxes: {len(over)} (largest {max(over, default=0):.1f}pt)")


def main():
    if OUT.exists() and not (OUT / "src").exists() and any(OUT.iterdir()):
        fail(f"{OUT}/ exists and does not look like an earlier output; not overwriting")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "stage"
        shutil.copytree(PAPER, stage, ignore=lambda d, names: [
            n for n in names if Path(d) == PAPER and n in ("build", "main.pdf")])
        transform(stage)
        check_log(pdflatex(stage, recorder=True))
        files = inputs(stage)

        if (OUT / "src").exists():
            shutil.rmtree(OUT / "src")
        for rel in files:
            (OUT / "src" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage / rel, OUT / "src" / rel)

        verify = Path(tmp) / "verify"                        # clean build of exactly the upload
        shutil.copytree(OUT / "src", verify)
        log = pdflatex(verify)
        check_log(log)
        shutil.copy2(verify / "main.pdf", OUT / "main.pdf")

    hits = [(rel, w) for rel in files for w in FORBIDDEN
            if w in (OUT / "src" / rel).read_text(errors="ignore").lower()]
    if hits:
        fail("venue/anonymity strings left in the upload: %s" % hits)
    anon = [str(r) for r in files if "Anonymous" in (OUT / "src" / r).read_text(errors="ignore")]
    if anon:
        print(f"note: 'Anonymous' still appears in {anon} (the MERIT reference)")

    stamp = time.gmtime(int(os.environ.get("SOURCE_DATE_EPOCH", "1791331200")))[:6]
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:  # fixed dates: byte-identical
        for rel in files:
            info = zipfile.ZipInfo(rel.as_posix(), date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, (OUT / "src" / rel).read_bytes())
    pages = re.search(r"Output written on main.pdf \((\d+) pages", log)[1]
    print(f"wrote {OUT}/src/ ({len(files)} files), {ZIP} ({ZIP.stat().st_size / 1e6:.2f} MB), "
          f"{OUT}/main.pdf ({pages} pages)")


if __name__ == "__main__":
    main()
