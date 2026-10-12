#!/bin/bash
# Compile paper/main.tex -> paper/main.pdf (pdflatex + bibtex; build products in paper/build/).
set -euo pipefail
cd "$(dirname "$0")/../../paper"
O=build; mkdir -p "$O"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$O" main.tex > "$O/pass1.log"
(cd "$O" && BIBINPUTS=..: bibtex main > bibtex.log)       # BIBINPUTS must point at paper/
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$O" main.tex > "$O/pass2.log"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$O" main.tex > "$O/pass3.log"
cp "$O/main.pdf" main.pdf
grep -E "Warning: (Citation|Reference)" "$O/main.log" | sort -u || true
# The main text must end on page 8 (AISTATS limit): the last line of Limitations.
if command -v pdftotext >/dev/null; then
    p8=$(pdftotext -f 8 -l 8 main.pdf -)          # no pipe into grep -q: SIGPIPE + pipefail
    if [[ $p8 == *"match–mismatch only"* ]]; then echo "page check: main text ends on p. 8"
    else echo "page check: WARNING -- 'match–mismatch only' is not on p. 8"; fi
fi
echo "wrote paper/main.pdf ($(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2}') pages)"
