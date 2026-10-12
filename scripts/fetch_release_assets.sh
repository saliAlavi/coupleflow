#!/bin/bash
# Download the per-run arrays that are too large for git and unpack them into runs/:
#   conf_margins.tar   runs/<tag>/<ds>_fold<k>/conf_margins.npz  (405 files, read by
#                      scripts/tables/confidence_appendix.py for App. G; written by
#                      scripts/results/dump_margins.py from each best.pt)
# The repository is private while the paper is under review, so the download needs a token:
#   gh auth login            (GitHub CLI), or
#   export GITHUB_TOKEN=...  (a token with read access to the repository)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${REPO:-saliAlavi/coupleflow}
TAG=${TAG:-v1.0-aistats2027}
ASSET=conf_margins.tar
mkdir -p .cache
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --pattern "$ASSET" --dir .cache --clobber
else
    : "${GITHUB_TOKEN:?set GITHUB_TOKEN (or log in with gh) to download from the private repository}"
    api=https://api.github.com/repos/$REPO
    id=$(curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" "$api/releases/tags/$TAG" |
         python -c "import json,sys; print(next(a['id'] for a in json.load(sys.stdin)['assets'] if a['name'] == '$ASSET'))")
    curl -fL -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/octet-stream" \
         -o ".cache/$ASSET" "$api/releases/assets/$id"
fi
tar -xf ".cache/$ASSET" -C runs
echo "unpacked $(tar -tf ".cache/$ASSET" | grep -c conf_margins.npz) conf_margins.npz files into runs/"
