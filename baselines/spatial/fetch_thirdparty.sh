#!/usr/bin/env bash
# Fetch the authors' DARNet, DBPNet and ListenNet repositories at the commits used in the paper.
# baselines/spatial/run.py imports their model definitions unmodified; the repositories carry no
# licence, so they are not redistributed with this code.
#
# Clones into $1, else $COUPLEFLOW_THIRDPARTY, else baselines/spatial/thirdparty (the default of
# coupleflow/paths.py), under the directory names run.py imports from:
#   darnet/     https://github.com/fchest/darnet     @ 6254e7ebd2aa2a63b64b407a1be3013088381c10  model.py
#   DBPNet/     https://github.com/fchest/DBPNet     @ 99a153fdbb8276f227ac6393529a24af9afc5dbc  model_module.py
#   ListenNet/  https://github.com/fchest/ListenNet  @ 286b8ac4bddc7e6baee870940457809131249737  dep/model.py
# Needs git and network access (run it on a login node; compute nodes may have no internet).
# A clone already at the pinned commit is left alone; one at another commit is fetched and
# checked out at the pinned commit (git refuses if it has local changes).  ~6 MB in all.
#   bash baselines/spatial/fetch_thirdparty.sh [DEST]
#   DRY_RUN=1 bash baselines/spatial/fetch_thirdparty.sh            # print the commands only
set -euo pipefail
cd "$(dirname "$0")/../.."
DRY_RUN=${DRY_RUN:-0}
DEST=${1:-${COUPLEFLOW_THIRDPARTY:-baselines/spatial/thirdparty}}
REPOS=(
    "darnet    https://github.com/fchest/darnet    6254e7ebd2aa2a63b64b407a1be3013088381c10 model.py"
    "DBPNet    https://github.com/fchest/DBPNet    99a153fdbb8276f227ac6393529a24af9afc5dbc model_module.py"
    "ListenNet https://github.com/fchest/ListenNet 286b8ac4bddc7e6baee870940457809131249737 dep/model.py"
)
run() { if [ "$DRY_RUN" = 1 ]; then echo "$*"; else "$@"; fi; }

if [ ! -d "$DEST" ]; then run mkdir -p "$DEST"; fi
for r in "${REPOS[@]}"; do
    read -r name url sha file <<<"$r"
    dir=$DEST/$name
    if [ -d "$dir/.git" ]; then
        if [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ]; then
            echo "# $dir is at $sha"
            continue
        fi
        run git -C "$dir" fetch origin
    else
        run git clone "$url" "$dir"
    fi
    run git -C "$dir" -c advice.detachedHead=false checkout "$sha"
    if [ "$DRY_RUN" != 1 ] && [ ! -f "$dir/$file" ]; then
        echo "$dir/$file missing after checkout" >&2
        exit 1
    fi
done
