#!/usr/bin/env python
"""Recent side classifiers (DARNet, DBPNet, ListenNet): their within-trial split vs our
content-disjoint folds, next to CoupleFlow on the content-disjoint folds.

    python scripts/tables/spatial_table.py --out paper/tables/spatial.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow.analysis.aggregate import CACHE, load, pooled  # noqa: E402

DS = [("dtu", "DTU"), ("esaa", "ESAA"), ("fuglsang2020", "Fuglsang'20"), ("uhd255", "KUL-255"),
      ("kuleuven", "KU Leuven$^\\dagger$")]
MODELS = [("darnet", "DARNet"), ("dbpnet", "DBPNet"), ("listennet", "ListenNet")]
WINS = ("1s", "5s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/tables/spatial.tex")
    ap.add_argument("--cf-tag", default="unified")
    args = ap.parse_args()
    rows, res = [], {}
    for ds, name in DS:
        cells = []
        for m, _ in MODELS:
            j = json.loads((CACHE / ds / f"spatial_{m}.json").read_text())
            for w in WINS:
                b = j["protocol_B"].get(w)
                a = j["protocol_A_subject_dep"].get(w)
                res[f"{ds}/{m}/{w}"] = {"B": b, "A_dep": a,
                                        "A_indep": j["protocol_A_subject_indep"].get(w)}
                cells.append(f"{b:.0f}/{a:.0f}" if (a is not None and b is not None) else "--")
        runs = load(args.cf_tag).get(ds, [])
        cf = [pooled(runs, f"competing_{w}", "acc_coupling") for w in WINS]
        res[f"{ds}/coupleflow"] = dict(zip(WINS, cf))
        rows.append(f"{name} & " + " & ".join(cells) + " & "
                    + " & ".join("--" if c is None else f"{100 * c:.0f}" for c in cf) + " \\\\")
    hdr = ("\\begin{tabular}{l" + "cc" * len(MODELS) + "|cc}\\toprule\n & "
           + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in MODELS)
           + " & \\multicolumn{2}{c}{CoupleFlow} \\\\\n"
           + "Dataset & " + " & ".join("1\\,s & 5\\,s" for _ in range(len(MODELS) + 1))
           + " \\\\\\midrule\n")
    tex = hdr + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n"
    Path(args.out).write_text(tex)
    Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=1))
    print(tex)


if __name__ == "__main__":
    main()
