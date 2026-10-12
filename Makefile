# Convenience targets; every target is a thin wrapper around a script in scripts/.
.PHONY: help install test assets paper tables pdf train baselines data

help:            ## list targets
	@grep -E "^[a-z-]+:.*##" $(MAKEFILE_LIST) | sed "s/:.*##/ -/"

install:         ## install the package and test tools (in an existing Python 3.10+ env)
	pip install -e ".[dev]"

test:            ## CPU unit tests (12, ~1 min)
	OMP_NUM_THREADS=4 python -m pytest -q tests

assets:          ## download the App. G margins (release asset) into runs/
	bash scripts/fetch_release_assets.sh

paper:           ## all tables, macros and figures from the shipped results, then the PDF
	bash scripts/build_paper.sh

tables:          ## tables, macros and figures only (no LaTeX)
	NO_PDF=1 bash scripts/build_paper.sh

pdf:             ## compile paper/main.tex only
	bash scripts/paper/build_pdf.sh

data:            ## (cluster) build bundles, content groups and checks; see scripts/data/README.md
	bash scripts/data/build_bundles.sh && bash scripts/data/stimulus_groups.sh && bash scripts/data/checks.sh

baselines:       ## (cluster) all baselines; see scripts/baselines/README.md
	bash scripts/baselines/linear.sh && bash scripts/baselines/deep.sh && bash scripts/baselines/spatial.sh

train:           ## (cluster, GPU) the main CoupleFlow runs; other experiments: scripts/train/README.md
	bash scripts/train/main.sh
