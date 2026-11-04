# GP<->RF head-to-head test (BNS GW170817).  Idiot-proof: `make help`.
#
# Produces the two paper figures from the committed test data (data/all.net):
#   relerr_vs_lnL.png  - relative lnL error vs lnL (GP-RF; GP/RF vs leave-some-out)
#   corner_test.png    - GP posterior (mu-frame NUTS) vs the production RF+AV benchmark
#
# Override anything on the command line, e.g.:
#   make all PY=/path/to/python BENCH='/my/cip_rf_*.xml.gz'

# Interpreter with RIFT + lal + the JAX stack (numpyro, tinygp, corner, matplotlib).
PY    ?= /home/oshaughn/.conda/envs/gwkokab/bin/python
# head_to_head -> export_likelihoods -> rift -> demo -> Code  (the dev RIFT to import)
CODE  := $(abspath ../../../..)
# Benchmark posterior for the corner overlay (your production RF+AV fleet). Optional:
# if absent, the corner is GP-only.
BENCH ?= /home/oshaughn/jaxcip_benchmark/out/cip_rf_*.xml.gz

ENV = PYTHONPATH=$(CODE):$$PYTHONPATH GW_SURROGATE='' BENCH='$(BENCH)'

.PHONY: help data surface posterior figures all clean
.DEFAULT_GOAL := help

help:
	@echo "Targets (run inside an env with RIFT + the JAX stack):"
	@echo "  make data       - verify the committed test data is present (md5)"
	@echo "  make surface    - fit RF+GP on a train split, evaluate held-out -> results/surface.npz"
	@echo "  make posterior  - fit GP on the backbone, mu-frame NUTS -> results/gp_posterior.npz"
	@echo "  make figures    - build paper/figures/{relerr_vs_lnL,corner_test}.png from the results"
	@echo "  make all        - surface + posterior + figures (the whole test; ~15 min on CPU)"
	@echo "  make clean      - remove results/*.npz"
	@echo ""
	@echo "  PY=$(PY)"
	@echo "  CODE=$(CODE)"
	@echo "  BENCH=$(BENCH)   (corner overlay; optional)"

data:
	@test -s data/all.net || (echo "MISSING data/all.net (see PROVENANCE.md)"; false)
	@echo "data/all.net present:"; md5sum data/all.net; wc -l data/all.net

surface: data
	$(ENV) $(PY) run_test.py --stage surface

posterior: data
	$(ENV) $(PY) run_test.py --stage posterior

figures:
	@test -s results/surface.npz      || (echo "run 'make surface' first";   false)
	@test -s results/gp_posterior.npz || (echo "run 'make posterior' first"; false)
	$(ENV) $(PY) make_figures.py
	@echo "Figures written under paper/figures/ (repo root)."

all: surface posterior figures
	@echo "Head-to-head test complete."

clean:
	rm -f results/*.npz
	@echo "cleaned results/ (data/ and figures kept)"
