# Standard pipeline-build demos for util_RIFT_pseudo_pipe.py
#
# These targets BUILD (do not submit) a RIFT DAG from a reference .ini + coinc
# using fake data, then verify the generated condor submit files.  They are
# fast, need no real frames/GPUs, and double as regression tests that the
# per-distance likelihood export flags (Plan A grid, Plan B slices) thread all
# the way through util_RIFT_pseudo_pipe.py -> create_event_parameter_pipeline_*
# -> the ILE_extr submit file.
#
# Run inside the RIFT environment, e.g.
#     pixi run --manifest-path ../../../../../pixi.toml make all
# or, with a pip-installed RIFT on PATH,
#     make all

RIFT_CODE_ROOT := $(abspath ../..)
REPO_ROOT := $(abspath ../../../..)

REF_INI ?= $(REPO_ROOT)/.travis/ref_ini/GW150914.ini
COINC   ?= $(REPO_ROOT)/.travis/ref_ini/coinc.xml
FAKE_CACHE ?= $(CURDIR)/fake.cache

# Build-only environment: pretend we have a singularity image + datafind, use
# fake data, and put the in-tree bin/ on PATH so the pipeline builder and ILE
# executable resolve.
ENV = RIFT_LOWLATENCY=True \
      SINGULARITY_RIFT_IMAGE=foo \
      SINGULARITY_BASE_EXE_DIR=/usr/bin/ \
      GW_SURROGATE='' \
      PATH=$(RIFT_CODE_ROOT)/bin:$${PATH} \
      PYTHONPATH=$(RIFT_CODE_ROOT):$${PYTHONPATH:-}

PIPE = util_RIFT_pseudo_pipe.py \
       --use-ini $(REF_INI) \
       --use-coinc $(COINC) \
       --fake-data-cache $(FAKE_CACHE) \
       --add-extrinsic

.PHONY: help inputs baseline grid slices validate-grid validate-slices all clean

help:
	@echo "Targets:"
	@echo "  make inputs          - verify reference ini/coinc are available"
	@echo "  make baseline        - build a standard pipeline (no per-distance export)"
	@echo "  make grid            - build + validate Plan-A distance-grid export"
	@echo "  make slices          - build + validate Plan-B distance-slice export"
	@echo "  make all             - run baseline, grid, and slices"
	@echo "  make clean           - remove generated run directories"

inputs:
	@test -s "$(REF_INI)"  || (echo "missing $(REF_INI)"  && false)
	@test -s "$(COINC)"    || (echo "missing $(COINC)"    && false)
	@touch "$(FAKE_CACHE)"
	@echo "Using ini:   $(REF_INI)"
	@echo "Using coinc: $(COINC)"

baseline: inputs
	rm -rf "$(CURDIR)/rundir_baseline"
	$(ENV) $(PIPE) --use-rundir "$(CURDIR)/rundir_baseline"
	@test -s "$(CURDIR)/rundir_baseline/ILE_extr.sub"
	@! grep -q -- "--export-marginal-distance-grid" "$(CURDIR)/rundir_baseline/ILE_extr.sub"
	@! grep -q -- "--export-distance-slices" "$(CURDIR)/rundir_baseline/ILE_extr.sub"
	@echo "OK: baseline pipeline built; no per-distance export leaked into ILE_extr.sub"

grid: inputs
	rm -rf "$(CURDIR)/rundir_grid"
	$(ENV) $(PIPE) --use-rundir "$(CURDIR)/rundir_grid" --export-marginal-distance-grid
	$(MAKE) validate-grid

validate-grid:
	@test -s "$(CURDIR)/rundir_grid/ILE_extr.sub"
	@grep -q -- "--export-marginal-distance-grid" "$(CURDIR)/rundir_grid/ILE_extr.sub"
	@grep -q -- "--internal-use-lnL"               "$(CURDIR)/rundir_grid/ILE_extr.sub"
	@! grep -q -- "--export-marginal-distance-grid" "$(CURDIR)/rundir_grid/ILE.sub"
	@grep -q -- "--distance-marginalization "      "$(CURDIR)/rundir_grid/args_ile.txt"
	@grep -q -- "--distance-marginalization "      "$(CURDIR)/rundir_grid/ILE.sub"
	@! grep -q -- "--distance-marginalization "    "$(CURDIR)/rundir_grid/ILE_extr.sub"
	@echo "OK: Plan-A grid export only on ILE_extr.sub; distance marginalization kept on intrinsic ILE.sub, disabled only at the extrinsic stage"

slices: inputs
	rm -rf "$(CURDIR)/rundir_slices"
	$(ENV) $(PIPE) --use-rundir "$(CURDIR)/rundir_slices" \
	    --export-distance-slices 10 \
	    --export-distance-slices-wing-delta-lnL 7.0 \
	    --export-distance-slices-skip-threshold 1.0
	$(MAKE) validate-slices

validate-slices:
	@test -s "$(CURDIR)/rundir_slices/ILE_extr.sub"
	@grep -q -- "--export-distance-slices 10"        "$(CURDIR)/rundir_slices/ILE_extr.sub"
	@grep -q -- "--distance-slice-wing-delta-lnL 7.0" "$(CURDIR)/rundir_slices/ILE_extr.sub"
	@grep -q -- "--distance-slice-skip-threshold 1.0" "$(CURDIR)/rundir_slices/ILE_extr.sub"
	@grep -q -- "--internal-use-lnL"                  "$(CURDIR)/rundir_slices/ILE_extr.sub"
	@! grep -q -- "--export-distance-slices"          "$(CURDIR)/rundir_slices/ILE.sub"
	@grep -q -- "--distance-marginalization "         "$(CURDIR)/rundir_slices/args_ile.txt"
	@grep -q -- "--distance-marginalization "         "$(CURDIR)/rundir_slices/ILE.sub"
	@! grep -q -- "--distance-marginalization "       "$(CURDIR)/rundir_slices/ILE_extr.sub"
	@echo "OK: Plan-B slice export only on ILE_extr.sub; distance marginalization kept on intrinsic ILE.sub, disabled only at the extrinsic stage"

all: baseline grid slices
	@echo "All pipeline-build demos passed."

clean:
	rm -rf "$(CURDIR)/rundir_baseline" "$(CURDIR)/rundir_grid" "$(CURDIR)/rundir_slices" "$(FAKE_CACHE)"
