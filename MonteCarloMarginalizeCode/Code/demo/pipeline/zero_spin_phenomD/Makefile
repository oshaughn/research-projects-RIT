# End-to-end validation: pseudo_pipe build + local ILE_extr run + .dgrid
# consolidation + util_ConstructEOSPosterior.py reconstruction.
#
# Uses the fake-data zero-noise BBH cache from .travis/ILE-GPU-Paper/demos with
# IMRPhenomD (zero spin) and the AV sampler, over a small mass grid.  Designed
# to run in a few minutes on a single laptop -- no condor, no GPU.
#
# Run (inside the RIFT environment):
#     pixi run --manifest-path ../../../../../pixi.toml make all
# or, with a pip-installed RIFT on PATH,
#     make all

RIFT_CODE_ROOT := $(abspath ../../..)
REPO_ROOT      := $(abspath ../../../../..)
CI_DEMO        := $(REPO_ROOT)/.travis/ILE-GPU-Paper/demos

REF_INI    ?= $(CURDIR)/zero_spin_phenomD.ini
COINC      ?= $(REPO_ROOT)/.travis/ref_ini/coinc.xml
CACHE      ?= $(CI_DEMO)/zero_noise.cache
PSD        ?= $(CI_DEMO)/HLV-ILIGO_PSD.xml.gz
GRID       ?= $(CI_DEMO)/overlap-grid.xml.gz

ILE_EXE    := $(RIFT_CODE_ROOT)/bin/integrate_likelihood_extrinsic_batchmode
CONS_EXE   := $(RIFT_CODE_ROOT)/bin/util_ConsolidateDistanceGrids.py
EOS_EXE    := $(RIFT_CODE_ROOT)/bin/util_ConstructEOSPosterior.py

RUN_BUILD := $(CURDIR)/rundir_build
RUN_EXTR  := $(CURDIR)/rundir_extr

# Tiny test: 3 grid rows, AV with low n_eff target -> ~30 s/row on a laptop.
N_EVENTS  ?= 3
N_EFF     ?= 50
N_MAX     ?= 30000

# Fake-data time window from the zero_noise cache (event at 1000000014.236...).
EVT_TIME  := 1000000014.236547946
DATA_T0   := 1000000008
DATA_T1   := 1000000016

ENV = GW_SURROGATE='' \
      RIFT_LOWLATENCY=True \
      SINGULARITY_RIFT_IMAGE=foo \
      SINGULARITY_BASE_EXE_DIR=/usr/bin/ \
      PATH=$(RIFT_CODE_ROOT)/bin:$${PATH} \
      PYTHONPATH=$(RIFT_CODE_ROOT):$${PYTHONPATH:-}

# ILE_extr arguments tuned for the zero-noise fake-data BBH demo (zero-spin
# IMRPhenomD, AV sampler, lnL mode, distance-grid export on).  These match
# what util_RIFT_pseudo_pipe.py + create_event_parameter_pipeline_* would
# emit at the extrinsic stage for the same configuration.
EXTR_ARGS := \
  --n-chunk 10000 --time-marginalization \
  --reference-freq 100.0 --adapt-weight-exponent 0.1 \
  --event-time $(EVT_TIME) --save-P 0.1 \
  --cache-file $(CACHE) \
  --fmin-template 10 --n-max $(N_MAX) --fmax 1700.0 \
  --save-deltalnL inf --l-max 2 --n-eff $(N_EFF) \
  --approximant IMRPhenomD --adapt-floor-level 0.1 \
  --force-xpy --d-max 1000 \
  --psd-file H1=$(PSD) --psd-file L1=$(PSD) \
  --channel-name H1=FAKE-STRAIN --channel-name L1=FAKE-STRAIN \
  --inclination-cosine-sampler --declination-cosine-sampler \
  --data-start-time $(DATA_T0) --data-end-time $(DATA_T1) \
  --inv-spec-trunc-time 0 --no-adapt-after-first --no-adapt-distance \
  --srate 4096 --sampler-method AV --internal-use-lnL \
  --export-marginal-distance-grid \
  --sim-xml overlap-grid.xml.gz \
  --n-events-to-analyze $(N_EVENTS) \
  --output-file demo_extr

.PHONY: help inputs build validate-build run-extr consolidate posterior all clean

help:
	@echo "Targets:"
	@echo "  make inputs         - verify fake-data inputs are available"
	@echo "  make build          - run util_RIFT_pseudo_pipe.py to build a pipeline (validates threading)"
	@echo "  make validate-build - check the build produced a .dgrid export and consolidation job"
	@echo "  make run-extr       - run ILE_extr locally (no condor) on the first $(N_EVENTS) grid rows"
	@echo "  make consolidate    - run util_ConsolidateDistanceGrids.py on the produced .dgrid files"
	@echo "  make posterior      - run util_ConstructEOSPosterior.py on the consolidated .dgrid -> joint posterior"
	@echo "  make all            - full chain"
	@echo "  make clean          - remove rundir_build, rundir_extr, fake.cache"

inputs:
	@test -s "$(CACHE)" || (echo "missing fake cache $(CACHE) (run the ILE-GPU-Paper demo Makefile first)" && false)
	@test -s "$(PSD)"   || (echo "missing PSD $(PSD)" && false)
	@test -s "$(GRID)"  || (echo "missing input grid $(GRID)" && false)
	@test -s "$(REF_INI)" && test -s "$(COINC)" || (echo "missing reference ini/coinc" && false)
	@touch "$(CURDIR)/fake.cache"
	@echo "All inputs present."

# Build a RIFT pipeline with the per-distance grid export turned on (forces
# AV sampler + IMRPhenomD + zero spin via CLI overrides on the GW150914
# reference ini).  We use --fake-data-cache to keep the build offline; the
# resulting pipeline could run on the real fake-data cache by setting the
# right event-time/coinc, but that is beyond this build-validation target.
build: inputs
	rm -rf "$(RUN_BUILD)"
	$(ENV) util_RIFT_pseudo_pipe.py \
	    --use-ini "$(REF_INI)" \
	    --use-coinc "$(COINC)" \
	    --use-rundir "$(RUN_BUILD)" \
	    --fake-data-cache "$(CURDIR)/fake.cache" \
	    --add-extrinsic \
	    --export-marginal-distance-grid \
	    --assume-nospin \
	    --approx IMRPhenomD \
	    --ile-sampler-method AV
	$(MAKE) validate-build

validate-build:
	@test -s "$(RUN_BUILD)/ILE_extr.sub"
	@grep -q -- "--export-marginal-distance-grid" "$(RUN_BUILD)/ILE_extr.sub"
	@grep -q -- "--sampler-method AV"             "$(RUN_BUILD)/ILE_extr.sub"
	@grep -q -- "--internal-use-lnL"              "$(RUN_BUILD)/ILE_extr.sub"
	@! grep -q -- "--distance-marginalization "   "$(RUN_BUILD)/ILE_extr.sub"
	@grep -q -- "IMRPhenomD"                      "$(RUN_BUILD)/ILE_extr.sub"
	@test -s "$(RUN_BUILD)/consolidate_dgrid.sub"
	@test -s "$(RUN_BUILD)/consolidate_dgrid.sh"
	@grep -q "consolidate_dgrid"                  $(RUN_BUILD)/*.dag
	@echo "OK: pipeline built; ILE_extr.sub carries AV+IMRPhenomD+grid-export, distance marg disabled only at extrinsic stage, consolidate_dgrid wired into DAG."

# Direct local run of the ILE_extr binary on a small subset of the grid -- no
# condor.  Produces real *.dgrid files we can then consolidate and reconstruct
# the posterior from.  Event-time/cache match the .travis/ILE-GPU-Paper fake
# data (the GW150914 ini used for the build step has a different event time
# and no matching frames; running directly here avoids that mismatch while
# still exercising the same ILE code path).
run-extr: inputs
	rm -rf "$(RUN_EXTR)"
	mkdir -p "$(RUN_EXTR)"
	cp "$(GRID)" "$(RUN_EXTR)/overlap-grid.xml.gz"
	@echo "Running ILE_extr on $(N_EVENTS) grid rows (AV, IMRPhenomD zero-spin, lnL mode, --export-marginal-distance-grid)..."
	cd "$(RUN_EXTR)" && $(ENV) "$(ILE_EXE)" $(EXTR_ARGS)
	@echo "Produced .dgrid files:"; ls "$(RUN_EXTR)"/demo_extr_*_.dgrid

consolidate: run-extr
	@echo "Consolidating per-event .dgrid files..."
	cd "$(RUN_EXTR)" && $(ENV) "$(CONS_EXE)" --input-glob 'demo_extr_*_.dgrid' --output all_dgrid.dat
	@test -s "$(RUN_EXTR)/all_dgrid.dat"
	@head -1 "$(RUN_EXTR)/all_dgrid.dat" | grep -q "^# lnL sigmaL m1 m2"
	@n_data=$$(grep -cv "^#" "$(RUN_EXTR)/all_dgrid.dat"); echo "all_dgrid.dat: $$n_data data rows"

posterior: consolidate
	@echo "Reconstructing joint (m1, m2, dist) posterior from consolidated grid..."
	cd "$(RUN_EXTR)" && $(ENV) "$(EOS_EXE)" \
	    --fname all_dgrid.dat \
	    --parameter m1 --parameter m2 --parameter dist \
	    --integration-parameter-range 'm1:[24,34]' \
	    --integration-parameter-range 'm2:[24,34]' \
	    --integration-parameter-range 'dist:[10,500]' \
	    --lnL-offset 30 \
	    --fname-output-samples joint_posterior \
	    --fname-output-integral joint_evidence \
	    --no-plots
	@test -s "$(RUN_EXTR)/joint_posterior.dat"
	@n_samples=$$(grep -cv "^#" "$(RUN_EXTR)/joint_posterior.dat"); echo "joint_posterior.dat: $$n_samples posterior samples"
	@python -c "import numpy as np; d=np.genfromtxt('$(RUN_EXTR)/joint_posterior.dat', names=True); print(' summary:', {n: (round(float(d[n].mean()),3), round(float(d[n].std()),3)) for n in d.dtype.names if n in ('m1','m2','dist','lnL')})"
	@echo "OK: end-to-end validation complete -- pipeline built, ILE_extr produced .dgrid output, consolidated, posterior reconstructed."

all: build run-extr consolidate posterior
	@echo "All validation steps passed."

clean:
	rm -rf "$(RUN_BUILD)" "$(RUN_EXTR)" "$(CURDIR)/fake.cache"
