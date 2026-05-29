# In-loop calibration marginalization demo (3 detectors: H1, L1, V1)
#
# Uses the zero-spin synthetic CI data shipped in .travis/ILE-GPU-Paper/demos.
# Runs the ILE binary directly (no condor) with distance marginalization and
# in-loop calibration marginalization, comparing:
#   - baseline : no calibration marginalization
#   - loop     : in-loop calmarg, Option B (cal_method='loop'; CPU or GPU)
#   - fused    : in-loop calmarg, Option C (fused CUDA kernel; GPU only)
#
# See README.md for the physics and what each target demonstrates.

RIFT_CODE_ROOT := $(abspath ../../..)
REPO_ROOT      := $(abspath ../../../../..)
CI_DEMO        := $(REPO_ROOT)/.travis/ILE-GPU-Paper/demos

CACHE ?= $(CI_DEMO)/zero_noise.cache
PSD   ?= $(CI_DEMO)/HLV-ILIGO_PSD.xml.gz
GRID  ?= $(CI_DEMO)/overlap-grid.xml.gz

ILE := $(RIFT_CODE_ROOT)/bin/integrate_likelihood_extrinsic_batchmode
ENV := PYTHONPATH=$(RIFT_CODE_ROOT):$${PYTHONPATH:-} PATH=$(RIFT_CODE_ROOT)/bin:$${PATH}

# Tunables.  NCHUNK/NMAX are kept small so the demo runs on a modest GPU; raise
# them (and NEFF) for a production-quality run on a larger card.
NCAL    ?= 40          # number of calibration realizations
NCHUNK  ?= 1000        # extrinsic samples per evaluation block
NMAX    ?= 2000        # max extrinsic samples
NEFF    ?= 100         # target effective samples
SEED    ?= 1234        # fixes BOTH the cal draws and the extrinsic draws
DMAX    ?= 1000

# Detector / data configuration for the synthetic injection
EVENT_TIME := 1000000014.236547946
IFO_PSD    := --psd-file H1=$(PSD) --psd-file L1=$(PSD) --psd-file V1=$(PSD)
IFO_CHAN   := --channel-name H1=FAKE-STRAIN --channel-name L1=FAKE-STRAIN --channel-name V1=FAKE-STRAIN

# Common ILE arguments (distance marginalization ON; 3 detectors)
COMMON := --vectorized --gpu \
  --n-chunk $(NCHUNK) --n-max $(NMAX) --n-eff $(NEFF) --seed $(SEED) \
  --time-marginalization --sim-xml $(GRID) --n-events-to-analyze 1 \
  --reference-freq 100.0 --event-time $(EVENT_TIME) --approximant SEOBNRv4 --l-max 2 \
  --cache-file $(CACHE) --fmin-template 10 --fmax 1700.0 \
  --d-min 1 --d-max $(DMAX) \
  --distance-marginalization --distance-marginalization-lookup-table distance_marg.npz \
  $(IFO_PSD) $(IFO_CHAN) \
  --inclination-cosine-sampler --declination-cosine-sampler \
  --data-start-time 1000000008 --data-end-time 1000000016 --inv-spec-trunc-time 0 \
  --srate 4096 --sampler-method GMM

CALMARG := --calibration-envelope-directory $(CURDIR)/cal_env --calibration-n-realizations $(NCAL)

.PHONY: help inputs run-baseline run-loop run-fused compare verify-exact all clean

help:
	@echo "Targets:"
	@echo "  make inputs       - build the distance-marginalization table + cal envelope files"
	@echo "  make verify-exact - DETERMINISTIC unit check: loop == fused == reference to ~1e-14"
	@echo "  make run-baseline - ILE, no calibration marginalization"
	@echo "  make run-loop     - ILE + in-loop calmarg, Option B (loop; CPU or GPU)"
	@echo "  make run-fused    - ILE + in-loop calmarg, Option C (fused GPU kernel)"
	@echo "  make compare      - print marginalized lnL from the three runs"
	@echo "  make all          - inputs + verify-exact + the three runs + compare"
	@echo "  make clean        - remove generated inputs/outputs"
	@echo ""
	@echo "Tunables (make VAR=...): NCAL=$(NCAL) NCHUNK=$(NCHUNK) NMAX=$(NMAX) NEFF=$(NEFF) SEED=$(SEED)"

inputs: distance_marg.npz cal_env/H1.txt

distance_marg.npz:
	$(ENV) $(RIFT_CODE_ROOT)/bin/util_InitMargTable --d-min 1 --d-max $(DMAX) --out distance_marg.npz

cal_env/H1.txt:
	$(ENV) python tools/make_cal_envelopes.py --out-dir $(CURDIR)/cal_env --ifos H1,L1,V1

# Deterministic exact check on identical inputs (bypasses the stochastic sampler):
# loop (Option B), fused (Option C) and a brute-force reference must agree to ~1e-14,
# using THIS demo's real distance-marginalization table.
verify-exact: distance_marg.npz
	$(ENV) python -m RIFT.calmarg.backtest_calmarg --backend gpu \
	  --loglikelihood distmarg --real-table distance_marg.npz \
	  --dets H1,L1,V1 --n-cal $(NCAL) --npts-extrinsic 512

run-baseline: inputs
	$(ENV) $(ILE) $(COMMON) --output-file out_baseline
	@echo "baseline lnL:"; cat out_baseline*.dat

run-loop: inputs
	$(ENV) $(ILE) $(COMMON) $(CALMARG) --output-file out_loop
	@echo "loop lnL:"; cat out_loop*.dat

run-fused: inputs
	$(ENV) $(ILE) $(COMMON) $(CALMARG) --calibration-fused-kernel --output-file out_fused
	@echo "fused lnL:"; cat out_fused*.dat

compare:
	$(ENV) python tools/compare_lnL.py \
	  baseline=out_baseline_0_.dat loop=out_loop_0_.dat fused=out_fused_0_.dat

all: inputs verify-exact run-baseline run-loop run-fused compare

clean:
	rm -f distance_marg.npz out_*.dat *.xml.gz
	rm -rf cal_env
