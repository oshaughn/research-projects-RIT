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

CACHE   ?= $(CI_DEMO)/zero_noise.cache
PSD     ?= $(CI_DEMO)/HLV-ILIGO_PSD.xml.gz
# Analyze the INJECTION itself as the (single) template, so the signal is actually
# present (matched template -> lnL ~ rho^2/2).  Using a far-off grid point instead
# gives lnL ~ 0 ("no signal"), which makes the calmarg effect invisible.
SIM_XML ?= $(CI_DEMO)/mdc.xml.gz

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
SAMPLER ?= AV          # adaptive-volume sampler: mature, stable GPU code path
                       #   (GMM is newer/heavier on GPU; override with SAMPLER=GMM)

# Detector / data configuration for the synthetic injection
EVENT_TIME := 1000000014.236547946
IFO_PSD    := --psd-file H1=$(PSD) --psd-file L1=$(PSD) --psd-file V1=$(PSD)
IFO_CHAN   := --channel-name H1=FAKE-STRAIN --channel-name L1=FAKE-STRAIN --channel-name V1=FAKE-STRAIN

# Common ILE arguments (distance marginalization ON; 3 detectors)
COMMON := --vectorized --gpu \
  --n-chunk $(NCHUNK) --n-max $(NMAX) --n-eff $(NEFF) --seed $(SEED) \
  --time-marginalization --sim-xml $(SIM_XML) --n-events-to-analyze 1 \
  --reference-freq 100.0 --event-time $(EVENT_TIME) --approximant SEOBNRv4 --l-max 2 \
  --cache-file $(CACHE) --fmin-template 10 --fmax 1700.0 \
  --d-min 1 --d-max $(DMAX) \
  --distance-marginalization --distance-marginalization-lookup-table distance_marg.npz \
  $(IFO_PSD) $(IFO_CHAN) \
  --inclination-cosine-sampler --declination-cosine-sampler \
  --data-start-time 1000000008 --data-end-time 1000000016 --inv-spec-trunc-time 0 \
  --srate 4096 --sampler-method $(SAMPLER)

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
	@echo "  make lowsnr-inputs- generate a fainter (~SNR 9) 3-IFO injection + cache"
	@echo "  make low-snr      - lowsnr-inputs, then 'make all' on the quiet cache"
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

# --- Low-SNR variant -------------------------------------------------------
# The bundled CI injection is network SNR ~17.5 (m1=35,m2=30 at 200 Mpc).  This
# generates a fainter copy (same source, larger distance -> ~SNR 9) on the fly so
# the full-sampler loop-vs-fused agreement is resolved well above Monte-Carlo
# noise.  No binaries are committed; the frames/cache are regenerated locally,
# exactly as the CI data itself is made.
INJ_M1     ?= 35
INJ_M2     ?= 30
INJ_DIST   ?= 400        # Mpc -> ~SNR 9 network (vs ~17.5 for the bundled 200 Mpc data)
INJ_APPROX ?= SEOBNRv4
INJ_FMIN   ?= 8
LOWSNR_CACHE := $(CURDIR)/lowsnr.cache

lowsnr-inputs: $(LOWSNR_CACHE)
$(LOWSNR_CACHE):
	$(ENV) util_WriteInjectionFile.py --parameter m1 --parameter-value $(INJ_M1) \
	  --parameter m2 --parameter-value $(INJ_M2) --fname mdc_lowsnr --approx $(INJ_APPROX) \
	  --parameter tref --parameter-value $(EVENT_TIME) \
	  --parameter dist --parameter-value $(INJ_DIST) --parameter fmin --parameter-value $(INJ_FMIN)
	$(ENV) util_WriteFrameAndCacheFromXML.sh mdc_lowsnr.xml.gz 0 lowsnr $(INJ_APPROX)
	@echo "# injected network SNR:"; $(ENV) util_FrameZeroNoiseSNR.py --cache $(LOWSNR_CACHE) \
	  --psd-file H1=$(PSD) --psd-file L1=$(PSD) --psd-file V1=$(PSD) 2>/dev/null | tail -1

# Run the full comparison on the quiet injection (override CACHE + matched template).
low-snr: lowsnr-inputs
	$(MAKE) all CACHE=$(LOWSNR_CACHE) SIM_XML=$(CURDIR)/mdc_lowsnr.xml.gz

clean:
	rm -f distance_marg.npz out_*.dat *.xml.gz lowsnr.cache snr_*.dat snr-report.txt
	rm -rf cal_env lowsnr_mdc
