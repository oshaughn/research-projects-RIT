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

# Review matrix toggles:
#   BACKEND=gpu (CUDA kernels) | cpu (numpy; runs on a laptop, no CUDA)
#   DMARG=1 (distance marginalization on, fused distmarg kernel)
#         | 0 (off, default-helper fused kernel)
BACKEND ?= gpu
DMARG   ?= 1

ifeq ($(BACKEND),cpu)
  # --gpu --force-xpy selects the xpy code path; on a machine WITHOUT cupy (e.g. a
  # Mac) that path is numpy, i.e. CPU.  (On a machine with cupy it stays on the GPU
  # xpy path -- there is no separate force-numpy switch -- so use BACKEND=cpu where
  # cupy is absent.  The deterministic 'make verify-exact BACKEND=cpu' always uses
  # numpy regardless, since the backtest selects the backend directly.)
  ACCEL := --vectorized --gpu --force-xpy
else
  ACCEL := --vectorized --gpu
endif

ifeq ($(DMARG),1)
  DMARG_FLAGS := --distance-marginalization --distance-marginalization-lookup-table $(CURDIR)/distance_marg.npz
  DMARG_DEP   := distance_marg.npz
  VERIFY_LL   := distmarg --real-table $(CURDIR)/distance_marg.npz
else
  DMARG_FLAGS :=
  DMARG_DEP   :=
  VERIFY_LL   := default
endif

# PHASE=1 adds analytic phase marginalization to the deterministic check (verify-exact).
PHASE ?= 0
ifeq ($(PHASE),1)
  VERIFY_PHASE := --phase-marginalization
else
  VERIFY_PHASE :=
endif

# Detector / data configuration for the synthetic injection
EVENT_TIME := 1000000014.236547946
IFO_PSD    := --psd-file H1=$(PSD) --psd-file L1=$(PSD) --psd-file V1=$(PSD)
IFO_CHAN   := --channel-name H1=FAKE-STRAIN --channel-name L1=FAKE-STRAIN --channel-name V1=FAKE-STRAIN

# Common ILE arguments (3 detectors)
COMMON := $(ACCEL) \
  --n-chunk $(NCHUNK) --n-max $(NMAX) --n-eff $(NEFF) --seed $(SEED) \
  --time-marginalization --sim-xml $(SIM_XML) --n-events-to-analyze 1 \
  --reference-freq 100.0 --event-time $(EVENT_TIME) --approximant SEOBNRv4 --l-max 2 \
  --cache-file $(CACHE) --fmin-template 10 --fmax 1700.0 \
  --d-min 1 --d-max $(DMAX) $(DMARG_FLAGS) \
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
	@echo "Review matrix:  BACKEND=gpu|cpu   DMARG=1|0   PHASE=0|1"
	@echo "  e.g.  make verify-exact BACKEND=cpu DMARG=0 PHASE=1   (laptop, no CUDA, no distance marg, phase marg)"

inputs: $(DMARG_DEP) cal_env/H1.txt

distance_marg.npz:
	$(ENV) $(RIFT_CODE_ROOT)/bin/util_InitMargTable --d-min 1 --d-max $(DMAX) --out distance_marg.npz

cal_env/H1.txt:
	$(ENV) python tools/make_cal_envelopes.py --out-dir $(CURDIR)/cal_env --ifos H1,L1,V1

# Deterministic exact check on identical inputs (bypasses the stochastic sampler):
# loop (Option B), fused (Option C) and a brute-force reference must agree to ~1e-14.
# Honours BACKEND (gpu/cpu) and DMARG (distmarg/default helper).
verify-exact: $(DMARG_DEP)
	$(ENV) python -m RIFT.calmarg.backtest_calmarg --backend $(BACKEND) \
	  --loglikelihood $(VERIFY_LL) $(VERIFY_PHASE) \
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

# --- Full DAG test (Option C adaptive cal pilot) on local condor ----------------
# Builds and (optionally) submits a real RIFT DAG that exercises the per-iteration
# calibration PILOT wiring: calpilot_N runs in parallel with CIP_N, learns a cal
# proposal, and SEEDS the wide ILE jobs of iteration N+1 via a breadcrumb.  Mirrors
# the known-good .travis/ILE-GPU-Paper test_workflow_batch_gpu_lowlatency target
# (calls create_event_parameter_pipeline_BasicIteration directly), 3-IFO + GPU + AV,
# with --calmarg-pilot added.  Designed to run on a single machine with condor + GPU
# (e.g. cardassia); select the card with CUDA_VISIBLE_DEVICES if needed.
RUN_DAG       := $(CURDIR)/rundir_dag
# Toggles: PILOT=1 adds the adaptive cal pilots (seed wide_{N+1}); PILOT=0 is VANILLA
# in-loop calmarg (no pilots -- the clean production path, best for faint sources).
# FUSED=1 uses the fused GPU kernel (Option C).  E.g. the priority vanilla-fused demo:
#     make dag-build PILOT=0 FUSED=1   &&   make dag-run PILOT=0 FUSED=1
PILOT         ?= 1
FUSED         ?= 0
N_IT_DAG      ?= 2        # >=2 so iteration 1 consumes the seed from calpilot_0 (PILOT=1)
NPTS_GRID_DAG ?= 80       # initial intrinsic grid size
NPTS_IT_DAG   ?= 12       # intrinsic points / ILE jobs per iteration
NCAL_DAG      ?= 20
NMAX_DAG      ?= 40000
NEFF_DAG      ?= 15
NCHUNK_DAG    ?= 10000
N_COPIES_DAG  ?= 1        # ILE job duplicates (builder default 2); 1 -> no redundancy,
                          # ~2x faster turnaround on a single-card test
DMARG_DAG     ?= 1        # distance marginalization in the wide ILE (default ON).  With it,
                          # distance is integrated analytically (not sampled), the clean
                          # production path.  (Aside: --no-adapt-distance is NOT a freeze --
                          # distance is then UNIFORMLY sampled, fine for low-SNR with a
                          # sensible d-min/d-max; it is simply moot once distmarg is on.)

ifeq ($(DMARG_DAG),1)
  DAG_DMARG_FLAG := --distance-marginalization --distance-marginalization-lookup-table $(CURDIR)/distance_marg.npz
  DAG_DMARG_DEP  := distance_marg.npz
else
  DAG_DMARG_FLAG :=
  DAG_DMARG_DEP  :=
endif

ifeq ($(FUSED),1)
  DAG_FUSED_FLAG := --calibration-fused-kernel
else
  DAG_FUSED_FLAG :=
endif
# Optional zero-cal extrinsic burn-in (task #24): set BURN_IN_NEFF to a target n_eff to
# adapt the sampler on the cheap zero-cal likelihood before the full cal-marg integral.
ifneq ($(BURN_IN_NEFF),)
  DAG_BURNIN_FLAG := --calibration-burn-in-neff $(BURN_IN_NEFF)
else
  DAG_BURNIN_FLAG :=
endif
ifeq ($(PILOT),1)
  DAG_PILOT_CEPP := --calmarg-pilot --calmarg-pilot-cadence 1 --calmarg-pilot-max-it 1
else
  DAG_PILOT_CEPP :=
endif

# NOTE: adapt-weight-exponent left at the ILE DEFAULT (1.0).  For the AV sampler used here
# this option is a NO-OP (it matters for the OTHER integrators -- GMM/portfolio -- so do
# not drop it globally).  n_eff is naturally low at small n-max (~3 at 100k) regardless:
# 100k samples is SHORT by RIFT standards; production runs use millions and let the AV
# integrator creep n_eff upward.  Baseline (no cal) and calmarg give ~equal n_eff here, so
# cal is not the bottleneck -- the moderate SNR (~17.5; "loud" is 40+) x sample count is.
DAG_ILE_ARGS := --n-chunk $(NCHUNK_DAG) --time-marginalization --sim-xml overlap-grid.xml.gz \
  --reference-freq 100.0 --event-time $(EVENT_TIME) --save-P 0.1 \
  --cache-file $(CACHE) --fmin-template 10 --n-max $(NMAX_DAG) --fmax 1700.0 --save-deltalnL inf \
  --l-max 2 --n-eff $(NEFF_DAG) --approximant SEOBNRv4 --d-min 1 --d-max $(DMAX) \
  --psd-file H1=$(PSD) --psd-file L1=$(PSD) --psd-file V1=$(PSD) \
  --channel-name H1=FAKE-STRAIN --channel-name L1=FAKE-STRAIN --channel-name V1=FAKE-STRAIN \
  --inclination-cosine-sampler --declination-cosine-sampler --data-start-time 1000000008 \
  --data-end-time 1000000016 --inv-spec-trunc-time 0 \
  --srate 4096 --sampler-method AV --vectorized --gpu $(DAG_DMARG_FLAG) \
  --calibration-envelope-directory $(CURDIR)/cal_env --calibration-n-realizations $(NCAL_DAG) \
  --calibration-spline-count 10 $(DAG_FUSED_FLAG) $(DAG_BURNIN_FLAG)
# --sigma-cut 5: cal draws (PRIOR for vanilla every iteration; PRIOR at iteration 0 for the
# pilot cold start) have high per-point MC error (~0.7-0.9 here), which the CIP default
# --sigma-cut 0.6 would strip entirely.  Relax it (cf. the helper_LDG_Events.py fix that
# relaxes the first CIP stage automatically when --calmarg-pilot is enabled).
DAG_CIP_ARGS  := --mc-range [23,35] --eta-range [0.20,0.24999] --parameter mc --parameter-implied eta --parameter-nofit delta_mc --fit-method gp --lnL-offset 120 --sigma-cut 5 --cap-points 12000 --n-output-samples 5000 --no-plots --n-eff 5000
DAG_TEST_ARGS := --always-succeed --method lame --parameter m1
DAG_PLOT_ARGS := --parameter m1 --parameter m2
DAG_FILE      := marginalize_intrinsic_parameters_BasicIterationWorkflow.dag

.PHONY: dag dag-build dag-validate dag-run tune-single tune-condor

# Single-event ILE at the injection as ONE condor job (a full DAG is overkill for tuning):
# persistent logs, watchdog-isolated (condor, not an interactive launch), restartable, and
# UNBUFFERED via `python -u` so n_eff creep is watchable live.  Submit from inside the env
# (getenv captures cupy):  conda run -n rift_gpu2 make tune-condor FUSED=1 NMAX_DAG=4000000
#   watch:  tail -f rundir_tune/tune_condor.out   ;   cat rundir_tune/tune_out*.dat
tune-condor: cal_env/H1.txt $(DAG_DMARG_DEP)
	rm -rf $(CURDIR)/rundir_tune; mkdir -p $(CURDIR)/rundir_tune
	@PYEXE=$$(command -v python); \
	{ echo "universe = vanilla"; \
	  echo "executable = $$PYEXE"; \
	  echo "arguments = -u $(ILE) $(DAG_ILE_ARGS) --sim-xml $(SIM_XML) --n-events-to-analyze 1 --output-file tune_out"; \
	  echo "request_GPUs = 1"; \
	  echo "getenv = True"; \
	  echo "initialdir = $(CURDIR)/rundir_tune"; \
	  echo "output = tune_condor.out"; echo "error = tune_condor.err"; echo "log = tune_condor.log"; \
	  echo "queue"; } > $(CURDIR)/rundir_tune/tune_condor.sub
	cd $(CURDIR)/rundir_tune && $(ENV) condor_submit tune_condor.sub
	@echo "Submitted.  Watch live:  tail -f $(CURDIR)/rundir_tune/tune_condor.out"
	@echo "Result when done:        cat $(CURDIR)/rundir_tune/tune_out*.dat   (lnL col -4, n_eff col -1)"

# Single-event ILE at the INJECTION (command-single style) using the SAME wide-ILE args
# the DAG will use -- the fast way to tune settings (n_eff!) for the event before paying
# for a full multi-iteration DAG.  Reports lnL (col -4) and n_eff (col -1).
#   make tune-single FUSED=1 DMARG_DAG=1 NMAX_DAG=200000 NEFF_DAG=200
tune-single: cal_env/H1.txt $(DAG_DMARG_DEP)
	rm -rf $(CURDIR)/rundir_tune; mkdir -p $(CURDIR)/rundir_tune
	cd $(CURDIR)/rundir_tune && $(ENV) $(ILE) $(DAG_ILE_ARGS) \
	  --sim-xml $(SIM_XML) --n-events-to-analyze 1 --output-file tune_out
	@echo "=== tune-single result (lnL +- err, n_eff) ==="
	@cat $(CURDIR)/rundir_tune/tune_out*.dat
	@echo "  (n_eff is the last column; aim for a healthy value before running the DAG)"

# Build the DAG (no condor submission) and validate the wiring.  PILOT/FUSED toggles above.
dag-build: cal_env/H1.txt $(DAG_DMARG_DEP)
	rm -rf $(RUN_DAG); mkdir -p $(RUN_DAG)
	cd $(RUN_DAG) && $(ENV) util_ManualOverlapGrid.py --parameter mc --parameter-range '[23,35]' \
	  --parameter delta_mc --parameter-range '[0.0,0.5]' --grid-cartesian-npts $(NPTS_GRID_DAG) \
	  --skip-overlap --approx SEOBNRv4
	cd $(RUN_DAG) && cp overlap-grid.xml.gz overlap-grid-0.xml.gz
	cd $(RUN_DAG) && printf 'X %s\n' "$(DAG_CIP_ARGS)"  > args_cip.txt
	cd $(RUN_DAG) && printf 'X %s\n' "$(DAG_TEST_ARGS)" > args_test.txt
	cd $(RUN_DAG) && printf 'X %s\n' "$(DAG_PLOT_ARGS)" > args_plot.txt
	# args_ile.txt: the wide ILE config; for PILOT=1 also the per-iteration seed breadcrumb
	# (the condor macro $(macroiterationprev) is emitted literally: $$ -> $ for Make, then
	# single-quoted for the shell -- exactly as util_RIFT_pseudo_pipe.py injects it).
	cd $(RUN_DAG) && { printf 'X %s' "$(DAG_ILE_ARGS)"; \
	  [ "$(PILOT)" = "1" ] && printf ' --calibration-proposal-breadcrumb %s/cal_consolidated_$$(macroiterationprev).npz' "$(RUN_DAG)"; \
	  printf '\n'; } > args_ile.txt
	cd $(RUN_DAG) && $(ENV) create_event_parameter_pipeline_BasicIteration --request-gpu-ILE \
	  --ile-n-events-to-analyze $(NPTS_IT_DAG) --input-grid overlap-grid-0.xml.gz \
	  --ile-exe $(ILE) \
	  --ile-args $(RUN_DAG)/args_ile.txt --cip-args $(RUN_DAG)/args_cip.txt \
	  --test-args $(RUN_DAG)/args_test.txt --plot-args $(RUN_DAG)/args_plot.txt \
	  --request-memory-CIP 2048 --request-memory-ILE 4096 --n-copies $(N_COPIES_DAG) \
	  --input-grid $(RUN_DAG)/overlap-grid.xml.gz --n-samples-per-job $(NPTS_IT_DAG) \
	  --working-directory $(RUN_DAG) --n-iterations $(N_IT_DAG) \
	  $(DAG_PILOT_CEPP)
	$(MAKE) dag-validate PILOT=$(PILOT) FUSED=$(FUSED)

# Validate the produced DAG (no condor needed).  Branch on PILOT.
dag-validate:
	@grep -q -- "--calibration-envelope-directory" "$(RUN_DAG)/ILE.sub" || (echo "FAIL: wide ILE.sub missing calibration envelope" && false)
ifeq ($(FUSED),1)
	@grep -q -- "--calibration-fused-kernel" "$(RUN_DAG)/ILE.sub" || (echo "FAIL: FUSED=1 but ILE.sub missing --calibration-fused-kernel" && false)
endif
ifeq ($(PILOT),1)
	@test -s "$(RUN_DAG)/CALPILOT.sub"                          || (echo "FAIL: no CALPILOT.sub" && false)
	@grep -q "util_CalPilotStage.py" "$(RUN_DAG)/CALPILOT.sub"  || (echo "FAIL: CALPILOT.sub does not run util_CalPilotStage.py" && false)
	@grep -q "CALPILOT" $(RUN_DAG)/$(DAG_FILE)                  || (echo "FAIL: no CALPILOT job in DAG" && false)
	@grep -q -- "--calibration-proposal-breadcrumb" "$(RUN_DAG)/ILE.sub" || (echo "FAIL: wide ILE.sub missing seed breadcrumb" && false)
	@grep -q "macroiterationprev" "$(RUN_DAG)/ILE.sub"          || (echo "FAIL: ILE.sub missing macroiterationprev" && false)
	@echo "OK: cal-PILOT DAG built -- CALPILOT runs util_CalPilotStage.py; wide ILE seeded via breadcrumb."
else
	@! test -s "$(RUN_DAG)/CALPILOT.sub" || (echo "FAIL: PILOT=0 but a CALPILOT.sub was produced" && false)
	@! grep -q -- "--calibration-proposal-breadcrumb" "$(RUN_DAG)/ILE.sub" || (echo "FAIL: PILOT=0 but ILE.sub has a seed breadcrumb" && false)
	@echo "OK: VANILLA calmarg DAG built (no pilots; FUSED=$(FUSED)) -- ILE.sub carries the cal envelope, no CALPILOT."
endif
	@echo "    Submit with:  make dag-run PILOT=$(PILOT) FUSED=$(FUSED)"

# Submit the DAG to the local condor (GPU).  Select the card with CUDA_VISIBLE_DEVICES.
dag-run:
	cd $(RUN_DAG) && $(ENV) condor_submit_dag -f $(DAG_FILE)
	@echo "Submitted.  Watch with: condor_q ; tail -f $(RUN_DAG)/$(DAG_FILE).dagman.out"
	@echo "After it finishes, the pilot products are $(RUN_DAG)/cal_consolidated_*.npz and"
	@echo "$(RUN_DAG)/cal_pilot_resp_*.npz ; the seeded wide-ILE logs report 'SEEDED from proposal breadcrumb'."

dag: dag-build

clean:
	rm -f distance_marg.npz out_*.dat *.xml.gz lowsnr.cache snr_*.dat snr-report.txt
	rm -rf cal_env lowsnr_mdc rundir_dag
