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

# --- Top-level pipeline thread-through (util_RIFT_pseudo_pipe.py) -------------------
# Unlike dag-build (which calls create_event_parameter_pipeline_BasicIteration directly
# with a hand-written args_ile.txt), this exercises the FULL top-level builder:
# util_RIFT_pseudo_pipe.py -> helper_LDG_Events.py -> args generation -> create_event_*.
# Goal: confirm the calmarg flags AND the time-sample (time-resampling) options thread all
# the way into args_ile.txt / the .sub files / the DAG.  Offline build-validate (the
# established pattern: .travis/test-build.sh + demo/pipeline/zero_spin_phenomD), so no GPU
# run needed.  Zero spin (reuses the zero-spin IMRPhenomD ini); iterations forced small.
PP_RUN   := $(CURDIR)/rundir_pp
PP_INI   ?= $(RIFT_CODE_ROOT)/demo/pipeline/zero_spin_phenomD/zero_spin_phenomD.ini
PP_COINC ?= $(REPO_ROOT)/.travis/ref_ini/coinc.xml
PP_NIT   ?= 2          # forced iteration count (small, under control)
PP_SRATE ?= 4096       # time-resampling output rate (--srate-resample-time-marginalization)
# PP_PILOT=1 adds the adaptive cal PILOT jobs to pp-run (harvest->dump->fit->consolidate->
# seed wide_{N+1}).  Default 0 (vanilla).  `make pp-run-pilot` is the convenience wrapper.
PP_PILOT ?= 0
ifeq ($(PP_PILOT),1)
  PP_PILOT_FLAGS := --calmarg-pilot --calmarg-pilot-cadence 1 --calmarg-pilot-max-it 1
else
  PP_PILOT_FLAGS :=
endif

.PHONY: pp pp-build pp-validate

pp-build: cal_env/H1.txt
	@test -s "$(PP_INI)"   || (echo "missing zero-spin ini $(PP_INI)" && false)
	@test -s "$(PP_COINC)" || (echo "missing coinc $(PP_COINC)" && false)
	rm -rf $(PP_RUN); touch $(CURDIR)/foo.cache
	$(ENV) util_RIFT_pseudo_pipe.py \
	  --use-ini $(PP_INI) --use-coinc $(PP_COINC) --use-rundir $(PP_RUN) \
	  --fake-data-cache $(CURDIR)/foo.cache \
	  --assume-nospin --approx IMRPhenomD --ile-sampler-method AV \
	  --add-extrinsic --add-extrinsic-time-resampling --internal-ile-srate-time-resampling $(PP_SRATE) \
	  --calmarg-envelope-directory $(CURDIR)/cal_env --calmarg-n-realizations $(NCAL_DAG) --calmarg-fused-kernel \
	  --internal-force-iterations $(PP_NIT) --ile-n-eff 30
	$(MAKE) pp-validate

# Confirm everything threaded through into the generated pipeline (no condor needed).
pp-validate:
	@test -s "$(PP_RUN)/args_ile.txt" || (echo "FAIL: pseudo_pipe produced no args_ile.txt" && false)
	@echo "--- args_ile.txt ---"; cat $(PP_RUN)/args_ile.txt
	@grep -q -- "--calibration-envelope-directory" $(PP_RUN)/args_ile.txt || (echo "FAIL: calmarg envelope not threaded to args_ile.txt" && false)
	@grep -q -- "--calibration-n-realizations"     $(PP_RUN)/args_ile.txt || (echo "FAIL: calmarg n-realizations not threaded" && false)
	@grep -q -- "--calibration-fused-kernel"       $(PP_RUN)/args_ile.txt || (echo "FAIL: calmarg fused-kernel not threaded" && false)
	@grep -q -- "--time-marginalization"           $(PP_RUN)/args_ile.txt || (echo "FAIL: --time-marginalization absent (time integral)" && false)
	@grep -q -- "--srate-resample-time-marginalization" $(PP_RUN)/args_ile.txt || (echo "FAIL: TIME SAMPLES (--srate-resample-time-marginalization) not threaded to ILE" && false)
	@ls $(PP_RUN)/*.dag >/dev/null 2>&1 || (echo "FAIL: no top-level DAG produced" && false)
	@test -s "$(PP_RUN)/ILE_extr.sub" || (echo "FAIL: no extrinsic (time-sample) stage ILE_extr.sub" && false)
	@echo "--- ILE_extr.sub (the extrinsic / time-sample output stage) ---"
	@grep -oE -- "--calibration-fused-kernel|--calibration-envelope-directory|--srate-resample-time-marginalization [0-9]+|--time-marginalization" $(PP_RUN)/ILE_extr.sub | sort -u
	@grep -q -- "--calibration-fused-kernel"            $(PP_RUN)/ILE_extr.sub || (echo "FAIL: extrinsic stage missing calmarg" && false)
	@grep -q -- "--srate-resample-time-marginalization" $(PP_RUN)/ILE_extr.sub || (echo "FAIL: extrinsic stage missing TIME SAMPLES resampling" && false)
	@echo "OK: top-level util_RIFT_pseudo_pipe.py threaded EVERYTHING --"
	@echo "    calmarg (envelope + n-realizations + fused-kernel) into args_ile.txt, ILE.sub AND ILE_extr.sub;"
	@echo "    TIME SAMPLES (--srate-resample-time-marginalization) into the wide AND extrinsic (ILE_extr) stages"
	@echo "    [--add-extrinsic-time-resampling drives the extrinsic stage at build time]; zero-spin IMRPhenomD;"
	@echo "    $(PP_NIT) iterations; full DAG produced."
	@echo "    (Offline build-validate -- the established pattern; no GPU/data run needed for the threading check.)"

pp: pp-build

# --- EXTRINSIC HANDOFF offline build-validate (separate rundir; never touches pp/run dirs) ---
# Builds a pipeline with the extrinsic handoff (GMM sampler) and confirms the per-event
# proposal output, the seed breadcrumb, and the EXTRCONSOLIDATE consolidation node all thread
# through.  Offline -- no GPU/condor/data run needed for the threading check.
# See RIFT/calmarg/DESIGN_extrinsic_handoff.md.
PP_EXTR_RUN := $(CURDIR)/rundir_pp_extr

# OPTIONAL distance marginalization for the pseudo_pipe calmarg path.  The fused calmarg kernel
# does NOT require distmarg -- it has both a non-distmarg kernel (Q_fused_calmarg_cupy) and a
# distmarg kernel (Q_fused_calmarg_distmarg_cupy), and the ILE binary wires whichever applies.
# PP_DMARG=1 adds --internal-marginalize-distance, so the helper builds the lookup table
# (util_InitMargTable) and distance is marginalized ANALYTICALLY -- it leaves the extrinsic
# sampler entirely.  RECOMMENDED with --extrinsic-handoff: it removes the distance dimension
# (and its hard [1,PP_DMAX] bound) from the GMM proposal, so a seeded distance Gaussian can't
# spill past the bound into NaN likelihood.  (Distinct from the direct-ILE dag-build DMARG knob,
# which uses a pre-built --distance-marginalization-lookup-table.)
PP_DMARG    ?= 0
PP_DMAX     ?= 1000
ifeq ($(PP_DMARG),1)
  PP_DMARG_FLAGS := --internal-marginalize-distance --internal-distance-max $(PP_DMAX)
else
  PP_DMARG_FLAGS :=
endif

.PHONY: extr extr-build extr-validate

extr-build: cal_env/H1.txt
	@test -s "$(PP_INI)"   || (echo "missing zero-spin ini $(PP_INI)" && false)
	@test -s "$(PP_COINC)" || (echo "missing coinc $(PP_COINC)" && false)
	rm -rf $(PP_EXTR_RUN); touch $(CURDIR)/foo.cache
	$(ENV) util_RIFT_pseudo_pipe.py \
	  --use-ini $(PP_INI) --use-coinc $(PP_COINC) --use-rundir $(PP_EXTR_RUN) \
	  --fake-data-cache $(CURDIR)/foo.cache \
	  --assume-nospin --approx IMRPhenomD --ile-sampler-method GMM \
	  --add-extrinsic --add-extrinsic-time-resampling --internal-ile-srate-time-resampling $(PP_SRATE) \
	  --calmarg-envelope-directory $(CURDIR)/cal_env --calmarg-n-realizations $(NCAL_DAG) --calmarg-fused-kernel \
	  --extrinsic-handoff $(PP_DMARG_FLAGS) \
	  --internal-force-iterations $(PP_NIT) --ile-n-eff 30
	$(MAKE) extr-validate PP_DMARG=$(PP_DMARG)

extr-validate:
	@test -s "$(PP_EXTR_RUN)/args_ile.txt" || (echo "FAIL: pseudo_pipe produced no args_ile.txt" && false)
	@echo "--- args_ile.txt (extrinsic handoff flags) ---"
	@grep -oE -- "--sampler-method [A-Za-z_]+|--extrinsic-proposal-output [^ ]+|--extrinsic-proposal-breadcrumb [^ ]+" $(PP_EXTR_RUN)/args_ile.txt | sort -u
	@grep -q -- "--sampler-method GMM"             $(PP_EXTR_RUN)/args_ile.txt || (echo "FAIL: extrinsic handoff needs the GMM sampler (--sampler-method GMM absent)" && false)
	@grep -q -- "--extrinsic-proposal-output"      $(PP_EXTR_RUN)/args_ile.txt || (echo "FAIL: per-event --extrinsic-proposal-output not threaded to ILE" && false)
	@grep -q -- "--extrinsic-proposal-breadcrumb"  $(PP_EXTR_RUN)/args_ile.txt || (echo "FAIL: seed --extrinsic-proposal-breadcrumb not threaded to ILE" && false)
	@if [ "$(PP_DMARG)" = "1" ]; then \
	   grep -q -- "--distance-marginalization" $(PP_EXTR_RUN)/args_ile.txt || (echo "FAIL: PP_DMARG=1 but --distance-marginalization not threaded" && false); \
	   grep -q -- "--distance-marginalization-lookup-table" $(PP_EXTR_RUN)/args_ile.txt || (echo "FAIL: PP_DMARG=1 but no distance-marginalization lookup table" && false); \
	   echo "  [PP_DMARG=1] distance marginalization ON: distance is marginalized analytically and removed from the GMM sampler"; \
	 else echo "  [PP_DMARG=0] distance SAMPLED (fused kernel without distmarg -- valid; pass PP_DMARG=1 to marginalize)"; fi
	@test -s "$(PP_EXTR_RUN)/EXTRCONSOLIDATE.sub" || (echo "FAIL: no EXTRCONSOLIDATE.sub consolidation stage" && false)
	@echo "--- EXTRCONSOLIDATE.sub ---"
	@grep -E "universe|executable|arguments|initialdir" $(PP_EXTR_RUN)/EXTRCONSOLIDATE.sub | head
	@grep -q "util_ExtrinsicConsolidate.py" $(PP_EXTR_RUN)/EXTRCONSOLIDATE.sub || (echo "FAIL: EXTRCONSOLIDATE.sub does not run util_ExtrinsicConsolidate.py" && false)
	@grep -q "universe = local"             $(PP_EXTR_RUN)/EXTRCONSOLIDATE.sub || (echo "FAIL: EXTRCONSOLIDATE should be a local-universe job" && false)
	@DAG=$$(ls $(PP_EXTR_RUN)/*BasicIterationWorkflow.dag 2>/dev/null | head -1); \
	  test -n "$$DAG" || (echo "FAIL: no per-iteration DAG produced" && false); \
	  grep -q "util_ExtrinsicConsolidate_py" $$DAG || (echo "FAIL: no EXTRCONSOLIDATE node in the DAG" && false); \
	  EC=$$(grep -oE "util_ExtrinsicConsolidate_py-[0-9a-f]+" $$DAG | head -1); \
	  grep -qE "PARENT unify_sh-[0-9a-f]+ CHILD $$EC" $$DAG || (echo "FAIL: EXTRCONSOLIDATE not gated behind the iteration's unify (ILE barrier)" && false); \
	  grep -qE "PARENT $$EC CHILD integrate_likelihood_extrinsic_batchmode" $$DAG || (echo "FAIL: next-iteration ILE jobs do not depend on EXTRCONSOLIDATE (seed barrier missing)" && false)
	@echo "OK: extrinsic handoff threaded EVERYTHING --"
	@echo "    --sampler-method GMM + per-event --extrinsic-proposal-output + seed --extrinsic-proposal-breadcrumb in args_ile.txt;"
	@echo "    EXTRCONSOLIDATE.sub (local universe, util_ExtrinsicConsolidate.py);"
	@echo "    DAG wiring: unify_{it} -> EXTRCONSOLIDATE_{it} -> wide ILE_{it+1} (consolidate barrier + seed barrier)."
	@echo "    (Offline build-validate; NOTE the ILE binary change is execute-point -- rebuild the container before an OSG/CIT run.)"

extr: extr-build

# --- RUNNABLE top-level pipeline on the CI fake data (breadcrumb for later) ----------
# Unlike pp-build (offline threading check), this builds a pipeline that ACTUALLY RUNS on
# the zero-noise CI data and SUBMITS it to condor: real coinc (util_SimInspiralToCoinc.py
# from the injection), the real zero_noise.cache + CI PSD + FAKE-STRAIN channels (a
# CI-matched ini), calmarg + time-resampling, zero spin, small forced iterations.  Produces
# extrinsic-stage TIME SAMPLES with calmarg on.  Single GPU on cardassia (CUDA_VISIBLE_DEVICES=0).
PP_RUN_REAL := $(CURDIR)/rundir_pp_run
PP_DAG      := marginalize_intrinsic_parameters_BasicIterationWorkflow.dag

# --- CIT / OSG container mode (auto-activates when SINGULARITY_RIFT_IMAGE is set) -----
# CIT is container-only: you must `export SINGULARITY_RIFT_IMAGE=/path/to/rift.sif` and run
# on OSG.  Exporting that env var (or passing OSG=1) auto-populates the OSG flags on pp-run:
#   --use-osg            -> the builder adds --use-singularity (reads SINGULARITY_RIFT_IMAGE)
#   --use-osg-cip        -> CIP jobs run in-container on OSG too
#   --use-osg-file-transfer + a frames_dir -> transfer the CI frames (no CVMFS for fake data)
#   --ile-additional-files-to-transfer -> ride the cal envelopes + PSD to the workers (no
#                                          shared filesystem on OSG)
# On a local shared-FS run (cardassia) leave SINGULARITY_RIFT_IMAGE unset / OSG=0 -> no-op.
#   CIT:  export SINGULARITY_RIFT_IMAGE=/cvmfs/.../rift.sif ; make pp-run
ifneq ($(SINGULARITY_RIFT_IMAGE),)
  OSG ?= 1
endif
OSG ?= 0
ifeq ($(OSG),1)
  # Container: ~4 GB disk for the .sif transfer + unpack (vs the MB-scale shared-FS baseline)
  PP_DISK ?= 4G
  PP_OSG_FLAGS := --use-osg --use-osg-cip --use-osg-file-transfer
  # On OSG the cal envelopes are auto-added to the transfer list by util_RIFT_pseudo_pipe.py
  # (referenced as '.'); the PSD is provided as run-dir <IFO>-psd.xml.gz (copied below), which
  # RIFT lists + transfers by basename.  So do NOT pass --use-online-psd-file (it would force
  # an unreachable absolute path into the args).
  PP_PSD_FLAG :=
else
  PP_DISK ?= 4M
  PP_OSG_FLAGS :=
  # Local shared FS: reference the CI PSD by absolute path (no transfer needed).
  PP_PSD_FLAG := --use-online-psd-file $(PSD)
endif
# Disk requests for ALL THREE job types come from the CLI here, ALWAYS (baseline 4M local,
# 4G container).  Must NOT also be set in the ini -- a [rift-pseudo-pipe] value there
# OVERRIDES the CLI, which previously left ILE stuck at 4M while cip/general got bumped.
PP_DISK_FLAGS := --internal-ile-request-disk $(PP_DISK) --internal-cip-request-disk $(PP_DISK) --internal-general-request-disk $(PP_DISK)

.PHONY: pp-run pp-run-build pp-coinc

pp-coinc: ci_coinc.xml
ci_coinc.xml:
	$(ENV) util_SimInspiralToCoinc.py --sim-xml $(SIM_XML) --event 0 \
	  --ifo H1 --ifo L1 --ifo V1 --output $(CURDIR)/ci_coinc.xml --injected-snr 17.5

# Build the runnable pipeline (no submit) and validate the runnable bits threaded through.
# OSG=1 (or SINGULARITY_RIFT_IMAGE set) layers on the container/OSG flags for CIT.
pp-run-build: cal_env/H1.txt ci_coinc.xml
ifeq ($(OSG),1)
	@test -n "$(SINGULARITY_RIFT_IMAGE)" || (echo "OSG=1 but SINGULARITY_RIFT_IMAGE is not set (CIT is container-only: export it first)" && false)
	@echo "  [OSG] container mode: SINGULARITY_RIFT_IMAGE=$(SINGULARITY_RIFT_IMAGE)"
endif
	rm -rf $(PP_RUN_REAL)
	$(ENV) util_RIFT_pseudo_pipe.py \
	  --use-ini $(CURDIR)/calmarg_ci.ini --use-coinc $(CURDIR)/ci_coinc.xml --use-rundir $(PP_RUN_REAL) \
	  --fake-data-cache $(CACHE) $(PP_PSD_FLAG) \
	  --assume-nospin --approx IMRPhenomD --ile-sampler-method AV \
	  --add-extrinsic --add-extrinsic-time-resampling --internal-ile-srate-time-resampling 4096 \
	  --calmarg-envelope-directory $(CURDIR)/cal_env --calmarg-n-realizations $(NCAL_DAG) --calmarg-fused-kernel \
	  $(PP_DMARG_FLAGS) \
	  --internal-force-iterations $(PP_NIT) --ile-n-eff 30 \
	  $(PP_DISK_FLAGS) $(PP_OSG_FLAGS) $(PP_PILOT_FLAGS)
ifeq ($(OSG),1)
	@# --use-osg-file-transfer expects a frames_dir of .gwf in the rundir.  Must be a FULL
	@# COPY, not a symlink: condor refuses "Transfer of symlinks to directories is not supported".
	rm -rf $(PP_RUN_REAL)/frames_dir; cp -r $(CI_DEMO)/zero_noise_mdc $(PP_RUN_REAL)/frames_dir
	@# The args reference per-IFO <IFO>-psd.xml.gz (basename, transferred); populate them from
	@# the CI PSD so the files exist in the run dir AND carry the real PSD (not fiducial).
	for ifo in H1 L1 V1; do cp $(PSD) $(PP_RUN_REAL)/$$ifo-psd.xml.gz; done
	@echo "  [OSG] copied CI frames -> frames_dir (full copy), CI PSD -> {H1,L1,V1}-psd.xml.gz"
endif
ifeq ($(OSG),1)
	@grep -q "local.cache" $(PP_RUN_REAL)/args_ile.txt 2>/dev/null || echo "  [OSG] note: cache is local.cache (built on the remote worker), not zero_noise.cache"
else
	@grep -q "zero_noise.cache"   $(PP_RUN_REAL)/args_ile.txt || (echo "FAIL: not the real CI cache" && false)
endif
	@grep -q "FAKE-STRAIN"        $(PP_RUN_REAL)/args_ile.txt || (echo "FAIL: not the FAKE-STRAIN channels" && false)
	@grep -q "1000000014"         $(PP_RUN_REAL)/args_ile.txt || (echo "FAIL: not the CI event time" && false)
	@grep -q -- "--srate-resample-time-marginalization" $(PP_RUN_REAL)/ILE_extr.sub || (echo "FAIL: extrinsic stage missing time resampling" && false)
	@grep -q -- "--calibration-fused-kernel" $(PP_RUN_REAL)/ILE_extr.sub || (echo "FAIL: extrinsic stage missing calmarg" && false)
	@test -s "$(PP_RUN_REAL)/$(PP_DAG)" || (echo "FAIL: no top-level DAG produced" && false)
ifeq ($(PP_PILOT),1)
	@test -s "$(PP_RUN_REAL)/CALPILOT.sub"                         || (echo "FAIL: PP_PILOT=1 but no CALPILOT.sub" && false)
	@# the stage exe is named in CALPILOT.sub (local) OR in calpilot_pre.sh (OSG: the sub's
	@# executable is the prescript, which builds local.cache then runs util_CalPilotStage.py)
	@grep -qs "util_CalPilotStage.py" "$(PP_RUN_REAL)/CALPILOT.sub" "$(PP_RUN_REAL)/calpilot_pre.sh" || (echo "FAIL: neither CALPILOT.sub nor calpilot_pre.sh runs util_CalPilotStage.py" && false)
	@grep -q "CALPILOT" "$(PP_RUN_REAL)/$(PP_DAG)"                 || (echo "FAIL: no CALPILOT job in DAG" && false)
	@grep -q -- "--calibration-proposal-breadcrumb" "$(PP_RUN_REAL)/args_ile.txt" || (echo "FAIL: wide ILE args missing seed breadcrumb" && false)
	@echo "OK: runnable cal-PILOT pipeline built on CI data -- CALPILOT (harvest->dump->fit->"
	@echo "    consolidate) wired; wide ILE seeded via --calibration-proposal-breadcrumb; $(PP_NIT) iters."
else
	@echo "OK: runnable calmarg pipeline built on the CI fake data (real cache + PSD + FAKE-STRAIN,"
	@echo "    event 1000000014.236, calmarg+fused, time-resampling, zero-spin, $(PP_NIT) iters)."
endif

# Build + SUBMIT to condor.  Watch: condor_q ; tail -f $(PP_RUN_REAL)/$(PP_DAG).dagman.out
# Extrinsic time-sample output lands in the last iteration's extrinsic stage.
pp-run: pp-run-build
	cd $(PP_RUN_REAL) && $(ENV) condor_submit_dag -f $(PP_DAG)
	@echo "Submitted.  Watch:  condor_q ; tail -f $(PP_RUN_REAL)/$(PP_DAG).dagman.out"
	@# CIT requires accounting identity in the environment, else jobs never queue.
	@test -n "$$LIGO_USER_NAME" || echo "  *** WARNING: LIGO_USER_NAME is unset -- condor jobs may not queue (required at CIT). export it. ***"
	@{ test -n "$$LIGO_ACCOUNTING" || test -n "$$LIGO_ACCOUNTING_GROUP"; } || echo "  *** WARNING: LIGO_ACCOUNTING (a.k.a. LIGO_ACCOUNTING_GROUP) is unset -- condor jobs may not queue (required at CIT). export it. ***"

# Runnable cal-PILOT pipeline: pp-run with the adaptive pilots enabled (harvest->dump->
# fit->consolidate->seed wide_{N+1}).  Needs >=2 iterations so the seed is consumed.
# Honours OSG (CIT) the same way as pp-run.
# NOTE: uses a SEPARATE run directory (rundir_pp_pilot) so it does NOT clobber a vanilla
# pp-run already running in rundir_pp_run.  (pp-run-build starts with `rm -rf`.)
.PHONY: pp-run-pilot pp-run-pilot-build
pp-run-pilot-build:
	$(MAKE) pp-run-build PP_PILOT=1 PP_NIT=2 PP_RUN_REAL=$(CURDIR)/rundir_pp_pilot
pp-run-pilot:
	$(MAKE) pp-run PP_PILOT=1 PP_NIT=2 PP_RUN_REAL=$(CURDIR)/rundir_pp_pilot

# --- EXTRINSIC HANDOFF: small real GPU+condor smoke run on cardassia --------------------
# Confirms the seeded extrinsic handoff actually runs: GMM sampler (so gmm_dict seeding is
# live), --extrinsic-handoff on, on the CI fake data.  Deliberately TINY so it gets through
# many intrinsic points per condor job (few restarts) and reaches the seeded 2nd iteration
# fast:  300 initial intrinsic points, 200/generation, 50 intrinsic evals PER ILE job
# (IMRPhenomD), >=2 iterations so iteration-1 ILE is seeded by iteration-0's consolidation.
# Separate rundir (rundir_pp_extr_run) -- never clobbers pp-run / pilot runs.
# Single GPU on cardassia:  CUDA_VISIBLE_DEVICES=0 make extr-run
EXTR_RUN      := $(CURDIR)/rundir_pp_extr_run
EXTR_NIT      ?= 2          # need >=2 so the iteration-0 proposal seeds iteration-1
EXTR_GRID0    ?= 300        # initial intrinsic grid points
EXTR_GRIDN    ?= 200        # intrinsic points proposed per generation (CIP output)
EXTR_PER_JOB  ?= 50         # intrinsic evaluations per ILE condor job (minimize restarts)
EXTR_NCHUNK   ?= 4000       # extrinsic block size; small keeps the NVS 510 off its watchdog
EXTR_NMAX     ?= 40000      # HARD cap on extrinsic samples/point -- the helper sets 4,000,000
                            # (production); with slow calmarg convergence that is hours/point on
                            # the NVS 510.  40000 -> ~20 s/point (measured), enough to fit+hand off.
EXTR_NEFF     ?= 30         # n_eff target -- smoke just confirms seeding; jobs stop at n-max

.PHONY: extr-run extr-run-build

# Several knobs live in calmarg_ci.ini's [rift-pseudo-pipe] section, and an ini value
# OVERRIDES the CLI -- so the per-job / per-generation / n_eff sizes must be set IN the ini,
# not on the command line.  Derive a run-specific ini from calmarg_ci.ini, overriding only
# those three lines (sed), leaving the shared ini untouched (so pp-run is unaffected).
EXTR_INI := $(CURDIR)/extr_calmarg_ci.ini

extr-run-build: cal_env/H1.txt ci_coinc.xml
	rm -rf $(EXTR_RUN)
	sed -e 's/^ile-jobs-per-worker=.*/ile-jobs-per-worker=$(EXTR_PER_JOB)/' \
	    -e 's/^n-output-samples=.*/n-output-samples=$(EXTR_GRIDN)/' \
	    -e 's/^ile-n-eff=.*/ile-n-eff=$(EXTR_NEFF)/' \
	    $(CURDIR)/calmarg_ci.ini > $(EXTR_INI)
	$(ENV) util_RIFT_pseudo_pipe.py \
	  --use-ini $(EXTR_INI) --use-coinc $(CURDIR)/ci_coinc.xml --use-rundir $(EXTR_RUN) \
	  --fake-data-cache $(CACHE) --use-online-psd-file $(PSD) \
	  --assume-nospin --approx IMRPhenomD --ile-sampler-method GMM \
	  --add-extrinsic --add-extrinsic-time-resampling --internal-ile-srate-time-resampling 4096 \
	  --calmarg-envelope-directory $(CURDIR)/cal_env --calmarg-n-realizations $(NCAL_DAG) --calmarg-fused-kernel \
	  --extrinsic-handoff $(PP_DMARG_FLAGS) \
	  --force-initial-grid-size $(EXTR_GRID0) --n-output-samples-last $(EXTR_GRIDN) \
	  --manual-extra-ile-args "--n-chunk $(EXTR_NCHUNK) --n-max $(EXTR_NMAX)" \
	  --internal-force-iterations $(EXTR_NIT) \
	  --internal-ile-request-disk 4M --internal-cip-request-disk 4M --internal-general-request-disk 4M
	@# the n-max bound must land AFTER the helper's production --n-max 4000000 (argparse last-wins);
	@# the manual-extra args are appended last, so the effective n-max is the bounded value.
	@awk '{for(i=1;i<=NF;i++)if($$i=="--n-max")v=$$(i+1)} END{print "  effective --n-max =",v}' $(EXTR_RUN)/args_ile.txt
	@awk '/--n-max/{n=split($$0,a," "); for(i=1;i<=n;i++)if(a[i]=="--n-max")last=a[i+1]} END{exit (last==$(EXTR_NMAX))?0:1}' $(EXTR_RUN)/args_ile.txt \
	  || (echo "FAIL: effective --n-max is not the bounded $(EXTR_NMAX) (jobs would run to the 4,000,000 production cap)" && false)
	@# confirm the per-job size took (ini value is overridden by the derived ini to $(EXTR_PER_JOB))
	@grep -q -- "--ile-n-events-to-analyze $(EXTR_PER_JOB)" $(EXTR_RUN)/command-single.sh 2>/dev/null \
	  || echo "  NOTE: verify --ile-n-events-to-analyze=$(EXTR_PER_JOB) in the builder invocation above"
	@# confirm the handoff threaded through into the runnable pipeline
	@grep -q -- "--sampler-method GMM"            $(EXTR_RUN)/args_ile.txt || (echo "FAIL: GMM sampler not set" && false)
	@grep -q -- "--extrinsic-proposal-output"     $(EXTR_RUN)/args_ile.txt || (echo "FAIL: per-event extrinsic output not threaded" && false)
	@grep -q -- "--extrinsic-proposal-breadcrumb" $(EXTR_RUN)/args_ile.txt || (echo "FAIL: extrinsic seed breadcrumb not threaded" && false)
	@grep -q -- "--n-chunk $(EXTR_NCHUNK)"        $(EXTR_RUN)/args_ile.txt || (echo "FAIL: n-chunk not threaded (NVS 510 watchdog guard)" && false)
	@test -s "$(EXTR_RUN)/EXTRCONSOLIDATE.sub"    || (echo "FAIL: no EXTRCONSOLIDATE.sub" && false)
	@test -s "$(EXTR_RUN)/$(PP_DAG)"              || (echo "FAIL: no top-level DAG" && false)
	@echo "OK: tiny GMM extrinsic-handoff pipeline built on CI data --"
	@echo "    $(EXTR_GRID0) initial / $(EXTR_GRIDN) per-gen intrinsic, $(EXTR_PER_JOB) evals/ILE job, n-chunk $(EXTR_NCHUNK),"
	@echo "    $(EXTR_NIT) iterations (iter-1 ILE seeded by iter-0's EXTRCONSOLIDATE).  Submit with: make extr-run"

# Build + SUBMIT.  After it runs, the iteration-1 ILE *.out logs should print
#   ' Extrinsic GMM SEEDED from .../extr_consolidated_0.npz for dim-groups [...]'
# and extr_consolidated_0.npz / extr_proposal_0_*.npz should appear in the rundir.
extr-run: extr-run-build
	cd $(EXTR_RUN) && $(ENV) condor_submit_dag -f $(PP_DAG)
	@echo "Submitted.  Watch:  condor_q ; tail -f $(EXTR_RUN)/$(PP_DAG).dagman.out"
	@echo "Confirm seeding:  grep -l 'Extrinsic GMM SEEDED' $(EXTR_RUN)/iteration_1_ile/logs/*.out"
	@test -n "$$LIGO_USER_NAME" || echo "  *** WARNING: LIGO_USER_NAME unset (jobs may not queue at CIT). ***"

clean:
	rm -f distance_marg.npz out_*.dat *.xml.gz lowsnr.cache snr_*.dat snr-report.txt foo.cache ci_coinc.xml extr_calmarg_ci.ini
	rm -rf cal_env lowsnr_mdc rundir_dag rundir_tune rundir_pp rundir_pp_run rundir_pp_pilot rundir_pp_extr rundir_pp_extr_run
