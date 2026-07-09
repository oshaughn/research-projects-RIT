# GPU rotation + freqresponse validation — breadcrumb / resume guide

## ✅ VALIDATED (2026-07-09, ldas-pcdev12, NVIDIA A100-SXM4-80GB, cc 8.0, cupy 13.6 / CUDA 11.8)
Ran directly in the RIFT container on this GPU host (`make local-gpu`), no Condor needed:
- **Path A/B rotation** GPU↔CPU parity: nearest `7.3e-12`, cubic `8.2e-12`  (< 1e-8 threshold)
- **Path D freqresponse** GPU↔CPU parity: nearest `7.3e-12`, cubic `7.3e-12`  (NEW GPU port)
- CPU baseline-vs-rotation still `3.6e-12`; freqresponse CPU V4 positive control still `+38.88` nats.

**A REAL BUG was caught that the no-cupy sandbox could not:** `TimeDelayFromEarthCenter`
(vectorized_lal_tools) defaults `xpy=xpy_default`, which is **cupy whenever cupy is importable**.
The rotation/freqresponse NoLoops keep the delay on the HOST (RA/DEC are `_h()` numpy copies), so the
default fed host arrays to `cupy.cos` and crashed on the GPU. Fixed by pinning `xpy=np` at both call
sites (`factored_likelihood_with_rotation.py`, `factored_likelihood_freqresponse.py`). Lesson: a
cupy-defaulting helper is invisible in a CPU-only sandbox; always validate on real hardware.

**Path D freqresponse GPU port + ILE wiring done this pass** (mirrors the rotation port exactly):
`DiscreteFactoredLogLikelihoodFreqResponseNoLoop` gained `xpy=np` + the fused-kernel term1 GPU branch;
ILE `--freqresponse` now accepts `--gpu` (n_cal=1, no glitch/cal marg) — guard relaxed, precompute
`cupy.asarray`'s the Q/U/V banks, GPU `likelihood_function` gained the freqresponse branch. Test:
`RIFT/likelihood/test_slowrot_freqresponse_gpu.py`.

**NOTE:** `setup_generic_env_vars.sh` no longer exports `SINGULARITY_RIFT_IMAGE`, so `make submit`
needs it set by hand (a cvmfs or local cc60-90 CUDA-11.8 image). `make local-gpu` sidesteps this by
pinning a local image (`RIFT_LOCAL_IMAGE` in the Makefile).

**Goal (original):** validate the GPU (`xpy=cupy`) rotation likelihood on real hardware. It was
implemented and CPU-verified, but the dev sandbox had **no cupy/CUDA** (`libcuda.so.1` absent), so the
GPU path was UNTESTED. This kit runs one Condor GPU job to confirm GPU↔CPU parity.

## What was done (commits on branch `rift_slowrot`)
- `d038f582` — GPU support for the rotation NoLoop (`factored_likelihood_with_rotation.py`):
  term1 reuses the baseline fused `Q_inner_product_{cubic,}_cupy` kernel **per elementary
  template** (A=conj(Ylm)) → no `(n_ext,npts,n_lms)` temporary, same GPU memory footprint as the
  baseline. term2 = small `|a_list|²` einsums. Antenna/Ylm/delay stay on host; only Q banks + U/V
  move to device. ILE (`integrate_likelihood_extrinsic_batchmode`): `--rotation-slow` now allowed
  with `--gpu` (needs n_cal=1, no glitch/cal marg); precompute `cupy.asarray`'s rho/U/V; the GPU
  `likelihood_function` (~line 2239) gained the rotation branch.
- `af68d7ec` — earlier: per-detector `--freqresponse-arm-length`, `--limit-*` zoom-box.
- CPU path is bit-identical: `test_slowrot_noloop` still 3.638e-12.
- **The validation test:** `RIFT/likelihood/test_slowrot_gpu.py` — runs the rotation NoLoop with
  `xpy=np` vs `xpy=cupy` on the same packed data (nearest + cubic), asserts `max|diff| < 1e-8`.
  Auto-SKIPS without a GPU.

## To validate on a GPU host (RESUME HERE)
```
source ~/setup_generic_env_vars.sh          # sets SINGULARITY_RIFT_IMAGE, RIFT_REQUIRE_GPUS,
                                            #      LIGO_ACCOUNTING, LIGO_USER_NAME
cd <this dir>            # .../Code/demo/rift/slowrot_gpu_validate
make submit                                 # tars the branch code + condor_submit gpu_test.sub
# ... wait for the 1 GPU job ...
make show                                   # prints gpu_test.out
```
**PASS criteria** (`gpu_test.out`): the cupy/device line prints (real GPU + CUDA), then
`(GPU) rotation NoLoop xpy=cupy vs xpy=np, interp=nearest/cubic : max|diff| = <~1e-9`, and the CPU
`ALL SLOWROT NOLOOP CHECKS PASSED`.

## How it runs
- `gpu_test.sub`: `+SingularityImage=$ENV(SINGULARITY_RIFT_IMAGE)` (cvmfs RIFT container, has
  cupy/CUDA), `request_gpus=1`, `Requirements = HAS_SINGULARITY && $ENV(RIFT_REQUIRE_GPUS)`
  (the pool's capability band, default `(Capability>=6.0)&&(<=9.0)` = CUDA-11.8 container band),
  accounting from `$ENV`. Transfers `rift_code.tar.gz` (the branch — production image predates it).
- `run_gpu_test.sh`: unpacks the tarball → `RIFT_CODE`, PYTHONPATH, prints GPU info, runs
  `test_slowrot_gpu.py` + the CPU sanity test.

## Likely gotchas / if it fails
- **cupy import fails** in the container → the image's CUDA doesn't match the matched GPU. Tighten
  `RIFT_REQUIRE_GPUS` to the container's band (see setup_generic_env_vars.sh comments re Blackwell /
  cc≤9.0 for the CUDA-11.8 image), or use a CUDA-12.8 image if matching Blackwell.
- **`optimized_gpu_tools.simps` / `Q_inner_product_cubic_cupy` errors** → those are the reused
  baseline GPU kernels; a failure there is a real bug in the GPU rotation path — check dtypes
  (`ifirst` int32, `frac` float64, Q contiguous complex128) in
  `factored_likelihood_with_rotation.py` term1 GPU branch.
- **NUMBA_CACHE_DIR**: set to `.` (job scratch) via the sub's `environment`.

## After it passes
- End-to-end GPU ILE run: same as the CPU rotation ILE but add `--gpu` (keep `--vectorized`,
  n_cal=1). Compare a short head-to-head vs the CPU run (should match to sampler noise).
- **Follow-ups not yet done:** (1) Path D (freqresponse) GPU port — identical fused-kernel pattern;
  (2) a dedicated benchmark container (see `~/rift_cit_build_container_family/` build kit in memory)
  if you want a self-contained image for end-users; (3) `create_event_parameter_pipeline`-based
  standard-pipeline example (like `.travis/ILE-GPU-Paper`).
- The finite-size sky-loc DAG (`~/RIFT_roboto_paper/analyses/slowrot_finite-size/3g/`) is a
  SEPARATE, already-launched run; `make post` there builds the figure once it finishes.
