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

## End-to-end GPU ILE — DONE (2026-07-09, commit 4c7ea1c2)
**Re-run any time with `make e2e`** (runs `run_e2e_consistency.sh` in the container on this host's GPU):
runs the real `integrate_likelihood_extrinsic_batchmode` four ways — {rotation,finite}×{cpu,gpu} — and
asserts CPU-vs-GPU marginalized lnL agree within `TOL_SIGMA` (4) × sampler error. **SELF-CONTAINED by
default**: `make_e2e_inputs.py` generates a throwaway H1/L1 IMRPhenomD BNS injection (frames/PSD/grid +
case.json, 40-km arm) — no paper-repo dependency. Set `RIFT_E2E_CASE=/path/to/case` to reuse an existing ILE
case dir instead. Args built by `e2e_mkargs.py`. Verified PASS on the A100 (self-contained: rotation
ΔlnL=0.039, freqresponse ΔlnL=0.005, both < 4σ; also on the finite-size CE-ET SNR30 inputs: ΔlnL 0.085/0.005
at n_eff~130k, GPU 2–4× faster). Three more bugs the unit tests could NOT catch (only the real GPU ILE did):
- **AV sampler `prior_prod`** fed the host CPU sample copy to mcsamplerGPU prior helpers that default
  `xpy=cupy` → `cupy.sin(numpy)` raised "Unsupported type numpy.ndarray" (broke ANY AV run in a cupy
  container). Fixed: pass `xpy=numpy` to helpers that accept it.
- **`mcsamplerGPU.cupy_pi`** was a device scalar (`cupy.array(pi)`) → `numpy_array/cupy_pi` re-dispatched to
  cupy → same error. Fixed: `cupy_pi = np.pi`.
- **freqresponse GPU port** missed the `_h()` host-copy of `P_vec` extrinsic params; under `--gpu` the ILE
  hands RA/DEC/incl as cupy arrays and `np.atleast_1d(cupy)` raised in ComputeYlmsArrayVector. Fixed.
**Invocation note:** `--force-xpy` alone is INERT (opts.gpu stays False unless `--gpu` is *also* passed and
cupy is absent, batchmode ~L520). Use `--gpu` (cupy present) to actually hit the xpy GPU likelihood_function.
Pure-CPU baseline (no `--gpu`) FAILS in a cupy container ("Unsupported dtype float128", old CPU-vectorized
path) — pre-existing, unrelated; use `--gpu` in a cupy container.

## Remaining follow-ups
- A dedicated benchmark container (see `~/rift_cit_build_container_family/` build kit in memory) for a
  self-contained end-user image; and a `create_event_parameter_pipeline`-based standard-pipeline example
  (like `.travis/ILE-GPU-Paper`).
- The finite-size sky-loc DAG (`~/RIFT_roboto_paper/analyses/slowrot_finite-size/3g/`) is a
  SEPARATE, already-launched run; `make post` there builds the figure once it finishes.
