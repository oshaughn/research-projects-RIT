# RIFT Container Pre-Compilation Assessment

## Summary

Pre-building parts of the RIFT container is feasible, but there are two distinct
classes of startup cost:

1. **Install/build cost**: dependency resolution, source checkout, editable
   install, Python bytecode generation, and optional packages. This is fully
   image-build-time work and should be moved out of job startup wherever possible.
2. **Runtime compiler cost**: CuPy `RawKernel`/`ElementwiseKernel` compilation and
   JAX/XLA compilation. This can be reduced with persistent caches and warmup
   scripts, but cache hits depend on CUDA version, GPU architecture, driver/JAX
   versions, backend, function shape, dtype, and selected likelihood mode.

The most practical near-term path is to add a survey-driven container warmup
phase that materializes persistent CuPy and JAX caches under a known in-image or
job-local cache path, plus runtime environment defaults that cap JAX/XLA thread
creation. For production on heterogeneous OSG/LDG GPUs, this should be paired
with the existing container-family mechanism so each image targets a bounded
CUDA/GPU capability range. A concrete design for that survey/warmup layer is in
`containers/SURVEY_SCAN_PROPOSAL.md`.

## Current State

The current Apptainer template (`containers/rift_container.def.in`) clones RIFT,
runs `pip3 install -e .`, installs the selected `cupy-cuda11x`/`cupy-cuda12x`
wheel, then installs the shared requirements. The top-level `rift_container.def`
does the same pattern inline. This means jobs receive source and dependencies,
but no application-specific GPU/JAX compilation has been warmed.

Core CuPy compilation sites:

- `RIFT.likelihood.Q_inner_product.Q_inner_product_cupy`: reads
  `cuda_Q_inner_product.cu` and constructs `cupy.RawKernel`.
- `RIFT.likelihood.Q_fused_calmarg`: lazily constructs `RawKernel` objects for
  `cuda_Q_fused_calmarg.cu` and `cuda_Q_fused_calmarg_distmarg.cu`.
- `RIFT.interpolators.interp_gpu`: memoizes a CuPy `ElementwiseKernel`.

Core JAX compilation sites:

- `RIFT.likelihood.jax_ile.wrapper` constructs jitted likelihood closures per
  likelihood object.
- `RIFT.likelihood.jax_ile.samplers._warmup_compile` already forces a small
  first-call compile and reports the latency, because some modes can spend
  tens of seconds in XLA compile.
- Additional `jax.jit`, `jax.grad`, and `jax.hessian` calls appear in polishing,
  Fisher, NUTS, and flowMC paths.

## Feasibility

### CuPy

CuPy cache warming is feasible and likely worth doing. The current kernels are
small, have stable source, and are concentrated in a few modules. A warmup script
can import CuPy, allocate representative arrays, call each kernel once, and
leave artifacts in `CUPY_CACHE_DIR`.

Important constraints:

- CuPy compiles for the active CUDA toolkit/driver/GPU architecture. A cache
  baked on one architecture may not serve another.
- Building a container usually has no GPU unless the builder is a GPU node and
  Apptainer is run with NVIDIA support. Without a GPU, build-time warmup cannot
  compile device-specific SASS.
- If the image must run across old and modern GPUs, the existing container-family
  split is the right place to bound compatibility and avoid one cache trying to
  serve every target.

Recommended first implementation:

- Add `RIFT/likelihood/warmup_gpu_kernels.py` or a `bin/rift_warmup_gpu_kernels`
  script.
- Set `CUPY_CACHE_DIR` to a stable path, for example
  `/opt/rift-cache/cupy` in the image or `${_CONDOR_SCRATCH_DIR}/.cupy/kernel_cache`
  at runtime.
- During container build, run the script only when a GPU is available; otherwise
  install the script and rely on a one-time prolog/warmup job on each GPU class.
- Stop using `CUPY_CACHE_IN_MEMORY=1` as the only default for production caching;
  it avoids disk writes but also prevents reuse across processes.

### JAX/XLA

JAX pre-compilation is feasible only as persistent-cache warming, not as a
single universal binary baked into the source install. The likelihood closes
over event data and compiles by argument shape and static branch choices:

- number of detectors;
- number of modes;
- `npts` / time grid length;
- distance grid size;
- phi/psi marginalization grid sizes;
- interpolation and phase-marginalization choices;
- sampler mode and use of value/grad/Hessian.

This means a generic warmup can populate common shapes, but real events with
different shapes may still compile. The most valuable cache targets are the
standard O4 production settings and the expensive modes already using
`_warmup_compile`.

Recommended first implementation:

- Add a JAX warmup command that builds a synthetic `JAXLikelihoodData` matching
  standard production shapes and instantiates the production wrappers.
- Enable JAX persistent compilation cache via environment variables before JAX
  import. For modern JAX this can be done with
  `JAX_COMPILATION_CACHE_DIR=/opt/rift-cache/jax` or the corresponding
  `jax.config` calls in the warmup command.
- Warm the common wrapper modes: fixed-distance, distance-marginalized,
  phi-marginalized, phi+psi-marginalized, and the value/grad/Hessian paths that
  samplers invoke.
- Keep runtime fallback behavior: if a shape misses the cache, it should compile
  once and continue.

Threading should be addressed independently. Container defaults should set a
conservative CPU-thread policy for JAX/XLA jobs, especially in Condor slots:

```sh
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

These should be opt-out or mode-aware if CPU-only JAX performance is important.

## Required Modifications

1. **Container recipes**
   - Add cache directories such as `/opt/rift-cache/cupy` and
     `/opt/rift-cache/jax`.
   - Export stable cache/thread defaults in `%environment`.
   - Optionally run warmup scripts during `%post` when a GPU is visible.
   - Add JAX/numpyro/flowMC dependencies to a separate JAX-enabled image flavor;
     they should not silently enter the minimal production image.

2. **Build family**
   - Extend `containers/build_family.sh` matrix with optional feature columns:
     CUDA family, target capability band, and JAX-enabled vs non-JAX image.
   - Keep a CPU-safe fallback image.
   - Publish warmed caches per image, not shared across CUDA major versions.

3. **Warmup scripts**
   - CuPy script: allocate tiny representative arrays and invoke
     `Q_inner_product_cupy`, `Q_fused_calmarg_cupy`,
     `Q_fused_calmarg_distmarg_cupy`, and `interp_gpu.interp`.
   - JAX script: construct synthetic likelihood data with configurable
     detectors/modes/time-grid/distance-grid sizes and call each wrapper's
     warmup path.
   - Scripts should report cache directory, backend, device, CUDA/JAX/CuPy
     versions, and elapsed compile time.

4. **Runtime pipeline**
   - Add optional Condor prolog or first-node warmup job per container/GPU class.
   - Ensure writable caches are available when the image cache is read-only.
     For Apptainer on CVMFS, a job-local cache path may be more reliable than
     trying to update an in-image cache.
   - Thread through environment for `CUPY_CACHE_DIR`,
     `JAX_COMPILATION_CACHE_DIR`, and XLA thread flags.

5. **Packaging**
   - Prefer a normal wheel install for release images instead of `pip install -e .`
     when the image should be immutable. Editable source installs are convenient
     for development but do not give stronger startup guarantees.
   - Keep `.cu` files as installed package data, or move them into package data
     explicitly rather than relying only on `data_files`, so warmup scripts and
     installed modules resolve the same paths.

## Risks and Limitations

- A cache warmed on one GPU architecture may miss on another; use the container
  family to limit variation.
- Build environments frequently lack GPUs, so some warming must happen as a
  deployment/prolog step rather than in `%post`.
- JAX cache portability is version-sensitive. Pinning JAX/JAXLIB/CUDA/Python is
  more important for JAX images than for the current unpinned container canary.
- JAX event-shape variation prevents complete elimination of first-call compile.
- Baked caches increase image size and should be measured against CVMFS/OSDF
  transfer cost.

## Suggested Sequence

1. Add `survey_scan survey` to record the target cluster's GPU/driver classes.
2. Add CuPy warmup script and persistent `CUPY_CACHE_DIR`; validate on one GPU
   per dominant image band.
3. Add a JAX-enabled container-family entry with pinned JAX/JAXLIB/numpyro/flowMC.
4. Add synthetic JAX warmup for the standard O4 shape/mode set.
5. Measure cold vs warm startup for:
   - NoLoop CuPy standard path;
   - fused calmarg path;
   - `integrate_likelihood_extrinsic_jax` distance/phi-marginalized modes.
6. Promote warmup to `%post` only for builders with GPU access; otherwise use a
   site prolog or one-time cache seeding job per published image/GPU class.
