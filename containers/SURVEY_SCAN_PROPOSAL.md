# `survey_scan` Proposal for RIFT Container Builds

## Goal

Add a `survey_scan` tool to the RIFT container build framework that surveys the
actual target cluster, classifies the small number of GPU/driver/container
combinations we care about, and runs representative warmup probes so a published
container family carries, or can seed, the most common CuPy and JAX startup
caches.

This is intentionally a "cover the common cases" tool. It does not need to find
every kernel, every RIFT executable, or every possible event shape. It should
reduce cold-start cost for the dominant NoLoop/CuPy and JAX ILE modes on the GPU
classes we actually schedule onto.

## Concrete Starting Point: CIT

The current CIT build kit on `ldas-grid-alt` already has the right shape:

- `~/rift_cit_build_container_family/build_cit_family.sh`
  builds the `cc60-90` CUDA 11.8 image and `cc90-120` CUDA 12.8 image.
- `~/rift_cit_build_container_family/build_jax_container.sh`
  builds a separate JAX GPU image.
- `built_containers/rift_container_family.cit.yaml`
  records the deployable family.
- `built_containers/rift_container_select.sh`
  selects the image at runtime on OSG-like pools when `MY.SingularityImage`
  expressions are not evaluated.
- `validate/gpu_check.py` and `validate_jax/` already prove how to submit
  real-GPU validation jobs.

A live CIT GPU census currently looks like:

```text
554  GeForce GTX 1050 Ti                         cc 6.1   4040 MB
164  NVIDIA RTX PRO 4000 Blackwell SFF Edition   cc 12.0  24027 MB
3    NVIDIA A30                                  cc 8.0   24188 MB
1    GeForce GTX 1650                            cc 7.5   3912 MB
3190 undefined/undefined/undefined
```

So for CIT, two non-JAX images remain the right default bands:

- `cc60-90`: CUDA 11.8 runtime, `cupy-cuda11x`, fallback / old-GPU image.
- `cc90-120`: CUDA 12.8 devel, `cupy-cuda12x`, Blackwell/Hopper image. The devel
  base matters because Blackwell may need NVRTC headers for first-use CuPy JIT.

The JAX GPU image can initially target `cc90-120`, where the CUDA 12.8 stack is
already validated.

## What `survey_scan` Should Do

### 1. Survey

Run on a build/login host with Condor tools available:

```sh
containers/survey_scan.sh survey --pool cit --out survey/cit-YYYYMMDD
```

Collect:

- `condor_status` grouped by GPU name, `GPUs_Capability`, memory, driver-ish
  attributes when advertised, and slot count.
- A normalized JSON summary with recommended image bands.
- The current build matrix and manifest labels, so the survey can say which
  observed classes are covered, uncovered, or only fallback-covered.

Suggested output:

```text
survey/cit-YYYYMMDD/
  gpu_inventory.tsv
  gpu_inventory.json
  recommended_matrix.json
  coverage.md
```

### 2. Generate Scan Jobs

Create Condor submit files that run a container-specific probe on one machine per
dominant GPU class:

```sh
containers/survey_scan.sh emit-jobs \
  --survey survey/cit-YYYYMMDD \
  --manifest built_containers/rift_container_family.cit.yaml \
  --out survey/cit-YYYYMMDD/jobs
```

Each job should:

- constrain to one GPU class or image band;
- run with `apptainer exec --nv`;
- set persistent cache directories;
- run a standard warmup probe;
- archive cache metadata and timing logs.

### 3. Warm CuPy Common Paths

The CuPy probe should run inside the selected image and exercise:

- `RIFT.likelihood.Q_inner_product.Q_inner_product_cupy`;
- `RIFT.likelihood.Q_fused_calmarg.Q_fused_calmarg_cupy`;
- `RIFT.likelihood.Q_fused_calmarg.Q_fused_calmarg_distmarg_cupy`;
- `RIFT.interpolators.interp_gpu.interp`.

It should use tiny, deterministic arrays with representative dtypes and shapes.
The important thing is to trigger compilation and confirm the cache path is
populated, not to benchmark production throughput.

Recommended environment:

```sh
CUPY_CACHE_DIR=/rift_cache/cupy/${image_label}/${gpu_capability}
CUPY_CACHE_IN_MEMORY=0
CUDA_PATH=/usr/local/cuda
```

For read-only published images, the runtime wrapper can instead seed or reuse a
job-local cache:

```sh
CUPY_CACHE_DIR=${_CONDOR_SCRATCH_DIR}/.rift_cache/cupy
```

### 4. Warm JAX Common Modes

The JAX probe should be separate and run only in JAX-enabled images. It should
build synthetic `JAXLikelihoodData` matching standard production shapes and
compile the expensive wrapper modes:

- fixed-distance `JAXExtrinsicLikelihood`;
- distance-marginalized `JAXDistanceMarginalizedLikelihood`;
- phi-marginalized and phi+psi-marginalized wrappers when present;
- `value_and_grad` and Hessian/Fisher paths used by samplers;
- the existing sampler `_warmup_compile` path.

Recommended environment:

```sh
JAX_COMPILATION_CACHE_DIR=/rift_cache/jax/${image_label}/${gpu_capability}
JAX_ENABLE_X64=1
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

The first version can target one standard O4 shape. Later versions can read a
small profile file, for example:

```yaml
jax_profiles:
  - name: o4_default_lmax4_hl
    detectors: [H1, L1]
    l_max: 4
    npts: 614
    distance_grid: 256
    phi_grid: 32
```

### 5. Report

After jobs finish:

```sh
containers/survey_scan.sh collect --survey survey/cit-YYYYMMDD
```

Produce:

- GPU classes observed;
- image selected for each class;
- CuPy/JAX versions;
- cold compile time;
- warm second-call time;
- cache size and file count;
- failures by GPU class;
- recommended manifest/build-matrix changes.

## Where This Should Live

Start in this repo under `containers/`, not in a separate repo.

Reasons:

- The probes need RIFT-specific imports, CLI names, and expected shapes.
- The container manifest and build-family code already live here.
- Keeping the first version local makes it easier for pipeline changes to evolve
  with the warmup profiles.

Split into a separate repository only after the tool has stable boundaries, for
example if it becomes a general IGWN GPU-container survey/warmup kit. A clean
future split would keep generic Condor/GPU inventory and cache-collection logic
outside RIFT, while RIFT keeps its own warmup profile scripts.

## Proposed File Layout

```text
containers/
  survey_scan.sh
  survey_scan/
    README.md
    gpu_inventory.py
    emit_condor_jobs.py
    collect_results.py
    profiles/
      rift_cupy_common.py
      rift_jax_ile_common.py
      o4_default.yaml
```

The Python files should be dependency-light: standard library plus optional
`PyYAML` when parsing manifests/profiles. They should not require JAX or CuPy on
the submit host; those imports happen inside the container probe.

## Integration with the Current CIT Kit

For the remote CIT kit, `survey_scan` can be used in place before or after image
builds:

1. `survey` before a build to confirm the matrix still covers the pool.
2. `emit-jobs` after images are built/staged to run one warmup job per class.
3. `collect` to decide whether caches should be baked into the next image or
   distributed as a cache tarball beside each image.

The existing `rift_container_select.sh` wrapper is a good runtime integration
point. It already detects compute capability and selects the image. It could also
set:

```sh
export RIFT_GPU_CAPABILITY="$cap"
export RIFT_CONTAINER_LABEL="${LABELS[$sel]}"
export CUPY_CACHE_DIR="${RIFT_CACHE_ROOT:-${_CONDOR_SCRATCH_DIR}/.rift_cache}/cupy/${LABELS[$sel]}"
export JAX_COMPILATION_CACHE_DIR="${RIFT_CACHE_ROOT:-${_CONDOR_SCRATCH_DIR}/.rift_cache}/jax/${LABELS[$sel]}"
```

For CVMFS deployments, if preseeded caches are published next to the SIF, the
wrapper can copy or bind the matching cache directory into the job scratch area
before running the real command.

## First Implementation Milestone

Do not start with full automatic cache baking. Start with observability and
repeatable probes:

1. Add `survey_scan survey` and commit its CIT inventory output format.
2. Add `rift_cupy_common.py` warmup and a Condor job emitter for one GPU class.
3. Run on `cc60-90` and `cc90-120`; record cold/warm timings and cache sizes.
4. Add the JAX warmup only for the JAX image after the CuPy path is stable.
5. Decide whether the next image build should bake caches in `%post`, publish a
   sidecar cache tarball, or simply rely on per-slot first-use warming.

