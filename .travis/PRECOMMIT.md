# CI tiers

`.github/workflows/ci.yml` gates every push/PR with ~22 jobs. That is already large, and
covering every backend and hardware corner in it would grow it without bound: no GitHub
runner has a GPU, and the JAX suite already costs real time. This tiers the existing suite;
nothing here removes or shortens a job.

## Tier 1 -- fast CI, every push/PR (unchanged)

Everything in `ci.yml` runs automatically: install, help-check, import-check,
dependency-compat-check, sim-manager-check, integrator-gate-accounting-check,
q-window-stencil-check, ci-roster-check, roster-verify-check, core-unit-check, slowrot-check,
jax-ile-check, lisa-check, calmarg-check, integration-check, asimov-integration,
rimsky-integration, test-run, container-dep-canary, container-swig-canary, docs. Plus
`.gitlab-ci.yml`'s default pipeline (coord/integrate/posterior/run/run-alts/build).

RECOMMENDATION for RO'S: `jax-ile-check` runs on every push, and its cost argues for tier 2 or
a schedule instead. `.travis/test-jax.sh:494` measures the gated 139-test baseline at 13m53s
(833s), against a 60-minute timeout; the suite has since grown to 577 tests
(`.travis/test-jax.sh:766`). `test_angle_marg_exact.py` (36 tests, excluded from this gate at
`.travis/test-jax.sh:484`) does not add to that per-push cost; run it by hand per the comment
there. Moving `jax-ile-check` to a nightly schedule would return the 833s to every push, since
nothing else in tier 1 imports jax. This PR leaves it in place: moving a job is a behavior
change and belongs in its own PR.

## Tier 2 -- strongly recommended before merging into rift_O4d, by hand on a GPU/big node

No GitHub runner has a GPU, so these either never run, or run a cupy-less CPU parity check
that cannot see a real device. Run `.travis/precommit-recommended.sh` before merging
anything touching the ILE likelihood, the NoLoop stencils, calibration marginalization, or
the JAX driver:

- `RIFT/likelihood/test_q_window_interp_gpu.py`, `test_noloop_gpu_stencils.py` (real cupy)
- `.travis/test-calmarg-gpu.sh` (fused calibration CUDA kernels)
- `test/jax/test_angle_marg_exact.py`, and the full `.travis/test-jax.sh`
- `test/expensive_before_merging/` (`RIFT_RUN_EXPENSIVE=1`)

The script exits 1 when cupy/CUDA or the jax stack is absent; it does not skip.

## Tier 3 -- scheduled or manual only

`.gitlab-ci.yml`'s `gpu_integration` job: a real GPU runner, running `test-integrate.sh` +
`test-calmarg-gpu.sh` + `test-lisa-gpu.sh` under CUDA. Its `rules:` block
(`.gitlab-ci.yml:180-184`) has two branches: web-triggered `when: manual`, and
`$CI_PIPELINE_SOURCE == "schedule"` with `when: on_success`. So it also runs automatically on a
scheduled pipeline. A nightly `jax-ile-check` would belong on the same schedule if RO'S takes
the tier-1 recommendation above.
