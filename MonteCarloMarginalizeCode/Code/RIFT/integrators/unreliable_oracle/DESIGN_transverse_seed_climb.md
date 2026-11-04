# Transverse-spin seed climb

Status: prototype (branch `transverse_seed_climb`, 2026-10-17). Off by default.

## Problem

The default initial grid is aligned-spin. `helper_LDG_Events.py` seeds only `s1x`,
at about 1e-5 (`RIFT/physics/teobresums_compat.py`, `initial_transverse_spin_range`).
For events with large in-plane spin the high-likelihood region is narrow in
(Mc, q, chi_eff) and shifted from the aligned-spin optimum. Points with random
in-plane spin rarely land on it. IR1 needed external Bilby bootstraps to place
points there (S250830bp, `~/LVK/IR1/s250830bp-high-transverse-bootstrap-20261010`).

## Method

`bin/util_TransverseSeedClimb.py` hillclimbs a fast likelihood and writes the
points it visits near the peak as a sim_inspiral grid.

**Chart.** x = (mu1, mu2, q, s2z, chi1_perp, phi1), optionally with
(chi2_perp, phi2). The (mu1, mu2) pair (`RIFT/misc/tools.py`) removes most of the
Mc-q-chi_eff correlation, so a climb in this chart moves along the narrow
direction at the right scale. A point outside the Kerr bound scores -inf.

**Objective** (`RIFT/likelihood/jax_ile/intrinsic_objective.py`). For each
intrinsic point:

1. the production precompute (`build_data_from_precompute`) builds Q, U, V from
   the lalsimulation modes of `--climb-approx` (default: the analysis `--approx`);
2. the JAX ILE kernel `fused_log_likelihood_distphipsimarg` marginalizes over
   time, distance, orbital phase and polarization;
3. the result is averaged over a fixed batch of (ra, dec, incl) drawn from an
   importance proposal: a BIC-selected Gaussian mixture in
   (ra, sin dec, cos incl), plus 10% prior.

One batch is fixed across all climbs (common random numbers), so their scores
are comparable. `--sky-readapt` restores adaptation before each climb. The per-point
arrays are jit arguments substituted into a template data object, so the kernel
compiles once per run. The stock wrappers close over one data object and would
recompile for every point.

**Climber** (`LadderClimbingOracle`, this directory). CMA-ES, with reflection at
box edges and wrapping of angles. It runs:

- free climbs from the best few points of the input grid, after scoring a subset
  of it;
- a ladder of climbs with chi1_perp pinned at fixed values (default
  0.3 and 0.6). Rungs run outward from the free optimum. Each starts from
  its neighbour's best point, projected inside the Kerr bound at fixed (Mc, q).

Default placement draws an equal share from each qualifying climb's final
Gaussian, broadening pinned coordinates between rungs. History placement is
also available. `--delta-lnL` (default 10) selects qualifying climbs. The ladder supplies
near-peak points at large chi1_perp with the in-plane orientation the data
prefer. Random seeding does not.

The seeds are proposals. ILE re-scores them, and the climb objective never enters
an evidence or posterior.

## Pipeline wiring

| Scheme | Option | Where it runs | Effect |
|---|---|---|---|
| (a) setup | `pseudo_pipe --transverse-seed-climb setup` | submit host, during setup, CPU unless a GPU is visible | appends seeds to `proposed-grid.xml.gz` |
| (b) DAG | `pseudo_pipe --transverse-seed-climb dag [--transverse-seed-climb-request-gpu]` | node `seed_climb.sub`, parent of every iteration-0 ILE job | appends seeds to `overlap-grid-0.xml.gz` |
| (c) parallel | `pseudo_pipe --transverse-seed-climb parallel` | node `seed_climb.sub` with no parent, alongside iteration 0 | writes `seed-0.xml.gz`; its `ILE_seed` jobs feed iteration-0 consolidation, which waits for both branches; ordinary iteration-0 ILE starts independently |
| (d) composite | `pseudo_pipe --transverse-seed-climb composite` | node `seed_climb.sub`, child of the iteration-0 unify, beside CIP 0 | climbs from the best lines of `all.net` (`--start-composite`), writes `seed-1.xml.gz`; its `ILE_seed` jobs feed the iteration-1 consolidation |

Scheme (d) is the intended seeding strategy. Iteration 0 runs unchanged. Its
best ILE-scored lines, usually near aligned spin, give the climb starts close
to the (mc, q, chi_eff) peak, so the climb spends its budget on transverse
spin. The climb and its ILE jobs run while CIP 0 and the iteration-1 ILE jobs
run, and add a few hundred re-scored points at large chi1_perp to the
iteration-1 fit. Later puffs spread around them. `all.net` is rewritten only by
the iteration-1 unify, which waits on the seed ILE jobs. The seed ILE jobs carry
the same iteration-0 calibration and extrinsic breadcrumb barriers as the puff
ILE jobs. It needs at least two iterations.

All take extra climber arguments through `--transverse-seed-climb-args`. The
climber reads data, PSD and waveform settings from `args_ile.txt`.

Scheme (b) uses the CEPP options `--seed-climb-exe`, `--seed-climb-args`,
`--seed-climb-n-max`, `--seed-climb-request-gpu` and `--seed-climb-request-memory`
(BasicIteration only). The iteration-0 ILE batches are sized at build time for
(input grid + `--seed-climb-n-max`) points. ILE exits without error on indices past
the end of the grid. The node is skipped under `--first-iteration-jumpstart`.
The sub file has no file transfer, so it needs the shared run directory
(CIT local pool on `/scratch`, as for the other non-worker jobs).

In a container (`--use-singularity`) the climb needs its own image, because no
RIFT ILE image carries JAX: `--seed-climb-singularity-image` (pseudo_pipe
`--transverse-seed-climb-singularity-image` or
`$SINGULARITY_RIFT_SEED_CLIMB_IMAGE`), or the build refuses. The sub sets
`MY.POOLS = "CIT"` and omits `flock_local`, which the CIT schedd rewrites to
"IGWN,CIT" (IGWN nodes lack `/scratch`). It requests a transfer sandbox so the
container has a `HOME`, and excludes `x86_64-v1` hosts: prebuilt jaxlib needs AVX,
which CIT's GTX 1050 Ti nodes lack. It takes the ILE condor commands (GPU requirements) and does
not transfer the executable, so the climber path must exist in the image or on
a bound filesystem. Until a RIFT image ships the climber, point
`--transverse-seed-climb-exe` at a `/scratch` checkout and set
`RIFT_SEED_CLIMB_PYTHONPATH` at build time. The container sub passes only
`*RIFT*` variables.

The final extrinsic stage reads the grid of the last fit, in seeded and
unseeded runs alike (RO, 2026-10-20: the earlier choice was an off-by-one).

## Limits

- The climb model must have the same precession as the analysis model. On IR1
  PP event 005 (XPHM injection, q = 0.09, chi1 = 0.94), IMRPhenomPv2 scores the
  true parameters 38 nats below XPHM. A Pv2 climb would follow the wrong ridge.
- `--device-precompute gpu` reduces Q/U/V on the GPU with
  `gpu_precompute.compound_precompute_arrays` (A = 1), the reduction written for
  the rotating finite-response likelihood. The first point is also built on CPU
  and compared. lalsim mode generation stays on CPU and is the floor: about
  1.6 s (l_max 2) and 3.3 s (l_max 4) per point on S250830bp (32 s at 4096 Hz).
- `--mode-backend ripple` takes IMRPhenomXPHM modes from ripple 0.4.1
  (`RIFT/likelihood/jax_ile/ripple_modes.py`), so the objective runs on the
  accelerator. ripple's h+, hx match lalsim XPHM to overlap 0.999999. The modes
  are J-frame and unconditioned, which is valid only because the objective
  marginalizes orientation. On PP 005 (pcdev11) the objective agrees with the
  lalsim XPHM path to within 1.4 nats over a 7 to 46 nat range, with the same
  ranking of six test points. ripple is an optional dependency, for targeting
  and development; the dependency-free cupy port is PR #453.
- The phiref and psi grids default to 8 l_max and 4 l_max points. lnL has
  phiref harmonics up to 2 l_max, and J-frame modes populate every m. On
  S250830bp at l_max 4 an 8 x 4 grid moved lnL by up to 20 nats and reversed the
  mass-ratio ranking (the ripple climb then settled at q = 0.28); at 32 x 16 the
  ripple and lalsim objectives agree to 0.4 nats at 5 of 6 test points. An
  earlier note here, that 8 x 4 kept the ranking, rested on one event and was
  wrong.
- ripple 0.4.1 XPHM itself departs from lalsim near q = 0.74 with s2z near -0.9
  (h+ overlap 0.96 to 0.97); elsewhere tested it matches to 1e-5.
- The sky batch adapts in (ra, sin dec) only and draws cos incl from its prior.
  Inclination depends on frame, and a batch concentrated in incl biased the
  comparison between intrinsic points.
- Placement defaults to covariance: two free climbs and rungs at chi1_perp 0.3
  and 0.6, then equal draws per climb from N(best point, (1.5)^2 x final CMA
  covariance), with the pinned chi1_perp spread by half the rung spacing.
- Reduced-band, batching and relative-binning timings from the October 18
  investigation remain historical observations. The archived reproduction script
  differs from its original manifest; those timings are excluded from acceptance
  evidence. Relative binning and batching remain opt-in performance choices.
- Decisions (RO, 2026-10-18): the climb is for targeting, and seeds need only
  support large transverse spin near the peak, since the puff spreads around
  them; points outside the Bilby box are acceptable. Speed matters more than
  accuracy. The climb uses l_max 2 by default (`--climb-l-max`). The intended
  production use is now composite mode, described above.
- One sky batch serves all climbs (`--sky-readapt` restores per-climb
  adaptation). A batch adapted at one climb's start scored that climb's best
  point 9 nats higher than a common batch did, so lnL from different climbs was
  not comparable. Four free starts by default.
- Complete production re-score (S250830bp, SEOBNRv5PHM ILE, peak lnL 178.7):
  all 993 seeds are finite. At chi1_perp 0.5 to 0.7, slow seeds (l_max 4,
  4096 Hz, fmax 1792 Hz) reach within 10 nats for 59 of 69 points; fast seeds
  (l_max 2, 1024 Hz, fmax 448 Hz) reach it for 0 of 48. Above 0.7 the counts
  are 8 of 40 and 0 of 19. The reduced-band setup fails the tested targeting task.
- A six-cell diagnostic on four fixed points separates l_max, frequency cutoff
  and sampling rate using one common sky batch. For the fast high-spin optimum,
  cutting fmax at 4096 Hz shifts the objective by +12.62 nats; downsampling at
  fmax 448 Hz then shifts it by -3.63 nats. Changing l_max shifts this point
  by less than one nat. Both band and rate matter (a conditional objective
  diagnostic on one sky batch).
- Keep the analysis sampling rate and frequency band. The reduced-band speed
  configuration is withdrawn. Evidence: paper repository
  `analyses/transverse_seed_climb/RESULTS_2026-10-19.md`.
- Warm start-up: JAX's compilation cache is keyed on lowered HLO, so it skips
  only the XLA compile; tracing ripple cost ~6 min per run. `export_cache.py`
  serializes the lowered ripple programs (`jax.export`; needs `flatbuffers`) under
  `$JAX_EXPORT_CACHE_DIR`, default `<compilation cache>/exported`. Single points
  run through the batched program. Warm start stage on S250830bp: 360 s to
  133 s, with identical lnL.
- On CIT head nodes, a second JAX process can hit the 500-thread cap
  (`pthread_create() failed`). Execute nodes have no such cap.
- Only BasicIteration has the three DAG schemes. Other builders can use setup mode.
- Intermittent all-NaN rows came from an uninitialized (2,-2) buffer in
  `lalsimutils.hlmoft_IMRPv2_dict` (IMRPhenomPv2; #488, fixed in #489). The
  tests use XPHM, which avoids that function. Non-finite rows score as zero
  likelihood and are counted in `stats()['n_nonfinite']`.

## Tests

- `test/integrators/test_ladder_climber.py`: toy ridge, box reflection, draws.
- `test/integrators/test_seed_climb_composite.py`: composite reader (duplicates, failed lines, mass order).
- Real-data runs and their numbers are in the paper-repo RESULTS note.
