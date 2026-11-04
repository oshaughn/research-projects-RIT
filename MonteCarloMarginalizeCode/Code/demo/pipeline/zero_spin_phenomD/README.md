# Zero-spin IMRPhenomD .dgrid end-to-end validation

End-to-end validation of the per-distance likelihood export pipeline using
**zero-spin IMRPhenomD**, the **AV** sampler, and a small mass grid of zero-noise
BBH points. The full chain runs in under a minute on a single laptop core --
no condor, no GPU.

## What it exercises

1. **Build** (`make build`). Calls `util_RIFT_pseudo_pipe.py` with
   `--add-extrinsic --export-marginal-distance-grid --assume-nospin --approx IMRPhenomD --ile-sampler-method AV`
   to produce a complete RIFT run directory. Verifies the resulting
   `ILE_extr.sub` carries the export flags, that distance marginalization is
   disabled **only** at the extrinsic stage, and that the
   `consolidate_dgrid.sub` consolidation job is wired into the DAG.

2. **Run** (`make run-extr`). Bypasses condor and directly invokes
   `integrate_likelihood_extrinsic_batchmode` on the first `N_EVENTS` rows of
   the fake-data zero-noise BBH grid (`.travis/ILE-GPU-Paper/demos/overlap-grid.xml.gz`)
   with arguments matching what the pipeline would emit at the extrinsic
   stage. Produces one `.dgrid` file per intrinsic point.

3. **Consolidate** (`make consolidate`). Runs
   `util_ConsolidateDistanceGrids.py` over the per-event `.dgrid` files,
   verifying header agreement and emitting a single `all_dgrid.dat` -- the
   "net" intrinsic + distance grid that downstream tools consume.

4. **Posterior** (`make posterior`). Feeds `all_dgrid.dat` into
   `util_ConstructEOSPosterior.py` with `--parameter m1 --parameter m2 --parameter dist`,
   reconstructing the joint (intrinsic + distance) posterior. Reports the
   sample count and per-parameter mean / std as a sanity check.

`make all` runs steps 1-4 sequentially; `make clean` removes the generated
run directories.

## Inputs

| input | source | notes |
| --- | --- | --- |
| `zero_spin_phenomD.ini` | local | minimal ini whose `[rift-pseudo-pipe]` section deliberately omits `approx`/`ile-sampler-method` so the CLI overrides win |
| coinc / fake cache / PSDs / grid | `.travis/...` | the same fake-data zero-noise BBH inputs used by the ILE-GPU-Paper demo and `.travis/test-build.sh` |

## Tunables

| variable | default | role |
| --- | --- | --- |
| `N_EVENTS` | 3 | number of grid rows to run locally in step 2 |
| `N_EFF`    | 50 | ILE `--n-eff` target |
| `N_MAX`    | 30000 | ILE `--n-max` cap |

## Expected output

```
OK: pipeline built; ILE_extr.sub carries AV+IMRPhenomD+grid-export, ...
Produced .dgrid files:
  rundir_extr/demo_extr_0_.dgrid
  rundir_extr/demo_extr_1_.dgrid
  rundir_extr/demo_extr_2_.dgrid
util_ConsolidateDistanceGrids.py: wrote 150 rows from 3 files to all_dgrid.dat
joint_posterior.dat: 2000 posterior samples
  summary: {'lnL': (0.0, 0.0), 'm1': (...), 'm2': (...), 'dist': (...)}
OK: end-to-end validation complete -- pipeline built, ILE_extr produced .dgrid output, consolidated, posterior reconstructed.
All validation steps passed.
```

The injected true signal is m1 = m2 = 35 Msun at d = 200 Mpc; with `N_EVENTS=3`
the test only covers the lower edge of the mass grid (m1 = m2 ~ 26-29 Msun),
so the recovered posterior mean is biased toward those points -- this is by
design (fast test, not an accuracy demo).  Increase `N_EVENTS` to cover the
full grid for a meaningful posterior.

## Notes

- Steps 2-4 use the same code paths the production pipeline does; only the
  condor layer is bypassed.
- The .ini section `[rift-pseudo-pipe]` overrides CLI flags. The minimal
  `zero_spin_phenomD.ini` here strips fields that would override
  `--approx` / `--ile-sampler-method` / `--assume-nospin`.
- `util_ConstructEOSPosterior.py` requires `--integration-parameter-range`
  for every fitted parameter; the Makefile supplies sensible ranges for
  m1, m2, dist.
