# Low-mass RF transverse-spin fitting

This is an opt-in change to the RF fit basis. `geometric4` replaces the four
Cartesian transverse fit coordinates of a full two-spin RF stage with total
transverse spin magnitude, its azimuth, and two sum-frame residuals. The stage
still fits 8 coordinates: delta_mc, mu1, mu2, chiMinus and the four geometric
ones. Priors, sampling coordinates, the L frame, ILE and the integrators are
unchanged. With the option omitted, every pipeline and the bundled Asimov
template behave as before.

## Command line

```
util_RIFT_pseudo_pipe.py ... --rf-transverse-spin-coordinates off|auto|geometric4|geometric4-phase-excess
helper_LDG_Events.py ... --rf-transverse-spin-coordinates off|auto|geometric4|geometric4-phase-excess
```

| Mode | Effect |
|---|---|
| omitted, `off` | No change to generated stages. |
| `auto` | Selects `geometric4` only for a precessing BBH whose detector-frame event chirp-mass estimate is finite, positive and below 20 Msun. |
| `geometric4`, `geometric4-phase-excess` | Activates at any mass; fails if the analysis is not eligible or no complete RF stage exists. |
| `physics3` | Retired and refused everywhere (see below). |

`auto` never uses a prior bound, source-frame mass, or the placeholder mass of an
event-time-only invocation. The 20 Msun boundary is excluded. Aligned, tides/EOS,
eccentric, high-q, total-mass, EOB-parameter and hyperbolic analyses are not
eligible: `auto` leaves them unchanged and an explicit mode fails.

## Retired: physics3

`physics3` appended three scalars to the eight native fit coordinates, so CIP fit
11 coordinates for 8 degrees of freedom. RF transverse modes require a
nonredundant fit basis; other CIP configurations, including matter fits, may fit
more features than they sample. The asimov validator, helper, pseudo_pipe and CIP refuse it, in every
argument form, including through `--manual-extra-cip-args`.

## What activation changes

When a mode activates, the helper:

1. Selects RF for every stage if no fit method was forced. An explicit non-RF
   choice is kept: `auto` then does nothing and an explicit mode fails.
2. Selects the native mu1/mu2 phase basis for every stage
   (`--internal-use-aligned-phase-coordinates`). This drops xi from the early
   stages and cuts the first stage from 3 to 2 iterations.
3. Appends `--rf-transverse-spin-coordinates geometric4 --fref FREF` to each
   complete two-spin RF stage. FREF is `engine.fref` for ini workflows, else the
   ILE reference frequency. CIP also uses `--fref` for its other spin
   conversions, so this stage no longer uses the CIP default of 20 Hz.

Step 1 happens before the initial grid is built, so the grid matches an explicit
rf run. Steps 2 and 3 happen after it, as they do for an explicit rf run.

When pseudo_pipe finds that the helper switched the fit to RF, it builds the DAG
as an explicit `--cip-fit-method rf` run: flat CIP workers
(`--cip-explode-jobs-flat`), 15000 MB CIP memory, and twice the ILE points per
iteration. CIP refuses RF transverse coordinates with `--fit-load-gp`, so
non-flat workers cannot be used.

| Starting configuration | Steps that change stages |
|---|---|
| Bare pipeline (gp default, mc basis) | 1, 2, 3 |
| Bundled Asimov template (rf, phase basis already set) | 3 only |

`test_rf_transverse_helper_generation.py` pins both rows.

pseudo_pipe refuses options that later replace delta_mc in CIP stages:
`--cip-internal-use-eta-in-sampler`, `--hierarchical-merger-prior-1g/2g` and
`--use-quadratic-early`. With an explicit mode it refuses them before running
the helper. With `auto` it refuses once a stage has activated, before the DAG is
built. A final check applies CIP's RF guard to every CIP line.

## Asimov

A silent ledger emits nothing, so the rendered ini is identical to the base
template. To opt in:

```yaml
sampler:
  cip:
    transverse spin coordinates: "auto"  # or "geometric4" / "geometric4-phase-excess" / "off"
```

Boolean `false` means `off`. Boolean `true` means `geometric4` at any mass; YAML
1.1 also loads `on` and `yes` as true. Other values, including `physics3`, fail
before the config is written. Quote the enum strings.

The template also reads two other ledger keys that the base template ignored. A
ledger that already sets them changes behavior:

| Key | Effect |
|---|---|
| `sampler.cip.fitting method` | Sets `cip-fit-method` (default `rf`); in-repo blueprints set `rf`. Unknown CIP fit methods fail before the config is written. |
| `scheduler.priority` | Integer passed as `condor_submit_dag -priority`. |

## Direct CIP

`--rf-transverse-spin-coordinates geometric4 --fref FREQ` requires a fresh RF fit
with exactly delta_mc, mu1, mu2, chiMinus and the four transverse Cartesian
coordinates, and at least as many sampled coordinates, with no sampled name
repeated. Cached GP
loading and incompatible models fail. Training, mirrored rows, predictions and
diagnostic output use the same implementation (`RIFT/misc/rf_transverse_spin.py`,
`geometric4`). The asimov README describes the chart.
