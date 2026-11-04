# Low-mass RF transverse-spin fitting

This is an opt-in change to the RF fit basis. `geometric4` replaces the four
Cartesian transverse fit coordinates of a two-spin RF stage with total
transverse spin magnitude, its azimuth, and two sum-frame residuals. Priors,
sampling coordinates, the L frame, ILE and the integrators are unchanged. With
the option omitted, every pipeline and the bundled Asimov template behave as
before.

The chart coordinates are ordinary CIP coordinates, defined in lalsimutils:

| Name | Meaning |
|---|---|
| `rf_total_perp` | total transverse spin \|S_perp\|/M^2 |
| `rf_phase_excess` | (J-\|J_parallel\|)/L_N, replaces `rf_total_perp` in geometric4-phase-excess |
| `rf_total_azimuth` | azimuth of S_perp in the L plane |
| `rf_residual_parallel`, `rf_residual_perpendicular` | sum-frame residuals |

They depend on the reference frequency: `extract_param` uses `P.fref`, and
`convert_waveform_coordinates` takes `fref=`. The helper option only decides
which stages use them.

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
11 coordinates for 8 degrees of freedom. The asimov validator, helper, pseudo_pipe and CIP refuse it, in every
argument form, including through `--manual-extra-cip-args`.

## What activation changes

When a mode activates, the helper:

1. Selects RF for every stage if no fit method was forced. An explicit non-RF
   choice is kept: `auto` then does nothing and an explicit mode fails.
2. Selects the native mu1/mu2 phase basis for every stage
   (`--internal-use-aligned-phase-coordinates`). This drops xi from the early
   stages and cuts the first stage from 3 to 2 iterations.
3. On each RF stage that fits all four of s1x, s1y, s2x, s2y, replaces them by
   the chart. An implied component (`--parameter-implied s1x`) becomes a chart
   name. A sampled one (`--parameter s1x`, as in volumetric-spin stages) becomes
   `--parameter-nofit s1x --parameter-implied <chart name>`, so the sampled set is
   unchanged. The helper then sets `--fref FREF`. FREF is `engine.fref` for ini
   workflows, else the ILE reference frequency. CIP also uses `--fref` for its
   other spin conversions, so this stage no longer uses the CIP default of 20 Hz.

Step 1 happens before the initial grid is built, so the grid matches an explicit
rf run. Steps 2 and 3 happen after it, as they do for an explicit rf run.

When pseudo_pipe finds that the helper switched the fit to RF, it builds the DAG
as an explicit `--cip-fit-method rf` run: flat CIP workers
(`--cip-explode-jobs-flat`), 15000 MB CIP memory, and twice the ILE points per
iteration.

| Starting configuration | Steps that change stages |
|---|---|
| Bare pipeline (gp default, mc basis) | 1, 2, 3 |
| Bundled Asimov template (rf, phase basis already set) | 3 only |

`test_rf_transverse_helper_generation.py` pins both rows.

The chart is computed from physical masses and spins, so options that later
rewrite the sampling coordinates (for example `--cip-internal-use-eta-in-sampler`)
do not conflict with it. pseudo_pipe still refuses `physics3` on every final CIP
line, including `--manual-extra-cip-args`.

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

Name the chart with `--parameter-implied` and set `--fref`:

```
--parameter-implied rf_total_perp --parameter-implied rf_total_azimuth
--parameter-implied rf_residual_parallel --parameter-implied rf_residual_perpendicular --fref FREQ
```

Any sampling coordinates that determine both spins work. RF fits handle the
azimuth seam at +/-pi; other fit methods print a warning. CIP's
`--rf-transverse-spin-coordinates` is a deprecated alias: it renames implied
s1x, s1y, s2x, s2y to the chart and does nothing else. The chart math is in
`RIFT/misc/rf_transverse_spin.py`. The asimov README describes the chart.
