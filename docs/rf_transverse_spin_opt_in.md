# Prototype low-mass RF transverse-spin fitting

This is an opt-in change to RF fitting features. It appends three scalars (cone
squared, phase deficit, torque squared) to full two-spin Cartesian RF fitting
stages. Every original fitting coordinate stays, including mu1/mu2 and all four
transverse spin components. Priors, sampling coordinates, the L frame, ILE and
the integrators are unchanged. With the option omitted, every pipeline and the
bundled Asimov template behave as before.

## Command line

```
util_RIFT_pseudo_pipe.py ... --rf-transverse-spin-coordinates off|auto|physics3
helper_LDG_Events.py ... --rf-transverse-spin-coordinates off|auto|physics3
```

| Mode | Effect |
|---|---|
| omitted, `off` | No change to generated stages. |
| `auto` | Activates only for a precessing BBH whose detector-frame event chirp-mass estimate is finite, positive and below 20 Msun. |
| `physics3` | Activates at any mass; fails if the analysis is not a precessing BBH or no complete RF stage exists. |

`auto` never uses a prior bound, source-frame mass, or the placeholder mass of an
event-time-only invocation. The 20 Msun boundary is excluded. Aligned, tides/EOS,
eccentric, high-q and total-mass schedules are outside this prototype.

## What activation changes

When `auto` or `physics3` activates, the helper:

1. Selects RF for every stage if no fit method was forced. An explicit non-RF
   choice is kept: `auto` then does nothing and `physics3` fails.
2. Selects the native mu1/mu2 phase basis for every stage
   (`--internal-use-aligned-phase-coordinates`). This drops xi from the early
   stages and cuts the first stage from 3 to 2 iterations.
3. Appends `--rf-transverse-spin-coordinates physics3 --fref FREF` to each
   complete two-spin RF stage. FREF is `engine.fref` for ini workflows, else the
   ILE reference frequency. CIP also uses `--fref` for its other spin
   conversions, so this stage no longer uses the CIP default of 20 Hz.

Step 1 happens before the initial grid is built, so the grid matches an explicit
rf run. Steps 2 and 3 happen after it, as they do for an explicit rf run.

When pseudo_pipe finds that the helper switched the fit to RF, it builds the DAG
as an explicit `--cip-fit-method rf` run: flat CIP workers
(`--cip-explode-jobs-flat`), 15000 MB CIP memory, and twice the ILE points per
iteration. CIP refuses `physics3` with `--fit-load-gp`, so non-flat workers
cannot be used.

| Starting configuration | Steps that change stages |
|---|---|
| Bare pipeline (gp default, mc basis) | 1, 2, 3 |
| Bundled Asimov template (rf, phase basis already set) | 3 only |

`test_rf_transverse_helper_generation.py` pins both rows.

pseudo_pipe refuses options that later replace delta_mc in CIP stages:
`--cip-internal-use-eta-in-sampler`, `--hierarchical-merger-prior-1g/2g` and
`--use-quadratic-early`. With `physics3` it refuses them before running the
helper. With `auto` it refuses once a stage has activated, before the DAG is
built. A final check applies CIP's physics3 guard to every activated stage.

## Asimov

A silent ledger emits nothing, so the rendered ini is identical to the base
template. To opt in:

```yaml
sampler:
  cip:
    transverse spin coordinates: "auto"  # or "physics3" / "off"
```

Boolean `false` means `off`. Boolean `true` means `physics3` at any mass; YAML 1.1
also loads `on` and `yes` as true. Other values fail before the config is written.
Quote the enum strings.

The template also reads two other ledger keys that the base template ignored. A
ledger that already sets them changes behavior:

| Key | Effect |
|---|---|
| `sampler.cip.fit method` | Sets `cip-fit-method` (default `rf`). Unknown CIP fit methods fail before the config is written. |
| `scheduler.priority` | Integer passed as `condor_submit_dag -priority`. |

## Direct CIP

`--rf-transverse-spin-coordinates physics3 --fref FREQ` requires a fresh RF fit
with delta_mc, mu1, mu2, chiMinus and all four transverse Cartesian coordinates.
Cached GP loading and incompatible models fail. Training, mirrored rows,
predictions and diagnostic output use the same scalar implementation.

## Definitions

For component masses in solar masses, x_i=m_i/M, eta=x_1*x_2,
v=(pi*M*4.9254909476412675e-6*fref)^(1/3), define L=eta/v,
D=L+x_1^2*s1z+x_2^2*s2z, T=x_1^2*s1perp+x_2^2*s2perp,
J=sqrt(D^2+|T|^2), G=(2+1.5*q)*x_1^2*s1perp+(2+1.5/q)*x_2^2*s2perp.
The features are |T|^2/[D^2+(0.1L)^2], (J-D)/L and |G|^2/(eta*v^2)^2.
For D>=0, (J-D) is evaluated as |T|^2/(J+D). These are fitting summaries, not
SEOBNRv5PHM Fisher coordinates.

## Evidence

Controlled tests improved both spin marginals in S250628am, S250205bk and
S250328ae. S250830bp improved partly. Finite-bin joint JS is descriptive only.
Calibration-treated references and provisional intrinsic CIP are different
statistical targets. This is a prototype, not a default.
