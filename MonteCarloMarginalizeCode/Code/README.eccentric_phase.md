# Optional effective eccentric-phase coordinates

`--eccentric-phase-coordinates` enables a cheap, invertible coordinate for
eccentricity/anomaly likelihood fitting and grid expansion. It does not redefine
the waveform inputs or the physical prior. It is off by default.

The input called `meanPerAno` is treated as the backend's original periodic
angle. Different backends may mean different things by that name. The coordinate
defaults to an effective shear, not a claim that this angle is a particular PN/EOB mean
anomaly or that the phase prediction is exact:

```
theta = (meanPerAno + A(m1, m2, eccentricity)) mod 2pi
meanPerAno = (theta - A(m1, m2, eccentricity)) mod 2pi
```

`A` is the full-eccentricity Peters quadrupole phase advance,
including the mass-dependent circular term. `RIFT.misc.eccentric_phase` provides
vectorized `phase_advance`, `forward`, `inverse` and periodic feature functions.
Component masses in this module are detector-frame solar masses. The waveform
object extractor handles SI masses, and the CIP projection uses the existing
source-redshift/native-mass conversion.

## Selectable procedures

Use `--eccentric-phase-procedure` (or the same key under Asimov's
`sampler: extra eccentric arguments`) to choose:

- `peters-small-e`: the legacy O(e²) shear, available explicitly.
- `peters-full-e`: the default and recommended full Peters quadrupole phase, retaining the circular
  term and using the same mean-frequency endpoints. It is independent of spin.
- `eob-tabulated`: a frozen numerical phase/clock map informed by public EOB
  dynamics, using detector masses, eccentricity and both spin-z components.

These are fitting and proposal coordinates. Invertibility and useful ridge
geometry are the acceptance criteria; exact alignment between waveform
approximations is neither required nor expected. The fitting model sees sine
and cosine features, while CIP samples the original parameters with the
original prior. No transformed-density Jacobian enters that native integral.
Puffing first assigns all proposed masses, spins and eccentricity, then inverts
at those proposed values. It never reuses the seed's phase shift.

The recommended configuration needs only `eccentric-phase-coordinates: true`.
Full-e Peters requires no EOB quadratures, coefficient payload, map file, or
additional container file transfer. The feature itself remains opt-in.

For an optional public-EOB correction, select:

```yaml
sampler:
  extra eccentric arguments:
    eccentric-phase-coordinates: true
    eccentric-phase-procedure: eob-tabulated
```

The helper records the coincidence component masses/spins and frequency anchor.
After resolving all CIP prior overrides, the outer build constructs a bounded
local chart using public LO spin-orbit/spin-spin Hamiltonian terms and relative
2PN radiation reaction (Khalil et al., https://arxiv.org/abs/2104.11705v2).
Initialization uses geometric eccentricity and Kepler mean anomaly. At most48
profiles are attempted; no waveform, likelihood or posterior archive is used.
The chart is clipped to the configured mass, eccentricity and spin support;
its exterior returns smoothly to full-e Peters. The original prior remains
unchanged. Unavailable returning orbits produce an explicit Peters fallback.

Generated coefficients and diagnostics are per-run artifacts, cached only for
identical sources, coincidence, priors and endpoints. No event map is shipped.
The finalized `eccentric_phase_build.json` records the source/input hashes and
fallback status. A bounded immutable base64 payload travels in ordinary worker
arguments to every eligible CIP/puff job. No separate map-file transfer or
shared filesystem is needed, including with containers. ILE receives no
coordinate flags. The high-mass identity fallback skips EOB generation entirely.
An explicit JSON override remains supported for advanced use; conflicting
choices are refused. Build hosts need SciPy for EOB; workers need only NumPy.

### Numerical map contract

The legacy `rift-eccentric-map-v1` stores normalized polynomial coefficients for the
unwrapped advance and sine harmonics of the native-to-radial clock. Features
are total detector mass M, primary-first mass difference delta, geometric e,
chi_eff and primary-first chi_a. The JSON supplies center, scale, domain bounds,
powers, coefficients, endpoint frequencies and nonempty provenance. No source
code is executed when loading a map.

Automatic builds use `rift-eccentric-map-v2`, with detector chirp mass, delta,
eccentricity and primary-first component spin-z features. They store a phase
correction to full-e Peters and clock harmonics.

Inside the declared box the map supplies its phase/clock prediction. A smooth
compact taper returns it to full-e Peters outside that box. Harmonics are
regularized so their derivative norm is at most0.8, guaranteeing an angular
derivative at least0.2. Bisection inverts this monotone circle map, including
periodic seams. This global extension avoids rejecting otherwise valid native
prior support. Native Kerr checks remain the native converter's responsibility.
The taper/regularization are fixed parts of the coordinate, not inferred
physics or sampler-dependent adjustments. New maps can represent other EOB
orders, models, or useful approximations through the same interface.

## Usage

The helper resolves the frequency once at workflow build time. Enable the
coordinate in an eccentric Asimov production with:

```yaml
sampler:
  extra eccentric arguments:
    eccentric-phase-coordinates: true
```

Keep the production's existing eccentricity bounds and priors in that block.
Asimov supplies `eccentric-phase-frequency-policy: lvk-mass` by default; the
pseudo-pipe forwards the request to `helper_LDG_Events.py`. The helper resolves
one value from the build event's **detector-frame** masses, writes
`eccentric_phase_build.json`, and emits numerical frequencies for eligible CIP
stages and puffing. The outer pseudo-pipe reads that record rather than
selecting again. Workers never select an anchor. The feature remains opt-in.

### LVK mass policy and high-mass fallback

For detector-frame total mass M in solar masses, compute

```
f_mass = 0.05**1.5 / (pi * 4.9254909476412675e-6 * M)  # Hz
```

This uses the dimensionless frequency x=(pi G M f/c^3)^(2/3)=0.05
(approximately a20M circular-orbit separation at Newtonian order). It is a
versioned early-inspiral coordinate policy, not a calibrated sensitivity
maximum or guarantee of PN accuracy. The LVK default early-band interval is
20--40Hz; its floor is raised to the lowest active detector fmin if needed,
and its ceiling respects the analysis fmax.

- If f_mass is within the interval, use it continuously, without quantizing.
- If f_mass exceeds the ceiling, use the ceiling.
- **High mass:** if f_mass falls below the interval floor, use the native epoch
  itself: an identity shear/native circular chart. Do not push a weak-field
  landmark into the later observed band merely to satisfy its lower bound.
- If the otherwise selected anchor precedes the native epoch, also use the
  identity shear. The receipt records the fallback and achieved x.

At native10Hz and network fmin20Hz this gives about40Hz for10Msun,
32.84Hz for22Msun,20.07Hz for36Msun, and an identity shear for70/120Msun.
The normal mass source is the pipeline's coincidence file: the helper reads
its detector-frame component masses before selecting the anchor. A bootstrapped
coincidence file works identically to a search-produced one; it need not contain
posterior estimates or independently recovered injection truth. Use the supplied
`--use-coinc` path (or the pipeline-resolved/downloaded coincidence) rather than
searching unrelated directories for another file. Explicit mass/target hints
remain optional overrides, not a requirement for bootstrapped files.

These are build-time event mass estimates, not sample-by-sample anchor choices.
Each coordinate uses the proposed point's detector masses at that fixed anchor.

The native epoch is inferred when template start and waveform reference agree.
For SEOBNRv5EHM it follows template start when they differ. Other ambiguous
backend conventions require an explicit `eccentric-phase-reference-frequency`.
Missing/placeholder event masses are refused; manual builds can provide
`--eccentric-phase-detector-mass M` (detector total mass) or a fixed target.

Explicit overrides remain available through Asimov's same block or either
builder CLI:

```
--eccentric-phase-coordinates \
--eccentric-phase-reference-frequency 10 \
--eccentric-phase-target-frequency 20
```

An explicit target wins over the mass policy. The legacy fixed20Hz policy is
available with `--eccentric-phase-frequency-policy fixed`. Negative/nonfinite
inputs, backward explicit transport, conflicting stage frequencies and late
manual ILE epoch or detector-band frequency changes fail before submission. A rebuild may select a
new value, recorded in the new build receipt; an executing workflow does not
retune. These frequencies configure **only the coordinate**, never waveform
fref/fmin or the physical priors. Direct CIP/puff calls continue requiring a
fixed numerical configuration (their legacy defaults remain10/20Hz).

CIP continues sampling native `meanPerAno` and the selected native eccentricity
coordinate, with the original bounds/prior options. Only its fitted angle is
replaced with `sin(theta), cos(theta)`; the two features are constrained to a
circle and are not two sampled physical parameters. Scalar, vector, diagnostic
and training feature conversions use the same map. Early stages configured with
`--parameter-nofit meanPerAno` remain unchanged. Pipeline propagation enables
the flag only on stages that actually fit eccentricity and anomaly.

Puffball continues accepting native parameter names, for example:

```
util_ParameterPuffball.py --inj-file grid.xml.gz --inj-file-out puff \
  --parameter mc --parameter delta_mc --parameter eccentricity \
  --parameter meanPerAno --parameter chieff_aligned --puff-factor 2 \
  --eccentric-phase-coordinates
```

Puffing replaces the angle internally with a circular covariance chart. It
retains the requested eccentricity representation; choose `eccentricity_ln`
instead of `eccentricity` if log-e puffing is wanted. All proposed non-angle
parameters are assigned first, then the inverse uses the **new** masses, spins
and eccentricity. The existing downselects act on physical/native parameters, and
the output XML has the original eccentricity and anomaly fields. Unchanged
waveform fields are retained from the input objects. `--force-away` minimizes
the correlated Mahalanobis distance over all integer lifts of the effective
angle, rather than creating a false discontinuity at its periodic seam.

The effective-coordinate puff writes `<output>_eccentric_phase.json` with its
configuration/chart origin and exported count. Optional `--random-seed` seeds
puff draws in either native or effective mode; its default preserves the
previous random behavior.

## Prior preservation and scope

The Peters procedures are angular shears with determinant one. The EOB clock
map generally has a nonunit angular Jacobian, so a uniform native anomaly
need not be uniform in the transformed angle. All procedures nevertheless
preserve the original physical prior because sampling and integration stay
in native coordinates. Fitting sine/cosine features adds no Jacobian to that
native integral. Grid proposals are not posterior draws. The code does not sample
a flat Cartesian eccentricity disk or impose a new prior.

The initial fit option rejects EOS/tidal runs, RF-transverse coordinate mode,
loading previously saved fits, internal `quadpuff`, and `--report-best-point`.
The `quadratic` fitter is also refused: its normal matrix is singular because
`sin(theta)^2 + cos(theta)^2 = 1`. Use RF, GP, or polynomial fitting instead.
Those paths need separate support for a constrained periodic feature manifold.
Puffing rejects independent `--random-parameter` overrides in this mode.
It also refuses `--reflect-parameter meanPerAno`: physical anomaly bounds cannot
be reflected in the effective angular chart. Native anomaly downselection remains
available through `--downselect-parameter meanPerAno`. The
legacy periodic-eccentricity fit option and AMR cannot be combined with the
top-level flag. No physical PN accuracy at high eccentricity or across waveform
families is claimed; approximation error changes ridge straightness, not the
explicit inverse or the waveform parameters.

Focused tests are `test/test_eccentric_phase.py`,
`test/test_eccentric_phase_puff.py` and `test/test_eccentric_phase_models.py`. They exercise native sampling preservation,
periodic seams, row-independent/redshift-aware conversion, inverse order when
masses change, exact circular distance, serialized frequency preservation,
and following a curved effective-phase band through the actual puff CLI.

Pipeline compatibility checks run after INI loading, so CLI and INI options
are validated together before workflow setup.

Build-policy regressions in `test/test_eccentric_phase_policy.py` exercise the
mass law, LVK bounds, high-mass identity fallback, real Asimov Liquid rendering,
explicit overrides, actual helper hook, saved decision and consistent CIP/puff
arguments. Production-readiness also requires supported-runtime checks of the
actual generated DAG/submit files, not just accepting new options.
