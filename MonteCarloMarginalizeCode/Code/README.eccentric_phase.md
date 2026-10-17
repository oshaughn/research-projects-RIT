# Optional effective eccentric-phase coordinates

`--eccentric-phase-coordinates` enables a cheap, invertible coordinate for
eccentricity/anomaly likelihood fitting and grid expansion. It does not redefine
the waveform inputs or the physical prior. It is off by default.

The input called `meanPerAno` is treated as the backend's original periodic
angle. Different backends may mean different things by that name. The coordinate
is an effective shear, not a claim that this angle is a particular PN/EOB mean
anomaly or that the phase prediction is exact:

```
theta = (meanPerAno + A(m1, m2, eccentricity)) mod 2pi
meanPerAno = (theta - A(m1, m2, eccentricity)) mod 2pi
```

`A` is the leading secular PN phase advance through eccentricity squared,
including the mass-dependent circular term. `RIFT.misc.eccentric_phase` provides
vectorized `phase_advance`, `forward`, `inverse` and periodic feature functions.
Component masses in this module are detector-frame solar masses. The waveform
object extractor handles SI masses, and the CIP projection uses the existing
source-redshift/native-mass conversion.

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

These are build-time event mass estimates, not sample-by-sample anchor choices;
the shear itself still uses each physical point's masses. The completed paired
pilot and archival plots retain their previously specified fixed anchors.

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
parameters are assigned first, then the inverse uses the **new** masses and
eccentricity. The existing downselects act on physical/native parameters, and
the output XML has the original eccentricity and anomaly fields. Unchanged
waveform fields are retained from the input objects. `--force-away` minimizes
the correlated Mahalanobis distance over all integer lifts of the effective
angle, rather than creating a false discontinuity at its periodic seam.

The effective-coordinate puff writes `<output>_eccentric_phase.json` with its
configuration/chart origin and exported count. Optional `--random-seed` seeds
puff draws in either native or effective mode; its default preserves the
previous random behavior.

## Prior preservation and scope

Keeping eccentricity and all other physical parameters unchanged makes this
an angular shear with determinant one. Uniform native anomaly remains uniform;
the original eccentricity prior, including a log-uniform prior, is unchanged.
Fitting sine/cosine features adds no Jacobian to an integral performed in native
coordinates. Grid proposals are not posterior draws. The code does not sample
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

Focused tests are `test/test_eccentric_phase.py` and
`test/test_eccentric_phase_puff.py`. They exercise native sampling preservation,
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
