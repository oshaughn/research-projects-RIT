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

Add the same configuration to eligible CIP and puff commands, or to
`util_RIFT_pseudo_pipe.py` / `helper_LDG_Events.py`:

```
--eccentric-phase-coordinates \
--eccentric-phase-reference-frequency 10 \
--eccentric-phase-target-frequency 20
```

These two frequencies configure **only the coordinate**. They do not set
`--fref`, `--fmin`, or a waveform-generator argument. Set the reference label
appropriately for the original eccentricity/angle convention. Defaults are
10 and 20 Hz. The helper/pipeline requires `--assume-eccentric --use-meanPerAno`.

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
