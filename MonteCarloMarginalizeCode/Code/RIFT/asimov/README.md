
RIFT asimov interface, attempting plugin form.
Based on 
* https://git.ligo.org/deanna.fernando/asimov/-/blob/review/asimov/configs/rift.ini
* https://git.ligo.org/deanna.fernando/asimov/-/blob/review/asimov/pipelines/rift.py?ref_type=heads

See related documentation and examples in 
* https://asimov.docs.ligo.org/asimov/master/pipelines-dev.html
* https://git.ligo.org/asimov/pipelines/gwdata/-/blob/master/datafind/asimov.py

Compatibility notes
-------------------

With ASIMOV versions that provide ``PESummaryPipeline``, RIFT retains the
legacy automatic PESummary completion job.  ASIMOV 0.7 and newer manage
PESummary as a separate postprocessing analysis, so RIFT marks the PE analysis
finished and does not submit a duplicate postprocessing job.

``Rift.collect_assets(absolute=True)`` publishes the ``rift-assets/v1``
contract for separate postprocessing adapters: samples (always a list), the
RIFT configuration, PSDs, calibration envelopes, likelihood products, and
basic event/analysis provenance.  Consumers should tolerate additional keys.

Rimsky integration
------------------

The ``rift-rimsky-analysis`` command generates a RIFT follow-up document for
Rimsky's ``sample_sink.asimov_configuration`` hook. It bootstraps from the
PESummary metafile produced by Rimsky's online Bilby analysis and normalizes
Rimsky's underscore-separated prior names for the RIFT template. See
``RIFT/rimsky/README.md`` for configuration and operational details.

### Low-mass transverse-spin RF coordinates

The pipeline option `--rf-transverse-spin-coordinates` (helper or
`util_RIFT_pseudo_pipe.py`) takes `off`, `auto`, `geometric4` or
`geometric4-phase-excess`. An active mode changes only the fit basis of fresh RF
stages that fit the native `delta_mc, mu1, mu2, chiMinus, s1x, s1y, s2x, s2y`
basis; reduced or aligned stages keep their coordinates. Sampling coordinates,
spin priors, likelihoods, ILE and physical products are unchanged.

The bundled Asimov template uses `auto`: eligible precessing BBH analyses with a
reliable native **detector-frame** chirp-mass estimate strictly below 20 solar
masses select the four-input `geometric4` chart described below. Missing or
placeholder mass estimates and unsupported analyses keep the existing recipe.
Override under `sampler.cip`:

```yaml
sampler:
  cip:
    transverse spin coordinates: "off"  # off, auto, geometric4, or geometric4-phase-excess
```

YAML boolean `false` (including an unquoted YAML 1.1 `off`) means `off`;
boolean `true` means `geometric4`. Quoted string modes avoid YAML ambiguity.

Geometric4 uses the actual ILE spin reference frequency (`engine.fref` in an
INI, otherwise the helper's template reference). It does not transport spins or
replace the physical prior.

The earlier `physics3` mode is retired and refused at every layer. It appended
three scalar features to the eight native coordinates, so CIP fit 11
coordinates for 8 degrees of freedom. This RF transverse mode requires a nonredundant fit basis. Other CIP
configurations, including matter fits, can use more fit features than sampled
coordinates.

If a pipeline option later rewrites an activated stage so that it no longer
fits the native basis (`--cip-internal-use-eta-in-sampler`,
`--use-quadratic-early`), the DAG build fails; set the option to `off`.

### Four-input Geometric4 chart

Set `sampler.cip.transverse spin coordinates: "geometric4"` (or pass
`--rf-transverse-spin-coordinates geometric4` to the helper/pipeline).
This replaces the four Cartesian transverse *fitting* inputs by total transverse
angular-momentum radius `|S_perp|/M^2`, total-spin azimuth in the L plane, and two signed
sum-frame residuals. It requires exactly the native eight-coordinate RF basis
`delta_mc, mu1, mu2, chiMinus, s1x, s1y, s2x, s2y`; reduced stages remain
unchanged. No physical sampling coordinate, prior, waveform, frame or Jacobian
changes. Zero total transverse spin uses azimuth zero, and the angular seam
remains; near zero total transverse spin the two residuals also flip sign with the azimuth.
At detector chirp mass of 20 or more, enabling either mode also switches every
helper stage to the mu1/mu2 aligned-phase basis. The separate opt-in `geometric4-phase-excess` replaces only that radius by
`(J-|J_parallel|)/L_N`, preserving the earlier tested H variant. This phase excess
differs from `(J-J_parallel)/L_N` when `J_parallel < 0`.
The raw radius has no aligned-spin/J rescaling. Historical phase-excess
fit results do not establish performance of this raw-radius option.
The `auto` policy selects `geometric4` for eligible low-mass analyses.
This is an experimental representation, not an end-to-end accuracy claim.


Eccentric posterior bootstrap
----------------------------

Based on Lange's upstream PR #186, eccentricity and mean anomaly supplied by a
posterior are retained when constructing the XML bootstrap grid. This includes
zero and tiny eccentricities. With an eccentric analysis, only missing
coordinates are synthesized: eccentricity is uniform between the configured
``sampler.force-ecc-min`` / ``sampler.force-ecc-max`` (defaults 0 / 0.1), and
missing mean anomaly is uniform on [0, 2 pi). Sampled coordinates take precedence
over these fallback bounds.

Both PESummary HDF5 metafiles and native Bilby JSON (including .json.gz) / HDF5
result files work through ``scheduler.bootstrap file`` or a dependency publishing
one posterior file. Native results require Bilby installed in the RIFT setup
environment. PESummary input requires ``dataset`` for multiple analysis labels.
For example::

    scheduler:
      bootstrap file: /path/to/bilby_result.json
      bootstrap size: 2000
    likelihood:
      assume: [eccentric, nonprecessing]
    sampler:
      force-ecc-min: 0.001
      force-ecc-max: 0.3

Bilby's detector-frame masses, Cartesian or standard polar spin names,
``eccentricity`` and ``mean_anomaly`` / ``mean_per_ano`` / ``meanPerAno`` are
recognized. No-spin eccentric posteriors need no synthetic spin columns.
True/relative anomaly is not treated as mean anomaly. Source
coordinates must refer to the target waveform reference frequency; they are
copied, not evolved between frequencies. When a nonzero eccentricity posterior
supplies a reference frequency, including native Bilby waveform metadata, a
mismatch with ``waveform.reference frequency`` is rejected. Conversion errors
stop the build instead of proceeding without a valid bootstrap grid.
