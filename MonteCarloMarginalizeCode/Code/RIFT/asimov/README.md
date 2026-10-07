
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

The pipeline remains opt-in: pass `--rf-transverse-spin-coordinates physics3`
to `helper_LDG_Events.py` or `util_RIFT_pseudo_pipe.py`. The helper enables this
only in fresh RF stages fitting the tested `delta_mc, mu1, mu2, chiMinus` basis
and both full Cartesian transverse spins. This single option selects the native
aligned-phase CIP schedule before adding the scalars; the initial grid is not
changed by this schedule selection. If no fitting method was requested, the
activated option selects RF for CIP; an explicit non-RF method is not overridden
(`physics3` cannot be honored and fails explicitly). Reduced
or aligned stages retain their existing coordinates. It appends three tested
L-frame cone, geometric phase-deficit, and precession-torque scalars; it retains
all native fitting coordinates, including `mu1`, `mu2` and the four transverse
components. Sampling coordinates, spherical spin priors, likelihoods, ILE and
recorded physical products are unchanged. No GP or GPU dependencies are added.

The bundled Asimov template uses `auto`: eligible precessing BBH analyses with a
reliable native **detector-frame** chirp-mass estimate strictly below 20 solar
masses select the four-input `geometric4` chart described below. Explicit
`physics3` retains the scalar augmentation described above. Missing/placeholder
mass estimates and unsupported analyses keep the existing recipe. Override under `sampler.cip`:

```yaml
sampler:
  cip:
    transverse spin coordinates: "off"  # off, auto, physics3, geometric4, or geometric4-phase-excess
```

YAML boolean `false` (including an unquoted YAML 1.1 `off`) means `off`;
boolean `true` means `physics3`. Quoted string modes avoid YAML ambiguity.

The scalars use the actual ILE spin reference frequency (`engine.fref` in an INI,
otherwise the helper's template reference). They do not transport spins or
replace the physical prior.
If a pipeline option later rewrites an activated stage so that it no longer
fits the native basis (`--cip-internal-use-eta-in-sampler`,
`--use-quadratic-early`), the DAG build fails; set the option to `off`. This is a prototype supported by controlled existing
grid comparisons, not a universal recovery guarantee; S250830bp remains a
separate partially improved case, and known sky/data/prior discrepancies require
separate assessment. Coordinate activation changes neither stopping criteria
nor posterior quotas.

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
differs from `(J-J_parallel)/L_N` when `J_parallel < 0`; the regularization
epsilon used by physics3 is absent from both four-coordinate charts.
The raw radius has no aligned-spin/J rescaling. Historical phase-excess
fit results do not establish performance of this raw-radius option.
The `auto` policy selects `geometric4` for eligible low-mass analyses.
Select `physics3` explicitly to retain the scalar augmentation.
This is an experimental representation, not an end-to-end accuracy claim.
