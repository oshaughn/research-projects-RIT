
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

### Low-mass transverse-spin RF prototype

The pipeline remains opt-in: pass `--rf-transverse-spin-coordinates physics3`
to `helper_LDG_Events.py` or `util_RIFT_pseudo_pipe.py`. The helper enables this
only in fresh RF stages fitting both full Cartesian transverse spins. Reduced
or aligned stages retain their existing coordinates. It appends three tested
L-frame cone, geometric phase-deficit, and precession-torque scalars; it retains
all native fitting coordinates, including `mu1`, `mu2` and the four transverse
components. Sampling coordinates, spherical spin priors, likelihoods, ILE and
recorded physical products are unchanged. No GP or GPU dependencies are added.

The bundled Asimov template uses `auto`: eligible precessing BBH analyses with a
reliable native **detector-frame** chirp-mass estimate strictly below 20 solar
masses opt in. Missing/placeholder mass estimates and unsupported analyses keep
the existing recipe. Override under `sampler.cip`:

```yaml
sampler:
  cip:
    transverse spin coordinates: off  # off, auto, or physics3
```

The scalars use the actual ILE spin reference frequency (`engine.fref` in an INI,
otherwise the helper's template reference). They do not transport spins or
replace the physical prior. This is a prototype supported by controlled existing
grid comparisons, not a universal recovery guarantee; S250830bp remains a
separate partially improved case, and known sky/data/prior discrepancies require
separate assessment. Coordinate activation changes neither stopping criteria
nor posterior quotas.
