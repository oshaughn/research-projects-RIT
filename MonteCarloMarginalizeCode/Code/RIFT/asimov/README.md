
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

### Four-input Geometric4 chart (opt-in)

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
The `auto` policy selects geometric4 for eligible precessing BBH analyses
with detector chirp mass below 20 solar masses. A YAML `true` explicitly
selects geometric4 at any mass.

The earlier `physics3` mode is retired and refused at every layer. It appended
three scalar features to the eight native coordinates, so CIP fit 11
coordinates for 8 degrees of freedom. This RF transverse mode requires a nonredundant fit basis. Other CIP
configurations, including matter fits, can use more fit features than sampled
coordinates.
This is an experimental representation, not an end-to-end accuracy claim.
