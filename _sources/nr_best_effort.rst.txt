Evidence-weighted NR intrinsic and extrinsic posteriors
=======================================================

``util_NRBestEffortPosterior.py`` constructs a best-effort posterior on a discrete
NR catalog using existing ILE likelihood results. It can also combine saved
conditional extrinsic posteriors. It does not require a continuous CIP fit, and
planning or combining ASCII files only requires NumPy. XML grid export additionally
requires the usual RIFT/lalsuite environment.

Installed instructions and modern catalog compatibility
-------------------------------------------------------

The RIFT wheel and source distribution include the CLI, this document, and
``demo/nr_best_effort/{README.rst,bootstrap_existing_run.py}``. Installed examples
are under ``sys.prefix/share/RIFT/nr-best-effort``. The utility is installed in
``sys.prefix/bin``. No NR waveform data or site-specific archives are bundled.

The ``catalog-check`` subcommand checks native intrinsic labels with the public
``WaveformModeCatalog(group,param,metadata_only=True).P`` interface::

   util_NRBestEffortPosterior.py catalog-check --manifest nr-best-effort/manifest.json \
     --group RIT-Five --output catalog-check.json

Configure ``NR_BASE`` and the module path for the same catalog revision used by
ILE. Both a direct ``NRWaveformCatalogManager3`` import and RIFT's
``nrcatalog.compat_nrwf`` backend are supported when the latter wraps an operational
waveform constructor. A registry-only backend whose constructor raises
``NotImplementedError`` cannot supply an ILE waveform. Catalog metadata checking
uses no waveform I/O and does not resolve multiple resolutions/extractions; its
report lists all matching identities and flags ambiguity. It preserves component
ordering rather than independently folding q to its reciprocal. Eccentricity
must retain the native NR labels, not a frequency-remapped CIP coordinate.

For an extrinsic-only followup of one existing run set, see the installed
``README.rst`` and ``bootstrap_existing_run.py``. They prepare an intrinsic
posterior grid, retain the source ILE data/prior/frequency settings, and require
one conditional draw per scheduled intrinsic draw. They refuse missing events,
wrong intrinsic coordinates, changed source hashes, and existing output paths.
A separate smoke output directory prevents short I/O checks entering the result.

What is already available
-------------------------

The ``rift_O4d`` audit at commit ``7909cbdc95763dc4b81b4ae5dab9dedba6dd9f86``
(2026-10-22) found several related workflows:

* ``create_event_nr_pipeline_with_cip`` and ``demo/nr_w_rift`` refine NR mass
  spokes and finish with CIP. Its ``--last-iteration-extrinsic`` stage follows
  the CIP intrinsic posterior; it does not implement a discrete evidence mixture
  over the original NR simulations.
* ``util_NRBestOfIndex.py`` ranks best mass/likelihood matches, optionally fitting
  peaks. A peak marginal likelihood is not a simulation's mass-integrated evidence.
* ``RIFT.misc.spokes`` groups/refines mass curves and consolidates repeated ILE
  estimates. ``weight_simulations`` mainly combines repeated Monte Carlo
  estimates; neither supplies an end-to-end catalog posterior command.
* ``demo/reconstruct/nr`` generates conditional extrinsic samples and strain
  reconstructions for one fixed simulation. It is useful for the ILE flag recipe,
  but does not supply the across-simulation mixture.

Historical reference and recovery
---------------------------------

Healy, Lousto, Lange and O'Shaughnessy, PRD 102, 124053 (2020),
`arXiv:2010.00108 <https://arxiv.org/abs/2010.00108>`_, Sec. II.E, describes
retaining RIFT samples for up to 100 high-ranking simulations to estimate
extrinsic posteriors. The simulation set supplies a discrete intrinsic prior;
the analysis does not correct its uneven coverage. Table III states the coverage
limitation and Fig. 4 compares GW170814 distance/inclination and sky inference.
The paper does not specify an exact mass-evidence quadrature algorithm. The
explicit quadrature below is a new implementation, not a recovered historical
script or an assertion of bit-for-bit reproduction of that analysis.

A bounded, read-only CIT search on 2026-10-22 recovered:

* ``/home/jacob.lange/unixhome/Projects/RIT-Catalog3-BFs/GW170814/``:
  ``all.net``, ``aligned/fitting.sh``, ``aligned/integral_result.dat``, and
  additional prior variants from Aug/Sep 2020. Other event directories also remain.
* ``/home/jacob.lange/resample_extrinsic.sub`` and ``postproc_extrinsic.sub``:
  the old generic resampling/conversion chain. Their existence does not verify
  how the historical NR mixture was assembled.
* ``/home/jacob.lange/for-NR-distance.txt``: an incomplete workflow note.
* ``/home/richard.oshaughnessy/unixhome/PersonalNRArchive/bin/``:
  NR metadata/comparison tools, without the sought posterior assembly script.
* An older project Makefile refers to
  ``/home/jacob.lange/lsc_Archive/PENR_methods/PENR_TheEvent/Python/dump-rankings-latex.py``;
  ``lsc_Archive`` was absent from both mounted homes. Tape/archive contents were
  not searched or recalled. No original NR mixture script was recovered.

Statistical definition
----------------------

For simulation/spoke ``s``, mass ``M``, and extrinsic coordinates ``theta``, use

.. math::

   Z_s = \int L_{\rm marg}(s,M) p(M\mid s)\,dM,
   \qquad P(s\mid d) \propto \pi_s Z_s,

   p(\theta\mid d) = \sum_s P(s\mid d)
       \int p(\theta\mid s,M,d)\,p(M\mid s,d)\,dM.

The planner uses composite trapezoid quadrature on the actual, possibly
nonuniform, mass nodes. Each component's unnormalized weight is
``pi_s * delta_M * p(M|s) * L_marg(s,M)``. Calculations stay in log space.
Uniform discrete spoke priors are the default. The mass prior is
``p(M) proportional to M**power`` on one explicitly supplied detector-frame
mass interval, normalized on that same interval for every spoke.

The default power is zero (uniform total mass). Power one gives the mass part
of a prior uniform in component masses at fixed mass ratio: the full continuous
Jacobian also contributes a mass-ratio factor, which belongs in the simulation
prior if that is your target. No catalog-density correction is inferred.

A physical spoke is identified using ``m2/m1``, six spins, and optionally
eccentricity and mean periastron anomaly. Coordinates are grouped with maximum
within-spoke span ``10**(-digits)`` (default digits 4), without transitive chaining.
Sorting makes grouping independent of row order; ambiguous nearby spokes are
rejected. This avoids the decimal-bin boundary splitting found in archived
GW170814 data when component masses were rounded to five decimal places. Manifest
signatures are rounded representative coordinates for identifying these groups. Different simulation
resolutions/catalog aliases with identical coordinates are not separate prior
atoms. Select a consistent waveform resolution/extraction before consolidation;
never merge physically different waveforms solely because their coordinates agree.
For distinct simulation identities with the same coordinates, supply an explicit
component manifest instead of the physical-spoke planner.

Bootstrap from a completed run
------------------------------

1. Use the run's cleaned, consolidated ``all.net`` (or ``.composite``). Supported
   layouts are ``index,m1,m2,6 spins,[eccentricity,meanPerAno],lnL,sigma/L,ntot,neff``.
   Enable ``--eccentric`` and, when appropriate, ``--mean-per-ano``. Distance,
   EOS, tidal, and other extended layouts are rejected. Repeated mass nodes are
   rejected by default: use ``util_CleanILE.py`` with the correct layout options
   first. Older rounded tables can retain near-duplicate masses on the same spoke
   even after cleaning individual intrinsic coordinates. For independent replicas
   of the same waveform, ``--coalesce-repeats`` combines these masses (rounded to
   ``digits`` places) by inverse absolute MC variance in likelihood, and records
   how many rows were merged. It requires positive relative uncertainties. Do not
   use it for distinct waveform resolutions/extractions. Monte Carlo uncertainty
   does not set the prior weight across different intrinsic points.

2. Plan the mixture, for example on a common 120--180 solar-mass interval::

      util_NRBestEffortPosterior.py plan \
        --composite /path/to/run/all.net --mass-range 120 180 \
        --mass-prior-power 0 --template-xml /path/to/run/overlap-grid-0.xml.gz \
        --n-output-samples 3000 --seed 17 --output-dir nr-best-effort

   The output directory must be new. Inspect ``manifest.json``: it includes
   per-spoke log evidence, posterior probabilities, maximum relative ILE errors,
   mass support, source hash, and per-cell weights. Large MC errors, a posterior
   against a mass boundary, or a sparse mass grid need further ILE/refinement.
   Repeat with a refined grid to check quadrature convergence; a successful
   command is not a scientific convergence assessment.

   By default every spoke must have nodes at both prior boundaries. To bootstrap
   older runs with different mass coverage, ``--allow-partial-mass-range`` uses
   only observed nodes inside the common interval and records a warning. It keeps
   the global mass-prior normalization: it does not renormalize each simulation
   to its own observed mass range. These are truncated evidences, with unmeasured
   tails omitted. Extend spokes before treating them as complete evidences.

   Custom simulation prior weights can be provided with ``--simulation-priors``:
   a JSON object keyed by the exact ``simulation`` signatures in a first plan's
   manifest. Every spoke must have one finite positive weight and no extra keys.
   Weights are normalized across the selected catalog.

3. The planner always produces ``intrinsic_posterior_samples.dat`` and an aligned
   ``intrinsic_component_ids.dat``. These are draws from the discrete mass-grid
   approximation. With ``--template-xml`` it also writes:

   * ``intrinsic-posterior.xml.gz``: evidence-weighted intrinsic draws, suitable
     for a normal conditional extrinsic followup, drawing the same number of
     extrinsic samples per intrinsic point. Such a followup already incorporates
     the mixture weights and must not be weighted a second time.
   * ``quadrature-grid.xml.gz``: one event per manifest component, with XML row
     number equal to component id. Use this for separate conditional followups
     and the manifest-based combination described next.

   An old refined XML can carry intrinsic labels that differ from those actually
   recorded by ILE (including older eccentricity labels). If XML is used only to
   supply common run settings, explicitly add ``--template-settings-only``:
   the first XML row supplies settings, and every intrinsic coordinate is
   overwritten from the composite. This does not remap the composite's labels;
   validate them with ``catalog-check`` against the intended archive. The
   one-run demo uses this mode and retains the exact source ILE arguments.

   For a posterior-grid followup, add ``--posterior-xml-only`` to skip the
   potentially large quadrature XML. A smaller refined template from the same
   run can be used if it contains every selected spoke; missing or ambiguous
   matches are rejected.

   Both XML files preserve settings from a representative point of the matching
   original-grid spoke, with masses replaced by the planned masses. Use a template
   with consistent reference frequency, time, and NR conventions. NR waveform
   lookup flags remain the responsibility of the ILE command.

Reuse saved conditional extrinsic samples
-----------------------------------------

A modern ILE fairdraw flag is not by itself proof of valid posterior sampling.
Check the ``*_integrator_status.json`` report: a collapsed live volume invalidates
its output even if XML serialization succeeded. Attach these reports in each
component's optional ``status_files`` list to make ``combine`` reject collapsed,
missing, or malformed status records and record their hashes. Without attached
status files, ``sample_kind=fairdraw`` is the operator's assertion that the saved
conditionals were validated (necessary for older runs with no status reports).
The one-run demo requires a noncollapsed modern status for every event, both at
conversion and at final assembly, and never silently omits a failed event.

For each component, put converted ASCII fair posterior draw files in its manifest
``samples`` list. Paths are relative to the manifest; absolute paths also work.
Pool only independent conditional runs of the same intrinsic point and analysis
configuration. The combiner checks common columns and all planned intrinsic
coordinates; it refuses a missing finite-weight component, reused file paths,
nonfinite samples, or mismatched coordinates. It does not infer the event, PSD,
distance prior, calibration setup, reference frequency, or waveform conventions
from ASCII tables: verify those against the original run arguments.

For a new extrinsic-only rerun, retain the original event/data/PSD/NR lookup
arguments and replace grid, event selection, and output arguments. For component
zero, the additional/replacement arguments are::

   --sim-xml nr-best-effort/quadrature-grid.xml.gz --event 0 --n-events-to-analyze 1
   --time-marginalization --vectorized --gpu --force-xpy
   --save-samples --fairdraw-extrinsic-output --resample-time-marginalization
   --output-file cell-0

These are fragments, not a runnable command without the original data and NR
arguments. Use the maintained NoLoop path and confirm it in the ILE log. The
fair-draw flags make the XML rows equally weighted and restore per-row coalescence
times. Convert the produced XML to ASCII with the existing converter::

   convert_output_format_ile2inference cell-0_0_.xml.gz > cell-0.dat

Add ``--export-eccentricity`` and ``--export-meanPerAno`` to the converter as needed
to retain planned intrinsic fields. Repeat for every component, setting its
event id and output name. Several independent files can appear in one component's
``samples`` list. Do not pass already fair-drawn samples through the old importance
resampler, or apply ``exp(lnL)*p/ps`` to them again. Raw importance samples need
conversion to a correct conditional fair posterior first; ``combine`` deliberately
does not accept them. Samples with time, phase, distance, or polarization still
marginalized cannot support joint inference for that missing coordinate.

After filling every component's ``samples`` list::

   util_NRBestEffortPosterior.py combine --manifest nr-best-effort/manifest.json \
     --n-output-samples 3000 --seed 18 --output extrinsic_posterior_samples.dat

The output uses the input column names and is compatible with the usual posterior
plotting tools. It also writes aligned component ids and a provenance JSON file
with manifest/file hashes and per-component available/drawn/unique counts.
Sampling is with replacement: 3000 exported rows do not imply 3000 independent
or effective samples. Each component's weight is independent of how many saved
conditional samples it has. The conditional pools still need adequate sampling.

Already mass-marginalized per-simulation posteriors
---------------------------------------------------

For existing runs that already contain a conditional joint ``p(M,theta|s,d)``,
write a minimal manifest instead of using ``plan``::

   {
     "schema_version": 1,
     "sample_kind": "fairdraw",
     "components": [
       {"id": 0, "simulation": "RIT-0001", "log_weight": 12.0,
        "samples": ["RIT-0001-posterior.dat"]},
       {"id": 1, "simulation": "RIT-0002", "log_weight": 11.5,
        "samples": ["RIT-0002-posterior.dat"]}
     ]
   }

Here ``log_weight = log(pi_s) + log(Z_s)`` includes the mass-integrated evidence
under the same mass prior as the conditional samples. Omit ``intrinsic`` because
masses vary within the conditional posterior. A manifest produced by ``plan``
instead has a fixed-mass component per node: do not replace its conditional
files with simulation-wide posteriors. Evidence normalizations must be comparable
between components; peak lnL values and sample counts are not valid substitutes.

Validation
----------

Run the fast analytic and CLI tests with::

   python MonteCarloMarginalizeCode/Code/test/test_nr_best_effort.py

They check known evidence ratios, nonuniform quadrature, explicit priors,
partial-support normalization, stability at lnL=1000, unequal conditional pool
sizes, reproducibility, missing/mismatched files, eccentric/anomaly grouping,
CLI planning/combination, and actual XML round trips when RIFT/lalsuite are present.
These tests do not evaluate an NR waveform or establish event-specific convergence.

The planner was also exercised on the recovered CIT GW170814 ``all.net``
(26,087 rows, 13 columns) after the current O4d ``util_CleanILE.py`` step, using
``--mass-range 40 80 --allow-partial-mass-range --coalesce-repeats``. It recovered
899 physical spokes and 25,957 mass cells, coalesced 29 repeated/near-repeated
rows, and marked 70 spokes with incomplete mass coverage. Simulation
probabilities summed to one. This checks real archived input compatibility; it
does not reproduce the historical paper posterior or validate the omitted mass
tails, waveform choices, or conditional extrinsic samples. The aligned run's
``integral_result.dat`` is a catalog-wide CIP result, not a table of per-simulation
evidences, and was not used as a component weight.
