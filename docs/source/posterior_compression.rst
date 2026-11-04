Posterior compression
=====================

RIFT can fit compact analytic representations to the joint posterior in
``extrinsic_posterior_samples.dat``. Each row contains intrinsic parameters and
their paired luminosity distance. The fitter uses these rows and their posterior
weights directly, preserving mass, spin and distance correlations.

The final CIP stage supplies the intrinsic prior, including the spin prior.
Earlier CIP stages can use different spin priors to explore the likelihood;
those exploratory priors must not be divided out of the final posterior.
The distance prior comes from the ILE/extrinsic configuration. These priors
are resolved from the production INI and checked against ``args_ile.txt`` and
the executed final stage selected by BasicIteration. New workflows save this
selection in ``auto_fit.final_cip_args.json``, bound to the stage-list hash.
An iteration limit can stop before the last listed CIP stage; that unexecuted
stage must not supply the prior.

A CIP posterior table does not supply the paired distances needed by these
fits. Compression does not read CIP posterior rows, join them to the extrinsic
samples or change the extrinsic samples' weights to match a CIP marginal.

Enable compression in a workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Append this section to the original high-level production INI:

.. code-block:: ini

   [rift-auto-fit]
   enabled = true
   component_counts = 1,2,4,8
   output_directory = auto_fit
   seed = 711
   max_iterations = 200
   validation_ks = .08
   request_cpus = 1
   request_memory_mb = 4096
   request_disk_mb = 2048
   max_runtime_seconds = 1800

Use the usual final extrinsic export options, such as ``--add-extrinsic
--batch-extrinsic --add-extrinsic-time-resampling``. Pseudo-pipe passes its
production INI to BasicIteration, which adds a bounded terminal compression
job after the joint extrinsic posterior is available. Omitting the section
leaves compression disabled. With ``targets`` omitted, final CIP spin
coordinates select both products for a precessing analysis and only
``aligned_distance`` for an aligned analysis. This follows the resolved run;
absence of ``assume-precessing`` alone does not imply alignment, since
pseudo-pipe can infer precession from the approximant. An explicit
``assume-nonprecessing`` override is checked against the final CIP coordinates.
Set ``targets = aligned_distance`` to request only that product in either case.

The default products are:

* ``precessing_source_redshift``: source chirp mass, eta, all six Cartesian
  spin components and redshift (nine dimensions).
* ``aligned_distance``: source chirp mass, eta, both aligned spin components
  and luminosity distance (five dimensions).

For precessing input, the aligned product marginalizes transverse spins under
the generating conditional prior. It does not impose an aligned waveform model.
Masses use solar masses, distances use Mpc, and spins retain the generating
L frame and reference frequency.

Apply to an existing run
~~~~~~~~~~~~~~~~~~~~~~~~

Use a RIFT installation containing the compression executable:

.. code-block:: console

   util_RIFT_compress_posterior.py --config /path/production.ini \
       --rundir /path/completed_run

The default data input is ``extrinsic_posterior_samples.dat`` under the run
directory. ``--posterior FILE`` selects another joint posterior table;
``--output DIRECTORY`` selects a new output directory. Relative paths resolve
under ``--rundir``. No new waveform calls or posterior sampling are performed.
The run's resolved argument files are needed to check the final prior and
automatically selected support; no CIP posterior sample file is required.
Legacy runs use the terminal ``CIP_prior.sub`` science arguments when available.
For a multistage legacy run without a saved terminal execution record, set ``final_cip_args = confirmed-final-args.txt`` in the auto-fit
section. This file must contain the science arguments of the actual terminal
CIP stage, confirmed from the generated workflow. Its hash is recorded. A
single-stage legacy run can use its sole listed stage. Remove legacy overrides
when a new workflow records its terminal stage automatically.

Fairdraw posterior rows have equal weights. For a weighted table, set
``weight_column`` in ``[rift-auto-fit]`` to its posterior-weight column.
Set ``group_column`` when the export records original intrinsic ancestry;
otherwise identical intrinsic rows share a group. Grouping controls training
and validation splits and does not reweight the posterior. Never apply earlier
importance weights again to fairdraw rows.

Existing output directories are preserved. A failed fit saves its candidate
and validation report and returns nonzero. Choose another output directory
for a new attempt. Missing distance columns, unresolved/custom priors and
unsupported post-reweighting contracts fail explicitly. Widened Cartesian
spin-radius cuts and unordered component-mass sampling with asymmetric
spin caps or cuts require separate prior adapters and are refused. Do not substitute
distance-slice likelihood rows or independent mass and distance marginals.

Optional pipeline diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the pipeline option to the production INI:

.. code-block:: ini

   [rift-auto-fit]
   enabled = true
   build-diagnostics = true

This adds two separate terminal jobs after fitting: draw a posterior from each
saved fit, then compare it with the original joint extrinsic posterior.
The fit executable does not run these jobs itself. Each job uses the existing
one-CPU resource and runtime settings. Local-universe memory/disk requests
are advisory; thread and runtime controls bound execution. Optional ``diagnostic_samples = 20000``
and ``diagnostic_bins = 64`` set the draw count and histogram resolution.

The sampler draws the fitted likelihood times its recorded native retained
prior. This is the supported fitted posterior, so mixture sampling replaces
another numerical CIP integration. It preserves the final CIP spin prior and
ILE distance prior/cosmology. It does not apply a new population prior or
reconstruct omitted nuisance parameters. The aligned product therefore reports
only retained-spin diagnostics, even when generated from a precessing run.

For each target, ``diagnostics/posterior_samples.dat`` contains equal-weight
draws and standard derived coordinates. ``diagnostics/js.json`` and ``js.tsv``
compare those draws against the held-out original extrinsic groups, excluded
from training and tuning. Original posterior weights are preserved. Parameters
include detector chirp/component masses, eta, q, effective spin ``xi``, both
aligned spins, distance, source masses and redshift. Full-spin fits additionally
include Cartesian spins, magnitudes, transverse spins, ``chi_p``, tilts and
azimuths. Undefined zero-spin reference tilts are listed as unavailable.
``phi12`` follows RIFT's L-frame range of zero to two pi, with zero returned
when either transverse spin vanishes.

JS is the squared base-2 Jensen-Shannon distance, reported in bits from zero
to one. The estimator uses weighted histograms with common edges covering both
distributions, and fixed circular domains for azimuths. It shares RIFT's base-2
convention but differs from the convergence script's KDE estimator. Each
parameter also reports one same-fit null comparison with matching row counts
and weights. This estimates histogram sampling noise; it ignores correlations
within original intrinsic groups. The report includes reference group ESS.
No fixed JS acceptance threshold is imposed. Small 1D divergences do not certify
joint correlations, arbitrary-prior tails or population integrals.

Standalone sampling and comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two tools can also be run independently after building a fit:

.. code-block:: console

   util_RIFT_sample_compressed.py \
       --model /run/auto_fit/aligned_distance/model.json \
       --output /run/auto_fit/aligned_distance/diagnostics/posterior_samples.dat

   util_RIFT_compare_compressed.py \
       --fit-directory /run/auto_fit --target aligned_distance \
       --samples /run/auto_fit/aligned_distance/diagnostics/posterior_samples.dat \
       --output /run/auto_fit/aligned_distance/diagnostics/js.json

Sampling accepts ``--n-samples`` and ``--seed``; comparison accepts ``--bins``.
Comparison outputs must end in ``.json``; the companion table uses ``.tsv``.
Existing outputs are preserved. Comparison checks the fit manifest and hashes
of the saved model, original posterior and generated samples. Retain the
original posterior and fit manifest when moving products to another host.

Cosmology and likelihood interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When omitted from the auto-fit section, cosmology and distance-prior choices
inherit from the analysis. Current RIFT cosmological distance priors use
Planck15. Conflicting declarations and inconsistent exported source columns
are rejected. The stored model retains its generating prior and cosmology.

The fitter estimates the native joint posterior density and divides by its
retained-coordinate prior to evaluate a scalar likelihood, up to one fixed
event constant. Omitted nuisance parameters retain the generating conditional
prior. Native-recovery validation does not certify arbitrary-prior tails,
population inference or absolute event evidence.

The stored chart uses detector masses and distance. Source-frame evaluation
substitutes ``Mc_detector = (1+z) Mc_source`` and ``dL = dL(z)`` without a
likelihood Jacobian. Densities and integration measures include their Jacobians.
An alternative evaluation cosmology changes this coordinate map without
changing the generating distance prior.

Efficient evaluation
~~~~~~~~~~~~~~~~~~~~

Load the model and cosmology context once, then evaluate arrays:

.. code-block:: python

   from RIFT.compression import CompressedFit, CosmologyContext
   from astropy.cosmology import Planck18

   fit = CompressedFit.load("auto_fit/aligned_distance/model.json")
   ctx = CosmologyContext.from_astropy(Planck18, distance_max=2500.)
   logL = fit.log_likelihood(points, mass_frame="source",
                            radial_coordinate="luminosity_distance",
                            cosmology=ctx)

Mixture Cholesky factors and checked cosmology interpolators are cached.
Cosmology tables cover the declared distance support and reject extrapolation;
set ``distance_max`` to cover the model's distance range.
For repeated integrals, reuse a proposal bank:

.. code-block:: python

   bank = fit.integration_bank(n=65536, seed=73, cosmology=ctx)
   log_integral, diagnostics = bank.integrate_log_density(log_population_density)

The callback receives source component masses, retained spins and redshift.
Bank weights include the physical measure and proposal density. Keep their
normalization fixed between population models and inspect the reported
importance ESS.
