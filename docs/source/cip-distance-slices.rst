###########################################
CIP and distance-slice operator workflow
###########################################

Use this workflow when an iterative RIFT analysis must both construct the next
intrinsic grid with CIP and retain the distance dependence of the
extrinsic-marginalized likelihood.  CIP consumes ILE likelihood evaluations and
exports posterior samples for the next iteration.  Distance-slice export is an
optional ILE product: it evaluates the extrinsic likelihood at fixed luminosity
distances and writes one ``.dslice`` table per event.

The two products answer different questions.  A CIP posterior is a draw over
intrinsic coordinates used to advance or finish the iterative analysis.  A
distance-slice table preserves fixed-distance likelihood information for later
inspection or re-marginalization.  Do not substitute one for the other.

Pipeline data flow
==================

The relevant stages and products are:

.. code-block:: text

   intrinsic grid
       |
       v
   integrate_likelihood_extrinsic_batchmode (ILE)
       |-- marginalized likelihood rows --> consolidated_*.composite
       `-- optional fixed-distance rows --> <output>_<event>_.dslice
                                              |
                                              `-- diagnostics / later re-marginalization

   consolidated_*.composite
       |
       v
   util_ConstructIntrinsicPosterior_GenericCoordinates (CIP)
       |-- posterior samples --> next intrinsic grid
       `-- final posterior --> downstream analysis

The generated ``ILE.sub`` and ``CIP.sub`` files are the first places to verify
the actual executable arguments.  See :doc:`using-pipeline` for the normal
run-directory layout and :doc:`executables/integrate_likelihood_extrinsic_batchmode`
and :doc:`executables/util_ConstructIntrinsicPosterior_GenericCoordinates`
for the command references.

CIP export contract
===================

CIP fits the marginalized likelihood over intrinsic coordinates, samples the
fit with the requested prior, and exports the samples used by the next stage.
The export uses systematic resampling of its weighted cache.  This keeps the
expected count for cache point :math:`i` equal to :math:`N w_i / \sum_j w_j`
and shuffles the returned indices, so truncating the draw does not privilege
high-weight rows at the front.

For the final non-Gaussian CIP argument group, pipeline generation adds
``--posterior-unique-draw``.  That option caps the delivered sample count at

.. math::

   \left\lfloor \frac{\sum_i w_i}{\max_i w_i} \right\rfloor,

the largest systematic-resampling draw that can remain duplicate-free.  The
delivered count may therefore be smaller than ``--n-output-samples``.  Treat
that as honest undersupply, not as a failed request.  Inspect the generated
``+annotation_export.dat`` sidecar, whose columns are ``n_requested``,
``n_delivered``, ``n_distinct``, and ``unique_draw_bound``.

Intermediate groups permit duplicate cache indices so that a requested draw
remains statistically fair even when it exceeds the unique bound.  A terminal
``G`` (Gaussian-resampling) group is not given the flag because that executable
does not accept it.  If unique final samples are required, do not end the CIP
schedule with a ``G`` group.  Convergence thresholds use distinct rows rather
than raw row count only when ``convergence_test_samples.py`` uses ``js_lame``
with ``--js-lame-auto-threshold``.  Other convergence methods do not provide
that duplicate-safety guarantee; see
:doc:`executables/convergence_test_samples`.

Enabling distance-slice export
==============================

Distance-slice export is disabled by default.  It runs only when all of the
following are true:

* ``--export-distance-slices K`` specifies a positive number of total slices;
* ``--internal-use-lnL`` is active;
* ILE has an output file; and
* distance marginalization is not active.

A representative ILE fragment is:

.. code-block:: bash

   integrate_likelihood_extrinsic_batchmode \
       ... \
       --internal-use-lnL \
       --export-distance-slices 10 \
       --distance-slice-wing-nmax 20000 \
       --distance-slice-wing-neff 30

Pass these options through the mechanism that owns ILE arguments in your
pipeline, then inspect ``ILE.sub`` before submission.  Do not copy the fragment
as a complete command: data, PSD, waveform, intrinsic-grid, and output options
remain analysis-specific.

Choosing core and fresh slices
------------------------------

The default mixed strategy divides ``K`` into a cheap importance-reweighted
core and fresh wing integrations:

* ``--n-distance-slice-core`` defaults to ``ceil(0.6*K)`` when set to zero;
* ``--n-distance-slice-wing`` defaults to the remaining rows when set to zero;
* core centers are fixed posterior-distance quantiles; and
* fresh wing centers extend beyond the core toward a default target drop of
  seven log-likelihood nats.

The core reuses the main ILE sampler's angular samples.  It is reliable only
when those samples form a healthy importance sample at each fixed distance.
At low main-loop effective sample count, especially with GMM below 50, use
``--distance-slice-all-fresh`` or increase the main integration budget.  ILE
also forces all-fresh mode when its stored rows are already posterior-resampled;
reweighting those rows again would double-count their importance weights.

All-fresh mode independently integrates the angular variables at every slice.
Its centers remain posterior-distance quantiles, but the precision of each row
comes from that row's own fixed-distance integration.  Add
``--distance-slice-randomize`` only with all-fresh mode when randomized
posterior quantiles are intended.  With ``K=1`` this draws one distance from
each intrinsic point's distance posterior instead of always choosing its
median.  Seed the ILE process with ``--seed`` when reproducible randomized
placement and Monte Carlo draws are required.

Fresh-slice budgets and safe parameterization
---------------------------------------------

The fresh integration defaults are:

* ``--distance-slice-wing-nmax 20000`` per slice;
* ``--distance-slice-wing-neff 30`` per slice; and
* ``--distance-slice-chunk`` unset, inheriting ILE's ``--n-chunk`` (default
  ``10000``).

The sample budget is block-granular, not a strict cap.  Adaptive-volume
integration checks the budget before drawing a whole block.  Keep ``nmax`` a
whole multiple of the resolved chunk size when predictable cost matters.
An inherited chunk below ``10000`` is raised to ``10000`` because smaller
blocks do not saturate the adaptive-volume threshold sample.  An explicitly
set ``--distance-slice-chunk`` is honored, must be positive, and can override
that protection.  Values below ``10000`` receive a stability warning; values
above ``15000`` receive a memory/quality warning.  Increase resource requests
after measuring the chosen waveform, detector network, and backend rather than
assuming that a larger block is always better.

The mixed strategy skips fresh wings when the peak core likelihood is below
``--distance-slice-skip-threshold`` (default ``1.0`` nat), treating the event as
effectively uninformative.  ``--distance-slice-wing-delta-lnL`` defaults to
``7.0`` nats for parabolic wing placement; a degenerate fit falls back to
log-uniform placement over the allowed distance range.  Review warnings and
the written method column before interpreting sparse or fallback placement.

CPU and GPU boundaries
======================

Fresh fixed-distance slices use the same array backend as the adaptive-volume
sample block.  On a CuPy-enabled ILE run, angular arrays are clipped and the
distance array is constructed on the device before the likelihood call; only
the likelihood vector and final summary values cross to the host for reduction
and output.  On a NumPy run, the same operations remain on the CPU.

The sampler deliberately retains a compatibility fallback: if the supplied
likelihood rejects device arrays with ``TypeError`` or ``ValueError``, it
retries on host arrays and remembers that choice for later blocks.  A run that
falls back is valid, but its transfer and memory profile differs from a
device-native run.  Do not infer GPU execution merely from requesting a GPU;
check the job environment and logs.  The focused regression tests skip their
CuPy assertions on hosts without a working GPU, so a CPU-only test pass is not
GPU validation.

Output schema and diagnostics
=============================

Each ``<output>_<event>_.dslice`` file has one row per intrinsic point and
slice distance.  The structured fields include:

``lnL``
   Fixed-distance, extrinsic-marginalized log likelihood on the same absolute
   overflow-corrected scale as the main ILE result.

``sigmaL``
   Monte Carlo standard error in log-likelihood space.

``neff`` and ``ntotal``
   Effective and total samples used by that row.  A non-finite likelihood,
   low ``neff``, or exhausted block-granular budget requires investigation.

``method``
   ``0`` for an importance-reweighted core row and ``1`` for a fresh
   fixed-distance integration.

``dist`` and ``ln_prior_d_sampling``
   Slice distance and the sampling-distance log prior evaluated there.

Intrinsic-coordinate columns are repeated on every slice row.  Use the method,
uncertainty, and sample-count columns together; a visually smooth sequence of
``lnL`` values is not by itself evidence of adequate Monte Carlo precision.
The table can be re-marginalized over distance with
``RIFT.misc.distance_slices.reconstruct_marginal_lnL``, which uses trapezoidal
integration across the available slice points.  That reconstruction is only as
reliable as the distance coverage and row uncertainties; it does not repair
missing tails or under-resolved slices.

Before submitting a large workflow, verify one representative ILE job
interactively and check:

#. the expected number of ``.dslice`` rows was written;
#. core/fresh counts and forced-all-fresh warnings match the intended strategy;
#. every ``lnL`` and ``sigmaL`` is finite;
#. ``neff`` and ``ntotal`` agree with the requested budgets;
#. chunk, low-information, fallback-placement, and GMM warnings are understood;
#. the CIP annotation reports the requested, delivered, and distinct counts;
   and
#. any duplicate-safety claim is limited to ``js_lame`` with
   ``--js-lame-auto-threshold``; other convergence methods require their own
   interpretation.

Reproducibility checklist
=========================

Archive the following together:

* the exact RIFT Git SHA and environment lock;
* generated ``ILE.sub``, ``CIP.sub``, CIP argument list, and test arguments;
* the ILE seed and all distance-slice options;
* ``.composite``, ``.dslice``, posterior, and ``+annotation_export.dat``
  products; and
* job logs containing backend selection, warnings, sample budgets, and software
  versions.

Fixed quantile centers are deterministic for fixed sampler rows.  Randomized
centers and Monte Carlo integration are not reproducible without control of the
random-number state.  Even with a seed, CPU and GPU libraries or different
dependency versions may not produce bitwise-identical draws.  Compare results
against their quoted Monte Carlo uncertainty and record the execution backend.
