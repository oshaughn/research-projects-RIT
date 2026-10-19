First batch mu puff with random transverse spins
===============================================

This is the XML-only O4c port of O4d PR 463. Enable with::

    util_RIFT_pseudo_pipe.py [usual precessing BBH arguments] \
        --internal-puff-mu-random-transverse \
        --rf-transverse-spin-coordinates geometric4

The new flag uses a covariance puff in ``mu1,mu2,q=m2/m1,s2z`` for the first
finite CIP batch, followed by independent RANDOM transverse-spin redraws.
An aligned grid has no transverse covariance, so transverse coordinates are
never included in that covariance. An atomic vectorized inverse reconstructs
both masses and aligned spins together. It supports physical negative mu1;
the relevant positive inverse target is ``mu1 - U02/U12 * mu2``. The mu
convention uses 200 Hz independently of waveform fref.

For each spin the transverse radius is uniform from zero to
``min(transverse_max,sqrt(1-siz**2))`` and azimuth is uniform on ``[0,2*pi)``.
Both spins are redrawn independently, even at puff-factor zero. This is an
exploration proposal distribution, not an isotropic-spin prior. Genuine zero
directions in the four-dimensional covariance remain zero; the new path does
not use the legacy singular-covariance fallback.

The flag enables O4c's existing ``internal_puff_transverse`` for later batches
and refinement subdags. It does not change that existing implementation or
import O4d's ASCII-grid machinery. Without the new flag, the
ordinary single-configuration puff settings and routing are unchanged.

``args_puff_list.txt`` pairs one puff line with each final ``args_cip_list.txt``
line, using identical iteration-count, ``G`` or ``Z`` prefixes. The first batch
count sets the discovery duration. Existing puff cadence and stop policy apply,
with the stop extended if needed to cover that first batch. Later batches use
the ordinary puff. A ``Z`` child inherits its stage's single puff configuration.
The first puff follows CIP-0 and is scored by ILE puff jobs in the next
iteration; it does not change the initial grid before ILE-0.

Direct BasicIteration users may pass ``--puff-args-list`` instead of
``--puff-args``, with a list paired line-for-line to ``--cip-args-list``.
Mismatched counts or prefixes are refused. ``args_puff.txt`` remains the
ordinary configuration for consumers of the legacy file.

Standalone proposal generation::

    util_ParameterPuffball.py --inj-file selected.xml.gz --inj-file-out puff \
        --puff-coordinate-system mu-random-transverse --puff-factor 1 \
        --random-seed 448 --fail-if-empty

Omit ``--parameter`` because the chart is fixed. The default makes one attempt
per input seed. ``--mu-puff-n-proposals N`` instead samples N parents uniformly.
The optional ``--mu-puff-transverse-max`` is in [0,1], default 1. Invalid inverse
rows and failed native cuts are discarded without clipping or retries. Other
waveform fields come from each parent, including tides and eccentricity; this
does not vary those nuisance coordinates. A ``<output>_mu_puff.json`` sidecar
records the proposal settings and attempted/accepted/exported counts.

Mass/spin downselection, duration bounds and fref/fmin overrides are supported.
Legacy coordinate, force-away, independent random override and
effective eccentric-phase options are refused in the standalone new mode.
The early pipeline stage preserves helper native cuts and scales, but omits
force-away exclusion: a nearby mass/aligned point can have a new transverse
orientation. Later stages retain the existing exclusion. The initial pipeline
prototype supports precessing BBH with BasicIteration; nuisance-coordinate
puffing and alternate/AMR builders are refused.

Core inverse/physical-bound, actual XML CLI and real DAG-routing tests accompany
the port. Production XPHM/geometric4 recovery qualification is tracked with the
O4d experiment. Proposal speed or geometric support alone is not evidence of
likelihood recovery; score all proposals with production ILE before inference.

General coordinate reflection
-----------------------------

The legacy puff also accepts repeatable ``--reflect-parameter NAME`` options.
Each name must be a puff ``--parameter`` with a finite, increasing
``--downselect-parameter-range`` (or the corresponding built-in range option).
For example::

    --parameter mc --reflect-parameter mc --mc-range '[20,40]'

Reflection folds excursions repeatedly into the interval, rather than clipping
or discarding points. Bounds use native coordinate units (solar masses for
mass coordinates); log-transformed coordinates are reflected after converting
back to native values. Native downselection still runs afterward. Explicit
reflection bounds are also checked on the finished waveform after sequential
setters and random overrides; inconsistent rows are rejected. This includes
legacy setters that skip unsupported values, such as eta below 1e-5.
Legacy setter order still matters: for example, assign ``q`` before ``mc``
to preserve both values (the q setter holds total mass fixed). Reversing that
order can reject the entire puff; ``--fail-if-empty`` exposes this as an error.
The four transverse spin box coordinates retain their automatic default
reflection. Explicit bounds override those defaults for a named coordinate.
Azimuth wrapping is unchanged; periodic export values can differ by an integer
period from the requested interval. Final checks compare equivalent angle lifts.
Native mean-anomaly reflection is refused when
using the effective eccentric-phase chart, whose angle is a different coordinate.
Mu mode retains its separate coupled q/s2z reflection before inversion.

Workflow integration tests
--------------------------

The pipeline regressions execute every generated PUFF submit command, including
ordinary PUFF, every scheduled stage and the refinement child. They validate
nonempty physical XML output and reject malformed schedules in the real builder::

    GW_SURROGATE='' python -m pytest -q \
        MonteCarloMarginalizeCode/Code/test/test_mu_puffball_pipeline.py

For real HTCondor/DAGMan execution, use the standalone integration driver on a
host with the usual RIFT dependencies. Choose a fresh directory for each run::

    python MonteCarloMarginalizeCode/Code/test/run_puffball_workflow_integration.py \
        --directory /tmp/puff-integration-scheduled --case scheduled \
        --local-toys --submit

    # After DAGMan finishes:
    python MonteCarloMarginalizeCode/Code/test/run_puffball_workflow_integration.py \
        --directory /tmp/puff-integration-scheduled --case scheduled --verify

Repeat with cases ``baseline``, ``general``, ``invalid`` and ``empty``. The last
two must fail at PUFF without exporting a grid or running dependent puff ILE.
The verifier treats those expected failures as successful negative checks.

Only ILE and CIP calculations are toys; PUFF, consolidation, unification,
conversion, convergence testing and refinement-grid fetching remain real.
``--local-toys`` changes only toy-worker submit files to local universe;
PUFF and other jobs retain the universe selected by the real builder.
The driver records build/submit logs, DAGMan metrics, toy execution traces and
``verification.json``. This tests workflow execution, not scientific recovery.
A jumpstarted child consumes inherited parent composites directly; consolidation
starts when the child actually runs ILE in its next iteration.
