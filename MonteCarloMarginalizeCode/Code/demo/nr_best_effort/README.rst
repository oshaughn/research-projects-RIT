NR posterior bootstrap from one existing run
============================================

This demo accompanies ``util_NRBestEffortPosterior.py``. It reads a completed
NR run's native ``all.net``, an original/refined XML grid, and ``args_ile.txt``. It writes
a posterior-only evidence-weighted intrinsic grid, checks the configured NR catalog, runs
conditional extrinsic followups, and exports one conditional draw per intrinsic
draw. It never submits jobs or writes into the source run.

Install RIFT normally (``pip install .``). The utility is installed in ``bin``;
this README, the full statistical instructions, and ``bootstrap_existing_run.py``
are installed under ``${VIRTUAL_ENV:-$CONDA_PREFIX}/share/RIFT/nr-best-effort``
(more generally ``sys.prefix/share/RIFT/nr-best-effort``). Source-tree copies
are under ``Code/demo/nr_best_effort`` and ``docs/source/nr_best_effort.rst``.

Configure ``NR_BASE`` and ``PYTHONPATH`` for the public
``NRWaveformCatalogManager3`` module and the same metadata/archive used by the
original analysis. Waveform I/O additionally needs the catalog's reader
dependencies (including ``romspline`` for compressed RIT mode data); a metadata
check alone cannot verify them. RIFT's backend selector also supports
``nrcatalog.compat_nrwf`` wrapping that module. Pure registry-only nrcatalog
installs whose waveform constructor is still a stub cannot run ILE. The
``catalog-check`` command uses the public ``WaveformModeCatalog(...,
metadata_only=True).P`` API, preserves component/spin ordering, and checks native
eccentricity labels. It does not depend on private JSON/pickle internals.

One source set
--------------

Choose exactly one completed source run with native NR labels, not an
eccentricity-remapped CIP table. For example::

   python bootstrap_existing_run.py prepare \
     --source-run /path/to/completed/rundir \
     --mass-range 150 1650 --group RIT-Five --samples 32 \
     --out ./nr-best-effort-example

The mass bounds above are an example, not defaults. They must be chosen for
your run. This prototype explicitly allows incomplete mass support and
coalesces repeated same-waveform estimates; it reports the resulting truncated
evidences. Review ``plan/manifest.json`` and ``catalog-check.json`` before using
the results scientifically. Multiple identities are reported rather than
silently resolving a waveform resolution/extraction. Followups preserve the
original run's NR lookup policy and all data/PSD/prior/frequency arguments.
The XML is used explicitly as a shared settings template; intrinsic labels are
written from the composite and separately checked against the catalog.
Use the same catalog revision as the original run; a different revision can
change a lookup choice even if its intrinsic labels match.

Run a bounded I/O smoke check first::

   python bootstrap_existing_run.py run --out ./nr-best-effort-example \
     --index 0 --smoke-n-max 20000

Inspect ``smoke/0/ile.log`` for the selected waveform, NoLoop path, data loading,
and completed export. The script rejects a missing or collapsed integrator
status; an undersampled smoke run may deliberately fail this quality check.
Its XML/logs can establish I/O, but its samples are not a posterior. Smoke
output is kept separate and cannot be
used by ``finish``. This checks execution, not posterior convergence.

Run one conditional followup under the full source-run sampling settings::

   python bootstrap_existing_run.py run --out ./nr-best-effort-example --index 0

Repeat for indices 1 through 31, locally or as independent batch jobs. It is
safe to dispatch different indices concurrently, subject to your cluster's
resource policies. Do not repeat an index into an existing output directory.
The demo caps numerical-library threads at two. ILE memory/GPU requirements
remain those of the source analysis.

Once every event has completed::

   python bootstrap_existing_run.py finish --out ./nr-best-effort-example

This writes ``extrinsic_posterior_samples.dat`` and provenance. It checks all
expected events and their intrinsic coordinates. Each scheduled intrinsic draw
contributes exactly one extrinsic row, so a run that happened to produce more
fairdraws cannot acquire more mixture weight. Increasing ``--samples`` in a new
preparation increases the number of scheduled intrinsic draws. Thirty-two rows
are a workflow prototype, not a publication-quality posterior.

No additional evidence weighting is applied at ``finish``: the intrinsic grid
already samples the evidence-weighted mass/simulation mixture. For separately
saved fixed-mass or mass-integrated simulation conditionals, use the utility's
``combine`` command instead; see ``nr_best_effort.rst`` for its manifest format,
prior definitions, limitations, and historical references.
