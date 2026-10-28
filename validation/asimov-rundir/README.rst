Native Asimov directory ownership regression
==========================================

This is development evidence, not frozen rc12 qualification. The candidate
package retains its branch version label; verify-development.py records its
local source URL and installed RECORD hashes separately. No editable install,
PYTHONPATH overlay, alternate builder, generated-file edits or scheduler jobs
are used. A newly published RC is required for production qualification.

On CIT, with an absolute new prefix and a source checkout at the reviewed commit::

    bash validation/asimov-rundir/setup-cit.sh /absolute/new/frontend
    source /absolute/new/frontend/bin/activate
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python -m pytest -q MonteCarloMarginalizeCode/Code/test/asimov_integration/test_asimov_rift_native_lifecycle.py

Activation is required so real builder child commands resolve to this installed
candidate. The lock pins Asimov 0.8.0, LALSuite 7.26.16 and HTCondor 24.12.24;
setup uses CVMFS CPython 3.11.14. The test uses real pipeline hooks, Asimov's
make_config, real builder subprocesses and provenance writers, with a minimal
production persistence interface. It generates scientific input fixtures and
workflow files, and submits no jobs. Missing dependencies fail rather than
being replaced by test doubles. The native regression is pinned to Asimov 0.8.0;
the older supported adapters retain the upstream no-op provenance hook.

Safety semantics
----------------

The native builder exclusively creates the run directory. Every preexisting
path is refused, including empty, provenance-only, partial and completed runs,
and dangling symlinks. Configuration does not capture environment provenance;
it is captured directly in the final directory only after successful generation
and a nonempty DAG. A failed or interrupted generation is retained and requires
a fresh destination. No adoption marker, automatic cleanup or deletion exists.
A dangerous-overwrite flag was considered and rejected because this collision
can be fixed without introducing permission to modify prior runs.

The adapter guards before configuration and build preparation; the unchanged
atomic builder mkdir decides concurrent attempts. Failed builder exit status
cannot be rescued by a stale DAG. Asimov owns use-rundir and use-ini; scheduler
path overrides, abbreviations, embedded-equals forms and early-success help
options are refused. The corresponding INI path overrides are refused too.

After successful generation, new environment.json and pip/conda manifests are
written and recorded at the final path. No existing provenance destination may
be replaced. A missing/empty manifest or swallowed capture error fails the
adapter and retains its partial run. PSD staging follows successful provenance
capture. Configuration may still provision versioned PSD/frame-cache inputs
outside the run directory as before; dry-run skips frame-cache generation.

This protects ordinary existing runs and competing native builders. It does
not sandbox arbitrary code in trusted INI expressions, replace filesystem
permissions, or defend against a hostile process renaming/replacing an owned
run directory during generation. No claim is made that direct standalone
pseudo_pipe calls are globally side-effect-free before mkdir (legacy GraceDB
lookup can write coinc.xml in its caller's working directory).
