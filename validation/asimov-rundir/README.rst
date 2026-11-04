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
production persistence interface. Negative contract cases separately inject
builder/capture/converter faults; native success and concurrency cases use the
real operations. It generates scientific input fixtures and
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

Native bootstrap input binding
------------------------------

A false or missing scheduler.bootstrap upstream flag does not request a
posterior bootstrap or reserve an INI manual grid. Null/empty optional bootstrap
file fields are also disabled. A nonempty explicit bootstrap file still requests
conversion even when upstream bootstrap is disabled.

Configured posterior bootstraps are converted directly into a new private input
directory for each build invocation. The selected manual grid and, when
scheduler.bootstrap coinc is explicitly true, the selected coincidence file
are bound after successful conversion. Canonical event coincidence files and
prior runs are never replaced. Private successful inputs are retained for
reproducibility. A configured but unresolved source, failed/invalid conversion,
non-boolean bootstrap-coinc value, or competing CLI/INI grid selection fails
before workflow generation. Legacy bootstrap: manual selects its supplied
existing grid read-only. Dry run prints a planned private coincidence input
without converting or creating files. Integer and float solar-mass prior bounds
are both accepted without mutating the source arrays.

Q-only development witness
--------------------------

Refinement requires GPU vectorized time marginalization; incompatible modes
fail before generating a workflow. Data, waveform requests and physical
integration nodes remain at 4096 Hz. No srate-internal change is used. The
versioned q2-worker-witness.py runs one bounded installed GPU-only worker using
unchanged generated args_ile.txt, records all bounded sampling overrides, and
checks generated and scientific input hashes before/after. It submits no jobs.
Its native generated run directory and grid must be supplied explicitly; it
must not be used to repair an existing run or strip incompatible arguments.

On pcdev13, select an available GPU and explicitly configure the installed CUDA
toolkit (the dependency lock alone does not provide NVRTC)::

    export CUDA_VISIBLE_DEVICES=3 CUDA_PATH=/usr/local/cuda-12.8
    export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64
    unset PYTHONPATH PYTHONHOME
    python validation/asimov-rundir/q2-worker-witness.py \
        --run-directory /absolute/native/generated/run \
        --grid /absolute/native/generated/run/proposed-grid.xml.gz \
        --output /absolute/new/witness

This bounded development witness does not qualify a published RC, a production
container, posterior accuracy, full PE, CIP or calibration.

O4d source configuration profile
--------------------------------

O4d now supports Q factors 1, 2, 4 and 8 with its private P.q_deltaT transport.
Refinement failures abort the worker; they cannot silently select coarse Q or
be swallowed by per-point error handling. O4d bandlimited time marginalization
owns continuous time draws and rejects a fixed time-export grid. Apply the
versioned profile after shared defaults and before event/analysis configuration::

    asimov apply -f validation/asimov-rundir/o4d-native-q2-profile.yaml

This source profile retains Q2 and disables the legacy fixed export rate. It
must be applied before native generation; do not edit generated arguments.
Strict frame-crop coverage remains required. The 10-second S241127aj portable
frames do not cover the padded interval, so that real-event recipe remains
unqualified pending genuinely longer input frames.

Refusal status and upstream persistence
---------------------------------------

A refused existing destination sets stuck in memory. Asimov 0.8 can catch that
exception without saving the event, so this does not guarantee that a later CLI
process skips it. Choose a new destination for a new build. The adapter does not
blindly save the whole event/ledger: the native YAML API can replace concurrent
job status and job-ID updates from a stale snapshot. Cross-process refusal
status persistence needs an upstream conditional update; it is not claimed here.

Cropped frame caches use file://localhost URLs, which the actual LAL FrameL
reader accepts in the locked environment. Each cropped frame is checked by that
reader before the output directory/cache is published. Input filenames may
contain spaces. Output run paths containing whitespace are refused explicitly:
LAL cache parsing truncates raw spaces and this FrameL reader does not decode
percent-encoded spaces. Coverage checks and the frame payload are unchanged.
