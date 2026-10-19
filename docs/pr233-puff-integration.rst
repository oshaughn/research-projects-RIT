PR 233 PUFF workflow integration
================================

Tested on Cardassia on 2026-10-19 with HTCondor 23.4.0, Python 3.12.4,
LALSuite 7.26.16 and igwn-ligolw 2.1.1 in an isolated virtual environment.
The source snapshot was PR 233 at 4cd0fc58a plus the jumpstart fix described
below. Production PUFF, proposal modules and DAG utilities were unchanged
from that source. All final test cases used the corrected builder.

Scope and checks
----------------

The standalone ``run_puffball_workflow_integration.py`` builds the actual
BasicIteration DAG. ILE validates its XML input and writes deterministic toy
likelihoods; CIP checks consolidated input and emits a deterministic physical
grid. PUFF and the consolidation, unification, conversion, convergence and
refinement-fetch executables are real. Toy worker jobs use local universe;
real jobs keep the builder's selected universe, including vanilla refinement
jobs. Global Condor configuration and existing analysis checkouts are untouched.

Each successful run checks DAGMan success, nonempty finite physical XML,
unit-spin bounds, actual ILE consumption of PUFF output, requested reflection
bounds and expected proposal-stage routing. The scheduled test uses finite
``1`` and ``G1`` stages followed by a ``Z`` child with two iterations. The
first stage alone must produce a mu sidecar; the child must run and consume
ordinary PUFF. Negative cases must fail at PUFF, produce no puff XML, and
prevent dependent puff ILE jobs from running.

Final Condor runs
-----------------

The run directories are under
``cardassia:/home/oshaughn/pr233-puff-integration-20261019/``, each with a ``-v4``
suffix. Earlier attempts are retained separately for diagnosis.

* ``baseline-v4`` (DAGMan 84558): passed, 12 nodes and two real PUFF executions.
* ``general-v4`` (DAGMan 84561): passed, 12 nodes and two real PUFF executions;
  explicitly reflects chirp mass and aligned spin while keeping automatic
  transverse spin box handling.
* ``scheduled-v4`` (DAGMan 84555): passed, all 19 ordinary parent nodes plus
  its refinement subdag succeeded. Five real PUFF executions produced five
  nonempty physical grids; the child completed successfully and its puff
  grid was consumed by ILE. Parent duration was 127.462 seconds.
* ``invalid-v4`` (DAGMan 84565): expected failure, invalid reflected coordinate;
  no puff XML and no dependent puff ILE execution.
* ``empty-v4`` (DAGMan 84570): expected failure, no completed waveforms satisfy
  the explicit eta bounds; ``--fail-if-empty`` stops downstream work.

Bug exposed and fixed
---------------------

The original scheduled run reached the refinement child and failed because
``--first-iteration-jumpstart`` omitted its initial ILE jobs but still scheduled
consolidation of their empty output directory. The real consolidator rejected
the empty data. The builder now omits that first consolidation node and its
unification dependency only for a jumpstarted iteration. The child unifies its
inherited parent composites; later ILE and consolidation remain active.
The subagent reviewed the dependency fix and found no remaining actionable
issues. An assertion checks both the absent first consolidation and present
next consolidation in the actual generated child DAG.

Two earlier toy-harness setup issues were also corrected: invoking a venv
interpreter through an external symlink lost its package environment, and
fixture paths could not depend on custom variables surviving DAGMan environment
filtering. The final harness uses an explicit interpreter launcher and embeds
its isolated fixture paths. These were harness changes, not PUFF changes.

Automated coverage and reproduction
-----------------------------------

The 73-test focused run passed across reflection, inverse/proposal/CLI,
pipeline and eccentric-phase tests. After the builder fix, all 14 pipeline
tests were rerun successfully, including real execution of every generated
PUFF stage and refinement submit command. Five negative builds reject schedule
length/prefix mismatches, conflicting puff options, missing CIP schedules and
missing cadence. The pipeline test file is already registered in GitLab CI.
``git diff --check`` and Python compilation checks pass.

See ``docs/mu-puffball.rst`` for commands to build, submit and verify each case.
Raw selected DAGs, submit descriptions, metrics, PUFF logs and toy traces are
preserved as evidence; environment dumps from consolidation are excluded.
The five recorded runtime source hashes were checked against the local change
being committed and match the tested Cardassia snapshot. The final queue was
empty after all test DAGs finished.
This is a workflow execution qualification with toy likelihood/posterior work.
It does not replace the independent scientific recovery qualification.
