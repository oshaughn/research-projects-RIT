CIT development validation, 2026-10-28
======================================

Branch: rift_O4c (target rift_O4c).
Base source: 2808c2fbe74aa1ba127ca8a88c8d97a046a59d21. Installed candidate version label: 0.0.17.14rc12.
These locally built, noneditable candidate packages are not published releases.

CPython 3.11.14; Asimov 0.8.0; LALSuite 7.26.16; HTCondor 24.12.24;
NumPy 2.2.6; SciPy 1.15.3; pytest 9.1.1. Exact dependency lock SHA256:
eeee1dd18806ef806b6e44a5369b6d1cc5ab0d6695498bebafeb3e5f33e73895.
Pip check passed. All 256 hashed installed wheel RECORD entries verified,
including pseudo_pipe, helper_LDG_Events and BasicIteration builder executables.

The native lifecycle, compatibility and build-contract test files passed:
105 passed, 90 deprecation warnings in 27.60 seconds. The native file supplies
90 cases, including actual before_config/make_config/build_dag/provenance,
regular DAG/submit output checks, scientific/partial/provenance/empty/stale-marker
refusal with bytes/mode/mtime snapshots, repeats, symlinks, parent/relative/fallback
aliases, real concurrent builders and full adapters, late competing stale DAG,
interrupted real builder, all reserved path/help option forms and values, occupied
provenance targets and genuine swallowed Asimov filesystem capture failure.
Only persistence/lookup objects are fixtures. No protected hooks, builder
subprocesses, provenance writes or directory guards are replaced. No scheduler
submission is invoked. A preexisting mocked command-contract test remains an
additional unit check and is not the basis for native lifecycle conclusions.

Two independent adversarial reviewers inspected both adaptations and rechecked
the final production code and realistic tests. Findings fixed: pre-refusal
writes, stale-DAG success after nonzero builder exit, argparse abbreviations and
embedded-equals path overrides, INI self-redirect, help early-success, stale or
partial capture metadata, and the conda manifest destination name. Their final
source reviews found no blocking issue under the cooperative concurrency model.

The unchanged rc12 native project diagnostic also passed at the final candidate:
real init/apply/manage build followed by the native build_dag hook, absent run
directory before builder, two generated DAGs, no generated-file patches and no
submission. This uses a new development project and does not qualify frozen rc12.

No merge, PyPI publication, production image replacement, existing-run change,
full PE or OSDF staging occurred. A new published RC and independent frozen
production qualification remain required; MR6 readiness is unchanged. Raw
receipts and logs remain in the owning private CIT/development workspace.

Adversarial review follow-up, 2026-10-28
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Host ldas-grid, environment built by setup-cit.sh. Changes:

- A refused run directory marks the production stuck. Asimov keeps a production
  ready after a transient submit failure. The PR otherwise logged the refusal on
  every later pass. The unchanged builder cannot resubmit from that directory,
  because condor_submit_dag refuses an existing ``.condor.sub``.
- A timed-out ``pip freeze`` cached by Asimov is retried. Otherwise one timeout
  failed every later build in the same process.
- The INI path check parses keys case-sensitively, as the builder does. A
  malformed INI raises PipelineException.
- Dry runs do not rewrite ``production.rundir`` or the status.
- The mocked build-contract test faked Asimov's ``pip freeze`` too. It passed only
  after another test had filled Asimov's package cache. It now fakes only the builder.
- New tests cover the DAG check, each capture-receipt clause, underscore
  scheduler keys, dry-run frame caches and the build-time guard.
- The receipt records the installed source commit. A clean install of 0d807952
  verifies 226 RECORD entries. The count above (256) did not reproduce.

Results: 138 passed in the three files above (was 105). Every guard
mutant tried (15) fails at least one test.
