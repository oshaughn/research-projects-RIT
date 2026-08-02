##############
``survey_scan``
##############

``containers/survey_scan.sh`` is an operator-facing companion for RIFT
container families.  It surveys a target HTCondor GPU pool, generates one
warmup job for each selected container/profile combination, and summarizes the
JSON reports returned by completed jobs.  It does not run an analysis or submit
the generated jobs itself.

For the container-family manifest and deployment model, see :doc:`../containers`.

Prerequisites and boundaries
=============================

The submit-side commands use Python's standard library.  Manifest parsing uses
PyYAML when it is installed and otherwise supports the simple RIFT
container-family YAML schema.  ``survey`` needs ``condor_status`` on the host.
To execute the jobs, the target environment needs HTCondor, a compatible GPU,
Apptainer, and an image containing the requested CuPy or JAX dependencies.

This command is an operator-run pool inventory and cache-warmup workflow.  The
reference documents the generated workflow; it is not evidence that a given
pool, container runtime, or image has been exercised successfully.

Survey a pool
=============

Run ``survey`` before selecting image bands or generating jobs::

   containers/survey_scan.sh survey \
     --out survey/cit-YYYYMMDD \
     --manifest container_family/rift_container_family.generated.yaml

The exact interface is::

   containers/survey_scan.sh survey [--out DIR] [--constraint EXPR] [--manifest FILE]

``--constraint`` is passed to ``condor_status`` and defaults to
``TotalGPUs > 0``.  ``--manifest`` is optional; when supplied, the inventory
also records which manifest labels cover each observed capability.  If ``--out``
is omitted, the command creates a timestamped directory under ``survey/``.

The survey directory contains:

* ``gpu_inventory.json`` — raw ClassAd fields, normalized summary, and optional
  manifest coverage;
* ``gpu_inventory.tsv`` — the summarized slot/device/capability/memory table;
* ``recommended_matrix.json`` — suggested capability bands; and
* ``coverage.md`` — a readable inventory, suggested bands, and manifest
  coverage when a manifest was supplied.

Generate warmup jobs
====================

Generate Condor submit files from a survey directory and a container-family
manifest::

   containers/survey_scan.sh emit-jobs \
     --survey survey/cit-YYYYMMDD \
     --manifest container_family/rift_container_family.generated.yaml

The exact interface is::

   containers/survey_scan.sh emit-jobs --survey DIR --manifest FILE [--out DIR] [--profiles LIST] [--request-disk REQUEST_DISK]

``--survey`` and ``--manifest`` are required.  ``--out`` defaults to
``DIR/jobs``; ``--profiles`` is a comma-separated list and defaults to
``cupy``; and ``--request-disk`` defaults to ``16000M``.  Supported profile
names are ``cupy`` and ``jax``.  For example, request both only for a
JAX-enabled image::

   containers/survey_scan.sh emit-jobs \
     --survey survey/cit-YYYYMMDD \
     --manifest container_family/rift_container_family.generated.yaml \
     --profiles cupy,jax

The ``cupy`` profile warms common NoLoop and fused-calmarg kernels.  The
``jax`` profile warms synthetic JAX ILE-wrapper shapes.  The generated directory
has a ``.sub`` and executable ``run_*.sh`` wrapper for each selected
container/profile pair, copied profile scripts, and ``submit_all.sh``.  Submit
them deliberately from that directory::

   cd survey/cit-YYYYMMDD/jobs
   ./submit_all.sh

Each wrapper sets cache locations such as ``CUPY_CACHE_DIR`` and
``JAX_COMPILATION_CACHE_DIR`` before running ``apptainer exec --nv``.  For an
``osdf://`` image URL, it fetches only that selected image, using ``stashcp`` or
``pelican``; one of those tools must therefore be available on the execute
node.  Size ``--request-disk`` for the selected image and its work area.

Collect results
===============

After jobs have returned their JSON outputs to the jobs directory, collect a
single summary::

   containers/survey_scan.sh collect --survey survey/cit-YYYYMMDD

The exact interface is::

   containers/survey_scan.sh collect --survey DIR [--out FILE]

``--survey`` is required.  By default, the command writes
``warmup_summary.json`` and its Markdown counterpart
``warmup_summary.md`` in the survey directory.  ``--out`` selects a different
JSON summary path; the Markdown report uses the same basename with a ``.md``
suffix.  Invalid result JSON is retained as an error entry so the summary can
show incomplete or malformed job output.

See also
========

* :doc:`../containers` for building and deploying a container family.
* ``containers/survey_scan/README.md`` in the source tree for a concise
  operator-oriented overview of the warmup profiles.
