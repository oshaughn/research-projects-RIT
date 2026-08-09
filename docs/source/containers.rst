Containers and multi-architecture deployment
=============================================

RIFT runs its compute jobs (ILE, CIP) inside a Singularity/Apptainer container
on HTCondor pools such as the OSG.  Historically the environment variable
``SINGULARITY_RIFT_IMAGE`` names a **single** image, and every job is pinned to
it::

    export SINGULARITY_RIFT_IMAGE=/cvmfs/singularity.opensciencegrid.org/.../rift:production

That still works exactly as before.  This page documents two additions:

* a **container *family*** — point ``SINGULARITY_RIFT_IMAGE`` at a YAML
  *manifest* describing several images that target different GPU compute
  capabilities, and let HTCondor pick the right one per matched machine; and
* a **multi-target build** that produces such a family from one template.

.. note::

   If ``SINGULARITY_RIFT_IMAGE`` is a plain ``.sif`` path or a single
   ``osdf://`` URL, behavior is **unchanged** — the manifest machinery is never
   engaged.  A manifest is recognized purely by its ``.yaml`` / ``.yml`` suffix.


Deploying a container family
----------------------------

Set ``SINGULARITY_RIFT_IMAGE`` to a manifest file instead of a single image::

    export SINGULARITY_RIFT_IMAGE=`pwd`/rift_container_family.yaml

Everything else — ``util_RIFT_pseudo_pipe.py``, ``--use-singularity``,
``--use-osg`` — is identical.  When the pipeline builds the ILE/CIP submit
files it reads the manifest and emits an *expression-valued* container
selection (see `What the pipeline generates`_ below).

Manifest format
~~~~~~~~~~~~~~~

.. code-block:: yaml

    version: 1

    # Machine ClassAd attribute the selection expression tests.
    # Default GPUs_Capability (see "GPU attribute names" below).
    capability_attr: GPUs_Capability

    # Catch-all image (innermost else of the selection); MUST be CPU-safe,
    # since it is also used when no GPU capability is advertised.
    fallback: default

    containers:
      # Broadly-compatible image for older machines. On CVMFS: referenced in
      # place and lazy-fetched (only the selected image is ever pulled), never
      # transferred.
      - label: default
        image: /cvmfs/singularity.opensciencegrid.org/oshaughn/rift_container_default.sif
        cuda_capability_min: 3.5     # inclusive lower bound for this image
        cuda_capability_max: 8.0     # informational; null = open-ended
        note: "cupy-cuda11x, ubuntu22.04/cuda11.8"

      # Newer image for higher-capability GPUs. Delivered via osdf: only the
      # matched machine fetches it (selective transfer).
      - label: modern
        image: osdf:///igwn/staging/oshaughn/rift_containers/rift_container_modern.sif
        cuda_capability_min: 8.0
        cuda_capability_max: null
        note: "cupy-cuda12x, ubuntu22.04/cuda12.4"

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``version``
     - Manifest schema version (currently ``1``).
   * - ``capability_attr``
     - Machine ClassAd attribute the selection expression tests (default
       ``GPUs_Capability``).
   * - ``fallback``
     - ``label`` of the catch-all image (innermost ``else``); **must be
       CPU-safe**.
   * - ``containers[].label``
     - Human id; also referenced by ``fallback``.
   * - ``containers[].image``
     - A CVMFS/local path (referenced in place, lazy-fetched) **or** an
       ``osdf://`` URL (selectively transferred).
   * - ``containers[].cuda_capability_min``
     - Inclusive lower capability bound for this image.
   * - ``containers[].cuda_capability_max``
     - Informational upper bound (``null`` = open-ended).
   * - ``containers[].note``
     - Free text.

A starting manifest lives at :code:`containers/rift_container_family.yaml` in the
source tree.

.. warning::

   **Keep the family consistent.** A *single* ``SINGULARITY_BASE_EXE_DIR`` is
   applied to **every** image in the family — the ILE/CIP jobs locate the
   executable as ``SINGULARITY_BASE_EXE_DIR + <exe name>``, with no per-image
   override. Every image in a manifest **must install RIFT's executables at the
   same in-container path** (and share a common layout / Python / entrypoints).
   Build them from the same ``rift_container.def.in`` template
   (``build_family.sh`` does this); do **not** hand-mix images with different
   internal layouts. The same applies to ``SINGULARITY_BASE_EXE_DIR_HYPERPIPE``
   if you use hyperpipe.


What the pipeline generates
---------------------------

For a manifest, the GPU ILE Condor submit files get:

* **``MY.SingularityImage``** — an *unquoted* ``ifThenElse`` expression that
  selects the highest-capability image the matched machine can run, with the
  ``fallback`` image as the innermost ``else`` (used when the machine's
  capability is below every threshold)::

      ifThenElse(TARGET.GPUs_Capability >= 8.0, "./rift_container_modern.sif", "/cvmfs/.../rift_container_default.sif")

* **Selective transfer** — only ``osdf://`` images are fetched, and only on the
  machine that selected them, via a single HTCondor ``$$()`` match-time token
  appended to ``transfer_input_files``.  CVMFS/local images are referenced in
  place and never transferred, so the **whole family is never pulled**::

      $$([ (TARGET.GPUs_Capability >= 8.0 ? "osdf:///.../rift_container_modern.sif" : "") ])

  ``request_disk`` is **not** auto-sized — set it to your largest single
  transferred image.

* **``require_gpus`` floor** — ``Capability >= <lowest min across the family>``,
  combined (``&&``) with any ``RIFT_REQUIRE_GPUS`` you set.  Both apply; neither
  is dropped.  This stops jobs matching a GPU that *no* image in the family
  supports.

* **A capability-defined ``Requirements`` clause** —
  ``TARGET.GPUs_Capability =!= undefined``.  The selection above is deliberately
  *not* undefined-guarded: a slot that does not advertise the machine-level
  capability rollup could be anything (including a Blackwell that hard-fails on
  the older fallback image), so the safe action is to **not match it** rather
  than guess.  Measured on the CIT pool, a large fraction of GPU slots satisfy
  the per-GPU ``require_gpus`` floor yet do not advertise the rollup attribute;
  without this clause those jobs go on hold with "Cannot expand $$ expression".

CPU-only jobs are handled differently.  **CIP requests no GPU**, so its matched
slot advertises no capability at all and a capability-keyed selection cannot
resolve — it would hold the job.  CIP therefore collapses to a **single fixed
container**: the manifest ``fallback`` image (hence the requirement that the
fallback be CPU-safe), quoted as a plain literal, with no ``$$()`` token and no
capability ``Requirements`` clause.


.. _osg-container-modes:

Choosing a delivery mode (legacy vs container universe)
-------------------------------------------------------

The expression-valued ``MY.SingularityImage`` above is evaluated on the
*execute* side.  That works on a local HTCondor pool, but **OSPool glidein
pilots read** ``SingularityImage`` **as a literal string**, so an ``ifThenElse``
lands verbatim and the job holds.  Two opt-in modes solve this; pick one with an
environment variable at DAG-build time.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Mode
     - Behaviour
   * - *(default)*
     - Legacy: ``universe = vanilla`` + expression-valued
       ``MY.SingularityImage``.  Correct on a local/CIT pool.  **Not OSG-safe.**
   * - ``RIFT_CONTAINER_UNIVERSE=1``
     - **Recommended for OSG.**  ``universe = container`` and
       ``container_image = $$([ ifThenElse(...) ])``.  ``$$()`` is HTCondor's
       documented *match-time* (schedd-side) machine-ad substitution, so the
       pilot only ever sees a literal image URL.  No ``MY.SingularityImage``,
       no ``MY.SingularityBindCVMFS``, no ``$$()`` transfer token — the image is
       delivered by ``container_image`` itself.  GPU access is automatic under
       ``request_gpus``.  Works on CIT-local too.
   * - ``RIFT_CONTAINER_RUNTIME_SELECT=1``
     - Older fallback, ILE only.  Condor runs a generated
       ``rift_container_select.sh`` on the bare execute node; the wrapper reads
       the *real* GPU capability from ``nvidia-smi``, fetches only the matching
       image (``stashcp``/``pelican``), and re-execs the command under
       ``apptainer exec --nv``.  Detects the actual device rather than trusting
       an advertised attribute, at the cost of running outside a container.

Under asimov, set the variable from the blueprint rather than the shell:

.. code-block:: yaml

    scheduler:
      singularity image: /path/to/rift_container_family.yaml
      singularity base exe directory: /usr/local/bin/
      environment variables:
        RIFT_CONTAINER_UNIVERSE: 1

.. note::

   With a family manifest whose images are ``osdf://`` URLs, the pipeline also
   enables the matching transfer credential automatically
   (``use_oauth_services = scitokens``, or ``igwn`` for ``igwn+osdf:``).  The
   single-image code path keys that off the ``SINGULARITY_RIFT_IMAGE`` string
   itself, which for a manifest is just a ``.yaml`` path — so the manifest's
   image URLs are inspected instead.


GPU attribute names
-------------------

Two different ClassAd namespaces are involved, and they are kept separate:

* The **image selection** ``ifThenElse`` reads the *machine* ad.  The default
  attribute is ``GPUs_Capability``.  Override it per run with the environment
  variable ``RIFT_GPU_CAPABILITY_ATTR``, or per manifest with ``capability_attr``.
  Verify what your pool advertises::

      condor_status -constraint 'TotalGPUs > 0' -autoformat GPUs_DeviceName GPUs_Capability GPUs_GlobalMemoryMb

  Not every GPU host advertises this; on such hosts the expression collapses to
  the fallback image and the ``require_gpus`` floor does the steering.

* The **``require_gpus`` floor** uses the require_gpus sub-ad attribute
  ``Capability`` (unprefixed — *not* ``TARGET.``, *not* ``GPUs_``).

.. note::

   These mechanisms have been validated on a real HTCondor pool + GPU: the
   attribute names, the ``require_gpus`` floor (matching a compatible GPU and
   excluding an incompatible one), the ``$$()`` match-time image selection, and
   tolerance of the empty-result case for a manifest that mixes CVMFS and
   ``osdf`` entries.  The expression-valued ``MY.SingularityImage`` is *not*
   OSPool-safe (see :ref:`osg-container-modes`); use
   ``RIFT_CONTAINER_UNIVERSE=1`` there.


Building a container family
---------------------------

The build lives under :code:`containers/`:

* :code:`rift_container.def.in` — an Apptainer definition template with
  ``@@BASE_IMAGE@@`` / ``@@CUPY_PKG@@`` placeholders.
* :code:`build_family.sh` — renders one ``.def`` per build-matrix entry and runs
  ``apptainer build``.  The **first** matrix entry keeps the current production
  base image, so the family always includes a broadly-compatible image for older
  machines.
* :code:`requirements-container.txt` — the shared, unpinned pip dependency set
  (the cupy wheel is the only per-entry difference).

.. code-block:: console

    # render the per-entry .def files only (no apptainer needed)
    containers/build_family.sh --render-only ./container_family

    # render and build each .sif (requires apptainer)
    containers/build_family.sh ./container_family

    # on shared clusters (e.g. CIT), build with --fakeroot to avoid the
    # unprivileged proot engine (whose mksquashfs step fails):
    containers/build_family.sh --fakeroot ./container_family

Each run also writes a ``rift_container_family.generated.yaml`` stub: fill in
each ``image:`` with where you published the ``.sif`` (a CVMFS path or
``osdf://`` URL) and you have a deployable manifest.

.. note::

   On clusters without setuid apptainer or unprivileged user namespaces, a plain
   build falls back to the ``proot`` engine and fails at ``mksquashfs``
   (``ptrace(TRACEME): Operation not permitted``). Use ``--fakeroot`` (needs
   ``/etc/subuid`` + ``/etc/subgid`` entries), or ``--sandbox`` to build a
   directory that skips ``mksquashfs`` and convert it to a ``.sif`` later on a
   capable host. See the build-troubleshooting section of
   :code:`containers/README.md`.

The top-level :code:`rift_container.def` is unchanged and remains the default
single-image build.


Catching dependency breakage early
----------------------------------

The container ships an *unpinned* dependency set and clones RIFT at build time,
so a fresh upstream release (for example ``swig>=4.4.0``) can silently break
RIFT and only surface when a container rebuild fails.  ``rift_O4d`` carries two
non-blocking GitHub Actions canaries for this (``container-dep-canary`` and
``container-swig-canary``); they are not part of this branch, whose CI is
GitLab-based.  Until an equivalent exists here, re-run the install of
``containers/requirements-container.txt`` plus a RIFT import check by hand before
a family rebuild.
