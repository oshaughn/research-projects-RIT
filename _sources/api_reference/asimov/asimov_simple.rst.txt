RIFT+Asimov: Simple Workflow
============================

.. note::
   This is a living document. RIFT and Asimov interfaces evolve on few-month timescales.
   Always cross-check with the latest authoritative references linked in the `References`_ section.

This section describes the practical processes for deploying RIFT analyses using the ``asimov`` framework.

RIFT+Asimov Workflow
--------------------

The deployment of a RIFT analysis in ``asimov`` follows a hierarchical "Blueprint" pattern. Instead of defining every single parameter for every event, settings are split into **Common Defaults** and **Analysis-Specific Specifications**.

The final configuration applied to ``rift.ini`` is a merge of these two layers. ``rift.ini`` is a Jinja2 template (see :ref:`rift-ini-mapping`) that maps Asimov YAML labels to RIFT command-line arguments.

1. The Common Defaults (Infrastructure Layer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Common Defaults define the environment and hardware requirements that remain constant across different physics models. These are typically stored in a global configuration file, such as ``production-pe-o4b.yaml``.

**Example Configuration Block:**

.. code-block:: yaml

   kind: configuration
   pipelines:
     rift:
       scheduler:
         osg: True
         accounting group: ligo.dev.o4.cbc.pe.rift
         request memory: 1024
         singularity image: /cvmfs/singularity.opensciencegrid.org/james-clark/research-projects-rit/rift:test
         gpu architectures:
           - Tesla K10.G1.8GB
           - Tesla P100-PCIE-16GB
       sampler:
         cip:
           fitting method: rf
           explode jobs auto: True
         ile:
           n eff: 10
           jobs per worker: 100

**What this defines:**

* **Cluster Access**: Tells ``asimov`` to use OSG and which accounting group to bill.
* **Environment**: Specifies the base Singularity image that contains the RIFT binaries.
* **Hardware Target**: Ensures the jobs are scheduled on nodes with compatible GPUs.
* **Global Sampler Logic**: Sets the default fitting method (Random Forest) and effective sample target.

.. note::
   The ``cip`` label maps to command-line arguments for the ``cip`` sampler tool. See :doc:`../../executables/util_ConstructIntrinsicPosterior_GenericCoordinates`.
   The ``ile`` label maps to command-line arguments for ``integrate_likelihood_extrinsic_batchmode``. See :doc:`../../executables/integrate_likelihood_extrinsic_batchmode`.

2. The Analysis Specification (Physics Layer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _cip-section:

The Analysis Specification describes the specific physics, waveform model, and analysis strategy. These are defined in analysis-specific files (e.g., ``analysis_rift_SEOBNRv5PHM.yaml``).

**Example Configuration Block:**

.. code-block:: yaml

   kind: analysis
   name: rift-v5EHM-calmarg
   pipeline: RIFT
   waveform:
       approximant: SEOBNRv5EHM
       reference frequency: 10
   likelihood:
       start frequency: 10
       assume:
            - eccentric
            - nonprecessing
   scheduler:
       osg: True
       singularity image: osdf:///igwn/cit/staging/richard.oshaughnessy/rift_containers/rift_container_ros_seobnr_20250527.sif
   sampler:
         cip:
           sampling method: AV
           request disk: "4G"
         ile:
           sampling method: AV
           manual extra args:
                   - --use-gwsignal

**What this defines:**

* **Physics Model**: Switches the approximant to ``SEOBNRv5EHM`` and sets the start frequency.
* **Analysis Constraints**: Explicitly enables eccentricity and disables precession.
* **Specialized Resources**: Overrides the base image with a model-specific container (e.g., from `osdf:///`).
* **Sampler Refinement**: Switches the sampling method to ``AV`` and increases requested disk space to accommodate the specific model's requirements.

The Merge Process
^^^^^^^^^^^^^^^^

When a user executes the ``asimov apply`` command, the framework merges these two blueprints. The **Analysis Specification** takes precedence over the **Common Defaults**.

**Pedagogical Example of the Merge:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Setting
     - Common Default
     - Analysis Spec
     - Final Result (Applied to ``rift.ini``)
   * - ``osg``
     - ``True``
     - ``True``
     - ``True``
   * - ``singularity image``
     - ``/cvmfs/.../rift:test``
     - ``osdf:///.../rift_container_ros...``
     - ``osdf:///.../rift_container_ros...`` (Override)
   * - ``sampling method``
     - (Not Specified)
     - ``AV``
     - ``AV``
   * - ``n eff``
     - ``10``
     - (Not Specified)
     - ``10`` (Inherited)

.. _rift-ini-mapping:

Label to Argument Mapping (rift.ini)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``rift.ini`` uses Jinja2 templating to map Asimov YAML labels to RIFT command-line arguments. For example:

* YAML ``sampler: ile: n eff: 10`` → ``rift.ini`` ``[ile] n-eff=10`` → passed as ``--n-eff 10`` to ``integrate_likelihood_extrinsic_batchmode``
* YAML ``sampler: cip: fitting method: rf`` → ``rift.ini`` ``[cip] fitting-method=rf`` → passed as ``--fitting-method rf`` to ``cip``

The authoritative ``rift.ini`` template is located at:

`MonteCarloMarginalizeCode/Code/RIFT/asimov/rift.ini <https://github.com/oshaughnessy-junior/research-projects-rit/blob/main/MonteCarloMarginalizeCode/Code/RIFT/asimov/rift.ini>`_ in this repository.

.. _asimov-rift-example:

Example of Asimov+RIFT: Full project + analysis setup
----------------------------------------------------

To deploy a full analysis for a specific event (e.g., ``S240426s``), follow this complete sequence adapted from the `Asimov GW150914 tutorial <https://asimov.docs.ligo.org/asimov/master/tutorials/analysing-gw150914.html>`_.

Pre-Analysis Setup (Mandatory Prerequisites)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Create project directory**:

   .. code-block:: bash

      mkdir S240426s-analysis
      cd S240426s-analysis

2. **Initialize Asimov project**:

   .. code-block:: bash

      asimov init "S240426s RIFT Analysis"

3. **Apply infrastructure blueprint** (defaults for all analyses):

   .. code-block:: bash

      asimov apply -f https://git.ligo.org/asimov/data/-/raw/main/defaults/production-pe.yaml

4. **Apply priors blueprint**:

   .. code-block:: bash

      asimov apply -f https://git.ligo.org/asimov/data/-/raw/main/defaults/production-pe-priors.yaml

5. **Add event metadata** (using event blueprint):

   .. code-block:: bash

      asimov apply -f https://git.ligo.org/asimov/data/-/raw/main/events/gwtc-2-1/GW150914_095045.yaml

6. **Apply physics/analysis blueprint**:

   .. code-block:: bash

      asimov apply -f analyses/rift-bbh/analysis_rift_SEOBNRv5PHM.yaml -e S240426s

Asimov Workflow Execution
^^^^^^^^^^^^^^^^^^^^^^^

7. **Build the analysis** (generate configuration and DAG):

   .. code-block:: bash

      asimov manage build

8. **Submit to cluster**:

   .. code-block:: bash

      asimov manage submit

9. **Monitor progress** (or use ``asimov start`` for automated monitoring):

   .. code-block:: bash

      asimov monitor
      # OR for automated monitoring:
      asimov start

This process ensures that infrastructure changes (like upgrading a base container) can be applied globally across all analyses by editing a single defaults file, while physics changes remain isolated to their respective analysis blueprints.

References
----------

* **RIFT GitHub Repository**: `oshaughnessy-junior/research-projects-rit <https://github.com/oshaughnessy-junior/research-projects-rit>`_
* **ILE (Integrate Likelihood Extrinsic) Documentation**: :doc:`../../executables/integrate_likelihood_extrinsic_batchmode`
* **CIP Sampler Documentation**: :doc:`../../executables/util_ConstructIntrinsicPosterior_GenericCoordinates`
* **cbcflow Documentation**: `cbcflow <https://cbc.docs.ligo.org/projects/cbcflow/index.html>`_
* **Asimov Documentation**: `asimov <https://asimov.docs.ligo.org/asimov/master/index.html>`_
* **Example YAML Configs**: Refer to Asimov configuration directories for infrastructure and physics blueprints
* **rift.ini Template**: `MonteCarloMarginalizeCode/Code/RIFT/asimov/rift.ini <https://github.com/oshaughnessy-junior/research-projects-rit/blob/rift_O4d_junior/MonteCarloMarginalizeCode/Code/RIFT/asimov/rift.ini>`_
* **Authoritative YAML Examples**: `defaults/ <https://git.ligo.org/asimov/data/-/tree/main/defaults>`_, `analyses/ <https://git.ligo.org/asimov/data/-/tree/main/events>`_, and `precursors/ <https://git.ligo.org/asimov/data/-/tree/main/precursors>`_ directories
