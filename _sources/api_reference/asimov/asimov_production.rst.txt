RIFT+Asimov: Production Workflow
===================================

.. note::
   This is a living document. RIFT and Asimov interfaces evolve on few-month timescales.
   Always cross-check with the latest authoritative references linked in the `References`_ section.

This section describes the production workflow for deploying RIFT analyses using the ``asimov`` framework. Unlike the :doc:`asimov_simple` workflow, which processes a single event with a minimal configuration, the production workflow handles **multiple events** and integrates **multiple pipelines** (RIFT, bilby, cbcflow, bayeswave) in a coordinated fashion.

Production vs Simple Workflow
-------------------------

The key differences between the simple and production workflows:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25

   * - Aspect
     - Simple (asimov_simple.rst)
     - Production (asimov_production.rst)
   * - Number of events
     - Single event
     - Multiple events (loop)
   * - Pipelines
     - RIFT only
     - RIFT + bilby + cbcflow + bayeswave
   * - Configuration layers
     - 2 (defaults + analysis)
     - 6+ (infrastructure, priors, cbcflow, localization, physics, bilby)
   * - Environment
     - Default Singularity container
     - Local conda environment + cvmfs
   * - Output
     - Basic asimov commands
     - cbcflow + webroot

Multi-Event Pattern
-------------------

The production workflow centers on the **multi-event pattern**. Instead of running a single analysis, production workflows typically process multiple events in a coordinated loop.

**Why Multiple Events?**

- **Efficiency**: Process many events with the same configuration
- **Consistency**: Ensure uniform treatment across events
- **Automation**: Reduce manual steps for each new event
- **Cross-validation**: Compare results across different waveforms

**Event Naming Convention**

Events in production are identified by their gracedb ID. The standard naming format is ``SYYMMDDx``, where:

- ``YY``: Year (e.g., 24 for 2024)
- ``MM``: Month (e.g., 04 for April)
- ``DD``: Day (e.g., 26 for 26th)
- ``x``: Letter suffix assigned by gracedb (varies)

See `gracedb <https://gracedb.ligo.org>`_ for details on event naming and IDs.

Examples: ``S240426s``, ``S240618ah``

Stage-by-Stage Guide
--------------------

This section explains each stage of the production workflow, referencing the script in :ref:`production-script`.

Stage 1: Project Setup
^^^^^^^^^^^^^^^^^^^^^

Create the project directory and initialize an Asimov project:

.. code-block:: bash

   mkdir -p $1
   cd $1
   asimov init my-project-$1

This creates a new Asimov project with a unique name. The project directory will store all configurations and outputs.

Stage 2: Environment Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure the RIFT environment. In production, this typically points to a specific conda environment rather than the default Singularity container:

.. code-block:: bash

   asimov configuration update rift/environment /path/to/your/rift-environment
   asimov configuration update pipelines/environment /path/to/your/rift-environment

**Why specify the environment?**

- Use a specific RIFT version for reproducibility
- Access to local build tools and debugging
- Consistent environment across all jobs

Stage 3: Infrastructure Blueprints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Apply the base infrastructure configuration. These define the cluster, accounting, and hardware requirements:

.. code-block:: bash

   asimov apply -f ${HERE}/../asimov-rift-data/defaults/production-pe.yaml
   asimov apply -f https://git.ligo.org/asimov/data/-/raw/main/defaults/cbcflow/testing.yaml
   asimov apply -f ${HERE}/../asimov-rift-data/defaults/production-pe-priors.yaml

**Key blueprints:**

1. **production-pe.yaml**: Infrastructure defaults
   - OSG (Open Science Grid) configuration
   - Accounting group (e.g., ``ligo.dev.o4.cbc.pe.rift``)
   - GPU requirements
   - Base Singularity image

2. **cbcflow/testing.yaml**: cbcflow configuration
   - Workflow manager settings

3. **production-pe-priors.yaml**: Prior distributions
   - Default prior settings for mass, spin, etc.

Stage 4: cbcflow Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

cbcflow manages the overall workflow for production PE. It coordinates multiple pipelines and handles result aggregation:

.. code-block:: bash

   git clone git@git.ligo.org:michael.williams/cbc-workflow-er16-o4b.git cbcflow-library
   (cd cbcflow-library; ./setup-cbcflow-merge-strategy.sh)
   asimov apply -f ${HERE}/cbcflow.yaml

.. warning::
   The ``cbcflow-library`` repository is **internal-use-only** and restricted to LVK personnel.

**What cbcflow does:**

- Coordinates data retrieval (graceDB)
- Manages multiple PE pipelines
- Handles result merging and aggregation
- Generates summary pages

Stage 5: Output Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure the webroot for output files:

.. code-block:: bash

   asimov configuration update condor/user ${USER}
   asimov configuration update general/webroot ${HERE}/$1/web_output/

This defines:
- The condor user for accounting
- The web directory for results

Stage 6: Localization Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Apply bayeswave localization settings:

.. code-block:: bash

   asimov apply -f ${HERE}/localize_bw.yaml

This allows bayeswave (the localization pipeline) to run on OSG.

Stage 7: Event Addition Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define a function to add events. This is the core of the multi-event pattern:

.. code-block:: bash

   function add_event () {
       # Add the event using cbcflow
       asimov apply -p cbcflow -e ${SID_HERE}

       # Apply analysis blueprints
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/get-data/o4b-production.yaml -e ${SID_HERE}
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/pe-configurator/standard.yaml -e ${SID_HERE}
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/bayeswave-psd/standard-settings.yaml -e ${SID_HERE}

       # RIFT analyses
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/rift-bbh/analysis_rift_SEOBNRv5PHM.yaml -e ${SID_HERE}
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/rift-bbh/analysis_rift_NRSur7dq4.yaml -e ${SID_HERE}

       # Bilby analyses
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/bilby-bbh/analysis_bilby_IMRPhenomXPHM-SpinTaylor.yaml -e ${SID_HERE}
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/bilby-bbh/analysis_bilby_SEOBNRv5PHM.yaml -e ${SID_HERE}
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/bilby-bbh/analysis_bilby_NRSur7Dq4.yaml -e ${SID_HERE}
   }

**What each blueprint provides:**

- **o4b-production.yaml**: Data retrieval from graceDB
- **standard.yaml**: PE configurator settings
- **standard-settings.yaml**: Bayeswave PSD settings
- **analysis_rift_SEOBNRv5PHM.yaml**: RIFT with SEOBNRv5PHM waveform
- **analysis_rift_NRSur7dq4.yaml**: RIFT with NRSur7dq4 waveform
- **bilby analyses**: Bilby with various waveforms (for cross-validation)

Stage 8: Event Loop
^^^^^^^^^^^^^^^^^^

Process multiple events in a loop:

.. code-block:: bash

   for i in S240426s S240618ah
   do
       SID_HERE=$i
       add_event ${SID_HERE}
   done

Each event is processed with all the configurations from Stage 7.

Multi-Pipeline Integration
--------------------

The production workflow integrates multiple parameter estimation pipelines to ensure robust and validated results.

Pipeline Roles
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Pipeline
     - Role
     - Description
   * - **RIFT**
     - Primary PE
     - Main parameter estimation. Supports SEOBNRv5PHM, NRSur7dq4.
   * - **bilby**
     - Cross-validation
     - Independent PE for validation and systematic errors.
   * - **bayeswave**
     - Detection & Localization
     - GW detection and sky localization.
   * - **cbcflow**
     - Workflow Manager
     - Coordinates data, pipelines, and result aggregation.


Waveform Comparison
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Waveform
     - Pipeline
     - Use Case
   * - SEOBNRv5PHM
     - RIFT + bilby
     - Standard effective Hamiltonian model
   * - NRSur7dq4
     - RIFT + bilby
     - Numerical relativity surrogate
   * - IMRPhenomXPHM
     - bilby
     - Phenomenological model

Configuration Layering
-----------------

The production workflow uses multiple configuration layers that are merged in a specific order. Later layers override earlier ones.

Layer Order
^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Layer
     - Description
   * - Infrastructure (production-pe.yaml)
     - Base hardware, cluster, accounting settings
   * - Priors (production-pe-priors.yaml)
     - Default mass/spin prior distributions
   * - cbcflow
     - Workflow manager settings
   * - Localization (localize_bw.yaml)
     - Bayeswave OSG settings
   * - Physics (analysis_rift_*.yaml)
     - Waveform-specific settings
   * - bilby
     - bilby-specific settings

**Merge Precedence**: Later layers override earlier ones. For example, the physics layer overrides the infrastructure layer.

Local vs Web Blueprints
^^^^^^^^^^^^^^^^^^^

Blueprints can come from:

1. **Web (URL)**: ``https://git.ligo.org/asimov/data/-/raw/main/defaults/production-pe.yaml``
2. **Local (file)**: ``${HERE}/../asimov-rift-data/defaults/production-pe.yaml``

**When to use each:**
- Web: Latest official defaults
- Local: Custom or event-specific configurations

Environment Setup
-----------------

Production workflows often require environments beyond the default asimov containers to support specific RIFT versions or local build tools.

Conda Environments
^^^^^^^^^^^^^^^^^

In many production cases, RIFT is run within a dedicated Conda environment. This is configured via the ``rift/environment`` and ``pipelines/environment`` settings in asimov.

.. code-block:: bash

   asimov configuration update rift/environment /path/to/your/conda/env
   asimov configuration update pipelines/environment /path/to/your/conda/env

Using a local Conda environment allows for:
- **Rapid Iteration**: Testing new code changes without rebuilding a container image.
- **Version Control**: Pinning exactly which version of RIFT and its dependencies are used for a specific production run.
- **Local Tooling**: Access to site-specific binaries or libraries.

CVMFS and Singularity
^^^^^^^^^^^^^^^^^^^^

While local environments are useful for development, production results are typically validated using containers stored on CVMFS (CernVM-FS).

- **Singularity Images**: Asimov can be configured to pull images from CVMFS paths (e.g., ``/cvmfs/singularity.opensciencegrid.org/...``).
- **Containerization**: Ensures that the environment is identical across all OSG nodes, eliminating "it works on my machine" issues.

Output and Webroot
^^^^^^^^^^^^^^^^^

The ``general/webroot`` setting defines where the final analysis results, plots, and summary pages are stored. In production, this is typically a public-facing directory on an IGWN cluster.

.. code-block:: bash

   asimov configuration update general/webroot /home/${USER}/public_html/asimov_results/project_name/

This allows collaborators to view the progress and results of the production run via a web browser without needing direct shell access to the cluster.

Building and Submitting
----------------------

Once the project is initialized and all event blueprints have been applied, the analysis must be built and submitted to the cluster.

1. **Build the Analysis**:
   The ``asimov manage build`` command generates the necessary configuration files (e.g., ``rift.ini``) and constructs the Condor DAGs for all events and pipelines.

   .. code-block:: bash

      asimov manage build

2. **Submit to Cluster**:
   The ``asimov manage submit`` command submits the generated DAGs to the Condor cluster.

   .. code-block:: bash

      asimov manage submit

Monitoring and Aggregation
^^^^^^^^^^^^^^^^^^^^^^^^^

In a production environment, monitoring multiple events and pipelines requires coordinated tracking.

**Asimov Monitoring**
Asimov provides built-in tools to track the status of jobs:

- ``asimov monitor``: Manually check the status of the current project's jobs.
- ``asimov start``: An automated monitoring mode that tracks progress and handles certain failures.

**cbcflow Integration**
For production-scale runs, ``cbcflow`` is used to manage the overall state and aggregate results. It provides:

- **Centralized Tracking**: Tracks the completion of multiple pipelines (RIFT, bilby, bayeswave) across multiple events.
- **Result Aggregation**: Automatically gathers posterior samples and logs from various nodes.
- **Summary Pages**: Generates HTML summary pages for the entire production run, accessible via the configured ``webroot``.

**Multi-Event Result Analysis**
After completion, results from multiple events can be analyzed collectively (e.g., for population studies) using the aggregated datasets provided by the ``cbcflow`` integration.

.. _production-script:

Full Production Script
---------------------

This is the generalized production script used in this documentation. It is based on ``demo_multiple_o4b_HL.sh`` from the integration testing directory.

.. code-block:: bash

   #!/bin/bash
   # =============================================================================
   # demo_multiple_o4b_HL.sh - Production Workflow for Multiple Events
   # =============================================================================
   # This script demonstrates the production workflow for running RIFT analyses
   # on multiple events using the asimov framework.
   #
   # IMPORTANT: This is a generalized prototype. The actual production workflow may differ.
   # Refer to the latest asimov documentation for up-to-date commands.
   #
   # Usage: ./demo_multiple_o4b_HL.sh <project_directory>
   # Example: ./demo_multiple_o4b_HL.sh my-production-project
   # =============================================================================

   # Variables
   # --------
   # HERE: Current directory (where the script is run from)
   # $1: Project directory name (passed as argument)
   # USER: Current user (for accounting)
   HERE=`pwd`

   # -----------------------------------------------------------------------------
   # Stage 1: Project Setup
   # -----------------------------------------------------------------------------
   mkdir -p $1
   cd $1
   asimov init my-project-$1

   # -----------------------------------------------------------------------------
   # Stage 2: Environment Configuration
   # -----------------------------------------------------------------------------
   asimov configuration update rift/environment /path/to/your/rift-environment
   asimov configuration update pipelines/environment /path/to/your/rift-environment

   # -----------------------------------------------------------------------------
   # Stage 3: Infrastructure Blueprints
   # -----------------------------------------------------------------------------
   asimov apply -f ${HERE}/../asimov-rift-data/defaults/production-pe.yaml
   asimov apply -f https://git.ligo.org/asimov/data/-/raw/main/defaults/cbcflow/testing.yaml
   asimov apply -f ${HERE}/../asimov-rift-data/defaults/production-pe-priors.yaml

   # -----------------------------------------------------------------------------
   # Stage 4: cbcflow Integration
   # -----------------------------------------------------------------------------
   git clone git@git.ligo.org:michael.williams/cbc-workflow-er16-o4b.git cbcflow-library
   (cd cbcflow-library; ./setup-cbcflow-merge-strategy.sh)
   asimov apply -f ${HERE}/cbcflow.yaml
   asimov configuration update condor/user ${USER}

   # -----------------------------------------------------------------------------
   # Stage 5: Output Configuration
   # -----------------------------------------------------------------------------
   asimov configuration update general/webroot ${HERE}/$1/web_output/

   # -----------------------------------------------------------------------------
   # Stage 6: Localization Configuration
   # -----------------------------------------------------------------------------
   asimov apply -f ${HERE}/localize_bw.yaml

   # -----------------------------------------------------------------------------
   # Stage 7: Event Addition Function
   # -----------------------------------------------------------------------------
   function add_event () {
       asimov apply -p cbcflow -e ${SID_HERE}
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/get-data/o4b-production.yaml -e ${SID_HERE}
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/pe-configurator/standard.yaml -e ${SID_HERE}
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/bayeswave-psd/standard-settings.yaml -e ${SID_HERE}
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/rift-bbh/analysis_rift_SEOBNRv5PHM.yaml -e ${SID_HERE}
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/rift-bbh/analysis_rift_NRSur7dq4.yaml -e ${SID_HERE}
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/bilby-bbh/analysis_bilby_IMRPhenomXPHM-SpinTaylor.yaml -e ${SID_HERE}
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/bilby-bbh/analysis_bilby_SEOBNRv5PHM.yaml -e ${SID_HERE}
       asimov apply -f ${HERE}/../asimov-rift-data/analyses/bilby-bbh/analysis_bilby_NRSur7Dq4.yaml -e ${SID_HERE}
       GID_HERE=`grep coinc.xml cbcflow-library/${SID_HERE}*.json | head -n 1| tr '/' ' ' | awk '{print $6}' `
   }

   # -----------------------------------------------------------------------------
   # Stage 8: Event Loop
   # -----------------------------------------------------------------------------
   for i in S240426s S240618ah  # Add more events here
   do
       SID_HERE=$i
       add_event ${SID_HERE}
   done

References
----------

* **RIFT GitHub Repository**: `oshaughnessy-junior/research-projects-rit <https://github.com/oshaughnessy-junior/research-projects-rit>`_
* **cbcflow Documentation**: `cbcflow <https://cbc.docs.ligo.org/projects/cbcflow/index.html>`_
* **Asimov Documentation**: `asimov <https://asimov.docs.ligo.org/asimov/master/index.html>`_
* **Production Script Source**: `/Users/rossma/LVK/O4/rapidpe_rift_review_o4/o4b/integration_testing/demo_multiple_o4b_HL.sh`
* **Simple Workflow**: :doc:`asimov_simple`
* **RIFT Asimov API**: :doc:`../executables/rift_asimov`
* **ILE Documentation**: :doc:`../../executables/integrate_likelihood_extrinsic_batchmode`
* **CIP Documentation**: :doc:`../../executables/util_ConstructIntrinsicPosterior_GenericCoordinates`