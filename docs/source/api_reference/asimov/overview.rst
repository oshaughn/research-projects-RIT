Asimov Interface Overview
==========================

.. contents::
   :local:

This section describes the integration of the RIFT pipeline into the ``asimov`` production framework.

Overview
--------

The ``asimov`` interface allows RIFT to be deployed as a production pipeline. This involves a Python wrapper (``rift.py``) and a configuration template (``rift.ini``) that map ``asimov`` ledger metadata to RIFT-specific execution arguments.

API Reference
-------------

The core integration is handled by:

* ``RIFT/asimov/rift.py``: The pipeline class that manages DAG construction, submission, and monitoring.
* ``RIFT/asimov/rift.ini``: The Jinja2 template used to generate the runtime configuration.

Asimov Use Case: Production PE
------------------------------

To run RIFT through ``asimov``, a user typically defines a production specification in a YAML file (the "ledger"). This specification is then mapped by ``rift.py`` and ``rift.ini`` into the final command line arguments for :doc:`../../executables/util_RIFT_pseudo_pipe`.

Mapping the Ledger to RIFT
--------------------------

The mapping process determines which RIFT flags are set based on the ledger's ``meta`` section.

**Key Mapping Examples:**

* **Waveform Approximant**: The ``waveform: approximant`` field in the ledger maps to the ``--approx`` flag in RIFT.
* **Physics Assumptions**: Fields under ``likelihood: assume`` (e.g., ``no spin``, ``precessing``, ``eccentric``) map to corresponding ``--assume-xxx`` flags.
* **Sampling Settings**: Fields under ``sampler: cip`` (e.g., ``sampling method``) map to ``cip-sampler-method`` in the ``ini`` file.
* **Priors**: Prior ranges defined in the ledger are mapped to specific ``ile-xxx-prior`` settings in the ``ini`` file.

Configuration File Note
-----------------------

It is important to note that the ``RIFT/asimov/rift.ini`` file contains many legacy or placeholder entries. The actual behavior of the pipeline is determined by which values are explicitly parsed by :doc:`../../executables/util_RIFT_pseudo_pipe` and :doc:`../../executables/helper_LDG_Events`. 

Users should focus on the **actionable items**—the flags and configuration keys that the low-level RIFT code actually reads—rather than the entirety of the ``ini`` template.

Deployment Workflow
-------------------

1. **Define Production**: Create a YAML production spec in the ``asimov`` ledger.
2. **Build**: Run ``asimov manage build``. The ``Rift`` pipeline class identifies the production and uses ``rift.ini`` to generate a specific configuration.
3. **Submit**: The ``Rift`` pipeline calls ``util_RIFT_pseudo_pipe.py`` to build a Condor DAG and then submits it to the cluster.
4. **Monitor**: ``asimov`` monitors the job status via ``detect_completion()``, which looks for the final posterior sample files.
