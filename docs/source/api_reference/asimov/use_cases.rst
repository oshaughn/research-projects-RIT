Asimov Use Cases and Mapping
==============================

.. contents::
   :local:

This document explains how to construct a YAML production specification for ``asimov`` and how those values map to the RIFT pipeline.

User-Facing Use Cases
--------------------

RIFT is integrated into ``asimov`` to support several primary analysis workflows:

1. **Production-style PE**: Full-scale parameter estimation runs.
2. **PP Plots**: Posterior predictive checks for consistency.
3. **Single-event Analysis**: Deep-dives into individual gravitational wave events.
4. **Population Analyses**: Statistical studies across multiple events.

Writing a Template YAML Specification
-------------------------------------

A production specification in the ``asimov`` ledger defines the "what" and "how" of a RIFT run. Below is a simplified example of a production entry:

.. code-block:: yaml

   - Prod_RIFT_01:
       pipeline: rift
       waveform:
          approximant: IMRPhenomPv3
          maximum mode: 4
       likelihood:
          assume:
             precessing: True
             eccentric: True
          start frequency: 20
       sampler:
          cip:
             sampling method: "AV"
             explode jobs: 3
          ile:
             n eff: 10
       priors:
          spin 1:
             maximum: 0.99
          luminosity distance:
             maximum: 5000
       needs:
         - Prod_Previous_PE

Mapping to RIFT Configuration
------------------------------

The values defined in the YAML specification are processed by ``rift.py`` and the ``rift.ini`` template to produce the final execution environment.

**Mapping Table**

+---------------------------+--------------------------------+------------------------------------------+
| Ledger Field              | RIFT Configuration/Flag         | Description                              |
+===========================+================================+==========================================+
| ``pipeline: rift``        | (Internal Check)               | Activates the ``Rift`` pipeline class     |
+---------------------------+--------------------------------+------------------------------------------+
| ``waveform: approximant`` | ``--approx <val>``              | Sets the waveform approximant           |
+---------------------------+--------------------------------+------------------------------------------+
| ``waveform: maximum mode``| ``l-max=<val>``                | Sets the maximum harmonic mode           |
+---------------------------+--------------------------------+------------------------------------------+
| ``likelihood: assume: precessing`` | ``--assume-precessing``  | Enables precessing spin models          |
+---------------------------+--------------------------------+------------------------------------------+
| ``likelihood: assume: eccentric``  | ``--assume-eccentric``    | Enables eccentric orbit models          |
+---------------------------+--------------------------------+------------------------------------------+
| ``sampler: cip: sampling method`` | ``cip-sampler-method=<val>`` | Determines the CIP sampling algorithm  |
+---------------------------+--------------------------------+------------------------------------------+
| ``sampler: ile: n eff``    | ``ile-n-eff=<val>``            | Sets the target number of effective samples|
+---------------------------+--------------------------------+------------------------------------------+
| ``priors: spin 1: maximum``| ``a_spin1-max=<val>``           | Sets the upper bound for spin 1          |
+---------------------------+--------------------------------+------------------------------------------+
| ``priors: luminosity distance: maximum`` | ``distance-max=<val>`` | Sets the maximum distance            |
+---------------------------+--------------------------------+------------------------------------------+

Actionable Configuration
-------------------------

When modifying the ``RIFT/asimov/rift.ini`` template, keep in mind that the low-level code (``util_RIFT_pseudo_pipe.py``) only consumes a subset of the provided keys. 

**Priority Actionable Items:**

* **Approximant and Modes**: Crucial for the underlying waveform generation.
* **Assumption Flags**: Directly control the physics of the likelihood calculation.
* **CIP/ILE Parameters**: Control the convergence and efficiency of the sampling process.
* **Priors**: Ensure the search space is correctly bounded.

Everything else in the ``ini`` file is generally considered boilerplate or legacy filler.
