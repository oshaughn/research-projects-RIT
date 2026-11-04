xmlutils
========

The ``RIFT.misc.xmlutils`` module provides the essential bridge between RIFT's internal data structures and the standardized LIGO-LW XML format. This ensures that RIFT outputs are compatible with other LIGO/Virgo/KAGRA data analysis tools.

Data Persistence Logic
----------------------

The module focuses on two primary data persistence tasks: archiving simulation parameters and recording likelihood results.

1. **Archiving Samples**:
   The ``append_samples_to_xmldoc`` function maps sampled parameter sets to the ``SimInspiralTable``. This is critical for preserving the "provenance" of an inference run, allowing researchers to see exactly which points in parameter space were evaluated.

2. **Recording Results**:
   The ``append_likelihood_result_to_xmldoc`` function records the resulting log-likelihood and convergence metrics into the ``SnglInspiralTable``.

The Parameter Mapping (CMAP)
----------------------------

Because RIFT uses intuitive parameter names (e.g., ``right_ascension``, ``declination``) that may differ from the formal LAL table column names (e.g., ``longitude``, ``latitude``), the module employs a central mapping dictionary called ``CMAP``.

This mapping ensures that:
- Parameters are correctly assigned to the corresponding LAL table columns.
- Specialized transformation functions (like ``assign_time``) are applied during the mapping process to handle unit conversions or time-stamping.

Database Integration
--------------------

For very large simulation sets where XML files become unwieldy, ``xmlutils`` provides SQLite integration:
- ``db_to_samples``: Efficiently extracts sets of parameters and results from a SQL database, returning them as Python named-tuples.
- ``db_identify_param``: Allows quick lookup of specific process parameters.

API Reference
-------------

.. automodule:: RIFT.misc.xmlutils
    :members:
    :undoc-members:
    :show-inheritance:
