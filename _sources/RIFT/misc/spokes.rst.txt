spokes
======

The ``RIFT.misc.spokes`` module implements the "Spoke" system, a specialized framework for managing and refining 1D slices of the parameter space. In RIFT, a "spoke" typically refers to a grid of simulations where the total mass is varied while all other intrinsic parameters (mass ratio, spins, and eccentricity) are held constant.

The Spoke Concept
-----------------

The spoke system is used to efficiently find the maximum likelihood for a specific set of intrinsic parameters. Instead of sampling the entire multi-dimensional space at high resolution, RIFT identifies a "spoke" (a 1D line in parameter space) and iteratively refines the grid along that line to pinpoint the peak of the likelihood.

Likelihood Refinement
---------------------

The module provides tools to analyze 1D likelihood grids and suggest higher-resolution sampling regions:

1. **Peak Fitting**: The ``FitSpokeNearPeak`` function identifies the region surrounding the likelihood maximum and fits a quadratic polynomial to the data. This allows for a sub-grid estimate of the peak position and value.
2. **Iterative Refinement**: The ``Refine`` function implements the logic for grid adaptation. It can:
   - **Extend**: If the peak is found at the edge of the current grid, it extends the grid in that direction.
   - **Narrow**: If the peak is interior, it suggests a new, denser grid centered on the peak.
   - **Fallback**: If the data is too sparse or flat, it provides safe fallback behavior to avoid numerical instability.

Spoke Identification and I/O
----------------------------

To manage thousands of spokes, the module uses a labeling system:
- **Spoke Labels**: The ``ChooseWaveformParams_to_spoke_label`` function creates a unique identifier based on the intrinsic parameters. This allows RIFT to group simulations belonging to the same spoke across different files.
- **Data Loading**: ``LoadSpokeDAT`` and ``LoadSpokeXML`` handle the ingestion of likelihood results and simulation parameters, automatically grouping them into their respective spokes.

Consolidation and Cleaning
--------------------------

When multiple simulations are performed at the same total mass within a single spoke, the results must be combined. The ``CleanSpokeEntries`` function performs this consolidation by:
1. Grouping entries by total mass.
2. Applying simulation weights (via ``RIFT.misc.weight_simulations``) to calculate a weighted average of the log-likelihoods.
3. Computing a net uncertainty ($\sigma$) for the consolidated entry.

API Reference
-------------

.. automodule:: RIFT.misc.spokes
    :members:
    :undoc-members:
    :show-inheritance:
