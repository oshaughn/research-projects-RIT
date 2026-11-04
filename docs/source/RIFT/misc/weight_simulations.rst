weight_simulations
===================

The ``RIFT.misc.weight_simulations`` module provides a centralized utility for aggregating results from multiple simulations. When RIFT performs multiple ILE (Inspiral-LAL-Engine) runs—either at the same intrinsic parameters with different draw counts or across different points in the intrinsic parameter space—the results must be combined into a single composite value.

The Weighted Average Framework
------------------------------

The core of the module is the ``AverageSimulationWeights`` function, which computes a set of weights $W_k$ such that a composite integral $I$ can be calculated as:

$$I = \sum_{k} W_k I_k$$

where $I_k$ is the result of the $k$-th simulation.

Weighting Strategies
-------------------

The module supports different strategies for determining the weights based on the `weight_at_intrinsic` parameter:

1. **Variance-Minimizing (``sigma``)**:
   Weights are assigned proportional to the inverse of the variance:
   $$W_k \propto \frac{1}{\sigma_k^2}$$
   This is the statistically optimal approach for combining independent measurements with known uncertainties, ensuring that the most precise simulations have the greatest influence on the final result.

2. **Uniform (``uniform``)**:
   Each simulation is assigned an equal weight ($W_k = 1/N$), treating all runs as equally reliable regardless of their individual variance.

Prior Volume Integration
------------------------

To correctly account for the distribution of simulations across the parameter space, the module can incorporate prior volumes. If a ``prior_volume_list`` is provided (containing the pre-integrated prior volume $\Delta P_k$ for each simulation), the weights are scaled accordingly:

$$W_{final} = W_{strategy} \times \Delta P_k$$

This ensures that simulations covering larger regions of the intrinsic prior volume contribute proportionally more to the composite result, effectively performing a discrete integration over the prior.

API Reference
-------------

.. automodule:: RIFT.misc.weight_simulations
    :members:
    :undoc-members:
    :show-inheritance:
