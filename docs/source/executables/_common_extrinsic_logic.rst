Functional Logic
----------------

1. **Data Loading**: Loads strain data from the cache and PSDs for each detector.
2. **Precomputation**: Calls ``factored_likelihood.PrecomputeLikelihoodTerms`` to compute terms that are independent of the extrinsic parameters.
3. **Likelihood Integration**:
    - Defines a likelihood function that computes the factored log-likelihood for specific extrinsic parameters.
    - Evaluates this function over the provided parameter grid (or uses a Monte Carlo sampler for marginalization).
    - Supports adaptive sampling or batch evaluation to focus on regions of high likelihood.
4. **Marginalization (Optional)**: If configured to integrate, computes the integral of the likelihood over the extrinsic parameters:
   
   .. math::

      \mathcal{L}_{marg} = \int \mathcal{L}(\theta_{ext}) d\theta_{ext}

5. **Maximization (Optional)**: If ``--maximize-only`` is set, it uses ``scipy.optimize.fmin`` to find the maximum likelihood point (MLP) in the extrinsic space.
