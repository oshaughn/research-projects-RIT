# weight_simulations.py
#
#  Library to provide black-box, single-point-of-contact tool to weight results of multiple ILE runs (at multiple intrinsic parameters)
#  to produce a composite result.  Provides
#      - weighting by priors over intrinsic parameters
#      - uniform or variance-minimizing 
#      

import numpy as np

def AverageSimulationWeights(param_list, length_list,sigma_list, prior_volume_list=None, weight_at_intrinsic="sigma"):
    """
    AverageSimulationWeights returns the factors W_k appearing in a simple average over
    N simulations, for example for the integral value
                   I = \sum_k W_k I_k
    For backward compatibility, weights can be chosen that depend on
         - the number of draws for each simulation
         - the variance of each simulation
         - the parameter values of each simulation
         - priors,  *if* the priors are provided as *pre-integrated volumes* (i.e., an array,
           one member per simulation).  The user is assumed to have already identified volumes surrounding each simulation and 
           integrated the prior accordingly.  The user is responsible for the (in)accuracy of the resulting average, if the prior and likelihood
           vary significantly versus intrinsic parameters.
    Specific implementations and use cases include:
         - single ILE run at fixed parameters, with different numbers of draws, either by variance-minimizing
                 I = (\sum_k  I_k /\sigma_k^2) / \sum_k 1/\sigma_k^2    
           or by a simple average over the N simulations
                 I = (\sum_k  I_k)/N         N=# of simulations

         - multiple ILE runs at different parameters, accounting for an intrinsic prior volume \Delta P_k for each simulation
           (normalized relative to the intrinsic prior: \sum_k \Delta P_k = (prior probability of lying in ellipsoid) << 1)
                 I =( \sum_k I_k \Delta P_k /\sigma_k^2)/ \sum_k 1/\sigma_k^2
    """

    if np.min(sigma_list) <0.:
        return np.ones(len(sigma_list))   # Allow the user to be a moron

    weights = np.ones(len(sigma_list))

    # Weighting at intrinsic parameters
    if (weight_at_intrinsic=="sigma"):
        weights *= 1./np.power(sigma_list,2)
        weights *= 1./np.sum(weights)          # explicitly renormalize *by intent*
    elif (weight_at_intrinsic=="uniform") or True:
        weights *= 1./len(weights)
    
    
    # Factoring in prior volume.  Note the 'box' average is less *significantly* less accurate than interpolating over prior parameters
    if prior_volume_list:
        weights *= prior_volume_list

    return weights

