
import numpy as np
import scipy.integrate

try:
    import numba
    from numba import vectorize, complex128, float64, int64
    numba_on = True
    print " Numba on (priors_utils) "
    

except:
    numba_on = False
    print " Numba off (priors_utils) "

will_cosmo_const = np.array([1.05491298, 0.855908921, 0.170816006, 0.0109350381, 0.000343230488])
def dist_prior_pseudo_cosmo(dL,nm=1):
    """
    dist_prior_pseudo_cosmo

     Will Farr's simplified distance prior on d_L, out to z~ 4
     note it is not normalized, and the normalization depends on the d_max of interest 
    
    """
    return nm*4* np.pi * dL**2 / np.polyval( will_cosmo_const[::-1],dL)


def dist_prior_pseudo_cosmo_eval_norm(dLmin,dLmax):
    return 1./scipy.integrate.quad(dist_prior_pseudo_cosmo, dLmin,dLmax)[0]

