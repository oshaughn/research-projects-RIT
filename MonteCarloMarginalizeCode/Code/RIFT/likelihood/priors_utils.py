
import numpy as np
import scipy.integrate

xpy_default=np
has_cupy = False
try:
    import cupy   # can proceed even if cupy doesn't actually work
    junk_to_check_installed = cupy.array(5)
    xpy_default=cupy
    has_cupy=True
except:
    has_cupy=False  # just to be sure
# try:
#     import numba
#     from numba import vectorize, complex128, float64, int64
#     numba_on = True
#     print(" Numba on (priors_utils) ")
    

# except:
#     numba_on = False
#     print(" Numba off (priors_utils) ")

# https://git.ligo.org/RatesAndPopulations/lalinfsamplereweighting/blob/reviewed-post-O2/approxprior.py
will_cosmo_const = np.array( [ 1.012306, 1.136740, 0.262462, 0.016732, 0.000387 ])
p_in = will_cosmo_const[::-1]
if has_cupy:
    p_in = cupy.asarray(p_in)
def dist_prior_pseudo_cosmo(dL,nm=1,xpy=np,p_in=p_in):
    """
    dist_prior_pseudo_cosmo.  dL needs to be in Gpc for the polynomial.
    By default, our code works with distances in Mpc.  So we divide by 1e3

     Will Farr's simplified distance prior on d_L, out to z~ 4
     note it is not normalized, and the normalization depends on the d_max of interest 
    
    """
    if isinstance(dL, np.ndarray):
        return nm*4* np.pi * dL**2 / np.polyval( will_cosmo_const[::-1], dL/1e3)
    return nm*4* np.pi * dL**2 / xpy.polyval( p_in,dL/1e3)


def dist_prior_pseudo_cosmo_eval_norm(dLmin,dLmax):
    return 1./scipy.integrate.quad(dist_prior_pseudo_cosmo, dLmin,dLmax)[0]

