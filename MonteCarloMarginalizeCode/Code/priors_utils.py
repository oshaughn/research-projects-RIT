
import numpy as np
import scipy

# Will Farr's simplified distance prior on d_L, out to z~ 4
# note it is not normalized, and the normalization depends on the d_max of interest 

will_cosmo_const = np.array([1.05491298, 0.855908921, 0.170816006, 0.0109350381, 0.000343230488])
def dist_prior_pseudo_cosmo(dL,nm=1):
    return nm*4* np.pi * dL**2 / np.polyval( will_cosmo_const[::-1],dL)


