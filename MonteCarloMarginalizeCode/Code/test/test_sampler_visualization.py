
# GOAL
#  -  visualize several samplers, to illustrate
#     how we approximate the likelihood in our sampler but preserve generality
#
# PLAN
#  - uniform sampler: make a histogram and
from lalsimutils import *
from matplotlib import pylab as plt

import RIFT.integrators.mcsampler as mcsampler
sampler = mcsampler.MCSampler()
# Sampling distribution
def uniform_samp(a, b, x):
   if type(x) is float:
       return 1/(b-a)
   else:
       return np.ones(x.shape[0])/(b-a)
import functools
# set up bounds on parameters
# Polarization angle
psi_min, psi_max = 0, 2*np.pi
# RA and dec
ra_min, ra_max = 0, 2*np.pi
dec_min, dec_max = -np.pi/2, np.pi/2
# Reference time
tref_min, tref_max = -0.1, 0.1
# Inclination angle
inc_min, inc_max = -np.pi/2, np.pi/2
# orbital phi
phi_min, phi_max = 0, 2*np.pi
# distance
dist_min, dist_max = 0.1, 100
# Uniform sampling, auto-cdf inverse
sampler.add_parameter("psi", functools.partial(uniform_samp, psi_min, psi_max), None, psi_min, psi_max)
sampler.add_parameter("ra", functools.partial(uniform_samp, ra_min, ra_max), None, ra_min, ra_max)
sampler.add_parameter("dec", functools.partial(uniform_samp, dec_min, dec_max), None, dec_min, dec_max)
sampler.add_parameter("tref", functools.partial(uniform_samp, tref_min, tref_max), None, tref_min, tref_max)
sampler.add_parameter("phi", functools.partial(uniform_samp, phi_min, phi_max), None, phi_min, phi_max)
sampler.add_parameter("inc", functools.partial(uniform_samp, inc_min, inc_max), None, inc_min, inc_max)
sampler.add_parameter("dist", functools.partial(uniform_samp, dist_min, dist_max), None, dist_min, dist_max)


ptarray = sampler.draw( 10, ("tref", "ra", "dec", "inc", "phi", "psi", "d"))
print(ptarray)
plt.hist(ptarray,50,normed=1)
plt.show()
