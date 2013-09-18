import sys
from optparse import OptionParser

import numpy

from glue.lal import Cache
import lalsimutils

"""
Simple program illustrating the precomputation step and calls to the
likelihood function.
"""

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>, Chris Pankow <pankow@gravity.phys.uwm.edu>"

from factored_likelihood import *

optp = OptionParser()
optp.add_option("-c", "--cache-file", default=None, help="LIGO cache file containing all data needed.")
optp.add_option("-C", "--channel-name", action="append", help="instrument=channel-name, e.g. H1=FAKE-STRAIN. Can be given multiple times for different instruments.")
opts, args = optp.parse_args()

det_dict = {}
if opts.channel_name is not None and opts.cache_file is None:
    print >>sys.stderr, "Cache file required when requesting channel data."	
    exit(-1)
elif opts.channel_name is not None:
    det_dict = dict(map(lambda cname: cname.split("="), opts.channel_name))

Niter = 5 # Number of times to call likelihood function
Tmax = 38. # max ref. time
Tmin = 36. # min ref. time
Dmax = 110. * 1.e6 * lal.LAL_PC_SI # max ref. time
Dmin = 90. * 1.e6 * lal.LAL_PC_SI # min ref. time

#
# Produce data with a coherent signal in H1, L1, V1
#
data_dict = {}
if len(det_dict) > 0:
    with open(opts.cache_file) as cfile:
        cachef = Cache.fromfile(cfile)

    for d, chan in det_dict.iteritems():
        data_dict[d] = lalsimutils.frame_data_to_hoff(cachef, chan)
else:

    Psig = ChooseWaveformParams(fmin = 10., radec=True, theta=1.2, phi=2.4,
            detector='H1', dist=100.*1.e6*lal.LAL_PC_SI)
    df = findDeltaF(Psig)
    Psig.deltaF = df
    data_dict['H1'] = non_herm_hoff(Psig)
    Psig.detector = 'L1'
    data_dict['L1'] = non_herm_hoff(Psig)
    Psig.detector = 'V1'
    data_dict['V1'] = non_herm_hoff(Psig)

# TODO: Read PSD from XML
psd_dict = {}
analyticPSD_Q = True # For simplicity, using an analytic PSD
psd_dict['H1'] = lal.LIGOIPsd
psd_dict['L1'] = lal.LIGOIPsd
psd_dict['V1'] = lal.LIGOIPsd

# Struct to hold template parameters
P = ChooseWaveformParams(fmin = 40., dist=100.*1.e6*lal.LAL_PC_SI, deltaF=df)
Lmax = 2 # sets which modes to include

#
# Perform the Precompute stage
#
rholms_intp, crossTerms, rholms = PrecomputeLikelihoodTerms(P, data_dict,
        psd_dict, Lmax, analyticPSD_Q)
print "Finished Precomputation..."

#
# Call the likelihood function for various extrinsic parameter values
#
def likelihood_function(phi, theta, tref, phiref, incl, psi, dist):
    lnL = numpy.zeros(phi.shape)
    i = 0
    for ph, th, tr, phr, ic, ps, di in zip(phi, theta, tref, phiref, incl, psi, dist):
        P.phi = ph # right ascension
        P.theta = th # declination
        P.tref = tr # ref. time
        P.phiref = phr # ref. orbital phase
        P.incl = ic # inclination
        P.psi = ps # polarization angle
        P.dist = di # luminosity distance

        lnL[i] = FactoredLogLikelihood(P, rholms_intp, crossTerms, Lmax)
        i+=1

        print "For (RA, DEC, tref, phiref, incl, psi, dist) ="
        print "\t", P.phi, P.theta, P.tref, P.phiref, P.incl, P.psi, P.dist
        print "\tlog likelihood is %g:" % lnL[-1]
    return numpy.exp(lnL)

import mcsampler
sampler = mcsampler.MCSampler()

# Sampling distribution
def uniform_samp(a, b, x):
   if type(x) is float:
       return 1/(b-a)
   else:
       return numpy.ones(x.shape[0])/(b-a)

# set up bounds on parameters
# Polarization angle
psi_min, psi_max = 0, 2*numpy.pi
# RA and dec
ra_min, ra_max = 0, 2*numpy.pi
dec_min, dec_max = -numpy.pi/2, numpy.pi/2
# Reference time
tref_min, tref_max = Tmin, Tmax
# Inclination angle
inc_min, inc_max = -numpy.pi/2, numpy.pi/2
# orbital phi
phi_min, phi_max = 0, 2*numpy.pi
# distance
dist_min, dist_max = Dmin, Dmax

import functools
# Uniform sampling, auto-cdf inverse
sampler.add_parameter("psi", functools.partial(uniform_samp, psi_min, psi_max), None, psi_min, psi_max)
sampler.add_parameter("ra", functools.partial(uniform_samp, ra_min, ra_max), None, ra_min, ra_max)
sampler.add_parameter("dec", functools.partial(uniform_samp, dec_min, dec_max), None, dec_min, dec_max)
sampler.add_parameter("tref", functools.partial(uniform_samp, tref_min, tref_max), None, tref_min, tref_max)
sampler.add_parameter("phi", functools.partial(uniform_samp, phi_min, phi_max), None, phi_min, phi_max)
sampler.add_parameter("inc", functools.partial(uniform_samp, inc_min, inc_max), None, inc_min, inc_max)
sampler.add_parameter("dist", functools.partial(uniform_samp, dist_min, dist_max), None, dist_min, dist_max)

res, var = sampler.integrate(likelihood_function, 1e6, "ra", "dec", "tref", "phi", "inc", "psi", "dist")
print res, numpy.sqrt(var)
