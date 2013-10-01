import sys
from optparse import OptionParser

import numpy
from matplotlib import pylab as plt

from glue.lal import Cache
import lalsimutils

"""
test_like_and_samp.py:  Testing the likelihood evaluation and sampler, working in conjunction

"""

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>, Chris Pankow <pankow@gravity.phys.uwm.edu>, R. O'Shaughnessy"

from factored_likelihood import *

rosUseDifferentWaveformLengths = False    
rosUseRandomTemplateStartingFrequency = False

rosUseTargetedDistance = True
rosShowSamplerInputDistributions = True

theEpochFiducial = lal.LIGOTimeGPS(1064023405.000000000)   # 2013-09-24 early am 

optp = OptionParser()
optp.add_option("-c", "--cache-file", default=None, help="LIGO cache file containing all data needed.")
optp.add_option("-C", "--channel-name", action="append", help="instrument=channel-name, e.g. H1=FAKE-STRAIN. Can be given multiple times for different instruments.")
opts, args = optp.parse_args()

det_dict = {}
rhoExpected ={}
if opts.channel_name is not None and opts.cache_file is None:
    print >>sys.stderr, "Cache file required when requesting channel data."	
    exit(-1)
elif opts.channel_name is not None:
    det_dict = dict(map(lambda cname: cname.split("="), opts.channel_name))

Niter = 5 # Number of times to call likelihood function
Tmax = 0.05 # max ref. time
Tmin = -0.05 # min ref. time
Dmax = 110. * 1.e6 * lal.LAL_PC_SI # max ref. time
Dmin = 1. * 1.e6 * lal.LAL_PC_SI   # min ref. time

fminWavesSignal = 25
if rosUseDifferentWaveformLengths: 
    fminWavesTemplate = fminWavesSignal+0.005
else:
    if rosUseRandomTemplateStartingFrequency:
         print "   --- Generating a random template starting frequency  ---- " 
         fminWavesTemplate = fminWavesSignal+5.*np.random.random_sample()
    else:
        fminWavesTemplate = fminWavesSignal

#
# Produce data with a coherent signal in H1, L1, V1
#
data_dict = {}
if len(det_dict) > 0:
    for d, chan in det_dict.iteritems():
        data_dict[d] = lalsimutils.frame_data_to_hoff(opts.cache_file, chan)
else:
    m1 = 4*lal.LAL_MSUN_SI
    m2 = 3*lal.LAL_MSUN_SI

    Psig = ChooseWaveformParams(
        m1 = m1,m2 =m2,
        fmin = 15, 
                                radec=True, theta=1.2, phi=2.4,
                                detector='H1', 
                                dist=25.*1.e6*lal.LAL_PC_SI,
                                tref = theEpochFiducial
                                )
    df = findDeltaF(Psig)
    Psig.print_params()
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

print " == Data report == "
detectors = data_dict.keys()
rho2Net = 0
print  " Amplitude report :"
fSample = 4096
fminSNR =30
for det in detectors:
    IP = ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=Psig.deltaF,psd=psd_dict[det])
    rhoExpected[det] = rhoDet = IP.norm(data_dict[det])
    rho2Net += rhoDet*rhoDet
    print det, " rho = ", rhoDet
print "Network : ", np.sqrt(rho2Net)



# Struct to hold template parameters
ampO =0 # sets which modes to include in the physical signal
Lmax = 2 # sets which modes to include
fref = 100
P = ChooseWaveformParams(fmin=fminWavesTemplate, radec=False, incl=0.0,phiref=0.0, theta=0.0, phi=0,psi=0.0,
         m1=m1,m2=m2,
         ampO=ampO,
         fref=fref,
         tref=theEpochFiducial,
         deltaT=1./fSample,
         dist=100*1.e6*lal.LAL_PC_SI,
         deltaF=df)

#
# Perform the Precompute stage
#
rholms_intp, crossTerms, rholms, epoch_post = PrecomputeLikelihoodTerms(theEpochFiducial,P, data_dict,psd_dict, Lmax, analyticPSD_Q)
print "Finished Precomputation..."


#
# Call the likelihood function for various extrinsic parameter values
#
nEvals = 0
def likelihood_function(phi, theta, tref, phiref, incl, psi, dist):
    global nEvals
    lnL = numpy.zeros(phi.shape)
    i = 0
    print " Likelihood results :  "
    print " iteration  lnL   sqrt(2max(lnL))  sqrt(2 lnLmarg)   <lnL> "
    for ph, th, tr, phr, ic, ps, di in zip(phi, theta, tref, phiref, incl, psi, dist):
        P.phi = ph # right ascension
        P.theta = th # declination
        P.tref = theEpochFiducial + tr # ref. time (rel to epoch for data taking)
        P.phiref = phr # ref. orbital phase
        P.incl = ic # inclination
        P.psi = ps # polarization angle
        P.dist = di # luminosity distance

        lnL[i] = FactoredLogLikelihood(theEpochFiducial,P, rholms_intp, crossTerms, Lmax)
        if (numpy.mod(i,200)==10 and i>100):
            print "\t Params ", i, " (RA, DEC, tref, phiref, incl, psi, dist) ="
            print "\t", i, P.phi, P.theta, float(P.tref-theEpochFiducial), P.phiref, P.incl, P.psi, P.dist/(1e6*lal.LAL_PC_SI), lnL[i]
            logLmarg =np.log(np.mean(np.exp(lnL[:i])))
#            print "\tlog likelihood is ",  lnL[i], ";   log-integral L_{marg} =", logLmarg , " with sqrt(2Lmarg)= ", np.sqrt(2*logLmarg), "; and  <lnL>=  ", np.mean(lnL[:i])
            print i,  lnL[i],   np.sqrt(np.max(lnL[:i])),  np.sqrt(2*logLmarg), np.mean(lnL[:i]) 
        i+=1
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
inc_min, inc_max = 0, numpy.pi
# orbital phi
phi_min, phi_max = 0, 2*numpy.pi
# distance
dist_min, dist_max = Dmin, Dmax

import functools
# Uniform sampling (in area) but nonuniform sampling in distance (*I hope*).  Auto-cdf inverse
# PROBLEM: Underlying prior samplers are not uniform.  We need two stages
sampler.add_parameter("psi", functools.partial(mcsampler.uniform_samp_vector, psi_min, psi_max), None, psi_min, psi_max)
sampler.add_parameter("ra", functools.partial(mcsampler.uniform_samp_vector, ra_min, ra_max), None, ra_min, ra_max)
sampler.add_parameter("dec", functools.partial(mcsampler.dec_samp_vector), None, dec_min, dec_max)
sampler.add_parameter("tref", functools.partial(mcsampler.uniform_samp_vector, tref_min, tref_max), None, tref_min, tref_max)
sampler.add_parameter("phi", functools.partial(mcsampler.uniform_samp_vector, phi_min, phi_max), None, phi_min, phi_max)
sampler.add_parameter("inc", functools.partial(mcsampler.cos_samp_vector), None, inc_min, inc_max)
if rosUseTargetedDistance:
    r0 = 25*1e6*lal.LAL_PC_SI
    sampler.add_parameter("dist", functools.partial(mcsampler.pseudo_dist_samp_vector,r0 ), None, dist_min, dist_max)
else:
    sampler.add_parameter("dist", functools.partial(mcsampler.uniform_samp_vector, dist_min, dist_max), None, dist_min, dist_max)


if rosShowSamplerInputDistributions:
    print " ====== Plotting prior and sampling distributions ==== "
    print "  PROBLEM: Build in/hardcoded via uniform limits on each parameter! Need to add measure factors "
    nFig = 0
    for param in sampler.params:
        nFig+=1
        plt.figure(nFig)
        xLow = sampler.llim[param]
        xHigh = sampler.rlim[param]
        xvals = np.linspace(xLow,xHigh,500)
        pdfPrior = lambda x: mcsampler.uniform_samp( float(xLow), float(xHigh), float(x))  # Force type conversion in case we have non-float limits for some reasona
        pdfvalsPrior = np.array(map(pdfPrior, xvals))  # do all the numpy operations by hand: no vectorization
        pdf = sampler.pdf[param]
        pdfvals = pdf(xvals)
        if param is  "dist":
            xvvals = xvals/(1e6*lal.LAL_PC_SI)       # plot in Mpc, not m.  Note PDF has to change
            pdfvalsPrior = pdfvalsPrior * (1e6*lal.LAL_PC_SI) # rescale units
            pdfvals = pdfvals * (1e6*lal.LAL_PC_SI) # rescale units
        plt.plot(xvals,pdfvalsPrior,label="prior:"+str(param),linestyle='--')
        plt.plot(xvals,pdfvals,label=str(param))
        plt.xlabel(str(param))
        plt.legend()
    plt.show()

    # print " ====== Plotting sampling distributions ==== "
    # nFig = 0
    # for param in sampler.params:
    #     nFig+=1
    #     plt.figure(nFig)
    #     pdf = sampler.pdf[param]
    #     xLow = sampler.llim[param]
    #     xHigh = sampler.rlim[param]
    #     xvals = np.linspace(xLow,xHigh,500)
    #     pdfvals = np.array(map(pdf, xvals))
    #     if param is  'dist':
    #         xvvals = xvals/(1e6*lal.LAL_PC_SI)       # plot in Mpc, not m.  Note PDF has to change
    #         pdfvals = pdfvals * (1e6*lal.LAL_PC_SI) # rescale units
    #     plt.plot(xvals,pdfvals,label=str(param))
    #     plt.xlabel(str(param))
    # plt.legend()
    # plt.show()

res, var = sampler.integrate(likelihood_function, 1e6, "ra", "dec", "tref", "phi", "inc", "psi", "dist")
print res, numpy.sqrt(var)
