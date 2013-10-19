# Copyright (C) 2013  Evan Ochsner, R. O'Shaughnessy
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Code to compute the log likelihood of parameters of a gravitational
waveform. Precomputes terms that depend only on intrinsic parameters
and computes the log likelihood for given values of extrinsic parameters

Requires python SWIG bindings of the LIGO Algorithms Library (LAL)
"""

import lal
import lalsimulation as lalsim
import lalsimutils as lsu
import numpy as np
from scipy import interpolate, integrate
from scipy import special
from itertools import product

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>, R. O'Shaughnessy"


distMpcRef = 100
tWindowReference = [-0.15,0.15]            # choose samples so we have this centered on the window
tWindowExplore = [-0.05, 0.05]             # smaller window.  Avoid interpolation errors on the edge.
rosDebugMessages = True
rosDebugUseCForQTimeseries =False
rosInterpolateOnlyTimeWindow = True       # Ability to only interpolate the target time window.
#rosInterpolateVia = 'AmplitudePhase'       #  Interpolate in amplitude and phase (NOT reliable!! DO NOT USE. Makes code slower and less reliable!)
rosInterpolateVia = "Other"
rosDoNotRollTimeseries = False           # must be done if you have zero padding.  Should always work, since epoch shifted too.
#rosDoNotUseMemoryMode = True       # strip the memory mode.  I am seeing some strange features.
rosAvoidNestedLoops = False               # itertools is actually pretty darned slow.  Let's not use it.
rosInterpolationMethod = "NearestNeighbor"
rosInterpolationMethod = "Interp1d"  # horribly slow!  Unusable prep time!
rosInterpolationMethod = "PrecomputeSpline"
rosInterpolationMethod = "ManualLinear"
rosInterpolationMethod = "InterpolatedUnivariateSpline"

#
# Main driver functions
#
def PrecomputeLikelihoodTerms(epoch,P, data_dict, psd_dict, Lmax,analyticPSD_Q=False):
    """
    Compute < h_lm(t) | d > and < h_lm | h_l'm' >

    Returns:
        - Dictionary of interpolating functions, keyed on detector, then (l,m)
          e.g. rholms_intp['H1'][(2,2)]
        - Dictionary of "cross terms" <h_lm | h_l'm' > keyed on (l,m),(l',m')
          e.g. crossTerms[((2,2),(2,1))]
        - Dictionary of discrete time series of < h_lm(t) | d >, keyed the same
          as the interpolating functions.
          Their main use is to validate the interpolating functions
    """
    global rosDebugMessages
    assert data_dict.keys() == psd_dict.keys()
    global distMpcRef
    detectors = data_dict.keys()
    rholms = {}
    rholms_intp = {}
    crossTerms = {}

    # Fix fiducial distance at which precomputations are performed: distance scaling applied later
    P.dist = distMpcRef*1e6*lal.LAL_PC_SI

    print "  ++++ Template data being computed for the following binary +++ "
    P.print_params()
    # Compute all hlm modes with l <= Lmax
    detectors = data_dict.keys()
    P.deltaF = data_dict[detectors[0]].deltaF   # FORCE DESIRED SIGNAL TIME
    hlms = lsu.hlmoff(P, Lmax)
    h22 = lalsim.SphHarmFrequencySeriesGetMode(hlms, 2, 2)
    h22Epoch = h22.epoch

    for det in detectors:
        # Compute time-shift-dependent mode SNRs < h_lm(t) | d >
        df = data_dict[det].deltaF
        fNyq = df*len(data_dict[det].data.data)/2
        if rosDebugMessages:
            print " : Computing for ", det, " df, fNyq  = ", df, fNyq
        # Compute cross terms < h_lm | h_l'm' >
        #print " :   ", det, " -  : Building cross term matrix "
        crossTerms[det] = ComputeModeCrossTermIP(hlms, psd_dict[det], P.fmin,1./2./P.deltaT, P.deltaF, analyticPSD_Q)
        rholms[det] = ComputeModeIPTimeSeries(epoch,hlms, data_dict[det],psd_dict[det], P.fmin, 1./2./P.deltaT, analyticPSD_Q)
        rho22 = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, 2)
        # FIXME: Need to handle geocenter-detector time shift properly
        # NOTE: This array is almost certainly wrapped in time via the inverse FFT and is NOT starting at the epoch
        tShift =  float( P.tref -  epoch)   # shift the data by the time difference between the target data
        nRollL = int((np.abs(tWindowReference[0]) - tShift )/rho22.deltaT)   # this is a DISCRETE timeshift.
        nRollL +=  int(float((rho22.epoch - epoch))/rho22.deltaT)   # roll by more points if the epochs between the rho and fiducial are very different.  Important b/c I only interpolate a narrow window...if I miss the right epoch, the lnL(t) function will look like quadratic garbage.
        if rosDoNotRollTimeseries:
            nRollL = 0
        tShiftDiscrete = nRollL*rho22.deltaT
        if rosDebugMessages:
            print "  : must shift by ", nRollL, " corresponding to a time shift = ", tShiftDiscrete, " to make sure the reference time ", float(P.tref), " has a long window", float(tShift)
        # Time correspondence for these events, relative to the *fiducial* GPSTime 'epoch'
        t = np.arange(rho22.data.length) * rho22.deltaT - tShiftDiscrete  # account for the timeseries, plus roll. This is always correct
        tShiftChangeTimeOrigin = float(rho22.epoch - epoch)   # rho22.epoch already includes signal length info: literally just origin change
        t = t + tShiftChangeTimeOrigin                           # account for zero of time being set to 'epoch'. 
        rholms_intp[det] =  InterpolateRholms(rholms[det], t,nRollL, Lmax)

        print " shifting time by ", -tShiftDiscrete, " to allow for wraparound and to center the desired time "
        epoch_interp = rho22.epoch - tShiftDiscrete  # Moving the zero of time back to account for rolling the timeseries

    return rholms_intp, crossTerms, rholms, epoch_interp

def FactoredLogLikelihood(epoch, extr_params, rholms_intp, crossTerms, Lmax):
    """
    Compute the log-likelihood = -1/2 < d - h | d - h > from:
        - extr_params is an object containing values of all extrinsic parameters
        - rholms_intp is a dictionary of interpolating functions < h_lm(t) | d >
        - crossTerms is a dictionary of < h_lm | h_l'm' >
        - Lmax is the largest l-index of any h_lm mode considered

    N.B. rholms_intp and crossTerms are the first two outputs of the function
    'PrecomputeLikelihoodTerms'
    """
    # Sanity checks
    assert rholms_intp.keys() == crossTerms.keys()
    detectors = rholms_intp.keys()

    RA = extr_params.phi
    DEC =  extr_params.theta
    tref = extr_params.tref # geocenter time
    phiref = extr_params.phiref
    incl = extr_params.incl
    psi = extr_params.psi
    dist = extr_params.dist

    Ylms = ComputeYlms(Lmax, incl, phiref)

    lnL = 0.
    for det in detectors:
        CT = crossTerms[det]
        F = ComplexAntennaFactor(det, RA, DEC, psi, tref)

        tshifted = ComputeArrivalTimeAtDetector(det, RA, DEC, tref)  -  epoch # detector time minus fiducial time zero used in rholms.
        shifted_rholms = {}  # preallocate space with the right keys
        for key in rholms_intp[det]: #  det_rholms_intp.keys():
            func = rholms_intp[det][key]
            shifted_rholms[key] = func(float(tshifted))

        lnL += SingleDetectorLogLikelihood(shifted_rholms, CT, Ylms, F, dist, Lmax)

    return lnL


#
# Internal functions
#
def SingleDetectorLogLikelihoodModel( crossTermsDictionary,tref, RA,DEC, thS,phiS,psi,  dist, Lmax, det):
    """
    DOCUMENT ME!!!
    """
    global distMpcRef

    crossTerms = crossTermsDictionary[det]
    Ylms = ComputeYlms(Lmax, thS,phiS)
    if (det == "Fake"):
        F=1
    else:
        F = ComplexAntennaFactor(det, RA,DEC,psi,tref)
    distMpc = dist/(lal.LAL_PC_SI*1e6)

    keys = Ylms.keys()

    # Eq. 26 of Richard's notes
    # APPROXIMATING V BY U (appropriately swapped).  THIS APPROXIMATION MUST BE FIXED FOR PRECSSING SOURCES
    term2 = 0.
    if rosAvoidNestedLoops:
        pairsOfPairs = product( keys,keys)  # loop over pairs might be faster than not
        for pPair in pairsOfPairs:
            pair1 = pPair[0]
            pair2 = pPair[1]
            term2 += F * np.conj(F) * ( crossTerms[(pair1,pair2)])* np.conj(Ylms[pair1]) * Ylms[pair2] + F*F*Ylms[pair1]*Ylms[pair2]*((-1)**pair1[0])*crossTerms[((pair1[0],-pair1[1]),pair2)]
    else:
        for pair1 in keys:
            for pair2 in keys:
                term2 += F * np.conj(F) * ( crossTerms[(pair1,pair2)])* np.conj(Ylms[pair1]) * Ylms[pair2] + F*F*Ylms[pair1]*Ylms[pair2]*((-1)**pair1[0])*crossTerms[((pair1[0],-pair1[1]),pair2)]
    term2 = -np.real(term2) / 4. /(distMpc/distMpcRef)**2
    return term2

def SingleDetectorLogLikelihoodData(epoch,rholmsDictionary,tref, RA,DEC, thS,phiS,psi,  dist, Lmax, det):
    """
    DOCUMENT ME!!!
    """
    global distMpcRef

    Ylms = ComputeYlms(Lmax, thS,phiS)
    if (det == "Fake"):
        F=1
        tshift= tref - epoch
    else:
        F = ComplexAntennaFactor(det, RA,DEC,psi,tref)
        detector = lalsim.DetectorPrefixToLALDetector(det)
        tshift = ComputeArrivalTimeAtDetector(det, RA,DEC, tref)  -  epoch   # detector time minus reference time (so far)
    rholms_intp = rholmsDictionary[det]
    distMpc = dist/(lal.LAL_PC_SI*1e6)

    term1 = 0.
    if rosAvoidNestedLoops:
        keys = lsu.constructLMIterator(Lmax)
        for pair in keys: #rholms_intp:
            #        print " adding term to lnLdata ", pair
            term1+= np.conj(F*Ylms[pair])*rholms_intp[pair]( float(tshift))
    else:
        for l in range(2,Lmax+1):
            for m in range(-l,l+1):
                term1 += np.conj(F * Ylms[(l,m)]) * rholms_intp[(l,m)]( float(tshift))
    term1 = np.real(term1) / (distMpc/distMpcRef)

    return term1

# Prototyping speed of time marginalization.  Not yet confirmed
def NetworkLogLikelihoodTimeMarginalized(epoch,rholmsDictionary,crossTerms, tref, RA,DEC, thS,phiS,psi,  dist, Lmax, detList):
    """
    DOCUMENT ME!!!
    """
    global distMpcRef

    Ylms = ComputeYlms(Lmax, thS,phiS)
    distMpc = dist/(lal.LAL_PC_SI*1e6)

    F = {}
    tshift= {}
    for det in detList:
        F[det] = ComplexAntennaFactor(det, RA,DEC,psi,tref)
        detector = lalsim.DetectorPrefixToLALDetector(det)
        tshift[det] = float(ComputeArrivalTimeAtDetector(det, RA,DEC, tref)  -  epoch)   # detector time minus reference time (so far)

    term2 = 0.
    for det in detList:
        for pair1 in rholmsDictionary[det]:
            for pair2 in rholmsDictionary[det]:
                term2 += F[det] * np.conj(F[det]) * ( crossTerms[det][(pair1,pair2)])* np.conj(Ylms[pair1]) * Ylms[pair2] + F[det]*F[det]*Ylms[pair1]*Ylms[pair2]*((-1)**pair1[0])*crossTerms[det][((pair1[0],-pair1[1]),pair2)]
    term2 = -np.real(term2) / 4. /(distMpc/distMpcRef)**2

    def fnIntegrand(dt):
        term1 = 0.
        for det in detList:
            for pair in rholmsDictionary[det]:
                term1+= np.conj(F[det]*Ylms[pair])*rholmsDictionary[det][pair]( float(tshift[det]) + dt)
        term1 = np.real(term1) / (distMpc/distMpcRef)
        return np.exp(np.max([term1+term2,-15.]))   # avoid hugely negative numbers.  This floor on the log likelihood here will not significantly alter any physical result.

    # empirically this procedure will find a gaussian with width less than  0.5 e (-3) times th window length.  This procedure *should* therefore work for a sub-s window
    LmargTime = integrate.quad(fnIntegrand, tWindowExplore[0], tWindowExplore[1],points=[0],limit=300)[0]  # the second return value is the error
#    LmargTime = integrate.quadrature(fnIntegrand, tWindowExplore[0], tWindowExplore[1],maxiter=400)  # very slow, not reliable
    return np.log(LmargTime)

# Prototyping speed of time marginalization.  Not yet confirmed
def NetworkLogLikelihoodPolarizationMarginalized(epoch,rholmsDictionary,crossTerms, tref, RA,DEC, thS,phiS,psi,  dist, Lmax, detList):
    """
    DOCUMENT ME!!!
    """
    global distMpcRef

    Ylms = ComputeYlms(Lmax, thS,phiS)
    distMpc = dist/(lal.LAL_PC_SI*1e6)

    F = {}
    tshift= {}
    for det in detList:
        F[det] = ComplexAntennaFactor(det, RA,DEC,psi,tref)
        detector = lalsim.DetectorPrefixToLALDetector(det)
        tshift[det] = float(ComputeArrivalTimeAtDetector(det, RA,DEC, tref)  -  epoch)   # detector time minus reference time (so far)

    term2a = 0.
    term2b = 0.
    for det in detList:
        for pair1 in rholmsDictionary[det]:
            for pair2 in rholmsDictionary[det]:
                term2a += F[det] * np.conj(F[det]) * ( crossTerms[det][(pair1,pair2)])* np.conj(Ylms[pair1]) * Ylms[pair2] 
                term2b += F[det]*F[det]*Ylms[pair1]*Ylms[pair2]*((-1)**pair1[0])*crossTerms[det][((pair1[0],-pair1[1]),pair2)]
    term2a = -np.real(term2a) / 4. /(distMpc/distMpcRef)**2
    term2b = -term2b/4./(distMpc/distMpcRef)**2   # coefficient of exp(-4ipsi)

    term1 = 0.
    for det in detList:
        for pair in rholmsDictionary[det]:
                term1+= np.conj(F[det]*Ylms[pair])*rholmsDictionary[det][pair]( float(tshift[det]) )
    term1 = term1 / (distMpc/distMpcRef)  # coefficient of exp(-2ipsi)

    # if the coefficients of the exponential are too large, do the integral by hand, in the gaussian limit? NOT IMPLEMENTED YET
    if False: #xgterm2a+np.abs(term2b)+np.abs(term1)>100:
#        psiCrit = - np.angle(term1/term2b)/2
#        return term2a+np.real(term2b*np.exp(-4.j*psiCrit) + np.real(term1*np.exp(-2.j*psiCrit)) 
#        return 0
        return term2a+ np.log(special.iv(0,np.abs(term1)))  # an approximation, ignoring term2b entirely! 
    else:
        # marginalize over phase.  Ideally done analytically. Only works if the terms are not too large -- otherwise overflow can occur. 
        # Should probably implement a special solution if overflow occurs
        def fnIntegrand(x):
            return np.exp( term2a+ np.real(term2b*np.exp(-4.j*x)+ term1*np.exp(+2.j*x)))/np.pi  # remember how the two terms enter -- note signs!
        LmargPsi = integrate.quad(fnIntegrand,0,np.pi,limit=100,epsrel=1e-4)[0]
        return np.log(LmargPsi)

def SingleDetectorLogLikelihood(rholm_vals, crossTerms, Ylms, F, dist, Lmax):
    """
    DOCUMENT ME!!!
    """
    global distMpcRef
    distMpc = dist/(lal.LAL_PC_SI*1e6)

    keys = lsu.constructLMIterator(Lmax)

    # Eq. 35 of Richard's notes
    term1 = 0.
    if rosAvoidNestedLoops:
        for key in keys: #rholm_vals:
            term1 += np.conj(F * Ylms[key]) * rholm_vals[key]
    else:
        for l in range(2,Lmax+1):
            for m in range(-l,l+1):
                term1 += np.conj(F * Ylms[(l,m)]) * rholm_vals[(l,m)]
    term1 = np.real(term1) / (distMpc/distMpcRef)

    # Eq. 26 of Richard's notes
    term2 = 0.
    if rosAvoidNestedLoops:
        pairsOfPairs = product(  keys,keys)  # loop over pairs might be faster than not
        for pPair in pairsOfPairs:
            pair1 = pPair[0]
            pair2 = pPair[1]
            term2 += F * np.conj(F) * ( crossTerms[(pair1,pair2)])* np.conj(Ylms[pair1]) * Ylms[pair2] + F*F*Ylms[pair1]*Ylms[pair2]*((-1)**pair1[0])*crossTerms[((pair1[0],-pair1[1]),pair2)]
    else:
        for pair1 in rholm_vals:
            for pair2 in rholm_vals:
                term2 += F * np.conj(F) * ( crossTerms[(pair1,pair2)])* np.conj(Ylms[pair1]) * Ylms[pair2] + F*F*Ylms[pair1]*Ylms[pair2]*((-1)**pair1[0])*crossTerms[((pair1[0],-pair1[1]),pair2)]
    # for l in range(2,Lmax+1):
    #     for m in range(-l,l+1):
    #         for lp in range(2,Lmax+1):
    #             for mp in range(-lp,lp+1):
    #                 term2 += F * np.conj(F) * ( crossTerms[((l,m),(lp,mp))])* np.conj(Ylms[(l,m)]) * Ylms[(lp,mp)] + F*F*Ylms[(l,m)]*Ylms[(lp,mp)]*((-1)**l)*crossTerms[((l,-m),(lp,mp))]
    term2 = -np.real(term2) / 4. /(distMpc/distMpcRef)**2

    return term1 + term2

def ComputeModeIPTimeSeries(epoch,hlms, data, psd, fmin, fNyq, analyticPSD_Q=False,
        tref=None, N=None):
    """
    Compute the complex-valued overlap between
    each member of a SphHarmFrequencySeries 'hlms' 
    and the interferometer data COMPLEX16FrequencySeries 'data',
    weighted the power spectral density REAL8FrequencySeries 'psd'.

    The integrand is non-zero in the range: [-fNyq, -fmin] \union [fmin, fNyq].
    This integrand is then inverse-FFT'd to get the inner product
    at a discrete series of time shifts.

    Returns a SphHarmTimeSeries object containing the complex inner product
    for discrete values of the reference time tref.  The epoch of the SphHarmTimeSeries object
    is set to account for the transformation

    Can optionally give arguments to return only the inner product reference
    times in the range: [tref - N * deltaT, tref + N * deltaT]
    where deltaT is the time step size stored in 'hlms'
    """
    # FIXME: For now not handling tref, N
    assert tref==None and N==None

    # Create an instance of class to compute inner product time series
    if analyticPSD_Q==False:
        assert data.deltaF == psd.deltaF
        IP = lsu.ComplexOverlap(fmin, fNyq, data.deltaF, psd.data.data, False, True)
        IPRegular = lsu.ComplexIP(fmin, fNyq, data.deltaF, psd.data.data, analyticPSD_Q=False)  # debugging, sanity checks
    else:
        IP = lsu.ComplexOverlap(fmin, fNyq, data.deltaF, psd, analyticPSD_Q=True, full_output=True)
        IPRegular = lsu.ComplexIP(fmin, fNyq, data.deltaF, psd)  # debugging, sanity checks

    print IP.fLow, IP.fNyq,IP.deltaF
    # Loop over modes and compute the overlap time series
    rholms = None
    h22 = lalsim.SphHarmFrequencySeriesGetMode(hlms,2,2)
    assert data.deltaF == h22.deltaF
    assert len(data.data.data) == len(h22.data.data)

    Lmax = lalsim.SphHarmFrequencySeriesGetMaxL(hlms)
    keys = lsu.constructLMIterator(Lmax)  # nested lists are very bad for python
    for pair in keys:
        l = int(pair[0])
        m = int(pair[1])
        hlm = lalsim.SphHarmFrequencySeriesGetMode(hlms, l, m)
        if hlm is None:
            # set a zero timeseries for that object
            deltaT = len(data.data.data)/(2*fNyq)
            rhoTS = lal.CreateCOMPLEX16TimeSeries("zero data",  lal.LIGOTimeGPS(0.), 0.,deltaT , lal.lalDimensionlessUnit,len(data.data.data))
        else:
            assert hlm.deltaF == data.deltaF
            rho, rhoTS, rhoIdx, rhoPhase = IP.ip(hlm, data)
        rhoTS.epoch = data.epoch -h22.epoch
        rholms = lalsim.SphHarmTimeSeriesAddMode(rholms, rhoTS, l, m)

    # RETURN: Do not window or readjust the timeseries here.  This is done in the interpolation step.
    # TIMING : Epoch set 
    return rholms

def InterpolateRholm(rholm, t,nRollL):
    global rosInterpolateVia
    nBinMax = 2*(tWindowReference[1]-tWindowReference[0])/rholm.deltaT
    hxdat = np.roll(np.imag(rholm.data.data),nRollL)[:nBinMax]
    hpdat = np.roll(np.real(rholm.data.data), nRollL)[:nBinMax]
    hx = interpolate.InterpolatedUnivariateSpline(t[:nBinMax], hxdat, k=3)
    hp = interpolate.InterpolatedUnivariateSpline(t[:nBinMax], hpdat, k=3)
    return lambda ti: hp(ti) + 1j*hx(ti)
        


def InterpolateRholms(rholms, t, nRollL, Lmax):
    """
    Return a dictionary keyed on mode index tuples, (l,m)
    where each value is an interpolating function of the overlap against data
    as a function of time shift:
    rholm_intp(t) = < h_lm(t) | d >

    'rholms' is a SphHarmTimeSeries containing discrete time series of
    < h_lm(t_i) | d >
    't' is an array of the discrete times:
    [t_0, t_1, ..., t_N]
    'Lmax' is the largest l index of the SphHarmTimeSeries
    """
    rholm_intp = {}
    for l in range(2, Lmax+1):
        for m in range(-l,l+1):
            rholm = lalsim.SphHarmTimeSeriesGetMode(rholms, l, m)
            rholm_intp[ (l,m) ] = InterpolateRholm(rholm,  t,nRollL)

    return rholm_intp

def ComputeModeCrossTermIP(hlms, psd, fmin, fNyq, deltaF, analyticPSD_Q=False):
    """
    Compute the 'cross terms' between waveform modes, i.e.
    < h_lm | h_l'm' >.
    The inner product is weighted by power spectral density 'psd' and
    integrated over the interval [-fNyq, -fmin] \union [fmin, fNyq]

    Returns a dictionary of inner product values keyed by tuples of mode indices
    i.e. ((l,m),(l',m'))
    """
    # Create an instance of class to compute inner product
    if analyticPSD_Q==False:
        assert deltaF == psd.deltaF
        IP = lsu.ComplexIP(fmin, fNyq, deltaF, psd.data.data, analyticPSD_Q=False)
    else:
        IP = lsu.ComplexIP(fmin, fNyq, deltaF, psd, analyticPSD_Q=True)

    # Loop over modes and compute the inner products, store in a dictionary
    Lmax = lalsim.SphHarmFrequencySeriesGetMaxL(hlms)

    crossTerms = {}

    if rosAvoidNestedLoops:
        keys = lsu.constructLMIterator(Lmax)
        pairsOfPairs = product(keys,keys)  # loop over pairs might be faster than not
        for pPair in pairsOfPairs:
            pair1 = pPair[0]
            pair2 = pPair[1]
            l = pair1[0]
            m = pair1[1]
            lp = pair2[0]
            mp = pair2[1]
            hlm = lalsim.SphHarmFrequencySeriesGetMode(hlms, int(l), int(m))
            hlpmp = lalsim.SphHarmFrequencySeriesGetMode(hlms, int(lp), int(mp))
            if hlm is None or hlpmp is None:
                crossTerms[ ((l,m),(lp,mp)) ] = 0
            else:
                crossTerms[ ((l,m),(lp,mp)) ] = IP.ip(hlm, hlpmp)  # need to be careful about left side to avoid type errors later when I do this loop
            if rosDebugMessages:
                print "       : U populated ", ((l,m), (lp,mp)), "  = ", crossTerms[(pair1,pair2) ]
    else:
        for l in range(2,Lmax+1):
            for m in range(-l,l+1):
                for lp in range(2,Lmax+1):
                    for mp in range(-lp,lp+1):
                        hlm = lalsim.SphHarmFrequencySeriesGetMode(hlms, l, m)
                        hlpmp = lalsim.SphHarmFrequencySeriesGetMode(hlms, lp, mp)
                        if hlm is None or hlpmp is None:
                            crossTerms[ ((l,m),(lp,mp)) ] = 0
                        else:
                            crossTerms[ ((l,m),(lp,mp)) ] = IP.ip(hlm, hlpmp)  # need to be careful about left side to avoid type errors later when I do this loop
                        if rosDebugMessages:
                            print "       : U populated ", ((l,m), (lp,mp)), "  = ", crossTerms[((l,m),(lp,mp)) ]

    return crossTerms

def ComplexAntennaFactor(det, RA, DEC, psi, tref):
    """
    Function to compute the complex-valued antenna pattern function:
    F+ + i Fx

    'det' is a detector prefix string (e.g. 'H1')
    'RA' and 'DEC' are right ascension and declination (in radians)
    'psi' is the polarization angle
    'tref' is the reference GPS time
    """
    detector = lalsim.DetectorPrefixToLALDetector(det)
    Fp, Fc = lal.ComputeDetAMResponse(detector.response, RA, DEC, psi, lal.GreenwichMeanSiderealTime(tref))

    return Fp + 1j * Fc

def ComputeYlms(Lmax, theta, phi):
    """
    Return a dictionary keyed by tuples
    (l,m)
    that contains the values of all
    -2Y_lm(theta,phi)
    with
    l <= Lmax
    -l <= m <= l
    """
    Ylms = {}
    if rosAvoidNestedLoops:
        keys = lsu.constructLMIterator(Lmax)
        for pair in keys:
            l = int(pair[0])
            m = int(pair[1])
            Ylms[ (l,m) ] = lal.SpinWeightedSphericalHarmonic(theta, phi,-2, l, m)
    else:
        for l in range(2,Lmax+1):
            for m in range(-l,l+1):
                Ylms[ (l,m) ] = lal.SpinWeightedSphericalHarmonic(theta, phi,-2, l, m)

    return Ylms

def ComputeArrivalTimeAtDetector(det, RA, DEC, tref): 
    """
    Function to compute the time of arrival at a detector
    from the time of arrival at the geocenter.

    'det' is a detector prefix string (e.g. 'H1')
    'RA' and 'DEC' are right ascension and declination (in radians)
    'tref' is the reference time at the geocenter.  It can be either a float (in which case the return is a float) or a GPSTime object (in which case it returns a GPSTime)
    """
    detector = lalsim.DetectorPrefixToLALDetector(det)
    return tref + lal.TimeDelayFromEarthCenter(detector.location, RA, DEC, tref)  # if tref is a float or a GPSTime object, it shoud be automagically converted in the appropriate way

# Create complex FD data that does not assume Hermitianity - i.e.
# contains positive and negative freq. content
# TIMING INFO: 
#    - epoch set so the merger event occurs at total time P.tref
def non_herm_hoff(P):
    hp, hc = lalsim.SimInspiralChooseTDWaveform(P.phiref, P.deltaT, P.m1, P.m2, 
            P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin, P.fref, P.dist, 
            P.incl, P.lambda1, P.lambda2, P.waveFlags, P.nonGRparams,
            P.ampO, P.phaseO, P.approx)
    hp.epoch = hp.epoch + P.tref
    hc.epoch = hc.epoch + P.tref
    hoft = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc,
             P.phi,  P.theta, P.psi,
            lalsim.InstrumentNameToLALDetector(P.detector))  # Propagates signal to the detector, including beampattern and time delay
    if rosDebugMessages:
        print " +++ Injection creation for detector ", P.detector, " ++ "
        print  "   : Creating signal for injection with epoch ", float(hp.epoch), " and event time centered at ", lsu.stringGPSNice(P.tref)
        Fp, Fc = lal.ComputeDetAMResponse(lalsim.InstrumentNameToLALDetector(P.detector).response, P.phi, P.theta, P.psi, lal.GreenwichMeanSiderealTime(hp.epoch))
        print "  : creating signal for injection with (det, t,RA, DEC,psi,Fp,Fx)= ", P.detector, float(P.tref), P.phi, P.theta, P.psi, Fp, Fc
    if P.taper != lalsim.LAL_SIM_INSPIRAL_TAPER_NONE: # Taper if requested
        lalsim.SimInspiralREAL8WaveTaper(hoft.data, P.taper)
    if P.deltaF == None:
        TDlen = nextPow2(hoft.data.length)
    else:
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen >= hoft.data.length

    fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)
    hoft = lal.ResizeREAL8TimeSeries(hoft, 0, TDlen)
    hoftC = lal.CreateCOMPLEX16TimeSeries("hoft", hoft.epoch, hoft.f0,
            hoft.deltaT, hoft.sampleUnits, TDlen)
    # copy h(t) into a COMPLEX16 array which happens to be purely real
    for i in range(TDlen):
        hoftC.data.data[i] = hoft.data.data[i]
    FDlen = TDlen
    hoff = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            hoft.epoch, hoft.f0, 1./hoft.deltaT/TDlen, lal.lalHertzUnit, 
            FDlen)
    lal.COMPLEX16TimeFreqFFT(hoff, hoftC, fwdplan)
    return hoff


def rollTimeSeries(series_dict, nRollRight):
    # Use the fact that we pass by value and that we swig bind numpy arrays
    for det in series_dict:
        print " Rolling timeseries ", nRollRight
        np.roll(series_dict.data.data, nRollRight)

def estimateUpperDistanceBoundInMpc(rholms,crossTerms):  
    # For nonprecessing sources, use the 22 mode to estimate the optimally oriented distance
    Qbar = 0
    nDet = 0
    for det in rholms:
        nDet+=1
        rho22 = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, 2)
        Qbar+=np.abs(crossTerms[det][(2,2), (2,2)])/np.max(np.abs(rho22.data.data))   # one value for each detector
    fudgeFactor = 1.1  # let's give ourselves a buffer -- we can't afford to be too tight
    return fudgeFactor*distMpcRef* Qbar/nDet *np.sqrt(5/(4.*np.pi))/2.

def estimateEventTimeRelative(theEpochFiducial,rholms, rholms_intp):
    
    return 0

def evaluateFast1dInterpolator(x,y,xlow,xhigh):
    indx = np.floor((x-xlow)/(xhigh-xlow) * len(y))
    if indx<0:
        indx = 0
    if indx > len(y):
        indx = len(y)-1
    return (y[indx]*(xhigh-x)  + y[indx+1]*(x-xlow))/(xhigh-xlow)
def makeFast1dInterpolator(y,xlow,xhigh):
    return lambda x: evaluateFast1dInterpolator(x,y, xlow, xhigh)

def NetworkLogLikelihoodTimeMarginalizedDiscrete(epoch,rholmsDictionary,crossTerms, tref, deltaTWindow, RA,DEC, thS,phiS,psi,  dist, Lmax, detList,array_output=False):
    """
    NetworkLogLikelihoodTimeMarginalizedDiscrete
    Uses DiscreteSingleDetectorLogLikelihoodData to calculate the lnL(t) on a discrete grid.
     - Mode 1: array_output = False
       Computes \int L dt/T over tref+deltaTWindow (GPSTime + pair, with |deltaTWindow|=T)
     - Mode 2: array_output = True
       Returns lnL(t) as a raw numpy array.
    """
    global distMpcRef

    Ylms = ComputeYlms(Lmax, thS,phiS)
    distMpc = dist/(lal.LAL_PC_SI*1e6)

    F = {}
    tshift= {}
    for det in detList:
        if (det == "Fake"):
            F[det]=1
            tshift= tref - epoch
        else:
            F[det] = ComplexAntennaFactor(det, RA,DEC,psi,tref)

    term2 = 0.

    keys = lsu.constructLMIterator(Lmax)
    for det in detList:
        for pair1 in keys:
            for pair2 in keys:
                term2 += F[det] * np.conj(F[det]) * ( crossTerms[det][(pair1,pair2)])* np.conj(Ylms[pair1]) * Ylms[pair2] + F[det]*F[det]*Ylms[pair1]*Ylms[pair2]*((-1)**pair1[0])*crossTerms[det][((pair1[0],-pair1[1]),pair2)]
    term2 = -np.real(term2) / 4. /(distMpc/distMpcRef)**2


    print detList
    rho22 = lalsim.SphHarmTimeSeriesGetMode(rholmsDictionary[detList[0]], 2,2)

    nBins =int( (deltaTWindow[1]-deltaTWindow[0])/rho22.deltaT)
    term1 =np.zeros(nBins)
    for det in detList:
        term1+=DiscreteSingleDetectorLogLikelihoodData(epoch, rholmsDictionary, tref+deltaTWindow[0], nBins,RA,DEC, thS,phiS,psi,  dist, Lmax, det)

    # Compute integral.  Note the NORMALIZATION interval is assumed to be tWindow.
    # This is equivalent to dividing by 1/N in *this case*.  That formula will not hold if the prior and integration region are different.  
    if array_output:
        return term1+term2  # output is lnL(t), NO marginalization
    else:
        LmargTime = rho22.deltaT*np.sum(np.exp(term1+term2))/(deltaTWindow[1]-deltaTWindow[0])  
        return np.log(LmargTime)

def DiscreteSingleDetectorLogLikelihoodData(epoch,rholmsDictionary, tStart,nBins, RA,DEC, thS,phiS,psi,  dist, Lmax, det):
    """
    DiscreteSingleDetectorLogLikelihoodData
    Returns lnLdata array, evaluated at the geocenter, based on a DISCRETE timeshift.
       - At low sampling rates, this procedure will be considerably time offset (~ ms). 
         It will also be undersampled for use in integration. 
    Return value is 
       - a RAW numpy array
       - associated with nBins following tStart
    Uses 'tStart' (a GPSTime) to identify the current detector orientations and hence time of flight delay.  
       - the assumption is that nBins will be very small
    Does NOT 
       - resample the  Q array : it is nearest-neighbor timeshifted before computing lnL
    """
    global distMpcRef

    Ylms = ComputeYlms(Lmax, thS,phiS)
    if (det == "Fake"):
        F=1
        tshift= tStart - epoch
    else:
        F = ComplexAntennaFactor(det, RA,DEC,psi,tStart)
        detector = lalsim.DetectorPrefixToLALDetector(det)
        tshift = ComputeArrivalTimeAtDetector(det, RA,DEC, tStart)  -  epoch   # detector time minus reference time (so far)
    rholms_grid = rholmsDictionary[det]
    distMpc = dist/(lal.LAL_PC_SI*1e6)

    rho22 = lalsim.SphHarmTimeSeriesGetMode(rholms_grid, 2,2)
    nShiftL = int(  float(tshift)/rho22.deltaT)

    term1 = 0.
    for l in range(2,Lmax+1):
        for m in range(-l,l+1):
            rhoTSnow  =     rho22 = lalsim.SphHarmTimeSeriesGetMode(rholms_grid, l,m)
            term1 += np.conj(F * Ylms[(l,m)]) * np.roll(rhoTSnow.data.data,nShiftL)
    term1 = np.real(term1) / (distMpc/distMpcRef)

    nBinLow = int(( tStart + tshift  - rho22.epoch )/rho22.deltaT)   # time interval is specified in GEOCENTER, but rho is at each IFO

    return term1[nBinLow:nBinLow+nBins]
