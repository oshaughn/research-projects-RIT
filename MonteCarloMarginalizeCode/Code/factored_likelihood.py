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

try:
	import NRWaveformCatalogManager as nrwf
	useNR =True
except:
	useNR=False

distMpcRef = 1000 # a fiducial distance for the template source.
tWindowExplore = [-0.05, 0.05] # Not used in main code.  Provided for backward compatibility for ROS. Should be consistent with t_ref_wind in ILE.
rosDebugMessages = True

#
# Main driver functions
#
def PrecomputeLikelihoodTerms(event_time_geo, t_window, P, data_dict,
        psd_dict, Lmax, fMax, analyticPSD_Q=False,
        inv_spec_trunc_Q=False, T_spec=0., verbose=True,
         NR_group=None,NR_param=None):
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
    assert data_dict.keys() == psd_dict.keys()
    global distMpcRef
    detectors = data_dict.keys()
    rholms = {}
    rholms_intp = {}
    crossTerms = {}

    # Compute hlms at a reference distance, distance scaling is applied later
    P.dist = distMpcRef*1e6*lsu.lsu_PC

    print "  ++++ Template data being computed for the following binary +++ "
    P.print_params()
    # Compute all hlm modes with l <= Lmax
    detectors = data_dict.keys()
    # Zero-pad to same length as data - NB: Assuming all FD data same resolution
    P.deltaF = data_dict[detectors[0]].deltaF
    if not (NR_group) or not (NR_param):
        hlms_list = lsu.hlmoff(P, Lmax) # a linked list of hlms
        hlms = lsu.SphHarmFrequencySeries_to_dict(hlms_list, Lmax) # a dictionary

    else: # NR signal required
        mtot = P.m1 + P.m2
        # Load the catalog
        wfP = nrwf.WaveformModeCatalog(NR_group, NR_param, \
                                           clean_initial_transient=True,clean_final_decay=True, shift_by_extraction_radius=True, 
                                       lmax=Lmax,align_at_peak_l2_m2_emission=True)
        # Overwrite the parameters in wfP to set the desired scale
        q = wfP.P.m2/wfP.P.m1
        wfP.P.m1 *= mtot/(1+q)
        wfP.P.m2 *= mtot*q/(1+q)
        wfP.P.dist =100*1e6*lal.PC_SI  # fiducial distance.

        hlms = wfP.hlmoff( deltaT=P.deltaT,force_T=1./P.deltaF)  # force a window

    # Print statistics on timeseries provided
    print " Mode  npts(data)   npts epoch  epoch/deltaT "
    for mode in hlms.keys():
        print mode, data_dict[detectors[0]].data.length, hlms[mode].data.length, hlms[mode].data.length*P.deltaT, hlms[mode].epoch, hlms[mode].epoch/P.deltaT

    for det in detectors:
        # This is the event time at the detector
        t_det = ComputeArrivalTimeAtDetector(det, P.phi, P.theta,event_time_geo)
        # The is the difference between the time of the leading edge of the
        # time window we wish to compute the likelihood in, and
        # the time corresponding to the first sample in the rholms
        rho_epoch = data_dict[det].epoch - hlms[hlms.keys()[0]].epoch
        t_shift =  float(float(t_det) - float(t_window) - float(rho_epoch))
#        assert t_shift > 0    # because NR waveforms may start at any time, they don't always have t_shift > 0 ! 
        # tThe leading edge of our time window of interest occurs
        # this many samples into the rholms
        N_shift = int( t_shift / P.deltaT )
        # Number of samples in the window [t_ref - t_window, t_ref + t_window]
        N_window = int( 2 * t_window / P.deltaT )
        # Compute cross terms < h_lm | h_l'm' >
        crossTerms[det] = ComputeModeCrossTermIP(hlms, psd_dict[det], P.fmin,
                fMax, 1./2./P.deltaT, P.deltaF, analyticPSD_Q,
                inv_spec_trunc_Q, T_spec)
        # Compute rholm(t) = < h_lm(t) | d >
        rholms[det] = ComputeModeIPTimeSeries(hlms, data_dict[det],
                psd_dict[det], P.fmin, fMax, 1./2./P.deltaT, N_shift, N_window,
                analyticPSD_Q, inv_spec_trunc_Q, T_spec)
        rhoXX = rholms[det][rholms[det].keys()[0]]
        # The vector of time steps within our window of interest
        # for which we have discrete values of the rholms
        # N.B. I don't do simply rho_epoch + t_shift, b/c t_shift is the
        # precise desired time, while we round and shift an integer number of
        # steps of size deltaT
        t = np.arange(N_window) * P.deltaT\
                + float(rho_epoch + N_shift * P.deltaT )
        if verbose:
            print "For detector", det, "..."
            print "\tData starts at %.20g" % float(data_dict[det].epoch)
            print "\trholm starts at %.20g" % float(rho_epoch)
            print "\tEvent time at detector is: %.18g" % float(t_det)
            print "\tInterpolation window has half width %g" % t_window
            print "\tComputed t_shift = %.20g" % t_shift
            print "\t(t_shift should be t_det - t_window - t_rholm = %.20g)" %\
                    (t_det - t_window - float(rho_epoch))
            print "\tInterpolation starts at time %.20g" % t[0]
            print "\t(Should start at t_event - t_window = %.20g)" %\
                    (float(rho_epoch + N_shift * P.deltaT))
        # The minus N_shift indicates we need to roll left
        # to bring the desired samples to the front of the array
        rholms_intp[det] =  InterpolateRholms(rholms[det], t)

    return rholms_intp, crossTerms, rholms

def FactoredLogLikelihood(extr_params, rholms_intp, crossTerms, Lmax):
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

    # N.B.: The Ylms are a function of - phiref b/c we are passively rotating
    # the source frame, rather than actively rotating the binary.
    # Said another way, the m^th harmonic of the waveform should transform as
    # e^{- i m phiref}, but the Ylms go as e^{+ i m phiref}, so we must give
    # - phiref as an argument so Y_lm h_lm has the proper phiref dependence
    Ylms = ComputeYlms(Lmax, incl, -phiref)

    lnL = 0.
    for det in detectors:
        CT = crossTerms[det]
        F = ComplexAntennaFactor(det, RA, DEC, psi, tref)

        # This is the GPS time at the detector
        t_det = ComputeArrivalTimeAtDetector(det, RA, DEC, tref)
        det_rholms = {}  # rholms evaluated at time at detector
        for key in rholms_intp[det]:
            func = rholms_intp[det][key]
            det_rholms[key] = func(float(t_det))

        lnL += SingleDetectorLogLikelihood(det_rholms, CT, Ylms, F, dist)

    return lnL

def FactoredLogLikelihoodTimeMarginalized(tvals, extr_params, rholms_intp, rholms, crossTerms, Lmax, interpolate=False):
    """
    Compute the log-likelihood = -1/2 < d - h | d - h > from:
        - extr_params is an object containing values of all extrinsic parameters
        - rholms_intp is a dictionary of interpolating functions < h_lm(t) | d >
        - crossTerms is a dictionary of < h_lm | h_l'm' >
        - Lmax is the largest l-index of any h_lm mode considered

    tvals is an array of timeshifts relative to the detector,
    used to compute the marginalized integral.
    It provides both the time prior and the sample points used for the integral.

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

    # N.B.: The Ylms are a function of - phiref b/c we are passively rotating
    # the source frame, rather than actively rotating the binary.
    # Said another way, the m^th harmonic of the waveform should transform as
    # e^{- i m phiref}, but the Ylms go as e^{+ i m phiref}, so we must give
    # - phiref as an argument so Y_lm h_lm has the proper phiref dependence
    Ylms = ComputeYlms(Lmax, incl, -phiref)

    lnL = 0.
    for det in detectors:
        CT = crossTerms[det]
        F = ComplexAntennaFactor(det, RA, DEC, psi, tref)

        # This is the GPS time at the detector
        t_det = ComputeArrivalTimeAtDetector(det, RA, DEC, tref)
        det_rholms = {}  # rholms evaluated at time at detector
        if ( interpolate ):
            # use the interpolating functions. 
            for key, func in rholms_intp[det].iteritems():
                det_rholms[key] = func(float(t_det)+tvals)
        else:
            # do not interpolate, just use nearest neighbors.
            for key, rhoTS in rholms[det].iteritems():
                tfirst = float(t_det)+tvals[0]
                ifirst = int(np.round(( float(tfirst) - float(rhoTS.epoch)) / rhoTS.deltaT) + 0.5)
                ilast = ifirst + len(tvals)
                det_rholms[key] = rhoTS.data.data[ifirst:ilast]

        lnL += SingleDetectorLogLikelihood(det_rholms, CT, Ylms, F, dist)

    maxlnL = np.max(lnL)
    return maxlnL + np.log(integrate.simps(np.exp(lnL - maxlnL), dx=tvals[1]-tvals[0]))


#
# Internal functions
#
def SingleDetectorLogLikelihoodModel( crossTermsDictionary,tref, RA,DEC, thS,phiS,psi,  dist, Lmax, det):
    """
    DOCUMENT ME!!!
    """
    global distMpcRef

    crossTerms = crossTermsDictionary[det]
    # N.B.: The Ylms are a function of - phiref b/c we are passively rotating
    # the source frame, rather than actively rotating the binary.
    # Said another way, the m^th harmonic of the waveform should transform as
    # e^{- i m phiref}, but the Ylms go as e^{+ i m phiref}, so we must give
    # - phiref as an argument so Y_lm h_lm has the proper phiref dependence
    Ylms = ComputeYlms(Lmax, thS, -phiS)
    if (det == "Fake"):
        F=1
    else:
        F = ComplexAntennaFactor(det, RA,DEC,psi,tref)
    distMpc = dist/(lsu.lsu_PC*1e6)

#    keys = Ylms.keys()
    keys = crossTermsDictionary.keys()[:0]

    # Eq. 26 of Richard's notes
    # APPROXIMATING V BY U (appropriately swapped).  THIS APPROXIMATION MUST BE FIXED FOR PRECSSING SOURCES
    term2 = 0.
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

    # N.B.: The Ylms are a function of - phiref b/c we are passively rotating
    # the source frame, rather than actively rotating the binary.
    # Said another way, the m^th harmonic of the waveform should transform as
    # e^{- i m phiref}, but the Ylms go as e^{+ i m phiref}, so we must give
    # - phiref as an argument so Y_lm h_lm has the proper phiref dependence
    Ylms = ComputeYlms(Lmax, thS, -phiS)
    if (det == "Fake"):
        F=1
        tshift= tref - epoch
    else:
        F = ComplexAntennaFactor(det, RA,DEC,psi,tref)
        detector = lalsim.DetectorPrefixToLALDetector(det)
        tshift = ComputeArrivalTimeAtDetector(det, RA,DEC, tref)
    rholms_intp = rholmsDictionary[det]
    distMpc = dist/(lsu.lsu_PC*1e6)

    term1 = 0.
    for key in rholms_intp.keys():
        l = key[0]
        m = key[1]
        term1 += np.conj(F * Ylms[(l,m)]) * rholms_intp[(l,m)]( float(tshift))
    term1 = np.real(term1) / (distMpc/distMpcRef)

    return term1

# Prototyping speed of time marginalization.  Not yet confirmed
def NetworkLogLikelihoodTimeMarginalized(epoch,rholmsDictionary,crossTerms, tref, RA,DEC, thS,phiS,psi,  dist, Lmax, detList):
    """
    DOCUMENT ME!!!
    """
    global distMpcRef

    # N.B.: The Ylms are a function of - phiref b/c we are passively rotating
    # the source frame, rather than actively rotating the binary.
    # Said another way, the m^th harmonic of the waveform should transform as
    # e^{- i m phiref}, but the Ylms go as e^{+ i m phiref}, so we must give
    # - phiref as an argument so Y_lm h_lm has the proper phiref dependence
    Ylms = ComputeYlms(Lmax, thS, -phiS)
    distMpc = dist/(lsu.lsu_PC*1e6)

    F = {}
    tshift= {}
    for det in detList:
        F[det] = ComplexAntennaFactor(det, RA,DEC,psi,tref)
        detector = lalsim.DetectorPrefixToLALDetector(det)
        tshift[det] = float(ComputeArrivalTimeAtDetector(det, RA,DEC, tref))   # detector time minus reference time (so far)

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

    # N.B.: The Ylms are a function of - phiref b/c we are passively rotating
    # the source frame, rather than actively rotating the binary.
    # Said another way, the m^th harmonic of the waveform should transform as
    # e^{- i m phiref}, but the Ylms go as e^{+ i m phiref}, so we must give
    # - phiref as an argument so Y_lm h_lm has the proper phiref dependence
    Ylms = ComputeYlms(Lmax, thS, -phiS)
    distMpc = dist/(lsu.lsu_PC*1e6)

    F = {}
    tshift= {}
    for det in detList:
        F[det] = ComplexAntennaFactor(det, RA,DEC,psi,tref)
        detector = lalsim.DetectorPrefixToLALDetector(det)
        tshift[det] = float(ComputeArrivalTimeAtDetector(det, RA,DEC, tref))   # detector time minus reference time (so far)

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
        return term2a+ np.log(special.iv(0,np.abs(term1)))  # an approximation, ignoring term2b entirely! 
    else:
        # marginalize over phase.  Ideally done analytically. Only works if the terms are not too large -- otherwise overflow can occur. 
        # Should probably implement a special solution if overflow occurs
        def fnIntegrand(x):
            return np.exp( term2a+ np.real(term2b*np.exp(-4.j*x)+ term1*np.exp(+2.j*x)))/np.pi  # remember how the two terms enter -- note signs!
        LmargPsi = integrate.quad(fnIntegrand,0,np.pi,limit=100,epsrel=1e-4)[0]
        return np.log(LmargPsi)

def SingleDetectorLogLikelihood(rholm_vals, crossTerms, Ylms, F, dist):
    """
    Compute the value of the log-likelihood at a single detector from
    several intermediate pieces of data.

    Inputs:
      - rholm_vals: A dictionary of values of inner product between data
            and h_lm modes, < h_lm(t*) | d >, at a single time of interest t*
      - crossTerms: A dictionary of inner products between h_lm modes:
            < h_lm | h_l'm' >
      - Ylms: Dictionary of values of -2-spin-weighted spherical harmonic modes
            for a certain inclination and ref. phase, Y_lm(incl, - phiref)
      - F: Complex-valued antenna pattern depending on sky location and
            polarization angle, F = F_+ + i F_x
      - dist: The distance from the source to detector in meters

    Outputs: The value of ln L for a single detector given the inputs.
    """
    global distMpcRef
    distMpc = dist/(lsu.lsu_PC*1e6)

    # Eq. 35 of Richard's notes
    term1 = 0.
    for mode in rholm_vals:
        term1 += np.conj(F * Ylms[mode]) * rholm_vals[mode]
    term1 = np.real(term1) / (distMpc/distMpcRef)

    # Eq. 26 of Richard's notes
    term2 = 0.
    for pair1 in rholm_vals:
        for pair2 in rholm_vals:
            term2 += F * np.conj(F) * ( crossTerms[(pair1,pair2)])\
                    * np.conj(Ylms[pair1]) * Ylms[pair2]\
                    + F*F*Ylms[pair1]*Ylms[pair2]*((-1)**pair1[0])\
                    * crossTerms[((pair1[0],-pair1[1]),pair2)]
    term2 = -np.real(term2) / 4. /(distMpc/distMpcRef)**2

    return term1 + term2

def ComputeModeIPTimeSeries(hlms, data, psd, fmin, fMax, fNyq,
        N_shift, N_window, analyticPSD_Q=False,
        inv_spec_trunc_Q=False, T_spec=0.):
    """
    Compute the complex-valued overlap between
    each member of a SphHarmFrequencySeries 'hlms'
    and the interferometer data COMPLEX16FrequencySeries 'data',
    weighted the power spectral density REAL8FrequencySeries 'psd'.

    The integrand is non-zero in the range: [-fNyq, -fmin] \union [fmin, fNyq].
    This integrand is then inverse-FFT'd to get the inner product
    at a discrete series of time shifts.

    Returns a SphHarmTimeSeries object containing the complex inner product
    for discrete values of the reference time tref.  The epoch of the
    SphHarmTimeSeries object is set to account for the transformation
    """
    rholms = {}
    assert data.deltaF == hlms[hlms.keys()[0]].deltaF
    assert data.data.length == hlms[hlms.keys()[0]].data.length
    deltaT = data.data.length/(2*fNyq)

    # Create an instance of class to compute inner product time series
    IP = lsu.ComplexOverlap(fmin, fMax, fNyq, data.deltaF, psd,
            analyticPSD_Q, inv_spec_trunc_Q, T_spec, full_output=True)

    # Loop over modes and compute the overlap time series
    for pair in hlms.keys():
        rho, rhoTS, rhoIdx, rhoPhase = IP.ip(hlms[pair], data)
        rhoTS.epoch = data.epoch - hlms[pair].epoch
        rholms[pair] = lal.CutCOMPLEX16TimeSeries(rhoTS, N_shift, N_window)  # Warning: code currently fails w/o this cut.

    return rholms

def InterpolateRholm(rholm, t):
    h_re = np.real(rholm.data.data)
    h_im = np.imag(rholm.data.data)
    if rosDebugMessages:
        print "Interpolation length check ", len(t), len(h_re)
    # spline interpolate the real and imaginary parts of the time series
    h_real = interpolate.InterpolatedUnivariateSpline(t, h_re[:len(t)], k=3)
    h_imag = interpolate.InterpolatedUnivariateSpline(t, h_im[:len(t)], k=3)
    return lambda ti: h_real(ti) + 1j*h_imag(ti)

    # Little faster
    #def anon_intp(ti):
        #idx = np.searchsorted(t, ti)
        #return rholm.data.data[idx]
    #return anon_intp

    #from pygsl import spline
    #spl_re = spline.cspline(len(t))
    #spl_im = spline.cspline(len(t))
    #spl_re.init(t, np.real(rholm.data.data))
    #spl_im.init(t, np.imag(rholm.data.data))
    #@profile
    #def anon_intp(ti):
        #re = spl_re.eval_e_vector(ti)
        #return re + 1j*im
    #return anon_intp

    # Doesn't work, hits recursion depth
    #from scipy.signal import cspline1d, cspline1d_eval
    #re_coef = cspline1d(np.real(rholm.data.data))
    #im_coef = cspline1d(np.imag(rholm.data.data))
    #dx, x0 = rholm.deltaT, float(rholm.epoch)
    #return lambda ti: cspline1d_eval(re_coef, ti) + 1j*cspline1d_eval(im_coef, ti)


def InterpolateRholms(rholms, t):
    """
    Return a dictionary keyed on mode index tuples, (l,m)
    where each value is an interpolating function of the overlap against data
    as a function of time shift:
    rholm_intp(t) = < h_lm(t) | d >

    'rholms' is a dictionary keyed on (l,m) containing discrete time series of
    < h_lm(t_i) | d >
    't' is an array of the discrete times:
    [t_0, t_1, ..., t_N]
    """
    rholm_intp = {}
    for mode in rholms.keys():
        rholm = rholms[mode]
        # The mode is identically zero, don't bother with it
        if sum(abs(rholm.data.data)) == 0.0:
            continue
        rholm_intp[ mode ] = InterpolateRholm(rholm, t)

    return rholm_intp

def ComputeModeCrossTermIP(hlms, psd, fmin, fMax, fNyq, deltaF,
        analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0., verbose=True):
    """
    Compute the 'cross terms' between waveform modes, i.e.
    < h_lm | h_l'm' >.
    The inner product is weighted by power spectral density 'psd' and
    integrated over the interval [-fNyq, -fmin] \union [fmin, fNyq]

    Returns a dictionary of inner product values keyed by tuples of mode indices
    i.e. ((l,m),(l',m'))
    """
    # Create an instance of class to compute inner product
    IP = lsu.ComplexIP(fmin, fMax, fNyq, deltaF, psd, analyticPSD_Q,
            inv_spec_trunc_Q, T_spec)

    crossTerms = {}

    for mode1 in hlms.keys():
        for mode2 in hlms.keys():
            crossTerms[ (mode1,mode2) ] = IP.ip(hlms[mode1], hlms[mode2])
            if verbose:
                print "       : U populated ", (mode1, mode2), "  = ",\
                        crossTerms[(mode1,mode2) ]

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
    # if tref is a float or a GPSTime object,
    # it shoud be automagically converted in the appropriate way
    return tref + lal.TimeDelayFromEarthCenter(detector.location, RA, DEC, tref)

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
            lalsim.InstrumentNameToLALDetector(str(P.detector)))  # Propagates signal to the detector, including beampattern and time delay
    if rosDebugMessages:
        print " +++ Injection creation for detector ", P.detector, " ++ "
        print  "   : Creating signal for injection with epoch ", float(hp.epoch), " and event time centered at ", lsu.stringGPSNice(P.tref)
        Fp, Fc = lal.ComputeDetAMResponse(lalsim.InstrumentNameToLALDetector(str(P.detector)).response, P.phi, P.theta, P.psi, lal.GreenwichMeanSiderealTime(hp.epoch))
        print "  : creating signal for injection with (det, t,RA, DEC,psi,Fp,Fx)= ", P.detector, float(P.tref), P.phi, P.theta, P.psi, Fp, Fc
    if P.taper != lsu.lsu_TAPER_NONE: # Taper if requested
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
        rho22 = rholms[det][( 2, 2)]
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

    # N.B.: The Ylms are a function of - phiref b/c we are passively rotating
    # the source frame, rather than actively rotating the binary.
    # Said another way, the m^th harmonic of the waveform should transform as
    # e^{- i m phiref}, but the Ylms go as e^{+ i m phiref}, so we must give
    # - phiref as an argument so Y_lm h_lm has the proper phiref dependence
    Ylms = ComputeYlms(Lmax, thS, -phiS)
    distMpc = dist/(lsu.lsu_PC*1e6)

    F = {}
    tshift= {}
    for det in detList:
        if (det == "Fake"):
            F[det]=1
            tshift= tref - epoch
        else:
            F[det] = ComplexAntennaFactor(det, RA,DEC,psi,tref)

    term2 = 0.

    keys = constructLMIterator(Lmax)
    for det in detList:
        for pair1 in keys:
            for pair2 in keys:
                term2 += F[det] * np.conj(F[det]) * ( crossTerms[det][(pair1,pair2)])* np.conj(Ylms[pair1]) * Ylms[pair2] + F[det]*F[det]*Ylms[pair1]*Ylms[pair2]*((-1)**pair1[0])*crossTerms[det][((pair1[0],-pair1[1]),pair2)]
    term2 = -np.real(term2) / 4. /(distMpc/distMpcRef)**2


    print detList
    rho22 = rholmsDictionary[detList[0]][( 2,2)]

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

    # N.B.: The Ylms are a function of - phiref b/c we are passively rotating
    # the source frame, rather than actively rotating the binary.
    # Said another way, the m^th harmonic of the waveform should transform as
    # e^{- i m phiref}, but the Ylms go as e^{+ i m phiref}, so we must give
    # - phiref as an argument so Y_lm h_lm has the proper phiref dependence
    Ylms = ComputeYlms(Lmax, thS, -phiS)
    if (det == "Fake"):
        F=1
        tshift= tStart - epoch
    else:
        F = ComplexAntennaFactor(det, RA,DEC,psi,tStart)
        detector = lalsim.DetectorPrefixToLALDetector(det)
        tshift = ComputeArrivalTimeAtDetector(det, RA,DEC, tStart)  -  epoch   # detector time minus reference time (so far)
    rholms_grid = rholmsDictionary[det]
    distMpc = dist/(lsu.lsu_PC*1e6)

    rho22 = rholms_grid[( 2,2)]
    nShiftL = int(  float(tshift)/rho22.deltaT)

    term1 = 0.
    # Only loop over terms available in the keys
    for key in rholms_grid.keys():
#    for l in range(2,Lmax+1):
#        for m in range(-l,l+1):
            l = int(key[0])
            m = int(key[1])
            rhoTSnow  = rholms_grid[( l,m)]
            term1 += np.conj(F * Ylms[(l,m)]) * np.roll(rhoTSnow.data.data,nShiftL)
    term1 = np.real(term1) / (distMpc/distMpcRef)

    nBinLow = int(( tStart + tshift  - rho22.epoch )/rho22.deltaT)   # time interval is specified in GEOCENTER, but rho is at each IFO

    if (nBinLow>-1):
        return term1[nBinLow:nBinLow+nBins]
    else:
        tmp = np.roll(term1,nBinLow) # Good enough
        return tmp[0:nBins] # Good enough

def constructLMIterator(Lmax):  # returns a list of (l,m) pairs covering all modes, as a list.  Useful for building iterators without nested lists
    mylist = []
    for L in np.arange(2, Lmax+1):
        for m in np.arange(-L, L+1):
            mylist.append((L,m))
    return mylist
