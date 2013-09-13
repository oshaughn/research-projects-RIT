# Copyright (C) 2013  Evan Ochsner
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

from lalsimutils import *

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>"


#
# Main driver functions
#
def PrecomputeLikelihoodTerms(P, data_dict, psd_dict, Lmax,analyticPSD_Q=False):
    """
    Compute < h_lm(t) | d > and < h_lm | h_l'm' >
    """
    assert data_dict.keys() == psd_dict.keys()
    detectors = data_dict.keys()
    rholms = {}
    rholms_intp = {}
    crossTerms = {}

    # Compute all hlm modes with l <= Lmax
    hlms = hlmoff(P, Lmax)

    for det in detectors:
        # Compute time-shift-dependent mode SNRs < h_lm(t) | d >
        rholms[det] = ComputeModeIPTimeSeries(hlms, data_dict[det],
                psd_dict[det], P.fmin, 1./2./P.deltaT, analyticPSD_Q)
        rho22 = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, 2)
        # FIXME: Need to handle geocenter-detector time shift properly
        t = np.arange(rho22.data.length) * rho22.deltaT
        rholms_intp[det] = InterpolateRholms(rholms[det], t, Lmax)
        # Compute cross terms < h_lm | h_l'm' >
        crossTerms[det] = ComputeModeCrossTermIP(hlms, psd_dict[det], P.fmin,
                1./2./P.deltaT, P.deltaF, analyticPSD_Q)

    return rholms_intp, crossTerms, rholms

def FactoredLogLikelihood(extr_params, rholms_intp, crossTerms, Lmax):
    """
    DOCUMENT ME!!!
    """
    # Sanity checks
    assert rholms_intp.keys() == crossTerms.keys()
    detectors = rholms_intp.keys()

    RA = extr_params.RA
    DEC = extr_params.DEC
    tref = extr_params.tref
    phiref = extr_params.phiref
    incl = extr.inclination
    psi = extr.polarization_angle
    dist = extr_params.distance

    Ylms = ComputeYlms(Lmax, incl, phiref)

    lnL = 0.
    for det in detectors:
        CT = crossTerms[det]
        F = ComplexAntennaFactor(det, RA, DEC, psi, tref)

        tshifted = ComputeTimeDelay(det, RA, DEC, tref)
        det_rholms_intp = rholms_intp[det]
        shifted_rholms = {}
        for key in det_rholms_intp.keys():
            func = det_rholms_intp[key]
            shifted_rholms[key] = func(tshifted)

        lnL += SingleDetectorLogLikelihood(shifted_rholms, CT, Ylms, F, dist)

    return lnL


#
# Internal functions
#
def ComputeModeIPTimeSeries(hlms, data, psd, fmin, fNyq, analyticPSD_Q=False,
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
    for discrete values of the reference time tref.

    Can optionally give arguments to return only the inner product reference
    times in the range: [tref - N * deltaT, tref + N * deltaT]
    where deltaT is the time step size stored in 'hlms'
    """
    # FIXME: For now not handling tref, N
    assert tref==None and N==None

    # Create an instance of class to compute inner product time series
    if analyticPSD_Q==False:
        assert data.deltaF == psd.deltaF
        IP = ComplexOverlap(fmin, fNyq, data.deltaF, psd.data.data, False, True)
    else:
        IP = ComplexOverlap(fmin, fNyq, data.deltaF, psd, True, True)

    # Loop over modes and compute the overlap time series
    rholms = None
    Lmax = lalsim.SphHarmFrequencySeriesGetMaxL(hlms)
    for l in range(2,Lmax+1):
        for m in range(-l,l+1):
            hlm = lalsim.SphHarmFrequencySeriesGetMode(hlms, l, m)
            rho, rhoTS, rhoIdx, rhoPhase = IP.ip(hlm, data)
            rholms = lalsim.SphHarmTimeSeriesAddMode(rholms, rhoTS, l, m)

    # FIXME: Add ability to cut down to a narrow time window

    return rholms

def InterpolateRholms(rholms, t, Lmax):
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
            amp = np.abs(rholm.data.data)
            phase = unwind_phase( np.angle(rholm.data.data) )
            ampintp = interpolate.InterpolatedUnivariateSpline(t, amp, k=3)
            phaseintp = interpolate.InterpolatedUnivariateSpline(t, phase, k=3)
            rholm_intp[ (l,m) ] = lambda x: ampintp(x)*np.exp(1j*phaseintp(x))

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
        IP = ComplexIP(fmin, fNyq, data.deltaF, psd.data.data, False)
    else:
        IP = ComplexIP(fmin, fNyq, deltaF, psd, True)

    # Loop over modes and compute the inner products, store in a dictionary
    Lmax = lalsim.SphHarmFrequencySeriesGetMaxL(hlms)
    crossTerms = {}
    for l in range(2,Lmax+1):
        for m in range(-l,l+1):
            for lp in range(2,Lmax+1):
                for mp in range(-lp,lp+1):
                    hlm = lalsim.SphHarmFrequencySeriesGetMode(hlms, l, m)
                    hlpmp = lalsim.SphHarmFrequencySeriesGetMode(hlms, lp, mp)
                    crossTerms[ ((l,m),(lp,mp)) ] = IP.ip(hlm, hlpmp)

    return crossTerms

# Create complex FD data that does not assume Hermitianity - i.e.
# contains positive and negative freq. content
def non_herm_hoff(P):
    hp, hc = lalsim.SimInspiralChooseTDWaveform(P.phiref, P.deltaT, P.m1, P.m2, 
            P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin, P.fref, P.dist, 
            P.incl, P.lambda1, P.lambda2, P.waveFlags, P.nonGRparams,
            P.ampO, P.phaseO, P.approx)
    hp.epoch = hp.epoch + P.tref
    hc.epoch = hc.epoch + P.tref
    hoft = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc,
            P.theta, P.phi, P.psi,
            lalsim.InstrumentNameToLALDetector(P.detector))
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

