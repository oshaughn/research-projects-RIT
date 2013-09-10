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
Code to compute the likelihood of intrinsic parameters of a gravitational
waveform marginalized over all extrinsic parameters given some interferometer
data.

Requires python SWIG bindings of the LIGO Algorithms Library (LAL)
"""

from lalsimutils import *

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>"



def MarginalizedLikelihood(P, data_dict, psd_dict, sample_dist):
    """
    Compute the likelihood of parameters P, given data, marginalized
    over all extrinsic parameters.
    
    'P' is a ChooseWaveformParams object.

    'data_dict' is a dictionary whose keys are names of detectors (e.g. 'H1')
        and whose values are pointers to REAL8TimeSeries of the 
        interferometer time series data

    'psd_dict' is a dictionary whose values are pointers to REAL8TimeSeries
        of the power spectral densities (PSDs) for the keyed interferometers.

    'sample_dist' is an ExtrinsicSampleDistribution object. It includes:
        - a list of extrinsic parameters names (keys)
        - the allowed range of each extrinsic parameter, keyed on the
            parameter names
        - functions to draw a random value of the variable from its
            sample distribution, keyed on the parameter names
    """
    # Sanity checks
    assert data_dict.keys() == psd_dict.keys()

    detectors = data_dict.keys()
    Ndet = len(detectors)
    Lmax = 2

    # These are the initial values of extrinsic parameters
    iota = P.incl
    psi = P.psi
    tref = P.tref
    phiref = P.phiref
    DEC = P.theta
    RA = P.phi
    dist = P.dist

    # Compute all FD h_lm waveform modes for masses (m1,m2).
    # Note that 'hlms' will be the linked list of modes in the geocenter frame
    # with the coalescence arriving at time tref=0
    hlms = hlmoff(P, Lmax)

    # Loop over detectors and compute the complex-valued SNR 
    # of each hlm vs the data for tref at each discrete sample of the hlms
    #
    # FIXME: Need to decide if we keep all time shifts, just some subset,
    #       or bypass FFT and do inner product integral for each of a
    #       small numbers of time shifts
    rholms = {}
    for det in detectors:
        # rholms is a dictionary, keyed on detector name, with each value a
        # linked list of SNR time series for that mode and detector
        rholms[det] = ComputeModeIPTimeSeries(hlms, data_dict[det],
                psd_dict[det], P.fmin, P.fNyq)
        # Compute < h_lm | h_l'm' >
        # crossTerms_dict is a dictionary keyed in detector name.
        # It's values crossTerms_dict[det] are each dictionaries keyed on
        # tuples of mode index pairs, e.g.
        # crossTerms_dict[det][ ( (l,m) , (l',m') ) ]
        # holds the value of < h_lm | h_l'm' > for detector 'det'
        crossTerms_dict[det] = ComputeModeCrossTermIP(hlms, psd_dict[det])

    # Draw random values of extrinsic parameters from our sampling distribution.
    # Compute the likelihood at each sample
    # Keep looping until some convergence criterion is met
    converged = False
    logL = []
    while converged is False:
        # Draw extrinsic parameters from the sampling distribution
        extr = DrawFromExtrinsicSampleDistribution(sample_dist)

        #
        # Compute extrinsic pieces of the likelihood
        #

        # If inclination or phiref orientation changed, recompute Ylms
        if extr.inclination != iota or extr.polarization_angle != psi:
            iota = extr.inclination
            phiref = extr.phiref
            Ylms = ComputeYlms(Lmax, iota, phiref)

        # Check if time, RA, DEC, or psi changed
        timeFlag = False
        antennaFlag = False
        if extr.tref != tref:
            timeFlag = True
            antennaFlag = True
        if extr.right_ascension != RA or extr.declination != DEC\
                or extr.polarization_angle != psi:
            RA = extr.right_ascension
            DEC = extr.declination
            psi = extr.polarization_angle
            antennaFlag = True

        dist = extr.distance

        # Loop over detectors
        temp = 0.
        for det in detectors:
            # If sky (or time) has change, recompute antenna pattern factor
            if antennaFlag == True:
                F = ComplexAntennaFactor(det, RA, DEC, psi, tref)

            # Introduce geocenter-to-detector time shift
            tshifted = ComputeTimeDelay(det, RA, DEC, tref)
            shifted_rholms = TimeShiftIPTimeSeries(rholms[det], tshifted)

            # Running total of log-likelihood in all detectors
            temp += ComputeLogLikelihood(shifted_rholms, crossTerms_dict[det],
                    Ylms, F, dist)

        # Add likelihood at current extrinsic parameters to an array
        # FIXME: What's the best way to store/sort this info?
        logL.append(temp)

        # Estimate the marginalized log-likelihood from our sampling of logL
        #margL = MarginalizeOverLikelihood(logL)
        margL = logL[-1]

        # Do some sort of convergence test to determine when to stop marg.
        converged = CheckMarginalizationConvergence(margL, logL)

    return margL


def ComputeModeIPTimeSeries(hlms, data, psd, fmin, fNyq, tref=None, N=None):
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
    # Sanity checks
    assert data.deltaF == psd.deltaF

    # FIXME: For now not handling tref, N
    assert tref==None and N==None

    # Create an instance of class to compute inner product time series
    IP = ComplexOverlap(fmin, fNyq, psd.deltaF, psd.data.data, False, True)

    # Loop over modes and compute the overlap time series
    rholms = None
    Lmax = lalsim.SphHarmFrequencySeriesGetMaxL(hlms)
    for l in range(2,Lmax+1):
        for m in range(-l,l+1):
            hlm = lalsim.SphHarmFrequencySeriesGetMode(hlms, l, m)
            rho, rhoTS, rhoIdx, rhoPhase = IP.ip(hlm, data)
            rholms = lalsim.SphHarmFrequencySeriesAddMode(rholms, rhoTS, l, m)

    # FIXME: Add ability to cut down to a narrow time window

    return rholms


def ComputeModeCrossTermIP(hlms, psd, fmin, fNyq):
    """
    Compute the 'cross terms' between waveform modes, i.e.
    < h_lm | h_l'm' >.
    The inner product is weighted by power spectral density 'psd' and
    integrated over the interval [-fNyq, -fmin] \union [fmin, fNyq]

    Returns a dictionary of inner product values keyed by tuples of mode indices
    i.e. ((l,m),(l',m'))
    """
    # Create an instance of class to compute inner product
    IP = ComplexIP(fmin, fNyq, psd.deltaF, psd.data.data, False)

    # Loop over modes and compute the inner products, store in a dictionary
    crossTerms = {}
    for l in range(2,Lmax+1):
        for m in range(-l,l+1):
            for lp in range(2,Lmax+1):
                for mp in range(-lp,lp+1):
                    hlm = lalsim.SphHarmFrequencySeriesGetMode(hlms, l, m)
                    hlpmp = lalsim.SphHarmFrequencySeriesGetMode(hlms, lp, mp)
                    temp = IP.ip(hlm, hlpmp)
                    crossTerms[ ((l,m),(lp,mp)) ] = temp

    return crossTerms

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
    for l in range(2,Lmax+2):
        for m in range(-l,l+1):
            Ylms[ (l,m) ] = lal.SpinWeightedSphericalHarmonic(theta, phi,
                    -2, l, m)

    return Ylms

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
    Fp, Fc = lal.ComputeDetAMResponse(detector.response, RA, DEC, psi, tref)

    return Fp + 1j * Fc

def ComputeTimeDelay(det, RA, DEC, tref):
    """
    Function to compute the time of arrival at a detector
    from the time of arrival at the geocenter.

    'det' is a detector prefix string (e.g. 'H1')
    'RA' and 'DEC' are right ascension and declination (in radians)
    'tref' is the reference time at the geocenter
    """
    detector = lalsim.DetectorPrefixToLALDetector(det)
    return tref + lal.TimeDelayFromEarthCenter(detector.location, RA, DEC, tref)

def ComputeLogLikelihood(rholms, crossTerms, Ylms, F, dist):
    """
    Compute the log-likelihood of a signal in a single detector:
    ln L = -1/2 < h(theta) - d | h(theta) - d >
    where h is the measured strain for a waveform with parameters theta
    and d is the data of a single detector.

    Depends on terms:
    'rholms' is a time series of < h_lm(tref) | d >
    'crossTerms' is a dictionary containing < h_lm | h_l'm' >
    'Ylms' is a dictionary containing the -2Y_lm's
    'F' is the complex-valued antenna pattern F+ + i Fx
    'dist' is the distance to the source (in meters)
    """
    # FIXME: Placeholder function
    return 1.

def CheckMarginalizationConvergence(margL, logL):
    """
    Perform a check whether the estimate of the marginalized likelihood (margL)
    has converged, given a collection of evaluation of the log-likelihood (logL)

    Returns True if the calculation has converged
    Returns False if it is not converged
    """
    # FIXME: Placeholder function
    limit = 1000
    if len(logL) >= limit:
        return True
    else:
        return False

def TimeShiftIPTimeSeries():
