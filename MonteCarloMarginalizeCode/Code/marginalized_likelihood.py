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

import numpy as np
import lal
import lalsimulation as lalsim

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>"



def MarginalizedLikelihood(m1, m2, data_dict, psd_dict, sample_dist):
    """
    Compute the likelihood of masses (m1,m2), given data, marginalized
    over all extrinsic parameters.
    
    m1, m2 should be given in solar masses.

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
    
    # FIXME: I'm hardcoding some params we may want as arguments later
    Ndet = len(data_dict.keys())
    phiref = 0.
    deltaT = 1./4096.
    s1x = 0.
    s1y = 0.
    s1z = 0.
    s2x = 0.
    s2y = 0.
    s2z = 0.
    fmin = 40.
    fref = 0.
    dist = 1.e6 * lal.LAL_PC_SI
    incl = 0.
    lambda1 = 0.
    lambda2 = 0.
    waveFlags = None
    nonGRparams = None
    ampO = 0
    phaseO = 7
    approx = lalsim.TaylorT4
    Lmax = 2

    # Create a linked list of all h_lm waveform modes for masses (m1,m2)
    # Note that 'hlms' will be the linked list of modes in the barycenter frame
    # with the coalescence arriving at time tref=0
    #
    # FIXME: The above computes TD modes, we will want the FD modes.
    # I'll write a LAL function to do the conversion
    hlms = lalsim.SimInspiralChooseTDModes(phiref, deltaT,
            m1*lal.LAL_MSUN_SI, m2*lal.LAL_MSUN_SI, s1x, s1y, s1z,
            s2x, s2y, s2z, fmin, fref, dist, incl, lambda1, lambda2,
            waveFlags, nonGRparams, ampO, phaseO, Lmax, approx)

    # Loop over detectors and compute the complex-valued SNR 
    # of each hlm vs the data for tref at each discrete sample of the hlms
    #
    # FIXME: Need to decide if we keep all time shifts, just some subset,
    #       or bypass FFT and do inner product integral for each of a
    #       small numbers of tiem shifts
    rholms = {}
    for det in data_dict:
        # rholms is a dictionary, keyed on detector name, with each value a
        # linked list of SNR time series for that mode and detector
        rholms[det] = ComputeModeSNRSeries(hlms, data_dict[det], psd_dict[det])

    # Draw random values of extrinsic parameters from our sampling distribution.
    # Compute the likelihood at each sample
    # Keep looping until some convergence criterion is met and we are satisfied
    # that we have the marginalized likelihood to sufficient accuracy
    converged = False
    logL = []
    while converged is False:
        # Draw extrinsic parameters from the sampling distribution
        extrParams = DrawFromExtrinsicSampleDistribution(sample_dist)

        # Compute extrinsic pieces of the likelihood
        # FIXME: We have LAL code to compute a single mode. Need to write a
        # wrapper function to compute all modes and put them in some object.
        # Maybe a python dictionary for the object???
        iota = extrParams.inclination
        psi = extrParams.polarization_angle
        Ylms = ComputeYlms(Lmax, iota, psi)

        # Loop over detectors
        temp = 0.
        for det in data_dict:
            tref = extrParams.tref
            RA = extrParams.right_ascension
            DEC = extrParams.declination

            # Compute detector-dependent extrinsic factors
            F = ComplexAntennaFactor(det, RA, DEC)

            # Introduce barycenter-to-detector time shift
            tshift = ComputeTimeDelay(tref, RA, DEC, det)
            det_data = TimeShiftCOMPLEX16FrequencySeries(det_data, tshift)

            # Running total of log-likelihood in all detectors
            temp += ComputeLogLikelihood(det_data, rholms, Ylms, F)

        # Add likelihood at current extrinsic parameters to an array
        # FIXME: What's the best way to store/sort this info?
        logL.append(temp)

        # Estimate the marginalized log-likelihood from our sampling of logL
        margL = MarginalizeOverLikelihood(logL)

        # Do some sort of convergence test to determine when to stop marg.
        converged = CheckMarginalizationConvergence(margL, logL)

    return margL


