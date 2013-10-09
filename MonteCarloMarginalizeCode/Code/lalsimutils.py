# Copyright (C) 2012  Evan Ochsner, R. O'Shaughnessy
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
A collection of useful data analysis routines
built from the SWIG wrappings of LAL and LALSimulation.
"""
import copy

import numpy as np
from numpy import sin, cos
from scipy import interpolate
from scipy import signal
#import scipy  # for decimate

from glue.ligolw import lsctables, table, utils # check all are needed
from glue.lal import Cache

import lal
import lalsimulation as lalsim
import lalinspiral
import lalmetaio

from pylal import frutils
from pylal.series import read_psd_xmldoc

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>, R. O'Shaughnessy"


rosDebugMessagesContainer = [True]
print "[Loading lalsimutils.py : MonteCarloMarginalization version]"


#
# Class to hold arguments of ChooseWaveform functions
#
class ChooseWaveformParams:
    """
    Class containing all the arguments needed for SimInspiralChooseTD/FDWaveform
    plus parameters theta, phi, psi, radec to go from h+, hx to h(t)

    if radec==True: (theta,phi) = (DEC,RA) and strain will be computed using
            XLALSimDetectorStrainREAL8TimeSeries
    if radec==False: then strain will be computed using a simple routine 
            that assumes (theta,phi) are spherical coord. 
            in a frame centered at the detector
    """
    def __init__(self, phiref=0., deltaT=1./4096., m1=10.*lal.LAL_MSUN_SI, 
            m2=10.*lal.LAL_MSUN_SI, s1x=0., s1y=0., s1z=0., 
            s2x=0., s2y=0., s2z=0., fmin=40., fref=0., dist=1.e6*lal.LAL_PC_SI,
            incl=0., lambda1=0., lambda2=0., waveFlags=None, nonGRparams=None,
            ampO=0, phaseO=7, approx=lalsim.TaylorT4, 
            theta=0., phi=0., psi=0., tref=0., radec=False, detector="H1",
            deltaF=None, fmax=0., # for use w/ FD approximants
            taper=lalsim.LAL_SIM_INSPIRAL_TAPER_NONE # for use w/TD approximants
            ):
        self.phiref = phiref
        self.deltaT = deltaT
        self.m1 = m1
        self.m2 = m2
        self.s1x = s1x
        self.s1y = s1y
        self.s1z = s1z
        self.s2x = s2x
        self.s2y = s2y
        self.s2z = s2z
        self.fmin = fmin
        self.fref = fref
        self.dist = dist
        self.incl = incl
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.waveFlags = waveFlags
        self.nonGRparams = nonGRparams
        self.ampO = ampO
        self.phaseO = phaseO
        self.approx = approx
        self.theta = theta     # DEC.  DEC =0 on the equator; the south pole has DEC = - pi/2
        self.phi = phi         # RA.   
        self.psi = psi
        self.tref = tref
        self.radec = radec
        self.detector = "H1"
        self.deltaF=deltaF
        self.fmax=fmax
        self.taper = taper

    def copy(self):
        """
        Create a deep copy, so copy and original can be changed separately
        """
        return copy.deepcopy(self)

    def print_params(self):
        """
        Print all key-value pairs belonging in the class instance
        """
        print "This ChooseWaveformParams has the following parameter values:"
        print "m1 =", self.m1 / lal.LAL_MSUN_SI, "(Msun)"
        print "m2 =", self.m2 / lal.LAL_MSUN_SI, "(Msun)"
        print "s1x =", self.s1x
        print "s1y =", self.s1y
        print "s1z =", self.s1z
        print "s2x =", self.s2x
        print "s2y =", self.s2y
        print "s2z =", self.s2z
        print "lambda1 =", self.lambda1
        print "lambda2 =", self.lambda2
        print "inclination =", self.incl
        print "distance =", self.dist / 1.e+6 / lal.LAL_PC_SI, "(Mpc)"
        print "reference orbital phase =", self.phiref
        print "time of coalescence =", float(self.tref)
        print "detector is:", self.detector
        if self.radec==False:
            print "Sky position relative to overhead detector is:"
            print "zenith angle =", self.theta, "(radians)"
            print "azimuth angle =", self.phi, "(radians)"
        if self.radec==True:
            print "Sky position relative to geocenter is:"
            print "declination =", self.theta, "(radians)"
            print "right ascension =", self.phi, "(radians)"
        print "polarization angle =", self.psi
        print "starting frequency is =", self.fmin
        print "reference frequency is =", self.fref
        print "Max frequency is =", self.fmax
        print "time step =", self.deltaT, "(s) <==>", 1./self.deltaT,\
                "(Hz) sample rate"
        print "freq. bin size is =", self.deltaF, "(Hz)"
        print "approximant is =", lalsim.GetStringFromApproximant(self.approx)
        print "phase order =", self.phaseO
        print "amplitude order =", self.ampO
        print "waveFlags struct is", self.waveFlags
        print "nonGRparams struct is", self.nonGRparams
        if self.taper==lalsim.LAL_SIM_INSPIRAL_TAPER_NONE:
            print "Tapering is set to LAL_SIM_INSPIRAL_TAPER_NONE"
        elif self.taper==lalsim.LAL_SIM_INSPIRAL_TAPER_START:
            print "Tapering is set to LAL_SIM_INSPIRAL_TAPER_START"
        elif self.taper==lalsim.LAL_SIM_INSPIRAL_TAPER_END:
            print "Tapering is set to LAL_SIM_INSPIRAL_TAPER_END"
        elif self.taper==lalsim.LAL_SIM_INSPIRAL_TAPER_STARTEND:
            print "Tapering is set to LAL_SIM_INSPIRAL_TAPER_STARTEND"
        else:
            print "Warning! Invalid value for taper:", self.taper

    def copy_sim_inspiral(self, row):
        """
        Fill this ChooseWaveformParams with the fields of a
        row of a SWIG wrapped lalmetaio.SimInspiral table

        NB: SimInspiral table does not contain deltaT, deltaF, fref, fmax,
        lambda1, lambda2, waveFlags, nonGRparams, or detector fields, but
        ChooseWaveformParams does have these fields.
        This function will not alter these fields, so their values will
        be whatever values the instance previously had.
        """
        self.phiref = row.coa_phase
        self.m1 = row.mass1 * lal.LAL_MSUN_SI
        self.m2 = row.mass2 * lal.LAL_MSUN_SI
        self.s1x = row.spin1x
        self.s1y = row.spin1y
        self.s1z = row.spin1z
        self.s2x = row.spin2x
        self.s2y = row.spin2y
        self.s2z = row.spin2z
        self.fmin = row.f_lower
        self.dist = row.distance * lal.LAL_PC_SI * 1.e6
        self.incl = row.inclination
        self.ampO = row.amp_order
        self.phaseO = lalsim.GetOrderFromString(row.waveform)
        self.approx = lalsim.GetApproximantFromString(row.waveform)
        self.theta = row.latitude # Declination
        self.phi = row.longitude # Right ascension
        self.radec = True # Flag to interpret (theta,phi) as (DEC,RA)
        self.psi = row.polarization
        self.tref = row.geocent_end_time + 1e-9*row.geocent_end_time_ns
        self.taper = lalsim.GetTaperFromString(row.taper)

    def copy_lsctables_sim_inspiral(self, row):
        """
        Fill this ChooseWaveformParams with the fields of a
        row of an lsctables.SimInspiral table
        (i.e. SimInspiral table in the format as read from a file)

        NB: SimInspiral table does not contain deltaT, deltaF, fref, fmax,
        lambda1, lambda2, waveFlags, nonGRparams, or detector fields, but
        ChooseWaveformParams does have these fields.
        This function will not alter these fields, so their values will
        be whatever values the instance previously had.

        Adapted from code by Chris Pankow
        """
        # Convert from lsctables.SimInspiral --> lalmetaio.SimInspiral
        swigrow = lalmetaio.SimInspiralTable()
        for simattr in lsctables.SimInspiralTable.validcolumns.keys():
            if simattr in ["waveform", "source", "numrel_data", "taper"]:
                # unicode -> char* doesn't work
                setattr( swigrow, simattr, str(getattr(row, simattr)) )
            else:
                setattr( swigrow, simattr, getattr(row, simattr) )
        # Call the function to read lalmetaio.SimInspiral format
        self.copy_sim_inspiral(swigrow)

def xml_to_ChooseWaveformParams_array(fname, minrow=None, maxrow=None,
        deltaT=1./4096., fref=0., lambda1=0., lambda2=0., waveFlags=None,
        nonGRparams=None, detector="H1", deltaF=None, fmax=0.):
    """
    Function to read an xml file 'fname' containing a SimInspiralTable,
    convert rows of the SimInspiralTable into ChooseWaveformParams instances
    and return an array of the ChooseWaveformParam instances

    Can optionally give 'minrow' and 'maxrow' to convert only rows
    in the range (starting from zero) [minrow, maxrow). If these arguments
    are not given, this function will convert the whole SimInspiral table.

    The rest of the optional arguments are the fields in ChooseWaveformParams
    that are not present in SimInspiral tables. Any of these arguments not given
    values will use the standard default values of ChooseWaveformParams.
    """
    xmldoc = utils.load_filename( fname )
    try:
        # Read SimInspiralTable from the xml file, set row bounds
        sim_insp = table.get_table(xmldoc, lsctables.SimInspiralTable.tableName)
        length = len(sim_insp)
        if not minrow and not maxrow:
            minrow = 0
            maxrow = length
        else:
            assert minrow >= 0
            assert minrow <= maxrow
            assert maxrow <= length
        rng = range(minrow,maxrow)
        # Create a ChooseWaveformParams for each requested row
        Ps = [ChooseWaveformParams(deltaT=deltaT, fref=fref, lambda1=lambda1,
            lambda2=lambda2, waveFlags=waveFlags, nonGRparams=nonGRparams,
            detector=detector, deltaF=deltaF, fmax=fmax) for i in rng]
        # Copy the information from requested rows to the ChooseWaveformParams
        [Ps[i-minrow].copy_lsctables_sim_inspiral(sim_insp[i]) for i in rng]
    except ValueError:
        print >>sys.stderr, "No SimInspiral table found in xml file"
    return Ps


#
# Classes for computing inner products of waveforms
#
class InnerProduct:
    """
    Base class for inner products
    """
    def __init__(self, fLow=10., fNyq=2048., deltaF=1./8.,
            psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower, analyticPSD_Q=True):
        self.fLow = fLow
        self.fNyq = fNyq
        self.deltaF = deltaF
        if analyticPSD_Q:
            self.psd = np.vectorize(psd)
        else:
            self.psd = psd
        self.minIdx = int(fLow/deltaF)
        self.FDlen = int(fNyq/deltaF)+1
        self.weights = np.zeros(self.FDlen)
        self.longweights = np.zeros(2*(self.FDlen-1))  # for not hermetian inner products
        self.analyticPSD_Q = analyticPSD_Q
        if analyticPSD_Q == True:
            if rosDebugMessagesContainer[0]:
                print "  ... populating inner product weight array using analytic PSD ... "
#            self.weights[self.minIdx:self.FDlen] = 1.0/self.psd(np.arange(self.minIdx, self.FDlen, 1))
#            # Take 1 sided PSD and make it 2 sided
#            self.longweights[1:1+len(self.weights)] = self.weights[::-1]
#            self.longweights[-(len(self.weights)+1):-1] = self.weights[:]
            for i in range(self.minIdx,self.FDlen):        # populate weights for both hermetian and non-hermetian products
                self.weights[i] = 1./self.psd(i*deltaF)
                length = 2*(self.FDlen-1)
                self.longweights[length/2 - i+1] = 1./self.psd(i*deltaF)
                self.longweights[length/2 + i-1] = 1./self.psd(i*deltaF)
            if rosDebugMessagesContainer[0]:
                print "  ... finished populating inner product weight array using analytic PSD ... "
        else:
            if rosDebugMessagesContainer[0]:
                print "  ... populating inner product weight array using a numerical PSD ... "
            for i in range(self.minIdx,self.FDlen):
                if psd[i] != 0.:
                    self.weights[i] = 1./psd[i]
                    length = 2*(self.FDlen-1)
                    self.longweights[length/2 - i+1] = 1./psd[i]
                    self.longweights[length/2 + i-1] = 1./psd[i]
            if rosDebugMessagesContainer[0]:
                print "  ... finished populating inner product weight array using a numerical PSD ... "

    def ip(self, h1, h2):
        """
        Compute inner product between two COMPLEX16Frequency Series
        """
        raise Exception("This is the base InnerProduct class! Use a subclass")

    def norm(self, h):
        """
        Compute norm of a COMPLEX16Frequency Series
        """
        raise Exception("This is the base InnerProduct class! Use a subclass")

class RealIP(InnerProduct):
    """
    Real-valued inner product. self.ip(h1,h2) computes

             fNyq
    4 Re \int      h1(f) h2*(f) / Sn(f) df
             fLow

    And similarly for self.norm(h1)

    DOES NOT maximize over time or phase
    """
    def ip(self, h1, h2):
        """
        Compute inner product between two COMPLEX16Frequency Series
        """
        assert h1.data.length <= self.FDlen
        assert h2.data.length <= self.FDlen
        assert abs(h1.deltaF-h2.deltaF)<=1.e-5 and abs(h1.deltaF-self.deltaF)<=1.e-5
        val = 0.
        maxIdx = min(h1.data.length,h2.data.length)
        val = np.sum(np.conj(h1.data.data)*h2.data.data*self.weights)
        # for i in range(self.minIdx,maxIdx):
        #     val += h1.data.data[i].conj() * h2.data.data[i]* self.weights[i]
        val = 4. * self.deltaF * np.real(val)
        return val

    def norm(self, h):
        """
        Compute norm of a COMPLEX16Frequency Series
        """
        assert h.data.length <= self.FDlen
        assert abs(h.deltaF-self.deltaF) <= 1.e-5
        val = 0.
        val = np.sum(np.conj(h.data.data)*h.data.data*self.weights)
        # for i in range(self.minIdx,h.data.length):
        #     val += h.data.data[i] * h.data.data[i].conj() * self.weights[i]
        val = np.sqrt( 4. * self.deltaF * np.abs(val) )
        return val

class HermitianComplexIP(InnerProduct):
    """
    Complex-valued inner product. self.ip(h1,h2) computes

          fNyq
    4 \int      h1(f) h2*(f) / Sn(f) df
          fLow

    And similarly for self.norm(h1)

    N.B. Assumes h1, h2 are Hermitian - i.e. they store only positive freqs.
         with negative freqs. given by h(-f) = h*(f)
    DOES NOT maximize over time or phase
    """
    def ip(self, h1, h2):
        """
        Compute inner product between two COMPLEX16Frequency Series
        """
        assert h1.data.length <= self.FDlen
        assert h2.data.length <= self.FDlen
        assert abs(h1.deltaF-h2.deltaF)<=1.e-5 and abs(h1.deltaF-self.deltaF)<=1.e-5
        val = 0.
        maxIdx = min(h1.data.length,h2.data.length)
        val = np.sum(np.conj(h1.data.data)*h2.data.data*self.weights)
        # for i in range(self.minIdx,maxIdx):
        #     val += h1.data.data[i].conj() * h2.data.data[i] * self.weights[i]
        val *= 4. * self.deltaF
        return val

    def norm(self, h):
        """
        Compute norm of a COMPLEX16Frequency Series
        """
        assert h.data.length <= self.FDlen
        assert abs(h.deltaF-self.deltaF) <= 1.e-5
        val = 0.
        for i in range(self.minIdx,h.data.length):
            val += h.data.data[i] * h.data.data[i].conj() * self.weights[i]
        val = np.sqrt( 4. * self.deltaF * np.abs(val) )
        return val

class ComplexIP(InnerProduct):
    """
    Complex-valued inner product. self.ip(h1,h2) computes

          fNyq
    2 \int      h1(f) h2*(f) / Sn(f) df
          -fNyq

    And similarly for self.norm(h1)

    N.B. DOES NOT assume h1, h2 are Hermitian - they should contain negative
         and positive freqs. packed as
    [ -N/2 * df, ..., -df, 0, df, ..., (N/2-1) * df ]
    DOES NOT maximize over time or phase
    """
    def ip(self, h1, h2):
        """
        Compute inner product between two COMPLEX16Frequency Series
        """
        assert h1.data.length==h2.data.length==2*(self.FDlen-1)
        assert abs(h1.deltaF-h2.deltaF)<=1.e-5 and abs(h1.deltaF-self.deltaF)<=1.e-5
        val = 0.
        val = np.sum( np.conj(h1.data.data)*h2.data.data *self.longweights)
        # for i in range(self.minIdx,length/2):
        #     val += (h1.data.data[length/2-i].conj() * h2.data.data[length/2-i]\
        #             + h1.data.data[length/2+i].conj()\
        #             * h2.data.data[length/2+i]) * self.weights[i]
        val *= 2. * self.deltaF
        return val

    def norm(self, h):
        """
        Compute norm of a COMPLEX16Frequency Series
        """
        assert h.data.length==2*(self.FDlen-1)
        assert abs(h.deltaF-self.deltaF) <= 1.e-5
        length = h.data.length
        val = 0.
        val = np.sum( np.conj(h.data.data)*h.data.data *self.longweights)
        # for i in range(self.minIdx,length/2):
        #     val += (h.data.data[length/2 - i] * h.data.data[length/2 -i].conj()\
        #             + h.data.data[length/2+i]\
        #             * h.data.data[length/2+i].conj()) * self.weights[i]
        val = np.sqrt( 2. * self.deltaF * np.abs(val) )
        return val

class Overlap(InnerProduct):
    """
    Inner product maximized over time and phase. self.ip(h1,h2) computes:

                  fNyq
    max 4 Abs \int      h1(f) h2*(f,tc) / Sn(f) df
     tc           fLow

    If self.full_output==False: returns
        The maximized (real-valued, > 0) overlap
    If self.full_output==True: returns
        The maximized overlap
        The entire COMPLEX16TimeSeries of overlaps for each possible time shift
        The index of the above time series at which the maximum occurs
        The phase rotation which maximizes the real-valued overlap
    """
    def __init__(self, fLow=10., fNyq=2048., deltaF=1./8.,
            psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower, analyticPSD_Q=True,
            full_output=False, revplan=None, intgd=None, ovlp=None):
        self.fLow = fLow
        self.fNyq = fNyq
        self.deltaF = deltaF
        self.psd = psd
        self.minIdx = int(fLow/deltaF)
        self.FDlen = int(fNyq/deltaF)+1
        self.TDlen = 2*(self.FDlen-1)
        self.deltaT = 1./self.deltaF/self.TDlen
        self.weights = np.zeros(self.FDlen)
        self.longweights = np.zeros(2*(self.FDlen-1))  # for not hermetian inner products
        self.analyticPSD_Q = analyticPSD_Q
        self.full_output=full_output
        self.revplan = revplan
        self.intgd=intgd
        self.ovlp=ovlp
        # Create FFT plan and workspace vectors if not provided
        if self.revplan==None:
            self.revplan=lal.CreateReverseCOMPLEX16FFTPlan(self.TDlen, 0)
        if self.intgd==None:
            self.intgd = lal.CreateCOMPLEX16FrequencySeries("SNR integrand", 
                lal.LIGOTimeGPS(0.), 0., self.deltaF,
                lal.lalHertzUnit, self.TDlen)
        if self.ovlp==None:
            self.ovlp = lal.CreateCOMPLEX16TimeSeries("Complex overlap", 
                lal.LIGOTimeGPS(0.), 0., self.deltaT, lal.lalDimensionlessUnit,
                self.TDlen)
        # Compute the weights
        if analyticPSD_Q == True:
            if rosDebugMessagesContainer[0]:
                print "  ... populating inner product weight array using analytic PSD ... "
            for i in range(self.minIdx,self.FDlen):
                self.weights[i] = 1./self.psd(i*deltaF)
                length = 2*(self.FDlen-1)
                self.longweights[length/2 - i+1] = 1./self.psd(i*deltaF)
                self.longweights[length/2 + i-1] = 1./self.psd(i*deltaF)
            if rosDebugMessagesContainer[0]:
                print "  ... finished populating inner product weight array using analytic PSD ... "
        else:
            if rosDebugMessagesContainer[0]:
                print "  ... populating inner product weight array using a numerical PSD ... "
            for i in range(self.minIdx,self.FDlen):
                if psd[i] != 0.:
                    self.weights[i] = 1./psd[i]
                    length = 2*(self.FDlen-1)
                    self.longweights[length/2 - i+1] = 1./psd[i]
                    self.longweights[length/2 + i-1] = 1./psd[i]
            if rosDebugMessagesContainer[0]:
                print "  ... finished populating inner product weight array using a numerical PSD ... "


    def ip(self, h1, h2):
        """
        Compute inner product between two COMPLEX16Frequency Series
        """
        assert h1.data.length <= self.FDlen
        assert h2.data.length <= self.FDlen
        # Tabulate the SNR integrand
        maxIdx = min(h1.data.length,h2.data.length)
        for i in range(self.TDlen):
            self.intgd.data.data[i] = 0.
        for i in range(self.minIdx,maxIdx):
            self.intgd.data.data[i] = 4.*np.conj(h1.data.data[i])*h2.data.data[i]*self.weights[i]
        # Reverse FFT to get overlap for all possible reference times
        lal.COMPLEX16FreqTimeFFT(self.ovlp, self.intgd, self.revplan)
        rhoSeries = np.abs(self.ovlp.data.data)
        rho = rhoSeries.max()
        if self.full_output==False:
            # Return overlap maximized over time, phase
            return rho
        else:
            # Return max overlap, full overlap time series and other info
            rhoIdx = rhoSeries.argmax()
            rhoPhase = np.angle(self.ovlp.data.data[rhoIdx])
            return rho, self.ovlp, rhoIdx, rhoPhase

    def norm(self, h):
        """
        Compute norm of a COMPLEX16Frequency Series
        """
        assert h.data.length <= self.FDlen
        assert abs(h.deltaF-self.deltaF) <= 1.e-5
        val = 0.
        val = np.sum( np.conj(h.data.data)*h.data.data *self.weights)
        # for i in range(self.minIdx,h.data.length):
        #     val += h.data.data[i] * h.data.data[i].conj() * self.weights[i]
        val = np.sqrt( 4. * self.deltaF * np.abs(val) )
        return val

    def wrap_times(self):
        """
        Return a vector of wrap-around time offsets, i.e.
        [ 0, dt, 2 dt, ..., N dt, -(N-1) dt, -(N-1) dt, ..., -2 dt, -dt ]

        This is useful in conjunction with the 'full_output' option to plot
        the overlap vs timeshift. e.g. do:

        IP = Overlap(full_output=True)
        t = IP.wrap_times()
        rho, ovlp, rhoIdx, rhoPhase = IP.ip(h1, h2)
        plot(t, abs(ovlp))
        """
        tShift = np.arange(self.TDlen) * self.deltaT
        for i in range(self.FDlen,self.TDlen):
            tShift[i] -= self.TDlen * self.deltaT
        return tShift

class ComplexOverlap(InnerProduct):
    """
    Inner product maximized over time and polarization angle. 
    This inner product does not assume Hermitianity and is therefore
    valid for waveforms that are complex in the TD, e.g. h+(t) + 1j hx(t).
    self.IP(h1,h2) computes:

                  fNyq
    max 2 Abs \int      h1(f) h2*(f,tc) / Sn(f) df
     tc          -fNyq


    If self.full_output==False: returns
        The maximized overlap
    If self.full_output==True: returns
        The maximized overlap
        The entire COMPLEX16TimeSeries of overlaps for each possible time shift
        The index of the above time series at which the maximum occurs
        The phase rotation which maximizes the real-valued overlap
    """
    def __init__(self, fLow=10., fNyq=2048., deltaF=1./8.,
            psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower, analyticPSD_Q=True,
            full_output=False, revplan=None, intgd=None, ovlp=None):
        self.fLow = fLow
        self.fNyq = fNyq
        self.deltaF = deltaF
        self.psd = psd
        self.minIdx = int(fLow/deltaF)
        self.wgtslen = int(fNyq/deltaF)+1
        self.wvlen = 2*(self.wgtslen-1)
        self.deltaT = 1./self.deltaF/self.wvlen
        self.weights = np.zeros(self.wgtslen)
        self.longweights = np.zeros(2*(self.wgtslen-1))  # for not hermetian inner products
        self.longpsdLAL = lal.CreateCOMPLEX16FrequencySeries("PSD",lal.LIGOTimeGPS(0.), 0., self.deltaF,lal.lalHertzUnit, self.wvlen)
        self.analyticPSD_Q = analyticPSD_Q
        self.full_output=full_output
        self.revplan = revplan
        self.intgd=intgd
        self.ovlp=ovlp
        # Create FFT plan and workspace vectors if not provided
        if self.revplan==None:
            self.revplan=lal.CreateReverseCOMPLEX16FFTPlan(self.wvlen, 0)
        if self.intgd==None:
            self.intgd = lal.CreateCOMPLEX16FrequencySeries("SNR integrand", 
                lal.LIGOTimeGPS(0.), 0., self.deltaF,
                lal.lalHertzUnit, self.wvlen)
        if self.ovlp==None:
            self.ovlp = lal.CreateCOMPLEX16TimeSeries("Complex overlap", 
                lal.LIGOTimeGPS(0.), 0., self.deltaT, lal.lalDimensionlessUnit,
                self.wvlen)
        # Compute the weights
        if analyticPSD_Q == True:
            for i in range(self.minIdx,self.wgtslen):
                self.weights[i] = 1./self.psd(i*deltaF)
                length = self.wvlen
                self.longweights[length/2 - i+1] = 1./self.psd(i*deltaF)
                self.longweights[length/2 + i-1] = 1./self.psd(i*deltaF)
                self.longpsdLAL.data.data[length/2-i+1] = self.psd(i*deltaF)
                self.longpsdLAL.data.data[length/2+i-1] = self.psd(i*deltaF)
        else:
            for i in range(self.minIdx,self.wgtslen):
                if psd[i] != 0.:
                    self.weights[i] = 1./psd[i]
                    length = self.wvlen
                    self.longweights[length/2 - i+1] = 1./psd[i]
                    self.longweights[length/2 + i-1] = 1./psd[i]
                    self.longpsdLAL.data.data[length/2-i+1] =1./ psd[i]
                    self.longpsdLAL.data.data[length/2+i-1] = 1./psd[i]

    def ip(self, h1, h2):
        """
        Compute inner product between two COMPLEX16Frequency Series
        """
        assert h1.data.length==h2.data.length==self.wvlen
        # Tabulate the SNR integrand
        for i in range(self.wvlen):
            self.intgd.data.data[i] = 0.
        # Note packing of h(f) is monotonic when h(t) is complex:
        # h(-N/2 df), ..., H(-df) h(0), h(df), ..., h(N/2 df)
        # In particular,freqs = +-i*df are in N/2+-i bins of array
        self.intgd.data.data = 2*np.conj(h1.data.data)*h2.data.data*self.longweights  # Dangerous!
        # for i in range(self.wvlen):
        #     self.intgd.data.data[i] = 2*( np.conj(h1.data.data[i])*h2.data.data[i]*self.weights[ np.abs(self.wvlen/2 -i)])
            # self.intgd.data.data[i] = 2* ( h1.data.data[self.wvlen/2-i]\
            #         * h2.data.data[self.wvlen/2-i].conj()\
            #         + h1.data.data[self.wvlen/2+i]\
            #         * h2.data.data[self.wvlen/2+i].conj() ) * self.weights[i]
        # for i in range(self.minIdx,self.wvlen/2):
            # self.intgd.data.data[i] = 2* ( h1.data.data[self.wvlen/2-i]\
            #         * h2.data.data[self.wvlen/2-i].conj()\
            #         + h1.data.data[self.wvlen/2+i]\
            #         * h2.data.data[self.wvlen/2+i].conj() ) * self.weights[i]
        # Reverse FFT to get overlap for all possible reference times
        lal.COMPLEX16FreqTimeFFT(self.ovlp, self.intgd, self.revplan)
        #self.ovlp.data.data = self.deltaF*np.fft.ifft(self.intgd.data.data)   # do it my own way, to be absolutely sure?
        rhoSeries = np.abs(self.ovlp.data.data)
        rho = rhoSeries.max()
        if self.full_output==False:
            # Return overlap maximized over time, phase
            return rho
        else:
            # Return max overlap, full overlap time series and other info
            rhoIdx = rhoSeries.argmax()
            rhoPhase = np.angle(self.ovlp.data.data[rhoIdx])
            return rho, self.ovlp, rhoIdx, rhoPhase

    def norm(self, h):
        """
        Compute inner product between two COMPLEX16Frequency Series
        """
        assert h.data.length==self.wvlen
        assert abs(h.deltaF-self.deltaF) <= 1.e-5
        val = 0.
        # Note monotonic packing of h(f)
        val = np.sum( np.conj(h.data.data)*h.data.data *self.longweights)
        # for i in range(self.minIdx,self.wvlen/2):
        #     val += ( h.data.data[self.wvlen/2-i]\
        #             * h.data.data[self.wvlen/2-i].conj()\
        #             + h.data.data[self.wvlen/2+i]\
        #             * h.data.data[self.wvlen/2+i].conj() ) * self.weights[i]
        val = np.sqrt( 2. * self.deltaF * np.abs(val) )
        return val

    def wrap_times(self):
        """
        Return a vector of wrap-around time offsets, i.e.
        [ 0, dt, 2 dt, ..., N dt, -(N-1) dt, -(N-1) dt, ..., -2 dt, -dt ]

        This is useful in conjunction with the 'full_output' option to plot
        the overlap vs timeshift. e.g. do:

        IP = ComplexOverlap(full_output=True)
        t = IP.wrap_times()
        rho, ovlp, rhoIdx, rhoPhase = IP.ip(h1, h2)
        plot(t, abs(ovlp))
        """
        tShift = np.arange(self.wvlen) * self.deltaT
        for i in range(self.FDlen,self.wvlen):
            tShift[i] -= self.wvlen * self.deltaT
        return tShift



#
# Antenna pattern functions
#
def Fplus(theta, phi, psi):
    """
    Antenna pattern as a function of polar coordinates measured from
    directly overhead a right angle interferometer and polarization angle
    """
    return 0.5*(1. + cos(theta)*cos(theta))*cos(2.*phi)*cos(2.*psi)\
            - cos(theta)*sin(2.*phi)*sin(2.*psi)

def Fcross(theta, phi, psi):
    """
    Antenna pattern as a function of polar coordinates measured from
    directly overhead a right angle interferometer and polarization angle
    """
    return 0.5*(1. + cos(theta)*cos(theta))*cos(2.*phi)*sin(2.*psi)\
            + cos(theta)*sin(2.*phi)*cos(2.*psi)

#
# Mass parameter conversion functions - note they assume m1 >= m2
#
def mass1(Mc, eta):
    """Compute larger component mass from Mc, eta"""
    return 0.5*Mc*eta**(-3./5.)*(1. + np.sqrt(1 - 4.*eta))

def mass2(Mc, eta):
    """Compute smaller component mass from Mc, eta"""
    return 0.5*Mc*eta**(-3./5.)*(1. - np.sqrt(1 - 4.*eta))

def mchirp(m1, m2):
    """Compute chirp mass from component masses"""
    return (m1*m2)**(3./5.)*(m1+m2)**(-1./5.)

def symRatio(m1, m2):
    """Compute symmetric mass ratio from component masses"""
    return m1*m2/(m1+m2)/(m1+m2)

def m1m2(Mc, eta):
    """Compute component masses from Mc, eta. Returns m1 >= m2"""
    m1 = 0.5*Mc*eta**(-3./5.)*(1. + np.sqrt(1 - 4.*eta))
    m2 = 0.5*Mc*eta**(-3./5.)*(1. - np.sqrt(1 - 4.*eta))
    return m1, m2

def Mceta(m1, m2):
    """Compute chirp mass and symmetric mass ratio from component masses"""
    Mc = (m1*m2)**(3./5.)*(m1+m2)**(-1./5.)
    eta = m1*m2/(m1+m2)/(m1+m2)
    return Mc, eta

#
# Other utility functions
#
def unwind_phase(phase,thresh=5.):
    """
    Unwind an array of values of a periodic variable so that it does not jump
    discontinuously when it hits the periodic boundary, but changes smoothly
    outside the periodic range.

    Note: 'thresh', which determines if a discontinuous jump occurs, should be
    somewhat less than the periodic interval. Empirically, 5 is usually a safe
    value of thresh for a variable with period 2 pi.
    """
    cnt = 0 # count number of times phase wraps around branch cut
    length = len(phase)
    unwound = np.zeros(length)
    unwound[0] = phase[0]
    for i in range(1,length):
        if phase[i-1] - phase[i] > thresh: # phase wrapped forward
            cnt += 1
        elif phase[i] - phase[i-1] > thresh: # phase wrapped backward
            cnt += 1
        unwound[i] = phase[i] + cnt * 2. * np.pi
    return unwound

def nextPow2(length):
    """
    Find next power of 2 <= length
    """
    return int(2**np.ceil(np.log2(length)))

def findDeltaF(P):
    """
    Given ChooseWaveformParams P, generate the TD waveform,
    round the length to the next power of 2,
    and find the frequency bin size corresponding to this length.
    This is useful b/c deltaF is needed to define an inner product
    which is needed for norm_hoft and norm_hoff functions
    """
    h = hoft(P)
    return 1./(nextPow2(h.data.length) * P.deltaT)

def sanitize_eta(eta, tol=1.e-10, exception='error'):
    """
    If 'eta' is slightly outside the physically allowed range for
    symmetric mass ratio, push it back in. If 'eta' is further
    outside the physically allowed range, throw an error
    or return a special value.
    Explicitly:
        - If 'eta' is in [tol, 0.25], return eta.
        - If 'eta' is in [0, tol], return tol.
        - If 'eta' in is (0.25, 0.25+tol], return 0.25
        - If 'eta' < 0 OR eta > 0.25+tol,
            - if exception=='error' raise a ValueError
            - if exception is anything else, return exception
    """
    MIN = 0.
    MAX = 0.25
    if eta < MIN or eta > MAX+tol:
        if exception=='error':
            raise ValueError("Value of eta outside the physicaly-allowed range of symmetric mass ratio.")
        else:
            return exception
    elif eta < tol:
        return tol
    elif eta > MAX:
        return MAX
    else:
        return eta

#
# Functions to generate waveforms
#
def hoft(P, Fp=None, Fc=None):
    """
    Generate a TD waveform from ChooseWaveformParams P
    You may pass in antenna patterns Fp, Fc. If none are provided, they will
    be computed from the information in ChooseWaveformParams.

    Returns a REAL8TimeSeries object
    """
    hp, hc = lalsim.SimInspiralChooseTDWaveform(P.phiref, P.deltaT, P.m1, P.m2, 
            P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin, P.fref, P.dist, 
            P.incl, P.lambda1, P.lambda2, P.waveFlags, P.nonGRparams,
            P.ampO, P.phaseO, P.approx)

    if Fp!=None and Fc!=None:
        hp.data.data *= Fp
        hc.data.data *= Fc
        hp = lal.AddREAL8TimeSeries(hp, hc)
        ht = hp
    elif P.radec==False:
        fp = Fplus(P.theta, P.phi, P.psi)
        fc = Fcross(P.theta, P.phi, P.psi)
        hp.data.data *= fp
        hc.data.data *= fc
        hp = lal.AddREAL8TimeSeries(hp, hc)
        ht = hp
    else:
        hp.epoch = hp.epoch + P.tref
        hc.epoch = hc.epoch + P.tref
        ht = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc, 
                P.phi, P.theta, P.psi, 
                lalsim.InstrumentNameToLALDetector(P.detector))
    if P.taper != lalsim.LAL_SIM_INSPIRAL_TAPER_NONE: # Taper if requested
        lalsim.SimInspiralREAL8WaveTaper(ht.data, P.taper)
    if P.deltaF is not None:
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen >= ht.data.length
        ht = lal.ResizeREAL8TimeSeries(ht, 0, TDlen)
    return ht

def hoff(P, Fp=None, Fc=None, fwdplan=None):
    """
    Generate a FD waveform from ChooseWaveformParams P.
    Will return a COMPLEX16FrequencySeries object.

    If P.approx is a FD approximant, hoff_FD is called.
    This path calls SimInspiralChooseFDWaveform
        fwdplan must be None for FD approximants.

    If P.approx is a TD approximant, hoff_TD is called.
    This path calls ChooseTDWaveform and performs an FFT.
        The TD waveform will be zero-padded so it's Fourier transform has
        frequency bins of size P.deltaT.
        If P.deltaF == None, the TD waveform will be zero-padded
        to the next power of 2.
    """
    # For FD approximants, use the ChooseFDWaveform path = hoff_FD
    if lalsim.SimInspiralImplementedFDApproximants(P.approx)==1:
        # Raise exception if unused arguments were specified
        if fwdplan is not None:
            raise ValueError('FFT plan fwdplan given with FD approximant.\nFD approximants cannot use this.')
        hf = hoff_FD(P, Fp, Fc)

    # For TD approximants, do ChooseTDWaveform + FFT path = hoff_TD
    else:
        hf = hoff_TD(P, Fp, Fc, fwdplan)

    return hf

def hoff_TD(P, Fp=None, Fc=None, fwdplan=None):
    """
    Generate a FD waveform from ChooseWaveformParams P
    by creating a TD waveform, zero-padding and
    then Fourier transforming with FFTW3 forward FFT plan fwdplan

    If P.deltaF==None, just pad up to next power of 2
    If P.deltaF = 1/X, will generate a TD waveform, zero-pad to length X seconds
        and then FFT. Will throw an error if waveform is longer than X seconds

    If you do not provide a forward FFT plan, one will be created.
    If you are calling this function many times, you may to create it
    once beforehand and pass it in, e.g.:
    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)

    You may pass in antenna patterns Fp, Fc. If none are provided, they will
    be computed from the information in ChooseWaveformParams

    Returns a COMPLEX16FrequencySeries object
    """
    ht = hoft(P, Fp, Fc)

    if P.deltaF == None: # h(t) was not zero-padded, so do it now
        TDlen = nextPow2(ht.data.length)
        ht = lal.ResizeREAL8TimeSeries(ht, 0, TDlen)
    else: # Check zero-padding was done to expected length
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen == ht.data.length
    
    if fwdplan==None:
        fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)
    FDlen = TDlen/2+1
    hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            ht.epoch, ht.f0, 1./ht.deltaT/TDlen, lal.lalHertzUnit, 
            FDlen)
    lal.REAL8TimeFreqFFT(hf, ht, fwdplan)
    return hf

def hoff_FD(P, Fp=None, Fc=None):
    """
    Generate a FD waveform for a FD approximant.
    Note that P.deltaF (which is None by default) must be set
    """
    if P.deltaF is None:
        raise ValueError('None given for freq. bin size P.deltaF')

    hptilde, hctilde = lalsim.SimInspiralChooseFDWaveform(P.phiref, P.deltaF,
            P.m1, P.m2, P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin,
            P.fmax, P.dist, P.incl, P.lambda1, P.lambda2, P.waveFlags,
            P.nonGRparams, P.ampO, P.phaseO, P.approx)
    if Fp is not None and Fc is not None:
        hptilde.data.data *= Fp
        hctilde.data.data *= Fc
        hptilde = lal.AddCOMPLEX16FrequencySeries(hptilde, hctilde)
        htilde = hptilde
    elif P.radec==False:
        fp = Fplus(P.theta, P.phi, P.psi)
        fc = Fcross(P.theta, P.phi, P.psi)
        hptilde.data.data *= fp
        hctilde.data.data *= fc
        hptilde = lal.AddCOMPLEX16FrequencySeries(hptilde, hctilde)
        htilde = hptilde
    else:
        raise ValueError('Must use P.radec=False for FD approximant (for now)')
    return htilde

def norm_hoff(P, IP, Fp=None, Fc=None, fwdplan=None):
    """
    Generate a normalized FD waveform from ChooseWaveformParams P.
    Will return a COMPLEX16FrequencySeries object.

    If P.approx is a FD approximant, norm_hoff_FD is called.
    This path calls SimInspiralChooseFDWaveform
        fwdplan must be None for FD approximants.

    If P.approx is a TD approximant, norm_hoff_TD is called.
    This path calls ChooseTDWaveform and performs an FFT.
        The TD waveform will be zero-padded so it's Fourier transform has
        frequency bins of size P.deltaT.
        If P.deltaF == None, the TD waveform will be zero-padded
        to the next power of 2.
    """
    # For FD approximants, use the ChooseFDWaveform path = hoff_FD
    if lalsim.SimInspiralImplementedFDApproximants(P.approx)==1:
        # Raise exception if unused arguments were specified
        if fwdplan is not None:
            raise ValueError('FFT plan fwdplan given with FD approximant.\nFD approximants cannot use this.')
        hf = norm_hoff_FD(P, IP, Fp, Fc)

    # For TD approximants, do ChooseTDWaveform + FFT path = hoff_TD
    else:
        hf = norm_hoff_TD(P, IP, Fp, Fc, fwdplan)

    return hf

def norm_hoff_TD(P, IP, Fp=None, Fc=None, fwdplan=None):
    """
    Generate a waveform from ChooseWaveformParams P normalized according
    to inner product IP by creating a TD waveform, zero-padding and
    then Fourier transforming with FFTW3 forward FFT plan fwdplan.
    Returns a COMPLEX16FrequencySeries object.

    If P.deltaF==None, just pad up to next power of 2
    If P.deltaF = 1/X, will generate a TD waveform, zero-pad to length X seconds
        and then FFT. Will throw an error if waveform is longer than X seconds

    If you do not provide a forward FFT plan, one will be created.
    If you are calling this function many times, you may to create it
    once beforehand and pass it in, e.g.:
    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)

    You may pass in antenna patterns Fp, Fc. If none are provided, they will
    be computed from the information in ChooseWaveformParams.

    N.B. IP and the waveform generated from P must have the same deltaF and 
        the waveform must extend to at least the highest frequency of IP's PSD.
    """
    hf = hoff_TD(P, Fp, Fc, fwdplan)
    norm = IP.norm(hf)
    hf.data.data /= norm
    return hf

def norm_hoff_FD(P, IP, Fp=None, Fc=None):
    """
    Generate a FD waveform for a FD approximant normalized according to IP.
    Note that P.deltaF (which is None by default) must be set.
    IP and the waveform generated from P must have the same deltaF and 
        the waveform must extend to at least the highest frequency of IP's PSD.
    """
    if P.deltaF is None:
        raise ValueError('None given for freq. bin size P.deltaF')

    htilde = hoff_FD(P, Fp, Fc)
    norm = IP.norm(htilde)
    htilde.data.data /= norm
    return htilde


def hlmoft(P, Lmax=2, Fp=None, Fc=None):
    """
    Generate the TD h_lm -2-spin-weighted spherical harmonic modes of a GW
    with parameters P. Returns a SphHarmTimeSeries, a linked-list of modes with
    a COMPLEX16TimeSeries and indices l and m for each node.

    The linked list will contain all modes with l <= Lmax
    and all values of m for these l.
    """
    global rosDebugMessagesContainer
    assert Lmax >= 2
    hlms = lalsim.SimInspiralChooseTDModes(P.phiref, P.deltaT, P.m1, P.m2,
            P.fmin, P.fref, P.dist, P.lambda1, P.lambda2, P.waveFlags,
            P.nonGRparams, P.ampO, P.phaseO, Lmax, P.approx)
    # FIXME: Add ability to taper
    # COMMENT: Add ability to generate hlmoft at a nonzero GPS time directly.
    #      USUALLY we will use the hlms in template-generation mode, so will want the event at zero GPS time
    # for L in np.arange(2,Lmax+1):
    #     for m in np.arange(-Lmax, Lmax+1):
    #         hxx = lalsim.SphHarmTimeSeriesGetMode(hlms,int(L),int(m))  
    #         if rosDebugMessagesContainer[0]:
    #             print " hlm(t) epoch after shift  (l,m)=", L,m,":  = ", stringGPSNice( hxx.epoch)
    #         hxx.epoch = hxx.epoch + P.tref  # edit the actual pointer's data.  Critical to make sure the epoch is propagated in full into the template hlm's, so I know what index corresponds to the P.tref time!
    #         if rosDebugMessagesContainer[0]:
    #             print " hlm(t) epoch after shift  (l,m)=", L,m,":  = ", stringGPSNice( hxx.epoch)

    if P.deltaF is not None:
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        hxx = lalsim.SphHarmTimeSeriesGetMode(hlms, 2, 2)
        assert TDlen >= hxx.data.length
        hlms = lalsim.ResizeSphHarmTimeSeries(hlms, 0, TDlen)

    # Debugging: Confirm with complete certainty that the epochs of all the modes are consistently propagated
    if rosDebugMessagesContainer[0]:
        for L in np.arange(2,Lmax+1):
            for m in np.arange(-Lmax, Lmax+1):
                hxx = lalsim.SphHarmTimeSeriesGetMode(hlms,int(L),int(m))  
                print " hlm(t) epoch after resize, (l,m) ", L,m, stringGPSNice( hxx.epoch), " and buffer duration =   ", hxx.deltaT*len(hxx.data.data)

    return hlms

def hlmoff(P, Lmax=2, Fp=None, Fc=None):
    """
    Generate the FD h_lm -2-spin-weighted spherical harmonic modes of a GW
    with parameters P. Returns a SphHarmTimeSeries, a linked-list of modes with
    a COMPLEX16TimeSeries and indices l and m for each node.

    The linked list will contain all modes with l <= Lmax
    and all values of m for these l.
    """
    global rosDebugMessagesContainer

    hlms = hlmoft(P, Lmax, Fp, Fc)
    hxx = lalsim.SphHarmTimeSeriesGetMode(hlms, 2, 2)
    if P.deltaF == None: # h_lm(t) was not zero-padded, so do it now
        TDlen = nextPow2(hxx.data.length)
        hlms = lalsim.ResizeSphHarmTimeSeries(hlms, 0, TDlen)
    else: # Check zero-padding was done to expected length
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen == hxx.data.length

    if rosDebugMessagesContainer[0]:
        print "  ... starting Fourier transform: hlm(t)->hlm(f)"
    # FFT the hlms
    Hlms = lalsim.SphHarmFrequencySeriesFromSphHarmTimeSeries(hlms)
    if rosDebugMessagesContainer[0]:
        print "  ... finished Fourier transform: hlm(t)->hlm(f)"

    # Fixme
    if rosDebugMessagesContainer[0]:
        for L in np.arange(2,Lmax+1):
            for m in np.arange(-Lmax, Lmax+1):
                hxx = lalsim.SphHarmFrequencySeriesGetMode(Hlms,int(L),int(m))  
                print " hlm(f) epoch after FFT, (l,m) ", L,m,  stringGPSNice(hxx.epoch)

    return Hlms


def complex_hoft(P, sgn=-1):
    """
    Generate a complex TD waveform from ChooseWaveformParams P
    Returns h(t) = h+(t) + 1j sgn hx(t)
    where sgn = -1 (default) or 1

    Returns a COMPLEX16TimeSeries object
    """
    assert sgn == 1 or sgn == -1
    hp, hc = lalsim.SimInspiralChooseTDWaveform(P.phiref, P.deltaT, P.m1, P.m2, 
            P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin, P.fref, P.dist, 
            P.incl, P.lambda1, P.lambda2, P.waveFlags, P.nonGRparams,
            P.ampO, P.phaseO, P.approx)
    if P.taper != lalsim.LAL_SIM_INSPIRAL_TAPER_NONE: # Taper if requested
        lalsim.SimInspiralREAL8WaveTaper(hp.data, P.taper)
        lalsim.SimInspiralREAL8WaveTaper(hc.data, P.taper)
    if P.deltaF is not None:
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen >= hp.data.length
        hp = lal.ResizeREAL8TimeSeries(hp, 0, TDlen)
        hc = lal.ResizeREAL8TimeSeries(hc, 0, TDlen)

    ht = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hp.epoch, hp.f0, 
            hp.deltaT, lal.lalDimensionlessUnit, hp.data.length)
    ht.epoch = ht.epoch + P.tref
    for i in range(ht.data.length):
        ht.data.data[i] = hp.data.data[i] + 1j * sgn * hc.data.data[i]
    return ht

def complex_hoff(P, sgn=-1, fwdplan=None):
    """
    CURRENTLY ONLY WORKS WITH TD APPROXIMANTS

    Generate a (non-Hermitian) FD waveform from ChooseWaveformParams P
    by creating a complex TD waveform of the form

    h(t) = h+(t) + 1j sgn hx(t)    where sgn = -1 (default) or 1

    If P.deltaF==None, just pad up to next power of 2
    If P.deltaF = 1/X, will generate a TD waveform, zero-pad to length X seconds
        and then FFT. Will throw an error if waveform is longer than X seconds

    If you do not provide a forward FFT plan, one will be created.
    If you are calling this function many times, it is best to create it
    once beforehand and pass it in, e.g.:
    fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)

    Returns a COMPLEX16FrequencySeries object
    """
    ht = complex_hoft(P, sgn)

    if P.deltaF == None: # h(t) was not zero-padded, so do it now
        TDlen = nextPow2(ht.data.length)
        ht = lal.ResizeCOMPLEX16TimeSeries(ht, 0, TDlen)
    else: # Check zero-padding was done to expected length
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen == ht.data.length

    if fwdplan==None:
        fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)

    FDlen = TDlen/2+1
    hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            ht.epoch, ht.f0, 1./ht.deltaT/TDlen, lal.lalHertzUnit, 
            TDlen)
    lal.COMPLEX16TimeFreqFFT(hf, ht, fwdplan)
    return hf

def complex_norm_hoff(P, IP, sgn=-1, fwdplan=None):
    """
    CURRENTLY ONLY WORKS WITH TD APPROXIMANTS

    Generate a (non-Hermitian) FD waveform from ChooseWaveformParams P
    by creating a complex TD waveform of the form

    h(t) = h+(t) + 1j sgn hx(t)    where sgn = -1 (default) or 1

    If P.deltaF==None, just pad up to next power of 2
    If P.deltaF = 1/X, will generate a TD waveform, zero-pad to length X seconds
        and then FFT. Will throw an error if waveform is longer than X seconds

    If you do not provide a forward FFT plan, one will be created.
    If you are calling this function many times, it is best to create it
    once beforehand and pass it in, e.g.:
    fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)

    Returns a COMPLEX16FrequencySeries object
    """
    htilde = complex_hoff(P, sgn, fwdplan)
    norm = IP.norm(htilde)
    htilde.data.data /= norm
    return htilde

#
# Functions to read an ASCII file in NINJA data format (see arXiv:0709.0093)
# and return REAL8TimeSeries or COMPLEX16FrequencySeries objects containing
# the waveform, possibly after rescaling the time and/or TD strain
#

def NINJA_data_to_hoft(fname, TDlen=-1, scaleT=1., scaleH=1., Fp=1., Fc=0.):
    """
    Function to read in data in the NINJA format, i.e.
    t_i   h+(t_i)   hx(t_i)
    and convert it to a REAL8TimeSeries holding the observed
    h(t) = Fp*h+(t) + Fc*hx(t)

    If TDlen == -1 (default), do not zero-pad the returned waveforms 
    If TDlen == 0, zero-pad returned waveforms to the next power of 2
    If TDlen == N, zero-pad returned waveforms to length N

    scaleT and scaleH can be used to rescale the time steps and strain resp.
    e.g. to get a waveform appropriate for a total mass M you would scale by
    scaleT = G M / c^3
    scaleH = G M / (D c^2)
    """
    t, hpdat, hcdat = np.loadtxt(fname, unpack=True)
    tmplen = len(t)
    if TDlen == -1:
        TDlen = tmplen
    elif TDlen==0:
        TDlen = nextPow2(tmplen)
    else:
        assert TDlen >= tmplen

    tStart = t[0]
    deltaT = (t[1] - t[0]) * scaleT

    ht = lal.CreateREAL8TimeSeries("h(t)", lal.LIGOTimeGPS(tStart), 0.,
            deltaT, lal.lalDimensionlessUnit, TDlen)

    for i in range(tmplen):
        ht.data.data[i] = (Fp*hpdat[i] + Fc*hcdat[i]) * scaleH
    for i in range(tmplen,TDlen):
        ht.data.data[i] = 0.

    return ht

def NINJA_data_to_hp_hc(fname, TDlen=-1, scaleT=1., scaleH=1., deltaT=0):
    """
    Function to read in data in the NINJA format in file 'fname', i.e.
    t_i   h+(t_i)   hx(t_i)
    and convert it to two REAL8TimeSeries holding polarizations hp(t) and hc(t)

    If TDlen == -1 (default), do not zero-pad the returned waveforms
    If TDlen == 0, zero-pad returned waveforms to the next power of 2
    If TDlen == N, zero-pad returned waveforms to length N

    scaleT and scaleH can be used to rescale the time steps and strain resp.
    e.g. if the input file provides time steps in t/M
    and the strain has mass and distance scaled out, to get a waveform
    appropriate for a total mass M and distance D you would scale by
    scaleT = G M / c^3
    scaleH = G M / (D c^2)

    Once time is properly scaled into seconds, you can interpolate the waveform
    to a different sample rate deltaT.
    e.g. if the file has time steps of t/M = 1 and you use scaleT to rescale
    for a 100 Msun binary, the time steps will be ~= 0.00049 s.
    If you provide the argument deltaT = 1./4096. ~= 0.00024 s
    the waveform will be interpolated and resampled at 4096 Hz.

    If deltaT==0, then no interpolation will be done

    NOTE: For improved accuracy, we convert
    h+ + i hx --> A e^(i Phi),
    interpolate and resample A and Phi, then convert back to h+, hx
    """
    t, hpdat, hcdat = np.loadtxt(fname, unpack=True)
    tmplen = len(t)
    tStart = t[0] * scaleT
    deltaT = (t[1] - t[0]) * scaleT
    hpdat *= scaleH
    hcdat *= scaleH

    if deltaT==0: # No need to interpolate or resample
        if TDlen == -1:
            TDlen = tmplen
        elif TDlen==0:
            TDlen = nextPow2(tmplen)
        else:
            assert TDlen >= tmplen

        hp = lal.CreateREAL8TimeSeries("hplus(t)", lal.LIGOTimeGPS(tStart),
                0., deltaT, lal.lalDimensionlessUnit, TDlen)
        hc = lal.CreateREAL8TimeSeries("hcross(t)", lal.LIGOTimeGPS(tStart),
                0., deltaT, lal.lalDimensionlessUnit, TDlen)

        for i in range(tmplen):
            hp.data.data[i] = hpdat[i]
            hc.data.data[i] = hcdat[i]
        for i in range(tmplen,TDlen):
            hp.data.data[i] = 0.
            hc.data.data[i] = 0.
        return hp, hc

    else: # do interpolation and resample at rate deltaT
        assert deltaT > 0
        times = tStart + np.arange(tmplen) * deltaT
        newlen = np.floor( (tmplen-1) * deltaT / deltaT)
        newtimes = tStart + np.arange(newlen) * deltaT 
        newlen = len(newtimes)
        if TDlen == -1:
            TDlen = newlen
        elif TDlen==0:
            TDlen = 1
            while TDlen < newlen:
                TDlen *= 2
        else:
            assert TDlen >= newlen
        hp = lal.CreateREAL8TimeSeries("hplus(t)", lal.LIGOTimeGPS(tStart),
                0., deltaT, lal.lalDimensionlessUnit, TDlen)
        hc = lal.CreateREAL8TimeSeries("hcross(t)", lal.LIGOTimeGPS(tStart),
                0., deltaT, lal.lalDimensionlessUnit, TDlen)

        # build complex waveform, cubic spline interpolate amp and phase
        hcmplx = hpdat + 1j * hcdat
        amp = np.abs(hcmplx)
        phase = unwind_phase( np.angle(hcmplx) )
        ampintp = interpolate.InterpolatedUnivariateSpline(times, amp, k=3)
        phaseintp = interpolate.InterpolatedUnivariateSpline(times, phase, k=3)
        # Resample interpolated waveform, convert back to hp, hc
        hcmplxnew = ampintp(newtimes) * np.exp(1j * phaseintp(newtimes) )
        hpnew = np.real(hcmplxnew)
        hcnew = np.imag(hcmplxnew)
        for i in range(newlen):
            hp.data.data[i] = hpnew[i]
            hc.data.data[i] = hcnew[i]
        for i in range(newlen,TDlen):
            hp.data.data[i] = 0.
            hc.data.data[i] = 0.
        return hp, hc


def NINJA_data_to_hoff(fname, TDlen=0, scaleT=1., scaleH=1., Fp=1., Fc=0.):
    """
    Function to read in data in the NINJA format, i.e.
    t_i   h+(t_i)   hx(t_i)
    and convert it to a COMPLEX16FrequencySeries holding the observed
    h(f) = FFT[ Fp*h+(t) + Fc*hx(t) ]

    If TDlen == -1, do not zero-pad the TD waveform before FFTing
    If TDlen == 0 (default), zero-pad the TD waveform to the next power of 2
    If TDlen == N, zero-pad the TD waveform to length N before FFTing

    scaleT and scaleH can be used to rescale the time steps and strain resp.
    e.g. to get a waveform appropriate for a total mass M you would scale by
    scaleT = G M / c^3
    scaleH = G M / (D c^2)
    """
    t, hpdat, hcdat = np.loadtxt(fname, unpack=True)
    tmplen = len(t)
    if TDlen == -1:
        TDlen = tmplen
    elif TDlen==0:
        TDlen = nextPow2(tmplen)
    else:
        assert TDlen >= tmplen

    tStart = t[0]
    deltaT = (t[1] - t[0]) * scaleT

    ht = lal.CreateREAL8TimeSeries("h(t)", lal.LIGOTimeGPS(tStart), 0.,
            deltaT, lal.lalDimensionlessUnit, TDlen)

    for i in range(tmplen):
        ht.data.data[i] = (Fp*hpdat[i] + Fc*hcdat[i]) * scaleH
    for i in range(tmplen,TDlen):
        ht.data.data[i] = 0.

    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)
    hf = lal.CreateCOMPLEX16FrequencySeries("h(f)", 
            ht.epoch, ht.f0, 1./deltaT/TDlen, lal.lalHertzUnit, 
            TDlen/2+1)
    lal.REAL8TimeFreqFFT(hf, ht, fwdplan)
    return hf

def NINJA_data_to_norm_hoff(fname, IP, TDlen=0, scaleT=1., scaleH=1.,
        Fp=1., Fc=0.):
    """
    Function to read in data in the NINJA format, i.e.
    t_i   h+(t_i)   hx(t_i)
    and convert it to a COMPLEX16FrequencySeries holding
    h(f) = FFT[ Fp*h+(t) + Fc*hx(t) ]
    normalized so (h|h)=1 for inner product IP

    If TDlen == -1, do not zero-pad the TD waveform before FFTing
    If TDlen == 0 (default), zero-pad the TD waveform to the next power of 2
    If TDlen == N, zero-pad the TD waveform to length N before FFTing

    scaleT and scaleH can be used to rescale the time steps and strain resp.
    e.g. to get a waveform appropriate for a total mass M you would scale by
    scaleT = G M / c^3
    scaleH = G M / (D c^2)
    """
    t, hpdat, hcdat = np.loadtxt(fname, unpack=True)
    tmplen = len(t)
    if TDlen == -1:
        TDlen = tmplen
    elif TDlen==0:
        TDlen = nextPow2(tmplen)
    else:
        assert TDlen >= tmplen

    tStart = t[0]
    deltaT = (t[1] - t[0]) * scaleT

    ht = lal.CreateREAL8TimeSeries("h(t)", lal.LIGOTimeGPS(tStart), 0.,
            deltaT, lal.lalDimensionlessUnit, TDlen)

    for i in range(tmplen):
        ht.data.data[i] = (Fp*hpdat[i] + Fc*hcdat[i]) * scaleH
    for i in range(tmplen,TDlen):
        ht.data.data[i] = 0.

    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)
    hf = lal.CreateCOMPLEX16FrequencySeries("h(f)", 
            ht.epoch, ht.f0, 1./deltaT/TDlen, lal.lalHertzUnit, 
            TDlen/2+1)
    lal.REAL8TimeFreqFFT(hf, ht, fwdplan)
    norm = IP.norm(hf)
    hf.data.data /= norm
    return hf

def frame_data_to_hoft(fname, channel, start=None, stop=None):
    """
    Function to read in data in the frame format and convert it to 
    a REAL8TimeSeries. fname is the path to a LIGO cache file.
    """
    global rosDebugMessagesContainer

    if rosDebugMessagesContainer[0]:
        print " ++ Loading from cache ", fname, channel
    with open(fname) as cfile:
        cachef = Cache.fromfile(cfile)
    for i in range(len(cachef))[::-1]:
        # FIXME: HACKHACKHACK
        if cachef[i].observatory != channel[0]:
            del cachef[i]
    if rosDebugMessagesContainer[0]:
        print cachef.to_segmentlistdict()
    fcache = frutils.FrameCache(cachef)
    # FIXME: Horrible, horrible hack -- will only work if all requested channels
    # span the cache *exactly*
    if start is None:
        start = cachef.to_segmentlistdict()[channel[0]][0][0]
    if stop is None:
        stop = cachef.to_segmentlistdict()[channel[0]][-1][-1]
    
    ht = fcache.fetch(channel, start, stop)
    tmp = lal.CreateREAL8TimeSeries("h(t)", 
            lal.LIGOTimeGPS(float(ht.metadata.segments[0][0])),
            0., ht.metadata.dt, lal.lalDimensionlessUnit, len(ht))
    print   "  ++ Frame data sampling rate ", 1./tmp.deltaT, " and epoch ", stringGPSNice(tmp.epoch)
    tmp.data.data[:] = ht
    return tmp

def frame_data_to_hoff(fname, channel, start=None, stop=None, TDlen=0):
    """
    Function to read in data in the frame format
    and convert it to a COMPLEX16FrequencySeries holding
    h(f) = FFT[ h(t) ]

    If TDlen == -1, do not zero-pad the TD waveform before FFTing
    If TDlen == 0 (default), zero-pad the TD waveform to the next power of 2
    If TDlen == N, zero-pad the TD waveform to length N before FFTing
    """
    ht = frame_data_to_hoft(fname, channel, start, stop)

    tmplen = ht.data.length
    if TDlen == -1:
        TDlen = tmplen
    elif TDlen==0:
        TDlen = nextPow2(tmplen)
    else:
        assert TDlen >= tmplen

    ht = lal.ResizeREAL8TimeSeries(ht, 0, TDlen)
    for i in range(tmplen,TDlen):
        ht.data.data[i] = 0.

    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)
    hf = lal.CreateCOMPLEX16FrequencySeries("h(f)", 
            ht.epoch, ht.f0, 1./deltaT/TDlen, lal.lalHertzUnit, 
            TDlen/2+1)
    lal.REAL8TimeFreqFFT(hf, ht, fwdplan)
    return hf


def frame_data_to_non_herm_hoff(fname, channel, start=None, stop=None, TDlen=0):
    """
    Function to read in data in the frame format
    and convert it to a COMPLEX16FrequencySeries 
    h(f) = FFT[ h(t) ]
    Create complex FD data that does not assume Hermitianity - i.e.
    contains positive and negative freq. content

    If TDlen == -1, do not zero-pad the TD waveform before FFTing
    If TDlen == 0 (default), zero-pad the TD waveform to the next power of 2
    If TDlen == N, zero-pad the TD waveform to length N before FFTing
    """
    hoft = frame_data_to_hoft(fname, channel, start, stop)

    tmplen = hoft.data.length
    if TDlen == -1:
        TDlen = tmplen
    elif TDlen==0:
        TDlen = nextPow2(tmplen)
    else:
        assert TDlen >= tmplen

    hoft = lal.ResizeREAL8TimeSeries(hoft, 0, TDlen)
    hoftC = lal.CreateCOMPLEX16TimeSeries("hoft", hoft.epoch, hoft.f0,
            hoft.deltaT, hoft.sampleUnits, TDlen)
    # copy h(t) into a COMPLEX16 array which happens to be purely real
    hoftC.data.data = hoft.data.data + 0j
#    for i in range(TDlen):
#        hoftC.data.data[i] = hoft.data.data[i]
    FDlen = TDlen
    fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)
    hoff = lal.CreateCOMPLEX16FrequencySeries("Template h(f)",
            hoft.epoch, hoft.f0, 1./hoft.deltaT/TDlen, lal.lalHertzUnit,
            FDlen)
    lal.COMPLEX16TimeFreqFFT(hoff, hoftC, fwdplan)
    if rosDebugMessagesContainer[0]:
        print " ++ Loaded data h(f) of length n= ", len(hoff.data.data), " (= ", len(hoff.data.data)*hoft.deltaT, "s) at sampling rate ", 1./hoft.deltaT    
    return hoff


def stringGPSNice(tgps):
    return str(tgps.gpsSeconds)+'.'+str(tgps.gpsNanoSeconds)

def constructLMIterator(Lmax):  # returns a list of (l,m) pairs covering all modes, as a list.  Useful for building iterators without nested lists
    mylist = []
    for L in np.arange(2, Lmax+1):
        for m in np.arange(-L, L+1):
            mylist.append((L,m))
    return mylist

def extend_psd_series_to_sampling_requirements(raw_psd, dfRequired, fNyqRequired):
    """
    extend_psd_series_to_sampling_requirements: 
    Takes a conventional 1-sided PSD and extends into a longer 1-sided PSD array by filling intervening samples, also strips the pylal binding and goes directly to a numpy array.
    The raw psd object is a pylal wrapper of the timeseries, which is different from the swig bindings and which is *also* different than the raw numpy array assumed in lalsimutils.py (above)
    """
    # Allocate new series
    n = len(raw_psd.data)                                     # odd number for one-sided PSD
    nRequired = int(fNyqRequired/dfRequired)+1     # odd number for one-sided PSD
    facStretch = int((nRequired-1)/(n-1))  # n-1 should be power of 2
    if rosDebugMessagesContainer[0]:
        print " extending psd of length ", n, " to ", nRequired, " elements requires a factor of ", facStretch
    #    psdNew = lal.CreateREAL8FrequencySeries("PSD", lal.LIGOTimeGPS(0.), 0., dfRequired,lal.lalHertzUnit, nRequired )
    #   psdNew.data.data = np.zeros(len(psdNew.data.data))
    # psdNew = np.zeros(nRequired)   
    # Populate the series.  Slow because it is a python loop
    # for i in np.arange(n):
    #     for j in np.arange(facStretch):
    #         psdNew[facStretch*i+j] = raw_psd.data[i]  # 
    psdNew = (np.array([raw_psd.data for j in np.arange(facStretch)])).transpose().flatten()  # a bit too large, but that's fine for our purposes
    return psdNew

def get_psd_series_from_xmldoc(fname, inst):
    return read_psd_xmldoc(utils.load_filename(fname))[inst]  # return value is pylal wrapping of the data type; index data by a.data[k]

def get_intp_psd_series_from_xmldoc(fname, inst):
    psd = get_psd_series_from_xmldoc(fname, inst)
    f = np.arange(psd.f0, psd.deltaF*len(psd.data), psd.deltaF)
    ifunc = interpolate.interp1d(f, psd.data)
    def intp_psd(freq):
        return float("inf") if freq > psd.deltaF*len(psd.data) else ifunc(freq)
    return np.vectorize(intp_psd)
