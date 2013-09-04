# Copyright (C) 2012  Evan Ochsner  (additions by Richard O'Shaughnessy)
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
from numbers import Number # type checking
import sys
import lal
import lalsimulation as lalsim
import lalinspiral
import lalmetaio
from glue.ligolw import lsctables, table, utils # check all are needed
import numpy as np
import pylab as plt
import copy
from numpy import sin, cos
from scipy import interpolate

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>"

# Bitmask to set debug level to report errors, warnings and info
# 1 = Errors, 2 = Warnings, 4 = Info, so 1 + 2 + 4 = everything
#lal.cvar.lalDebugLevel = 3
#print "lalDebuglevel has been set to:", lal.cvar.lalDebugLevel

# IMPROVEMENTS NEEDED
#   - Better wrapping of arrays
#   - Sane use of TDlen parameter : easy to get lost with which parameter is setting the length of which array.
#     at what time.


#
# Classes to hold arguments of ChooseWaveform functions
#


thePrefix = " --"

def vecCross(v1,v2):
    return [v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]]

def vecDot(v1,v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
       

# http://code.activestate.com/recipes/410692/
# This class provides the functionality we want. You only need to look at
# this if you want to know how this works. It only needs to be defined
# once, no need to muck around with its internals.
class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

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
    def __init__(self, phiref=0., deltaT=1./4096., m1=1.*lal.LAL_MSUN_SI, 
            m2=1.*lal.LAL_MSUN_SI, s1x=0., s1y=0., s1z=0., 
            s2x=0., s2y=0., s2z=0., fmin=40., fref=0., dist=1.e6*lal.LAL_PC_SI,
            incl=0., lambda1=0., lambda2=0., waveFlags=None, nonGRparams=None,
            ampO=0, phaseO=7, approx=lalsim.TaylorT4, 
            theta=0., phi=0., psi=0., tref=0., radec=False, detector="H1",
            deltaF=None, fMax=0., # for use w/ FD approximants
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
        self.theta = theta
        self.phi = phi
        self.psi = psi
        self.tref = tref
        self.radec = radec
        self.detector = "H1"
        self.deltaF=deltaF
        self.fMax=fMax
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
        print thePrefix, "m1 =", self.m1 / lal.LAL_MSUN_SI, "(Msun)"
        print thePrefix, "m2 =", self.m2 / lal.LAL_MSUN_SI, "(Msun)"
        print thePrefix, " : mchirp =", mchirp(self.m1,self.m2)/lal.LAL_MSUN_SI, "(Msun)"
        print thePrefix, " : eta =", symRatio(self.m1,self.m2), "(1)"
        print thePrefix,  "s1x =", self.s1x
        print thePrefix, "s1y =", self.s1y
        print thePrefix, "s1z =", self.s1z
        print thePrefix, "s2x =", self.s2x
        print thePrefix, "s2y =", self.s2y
        print thePrefix,  "s2z =", self.s2z
        print thePrefix,  " : Vector spin products"
        print thePrefix,  " : s1.s2 = ",  vecDot([self.s1x,self.s1y,self.s1z],[self.s2x,self.s2y,self.s2z])
        print thePrefix,  " : L. s1 x s2 =  ",  vecDot( [np.sin(self.incl),0,np.cos(self.incl)] , vecCross([self.s1x,self.s1y,self.s1z],[self.s2x,self.s2y,self.s2z]))
        print thePrefix, "lambda1 =", self.lambda1
        print thePrefix, "lambda2 =", self.lambda2
        print thePrefix, "inclination =", self.incl
        print thePrefix, "distance =", self.dist / 1.e+6 / lal.LAL_PC_SI, "(Mpc)"
        print thePrefix, "reference orbital phase =", self.phiref
        print thePrefix,  "time of coalescence =", self.tref
        print thePrefix, "detector is:", self.detector
        if self.radec==False:
            print thePrefix, "Sky position relative to overhead detector is:"
            print thePrefix,  "zenith angle =", self.theta, "(radians)"
            print thePrefix, "azimuth angle =", self.phi, "(radians)"
        if self.radec==True:
            print thePrefix, "Sky position relative to geocenter is:"
            print thePrefix, "declination =", self.theta, "(radians)"
            print thePrefix,  "right ascension =", self.phi, "(radians)"
        print thePrefix, "polarization angle =", self.psi
        print  thePrefix, "starting frequency is =", self.fmin
        print  thePrefix, "reference frequency is =", self.fref
        print thePrefix,  "Max frequency is =", self.fMax
        print thePrefix, "time step =", self.deltaT, "(s) <==>", 1./self.deltaT,\
                "(Hz) sample rate"
        print  thePrefix, "freq. bin size is =", self.deltaF, "(Hz)"
        print thePrefix, "approximant is =", lalsim.GetStringFromApproximant(self.approx)
        print thePrefix,  "phase order =", self.phaseO
        print thePrefix, "amplitude order =", self.ampO
        print thePrefix, "waveFlags struct is", self.waveFlags
        print thePrefix, " :  Spin order " , lalsim.SimInspiralGetSpinOrder(self.waveFlags)
        print thePrefix, " :  Tidal order " , lalsim.SimInspiralGetTidalOrder(self.waveFlags)
        print thePrefix, "nonGRparams struct is", self.nonGRparams
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
    def VelocityAtFrequency(self,f):  # in units of c
        m1 = self.m1* lal.LAL_G_SI / lal.LAL_C_SI**3
        m2 = self.m2*lal.LAL_G_SI / lal.LAL_C_SI**3
        return ( (m1+m2) * lal.LAL_PI * f)**(1./3.)
    def OrbitalAngularMomentumAtReference(self):   # in units of kg in SI
        v = self.VelocityAtFrequency(max(self.fref,self.fmin));
        Lhat = np.array( [np.sin(self.incl),0,np.cos(self.incl)])  # does NOT correct for psi polar angle!
        M = (self.m1+self.m2)
        eta = symRatio(self.m1,self.m2)   # dimensionless
        return Lhat*M*M*eta/v     # in units of kg in SI
    def OrbitalAngularMomentumAtReferenceOverM2(self):
        L = self.OrbitalAngularMomentumAtReference()
        return L/(self.m1+self.m2)/(self.m1+self.m2)
    def TotalAngularMomentumAtReference(self):    # does NOT correct for psi polar angle, per convention
        L = self.OrbitalAngularMomentumAtReference()
        S1 = self.m1*self.m1 * np.array([self.s1x,self.s1y, self.s1z])
        S2 = self.m2*self.m2 * np.array([self.s2x,self.s2y, self.s2z])
        return L+S1+S2
    def TotalAngularMomentumAtReferenceOverM2(self):
        J = self.TotalAngularMomentumAtReference()
        return J/(self.m1+self.m2)/(self.m1+self.m2)

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
        nonGRparams=None, detector="H1", deltaF=None, fMax=0.):
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
            detector=detector, deltaF=deltaF, fMax=fMax) for i in rng]
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
        self.psd = psd
        self.minIdx = int(fLow/deltaF)
        self.FDlen = int(fNyq/deltaF)+1
        self.weights = np.zeros(self.FDlen)
        self.analyticPSD_Q = analyticPSD_Q
        if analyticPSD_Q == True:
            for i in range(self.minIdx,self.FDlen):
                self.weights[i] = 1./self.psd(i*deltaF)
        else:
            for i in range(self.minIdx,self.FDlen):
                if psd[i] != 0.:
                    self.weights[i] = 1./psd[i]   # Pass an array, not a function.

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
        for i in range(self.minIdx,maxIdx):
            val += h1.data.data[i] * h2.data.data[i].conj() * self.weights[i]
        val = 4. * self.deltaF * np.real(val)
        return val

    def norm(self, h):
        """
        Compute norm of a COMPLEX16Frequency Series
        """
        assert h.data.length <= self.FDlen
        assert abs(h.deltaF-self.deltaF) <= 1.e-6
        val = 0.
        for i in range(self.minIdx,h.data.length):
            val += h.data.data[i] * h.data.data[i].conj() * self.weights[i]
        val = np.sqrt( 4. * self.deltaF * np.abs(val) )
        return val

class ComplexIP(InnerProduct):
    """
    Complex-valued inner product. self.ip(h1,h2) computes

          fNyq
    4 \int      h1(f) h2*(f) / Sn(f) df
          fLow

    And similarly for self.norm(h1)

    DOES NOT maximize over time or phase
    NOT YET FIXED TO WORK WITH COMPLEX PACKING - still mangled!
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
        for i in range(self.minIdx,maxIdx):
            val += h1.data.data[i] * h2.data.data[i].conj() * self.weights[i]
        val *= 4. * self.deltaF
        return val

    def norm(self, h):
        """
        Compute norm of a COMPLEX16Frequency Series
        """
        assert h.data.length <= 2*self.FDlen
        assert abs(h.deltaF-self.deltaF) <= 1.e-5
        val = 0.
        for i in range(self.minIdx,self.FDlen/2):
            val += 2*(h.data.data[self.FDlen - i]*h.data.data[self.FDlen -i].conj()\
                    + h.data.data[self.FDlen+i]*h.data.data[self.FDlen+i].conj())\
                    *self.weights[i]
        val = np.sqrt( 4. * self.deltaF * np.abs(val) )
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
        The entire COMPLEX time series of overlaps for each possible time shift
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
            for i in range(self.minIdx,self.FDlen):
                self.weights[i] = 1./self.psd(i*deltaF)
        else:
            for i in range(self.minIdx,self.FDlen):
                if psd[i] != 0.:
                    self.weights[i] = 1./psd[i]

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
            self.intgd.data.data[i] = 4.*h1.data.data[i]*h2.data.data[i].conj()\
                    *self.weights[i]
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
            return rho, self.ovlp.data.data, rhoIdx, rhoPhase

    def norm(self, h):
        """
        Compute norm of a COMPLEX16Frequency Series
        """
#        print "h(deltaF) = ", h.deltaF, " IP(deltaF) = ", self.deltaF
        assert h.data.length <= self.FDlen
        assert abs(h.deltaF-self.deltaF) <= 1.e-5
        val = 0.
        for i in range(self.minIdx,h.data.length):
            val += h.data.data[i] * h.data.data[i].conj() * self.weights[i]
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
        The entire time series of overlaps for each possible time shift
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
        else:
            for i in range(self.minIdx,self.FDlen):
                if psd[i] != 0.:
                    self.weights[i] = 1./psd[i]

    def ip(self, h1, h2):
        """
        Compute inner product between two COMPLEX16Frequency Series.
        Remember LAL's packing for frequency-domain FFTs
        """
        assert h1.data.length==h2.data.length==self.wvlen
        # Tabulate the SNR integrand
        for i in range(self.wvlen):
            self.intgd.data.data[i] = 0.
        # Beware the complex packing
        for i in range(self.minIdx,self.wvlen/2):
            self.intgd.data.data[i] = 2*(h1.data.data[self.wvlen/2 - i]*h2.data.data[self.wvlen/2 -i].conj()\
                    + h1.data.data[self.wvlen/2+i]*h2.data.data[self.wvlen/2+i].conj())\
                    *self.weights[i]
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
            return rho, self.ovlp.data.data, rhoIdx, rhoPhase

    def norm(self, h):
        """
        Compute inner product between two COMPLEX16Frequency Series
        """
        assert h.data.length==self.wvlen
        assert abs(h.deltaF-self.deltaF) <= 1.e-5
        val = 0.
        # Beware the complex packing
        for i in range(self.minIdx,self.wvlen/2):
            val += (h.data.data[self.wvlen/2 - i] * h.data.data[self.wvlen/2 -i].conj() +\
                    h.data.data[self.wvlen/2+i]\
                    * h.data.data[self.wvlen/2+i].conj()) * self.weights[i]
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
    return 0.5*(1. + cos(theta)*cos(theta))*cos(2.*phi)*cos(2.*psi) - cos(theta)*sin(2.*phi)*sin(2.*psi)

def Fcross(theta, phi, psi):
    """
    Antenna pattern as a function of polar coordinates measured from
    directly overhead a right angle interferometer and polarization angle
    """
    return 0.5*(1. + cos(theta)*cos(theta))*cos(2.*phi)*sin(2.*psi) + cos(theta)*sin(2.*phi)*cos(2.*psi)

#
# Mass parameter conversion functions - note they assume m1 >= m2
#
def mass1(Mc, eta):
    """Compute larger component mass from Mc, eta"""
#    if (Mc< 0) or (eta > 1/4) or (eta < 0):
#       return 0
#    else:
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
            cnt -= 1
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

def findFEnd(P):
    """
    Generate the TD waveform, then find the termination frequency.
    should modify: use MODES if available, as that will be more reliable.
    """
    P1 = P.copy()
    P1.deltaT = 1./(2*16384)   # insure sampling rate is high, so I can terminate at a high frequency. Warning: termination frequency depends on sampling rate!  Be very careful
    P1.fmin=80       # unlikely I will care if I start that early
    P1.ampO=0       # really want the structure without higher harmonics.
    hc =complex_hoft(P1)
    ph = unwind_phase(np.angle(hc.data.data))
    return (ph[-3]-ph[-2])/P1.deltaT/(2*np.pi)

def findPhiOrbRefViaWaveform(P,fForRef):   # assumes 22 mode.  WARNING: complex_hoft includes psi!
    """
    Generate the TD waveform, then find phi_ref.  
    I assume you know what you're doing (i.e., only use this for a 22-mode dominated signal)
    (Alternative implementation: I *should* implement this by solving the orbital ODEs, to get something)
    [should modify: use MODES if available, as that will be more reliable.]
    """
    P1 = P.copy()
    P1.deltaT = 1./(4096)   # insure sampling rate is 'good enough'
    hc =complex_hoft(P1)
    # Find the index associated with a specific orbital frequency -- by doing the t(f) construction explicitly
    # NOTE THE NEGATIVE FACTOR USED (so frequencies and phases are positive) WHICH MUST BE UNDONE
    ph = (-1)*unwind_phase(np.angle(hc.data.data))  # phase array
    freq = np.zeros(len(ph))
    for i in np.arange(len(ph)-1):
        freq[i]= np.abs((ph[i+1]-ph[i])/(2*np.pi)/hc.deltaT)   # insure positive frequencies. (assume monotonic, in fact)
    freq[-1]=freq[-2]+1   # insure final frequency is a little higher
#    plt.plot(freq,np.array(ph)/2,'r-',label='waveform')
    fnPhase = interpolate.interp1d(freq,(-1)*ph,kind='quadratic')   # undo the sign flip, so i interpoalte true phase
    # Remember h22 ~ exp ( i (2 \phi - 2\psi - 2 \phi_{orb}(t)))
    return np.mod( - (  fnPhase(fForRef) + 2*P.psi)/2, 2 *np.pi)

def findPhiOrbRefViaOrbit(Praw,fForRef):  # Finds the orbital phase at the reference frequency
    P = Praw.copy()
    P.deltaT = 1./(4096)
    for case in switch(P.approx):
        if (case(lalsim.SpinTaylorT4)):
            LN0  = np.array( [np.sin(P.incl),0,np.cos(P.incl)])
            E10 = np.array( [np.cos(P.incl),0,np.sin(P.incl)])
            V, phi,S1x,S1y,S1z,S2x,S2y,S2z,LNx,LNy,LNz,E1x,E1y,E1z = lalsim.SimInspiralSpinTaylorPNEvolveOrbit(
                P.deltaT, P.m1, P.m2, P.fmin, P.fref, 
                P.s1x, P.s1y,P.s1z,
                P.s2x,P.s2y,P.s2z,
                LN0[0],LN0[1],LN0[2],
                E10[0],E10[1],E10[2],
                P.lambda1,P.lambda2,
                1,1,   # hardcoded
                lalsim.SimInspiralGetSpinOrder(P.waveFlags),
                lalsim.SimInspiralGetTidalOrder(P.waveFlags),
                P.phaseO,
                P.approx
                )
            break
        if (case(lalsim.TaylorT4)):
            V, phi = lalsim.SimInspiralTaylorT4PNEvolveOrbit(P.phiref, P.deltaT, P.m1, P.m2, P.fmin, P.fref, P.lambda1, P.lambda2, 0, P.phaseO)
            break
        if (case(lalsim.TaylorT2)):
            V, phi = lalsim.SimInspiralTaylorT2PNEvolveOrbit(P.phiref, P.deltaT, P.m1, P.m2, P.fmin, P.fref, P.lambda1, P.lambda2, 0, P.phaseO) 
            break
        if (case(lalsim.TaylorT1)):
            V, phi = lalsim.SimInspiralTaylorT1PNEvolveOrbit(P.phiref, P.deltaT, P.m1, P.m2, P.fmin, P.fref, P.lambda1, P.lambda2, 0, P.phaseO) 
            break
        # default, insure something is done to get a zeroth-order estimate
        print " FAILURE : using default behavior to create orbit "
        V, phi = lalsim.SimInspiralTaylorT4PNEvolveOrbit(P.phiref, P.deltaT, P.m1, P.m2, P.fmin, P.fref, P.lambda1, P.lambda2, 0, P.phaseO)
        
    # WARNING:
    #  - P.phiref is NOT provided in the evolve orbit code for SpinTaylor...I need to regenerate it
    ph=phi.data.data
    freq = np.zeros(len(ph))
    for i in np.arange(len(ph)-1):
        freq[i]= np.abs((ph[i+1]-ph[i])/(2*np.pi)/phi.deltaT)   # insure positive frequencies. (assume monotonic, in fact). Not centered!
    freq[-1]=freq[-2]+1   # insure final frequency is a little higher
    # plt.plot(2*freq,np.array(ph),'b-',label='orbit')
    # plt.legend(loc='upper left')
    # plt.xlabel('f_{22}=f_{orb}/2')
    # plt.ylabel('phase $\phi_{orb}$')
    # plt.show()
    fnPhaseOrb = interpolate.interp1d(2*freq,ph) # scale orbital to GW frequency in frequency array
#        return np.mod(fnPhaseOrb(fForRef), 2 *np.pi)   # Beware: the GW phase has a gauge term that shifts the origin/relationship. Not safe.
    if (P.fref==0):
        return np.mod(fnPhaseOrb(fForRef) - ph[-1]+P.phiref, 2 *np.pi)   # Beware: the GW phase has a gauge term that shifts the origin/relationship. Not safe.
    else:
        return np.mod(fnPhaseOrb(fForRef) - fnPhaseOrb(P.fref)+P.phiref, 2 *np.pi)   # Beware: the GW phase has a gauge term that shifts the origin/relationship

def findFEndViaOrbit(Praw):  # Finds the termination frequency 
    P = Praw.copy()
    P.fmin = 90
    P.fref = 100
    P.deltaT = 1./(2*16384)   # insure sampling rate is high, so I can terminate at a high frequency. Warning: termination frequency depends on sampling rate!  Be very careful
    for case in switch(P.approx):
        if (case(lalsim.SpinTaylorT4)):
            LN0  = np.array( [np.sin(P.incl),0,np.cos(P.incl)])
            E10 = np.array( [np.cos(P.incl),0,np.sin(P.incl)])
            V, phi,S1x,S1y,S1z,S2x,S2y,S2z,LNx,LNy,LNz,E1x,E1y,E1z = lalsim.SimInspiralSpinTaylorPNEvolveOrbit(
                P.deltaT, P.m1, P.m2, P.fmin, P.fref, 
                P.s1x, P.s1y,P.s1z,
                P.s2x,P.s2y,P.s2z,
                LN0[0],LN0[1],LN0[2],
                E10[0],E10[1],E10[2],
                P.lambda1,P.lambda2,
                1,1,   # hardcoded
                lalsim.SimInspiralGetSpinOrder(P.waveFlags),
                lalsim.SimInspiralGetTidalOrder(P.waveFlags),
                P.phaseO,
                P.approx
                )
            break
        if (case(lalsim.TaylorT4)):
            V, phi = lalsim.SimInspiralTaylorT4PNEvolveOrbit(P.phiref, P.deltaT, P.m1, P.m2, P.fmin, P.fref, P.lambda1, P.lambda2, 0, P.phaseO)
            break
        if (case(lalsim.TaylorT2)):
            V, phi = lalsim.SimInspiralTaylorT2PNEvolveOrbit(P.phiref, P.deltaT, P.m1, P.m2, P.fmin, P.fref, P.lambda1, P.lambda2, 0, P.phaseO) 
            break
        if (case(lalsim.TaylorT1)):
            V, phi = lalsim.SimInspiralTaylorT1PNEvolveOrbit(P.phiref, P.deltaT, P.m1, P.m2, P.fmin, P.fref, P.lambda1, P.lambda2, 0, P.phaseO) 
            break
        # default, insure something is done to get a zeroth-order estimate
        print " FAILURE : using default behavior to create orbit "
        V, phi = lalsim.SimInspiralTaylorT4PNEvolveOrbit(P.phiref, P.deltaT, P.m1, P.m2, P.fmin, P.fref, P.lambda1, P.lambda2, 0, P.phaseO)
        
    ph=phi.data.data
    freq = np.zeros(len(ph))
    for i in np.arange(len(ph)-1):
        freq[i]= np.abs((ph[i+1]-ph[i])/(2*np.pi)/phi.deltaT)   # insure positive frequencies. (assume monotonic, in fact)
    freq[-1]=freq[-2]+1   # insure final frequency is a little higher
    # plt.plot(2*freq,np.array(ph),'b-',label='orbit')
    # plt.legend(loc='upper left')
    # plt.xlabel('f_{22}=f_{orb}/2')
    # plt.ylabel('phase $\phi_{orb}$')
    # plt.show()
    return 2*freq[-2]  # twice orbital frequency


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
        hoft = hp
    elif P.radec==False:
        fp = Fplus(P.theta, P.phi, P.psi)
        fc = Fcross(P.theta, P.phi, P.psi)
        hp.data.data *= fp
        hc.data.data *= fc
        hp = lal.AddREAL8TimeSeries(hp, hc)
        hoft = hp
    else:
        hoft = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc, 
                P.phi, P.theta, P.psi, 
                lalsim.InstrumentNameToLALDetector(P.detector))
    hoft.epoch = hoft.epoch + P.tref
    if P.taper != lalsim.LAL_SIM_INSPIRAL_TAPER_NONE: # Taper if requested
        lalsim.SimInspiralREAL8WaveTaper(hoft.data, P.taper)
    return hoft

def hoff(P, TDlen=0, fwdplan=None, Fp=None, Fc=None):
    """
    Generate a FD waveform from ChooseWaveformParams P.
    Will return a COMPLEX16FrequencySeries object.

    If P.approx is a FD approximant, hoff_FD is called.
    This path calls SimInspiralChooseFDWaveform
        If TDlen != 0, freq. bin size will be set from TDlen and P.deltaT
            and output FrequencySeries will be zero-padded to length TDlen/2+1
        If TDlen == 0, freq. bin size will be P.deltaF, output not zero-padded 
        fwdplan must be None for FD approximants.

    If P.approx is a TD approximant, hoff_TD is called.
    This path calls ChooseTDWaveform and performs an FFT.
    """
    # For FD approximants, use the ChooseFDWaveform path = hoff_FD
    if lalsim.SimInspiralImplementedFDApproximants(P.approx)==1:
        # Raise exception if unused arguments were specified
        if fwdplan is not None:
            raise ValueError('FFT plan fwdplan given with FD approximant.\nFD approximants cannot use this.')
        if TDlen!=0: # Set values of P.deltaF from TDlen, P.deltaT
            P.deltaF = 1./P.deltaT/TDlen
        hf = hoff_FD(P, TDlen, Fp, Fc)

    # For TD approximants, do ChooseTDWaveform + FFT path = hoff_TD
    else:
        hf = hoff_TD(P, TDlen, fwdplan, Fp, Fc)

    return hf

def hoff_TD(P, TDlen=0, fwdplan=None, Fp=None, Fc=None):
    """
    Generate a FD waveform from ChooseWaveformParams P
    by creating a TD waveform padded to TDlen samples,
    then Fourier transform with FFTW3 forward FFT plan fwdplan

    If TDlen==0, just pad up to next power of 2

    If you do not provide a forward FFT plan, one will be created.
    If you are calling this function many times, it is best to create it
    once beforehand and pass it in, e.g.:
    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)

    You may pass in antenna patterns Fp, Fc. If none are provided, they will
    be computed from the information in ChooseWaveformParams

    Returns a COMPLEX16FrequencySeries object
    """
    # hp, hc = lalsim.SimInspiralChooseTDWaveform(P.phiref, P.deltaT, P.m1, P.m2, 
    #         P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin, P.fref, P.dist, 
    #         P.incl, P.lambda1, P.lambda2, P.waveFlags, P.nonGRparams,
    #         P.ampO, P.phaseO, P.approx)

    # if Fp is not None and Fc is not None:
    #     hp.data.data *= Fp
    #     hc.data.data *= Fc
    #     hp = lal.AddREAL8TimeSeries(hp, hc)
    #     hoft = hp
    # elif P.radec==False:
    #     fp = Fplus(P.theta, P.phi, P.psi)
    #     fc = Fcross(P.theta, P.phi, P.psi)
    #     hp.data.data *= fp
    #     hc.data.data *= fc
    #     hp = lal.AddREAL8TimeSeries(hp, hc)
    #     hoft = hp
    # else:
    #     hoft = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc, 
    #             P.theta, P.phi, P.psi, 
    #             lalsim.InstrumentNameToLALDetector(P.detector))
    # hoft.epoch = hoft.epoch + P.tref
    # if P.taper != lalsim.LAL_SIM_INSPIRAL_TAPER_NONE: # Taper if requested
    #     lalsim.SimInspiralREAL8WaveTaper(hoft.data, P.taper)
    assert (isinstance(TDlen , Number))   # type checking.  Bad argument passing by accident does happen
    ht = hoft(P,Fp,Fc)
    #print TDlen, ht.data.length
    if TDlen == 0:
        TDlen = nextPow2(ht.data.length)
    else:
        assert TDlen >= ht.data.length
    
    if fwdplan==None:
        fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)

    ht = lal.ResizeREAL8TimeSeries(ht, 0, TDlen)
    FDlen = TDlen/2+1
    hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            ht.epoch, ht.f0, 1./ht.deltaT/TDlen, lal.lalHertzUnit, 
            FDlen)
    lal.REAL8TimeFreqFFT(hf, ht, fwdplan)
    return hf

def hoff_FD(P, TDlen=0, Fp=None, Fc=None):
    """
    Generate a FD waveform for a FD approximant.
    Note that P.deltaF must be set to call this function
    If TDlen != 0, output FrequencySeries will have zero-padded length TDlen/2+1
    """
    assert (isinstance(TDlen , Number))   # type checking.  Bad argument passing by accident does happen
    if P.deltaF is None:
        raise ValueError('None given for freq. bin size P.deltaF')

    hptilde, hctilde = lalsim.SimInspiralChooseFDWaveform(P.phiref, P.deltaF,
            P.m1, P.m2, P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin,
            P.fMax, P.dist, P.incl, P.lambda1, P.lambda2, P.waveFlags,
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
    if TDlen > 0:
        assert TDlen/2+1 >= htilde.data.length
        htilde = lal.ResizeCOMPLEX16FrequencySeries(htilde, 0, TDlen/2+1)
    return htilde

def norm_hoff(P, IP, TDlen=0, fwdplan=None, Fp=None, Fc=None):
    """
    Generate a normalized FD waveform from ChooseWaveformParams P.
    Will return a COMPLEX16FrequencySeries object.

    If P.approx is a FD approximant, norm_hoff_FD is called.
    This path calls SimInspiralChooseFDWaveform
        If TDlen != 0, freq. bin size will be set from TDlen and P.deltaT
            and output FrequencySeries will be zero-padded to length TDlen/2+1
        If TDlen == 0, freq. bin size will be P.deltaF, output not zero-padded 
        fwdplan must be None for FD approximants.

    If P.approx is a TD approximant, norm_hoff_TD is called.
    This path calls ChooseTDWaveform and performs an FFT.
    """
    # For FD approximants, use the ChooseFDWaveform path = hoff_FD
    if lalsim.SimInspiralImplementedFDApproximants(P.approx)==1:
        # Raise exception if unused arguments were specified
        if fwdplan is not None:
            raise ValueError('FFT plan fwdplan given with FD approximant.\nFD approximants cannot use this.')
        if TDlen!=0: # Set value of P.deltaF from TDlen, P.deltaT
            P.deltaF = 1./P.deltaT/TDlen
        hf = norm_hoff_FD(P, IP, TDlen, Fp, Fc)

    # For TD approximants, do ChooseTDWaveform + FFT path = hoff_TD
    else:
        hf = norm_hoff_TD(P, IP, TDlen, fwdplan, Fp, Fc)

    return hf

def norm_hoff_TD(P, IP, TDlen=0, fwdplan=None, Fp=None, Fc=None):
    """
    Generate a normalized FD waveform from ChooseWaveformParams P
    by creating a TD waveform padded to length TDlen, Fourier transform with
    FFTW3 forward FFT plan fwdplan and normalizing using inner product IP

    If TDlen==0, just pad up to next power of 2

    If you do not provide a forward FFT plan, one will be created.
    If you are calling this function many times, it is best to create it
    once beforehand and pass it in, e.g.:
    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)

    You may pass in antenna patterns Fp, Fc. If none are provided, they will
    be computed from the information in ChooseWaveformParams

    Returns a COMPLEX16FrequencySeries object
    """
    # hp, hc = lalsim.SimInspiralChooseTDWaveform(P.phiref, P.deltaT, P.m1, P.m2, 
    #         P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin, P.fref, P.dist, 
    #         P.incl, P.lambda1, P.lambda2, P.waveFlags, P.nonGRparams,
    #         P.ampO, P.phaseO, P.approx)

    # if Fp!=None and Fc!=None:
    #     hp.data.data *= Fp
    #     hc.data.data *= Fc
    #     hp = lal.AddREAL8TimeSeries(hp, hc)
    #     hoft = hp
    # elif P.radec==False:
    #     fp = Fplus(P.theta, P.phi, P.psi)
    #     fc = Fcross(P.theta, P.phi, P.psi)
    #     hp.data.data *= fp
    #     hc.data.data *= fc
    #     hp = lal.AddREAL8TimeSeries(hp, hc)
    #     hoft = hp
    # else:
    #     hoft = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc, 
    #             P.theta, P.phi, P.psi, 
    #             lalsim.InstrumentNameToLALDetector(P.detector))
    # hoft.epoch = hoft.epoch + P.tref
    # if P.taper != lalsim.LAL_SIM_INSPIRAL_TAPER_NONE: # Taper if requested
    #     lalsim.SimInspiralREAL8WaveTaper(hoft.data, P.taper)
    ht = hoft(P,Fp,Fc)

    if TDlen == 0:
        TDlen = nextPow2(ht.data.length)
    else:
        assert TDlen >= ht.data.length

    if fwdplan==None:
        fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)

    # Warning: the IP has an expected deltaF.  This had better agree!
    ht = lal.ResizeREAL8TimeSeries(ht, 0, TDlen)
    deltaFOutput = 1./ht.deltaT/TDlen
    # Error checking: you have probably screwed up badly if you see this message, by passing inconsistent parameters
    if ( np.abs(IP.deltaF -deltaFOutput) > 1e-5 ):
        print "IP: ", IP.deltaF, " vs calculated ", deltaFOutput
        print "Manual length ", TDlen, " vs (post-extension) ", ht.data.length
        print "Using this number times sampling rate ", int(1/ht.deltaT), " which agrees with ", int(1/P.deltaT)
        print "Buffer time duration is ", ht.deltaT*TDlen

    FDlen = TDlen/2+1
    hoff = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            ht.epoch, ht.f0, deltaFOutput, lal.lalHertzUnit, 
            FDlen)
    lal.REAL8TimeFreqFFT(hoff, ht, fwdplan)
    norm = IP.norm(hoff)
    hoff.data.data /= norm
    return hoff

def norm_hoff_FD(P, IP, TDlen=0, Fp=None, Fc=None):
    """
    Generate a normalized FD waveform for a FD approximant.
    Note that P.deltaF must be set to call this function
    If TDlen != 0, output FrequencySeries will have zero-padded length TDlen/2+1
    """
    if P.deltaF is None:
        raise ValueError('None given for freq. bin size P.deltaF')

    hptilde, hctilde = lalsim.SimInspiralChooseFDWaveform(P.phiref, P.deltaF,
            P.m1, P.m2, P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin,
            P.fMax, P.dist, P.incl, P.lambda1, P.lambda2, P.waveFlags,
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
    if TDlen > 0:
        assert TDlen/2+1 >= htilde.data.length
        htilde =lal.ResizeCOMPLEX16FrequencySeries(htilde, 0, TDlen/2+1)
    norm = IP.norm(htilde)
    htilde.data.data /= norm
    return htilde


def complex_hoft(P, sgn=-1):
    """
    Generate a complex TD waveform from ChooseWaveformParams P
    Returns h(t) = h+(t) + 1j sgn hx(t)
    where sgn = -1 (default)  or +1

    Returns a COMPLEX16TimeSeries object
    """
    assert lalsim.SimInspiralImplementedTDApproximants(P.approx)
    hp, hc = lalsim.SimInspiralChooseTDWaveform(P.phiref, P.deltaT, P.m1, P.m2, 
            P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin, P.fref, P.dist, 
            P.incl, P.lambda1, P.lambda2, P.waveFlags, P.nonGRparams,
            P.ampO, P.phaseO, P.approx)
    if P.taper != lalsim.LAL_SIM_INSPIRAL_TAPER_NONE: # Taper if requested
        lalsim.SimInspiralREAL8WaveTaper(hp.data, P.taper)
        lalsim.SimInspiralREAL8WaveTaper(hc.data, P.taper)

    hoft = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hp.epoch, hp.f0, 
            hp.deltaT, lal.lalDimensionlessUnit, hp.data.length)
    hoft.epoch = hoft.epoch + P.tref
    for i in range(hoft.data.length):
        # make complex object.  Add polarization phase HERE
        hoft.data.data[i] = np.exp(-2*sgn*1j*P.psi)* (hp.data.data[i] + 1j * sgn * hc.data.data[i])
    return hoft

def complex_hoft_TD_polarized(P, TDlen=0, fwdplan=None, sgn=1, signPolarize=1):
    """
    Returns the + or - polarized part of the time domain complex waveform.
    Returns a COMPLEX16TimesSeries object
    """

    # Generate
    hComplex = complex_hoft(P)
#    print hComplex.data.length, "\n"

    # Resize
    if TDlen == 0:
        TDlen = nextPow2(hComplex.data.length)
    else:
        assert TDlen >= hComplex.data.length
    hComplex = lal.ResizeCOMPLEX16TimeSeries(hComplex, 0, TDlen)
#    print hComplex.data.length, "\n"
    
    # Fourier transform
#    FDlen = TDlen/2+1
    FDlen=TDlen
    if fwdplan==None:
        fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)

    hoff = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            hComplex.epoch, hComplex.f0, 1./hComplex.deltaT/TDlen, lal.lalHertzUnit, 
            FDlen)
#    print hoff.data.length, " ", hComplex.data.length, " ", TDlen, "\n"
    lal.COMPLEX16TimeFreqFFT(hoff, hComplex, fwdplan)
    
    # Polarize in the frequency domain:
    #    - set negative frequencies to zero.  These frequencies are above the midpoint
    if signPolarize > 0:
        for i in range( (FDlen)/2):
            hoff.data.data[i+FDlen/2] = 0
    else:
        for i in range(FDlen/2):
            hoff.data.data[i] = 0


    # create space for object
    # https://www.lsc-group.phys.uwm.edu/daswg/projects/lal/nightly/docs/html/group__TimeFreqFFT__h.html
    hoftPolarized = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hComplex.epoch, hComplex.f0, 
            hComplex.deltaT, lal.lalDimensionlessUnit, TDlen)
    rvplan=lal.CreateReverseCOMPLEX16FFTPlan(TDlen,0)
    lal.COMPLEX16FreqTimeFFT(hoftPolarized, hoff, rvplan)
    
    return hoftPolarized


def complex_hoff(P, sgn=-1, TDlen=0, fwdplan=None):
    """
    CURRENTLY ONLY WORKS WITH TD APPROXIMANTS

    Generate a (non-Hermitian) FD waveform from ChooseWaveformParams P
    by creating a complex TD waveform of the form

    h(t) = h+(t) + 1j sgn hx(t)    where sgn = 1 (default) or -1

    Then pad to length TDlen and Fourier transform to get a
    non-Hermitian FD waveform using FFTW3 forward FFT plan fwdplan

    If TDlen==0, just pad up to next power of 2

    If you do not provide a forward FFT plan, one will be created.
    If you are calling this function many times, it is best to create it
    once beforehand and pass it in, e.g.:
    fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)

    You may pass in antenna patterns Fp, Fc. If none are provided, they will
    be computed from the information in ChooseWaveformParams

    Returns a COMPLEX16FrequencySeries object
    """
    # assert lalsim.SimInspiralImplementedTDApproximants(P.approx)
    # hp, hc = lalsim.SimInspiralChooseTDWaveform(P.phiref, P.deltaT, P.m1, P.m2, 
    #         P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin, P.fref, P.dist, 
    #         P.incl, P.lambda1, P.lambda2, P.waveFlags, P.nonGRparams,
    #         P.ampO, P.phaseO, P.approx)
    # if P.taper != lalsim.LAL_SIM_INSPIRAL_TAPER_NONE: # Taper if requested
    #     lalsim.SimInspiralREAL8WaveTaper(hp.data, P.taper)
    #     lalsim.SimInspiralREAL8WaveTaper(hc.data, P.taper)

    # hoft = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hp.epoch, hp.f0, 
    #         hp.deltaT, lal.lalDimensionlessUnit, hp.data.length)
    # hoft.epoch = hoft.epoch + P.tref
    # for i in range(hoft.data.length):
    #     hoft.data.data[i] = hp.data.data[i] + 1j*sgn * hc.data.data[i]

    # if TDlen == 0:
    #     TDlen = nextPow2(hoft.data.length)
    # else:
    #     assert TDlen >= hoft.data.length
    # hoft = lal.ResizeCOMPLEX16TimeSeries(hoft, 0, TDlen)
    hoft = complex_hoft(P,sgn)
 
    if fwdplan==None:
        fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)

    FDlen = TDlen/2+1
    hoff = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            hoft.epoch, hoft.f0, 1./hoft.deltaT/TDlen, lal.lalHertzUnit, 
            TDlen)
    lal.COMPLEX16TimeFreqFFT(hoff, hoft, fwdplan)
    return hoff

def complex_norm_hoff(P, IP, TDlen=0, fwdplan=None,sgn=-1):
    """
    CURRENTLY ONLY WORKS WITH TD APPROXIMANTS

    Generate a (non-Hermitian) FD waveform from ChooseWaveformParams P
    by creating a complex TD waveform of the form

    h(t) = h+(t) + 1j sgn hx(t)    where sgn = 1 (default) or -1

    Then pad to length TDlen and Fourier transform to get a
    non-Hermitian FD waveform using FFTW3 forward FFT plan fwdplan
    and normalize by using inner product IP

    If TDlen==0, just pad up to next power of 2

    If you do not provide a forward FFT plan, one will be created.
    If you are calling this function many times, it is best to create it
    once beforehand and pass it in, e.g.:
    fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)

    You may pass in antenna patterns Fp, Fc. If none are provided, they will
    be computed from the information in ChooseWaveformParams

    Returns a COMPLEX16FrequencySeries object
    """
    assert lalsim.SimInspiralImplementedTDApproximants(P.approx)
    hp, hc = lalsim.SimInspiralChooseTDWaveform(P.phiref, P.deltaT, P.m1, P.m2, 
            P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin, P.fref, P.dist, 
            P.incl, P.lambda1, P.lambda2, P.waveFlags, P.nonGRparams,
            P.ampO, P.phaseO, P.approx)
    if P.taper != lalsim.LAL_SIM_INSPIRAL_TAPER_NONE: # Taper if requested
        lalsim.SimInspiralREAL8WaveTaper(hp.data, P.taper)
        lalsim.SimInspiralREAL8WaveTaper(hc.data, P.taper)

    hoft = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hp.epoch, hp.f0, 
            hp.deltaT, lal.lalDimensionlessUnit, hp.data.length)
    hoft.epoch = hoft.epoch + P.tref
    for i in range(hoft.data.length):
        hoft.data.data[i] = hp.data.data[i] + 1j *sgn* hc.data.data[i]

    if TDlen == 0:
        TDlen = nextPow2(hoft.data.length)
    else:
        assert TDlen >= hoft.data.length
    hoft = lal.ResizeCOMPLEX16TimeSeries(hoft, 0, TDlen)

    if fwdplan==None:
        fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)

    FDlen = TDlen/2+1
    hoff = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            hoft.epoch, hoft.f0, 1./hoft.deltaT/TDlen, lal.lalHertzUnit, 
            TDlen)
    lal.COMPLEX16TimeFreqFFT(hoff, hoft, fwdplan)
    norm = IP.norm(hoff)
    hoff.data.data /= norm
    return hoff


def complex_hoft(P, sgn=-1):
    """
    Generate a complex TD waveform from ChooseWaveformParams P
    Returns h(t) = h+(t) + 1j sgn hx(t)
    where sgn = 1 (default) or -1

    Returns a COMPLEX16TimeSeries object
    """
    assert lalsim.SimInspiralImplementedTDApproximants(P.approx)
    hp, hc = lalsim.SimInspiralChooseTDWaveform(P.phiref, P.deltaT, P.m1, P.m2, 
            P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin, P.fref, P.dist, 
            P.incl, P.lambda1, P.lambda2, P.waveFlags, P.nonGRparams,
            P.ampO, P.phaseO, P.approx)
    if P.taper != lalsim.LAL_SIM_INSPIRAL_TAPER_NONE: # Taper if requested
        lalsim.SimInspiralREAL8WaveTaper(hp.data, P.taper)
        lalsim.SimInspiralREAL8WaveTaper(hc.data, P.taper)

    hoft = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hp.epoch, hp.f0, 
            hp.deltaT, lal.lalDimensionlessUnit, hp.data.length)
    hoft.epoch = hoft.epoch + P.tref
    for i in range(hoft.data.length):
        # make complex object.  Add polarization phase HERE
        hoft.data.data[i] = np.exp(1j*P.psi)* (hp.data.data[i] + 1j * sgn * hc.data.data[i])
    return hoft

def modes_hoft_TD(P, TDlen=0, sgn=1,lmax=2):
    """
    Returns a SphHarmTimeSeries for modes.
    NOT YET WORKING RELIABLY
    """

    # Not implemented except for a handful of cases!
    if  not (P.approx == lalsim.TaylorT1  or P.approx == lalsim.TaylorT2 or P.approx == lalsim.TaylorT3 or P.approx == lalsim.TaylorT4):
        return None

    # Does not currently work -- I need to make the modes manually
    # hlmSequence  = lalsim.SimInspiralChooseTDModes(P.phiref, P.deltaT, P.m1, P.m2, 
    #                                                P.fmin, P.fref, P.dist, P.lambda1, P.lambda2, 
    #                                                P.waveFlags, None, P.ampO, P.phaseO, lmax, P.approx)
    #  return hlmSequence

    # Evolve orbit
    V, phi = lalsim.SimInspiralTaylorT1PNEvolveOrbit(P.phiref, P.deltaT, P.m1, P.m2, P.fmin, P.fref, P.lambda1, P.lambda2, 0, P.phaseO)

    # create from mode.  I will just create one
    v0=1  # gauge term
    hlmExample = lalsim.CreateSimInspiralPNModeCOMPLEX16TimeSeries(V,phi, v0,P.m1,P.m2, P.dist, P.ampO, 2,1)

    return hlmExample

#
# Functions to parse string data
#
def StringToLALApproximant(approxString):
    ret = 0;
    return lalsim.GetApproximantFromString(approxString)
    # if (approxString=='SpinTaylorT4'):
    #     ret = lalsim.SpinTaylorT4
    # elif (approxString=='SpinTaylorT2'):
    #     ret=lalsim.SpinTaylorT2
    # elif (approxString=='SpinTaylorF2'):
    #     ret=lalsim.SpinTaylorF2
    # elif (approxString=='TaylorT4'):
    #     ret=lalsim.TaylorT4
    # elif (approxString=='TaylorT2'):
    #     ret=lalsim.TaylorT2
    # elif (approxString=='TaylorF2'):
    #     ret= lalsim.TaylorF2
    #   #  ret = 4  # manually hardcoding
    # elif (approxString=='TaylorF2RedSpin'):
    #     ret ==lalsim.TaylorF2RedSpin
    #     ret = 5
    # elif (approxString=='IMRPhenomB'):
    #   #  ret==lalsim.IMRPhenomB
    #     ret = 32   # the assignments aren't working
    # else:
    #     print "Unrecognized hardcoded approximant"
    #     sys.exit(1)

    # return ret

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

    hoft = lal.CreateREAL8TimeSeries("h(t)", lal.LIGOTimeGPS(tStart), 0.,
            deltaT, lal.lalDimensionlessUnit, TDlen)

    for i in range(tmplen):
        hoft.data.data[i] = (Fp*hpdat[i] + Fc*hcdat[i]) * scaleH
    for i in range(tmplen,TDlen):
        hoft.data.data[i] = 0.

    return hoft

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

    hoft = lal.CreateREAL8TimeSeries("h(t)", lal.LIGOTimeGPS(tStart), 0., deltaT,
            lal.lalDimensionlessUnit, TDlen)

    for i in range(tmplen):
        hoft.data.data[i] = (Fp*hpdat[i] + Fc*hcdat[i]) * scaleH
    for i in range(tmplen,TDlen):
        hoft.data.data[i] = 0.

    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)
    hoff = lal.CreateCOMPLEX16FrequencySeries("h(f)", 
            hoft.epoch, hoft.f0, 1./deltaT/TDlen, lal.lalHertzUnit, 
            TDlen/2+1)
    lal.REAL8TimeFreqFFT(hoff, hoft, fwdplan)
    return hoff

def NINJA_data_to_norm_hoff(fname, IP, TDlen=0, scaleT=1., scaleH=1., Fp=1., Fc=0.):
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

    hoft = lal.CreateREAL8TimeSeries("h(t)", lal.LIGOTimeGPS(tStart), 0., deltaT,
            lal.lalDimensionlessUnit, TDlen)

    for i in range(tmplen):
        hoft.data.data[i] = (Fp*hpdat[i] + Fc*hcdat[i]) * scaleH
    for i in range(tmplen,TDlen):
        hoft.data.data[i] = 0.

    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)
    hoff = lal.CreateCOMPLEX16FrequencySeries("h(f)", 
            hoft.epoch, hoft.f0, 1./deltaT/TDlen, lal.lalHertzUnit, 
            TDlen/2+1)
    lal.REAL8TimeFreqFFT(hoff, hoft, fwdplan)
    norm = IP.norm(hoff)
    hoff.data.data /= norm
    return hoff



# SUPPORT ROUTINES
