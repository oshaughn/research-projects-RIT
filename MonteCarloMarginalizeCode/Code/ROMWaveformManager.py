#

debug_output = False
rosDebug = False


import numpy as np
import os, sys
import gwsurrogate as gws
import lalsimutils
import lalsimulation as lalsim
import lal

from scipy.interpolate import interp1d
from scipy.linalg import inv
from scipy.interpolate import splrep as _splrep

import pickle
import h5py

try:
    dirBaseFiles =os.environ["GW_SURROGATE"] # surrogate base directory
except:
    print " Make sure you set the GW_SURROGATE environment variable"
    sys.exit(0)

print " ROMWaveformManager: ILE version"

#default_interpolation_kind = 'quadratic'  # spline interpolation   # very slow! 
default_interpolation_kind = 'linear'  # robust, fast

internal_ParametersAvailable ={}
# For each interesting simulation, store the definitions in a file
# Use 'execfile' to load those defintions now

MsunInSec = lal.MSUN_SI*lal.G_SI/lal.C_SI**3



#execfile(dirBaseFiles + "/"+"Sequence-GT-Aligned-UnequalMass/interface.py")

def myzero(arg):
    return 0

def RangeWrap1d(bound, val,fn):
     """
     RangeWrap1d: Uses np.piecewise to construct a piecewise function which is =fn inside the boundary, and 0 outside.
     SHOULD be syntactic sugar, but depending on the python version the language needed to implement this changes.
     """
# #    return (lambda x: fn(x) if  (x>bound[0] and x<bound[1]) else val)
# #  WARNING: piecewise is much faster, but will fail for numpy versions less than 1.8-ish :http://stackoverflow.com/questions/20800324/scipy-pchipinterpolator-error-array-cannot-be-safely-cast-to-required-type
# #     Unfortunately that is the version LIGO uses on their clusters.
     return (lambda x: np.piecewise( x,        [
                 np.logical_and(x> bound[0], x<bound[1]), 
                 np.logical_not(np.logical_and(x> bound[0], x<bound[1])) 
                 ], [fn, myzero]))
#    return (lambda x: np.where( np.logical_and(x> bound[0], x<bound[1]), fn(x),0))  # vectorized , but does not protect the call


def ModeToString(pair):
    return "l"+str(pair[0])+"_m"+str(pair[1]) 

def CreateCompatibleComplexOverlap(hlmf,**kwargs):
    modes = hlmf.keys()
    hbase = hlmf[modes[0]]
    deltaF = hbase.deltaF
    fNyq = np.max(lalsimutils.evaluate_fvals(hbase))
    if debug_output:
#        for key, value in kwargs.iteritems():
#            print (key, value)
        print kwargs
        print "dF, fNyq, npts = ",deltaF, fNyq, len(hbase.data.data)
    IP = lalsimutils.ComplexOverlap(fNyq=fNyq, deltaF=deltaF, **kwargs)
    return IP

def CreateCompatibleComplexIP(hlmf,**kwargs):
    """
    Creates complex IP (no maximization)
    """
    modes = hlmf.keys()
    hbase = hlmf[modes[0]]
    deltaF = hbase.deltaF
    fNyq = np.max(lalsimutils.evaluate_fvals(hbase))
    if debug_output:
#        for key, value in kwargs.iteritems():
#            print (key, value)
        print kwargs
        print "dF, fNyq, npts = ",deltaF, fNyq, len(hbase.data.data)
    IP = lalsimutils.ComplexIP(fNyq=fNyq, deltaF=deltaF, **kwargs)
    return IP

class NRError(Exception):
    """Base class for this module"""
    pass
class NRNoSimulation(NRError):
    """Nothing"""
    def __init__(self,expr,msg):
        print "No known simulation ", expr, msg
    pass

def SurrogateDimensionlessBasisFunction(sur,k):    
    def w(t):
        return sur.amp_fit_func(k,t)*np.exp(1j*sur.phase_fit_func(k,t))

    return w

def sur_identity(t,hp,hc):
    return t, hp, hc
def sur_conj(t,hp,hc):
    return t, hp, -hc


def ConvertWPtoSurrogateParams(P,**kwargs):
    """
    Takes P, returns arguments of the form usually used in gwsurrogate.
    (currently, just returns 1/q = P.m1/P.m1, the mass ratio parameter usually accepted)
    """

    q = P.m2/P.m1
    return 1./q


class WaveformModeCatalog:
    """
    Class containing ROM model.
    API is currently **unsafe** for precessing binaries (=ambiguous reference time)
    Reference for underlying notation:   Eq. (30) in http://arxiv.org/pdf/1308.3565v2
    """


    def __init__(self, group ,param, lmax=2, 
                 strain_basis_functions_dimensionless=None,
                 mode_list_to_load=[(2,2)],build_fourier_time_window=1000,reflection_symmetric=True,max_nbasis_per_mode=None):
        self.group  = group
        self.param = param 
        self.deltaToverM =0
        self.fOrbitLower =0.    #  Used to clean results.  Based on the phase of the 22 mode
        self.fMinMode ={}
        self.sur_dict = {}
        self.post_dict = {}
        self.post_dict_complex ={}
        self.post_dict_complex_coef ={}
        self.parameter_convert = {}
        self.nbasis_per_mode ={}   # number of basis functions

        self.sur =  gws.EvaluateSurrogate(dirBaseFiles +'/'+group+param) # straight up filename.  MODIFY to change to use negative modes
        raw_modes = self.sur.all_model_modes()
        self.modes_available = []
        # Load surrogates from a mode-by-mode basis, and their conjugates
        for mode in raw_modes:
            print " Loading mode ", mode
            self.modes_available.append(mode)
            self.sur_dict[mode] = self.sur.single_mode(mode)
            self.post_dict[mode] = sur_identity
            self.post_dict_complex[mode]  = lambda x: x   # to mode
            self.post_dict_complex_coef[mode] = lambda x:x  #  to coefficients.
            self.parameter_convert[mode] = ConvertWPtoSurrogateParams   # default conversion routine
            print ' mode ', mode, self.sur_dict[mode].B.shape
            self.nbasis_per_mode[mode] = (self.sur_dict[mode].B.shape)[1]
            if max_nbasis_per_mode != None  and self.sur_dict[mode].surrogate_mode_type == 'waveform_basis':
             if max_nbasis_per_mode >0: # and max_nbasis_per_mode < self.nbasis_per_mode[mode]:
                # See  https://arxiv.org/pdf/1308.3565v2.pdf  Eqs. 13 - 19
                # Must truncate *orthogonal* basis.
                # Works only for LINEAR basis
                # B are samples of the basis on some long time, V
                sur = self.sur_dict[mode]
                print " Truncating basis for mode ", mode, " to size ", max_nbasis_per_mode,  " but note the number of EIM points remains the same..."
                V = self.sur_dict[mode].V
                n_basis = len(V)
                V_inv = inv(V)
                mtx_E = np.dot(self.sur_dict[mode].B,V)
#                print "E ", mtx_E.shape
                # Zero out the components we don't want
                if max_nbasis_per_mode < n_basis:
                    mtx_E[:,max_nbasis_per_mode:n_basis] *=0
                # Regenerate
                sur.B = np.dot(mtx_E , V_inv)
                sur.reB_spline_params = [_splrep(sur.times, sur.B[:,jj].real, k=3) for jj in range(sur.B.shape[1])]
                sur.imB_spline_params = [_splrep(sur.times, sur.B[:,jj].imag, k=3) for jj in range(sur.B.shape[1])]
                self.nbasis_per_mode[mode] = len(self.sur_dict[mode].V) # if you truncate over the orthogonal basis, you still need to use the fit at all the EIM points!
                # This SHOULD update the copies inside the surrogate, so the later interpolation called by EvaluateSingleModeSurrogate will interpolate this data
                
            if reflection_symmetric and raw_modes.count((mode[0],-mode[1]))<1:
                mode_alt = (mode[0],-mode[1])
                self.nbasis_per_mode[mode_alt] = self.nbasis_per_mode[mode]
#                if max_nbasis_per_mode:
 #                   self.nbasis_per_mode[mode_alt] = np.max([int(max_nbasis_per_mode),1])    # INFRASTRUTCTURE PLAN: Truncate t                print " Loading mode ", mode_alt, " via reflection symmetry "
                self.modes_available.append(mode_alt)
                self.post_dict[mode_alt] = sur_conj
                self.post_dict_complex_coef[mode_alt] = lambda x,l=mode[0]: np.power(-1,l)*np.conj(x)  # beware, do not apply this twice.
                self.post_dict_complex[mode_alt] = np.conj  # beware, do not apply this twice.
                self.sur_dict[mode_alt] = self.sur_dict[mode]
                self.parameter_convert[mode_alt] = ConvertWPtoSurrogateParams
        # CURRENTLY ONLY LOAD THE 22 MODE and generate the 2,-2 mode by symmetr
        t, hp, hc = self.sur_dict[(2,2)](q=1);
        self.ToverMmin = t.min()
        self.ToverMmax = t.max()
        self.ToverM_peak = t[np.argmax(np.abs(hp**2+hc**2))]  # discrete maximum time
        if rosDebug:
            print " Peak time for ROM ", self.ToverM_peak

        # BASIS MANAGEMENT: Not yet implemented
        # Assume a DISTINCT BASIS SET FOR AL MODES, which will be ANNOYING
        self.strain_basis_functions_dimensionless_data ={}
        self.strain_basis_functions_dimensionless = self.sur_dict[(2,2)].resample_B  # We may need to add complex conjugate functions too. And a master index for basis functions associated with different modes

    def print_params(self):
        print " Surrogate model "
        print "   Modes available "
        for mode in self.sur_dict:
            print  "    " , mode, " nbasis = ", self.nbasis_per_mode[mode]

    def complex_hoft(self, P, force_T=False, deltaT=1./16384, time_over_M_zero=0.,sgn=-1):
        hlmT = self.hlmoft(P, force_T, deltaT,time_over_M_zero)
        npts = hlmT[(2,2)].data.length
        wfmTS = lal.CreateCOMPLEX16TimeSeries("h", lal.LIGOTimeGPS(0.), 0., deltaT, lalsimutils.lsu_DimensionlessUnit, npts)
        wfmTS.data.data[:] = 0   # SHOULD NOT BE NECESARY, but the creation operator doesn't robustly clean memory
        wfmTS.epoch = hlmT[(2,2)].epoch
        for mode in hlmT.keys():
            # PROBLEM: Be careful with interpretation. The incl and phiref terms are NOT tied to L.
            if rosDebug:
                print mode, np.max(hlmT[mode].data.data), " running max ",  np.max(np.abs(wfmTS.data.data))
            wfmTS.data.data += np.exp(-2*sgn*1j*P.psi)* hlmT[mode].data.data*lal.SpinWeightedSphericalHarmonic(P.incl,-P.phiref,-2, int(mode[0]),int(mode[1]))
        return wfmTS
    def complex_hoff(self,P, force_T=False):
        htC  = self.complex_hoft(P, force_T=force_T,deltaT= P.deltaT)
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen == htC.data.length
        hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)",
                                                htC.epoch, htC.f0, 1./htC.deltaT/htC.data.length, lalsimutils.lsu_HertzUnit, 
                                                htC.data.length)
        fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(htC.data.length,0)
        lal.COMPLEX16TimeFreqFFT(hf, htC, fwdplan)
        return hf
    def real_hoft(self,P,Fp=None, Fc=None):
        """
        Returns the real-valued h(t) that would be produced in a single instrument.
        Translates epoch as needed.
        Based on 'hoft' in lalsimutils.py
        """
        # Create complex timessereis
        htC = self.complex_hoft(force_T=1./P.deltaF, deltaT= P.deltaT)  # note P.tref is NOT used in the low-level code
        TDlen  = htC.data.length
        if rosDebug:
            print  "Size sanity check ", TDlen, 1/(P.deltaF*P.deltaT)
            print " Raw complex magnitude , ", np.max(htC.data.data)
            
        # Create working buffers to extract data from it -- wasteful.
        hp = lal.CreateREAL8TimeSeries("h(t)", htC.epoch, 0.,
            P.deltaT, lalsimutils.lsu_DimensionlessUnit, TDlen)
        hc = lal.CreateREAL8TimeSeries("h(t)", htC.epoch, 0.,
            P.deltaT, lalsimutils.lsu_DimensionlessUnit, TDlen)
        hT = lal.CreateREAL8TimeSeries("h(t)", htC.epoch, 0.,
            P.deltaT, lalsimutils.lsu_DimensionlessUnit, TDlen)
        # Copy data components over
        hp.data.data = np.real(htC.data.data)
        hc.data.data = np.imag(htC.data.data)
        # transform as in lalsimutils.hoft
        if Fp!=None and Fc!=None:
            hp.data.data *= Fp
            hc.data.data *= Fc
            hp = lal.AddREAL8TimeSeries(hp, hc)
            hoft = hp
        elif P.radec==False:
            fp = lalsimutils.Fplus(P.theta, P.phi, P.psi)
            fc = lalsimutils.Fcross(P.theta, P.phi, P.psi)
            hp.data.data *= fp
            hc.data.data *= fc
            hp.data.data  = lal.AddREAL8TimeSeries(hp, hc)
            hoft = hp
        else:
            # Note epoch must be applied FIRST, to make sure the correct event time is being used to construct the modulation functions
            hp.epoch = hp.epoch + P.tref
            hc.epoch = hc.epoch + P.tref
            if rosDebug:
                print " Real h(t) before detector weighting, ", np.max(hp.data.data), np.max(hc.data.data)
            hoft = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc,    # beware, this MAY alter the series length??
                P.phi, P.theta, P.psi, 
                lalsim.DetectorPrefixToLALDetector(str(P.detector)))
            hoft = lal.CutREAL8TimeSeries(hoft, 0, hp.data.length)       # force same length as before??
            if rosDebug:
                print "Size before and after detector weighting " , hp.data.length, hoft.data.length
        if rosDebug:
            print  " Real h_{IFO}(t) generated, pre-taper : max strain =", np.max(hoft.data.data)
        if P.taper != lalsimutils.lsu_TAPER_NONE: # Taper if requested
            lalsim.SimInspiralREAL8WaveTaper(hoft.data, P.taper)
        if P.deltaF is not None:
            TDlen = int(1./P.deltaF * 1./P.deltaT)
            print "Size sanity check 2 ", int(1./P.deltaF * 1./P.deltaT), hoft.data.length
            assert TDlen >= hoft.data.length
            npts = hoft.data.length
            hoft = lal.ResizeREAL8TimeSeries(hoft, 0, TDlen)
            # Zero out the last few data elements -- NOT always reliable for all architectures; SHOULD NOT BE NECESSARY
            hoft.data.data[npts:TDlen] = 0

        if rosDebug:
            print  " Real h_{IFO}(t) generated : max strain =", np.max(hoft.data.data)
        return hoft

    def non_herm_hoff(self,P):
        """
        Returns the 2-sided h(f) associated with the real-valued h(t) seen in a real instrument.
        Translates epoch as needed.
        Based on 'non_herm_hoff' in lalsimutils.py
        """
        htR = self.real_hoft() # Generate real-valued TD waveform, including detector response
        if P.deltaF == None: # h(t) was not zero-padded, so do it now
            TDlen = nextPow2(htR.data.length)
            htR = lal.ResizeREAL8TimeSeries(htR, 0, TDlen)
        else: # Check zero-padding was done to expected length
            TDlen = int(1./P.deltaF * 1./P.deltaT)
            assert TDlen == htR.data.length
        fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(htR.data.length,0)
        htC = lal.CreateCOMPLEX16TimeSeries("hoft", htR.epoch, htR.f0,
            htR.deltaT, htR.sampleUnits, htR.data.length)
        # copy h(t) into a COMPLEX16 array which happens to be purely real
        htC.data.data[:htR.data.length] = htR.data.data
#        for i in range(htR.data.length):
#            htC.data.data[i] = htR.data.data[i]
        hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)",
            htR.epoch, htR.f0, 1./htR.deltaT/htR.data.length, lalsimutils.lsu_HertzUnit, 
            htR.data.length)
        lal.COMPLEX16TimeFreqFFT(hf, htC, fwdplan)
        return hf


    def estimateFminHz(self,P,fmin=10.):
        return 2*self.fOrbitLower/(MsunInSec*(P.m1+P.m2)/lal.MSUN_SI)

    def estimateDurationSec(self,P,fmin=10.):
        """
        estimateDuration uses fmin*M from the (2,2) mode to estimate the waveform duration from the *well-posed*
        part.  By default it uses the *entire* waveform duration.
        CURRENTLY DOES NOT IMPLEMENT frequency-dependent duration
        """
        return None

    def basis_oft(self,  P, force_T=False, deltaT=1./16384, time_over_M_zero=0.,return_numpy=False):
        m_total_s = MsunInSec*(P.m1+P.m2)/lal.MSUN_SI
        # Create a suitable set of time samples.  Zero pad to 2^n samples.
        T_estimated =  20 # FIXME. Time in seconds
        npts=0
        if not force_T:
            npts_estimated = int(T_estimated/deltaT)
            npts = lalsimutils.nextPow2(npts_estimated)
        else:
            npts = int(force_T/deltaT)
            print " Forcing length T=", force_T, " length ", npts
        tvals = (np.arange(npts)-npts/2)*deltaT   # Use CENTERED time to make sure I handle CENTERED NR signal (usual)
        if rosDebug:
            print " time range being sampled ", [min(tvals),max(tvals)], " corresponding to dimensionless range", [min(tvals)/m_total_s,max(tvals)/m_total_s], " for mtotal ", (P.m1+P.m2)/lal.MSUN_SI

        # Identify natural time region, and prepare to taper accordingly

        # Basis function matrix on grid
        # BEWARE: we need to zero out anything not well within the region of validity. 
        basis_grid = {}
        for mode in self.sur_dict:
            new_B = self.sur_dict[mode].resample_B(tvals/m_total_s + self.ToverM_peak,ext=1)  # should return 0 outside the domain

            # Identify start and end time, for tapering if desired (probably not)
            tmin = (self.sur_dict[mode].tmin-self.ToverM_peak)*m_total_s
            tmax = (self.sur_dict[mode].tmax-self.ToverM_peak)*m_total_s
            for indx in np.arange(new_B.shape[1]):
                if indx >= self.nbasis_per_mode[mode]:    # Allow truncation of model. Note fencepost
                    continue
                how_to_store = (mode[0], mode[1], indx)
                if rosDebug:
                    print " Storing resampled copy : ", how_to_store, " expected start and end ", tmin, tmax, " for mtot = ", (P.m1+P.m2)/lal.MSUN_SI
                basis_grid[how_to_store] = self.post_dict_complex[mode](new_B[:,indx])

        # Optional: return just this way
        if return_numpy:
            return tvals, basis_grid

        # Otherwise, pack into a set of LAL data structures, with epoch etc set appropriately, so we can use in ILE without changes
        ret = {}

        # Taper design.  Note taper needs to be applied IN THE RIGHT PLACE.  
        print " Tapering basis "
        t_start_taper = -self.ToverM_peak*m_total_s
        t_end_taper = t_start_taper +100*m_total_s  # taper over around 100 M
#        print t_start_taper, t_end_taper, tvals
        def fn_taper(x):
            return np.piecewise(x,[x<t_start_taper,x>t_end_taper,np.logical_and(x>=t_start_taper, x<=t_end_taper)], [0,1,lambda z: 0.5-0.5*np.cos(np.pi* (z-t_start_taper)/(t_end_taper-t_start_taper))])
        vectaper= fn_taper(tvals)

        for ind in basis_grid:
            wfmTS = lal.CreateCOMPLEX16TimeSeries("h", lal.LIGOTimeGPS(0.), 0., deltaT, lalsimutils.lsu_DimensionlessUnit, npts)
            wfmTS.epoch = tvals[0]
            wfmTS.data.data = vectaper*basis_grid[ind]
            ret[ind] = wfmTS

        return ret

    def basis_off(self,  P, force_T=False, deltaT=1./16384, time_over_M_zero=0.):
        """
        basis_off takes fourier transforms of LAL timeseries generated from basis_oft.
        All modes have physical units, appropriate to a physical signal.
        """
        # Code is operationally identical to hlmoff, hence the names
        hlmF ={}
        hlmT = self.basis_oft(P,force_T=force_T,deltaT=deltaT,time_over_M_zero=time_over_M_zero, return_numpy=False)
        for mode in hlmT.keys():
            wfmTS=hlmT[mode]
            # Take the fourier transform
            wfmFD = lalsimutils.DataFourier(wfmTS)  # this creates a new f series for *each* call.
            # Store the resulting mode
            hlmF[mode] = wfmFD
        return hlmF


    def coefficients(self,P,**kwargs):
        """
        Returns the values of the ROM coefficients for the parameter P.  
        Usees the key-value pairing convention described in basis_oft.
        Note that the coefficients are created on a mode by mode basis, so the conversion call is VERY redundant.
        """

        coefs = {}
        for mode in self.sur_dict:
            params = self.parameter_convert[mode](P,**kwargs)  # very redundant, almost certainly the same for all modes in the surrogate.
            params_surrogate = self.sur_dict[mode].get_surr_params(params)
            if rosDebug:
                print " passing params to mode : ", mode, params
                print " surrogate natural parameter is ", params_surrogate
            x0= self.sur_dict[mode]._affine_mapper(params_surrogate)
            amp_eval = self.sur_dict[mode]._amp_eval(x0)
            phase_eval = self.sur_dict[mode]._phase_eval(x0)
            norm_eval = self.sur_dict[mode]._norm_eval(x0)
            h_EIM = np.zeros(len(amp_eval))
            if self.sur_dict[mode].surrogate_mode_type  == 'waveform_basis':
                h_EIM = norm_eval*amp_eval*np.exp(1j*phase_eval)
            for indx in np.arange(self.nbasis_per_mode[mode]):  
                how_to_store = (mode[0], mode[1], indx)
                coefs[how_to_store]  = self.post_dict_complex_coef[mode](h_EIM[indx])   # conjugation as needed

        return coefs

    def hlmoft(self,  P, force_T=False, deltaT=1./16384, time_over_M_zero=0.,use_basis=False):
        """
        hlmoft uses the dimensionless ROM basis functions to extract hlm(t) in physical units, in a LAL array.
        The argument 'P' is a ChooseWaveformParaams object
        FIXME: Add tapering option!
        """
        hlmT ={}
        # Define units
        m_total_s = MsunInSec*(P.m1+P.m2)/lal.MSUN_SI
        distance_s = P.dist/lal.C_SI  # insures valid units

        # Create a suitable set of time samples.  Zero pad to 2^n samples.
        T_estimated =  20 # FIXME. Time in seconds
        npts=0
        if not force_T:
            npts_estimated = int(T_estimated/deltaT)
            npts = lalsimutils.nextPow2(npts_estimated)
        else:
            npts = int(force_T/deltaT)
            print " Forcing length T=", force_T, " length ", npts
        tvals = (np.arange(npts)-npts/2)*deltaT   # Use CENTERED time to make sure I handle CENTERED NR signal (usual)
        if rosDebug:
            print " time range being sampled ", [min(tvals),max(tvals)], " corresponding to dimensionless range", [min(tvals)/m_total_s,max(tvals)/m_total_s]

        # Populate basis coefficients, if we need it
        if use_basis:
            bT = self.basis_oft(P, force_T=force_T, deltaT=deltaT, time_over_M_zero=time_over_M_zero)
            coefs = self.coefficients(P)

        # extract arguments
        q = P.m1/P.m2  # NOTE OPPOSITE CONVENTION ADOPTED BY SURROGATE GROUP. Code WILL FAIL if q<1
        if q<1:
            q=1./q  # this flips the objects, swapping phase, but it is a good place to start
        # Loop over all modes in the system
        for mode in self.modes_available: # loop over modes
            # Copy into a new LIGO time series object
            wfmTS = lal.CreateCOMPLEX16TimeSeries("h", lal.LIGOTimeGPS(0.), 0., deltaT, lalsimutils.lsu_DimensionlessUnit, npts)
            wfmTS.data.data *=0;  # init - a sanity check

            # Option 1: Use the surrogate functions themselves
            if not use_basis:
                t_phys, hp_dim, hc_dim = self.post_dict[mode](*self.sur_dict[mode](q=q,samples=tvals/m_total_s + self.ToverM_peak))  # center time values AT PEAK
                wfmTS.data.data =  m_total_s/distance_s * (hp_dim - 1j*hc_dim)
                # Zero out data after tmax and before tmin
                wfmTS.data.data[tvals/m_total_s<self.ToverMmin-self.ToverM_peak] = 0
                wfmTS.data.data[tvals/m_total_s>self.ToverMmax-self.ToverM_peak] = 0
                # Apply taper *at times where ROM waveform starts*
                if P.taper:
                    n0 = np.argmin(np.abs(tvals)) # origin of time; this will map to the peak
                    # start taper
                    nstart = n0+int((self.ToverMmin-self.ToverM_peak)*m_total_s/deltaT)        # note offset for tapering
                    ntaper = int( (self.ToverM_peak-self.ToverMmin)*m_total_s/deltaT*0.1)  # SHOULD BE set by a few Hz in time fixed 1% of waveform length
                    vectaper= 0.5 - 0.5*np.cos(np.pi*np.arange(ntaper)/(1.*ntaper))
                    print " Tapering ROM hlm(t) for ", mode, " over range ", nstart, nstart+ntaper, " or time offset ", nstart*deltaT, " and window ", ntaper*deltaT
                    wfmTS.data.data[nstart:nstart+ntaper]*=vectaper

                    # end taper
                    ntaper = int( (self.ToverMmax-self.ToverM_peak)*m_total_s/deltaT*0.1)  # SHOULD BE set by a few Hz in time fixed 1% of waveform length
                    nend = n0 + int((self.ToverMmax-self.ToverM_peak)*m_total_s/deltaT)
                    vectaper= 0.5 - 0.5*np.cos(np.pi* (1-np.arange(ntaper)/(1.*ntaper)))
                    wfmTS.data.data[-ntaper+nend:nend]*=vectaper
                    print " Tapering ROM hlm(t) end for ", mode, " over range ", nend-ntaper, nend, " or time offset ", nend*deltaT, ntaper*deltaT


            else:
                # Option 2: reconstruct with basis.  Note we will previously have called the basis function generator.
                # select index list
                indx_list_ok = [indx for indx in coefs.keys()  if indx[0]==mode[0] and indx[1]==mode[1]]
                if rosDebug:
                    print " To reconstruct ", mode, " using the following basis entries ",  indx_list_ok
                fudge_factor = 1    # FIXME, TERRIBLE!!!
                for indx in indx_list_ok: # loop over basis
#                    print " Adding " , bT[indx].data.length, wfmTS.data.length, indx, " of " ,len(coefs), (coefs[indx]*bT[indx].data.data)[len(bT[indx].data.data)/2]
                    wfmTS.data.data +=  fudge_factor*m_total_s/distance_s * np.array(coefs[indx]*bT[indx].data.data,dtype=np.cfloat)

                # CLEAN UP
                # Zero out data after tmax and before tmin
                wfmTS.data.data[tvals/m_total_s<self.ToverMmin-self.ToverM_peak] = 0
                wfmTS.data.data[tvals/m_total_s>self.ToverMmax-self.ToverM_peak] = 0
                # Apply taper *at times where ROM waveform starts*
                if P.taper:
                    n0 = np.argmin(np.abs(tvals)) # origin of time; this will map to the peak
                    # start taper
                    nstart = n0+int((self.ToverMmin-self.ToverM_peak)*m_total_s/deltaT)        # note offset for tapering
                    ntaper = int( (self.ToverM_peak-self.ToverMmin)*m_total_s/deltaT*0.1)  # SHOULD BE set by a few Hz in time fixed 1% of waveform length
                    vectaper= 0.5 - 0.5*np.cos(np.pi*np.arange(ntaper)/(1.*ntaper))
                    print " Tapering ROM hlm(t) for ", mode, " over range ", nstart, nstart+ntaper, " or time offset ", nstart*deltaT, " and window ", ntaper*deltaT
                    wfmTS.data.data[nstart:nstart+ntaper]*=vectaper

                    # end taper
                    ntaper = int( (self.ToverMmax-self.ToverM_peak)*m_total_s/deltaT*0.1)  # SHOULD BE set by a few Hz in time fixed 1% of waveform length
                    nend = n0 + int((self.ToverMmax-self.ToverM_peak)*m_total_s/deltaT)
                    vectaper= 0.5 - 0.5*np.cos(np.pi* (1-np.arange(ntaper)/(1.*ntaper)))
                    wfmTS.data.data[-ntaper+nend:nend]*=vectaper
                    print " Tapering ROM hlm(t) end for ", mode, " over range ", nend-ntaper, nend, " or time offset ", nend*deltaT, ntaper*deltaT



            # Set the epoch for the time series correctly: should have peak near center of series by construction
            # MUST USE TIME CENTERING INFO FROM SURROGATE
            wfmTS.epoch = -deltaT*wfmTS.data.length/2

            # Store the resulting mode
            hlmT[mode] = wfmTS

        return hlmT

    def hlmoff(self, P,force_T=False, deltaT=1./16384, time_over_M_zero=0.,use_basis=False):
        """
        hlmoff takes fourier transforms of LAL timeseries generated from hlmoft.
        All modes have physical units, appropriate to a physical signal.
        """
        hlmF ={}
        hlmT = self.hlmoft(P,force_T=force_T,deltaT=deltaT,time_over_M_zero=time_over_M_zero,use_basis=use_basis)
        for mode in hlmT.keys():
            wfmTS=hlmT[mode]
            # Take the fourier transform
            wfmFD = lalsimutils.DataFourier(wfmTS)  # this creates a new f series for *each* call.
            # Store the resulting mode
            hlmF[mode] = wfmFD
        return hlmF

    def conj_hlmoff(self, P,force_T=False, deltaT=1./16384, time_over_M_zero=0.,use_basis=False):
        """
        conj_hlmoff takes fourier transforms of LAL timeseries generated from hlmoft, but after complex conjugation.
        All modes have physical units, appropriate to a physical signal.
        """
        hlmF ={}
        hlmT = self.hlmoft(P,force_T=force_T,deltaT=deltaT,time_over_M_zero=time_over_M_zero,use_basis=use_basis)
        for mode in hlmT.keys():
            wfmTS=hlmT[mode]
            wfmTS.data.data = np.conj(wfmTS.data.data)  # complex conjugate
            # Take the fourier transform
            wfmFD = lalsimutils.DataFourier(wfmTS)  # this creates a new f series for *each* call.
            # Store the resulting mode
            hlmF[mode] = wfmFD
        return hlmF


class WaveformMode:
    """
    Class representing a dimensionless timeseries X(\tau=t/M)
    """

###
### FFT syntatic sugar
###
def DataFourierNumpy(wfComplex):    # assume (n,2) size array of [tvals, g(t)]; return fvals, tilde(g). 
    # FFT
    # Unroll -- it will save me time later if we are continuous
    T = wfComplex[-1,0] - wfComplex[0,0]
    n = len(wfComplex)
    dt = T/n
    gtilde = np.fft.fft(wfComplex[:,1])*dt   
    # Option 1: Raw pairing, not caring about continuity in the frequency array.
#    fvals = 1./T* np.array([ k if  k<=n/2 else k-n for k in np.arange(n)]) 
    # Option 2: Unroll
    gtilde = np.roll(gtilde, n/2)
    fvals = 1./T*(np.arange(n) - n/2+1)     # consistency with DataFourier: \int dt e(i2\pi f t), note REVERSING
    wfComplexF = np.array([fvals,gtilde[::-1]]).T
    return wfComplexF

def DataInverseFourierNumpy(wfComplex):    # assume (n,2) size array of [fvals, gtilde(f)]; return tvals, g
#    print "NAN check ", wfComplex[np.isnan(wfComplex[:,1]),1]
    df = wfComplex[1,0] - wfComplex[0,0]
    n = len(wfComplex)
    T = 1./df
    dt = T/n
    # Undo the roll, then perform the FFT
    datReversed = wfComplex[:,1][::-1]
    g = np.fft.ifft(np.roll(datReversed, -n/2))*n*df # undo the reverse and roll
    # Assign correct time values.  
    #    - Note the zero of time is now centered in the array -- we don't carry a time reference with us.
    tvals = dt*(np.arange(n) -n/2+1)
    wfComplexT = np.array([tvals,g]).T
    return wfComplexT

###
### Mode syntactic sugar
###
def RawGetModePeakTime(wfMode):   # assumed applied to complex data sequence
    nmax = np.argmax(np.abs(wfMode[:,1]))
    return np.real(wfMode[nmax][0])



###
### Plotting syntactic sugar
###

# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot    
mode_line_style= {}
mode_line_style[(2,2)] = 'r'
mode_line_style[(2,1)] = 'b'
mode_line_style[(2,-1)] = 'b'
mode_line_style[(2,-2)] = 'r'
mode_line_style[(3,3)] = 'r-'
mode_line_style[(3,-3)] = 'r-'
mode_line_style[(3,1)] = 'b-'
mode_line_style[(3,-1)] = 'b-'

mode_line_style[(2,0)] = 'g'


for l in np.arange(2,5):
    for m in np.arange(-l,l+1):
        if not ((l,m) in mode_line_style.keys()):
            mode_line_style[(l,m)] = 'k'

