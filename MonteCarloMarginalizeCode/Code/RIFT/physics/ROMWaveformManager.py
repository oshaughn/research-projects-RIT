#

debug_output = False
rosDebug = False


import numpy as np
import os, sys
try:
    import gwtools
    import gwsurrogate as gws
    print("  gwsurrogate: ",  gws.__file__)
    print("  gwtools: ",  gwtools.__file__)
except:
    print(" - no gwsurrogate - (almost everything from ROMWaveformManager will hard fail if you use it) ")
try:
    import NRSur7dq2
    print("  NRSur7dq2: ", NRSur7dq2.__version__, NRSur7dq2.__file__)
except:
    print(" - no NRSur7dq2 - ")

import lalsimulation as lalsim
import lal

from .. import lalsimutils
try:
    import LALHybrid
except:
    print(" - no hybridization - ")

from scipy.interpolate import interp1d
from scipy.linalg import inv
from scipy.interpolate import splrep as _splrep

import pickle
import h5py

try:
    dirBaseFiles =os.environ["GW_SURROGATE"] # surrogate base directory
except:
    print( " ==> WARNING:  GW_SURROGATE environment variable is not set <== ")
    print( "    Only surrogates with direct implementation are available (NRSur7dq2) ")

print(" ROMWaveformManager: ILE version")

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
#        for key, value in kwargs.items():
#            print (key, value)
        print(kwargs)
        print("dF, fNyq, npts = ",deltaF, fNyq, len(hbase.data.data))
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
#        for key, value in kwargs.items():
#            print (key, value)
        print(kwargs)
        print("dF, fNyq, npts = ",deltaF, fNyq, len(hbase.data.data))
    IP = lalsimutils.ComplexIP(fNyq=fNyq, deltaF=deltaF, **kwargs)
    return IP

class NRError(Exception):
    """Base class for this module"""
    pass
class NRNoSimulation(NRError):
    """Nothing"""
    def __init__(self,expr,msg):
        print("No known simulation ", expr, msg)
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
#    return {"q":1./q}
    return 1./q

def ConvertWPtoSurrogateParamsAligned(P,**kwargs):
    """
    Takes P, returns arguments of the form used in gwsurrogate for a nonprecessing binary
    """
    q = P.m2/P.m1
    chi1 = np.array([0.0,0.0,P.s1z])
    chi2 = np.array([0.0,0.0,P.s2z])
    mtot=P.m1+P.m2
    tidal = {'Lambda1': P.lambda1,'Lambda2': P.lambda2}
    dist_mpc = P.dist/1e6/lal.PC_SI
    val =[1./q, chi1, chi2, mtot, dist_mpc, tidal]
    return val



def ConvertWPtoSurrogateParamsPrecessing(P,**kwargs):
    """
    Takes P, returns arguments of the form usually used in gwsurrogate.
    (currently, just returns 1/q = P.m1/P.m1, the mass ratio parameter usually accepted)
    """

    q = P.m2/P.m1
    chi1 = P.extract_param('chi1')
    theta1=phi1 =0
    if np.abs(chi1)>1e-5:
        theta1 = np.arccos( P.s1z/chi1)
        phi1 = np.arctan2(P.s1x,P.s1y)
#    return {"q":1./q, "chi1": chi1,"theta1":theta1,"phi1":phi1,"chi2z":P.s2z}
    val =np.array([1./q, chi1,theta1,phi1,P.s2z])
    return val

def ConvertWPtoSurrogateParamsPrecessingFull(P,**kwargs):
    """
    Takes P, returns arguments of the form usually used in gwsurrogate.
    (currently, just returns 1/q = P.m1/P.m1, the mass ratio parameter usually accepted)
    """

    q = P.m2/P.m1
    val =[1./q, np.array([P.s1x,P.s1y,P.s1z]), np.array([P.s2x,P.s2y,P.s2z]) ]
    return val



class WaveformModeCatalog:
    """
    Class containing ROM model.
    API is currently **unsafe** for precessing binaries (=ambiguous reference time)
    Reference for underlying notation:   Eq. (30) in http://arxiv.org/pdf/1308.3565v2
       group
       param
       lmax                          # specifies modes to attempt to load. Not guaranteed to/required to find all.
       strain_basis_functions_dimensionless   # don't recall
       mode_list_to_load        # ability to constrain the mode list.  Passed directly to low-level code
       build_fourier_time_window  # window for FT. NOT USED
       reflection_symmetric     # reflection symmetry used
       max_nbasis_per_mode   # constrain basis size
       coord_names_internal    # coordinate names used by the basis.  FUTURE

    """


    def __init__(self, group ,param, lmax=2, 
                 strain_basis_functions_dimensionless=None,
                 mode_list_to_load=None,build_fourier_time_window=1000,reflection_symmetric=True,max_nbasis_per_mode=None,coord_names_internal=['q']):
        self.group  = group
        self.param = param 
        self.deltaToverM =0
        self.lmax =lmax
        self.coord_names=coord_names_internal
        self.fOrbitLower =0.    #  Used to clean results.  Based on the phase of the 22 mode
        self.fMinMode ={}
        self.sur_dict = {}
        self.post_dict = {}
        self.post_dict_complex ={}
        self.post_dict_complex_coef ={}
        self.parameter_convert = {}
        self.single_mode_sur = True
        self.nbasis_per_mode ={}   # number of basis functions
        self.reflection_symmetric = reflection_symmetric

        lm_list=None
        lm_list = []
        if rosDebug:
            print(" WARNING: Using a restricted mode set requires a custom modification to gwsurrogate ")
        Lmax =lmax
        for l in np.arange(2,Lmax+1):
            for m in np.arange(-l,l+1):
                if m<0 and reflection_symmetric:
                    continue
                lm_list.append( (l,m))
        if not(mode_list_to_load is None):
            lm_list = mode_list_to_load  # overrride
        if rosDebug:
            print(" ROMWaveformManager: Loading restricted mode set ", lm_list)

        my_converter = ConvertWPtoSurrogateParams
        if 'NRSur4d' in param:
            print(" GENERATING ROM WAVEFORM WITH SPIN PARAMETERS ")
            my_converter = ConvertWPtoSurrogateParamsPrecessing
            reflection_symmetric=False
        if 'NRHyb' in param and not'Tidal' in param:
            print(" GENERATING hybrid ROM WAVEFORM WITH ALIGNED SPIN PARAMETERS ")
            my_converter = ConvertWPtoSurrogateParamsAligned
            self.single_mode_sur=False
        if 'Tidal' in param:
            print(" GENERATING hybrid ROM WAVEFORM WITH ALIGNED SPIN AND TIDAL PARAMETERS ")
            my_converter = ConvertWPtoSurrogateParamsAligned
            self.single_mode_sur=False
        if 'NRSur7d' in param:
            if  rosDebug:
                print(" GENERATING ROM WAVEFORM WITH FULL SPIN PARAMETERS ")
            my_converter = ConvertWPtoSurrogateParamsPrecessingFull
            self.single_mode_sur=False
            reflection_symmetric=False
        # PENDING: General-purpose interface, based on the coordinate string specified. SHOULD look up these names from the surrogate!
        def convert_coords(P):
            vals_out = np.zeros(len(coord_names_internal))
            for indx in np.arange(len(coord_names_internal)):
                vals_out[indx] = P.extract_param( coord_names_internal[indx])
                if coord_names_internal[indx] == 'q':
                    vals_out[indx] = 1./vals_out[indx]
                return vals_out

        raw_modes =[]
        if self.single_mode_sur: #(not 'NRSur7d' in param) and (not 'NRHyb' in param):
            self.sur =  gws.EvaluateSurrogate(dirBaseFiles +'/'+group+param,use_orbital_plane_symmetry=reflection_symmetric, ell_m=None) # lm_list) # straight up filename.  MODIFY to change to use negative modes
                # Modified surrogate import call to load *all* modes all the time
            raw_modes = self.sur.all_model_modes()
            self.modes_available=[]
        elif 'NRHybSur' in param:
            if 'Tidal' in param:
                self.sur = gws.LoadSurrogate(dirBaseFiles +'/'+group,surrogate_name_spliced=param)   # get the dimensinoless surrogate file?
            else:
                self.sur = gws.LoadSurrogate(dirBaseFiles +'/'+group+param)   # get the dimensinoless surrogate file?
            raw_modes = self.sur._sur_dimless.mode_list  # raw modes
            reflection_symmetric = True
            self.modes_available=[]
#            self.modes_available=[(2, 0), (2, 1), (2,-1), (2, 2),(2,-2), (3, 0), (3, 1),(3,-1), (3, 2),(3,-2), (3, 3),(3,-3), (4, 2),(4,-2), (4, 3),(4,-3), (4, 4), (4,-4),(5, 5), (5,-5)]  # see sur.mode_list
            t = self.sur._sur_dimless.domain
            self.ToverMmin = t.min()
            self.ToverMmax = t.max()
            self.ToverM_peak=0   # Need to figure out where this is?  Let's assume it is zero to make my life easier
            # for mode in self.modes_available:
            #     # Not used, bt populate anyways
            #     self.post_dict[mode] = sur_identity
            #     self.post_dict_complex[mode]  = lambda x: x   # to mode
            #     self.post_dict_complex_coef[mode] = lambda x:x  #  to coefficients.
            #     self.parameter_convert[mode] =  my_converter #  ConvertWPtoSurrogateParams   # default conversion routine
#            return
        elif 'NRSur7dq4' in param:
            print(param)
            self.sur = gws.LoadSurrogate(dirBaseFiles +'/'+group+param)   # get the dimensinoless surrogate file?
            raw_modes = self.sur._sur_dimless.mode_list  # raw modes
            reflection_symmetric = False
            self.modes_available=[]
            print(raw_modes)
            self.modes_available=raw_modes
            t = self.sur._sur_dimless.t_coorb
            self.ToverMmin = t.min()
            self.ToverMmax = t.max()
            self.ToverM_peak=0   # Need to figure out where this is?  Let's assume it is zero to make my life easier
            for mode in raw_modes:
            #     # Not used, bt populate anyways
                self.post_dict[mode] = sur_identity
                self.post_dict_complex[mode]  = lambda x: x   # to mode
                self.post_dict_complex_coef[mode] = lambda x:x  #  to coefficients.
                self.parameter_convert[mode] =  my_converter #  ConvertWPtoSurrogateParams   # default conversion routine
            return
        else:
            self.sur = NRSur7dq2.NRSurrogate7dq2()
            reflection_symmetric = False
            self.modes_available = [(2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (3, -3), (3, -2), (3, -1), (3, 0), (3, 1), (3, 2), (3, 3), (4, -4), (4, -3), (4, -2), (4, -1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)];    
            t = self.sur.t_coorb
            self.ToverMmin = t.min()
            self.ToverMmax = t.max()
            self.ToverM_peak=0
            for mode in self.modes_available:
                # Not used, bt populate anyways
                self.post_dict[mode] = sur_identity
                self.post_dict_complex[mode]  = lambda x: x   # to mode
                self.post_dict_complex_coef[mode] = lambda x:x  #  to coefficients.
                self.parameter_convert[mode] =  my_converter #  ConvertWPtoSurrogateParams   # default conversion routine
            return
        # Load surrogates from a mode-by-mode basis, and their conjugates
        for mode in raw_modes:
          if mode[0]<=self.lmax and mode in lm_list:  # latter SHOULD be redundant (because of ell_m=lm_list)
            print(" Loading mode ", mode)
            self.modes_available.append(mode)
            self.post_dict[mode] = sur_identity
            self.post_dict_complex[mode]  = lambda x: x   # to mode
            self.post_dict_complex_coef[mode] = lambda x:x  #  to coefficients.
            self.parameter_convert[mode] =  my_converter #  ConvertWPtoSurrogateParams   # default conversion routine
            if self.single_mode_sur:
                self.sur_dict[mode] = self.sur.single_mode(mode)
                print(' mode ', mode, self.sur_dict[mode].B.shape)
                self.nbasis_per_mode[mode] = (self.sur_dict[mode].B.shape)[1]
            if max_nbasis_per_mode != None  and self.sur_dict[mode].surrogate_mode_type == 'waveform_basis':
             if max_nbasis_per_mode >0: # and max_nbasis_per_mode < self.nbasis_per_mode[mode]:
                # See  https://arxiv.org/pdf/1308.3565v2.pdf  Eqs. 13 - 19
                # Must truncate *orthogonal* basis.
                # Works only for LINEAR basis
                # B are samples of the basis on some long time, V
                sur = self.sur_dict[mode]
                print(" Truncating basis for mode ", mode, " to size ", max_nbasis_per_mode,  " but note the number of EIM points remains the same...")
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
                if rosDebug:
                    print(" Adjoining postprocessing to enable complex conjugate for reflection symmetric case", mode_alt)
#                if max_nbasis_per_mode:
 #                   self.nbasis_per_mode[mode_alt] = np.max([int(max_nbasis_per_mode),1])    # INFRASTRUTCTURE PLAN: Truncate t                print " Loading mode ", mode_alt, " via reflection symmetry "
                self.modes_available.append(mode_alt)
                self.post_dict[mode_alt] = sur_conj
                self.post_dict_complex_coef[mode_alt] = lambda x,l=mode[0]: np.power(-1,l)*np.conj(x)  # beware, do not apply this twice.
                self.post_dict_complex[mode_alt] = np.conj  # beware, do not apply this twice.
                self.parameter_convert[mode_alt] = my_converter
                if self.single_mode_sur: 
                    self.nbasis_per_mode[mode_alt] = self.nbasis_per_mode[mode]
                    self.sur_dict[mode_alt] = self.sur_dict[mode]
        if not self.single_mode_sur:
            # return after performing all the neat reflection symmetrization setup described above, in case model is *not* a single-mode surrogate
            print("  ... done setting mode symmetry requirements", self.modes_available)
#            print raw_modes, self.post_dict
            return  
        # CURRENTLY ONLY LOAD THE 22 MODE and generate the 2,-2 mode by symmetr
        t = self.sur_dict[(2,2)].times  # end time
        self.ToverMmin = t.min()
        self.ToverMmax = t.max()
        P=lalsimutils.ChooseWaveformParams() # default is q=1 object
        params_tmp = self.parameter_convert[(2,2)](P)
        if rosDebug:
            print(" Passing temporary parameters ", params_tmp, " to find the peak time default ")
        # print dir(self.sur_dict[(2,2)])
        # print self.sur_dict[(2,2)].__dict__.keys()
        # print self.sur_dict[(2,2)].parameterization

#        t, hp, hc = self.sur_dict[(2,2)]( **params_tmp  );   # calculate merger time -- addresses conventions on peak time location, and uses named arguments
        t, hp, hc = self.sur_dict[(2,2)]( params_tmp  );   # calculate merger time -- addresses conventions on peak time location, and uses named arguments
        self.ToverM_peak = t[np.argmax(np.abs(hp**2+hc**2))]  # discrete maximum time. Sanity check
        if rosDebug:
            print(" Peak time for ROM ", self.ToverM_peak)

        # BASIS MANAGEMENT: Not yet implemented
        # Assume a DISTINCT BASIS SET FOR AL MODES, which will be ANNOYING
        self.strain_basis_functions_dimensionless_data ={}
        self.strain_basis_functions_dimensionless = self.sur_dict[(2,2)].resample_B  # We may need to add complex conjugate functions too. And a master index for basis functions associated with different modes

    def print_params(self):
        print(" Surrogate model ")
        print("   Modes available ")
        for mode in self.sur_dict:
            print("    " , mode, " nbasis = ", self.nbasis_per_mode[mode])

    # same arguments as hlm
    def complex_hoft(self, P, force_T=False, deltaT=1./16384, time_over_M_zero=0.,sgn=-1):
        hlmT = self.hlmoft(P, force_T, deltaT,time_over_M_zero)
        npts = hlmT[(2,2)].data.length
        wfmTS = lal.CreateCOMPLEX16TimeSeries("h", lal.LIGOTimeGPS(0.), 0., deltaT, lalsimutils.lsu_DimensionlessUnit, npts)
        wfmTS.data.data[:] = 0   # SHOULD NOT BE NECESARY, but the creation operator doesn't robustly clean memory
        wfmTS.epoch = hlmT[(2,2)].epoch
        for mode in hlmT.keys():
            # PROBLEM: Be careful with interpretation. The incl and phiref terms are NOT tied to L.
            if rosDebug:
                print(mode, np.max(hlmT[mode].data.data), " running max ",  np.max(np.abs(wfmTS.data.data)))
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
        htC = self.complex_hoft(P,force_T=1./P.deltaF, deltaT= P.deltaT)  # note P.tref is NOT used in the low-level code
        TDlen  = htC.data.length
        if rosDebug:
            print("Size sanity check ", TDlen, 1/(P.deltaF*P.deltaT))
            print(" Raw complex magnitude , ", np.max(htC.data.data))
            
        # Create working buffers to extract data from it -- wasteful.
        hp = lal.CreateREAL8TimeSeries("h(t)", htC.epoch, 0.,
            P.deltaT, lalsimutils.lsu_DimensionlessUnit, TDlen)
        hc = lal.CreateREAL8TimeSeries("h(t)", htC.epoch, 0.,
            P.deltaT, lalsimutils.lsu_DimensionlessUnit, TDlen)
        hT = lal.CreateREAL8TimeSeries("h(t)", htC.epoch, 0.,
            P.deltaT, lalsimutils.lsu_DimensionlessUnit, TDlen)
        # Copy data components over
        #  - note htC is hp - i hx
        hp.data.data = np.real(htC.data.data)
        hc.data.data = (-1) * np.imag(htC.data.data)
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
                print(" Real h(t) before detector weighting, ", np.max(hp.data.data), np.max(hc.data.data))
            hoft = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc,    # beware, this MAY alter the series length??
                P.phi, P.theta, P.psi, 
                lalsim.DetectorPrefixToLALDetector(str(P.detector)))
            hoft = lal.CutREAL8TimeSeries(hoft, 0, hp.data.length)       # force same length as before??
            if rosDebug:
                print("Size before and after detector weighting " , hp.data.length, hoft.data.length)
        if rosDebug:
            print(" Real h_{IFO}(t) generated, pre-taper : max strain =", np.max(hoft.data.data))
        if P.taper != lalsimutils.lsu_TAPER_NONE: # Taper if requested
            lalsim.SimInspiralREAL8WaveTaper(hoft.data, P.taper)
        if P.deltaF is not None:
            TDlen = int(1./P.deltaF * 1./P.deltaT)
            print("Size sanity check 2 ", int(1./P.deltaF * 1./P.deltaT), hoft.data.length)
            assert TDlen >= hoft.data.length
            npts = hoft.data.length
            hoft = lal.ResizeREAL8TimeSeries(hoft, 0, TDlen)
            # Zero out the last few data elements -- NOT always reliable for all architectures; SHOULD NOT BE NECESSARY
            hoft.data.data[npts:TDlen] = 0

        if rosDebug:
            print(" Real h_{IFO}(t) generated : max strain =", np.max(hoft.data.data))
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
        # This SHOULD use information from the ROM
        return 2*self.fMin/(MsunInSec*(P.m1+P.m2)/lal.MSUN_SI)

    def estimateDurationSec(self,P,fmin=10.):
        """
        estimateDuration uses fmin*M from the (2,2) mode to estimate the waveform duration from the *well-posed*
        part.  By default it uses the *entire* waveform duration.
        CURRENTLY DOES NOT IMPLEMENT frequency-dependent duration
        """
        return (self.ToverMmax - self.ToverMmin)*MsunInSec*(P.m1+P.m2)/lal.MSUN_SI
        return None

    def basis_oft(self,  P, force_T=False, deltaT=1./16384, time_over_M_zero=0.,return_numpy=False):
        m_total_s = MsunInSec*(P.m1+P.m2)/lal.MSUN_SI
        # Create a suitable set of time samples.  Zero pad to 2^n samples.
        T_estimated = np.abs(self.sur_dict[(2,2)].tmin)*m_total_s
        print(" Estimated duration ", T_estimated)
#        T_estimated =  20 # FIXME. Time in seconds
        npts=0
        if not force_T:
            npts_estimated = int(T_estimated/deltaT)
            npts = lalsimutils.nextPow2(npts_estimated)
        else:
            npts = int(force_T/deltaT)
            if rosDebug:
                print(" Forcing length T=", force_T, " length ", npts)
        tvals = (np.arange(npts)-npts/2)*deltaT   # Use CENTERED time to make sure I handle CENTERED NR signal (usual)
        if rosDebug:
            print(" time range being sampled ", [min(tvals),max(tvals)], " corresponding to dimensionless range", [min(tvals)/m_total_s,max(tvals)/m_total_s], " for mtotal ", (P.m1+P.m2)/lal.MSUN_SI)

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
                    print(" Storing resampled copy : ", how_to_store, " expected start and end ", tmin, tmax, " for mtot = ", (P.m1+P.m2)/lal.MSUN_SI)
                basis_grid[how_to_store] = self.post_dict_complex[mode](new_B[:,indx])

        # Optional: return just this way
        if return_numpy:
            return tvals, basis_grid

        # Otherwise, pack into a set of LAL data structures, with epoch etc set appropriately, so we can use in ILE without changes
        ret = {}

        # Taper design.  Note taper needs to be applied IN THE RIGHT PLACE.  
        print(" Tapering basis part 1: start ")
        vectaper = np.zeros(len(tvals))
        t_start_taper = (self.ToverMmin - self.ToverM_peak)*m_total_s#-self.ToverM_peak*m_total_s
        t_end_taper = t_start_taper +100*m_total_s  # taper over around 100 M
        print("  t interval (s) =  ", t_start_taper, t_end_taper)
#        print t_start_taper, t_end_taper, tvals
        def fn_taper(x):
            return np.piecewise(x,[x<t_start_taper,x>t_end_taper,np.logical_and(x>=t_start_taper, x<=t_end_taper)], [0,1,lambda z: 0.5-0.5*np.cos(np.pi* (z-t_start_taper)/(t_end_taper-t_start_taper))])
        vectaper= fn_taper(tvals)

        print(" Tapering basis part 2: end ")  # IMPORTANT: ROMs can be discontinuous at end, which casues problems
        vectaper2 = np.zeros(len(tvals))
        t_start_taper = (self.ToverMmax-self.ToverM_peak-10)*m_total_s  # taper over last 5 M
        t_end_taper = (self.ToverMmax-self.ToverM_peak)*m_total_s  
        print("  t interval (s) =  ", t_start_taper, t_end_taper)
#        print t_start_taper, t_end_taper, tvals
        def fn_taper2(x):
            return np.piecewise(x,[x<t_start_taper,x>t_end_taper,np.logical_and(x>=t_start_taper, x<=t_end_taper)], [1,0,lambda z: 0.5+0.5*np.cos(np.pi* (z-t_start_taper)/(t_end_taper-t_start_taper))])
        vectaper2= fn_taper2(tvals)

        for ind in basis_grid:
            wfmTS = lal.CreateCOMPLEX16TimeSeries("h", lal.LIGOTimeGPS(0.), 0., deltaT, lalsimutils.lsu_DimensionlessUnit, npts)
            wfmTS.epoch = tvals[0]
            wfmTS.data.data = vectaper*basis_grid[ind]
            wfmTS.data.data *= vectaper2
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
                print(" passing params to mode : ", mode, params)
                print(" surrogate natural parameter is ", params_surrogate)

            # New version: gw-surrogate-0.5
            h_EIM = self.sur_dict[mode].eim_coeffs(params_surrogate, 'waveform_basis')
            # OLD VERSION: gw-surrogate-0.4.2 and earlier
            # x0= self.sur_dict[mode]._affine_mapper(params_surrogate)
            # amp_eval = self.sur_dict[mode]._amp_eval(x0)
            # phase_eval = self.sur_dict[mode]._phase_eval(x0)
            # norm_eval = self.sur_dict[mode]._norm_eval(x0)
            # h_EIM = np.zeros(len(amp_eval))
            # if self.sur_dict[mode].surrogate_mode_type  == 'waveform_basis':
            #     h_EIM = norm_eval*amp_eval*np.exp(1j*phase_eval)


            for indx in np.arange(self.nbasis_per_mode[mode]):  
                how_to_store = (mode[0], mode[1], indx)
                coefs[how_to_store]  = self.post_dict_complex_coef[mode](h_EIM[indx])   # conjugation as needed

        return coefs

    # See NR code 
    def hlmoft(self,  P, force_T=False, deltaT=1./16384, time_over_M_zero=0.,use_basis=False,Lmax=np.inf,hybrid_time=None,hybrid_use=False,hybrid_method='taper_add',hybrid_frequency=None,verbose=False,rom_taper_start=False,rom_taper_end=True,use_reference_spins=False):
        """
        hlmoft uses the dimensionless ROM basis functions to extract hlm(t) in physical units, in a LAL array.
        The argument 'P' is a ChooseWaveformParaams object
        FIXME: Add tapering option!

        rom_taper_start  # use manual tapering in this routine.  Should taper if this is true, OR the P.taper includes start tapering
        rom_taper_end  # use gwsurrogate built-in tapering as appropriate (HybSur only), with hard-coded taper time from scott (40M out of 140 M)
        """
        hybrid_time_viaf = hybrid_time
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
            if rosDebug:
                print(" Forcing length T=", force_T, " length ", npts)
        tvals = (np.arange(npts)-npts/2)*deltaT   # Use CENTERED time to make sure I handle CENTERED NR signal (usual)
        if rosDebug:
            print(" time range being sampled ", [min(tvals),max(tvals)], " corresponding to dimensionless range", [min(tvals)/m_total_s,max(tvals)/m_total_s])

        # Populate basis coefficients, if we need it
        if use_basis:
            bT = self.basis_oft(P, force_T=force_T, deltaT=deltaT, time_over_M_zero=time_over_M_zero)
            coefs = self.coefficients(P)


        # extract arguments
        q = P.m1/P.m2  # NOTE OPPOSITE CONVENTION ADOPTED BY SURROGATE GROUP. Code WILL FAIL if q<1
        if q<1:
            q=1./q  # this flips the objects, swapping phase, but it is a good place to start

        # Option 0: Use NRSur7dsq approach (i.e., generate an hlmoft dictionary)
        hlmT_dimensionless={}
        if 'NRSur7dq2' in self.param:
            params_here = self.parameter_convert[(2,2)](P)
            tvals_dimensionless= tvals/m_total_s + self.ToverM_peak
            indx_ok = np.logical_and(tvals_dimensionless  > self.ToverMmin , tvals_dimensionless < self.ToverMmax)
            hlmT ={}
            taper_end_duration =None
            if rom_taper_end:
                taper_end_duration =40.0
            if P.fref >0 and use_reference_spins:
                hlmT_dimensionless_narrow = self.sur(params_here[0], params_here[1],params_here[2],f_ref=P.fref, MTot=(P.m1+P.m2)/lal.MSUN_SI, t=tvals_dimensionless[indx_ok]*m_total_s) #,f_low=0)   
            else:
                hlmT_dimensionless_narrow = self.sur(params_here[0], params_here[1],params_here[2],t=tvals_dimensionless[indx_ok]) #,f_low=0)
            for mode in self.modes_available:
                hlmT_dimensionless[mode] = np.zeros(len(tvals_dimensionless),dtype=complex)
                hlmT_dimensionless[mode][indx_ok] = hlmT_dimensionless_narrow[mode]
           
        if 'NRSur7dq4' in self.param:
            print(self.sur)
            params_here = self.parameter_convert[(2,2)](P)
            tvals_dimensionless= tvals/m_total_s + self.ToverM_peak
            indx_ok = np.logical_and(tvals_dimensionless  > self.ToverMmin , tvals_dimensionless < self.ToverMmax)
            hlmT ={}
            taper_end_duration =None
            if rom_taper_end:
                taper_end_duration =40.0
                print(params_here[0],params_here[1],params_here[2])
            if P.fref >0 and use_reference_spins:
                time,hlmT_dimensionless_narrow,dym = self.sur(params_here[0], params_here[1],params_here[2],f_ref=P.fref, MTot=(P.m1+P.m2)/lal.MSUN_SI, times=tvals_dimensionless[indx_ok]*m_total_s,f_low=0,taper_end_duration=taper_end_duration) #,f_low=0)
            else:
                time,hlmT_dimensionless_narrow,dym = self.sur(params_here[0],params_here[1],params_here[2],times=tvals_dimensionless[indx_ok],f_low=0,taper_end_duration=taper_end_duration)
            for mode in self.modes_available:
                hlmT_dimensionless[mode] = np.zeros(len(tvals_dimensionless),dtype=complex)
                hlmT_dimensionless[mode][indx_ok] = hlmT_dimensionless_narrow[mode]
        # Option 1: Use NRHybXXX approach (i.e., generate an hlmoft dictionary...but with its OWN time grid and scaling...very annoying)
        if 'NRHyb' in self.param:
            params_here = self.parameter_convert[(2,2)](P)
            f_low = P.fmin  # need to convert to dimensionless time
            tvals_dimensionless= tvals/m_total_s + self.ToverM_peak
            indx_ok = np.logical_and(tvals_dimensionless  > self.ToverMmin , tvals_dimensionless < self.ToverMmax)
            hlmT ={}
            taper_end_duration =None
            if rom_taper_end:
                taper_end_duration =40.0
            if 'Tidal' in self.param:
                time, hlmT_dimensionless_narrow,dym = self.sur(params_here[0],params_here[1],params_here[2],times=tvals_dimensionless[indx_ok],taper_end_duration=taper_end_duration,f_low=None,tidal_opts=params_here[5])
            else:
                time, hlmT_dimensionless_narrow,dym = self.sur(params_here[0],params_here[1],params_here[2],times=tvals_dimensionless[indx_ok],taper_end_duration=taper_end_duration,f_low=0)
#            hlmT_dimensionless_narrow = self.sur(params_here[0],params_here[1],params_here[2],dt=P.deltaT,f_low=0,mode_list=self.modes_available,M=params_here[3],dist_mpc=params_here[4],tidal_opts=params_here[5])
#            hlmT_dimensionless_narrow = self.sur(params_here,times=tvals_dimensionless[indx_ok],f_low=0,taper_end_duration=taper_end_duration)
            # Build taper for start
            taper_start_window = np.ones(len(hlmT_dimensionless_narrow[(2,2)]))
            if rom_taper_start or P.taper  != lalsimutils.lsu_TAPER_NONE:
                print(" HybSur: Preparing manual tapering for first ~ 12 cycles ")
                # Taper the start of the waveform by considering a fixed number of cyles for 2,2 mode
                # Taper window must be applied later on, as we want to measure the starting frequency
                # mode-by-mode
                fM_start_22_mode = gwtools.find_instant_freq(np.real(hlmT_dimensionless_narrow[(2,2)]),\
                    np.imag(hlmT_dimensionless_narrow[(2,2)]),tvals_dimensionless[indx_ok])
                taper_start_duration = (2.0 * np.pi) / np.abs(fM_start_22_mode) # about (2pi)^2 radians from 22 mode
                taper_start_window = gwtools.gwutils.window(tvals_dimensionless[indx_ok],\
                    tvals_dimensionless[indx_ok][0], tvals_dimensionless[indx_ok][0]+taper_start_duration,\
                    rolloff=0, windowType='planck')
                taper_fraction = taper_start_duration/(tvals_dimensionless[indx_ok][-1] - tvals_dimensionless[indx_ok][0])
                print("Taper duration "+str(taper_start_duration)+" (M). Fraction of signal: "+str(taper_fraction))

            for mode in self.modes_available:
                hlmT_dimensionless[mode] = np.zeros(len(tvals_dimensionless),dtype=complex)
                if mode[1]<0 and self.reflection_symmetric:   
                    # Perform reflection symmetry
                    mode_alt = (mode[0],-mode[1])
                    hlmT_dimensionless[mode][indx_ok] = np.power(-1, mode[0])*np.conj( taper_start_window* hlmT_dimensionless_narrow[mode_alt])
                else:
                    hlmT_dimensionless[mode][indx_ok] = taper_start_window* hlmT_dimensionless_narrow[mode]

        # Loop over all modes in the system
        for mode in self.modes_available: # loop over modes
          if mode[0] <= Lmax:
            # Copy into a new LIGO time series object
            wfmTS = lal.CreateCOMPLEX16TimeSeries("h", lal.LIGOTimeGPS(0.), 0., deltaT, lalsimutils.lsu_DimensionlessUnit, npts)
            wfmTS.data.data *=0;  # init - a sanity check


            # Option 1: Use the surrogate functions themselves
            if not use_basis:
                params_here = self.parameter_convert[mode](P)
                # Option 0: Use NRSur7dsq approach (i.e., use the stored hlmT data )
                if  not self.single_mode_sur: # 'NRSur7d' in self.param:
                    wfmTS.data.data = m_total_s/distance_s * hlmT_dimensionless[mode]
                else:
                    t_phys, hp_dim, hc_dim = self.post_dict[mode](*self.sur_dict[mode](q=params_here,samples=tvals/m_total_s + self.ToverM_peak))  # center time values AT PEAK
                    wfmTS.data.data =  m_total_s/distance_s * (hp_dim + 1j*hc_dim)  # ARGH! Is this consistent across surrogates?
                # Zero out data after tmax and before tmin
                wfmTS.data.data[tvals/m_total_s<self.ToverMmin-self.ToverM_peak] = 0
                wfmTS.data.data[tvals/m_total_s>self.ToverMmax-self.ToverM_peak] = 0
                # Apply taper *at times where ROM waveform starts*
                if P.taper:
                    n0 = np.argmin(np.abs(tvals)) # origin of time; this will map to the peak
                    # start taper
                    nstart = n0+ int((self.ToverMmin-self.ToverM_peak)*m_total_s/deltaT)        # note offset for tapering. Assumes time interval longer than requested...for NRHyb, ToverMmin does NOT relate to returned array size, beware!
                    nstart = np.max([0,nstart]) # in principle, waveform returned could be massively longer than available time
                    if nstart >0:
                        ntaper = int( (self.ToverM_peak-self.ToverMmin)*m_total_s/deltaT*0.1)  # SHOULD BE set by a few Hz in time fixed 1% of waveform length
                    else:
                        ntaper = int(0.05*len(tvals))   # force 1% length taper
                    vectaper= 0.5 - 0.5*np.cos(np.pi*np.arange(ntaper)/(1.*ntaper))
                    if rosDebug:
                        print(" Tapering ROM hlm(t) for ", mode, " over range ", nstart, nstart+ntaper, " or time offset ", nstart*deltaT, " and window ", ntaper*deltaT)
                        print(len(vectaper), ntaper, n0,nstart, len(tvals), wfmTS.data.length)
                    wfmTS.data.data[nstart:nstart+ntaper]*=vectaper

                    # end taper
                    ntaper = int( (self.ToverMmax-self.ToverM_peak)*m_total_s/deltaT*0.1)  # SHOULD BE set by a few Hz in time fixed 1% of waveform length
                    nend = n0 + int((self.ToverMmax-self.ToverM_peak)*m_total_s/deltaT)
                    vectaper= 0.5 - 0.5*np.cos(np.pi* (1-np.arange(ntaper)/(1.*ntaper)))
                    wfmTS.data.data[-ntaper+nend:nend]*=vectaper
                    if rosDebug:
                        print(" Tapering ROM hlm(t) end for ", mode, " over range ", nend-ntaper, nend, " or time offset ", nend*deltaT, ntaper*deltaT)


            else:
                # Option 2: reconstruct with basis.  Note we will previously have called the basis function generator.
                # select index list
                indx_list_ok = [indx for indx in coefs.keys()  if indx[0]==mode[0] and indx[1]==mode[1]]
                if rosDebug:
                    print(" To reconstruct ", mode, " using the following basis entries ",  indx_list_ok)
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
                    print(" Tapering ROM hlm(t) for ", mode, " over range ", nstart, nstart+ntaper, " or time offset ", nstart*deltaT, " and window ", ntaper*deltaT)
                    wfmTS.data.data[nstart:nstart+ntaper]*=vectaper

                    # end taper
                    ntaper = int( (self.ToverMmax-self.ToverM_peak)*m_total_s/deltaT*0.1)  # SHOULD BE set by a few Hz in time fixed 1% of waveform length
                    nend = n0 + int((self.ToverMmax-self.ToverM_peak)*m_total_s/deltaT)
                    vectaper= 0.5 - 0.5*np.cos(np.pi* (1-np.arange(ntaper)/(1.*ntaper)))
                    wfmTS.data.data[-ntaper+nend:nend]*=vectaper
                    print(" Tapering ROM hlm(t) end for ", mode, " over range ", nend-ntaper, nend, " or time offset ", nend*deltaT, ntaper*deltaT)



            # Set the epoch for the time series correctly: should have peak near center of series by construction
            # MUST USE TIME CENTERING INFO FROM SURROGATE
            wfmTS.epoch = -deltaT*wfmTS.data.length/2

            # Store the resulting mode
            hlmT[mode] = wfmTS

            # if the 22 mode, use to identify the natural frequency.  Can afford to be a bit sloppy, since only used for hybridization
            if hybrid_use and hybrid_time == None and mode[0]==2 and mode[1]==2 :
                if hybrid_frequency:
                    phase_vals = np.angle(wfmTS.data.data)
                    phase_vals = lalsimutils.unwind_phase(phase_vals)
                    datFreqReduced = (-1)* (np.roll(phase_vals,-1) - phase_vals)/deltaT/(2*np.pi)   # Hopefully this is smooth and monotonic
                    indx_max = np.argmax(np.abs(wfmTS.data.data))
                    t_max_location = tvals[indx_max]
                    indx_ok =  np.logical_and(np.abs(datFreqReduced) < hybrid_frequency*1.001, tvals < t_max_location)  # Use peak time to insure matching occurs early on
                    tvals_ok = tvals[indx_ok]
                    f_ok = datFreqReduced[indx_ok]
                    hybrid_time_viaf = tvals_ok[np.argmax(f_ok)]   # rely on high sampling rate. No interpolation!
#                    if verbose:
#                        print " NR catalog: revising hybridization time from 22 mode  to ", hybrid_time_viaf, " given frequency ", hybrid_frequency

        # hybridize
        if hybrid_use:
            my_hybrid_time = hybrid_time_viaf
#            HackRoundTransverseSpin(self.P) # Hack, sub-optimal
            if my_hybrid_time == None:
                my_hybrid_time = -0.5*self.estimateDurationSec(P)  # note fmin is not used. Note this is VERY conservative
            if verbose:
                print("  hybridization performed for ", self.group, self.param, " at time ", my_hybrid_time)
            P.deltaT = deltaT # sanity
            # HACK: round digits, so I can get a spin-aligned approximant if I need it
            hlmT_hybrid = LALHybrid.hybridize_modes(hlmT,P,hybrid_time_start=my_hybrid_time,hybrid_method=hybrid_method)
            return hlmT_hybrid
        else:
            if rosDebug:
                print(" ------ NO HYBRIDIZATION PERFORMED AT LOW LEVEL (=automatic) for ", self.group, self.param, "----- ")
        return hlmT

    def hlmoff(self, P,force_T=False, deltaT=1./16384, time_over_M_zero=0.,use_basis=False,Lmax=np.inf,**kwargs):
        """
        hlmoff takes fourier transforms of LAL timeseries generated from hlmoft.
        All modes have physical units, appropriate to a physical signal.
        """
        hlmF ={}
        hlmT = self.hlmoft(P,force_T=force_T,deltaT=deltaT,time_over_M_zero=time_over_M_zero,use_basis=use_basis,Lmax=Lmax,**kwargs)
        for mode in hlmT.keys():
            wfmTS=hlmT[mode]
            # Take the fourier transform
            wfmFD = lalsimutils.DataFourier(wfmTS)  # this creates a new f series for *each* call.
            # Store the resulting mode
            hlmF[mode] = wfmFD
        return hlmF

    def conj_hlmoff(self, P,force_T=False, deltaT=1./16384, time_over_M_zero=0.,use_basis=False,Lmax=np.inf,**kwargs):
        """
        conj_hlmoff takes fourier transforms of LAL timeseries generated from hlmoft, but after complex conjugation.
        All modes have physical units, appropriate to a physical signal.
        """
        hlmF ={}
        hlmT = self.hlmoft(P,force_T=force_T,deltaT=deltaT,time_over_M_zero=time_over_M_zero,use_basis=use_basis,Lmax=Lmax,**kwargs)
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

