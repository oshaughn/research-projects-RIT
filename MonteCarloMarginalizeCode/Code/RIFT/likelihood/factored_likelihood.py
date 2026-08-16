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

from __future__ import print_function

log_loud = False


import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lsu  # problem of relative comprehensive import - dangerous due to package name
log_loud = lsu.log_loud
import numpy as np
from RIFT.precision import RiftFloat  # platform-portable replacement for np.float128
try:
  import cupy
  from . import optimized_gpu_tools
  from . import Q_inner_product
  xpy_default=cupy
  junk_to_check_installed = cupy.array(5)  # this will fail if GPU not installed correctly
except:
  print(' no cupy (factored)')
  cupy=np #import numpy as cupy  # make sure pointer is identical
  optimized_gpu_tools=None
  Q_inner_product=None
  xpy_default=np

# Old code
#from SphericalHarmonics_gpu_orig import SphericalHarmonicsVectorized_orig as SphericalHarmonicsVectorized
# New code
from .SphericalHarmonics_gpu import SphericalHarmonicsVectorized


from scipy import interpolate, integrate
my_simps = None
if hasattr(integrate, 'simpson'):
  my_simps = integrate.simpson
else:
  my_simps = integrate.simps  # old name

from scipy import special
from itertools import product, combinations
import math

from .vectorized_lal_tools import ComputeDetAMResponse,TimeDelayFromEarthCenter

import os
if 'PROFILE' not in os.environ:
   def profile(fn):
      return fn

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>, R. O'Shaughnessy"


has_GWS=False  # make sure defined in top-level scope
try:
   if not('RIFT_NO_GWSIGNAL' in os.environ):
        import RIFT.physics.GWSignal as rgws
        has_GWS=True
   else:
        has_GWS=False
except:
        has_GWS=False

if not( 'RIFT_LOWLATENCY'  in os.environ):
 # Dont support external packages in low latency.
 # Resolution order (see RIFT/physics/_nrwf_loader.py):
 #   1. nrcatalog.compat_nrwf  (cleanup_2026 NR catalog package)
 #   2. NRWaveformCatalogManager3  (legacy)
 #   3. None
 # Override with $RIFT_NRWF_BACKEND={auto,new,legacy,none}.
 from RIFT.physics._nrwf_loader import get_nrwf as _rift_get_nrwf
 nrwf, useNR = _rift_get_nrwf()
 if useNR and log_loud:
        backend = "nrcatalog" if getattr(nrwf, "is_using_legacy", None) else "NRWaveformCatalogManager3"
        legacy_note = ""
        if getattr(nrwf, "is_using_legacy", None) is not None:
              legacy_note = " (legacy wrapped)" if nrwf.is_using_legacy() else " (no legacy)"
        print(" factored_likelihood.py : NR backend = " + backend + legacy_note)


 try:
        import RIFT.physics.ROMWaveformManager as romwf
        if log_loud:
          print(" factored_likelihood.py: ROMWaveformManager as romwf")
        useROM=True
        rom_basis_scale = 1.0*1e-21   # Fundamental problem: Inner products with ROM basis vectors/Sh are tiny. Need to rescale to avoid overflow/underflow and simplify comparisons
 except ImportError:
        useROM=False
        if log_loud:
          print(" factored_likelihood.py: - no ROM - ")
        rom_basis_scale =1

 try:
    hasEOB=True
    import RIFT.physics.EOBTidalExternalC as eobwf
 #    import EOBTidalExternal as eobwf
 except:
    hasEOB=False
    if log_loud:
      print(" factored_likelihood: no EOB ")
else:
  hasEOB=False
  useROM=False; rom_basis_scale=1
  useNR=False

distMpcRef = 1000 # a fiducial distance for the template source.
tWindowExplore = [-0.15, 0.15] # Not used in main code.  Provided for backward compatibility for ROS. Should be consistent with t_ref_wind in ILE.
rosDebugMessages = True
rosDebugMessagesDictionary = {}   # Mutable after import (passed by reference). Not clear if it can be used by caling routines
                                                  # BUT if every module has a `PopulateMessagesDictionary' module, I can set their internal copies
rosDebugMessagesDictionary["DebugMessages"] = False
rosDebugMessagesDictionary["DebugMessagesLong"] = False


def internal_hlm_generator(P, 
        Lmax, 
        verbose=True,quiet=False,
        NR_group=None,NR_param=None,
        extra_waveform_kwargs={},
        use_gwsignal=False,
        use_gwsignal_approx=None,
       use_external_EOB=False,nr_lookup=False,nr_lookup_valid_groups=None,no_memory=True,perturbative_extraction=False,perturbative_extraction_full=False,hybrid_use=False,hybrid_method='taper_add',use_provided_strain=False,ROM_group=None,ROM_param=None,ROM_use_basis=False,ROM_limit_basis_size=None,skip_interpolation=False,**kwargs):
    """
    internal_hlm_generator: top-level front end to all waveform generators used.
    Needs to be restructured so it works on a 'hook' basis, so we are not constantly changing the source code
    """
    P.taper =  lsu.lsu_TAPER_START # use to pass to romwf, rgws/gwsignal. Pass through
    if not( ROM_group is None) and not (ROM_param is None):
       # For ROM, use the ROM basis. Note that hlmoff -> basis_off henceforth
       acatHere= romwf.WaveformModeCatalog(ROM_group,ROM_param,max_nbasis_per_mode=ROM_limit_basis_size,lmax=Lmax)
       if ROM_use_basis:
            if hybrid_use:
               # WARNING
               #    - Hybridization is NOT enabled 
                    print(" WARNING: Hybridization will not be applied (obviously) if you are using a ROM basis. ")
            bT = acatHere.basis_oft(P,return_numpy=False,force_T=1./P.deltaF)
            # Fake names, to re-use the code below.  
            hlms = {}
            hlms_conj = {}
            for mode in bT:
              if mode[0]<=Lmax:  # don't report waveforms from modes outside the target L range
                if rosDebugMessagesDictionary["DebugMessagesLong"]:
                        print(" FFT for mode ", mode, bT[mode].data.length, " note duration = ", bT[mode].data.length*bT[mode].deltaT)
                hlms[mode] = lsu.DataFourier(bT[mode])
#                print " FFT for conjugate mode ", mode, bT[mode].data.length
                bT[mode].data.data = np.conj(bT[mode].data.data)
                hlms_conj[mode] = lsu.DataFourier(bT[mode])

                # APPLY SCALE FACTOR
                hlms[mode].data.data *=rom_basis_scale
                hlms_conj[mode].data.data *=rom_basis_scale
       else:
           # enforce tapering of this waveform at start, based on discussion with Aasim
           # this code is modular but inefficient: the waveform is regenerated twice
           hlms = acatHere.hlmoff(P, use_basis=False,deltaT=P.deltaT,force_T=1./P.deltaF,Lmax=Lmax,hybrid_use=hybrid_use,hybrid_method=hybrid_method,**extra_waveform_kwargs)  # Must force duration consistency, very annoying
           hlms_conj = acatHere.conj_hlmoff(P, force_T=1./P.deltaF, use_basis=False,deltaT=P.deltaT,Lmax=Lmax,hybrid_use=hybrid_use,hybrid_method=hybrid_method,**extra_waveform_kwargs)  # Must force duration consistency, very annoying
           mode_list = list(hlms.keys())  # make copy: dictionary will change during iteration
           for mode in mode_list:
                   if no_memory and mode[1]==0 and P.SoftAlignedQ():
                           # skip memory modes if requested to do so. DANGER
                        print(" WARNING: Deleting memory mode in precompute stage ", mode)
                        del hlms[mode]
                        del hlms_conj[mode]
                        continue


    elif use_gwsignal and (has_GWS):  # this MUST be called first, so the P.approx is never tested
        if not quiet:
            print( "  FACTORED LIKELIHOOD WITH hlmoff (GWsignal) " )            
        hlms, hlms_conj = rgws.std_and_conj_hlmoff(P,Lmax,approx_string=use_gwsignal_approx,**extra_waveform_kwargs)

    elif (not nr_lookup) and (not NR_group) and ( P.approx ==lalsim.SEOBNRv2 or P.approx == lalsim.SEOBNRv1 or P.approx==lalsim.SEOBNRv3 or P.approx == lsu.lalSEOBv4 or P.approx ==lsu.lalSEOBNRv4HM or P.approx == lalsim.EOBNRv2 or P.approx == lsu.lalTEOBv2 or P.approx==lsu.lalTEOBv4 ):
        # note: alternative to this branch is to call hlmoff, which will actually *work* if ChooseTDModes is propertly implemented for that model
        #   or P.approx == lsu.lalSEOBNRv4PHM or P.approx == lsu.lalSEOBNRv4P  
        if not quiet:
                print("  FACTORED LIKELIHOOD WITH SEOB ")
        hlmsT = {}
        hlmsT = lsu.hlmoft(P,Lmax)  # do a standard function call NOT anything special; should be wrapped properly now!
        # if P.approx == lalsim.SEOBNRv3:
        #         hlmsT = lsu.hlmoft_SEOBv3_dict(P)  # only 2,2 modes -- Lmax irrelevant
        # else:
        #         if useNR:
        #                 nrwf.HackRoundTransverseSpin(P) # HACK, to make reruns of NR play nicely, without needing to rerun

        #         hlmsT = lsu.hlmoft_SEOB_dict(P, Lmax)  # only 2,2 modes -- Lmax irrelevant
        if not quiet:
                print("  hlm generation complete ")
        if P.approx == lalsim.SEOBNRv3 or  P.deltaF == None: # h_lm(t) should be zero-padded properly inside code
                TDlen = int(1./(P.deltaF*P.deltaT))#TDlen = lsu.nextPow2(hlmsT[(2,2)].data.length)
                if not quiet:
                        print(" Resizing to ", TDlen, " from ", hlmsT[(2,2)].data.length)
                for mode in hlmsT:
                        hlmsT[mode] = lal.ResizeCOMPLEX16TimeSeries(hlmsT[mode],0, TDlen)
                #h22 = hlmsT[(2,2)]
                #h2m2 = hlmsT[(2,-2)]
                #hlmsT[(2,2)] = lal.ResizeCOMPLEX16TimeSeries(h22, 0, TDlen)
                #hlmsT[(2,-2)] = lal.ResizeCOMPLEX16TimeSeries(h2m2, 0, TDlen)
        hlms = {}
        hlms_conj = {}
        for mode in hlmsT:
                if verbose:
                        print(" FFT for mode ", mode, hlmsT[mode].data.length, " note duration = ", hlmsT[mode].data.length*hlmsT[mode].deltaT)
                hlms[mode] = lsu.DataFourier(hlmsT[mode])
                if verbose:
                        print(" -> ", hlms[mode].data.length)
                        print(" FFT for conjugate mode ", mode, hlmsT[mode].data.length)
                hlmsT[mode].data.data = np.conj(hlmsT[mode].data.data)
                hlms_conj[mode] = lsu.DataFourier(hlmsT[mode])
    elif (not (NR_group) or not (NR_param)) and  (not use_external_EOB) and (not nr_lookup) and (not use_gwsignal):
        if not quiet:
                print( "  FACTORED LIKELIHOOD WITH hlmoff (default ChooseTDModes) " )
        # hlms_list = lsu.hlmoff(P, Lmax) # a linked list of hlms
        # if not isinstance(hlms_list, dict):
        #         hlms = lsu.SphHarmFrequencySeries_to_dict(hlms_list, Lmax) # a dictionary
        # else:
        #         hlms = hlms_list
        # hlms_conj_list = lsu.conj_hlmoff(P, Lmax)
        # if not isinstance(hlms_list,dict):
        #         hlms_conj = lsu.SphHarmFrequencySeries_to_dict(hlms_conj_list, Lmax) # a dictionary
        # else:
        #         hlms_conj = hlms_conj_list
        if not('fd_standoff_factor' in extra_waveform_kwargs):
          extra_waveform_kwargs['fd_standoff_factor'] = 0.9  # IMPORTANT to match SimInspiralTD. But allow user to override
        hlms, hlms_conj = lsu.std_and_conj_hlmoff(P,Lmax,**extra_waveform_kwargs)
    elif (nr_lookup or NR_group) and useNR:
	    # look up simulation
	    # use nrwf to get hlmf
        print(" Using NR waveforms ")
        group = None
        param = None
        if nr_lookup:
                compare_dict = {}
                compare_dict['q'] = P.m2/P.m1 # Need to match the template parameter. NOTE: VERY IMPORTANT that P is updated with the event params
                compare_dict['s1z'] = P.s1z
                compare_dict['s1x'] = P.s1x
                compare_dict['s1y'] = P.s1y
                compare_dict['s2z'] = P.s2z
                compare_dict['s2x'] = P.s2x
                compare_dict['s2y'] = P.s2y
                if P.eccentricity is not None and P.eccentricity > 0:
                  compare_dict['eccentricity'] = P.eccentricity
                print(" Parameter matching condition ", compare_dict)
                good_sim_list = nrwf.NRSimulationLookup(compare_dict,valid_groups=nr_lookup_valid_groups)
                if len(good_sim_list)< 1:
                        print(" ------- NO MATCHING SIMULATIONS FOUND ----- ")
                        import sys
                        sys.exit(0)
                        print(" Identified set of matching NR simulations ", good_sim_list)
                try:
                        print("   Attempting to pick longest simulation matching  the simulation  ")
                        MOmega0  = 1
                        good_sim = None
                        for key in good_sim_list:
                                print(key, nrwf.internal_EstimatePeakL2M2Emission[key[0]][key[1]])
                                if nrwf.internal_WaveformMetadata[key[0]][key[1]]['Momega0'] < MOmega0:
                                        good_sim = key
                                        MOmega0 = nrwf.internal_WaveformMetadata[key[0]][key[1]]['Momega0']
                                print(" Picked  ",key,  " with MOmega0 ", MOmega0, " and peak duration ", nrwf.internal_EstimatePeakL2M2Emission[key[0]][key[1]])
                except:
                        good_sim  = good_sim_list[0] # pick the first one.  Note we will want to reduce /downselect the lookup process
                group = good_sim[0]
                param = good_sim[1]
        else:
                group = NR_group
                param = NR_param
        print(" Identified matching NR simulation ", group, param)
        mtot = P.m1 + P.m2
        q = P.m2/P.m1
        # Load the catalog
        wfP = nrwf.WaveformModeCatalog(group, param, \
                                       clean_initial_transient=True,clean_final_decay=True, shift_by_extraction_radius=True,perturbative_extraction_full=perturbative_extraction_full,perturbative_extraction=perturbative_extraction,lmax=Lmax,align_at_peak_l2_m2_emission=True, build_strain_and_conserve_memory=True,use_provided_strain=use_provided_strain)
        # Overwrite the parameters in wfP to set the desired scale
        wfP.P.m1 = mtot/(1+q)
        wfP.P.m2 = mtot*q/(1+q)
        wfP.P.dist =distMpcRef*1e6*lal.PC_SI  # fiducial distance
        wfP.P.approx = P.approx
        wfP.P.deltaT = P.deltaT
        wfP.P.deltaF = P.deltaF
        wfP.P.fmin = P.fmin

        hlms = wfP.hlmoff( deltaT=P.deltaT,force_T=1./P.deltaF,hybrid_use=hybrid_use,hybrid_method=hybrid_method)  # force a window.  Check the time
        hlms_conj = wfP.conj_hlmoff( deltaT=P.deltaT,force_T=1./P.deltaF,hybrid_use=hybrid_use)  # force a window.  Check the time

        #if rosDebugMessages:
        #        print("NR variant: Length check: ",hlms[(2,2)].data.length, first_data.data.length)
        # Remove memory modes (ALIGNED ONLY: Dangerous for precessing spins)
        if no_memory and wfP.P.SoftAlignedQ():
                for key in hlms.keys():
                        if key[1]==0:
                                hlms[key].data.data *=0.
                                hlms_conj[key].data.data *=0.


    elif hasEOB and use_external_EOB:
            print("    Using external EOB interface (Bernuzzi)    ")
            # Code WILL FAIL IF LAMBDA=0
            P.taper = lsu.lsu_TAPER_START
            lambda_crit=1e-3  # Needed to have adequate i/o output 
            if P.lambda1<lambda_crit:
                    P.lambda1=lambda_crit
            if P.lambda2<lambda_crit:
                    P.lambda2=lambda_crit
            if P.deltaT > 1./16384:
                    print(" Bad idea to use such a low sampling rate for EOB tidal ")
            wfP = eobwf.WaveformModeCatalog(P,lmax=Lmax)
            hlms = wfP.hlmoff(force_T=1./P.deltaF,deltaT=P.deltaT)
            # Reflection symmetric
            hlms_conj = wfP.conj_hlmoff(force_T=1./P.deltaF,deltaT=P.deltaT)

            # Code will not make the EOB waveform shorter, so the code can fail if you have insufficient data, later
#            print(" External EOB length check ", hlms[(2,2)].data.length, first_data.data.length, first_data.data.length*P.deltaT)
            print(" External EOB length check (in M) ", end=' ')
            print(" Comparison EOB duration check vs epoch vs window size (sec) ", wfP.estimateDurationSec(),  -hlms[(2,2)].epoch, 1./hlms[(2,2)].deltaF)
#            assert hlms[(2,2)].data.length ==first_data.data.length
            if rosDebugMessagesDictionary["DebugMessagesLong"]:
                    hlmT_ref = lsu.DataInverseFourier(hlms[(2,2)])
                    print(" External EOB: Time offset of largest sample (should be zero) ", hlms[(2,2)].epoch + np.argmax(np.abs(hlmT_ref.data.data))*P.deltaT)
    elif useNR: # NR signal required
        mtot = P.m1 + P.m2
        # Load the catalog
        wfP = nrwf.WaveformModeCatalog(NR_group, NR_param, \
                                           clean_initial_transient=True,clean_final_decay=True, shift_by_extraction_radius=True, 
                                       lmax=Lmax,align_at_peak_l2_m2_emission=True,use_provided_strain=use_provided_strain)
        # Overwrite the parameters in wfP to set the desired scale
        q = wfP.P.m2/wfP.P.m1
        wfP.P.m1 *= mtot/(1+q)
        wfP.P.m2 *= mtot*q/(1+q)
        wfP.P.dist =distMpcRef*1e6*lal.PC_SI  # fiducial distance.

        hlms = wfP.hlmoff( deltaT=P.deltaT,force_T=1./P.deltaF)  # force a window
    else:
            print(" No waveform available ")
            import sys
            sys.exit(0)

    return hlms, hlms_conj

#
# Main driver functions
#
def PrecomputeLikelihoodTerms(event_time_geo, t_window, P, data_dict,
        psd_dict, Lmax, fMax, analyticPSD_Q=False,
        inv_spec_trunc_Q=False, T_spec=0., verbose=True,quiet=False,
        internal_fast_precompute=True,
        NR_group=None,NR_param=None,
        ignore_threshold=1e-4,   # dangerous for peak lnL of 25^2/2~300 : biases
        extra_waveform_kwargs={},
        use_gwsignal=False,
        use_gwsignal_approx=None,
        use_external_EOB=False,nr_lookup=False,nr_lookup_valid_groups=None,no_memory=True,perturbative_extraction=False,perturbative_extraction_full=False,hybrid_use=False,hybrid_method='taper_add',use_provided_strain=False,ROM_group=None,ROM_param=None,ROM_use_basis=False,ROM_limit_basis_size=None,skip_interpolation=False, calibration_realizations=None, calibration_conjugate=False, return_calibration_crossterms=False, calibration_self_term=True):
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
    detectors = list(data_dict.keys())
    first_data = data_dict[detectors[0]]
    rholms = {}
    rholms_intp = {}
    crossTerms = {}
    crossTermsV = {}
    # Per-realization |C_c|^2-weighted cross terms for the fused-calmarg self-term
    # fix (None unless a per-detector calibration_realizations dict is supplied).
    # crossTermsCal[det] is a list of n_cal cross-term dicts (rho_sq_c = <C_c h|C_c h>).
    crossTermsCal = None
    crossTermsCalV = None
    _have_cal = (not (calibration_realizations is None)) and isinstance(calibration_realizations, dict)
    # The (potentially expensive) per-realization |C_c|^2-weighted self-term cross terms
    # are built ONLY when the caller will actually receive them (return_calibration_crossterms)
    # AND wants the complete route (calibration_self_term=True, the default).  So a six-value-
    # API caller that happens to supply calibration realizations never pays the (potentially
    # dominant) SVD + rank integrations just to discard the result, and the cheaper global-norm
    # route (calibration_self_term=False) skips them and falls back to the cal-independent
    # <h|h>.  When the build is skipped the trailing crossTermsCal/crossTermsCalV are None.
    _build_cal_ct = _have_cal and return_calibration_crossterms and calibration_self_term
    if _build_cal_ct:
        crossTermsCal = {}
        crossTermsCalV = {}

    # Compute hlms at a reference distance, distance scaling is applied later
    P.dist = distMpcRef*1e6*lsu.lsu_PC

    if not quiet:
            print("  ++++ Template data being computed for the following binary +++ ")
            P.print_params()
    if use_external_EOB:
            # Mass sanity check "
            if  (P.m1/lal.MSUN_SI)>3 or P.m2/lal.MSUN_SI>3:
                    print(" ----- external EOB code: MASS DANGER ---")
    # Compute all hlm modes with l <= Lmax
    # Zero-pad to same length as data - NB: Assuming all FD data same resolution
    P.deltaF = first_data.deltaF

    # call internal waveform generator
    hlms, hlms_conj = internal_hlm_generator(P,Lmax, verbose=verbose, quiet=quiet,
                                             NR_group=NR_group,NR_param=NR_param, nr_lookup=nr_lookup,nr_lookup_valid_groups=nr_lookup_valid_groups,
                                             no_memory=no_memory,perturbative_extraction=perturbative_extraction,perturbative_extraction_full=perturbative_extraction_full,
                                             hybrid_use=hybrid_use,hybrid_method=hybrid_method,use_provided_strain=use_provided_strain,
                                             ROM_group=ROM_group,ROM_param=ROM_param,ROM_use_basis=ROM_use_basis,ROM_limit_basis_size=ROM_limit_basis_size,
                                             extra_waveform_kwargs=extra_waveform_kwargs,use_gwsignal=use_gwsignal,use_gwsignal_approx=use_gwsignal_approx,
                                             skip_interpolation=skip_interpolation)
                                             

    if not(ignore_threshold is None) and (not ROM_use_basis):
            crossTermsFiducial = ComputeModeCrossTermIP(hlms,hlms, psd_dict[detectors[0]], 
                                                        P.fmin, fMax,
                                                        1./2./P.deltaT, P.deltaF, analyticPSD_Q=analyticPSD_Q, inv_spec_trunc_Q=inv_spec_trunc_Q, T_spec=T_spec,verbose=verbose,same_waveform_Q=internal_fast_precompute)
            theWorthwhileModes =  IdentifyEffectiveModesForDetector(crossTermsFiducial, ignore_threshold, detectors)
            # Make sure worthwhile modes satisfy reflection symmetry! Do not truncate egregiously!
            theWorthwhileModes  = theWorthwhileModes.union(  set([(p,-q) for (p,q) in theWorthwhileModes]))
            print("  Worthwhile modes : ", theWorthwhileModes)
            hlmsNew = {}
            hlmsConjNew = {}
            for pair in theWorthwhileModes:
                    hlmsNew[pair]=hlms[pair]
                    hlmsConjNew[pair] = hlms_conj[pair]
            hlms =hlmsNew
            hlms_conj= hlmsConjNew
            if len(hlms.keys()) == 0:
                    print(" Failure ")
                    import sys
                    sys.exit(0)


    # Print statistics on timeseries provided
    if verbose:
      print(" Mode  npts(data)   npts epoch  epoch/deltaT ")
      for mode in hlms.keys():
        print(mode, first_data.data.length, hlms[mode].data.length, hlms[mode].data.length*P.deltaT, hlms[mode].epoch, hlms[mode].epoch/P.deltaT)

    for det in detectors:
        # This is the event time at the detector
        t_det = ComputeArrivalTimeAtDetector(det, P.phi, P.theta,event_time_geo)
        # The is the difference between the time of the leading edge of the
        # time window we wish to compute the likelihood in, and
        # the time corresponding to the first sample in the rholms
        rho_epoch = data_dict[det].epoch - hlms[list(hlms.keys())[0]].epoch
        t_shift =  float(float(t_det) - float(t_window) - float(rho_epoch))
#        assert t_shift > 0    # because NR waveforms may start at any time, they don't always have t_shift > 0 ! 
        # tThe leading edge of our time window of interest occurs
        # this many samples into the rholms
        N_shift = int( t_shift / P.deltaT + 0.5 )  # be careful about rounding: might be one sample off!
        # Number of samples in the window [t_ref - t_window, t_ref + t_window]
        N_window = int( 2 * t_window / P.deltaT )
        # Compute cross terms < h_lm | h_l'm' >
        crossTerms[det] = ComputeModeCrossTermIP(hlms, hlms, psd_dict[det], P.fmin,
                fMax, 1./2./P.deltaT, P.deltaF, analyticPSD_Q,
                inv_spec_trunc_Q, T_spec,verbose=verbose,same_waveform_Q=internal_fast_precompute)
        crossTermsV[det] = ComputeModeCrossTermIP(hlms_conj, hlms, psd_dict[det], P.fmin,
                fMax, 1./2./P.deltaT, P.deltaF, analyticPSD_Q,
                inv_spec_trunc_Q, T_spec,prefix="V",verbose=verbose,same_waveform_Q=internal_fast_precompute)
        # Compute rholm(t) = < h_lm(t) | d >
        cal_realization = None
        if _have_cal:
          cal_realization=calibration_realizations[det]
        if _build_cal_ct:
          # Per-realization |C_c|^2-weighted cross terms (fused-calmarg self-term fix).
          # U-type uses <h_lm | h_l'm'>_{|C|^2/S}, V-type uses <h_lm^* | h_l'm'>_{|C|^2/S},
          # mirroring crossTerms/crossTermsV.  Build the low-rank |C_c|^2 basis ONCE per
          # detector (SVD; rank ~ n_spline^2/2, NOT n_cal), so each realization's cross
          # terms are a cheap linear combo -- NO per-draw band integral, and the same
          # basis serves both the U and V calls.  See BuildCalibrationSelfTermBasis.
          # (Skipped entirely for the cheaper global-norm route, so its cost is not paid.)
          _cal_IP = lsu.ComplexIP(P.fmin, fMax, 1./2./P.deltaT, P.deltaF, psd_dict[det],
                analyticPSD_Q, inv_spec_trunc_Q, T_spec)
          _cal_basis = BuildCalibrationSelfTermBasis(cal_realization, _cal_IP.weights2side.copy(), verbose=verbose)
          crossTermsCal[det] = CalibrationSelfTermCrossTermsFromBasis(
                _cal_IP, hlms, hlms, _cal_basis, prefix="U", same_waveform_Q=internal_fast_precompute)
          crossTermsCalV[det] = CalibrationSelfTermCrossTermsFromBasis(
                _cal_IP, hlms_conj, hlms, _cal_basis, prefix="V", same_waveform_Q=internal_fast_precompute)
        rholms[det] = ComputeModeIPTimeSeries(hlms, data_dict[det],
                psd_dict[det], P.fmin, fMax, 1./2./P.deltaT, N_shift, N_window,
                analyticPSD_Q, inv_spec_trunc_Q, T_spec, calibration_realizations=cal_realization,
                calibration_conjugate=calibration_conjugate)
#        rhoXX = rholms[det][list(rholms[det].keys())[0]]
        # The vector of time steps within our window of interest
        # for which we have discrete values of the rholms
        # N.B. I don't do simply rho_epoch + t_shift, b/c t_shift is the
        # precise desired time, while we round and shift an integer number of
        # steps of size deltaT
        t = np.arange(N_window) * P.deltaT\
                + float(rho_epoch + N_shift * P.deltaT )
        if verbose:
            print("For detector", det, "...")
            print("\tData starts at %.20g" % float(data_dict[det].epoch))
            print("\trholm starts at %.20g" % float(rho_epoch))
            print("\tEvent time at detector is: %.18g" % float(t_det))
            print("\tInterpolation window has half width %g" % t_window)
            print("\tComputed t_shift = %.20g" % t_shift)
            print("\t(t_shift should be t_det - t_window - t_rholm = %.20g)" %\
                    (t_det - t_window - float(rho_epoch)))
            print("\tInterpolation starts at time %.20g" % t[0])
            print("\t(Should start at t_event - t_window = %.20g)" %\
                    (float(rho_epoch + N_shift * P.deltaT)))
        # The minus N_shift indicates we need to roll left
        # to bring the desired samples to the front of the array
        if not skip_interpolation:
          rholms_intp[det] =  InterpolateRholms(rholms[det], t,verbose=verbose)
        else:
          rholms_intp[det] = None

    guess_snr=None
    if True: 
      # estimate peak snr based on rholms.  
      #   - Note U, Q are scaled to fiducial distance.  Likelihood eqution (Pankow et al) :  -0.25 x^2 U + x Q = L where x=1/d
      #   - complete square, maximize in x gives us estimate.  
      #   - don't do precisely correctly, since we're not using sky location, emission direction, or detector response factors, nor coherence between modes. Add in quadrature
      rho_max = 0
      
      for ifo in rholms:
        rho_max_ifo = 0
        for mode in rholms[ifo]:
          if np.real(crossTerms[ifo][(mode,mode)]) >0:
             # Skip SNR sanity check for modes which don't contribute
             rho_max_ifo= np.max(np.abs(rholms[ifo][mode].data.data))
             delta_here= rho_max_ifo**2/np.real(crossTerms[ifo][(mode,mode)])
             if verbose:
               print("  ... ", ifo, mode,rho_max_ifo,np.real(crossTerms[ifo][(mode,mode)]), delta_here)
             rho_max+= delta_here
      rho_max = np.sqrt(rho_max)/2.3  # guesstimate, keep in mind there are sky location terms, coherence, etc
      print("SNR guess (internal, from det response) ", rho_max,rho_max**2/2)
      guess_snr= rho_max

    # Backward-compatible return.  DEFAULT is the historical 6-tuple, so every existing
    # caller (integrate_likelihood_extrinsic[_batchmode[_lisa]], ile_postproc_add_time,
    # jax_ile, test/*) is unaffected.  Only callers that opt in with
    # return_calibration_crossterms=True get the two trailing fused-calmarg self-term
    # cross-term structures (crossTermsCal, crossTermsCalV; None unless a per-detector
    # calibration_realizations dict was supplied).
    _rest = None if not ROM_use_basis else acatHere  # labels are misleading for use_rom_basis
    if return_calibration_crossterms:
            return rholms_intp, crossTerms, crossTermsV,  rholms, guess_snr, _rest, crossTermsCal, crossTermsCalV
    return rholms_intp, crossTerms, crossTermsV,  rholms, guess_snr, _rest

def ReconstructPrecomputedLikelihoodTermsROM(P,acat_rom,rho_intp_rom,crossTerms_rom, crossTermsV_rom, rho_rom,verbose=True):
        """
        Using a set of ROM coefficients for hlm[lm] = coef[l,m,basis] w[basis], reconstructs <h[lm]|data>, <h[lm]|h[l'm']>
        Requires ROM also be loaded in top level, for simplicity
        """
        # Extract coefficients
        coefs = acat_rom.coefficients(P)
        # Identify available modes
        modelist = acat_rom.modes_available      

        detectors = crossTerms_rom.keys()
        rholms = {}
        rholms_intp = {}
        crossTerms = {}
        crossTermsV = {}

        # Reproduce rholms and rholms_intp
        # Loop over detectors
        for det in detectors:
              rholms[det] ={}
              rholms_intp[det] ={}
              # Loop over available modes
              for mode in modelist:
                # Identify relevant terms in the sum
                indx_list_ok = [indx for indx in coefs.keys()  if indx[0]==mode[0] and indx[1]==mode[1]]
                # Discrete case: 
                #   - Create data structure to hold it
                indx0 = indx_list_ok[0]
                rhoTS = lal.CreateCOMPLEX16TimeSeries("rho",rho_rom[det][indx0].epoch,rho_rom[det][indx0].f0,rho_rom[det][indx0].deltaT,rho_rom[det][indx0].sampleUnits,rho_rom[det][indx0].data.length)
                rhoTS.data.data = np.zeros( rho_rom[det][indx0].data.length)   # problems with data initialization common with LAL
                #  - fill the data structure
                fn_list_here = []
                wt_list_here = []
                for indx in indx_list_ok:
                        rhoTS.data.data+= np.conj(coefs[indx])*rho_rom[det][indx].data.data
                        wt_list_here.append(np.conj(coefs[indx]) )
                        fn_list_here = rho_intp_rom[det][indx]
                rholms[det][mode]=rhoTS
                # Interpolated case
                #   - create a lambda structure for it, holding the coefficients.  NOT IMPLEMENTED since not used in production
                if verbose:
                        print(" factored_likelihood: ROM: interpolated timeseries ", det, mode, " NOT CREATED")
                wt_list_here = np.array(wt_list_here)
                rholms_intp[det][mode] = lambda t, fns=fn_list_here, wts=wt_list_here: np.sum(np.array(map(fn_list_here,t))*wt_list_here )
        # Reproduce  crossTerms, crossTermsV
        for det in detectors:
              crossTerms[det] ={}
              crossTermsV[det] ={}
              for mode1 in modelist:
                      indx_list_ok1 = [indx for indx in coefs.keys()  if indx[0]==mode1[0] and indx[1]==mode1[1]]
                      for mode2 in modelist:
                              crossTerms[det][(mode1,mode2)] =0.j
                              indx_list_ok2 = [indx for indx in coefs.keys()  if indx[0]==mode2[0] and indx[1]==mode2[1]]
                              crossTerms[det][(mode1,mode2)] = np.sum(np.array([ np.conj(coefs[indx1])*coefs[indx2]*crossTerms_rom[det][(indx1,indx2)] for indx1 in indx_list_ok1 for indx2 in indx_list_ok2]))
                              crossTermsV[det][(mode1,mode2)] = np.sum(np.array([ coefs[indx1]*coefs[indx2]*crossTermsV_rom[det][(indx1,indx2)] for indx1 in indx_list_ok1 for indx2 in indx_list_ok2]))
                              if verbose:
                                      print("       : U populated ", (mode1, mode2), "  = ",crossTerms[det][(mode1,mode2) ])
                                      print("       : V populated ", (mode1, mode2), "  = ",crossTermsV[det][(mode1,mode2) ])
                    
        return rholms_intp, crossTerms, crossTermsV, rholms, None  # Same return pattern as Precompute...


def FactoredLogLikelihood(extr_params, rholms,rholms_intp, crossTerms, crossTermsV,  Lmax,interpolate=True):
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
    # In practice, all detectors have the same set of Ylms selected, so we only compute for a subset
    Ylms = ComputeYlms(Lmax, incl, -phiref, selected_modes=rholms_intp[list(rholms_intp.keys())[0]].keys())

    lnL = 0.
    for det in detectors:
        CT = crossTerms[det]
        CTV = crossTermsV[det]
        F = ComplexAntennaFactor(det, RA, DEC, psi, tref)

        # This is the GPS time at the detector
        t_det = ComputeArrivalTimeAtDetector(det, RA, DEC, tref)
        det_rholms = {}  # rholms evaluated at time at detector
        if (interpolate):
                for key in rholms_intp[det]:
                        func = rholms_intp[det][key]
                        det_rholms[key] = func(float(t_det))
        else:
            # do not interpolate, just use nearest neighbor.
            for key, rhoTS in rholms[det].items():
                tfirst = t_det
                ifirst = int(np.round(( float(tfirst) - float(rhoTS.epoch)) / rhoTS.deltaT) + 0.5)
                det_rholms[key] = rhoTS.data.data[ifirst]


        lnL += SingleDetectorLogLikelihood(det_rholms, CT, CTV,Ylms, F, dist)

    return lnL

def FactoredLogLikelihoodTimeMarginalized(tvals, extr_params, rholms_intp, rholms, crossTerms, crossTermsV, Lmax, interpolate=False):
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
    Ylms = ComputeYlms(Lmax, incl, -phiref, selected_modes=rholms_intp[list(rholms.keys())[0]].keys())

#    lnL = 0.
    lnL = np.zeros(len(tvals),dtype=RiftFloat)
    for det in detectors:
        CT = crossTerms[det]
        CTV = crossTermsV[det]
        F = ComplexAntennaFactor(det, RA, DEC, psi, tref)

        # This is the GPS time at the detector
        t_det = ComputeArrivalTimeAtDetector(det, RA, DEC, tref)
        det_rholms = {}  # rholms evaluated at time at detector
        if ( interpolate ):
            # use the interpolating functions. 
            for key, func in rholms_intp[det].items():
                det_rholms[key] = func(float(t_det)+tvals)
        else:
            # do not interpolate, just use nearest neighbors.
            for key, rhoTS in rholms[det].items():
                tfirst = float(t_det)+tvals[0]
                ifirst = int(np.round(( float(tfirst) - float(rhoTS.epoch)) / rhoTS.deltaT) + 0.5)
                ilast = ifirst + len(tvals)
                det_rholms[key] = rhoTS.data.data[ifirst:ilast]

        lnL += SingleDetectorLogLikelihood(det_rholms, CT, CTV, Ylms, F, dist)

    maxlnL = np.max(lnL)
    return maxlnL + np.log(my_simps(np.exp(lnL - maxlnL), dx=tvals[1]-tvals[0]))


#
# Internal functions
#
def SingleDetectorLogLikelihoodModel( crossTermsDictionary,crossTermsVDictionary, tref, RA,DEC, thS,phiS,psi,  dist, Lmax, det):
    """
    DOCUMENT ME!!!
    """
    global distMpcRef

    crossTerms = crossTermsDictionary[det]
    crossTermsV = crossTermsVDictionary[det]
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
            term2 += F * np.conj(F) * ( crossTerms[(pair1,pair2)])* np.conj(Ylms[pair1]) * Ylms[pair2] + F*F*Ylms[pair1]*Ylms[pair2]*crossTermsV[(pair1,pair2)]  #((-1)**pair1[0])*crossTerms[((pair1[0],-pair1[1]),pair2)]
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
def NetworkLogLikelihoodTimeMarginalized(epoch,rholmsDictionary,crossTerms,crossTermsV, tref, RA,DEC, thS,phiS,psi,  dist, Lmax, detList):
    """
    DOCUMENT ME!!!
    """
    global distMpcRef

    # N.B.: The Ylms are a function of - phiref b/c we are passively rotating
    # the source frame, rather than actively rotating the binary.
    # Said another way, the m^th harmonic of the waveform should transform as
    # e^{- i m phiref}, but the Ylms go as e^{+ i m phiref}, so we must give
    # - phiref as an argument so Y_lm h_lm has the proper phiref dependence
    Ylms = ComputeYlms(Lmax, thS, -phiS, selected_modes = rholmsDictionary[list(rholmsDictionary.keys())[0]].keys())
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
                term2 += F[det] * np.conj(F[det]) * ( crossTerms[det][(pair1,pair2)])* np.conj(Ylms[pair1]) * Ylms[pair2]  \
                    + F[det]*F[det]*Ylms[pair1]*Ylms[pair2]*crossTermsV[det][(pair1,pair2)] #((-1)**pair1[0])*crossTerms[det][((pair1[0],-pair1[1]),pair2)]
#                    + F[det]*F[det]*Ylms[pair1]*Ylms[pair2]*((-1)**pair1[0])*crossTerms[det][((pair1[0],-pair1[1]),pair2)]
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
def NetworkLogLikelihoodPolarizationMarginalized(epoch,rholmsDictionary,crossTerms, crossTermsV, tref, RA,DEC, thS,phiS,psi,  dist, Lmax, detList):
    """
    DOCUMENT ME!!!
    """
    global distMpcRef

    # N.B.: The Ylms are a function of - phiref b/c we are passively rotating
    # the source frame, rather than actively rotating the binary.
    # Said another way, the m^th harmonic of the waveform should transform as
    # e^{- i m phiref}, but the Ylms go as e^{+ i m phiref}, so we must give
    # - phiref as an argument so Y_lm h_lm has the proper phiref dependence
    Ylms = ComputeYlms(Lmax, thS, -phiS, selected_modes = rholmsDictionary[list(rholmsDictionary.keys())[0]].keys())
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
                term2b += F[det]*F[det]*Ylms[pair1]*Ylms[pair2]*crossTermsV[(pair1,pair2)] #((-1)**pair1[0])*crossTerms[det][((pair1[0],-pair1[1]),pair2)]
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

def SingleDetectorLogLikelihood(rholm_vals, crossTerms,crossTermsV, Ylms, F, dist):
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
    invDistMpc = distMpcRef/distMpc
    Fstar = np.conj(F)

    # Eq. 35 of Richard's notes
    term1 = 0.
#    for mode in rholm_vals:
    for mode, Ylm in Ylms.items():
        term1 += Fstar * np.conj( Ylms[mode]) * rholm_vals[mode]
    term1 = np.real(term1) *invDistMpc

    # Eq. 26 of Richard's notes
    term2 = 0.
    for pair1 in rholm_vals:
        for pair2 in rholm_vals:
            term2 += F * np.conj(F) * ( crossTerms[(pair1,pair2)])* np.conj(Ylms[pair1]) * Ylms[pair2] \
                + F*F*Ylms[pair1]*Ylms[pair2]*crossTermsV[pair1,pair2] #((-1)**pair1[0])* crossTerms[((pair1[0],-pair1[1]),pair2)]
#                + F*F*Ylms[pair1]*Ylms[pair2]*((-1)**pair1[0])* crossTerms[((pair1[0],-pair1[1]),pair2)]
    term2 = -np.real(term2) / 4. /(distMpc/distMpcRef)**2

    return term1 + term2

def ComputeModeIPTimeSeries(hlms, data, psd, fmin, fMax, fNyq,
        N_shift, N_window, analyticPSD_Q=False,
        inv_spec_trunc_Q=False, T_spec=0., calibration_realizations=None,
        calibration_conjugate=False):
    r"""
    Compute the complex-valued overlap between
    each member of a SphHarmFrequencySeries 'hlms'
    and the interferometer data COMPLEX16FrequencySeries 'data',
    weighted the power spectral density REAL8FrequencySeries 'psd'.

    The integrand is non-zero in the range: [-fNyq, -fmin] union [fmin, fNyq].
    This integrand is then inverse-FFT'd to get the inner product
    at a discrete series of time shifts.

    Returns a SphHarmTimeSeries object containing the complex inner product
    for discrete values of the reference time tref.  The epoch of the
    SphHarmTimeSeries object is set to account for the transformation
    """
    rholms = {}
    assert data.deltaF == hlms[list(hlms.keys())[0]].deltaF
    assert data.data.length == hlms[list(hlms.keys())[0]].data.length
    deltaT = 1./(2*fNyq)

    # Create an instance of class to compute inner product time series
    IP = lsu.ComplexOverlap(fmin, fMax, fNyq, data.deltaF, psd,
            analyticPSD_Q, inv_spec_trunc_Q, T_spec, full_output=True)

    # Loop over modes and compute the overlap time series
    if calibration_realizations is None:
      for pair in hlms.keys():
        rho, rhoTS, rhoIdx, rhoPhase = IP.ip(hlms[pair], data)
        rhoTS.epoch = data.epoch - hlms[pair].epoch
#        rholms[pair] = lal.CutCOMPLEX16TimeSeries(rhoTS, N_shift, N_window)  # Warning: code currently fails w/o this cut.
        tmp= lsu.DataRollBins(rhoTS, N_shift)  # restore functionality for bidirectional shifts: waveform need not start at t=0
        rholms[pair] =lal.CutCOMPLEX16TimeSeries(rhoTS, 0, N_window)
    else:
      data_now =  lal.CreateCOMPLEX16FrequencySeries("data", 
                             data.epoch, data.f0, data.deltaF , lsu.lsu_HertzUnit, data.data.length)
      for pair in hlms.keys():
         rholms_so_far = lal.CreateCOMPLEX16TimeSeries("rho", data.epoch, data.f0, deltaT, lal.DimensionlessUnit,  N_window*len(calibration_realizations.T))

         # Create multiple data realizations from the realizations, and construct a longer IP item.
         for index, calib_array in enumerate(calibration_realizations.T):
          #print(calib_array.shape, data.data.length, calibration_realizations.shape)
          # Apply calibration to the DATA (d -> C(f) d).  The data-side cross term
          # kappa = <h|C d> then carries the calibration; the (unweighted) U,V cross
          # terms feeding kappa are cal-independent.  NOTE: the template self-term is
          # NOT -- the complete per-realization norm is rho_sq_c = <C_c h|C_c h>,
          # supplied separately via the |C_c|^2-weighted cross terms
          # (ComputeModeCrossTermIPCal) and used in place of <h|h> in the reduction.
          #
          # Phase-convention caveat (the calmarg self-term-bias analysis note, sec 6):
          # applying C to the DATA gives kappa = <h|C d>, whereas the template-side
          # (bilby) matched filter is <d|C h>.  The IDENTITY <h|conj(C) d> = conj(<d|C h>)
          # means using conj(C) on the data makes |kappa| EXACTLY equal the template-side
          # |<d|C h>| (any phase size) and Re<h|conj(C) d> = Re<d|C h> (correct-SIGN phase
          # tracking).  With plain C on the data the phase tracks the WRONG sign for a
          # phase-unmarginalized or large-phase treatment; under phase marginalization and
          # small phase the two conventions agree to <0.05 nats.  |C|^2 (the self-term
          # weight) is conjugation-invariant, so this only affects kappa, never rho_sq_c.
          # Default keeps the historical C-on-data behavior; set calibration_conjugate=True
          # for the correct-sign / phase-unmarginalized convention.
          _cal_here = np.conj(calib_array) if calibration_conjugate else calib_array
          data_now.data.data = _cal_here * data.data.data
          rho, rhoTS, rhoIdx, rhoPhase = IP.ip(hlms[pair], data_now)
          rhoTS.epoch = data.epoch - hlms[pair].epoch
          tmp= lsu.DataRollBins(rhoTS, N_shift)  # restore functionality for bidirectional shifts: waveform need not start at t=0
          rholms_here = lal.CutCOMPLEX16TimeSeries(rhoTS, 0, N_window)
          # CRITICAL: the concatenated series must carry the SAME epoch as the
          # non-calibration branch (i.e. block 0's per-block epoch, after the roll/cut).
          # Otherwise the time reference is wrong and ifirst lands in the wrong block,
          # zeroing the signal -> the calmarg likelihood collapses.  The within-block
          # offset (ifirst) is identical for every block, so block 0's epoch is the
          # reference for all of them.
          if index == 0:
            rholms_so_far.epoch = rholms_here.epoch
          indx_start = index*N_window
          rholms_so_far.data.data[indx_start:indx_start+N_window] = rholms_here.data.data
         rholms[pair] = rholms_so_far
         print(pair, rholms[pair].data.length, rholms[pair].data.length*deltaT, N_window, N_window*deltaT, 1./deltaT) 

    return rholms

def InterpolateRholm(rholm, t,verbose=False):
    h_re = np.real(rholm.data.data)
    h_im = np.imag(rholm.data.data)
    if verbose:
        print("Interpolation length check ", len(t), len(h_re))
    # spline interpolate the real and imaginary parts of the time series
    h_real = interpolate.InterpolatedUnivariateSpline(t, h_re[:len(t)], k=3,ext='zeros')
    h_imag = interpolate.InterpolatedUnivariateSpline(t, h_im[:len(t)], k=3,ext='zeros')
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


def InterpolateRholms(rholms, t,verbose=False):
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
        rholm_intp[ mode ] = InterpolateRholm(rholm, t,verbose)

    return rholm_intp

def ComputeModeCrossTermIP(hlmsA, hlmsB, psd, fmin, fMax, fNyq, deltaF, 
        analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0., verbose=True,prefix="U",same_waveform_Q=False):
    """
    Compute the 'cross terms' between waveform modes, i.e.
    < h_lm | h_l'm' >.
    The inner product is weighted by power spectral density 'psd' and
    integrated over the interval [-fNyq, -fmin] union [fmin, fNyq]

    Returns a dictionary of inner product values keyed by tuples of mode indices
    i.e. ((l,m),(l',m'))

    same_waveform_Q:  if true, it will do a loop that assumes (a) the same keys in both loops and (b) the same waveform in both (i.e., conjugate transpose)
        Note we WILL use the 'prefix' quantity to identify if we are computing 'V' and therefore using a different symmetry!
    """
    # Create an instance of class to compute inner product
    IP = lsu.ComplexIP(fmin, fMax, fNyq, deltaF, psd, analyticPSD_Q,
            inv_spec_trunc_Q, T_spec)

    crossTerms = {}

    if same_waveform_Q:
      mode_list = list(hlmsA.keys())
      pairs = combinations(mode_list, 2)  # all pairs, no diagonal terms, sorted order ! 
      # do diagonal
      for mode in mode_list:
        crossTerms[ (mode,mode) ] = IP.ip(hlmsA[mode], hlmsB[mode])
      for mode1, mode2 in pairs:
        crossTerms[ (mode1,mode2) ] = IP.ip(hlmsA[mode1], hlmsB[mode2])
        if prefix == "V":
          crossTerms[ (mode2,mode1) ] = crossTerms[(mode1,mode2)]
        else:
          crossTerms[ (mode2,mode1) ] = np.conj(crossTerms[(mode1,mode2)])
    else:
     for mode1 in hlmsA.keys():
        for mode2 in hlmsB.keys():
            crossTerms[ (mode1,mode2) ] = IP.ip(hlmsA[mode1], hlmsB[mode2])
            if verbose:
                print("       : ", prefix, " populated ", (mode1, mode2), "  = ",\
                        crossTerms[(mode1,mode2) ])

    return crossTerms


_cal_selfterm_basis_cache = {}   # key -> (weakref(cal), basis dict); bounded to a few entries


def _cal_selfterm_key(cal, base_weights2side):
    """Collision-resistant cache key for the SVD self-term basis, which depends on the
    draws |C_c|^2 AND the 1/S band weights (band + values) -- NOT on the template -- so it
    is constant over the intrinsic grid for a fixed draw set + PSD, but MUST change if the
    PSD/band changes.  The band weights are digested in FULL (they are small next to the
    draws, and a strided sample can collide -- two different PSDs agreeing on the sampled
    bins returned a stale basis).  The draws are keyed by object identity + shape/dtype and
    re-verified against the stored weakref on a hit (see BuildCalibrationSelfTermBasis), so
    an id reused after garbage collection cannot return another array's basis."""
    import hashlib
    w = np.ascontiguousarray(base_weights2side, dtype=np.float64)
    wdig = hashlib.blake2b(w.tobytes(), digest_size=16).digest()
    return (id(cal), tuple(cal.shape), str(cal.dtype), wdig)


def BuildCalibrationSelfTermBasis(calibration_realizations, base_weights2side,
        rank_tol=1e-10, rank_max=None, verbose=False, use_cache=True):
    r"""
    Low-rank basis for the per-realization |C_c(f)|^2 profiles used by the fused-
    calmarg self-term fix.  The calibration factors are random SPLINE draws made
    in-flight (per iteration / proposal round), so |C_c|^2 must be handled PER DRAW
    at runtime -- but NOT by a per-draw band integral (that is what this avoids).

    The SVD is the ONE-TIME "expand on a fixed basis" step; it depends only on the
    draws + PSD band (not the template), so it is memoized on a content fingerprint and
    reused across the intrinsic grid (a multi-point ILE job pays the SVD once, then only
    ``rank`` band integrals per point + a cheap per-draw linear combo).

    |C_c(f)|^2 = amp_c(f)^2 with amp_c a cubic spline on ~n_spline (~10) log-f nodes,
    so the SET {|C_c|^2}_c spans a small fixed subspace (dim <= n_spline*(n_spline+1)/2,
    independent of n_cal).  An SVD of the in-band |C_c(f)|^2 matrix recovers an
    orthonormal basis {b_k(f)} of that subspace and the per-draw coefficients:
        |C_c(f)|^2 = sum_k  alpha[k,c]  b_k(f)   (exact to rank_tol).
    Then the per-realization weighted cross terms are a cheap linear combination of
    a FIXED, once-computed set of basis-weighted cross-term tensors (see
    CalibrationSelfTermCrossTermsFromBasis); no band integral is redone per draw, and
    a freshly drawn C_c only needs its coefficients alpha_c = B^+ |C_c|^2 (a matmul
    against the cached basis), not a new integral.

    base_weights2side : ComplexIP.weights2side (two-sided 1/S, zero out of band).
        The band (nonzero-weight) support defines where |C_c|^2 must be represented;
        outside it the integrand h_lm* h_l'm'/S vanishes, so |C_c|^2 there is irrelevant.

    Returns a dict with:
        weights2side : (rank, len2side) float64 -- base_weights2side * b_k(f) (0 out of band),
                       ready to drop into ComplexIP.weights2side.
        alpha        : (rank, n_cal) float64   -- per-draw expansion coefficients.
        rank, resid, n_cal.
    """
    base_weights2side = np.asarray(base_weights2side, dtype=np.float64)
    cal = np.asarray(calibration_realizations)
    n_cal = cal.shape[1]
    if use_cache:
        _key = _cal_selfterm_key(cal, base_weights2side)
        _hit = _cal_selfterm_basis_cache.get(_key)
        if _hit is not None:
            _ref, _basis = _hit
            if _ref() is cal:            # exact identity: guards id-reuse after GC
                return _basis
            _cal_selfterm_basis_cache.pop(_key, None)   # stale id -> drop and rebuild
    band = base_weights2side != 0.0
    absC2_band = (np.abs(cal[band, :]) ** 2).astype(np.float64)     # (n_band, n_cal)
    # economy SVD: absC2_band = Ub @ diag(S) @ Vt ; columns live in a low-dim subspace
    Ub, S, Vt = np.linalg.svd(absC2_band, full_matrices=False)
    keep = S > (rank_tol * (S[0] if S.size else 1.0))
    rank = int(np.count_nonzero(keep))
    if rank_max is not None:
        rank = min(rank, int(rank_max))
    rank = max(rank, 1)
    b_band = Ub[:, :rank]                                            # (n_band, rank)
    alpha = (S[:rank, None] * Vt[:rank, :])                          # (rank, n_cal)
    resid = float(np.max(np.abs(b_band @ alpha - absC2_band))) if n_cal else 0.0
    # embed each basis vector into the full two-sided grid, pre-multiplied by 1/S
    w2 = np.zeros((rank, base_weights2side.shape[0]), dtype=np.float64)
    bw_band = base_weights2side[band]
    for k in range(rank):
        w2[k, band] = bw_band * b_band[:, k]
    if verbose:
        print("       : cal self-term basis rank=%d / n_cal=%d  (recon resid=%.2e)"
              % (rank, n_cal, resid))
    basis = dict(weights2side=w2, alpha=alpha, rank=rank, resid=resid, n_cal=n_cal)
    if use_cache:
        import weakref
        if len(_cal_selfterm_basis_cache) >= 8:      # bound memory: evict an arbitrary entry
            _cal_selfterm_basis_cache.pop(next(iter(_cal_selfterm_basis_cache)))
        # store a weakref to the draws so a hit can re-verify identity (see lookup above)
        _cal_selfterm_basis_cache[_key] = (weakref.ref(cal), basis)
    return basis


def CalibrationSelfTermCrossTermsFromBasis(IP, hlmsA, hlmsB, cal_basis,
        prefix="U", same_waveform_Q=False):
    r"""
    Per-realization |C_c|^2-weighted mode cross terms (fused-calmarg self-term fix),
    formed from the pre-built low-rank basis (BuildCalibrationSelfTermBasis) WITHOUT a
    per-draw band integral.

    For each of the ``rank`` basis functions we compute the basis-weighted cross-term
    tensor  U^(k)_{lm,l'm'} = 2 dF sum_f  b_k(f)/S(f)  h_lm*(f) h_l'm'(f)  (rank band
    integrals, ONCE), then each realization's cross terms are the cheap linear combo
        U_c = sum_k  alpha[k,c]  U^(k).
    Returns a LIST of n_cal cross-term dicts, keyed exactly like ComputeModeCrossTermIP.

    Because |C_c(f)|^2 = sum_k alpha[k,c] b_k(f) exactly (to the basis rank_tol), this
    reproduces the per-realization <C_c h|C_c h> to that tolerance.  alpha is real, so
    the U-hermitian / V-symmetric structure of each U^(k) (real weight) carries to U_c.
    """
    w2 = cal_basis["weights2side"]
    alpha = cal_basis["alpha"]
    rank = cal_basis["rank"]
    n_cal = cal_basis["n_cal"]
    mode_list = list(hlmsA.keys())
    pairs_upper = list(combinations(mode_list, 2))
    base_w2 = IP.weights2side  # preserve to restore

    # rank band integrals per (upper-triangular) mode pair, ONCE
    Uk = {}   # (mode1,mode2) -> (rank,) complex
    diag_keys = [(m, m) for m in mode_list]
    for key in diag_keys + list(pairs_upper):
        vals = np.empty(rank, dtype=np.complex128)
        m1, m2 = key
        for k in range(rank):
            IP.weights2side = w2[k]
            vals[k] = IP.ip(hlmsA[m1], hlmsB[m2])
        Uk[key] = vals
    IP.weights2side = base_w2   # restore

    out_list = []
    for c in range(n_cal):
        a_c = alpha[:, c]
        crossTerms = {}
        if same_waveform_Q:
            for key in diag_keys:
                crossTerms[key] = complex(np.dot(a_c, Uk[key]))
            for (m1, m2) in pairs_upper:
                val = complex(np.dot(a_c, Uk[(m1, m2)]))
                crossTerms[(m1, m2)] = val
                crossTerms[(m2, m1)] = val if prefix == "V" else np.conj(val)
        else:
            # non-fast path: fill every ordered pair (compute missing lower-tri tensors)
            for m1 in mode_list:
                for m2 in mode_list:
                    if (m1, m2) not in Uk:
                        vals = np.empty(rank, dtype=np.complex128)
                        for k in range(rank):
                            IP.weights2side = w2[k]
                            vals[k] = IP.ip(hlmsA[m1], hlmsB[m2])
                        Uk[(m1, m2)] = vals
                    crossTerms[(m1, m2)] = complex(np.dot(a_c, Uk[(m1, m2)]))
            IP.weights2side = base_w2
        out_list.append(crossTerms)
    return out_list


def ComputeModeCrossTermIPCal(hlmsA, hlmsB, psd, fmin, fMax, fNyq, deltaF,
        calibration_realizations,
        analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0., verbose=False,
        prefix="U", same_waveform_Q=False, cal_basis=None):
    r"""
    Per-realization |C_c(f)|^2-weighted mode cross terms for the fused-calmarg
    self-term fix (rho_sq_c = <C_c h | C_c h>; the calmarg self-term-bias analysis note).

    Uses the low-rank basis expansion (BuildCalibrationSelfTermBasis +
    CalibrationSelfTermCrossTermsFromBasis): rank (<= n_spline*(n_spline+1)/2, NOT n_cal)
    band integrals ONCE, then each realization's U_c is a cheap linear combination -- no
    per-draw band integral.  Pass a pre-built ``cal_basis`` (dict) to share the SVD
    across the U and V calls; otherwise it is built here from the local IP weights.

    Returns a LIST of n_cal cross-term dicts, keyed like ComputeModeCrossTermIP.
    """
    IP = lsu.ComplexIP(fmin, fMax, fNyq, deltaF, psd, analyticPSD_Q,
            inv_spec_trunc_Q, T_spec)
    cal = np.asarray(calibration_realizations)
    assert cal.shape[0] == IP.len2side, \
        "calibration realization length %d != IP.len2side %d" % (cal.shape[0], IP.len2side)
    if cal_basis is None:
        cal_basis = BuildCalibrationSelfTermBasis(cal, IP.weights2side.copy(), verbose=verbose)
    return CalibrationSelfTermCrossTermsFromBasis(
        IP, hlmsA, hlmsB, cal_basis, prefix=prefix, same_waveform_Q=same_waveform_Q)


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

def ComputeYlms(Lmax, theta, phi, selected_modes=None):
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
            if selected_modes is not None and (l,m) not in selected_modes:
                continue
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


def ComputeArrivalTimeAtDetectorWithoutShift(det, RA, DEC, tref):
    """
    Function to compute the time of arrival at a detector
    from the time of arrival at the geocenter.

    'det' is a detector prefix string (e.g. 'H1')
    'RA' and 'DEC' are right ascension and declination (in radians)
    'tref' is the reference time at the geocenter.  It can be either a float (in which case the return is a float) or a GPSTime object (in which case it returns a GPSTime)
    """
    detector = lalsim.DetectorPrefixToLALDetector(det)
    print(detector, detector.location)
    # if tref is a float or a GPSTime object,
    # it shoud be automagically converted in the appropriate way
    return lal.TimeDelayFromEarthCenter(detector.location, RA, DEC, tref)


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
        print(" +++ Injection creation for detector ", P.detector, " ++ ")
        print("   : Creating signal for injection with epoch ", float(hp.epoch), " and event time centered at ", lsu.stringGPSNice(P.tref))
        Fp, Fc = lal.ComputeDetAMResponse(lalsim.InstrumentNameToLALDetector(str(P.detector)).response, P.phi, P.theta, P.psi, lal.GreenwichMeanSiderealTime(hp.epoch))
        print("  : creating signal for injection with (det, t,RA, DEC,psi,Fp,Fx)= ", P.detector, float(P.tref), P.phi, P.theta, P.psi, Fp, Fc)
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


# def rollTimeSeries(series_dict, nRollRight):
#         """
#         rollTimeSeries
#         Deprecated -- see DataRollBins in lalsimutils.py
#         """
#     # Use the fact that we pass by value and that we swig bind numpy arrays
#     for det in series_dict:
#         print " Rolling timeseries ", nRollRight
#         np.roll(series_dict.data.data, nRollRight)

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
    r"""
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
                term2 += F[det] * np.conj(F[det]) * ( crossTerms[det][(pair1,pair2)])* np.conj(Ylms[pair1]) * Ylms[pair2] + F[det]*F[det]*Ylms[pair1]*Ylms[pair2]*crossTermsV[(pair1,pair2)] #((-1)**pair1[0])*crossTerms[det][((pair1[0],-pair1[1]),pair2)]
    term2 = -np.real(term2) / 4. /(distMpc/distMpcRef)**2


    print(detList)
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


def IdentifyEffectiveModesForDetector(crossTermsOneDetector, fac,det):
    # extract a list of possible pairs
    pairsOfPairs = crossTermsOneDetector.keys()
    pairsUnion = []
    for x in pairsOfPairs:
        pairsUnion.append(x[1])
    pairsUnion = set(pairsUnion)  # a list of unique pairs that occur in crossTerms' first index
    
    # Find modes which are less effective
    pairsIneffective = []
    for pair in pairsUnion:
        isEffective = False
        threshold = crossTermsOneDetector[((2,2),(2,2))]*fac
        for pair2 in pairsUnion:
            if crossTermsOneDetector[(pair,pair2)] > threshold:
                isEffective = True
        if not isEffective:
            pairsIneffective.append(pair)
            if rosDebugMessagesDictionary["DebugMessages"]:
                print("   ", pair, " - no significant impact on U, less than ", threshold)

    return pairsUnion - set(pairsIneffective)

####
#### Reimplementation with arrays   [NOT YET GENERALIZED TO USE V]
####

def PackLikelihoodDataStructuresAsArrays(pairKeys, rholms_intpDictionaryForDetector, rholmsDictionaryForDetector,crossTermsForDetector, crossTermsForDetectorV):
    """
    Accepts list of LM pairs, dictionary for rholms against keys, and cross terms (a dictionary)

    PROBLEM: Different detectors may have different time zeros. User must use the returned arrays with great care.
    """
    #print pairKeys, rholmsDictionaryForDetector
    nKeys  = len(pairKeys)
    keyRef = list(pairKeys)[0]
    npts = rholmsDictionaryForDetector[keyRef].data.length


    ### Step 0: Create two lookup tables: index->pair and pair->index
    lookupNumberToKeys = np.zeros((nKeys,2),dtype=int)
    lookupKeysToNumber = {}
    for indx, val in enumerate(pairKeys):
        lookupNumberToKeys[indx][0]= val[0]  
        lookupNumberToKeys[indx][1]= val[1]  
        lookupKeysToNumber[val] = indx
    # Now create a *second* lookup table, for complex-conjugation-in-time: (l,m)->(l,-m)
    lookupNumberToNumberConjugation = np.zeros(nKeys,dtype=int)
    for indx in np.arange(nKeys):
        l = lookupNumberToKeys[indx][0]
        m = lookupNumberToKeys[indx][1]
        indxOut = lookupKeysToNumber[(l,-m)]
        lookupNumberToNumberConjugation[indx] = indxOut
        
    ### Step 1: Convert crossTermsForDetector explicitly into a matrix
    crossTermsArrayU = np.zeros((nKeys,nKeys),dtype=np.complex128)   # Make sure complex numbers can be stored
    crossTermsArrayV = np.zeros((nKeys,nKeys),dtype=np.complex128)   # Make sure complex numbers can be stored
    for pair1 in pairKeys:
        for pair2 in pairKeys:
            indx1 = lookupKeysToNumber[pair1]
            indx2 = lookupKeysToNumber[pair2]
            crossTermsArrayU[indx1][indx2] = crossTermsForDetector[(pair1,pair2)]
            crossTermsArrayV[indx1][indx2] = crossTermsForDetectorV[(pair1,pair2)]
#            pair1New = (pair1[0], -pair1[1])
#            crossTermsArrayV[indx1][indx2] = (-1)**pair1[0]*crossTermsForDetector[(pair1New,pair2)]   # this actually should be a seperate array in general; we are assuming reflection symmetry to populate it
    if rosDebugMessagesDictionary["DebugMessagesLong"]:
        print(" Built cross-terms matrix ", crossTermsArray)

    ### Step 2: Convert rholmsDictionaryForDetector
    rholmArray = np.zeros((nKeys,npts),dtype=np.complex128)
    for pair1 in pairKeys:
        indx1 = lookupKeysToNumber[pair1]
        rholmArray[indx1][:] = rholmsDictionaryForDetector[pair1].data.data  # Copy the array of time values.

    ### Step 3: Create rholm_intp array-ized structure
    rholm_intpArray = [None] * nKeys   # create a flexible python array of the desired size, to hold function pointers
    if rholms_intpDictionaryForDetector:
        for pair1 in pairKeys:
            indx1 = lookupKeysToNumber[pair1]
            rholm_intpArray[indx1] = rholms_intpDictionaryForDetector[pair1]
            
    ### step 4: create dictionary (one per detector) with epoch  associated with the starting point for that IFO.  (should be the same for all modes for a given IFO)
    epochHere  = float(rholmsDictionaryForDetector[pair1].epoch)
    
    return lookupNumberToKeys,lookupKeysToNumber, lookupNumberToNumberConjugation, crossTermsArrayU,crossTermsArrayV, rholmArray, rholm_intpArray, epochHere


def PackCalCrossTermsAsArrays(pairKeys, lookupKeysToNumber, crossTermsCalList, crossTermsCalListV):
    """Pack the per-realization |C_c|^2-weighted cross terms (fused-calmarg self-term
    fix) into arrays, using the SAME (l,m)->index mapping (lookupKeysToNumber) that
    PackLikelihoodDataStructuresAsArrays produced for the cal-independent U,V.  This
    guarantees ctUArrayDict_cal[det][c] is index-aligned with ctUArrayDict[det].

    crossTermsCalList, crossTermsCalListV : lists of n_cal cross-term dicts (the
        ComputeModeCrossTermIPCal output for one detector).

    Returns (U_cal, V_cal), each (n_cal, nKeys, nKeys) complex128.
    """
    nKeys = len(pairKeys)
    n_cal = len(crossTermsCalList)
    U_cal = np.zeros((n_cal, nKeys, nKeys), dtype=np.complex128)
    V_cal = np.zeros((n_cal, nKeys, nKeys), dtype=np.complex128)
    for c in range(n_cal):
        ctU_c = crossTermsCalList[c]
        ctV_c = crossTermsCalListV[c]
        for pair1 in pairKeys:
            indx1 = lookupKeysToNumber[pair1]
            for pair2 in pairKeys:
                indx2 = lookupKeysToNumber[pair2]
                U_cal[c, indx1, indx2] = ctU_c[(pair1, pair2)]
                V_cal[c, indx1, indx2] = ctV_c[(pair1, pair2)]
    return U_cal, V_cal


def SingleDetectorLogLikelihoodDataViaArray(epoch,lookupNK, rholms_intpArrayDict,tref, RA,DEC, thS,phiS,psi,  dist, det):
    """
    SingleDetectorLogLikelihoodDataViaArray evaluates everything using *arrays* for each (l,m) pair
    Note arguments passed are STILL SCALARS

    DEPRECATED: use DiscreteFactoredLogLikelihoodViaArray for end-to-end uuse
    USED IN : FactoredLogLikelihoodViaArray
    """
    global distMpcRef

    # N.B.: The Ylms are a function of - phiref b/c we are passively rotating
    # the source frame, rather than actively rotating the binary.
    # Said another way, the m^th harmonic of the waveform should transform as
    # e^{- i m phiref}, but the Ylms go as e^{+ i m phiref}, so we must give
    # - phiref as an argument so Y_lm h_lm has the proper phiref dependence
    Ylms = ComputeYlmsArray(lookupNK[det], thS,-phiS)
    if (det == "Fake"):
        F=np.exp(-2.*1j*psi)  # psi is applied through *F* in our model
        tshift= tref - epoch
    else:
        F = ComplexAntennaFactor(det, RA,DEC,psi,tref)
        detector = lalsim.DetectorPrefixToLALDetector(det)
        tshift = ComputeArrivalTimeAtDetector(det, RA,DEC, tref)  -  epoch   # detector time minus reference time (so far)
    rholmsArray = np.array(map( lambda x : complex(x( float(tshift))) , rholms_intpArrayDict[det]),dtype=complex)  # Evaluate interpolating functions at target time
    distMpc = dist/(lal.PC_SI*1e6)

    # Following loop *should* be implemented as an array multiply!
    term1 = 0.j
    term1 = np.dot(np.conj(F*Ylms),rholmsArray)   # be very careful re how this multiplication is done: suitable to use this form of multiply
    term1 = np.real(term1) / (distMpc/distMpcRef)

    return term1

def DiscreteSingleDetectorLogLikelihoodDataViaArray(tvals,extr_params,lookupNK, rholmsArrayDict,Lmax=2,det='H1'):
    """
    SingleDetectorLogLikelihoodDataViaArray evaluates everything using *arrays* for each (l,m) pair.
    Uses discrete arrays.  Compare to  FactoredLogLikelihoodTimeMarginalized
    """
    global distMpcRef


    RA = extr_params.phi
    DEC =  extr_params.theta
    tref = extr_params.tref # geocenter time
    phiref = extr_params.phiref
    incl = extr_params.incl
    psi = extr_params.psi
    dist = extr_params.dist

    npts = len(tvals)
    deltaT = extr_params.deltaT

    Ylms = ComputeYlmsArray(lookupNK[det], incl,-phiref)
    if (det == "Fake"):
        F=np.exp(-2.*1j*psi)  # psi is applied through *F* in our model
        tshift= tref - epoch
    else:
        F = ComplexAntennaFactor(det, RA,DEC,psi,tref)
        detector = lalsim.DetectorPrefixToLALDetector(det)
        t_det = ComputeArrivalTimeAtDetector(det, RA, DEC, tref)
        tshift= t_det - tref
    rhoTS = rholmsArrayDict[det]
    distMpc = dist/(lal.PC_SI*1e6)

    npts = len(rhoTS[0])
    # Following loop *should* be implemented as an array multiply!
    term1 = np.zeros(npts,dtype=complex)
    term1 = np.dot(np.conj(F*Ylms),rhoTS)   # be very careful re how this multiplication is done: suitable to use this form of multiply
    term1 = np.real(term1) / (distMpc/distMpcRef)

    # Apply timeshift *at end*, without loss of generality: this is a single detector. Note no subsample interpolation
    # This timeshift should *only* be applied if all detectors start at the same array index!
    nShiftL =int(np.round(float(tshift)/deltaT))
    # return structure: different shifts
    term1 = np.roll(term1,-nShiftL)

    return term1

def SingleDetectorLogLikelihoodModelViaArray(lookupNKDict,ctUArrayDict,ctVArrayDict, tref, RA,DEC, thS,phiS,psi,  dist,det):
    """
    DOCUMENT ME!!!
    """
    global distMpcRef

    # N.B.: The Ylms are a function of - phiref b/c we are passively rotating
    # the source frame, rather than actively rotating the binary.
    # Said another way, the m^th harmonic of the waveform should transform as
    # e^{- i m phiref}, but the Ylms go as e^{+ i m phiref}, so we must give
    # - phiref as an argument so Y_lm h_lm has the proper phiref dependence
    U = ctUArrayDict[det]
    V = ctVArrayDict[det]
    Ylms = ComputeYlmsArray(lookupNKDict[det], thS,-phiS)
    if (det == "Fake"):
        F=np.exp(-2.*1j*psi)  # psi is applied through *F* in our model
    else:
        F = ComplexAntennaFactor(det, RA,DEC,psi,tref)
    distMpc = dist/(lal.PC_SI*1e6)


    # Term 2 part 1 : conj(Ylms*F)*crossTermsU*F*Ylms
    # Term 2 part 2:  Ylms*F*crossTermsV*F*Ylms
    term2 = 0.j
    term2 += F*np.conj(F)*(np.dot(np.conj(Ylms), np.dot(U,Ylms)))
    term2 += F*F*np.dot(Ylms,np.dot(V,Ylms))
    term2 = np.sum(term2)

    term2 = -np.real(term2) / 4. /(distMpc/distMpcRef)**2
    return term2


def  FactoredLogLikelihoodViaArray(epoch, P, lookupNKDict, rholms_intpArrayDict, ctUArrayDict,ctVArrayDict):
    """
    FactoredLogLikelihoodViaArray uses the array-ized data structures to compute the log likelihood, a single scalar value.
    This generally is marginally faster, particularly if Lmax is large.
    The timeseries quantities are computed via interpolation onto the desired grid

    Speed-wise, because we extract a *single* scalar value, this code has the same efficiency as FactoredLogLikelihoood

    Note 'P' must have the *sampling rate* set to correctly interpret the event time.
    Note arguments passed are STILL SCALARS
    """
    global distMpcRef

    detectors = rholms_intpArrayDict.keys()

    RA = P.phi
    DEC =  P.theta
    tref = P.tref # geocenter time
    phiref = P.phiref
    incl = P.incl
    psi = P.psi
    dist = P.dist

    deltaT = P.deltaT

    term1 = 0.
    term2 = 0.
    for det in detectors:
        term1 += SingleDetectorLogLikelihoodDataViaArray(epoch,lookupNKDict, rholms_intpArrayDict,tref, RA, DEC, incl,phiref,psi,dist,det)
        term2 += SingleDetectorLogLikelihoodModelViaArray(lookupNKDict, ctUArrayDict, ctVArrayDict, tref, RA, DEC, incl,phiref,psi,dist,det)


    return term1+term2


def  DiscreteFactoredLogLikelihoodViaArray(tvals, P, lookupNKDict, rholmsArrayDict, ctUArrayDict,ctVArrayDict,epochDict,Lmax=2,array_output=False):
    """
    DiscreteFactoredLogLikelihoodViaArray uses the array-ized data structures to compute the log likelihood,
    either as an array vs time *or* marginalized in time. 
    This generally is marginally faster, particularly if Lmax is large.

    The timeseries quantities are computed via discrete shifts of an existing grid

    Note 'P' must have the *sampling rate* set to correctly interpret the event time.
     Note arguments passed are STILL SCALARS
    """
    global distMpcRef

    detectors = rholmsArrayDict.keys()
    npts = len(tvals)

    RA = P.phi
    DEC =  P.theta
    tref = P.tref # geocenter time?
    phiref = P.phiref
    incl = P.incl
    psi = P.psi
    dist = P.dist
    distMpc = dist/(lal.PC_SI*1e6)
    invDistMpc = distMpcRef/distMpc

    deltaT = P.deltaT

    lnL = np.zeros(npts,dtype=RiftFloat)


    for det in detectors:
            assert len(tvals) <= len(rholmsArrayDict[det][0])   # code cannot work if window too large!

            U = ctUArrayDict[det]
            V = ctVArrayDict[det]
            Ylms = ComputeYlmsArray(lookupNKDict[det], incl,-phiref)

            t_ref = epochDict[det]

            # BE CAREFUL ABOUT DETECTOR ROTATION: need time not at start time in general! Use 'reference time' to do better
            F = ComplexAntennaFactor(det, RA, DEC, psi, tref)
            invDistMpc = distMpcRef/distMpc

            # This is the GPS time at the detector
            t_det = ComputeArrivalTimeAtDetector(det, RA, DEC, tref) # target time to explore around; should be CENTERED in interval
            tfirst = float(t_det)+tvals[0]
            ifirst = int(round(( float(tfirst) - t_ref) / P.deltaT) + 0.5) # this should be fast, done once. Should also be POSITIVE
            ilast = ifirst + npts

            det_rholms = np.zeros(( len(lookupNKDict[det]),npts),dtype=np.complex64)  # rholms evaluated at time at detector, in window, packed. Do NOT 
            # do not interpolate, just use nearest neighbors.
            for indx in np.arange(len(lookupNKDict[det])):
                det_rholms[indx] = rholmsArrayDict[det][indx][ifirst:ilast]

            # Quadratic term: SingleDetectorLogLikelihoodModelViaArray
            term2 = 0.j
            term2 += F*np.conj(F)*(np.dot(np.conj(Ylms), np.dot(U,Ylms)))
            term2 += F*F*np.dot(Ylms,np.dot(V,Ylms))
            term2 = np.sum(term2)
            term2 = -np.real(term2) / 4. /(distMpc/distMpcRef)**2

            # Linear term
            term1 = np.zeros(len(tvals), dtype=complex)
            term1 = np.dot(np.conj(F*Ylms),det_rholms)   # be very careful re how this multiplication is done: suitable to use this form of multiply
            term1 = np.real(term1) / (distMpc/distMpcRef)

            lnL+= term1+term2

    if  array_output:  # return the raw array
        return lnL
    else:  # return the marginalized lnL in time
        lnLmax = np.max(lnL)
        lnLmargT = np.log(my_simps(np.exp(lnL-lnLmax), dx=deltaT)) + lnLmax
        return lnLmargT

def  DiscreteFactoredLogLikelihoodViaArrayVector(tvals, P_vec, lookupNKDict, rholmsArrayDict, ctUArrayDict,ctVArrayDict,epochDict,Lmax=2,array_output=False,xpy=xpy_default):
    """
    DiscreteFactoredLogLikelihoodViaArray uses the array-ized data structures to compute the log likelihood,
    either as an array vs time *or* marginalized in time. 
    This generally is marginally faster, particularly if Lmax is large.

    The timeseries quantities are computed via discrete shifts of an existing grid

    Note 'P' must have the *sampling rate* set to correctly interpret the event time.
     Note arguments passed are NOW ARRAYS, in contrast to similar function which does not have 'Vector' postfix
    """
    detectors = rholmsArrayDict.keys()
    npts = len(tvals)
    npts_extrinsic = len(P_vec.phi)

    # All arrays of length `npts_extrinsic`, except for `tref` which is a scalar
    RA = P_vec.phi
    DEC = P_vec.theta
    tref = P_vec.tref  # geocenter time, stored as a scalar
    phiref = P_vec.phiref
    incl = P_vec.incl
    psi = P_vec.psi
    dist = P_vec.dist
    distMpc = dist/(lal.PC_SI*1e6)
    invDistMpc = distMpcRef/distMpc


    deltaT = P_vec.deltaT # this is stored as a scalar

    # Array to use for work
    lnL = np.zeros(npts,dtype=RiftFloat)
    lnL_array = np.zeros((npts_extrinsic,npts),dtype=RiftFloat)
    # Array to use for output
    lnLmargOut = np.zeros(npts_extrinsic,dtype=RiftFloat)
#    term1  = np.zeros(npts, dtype=complex) # workspace

    for det in detectors:  # strings right now - need to change to make ufunc-able
      # these do not depend on extrinsic params
      U= ctUArrayDict[det]
      V = ctVArrayDict[det]

      # these do depend on extrinsic params
      Ylms_vec = ComputeYlmsArrayVector(lookupNKDict[det], incl,-phiref)
      F_vec = lalF(det, RA, DEC, psi, tref)
      invDistMpc = distMpcRef/distMpc

      t_ref = epochDict[det]  # a constant for each IFO

      # This is the GPS time at the detector...an arra y
      t_det = lalT(det, RA, DEC, tref)
      for indx_ex in np.arange(npts_extrinsic):  # effectively a loop over RA, DEC
            tfirst = float(t_det[indx_ex])+tvals[0]
            d_here = distMpc[indx_ex]
      
            # pull out scalars
            Ylms = Ylms_vec.T[indx_ex].T  # yank out Ylms for this specific set of parameters
            F = complex(F_vec.T[indx_ex])  # should be scalar

            # these are scalars
            ifirst = int(round(( float(tfirst) - t_ref) / P_vec.deltaT) + 0.5) # this should be fast, done once
            ilast = ifirst + npts

            det_rholms = np.zeros(( len(lookupNKDict[det]),npts),dtype=np.complex64)  # rholms evaluated at time at detector, in window, packed. Do NOT 
            # do not interpolate, just use nearest neighbors.
            for indx in np.arange(len(lookupNKDict[det])):
                det_rholms[indx] = rholmsArrayDict[det][indx][ifirst:ilast]

            # Quadratic term: SingleDetectorLogLikelihoodModelViaArray
            term2 = 0.j
            term2 += F*np.conj(F)*(np.dot(np.conj(Ylms), np.dot(U,Ylms)))
            term2 += F*F*np.dot(Ylms,np.dot(V,Ylms))
            term2 = np.sum(term2)
            term2 = -np.real(term2) / 4. /(d_here/distMpcRef)**2

            # Linear term
            term1  = np.zeros(len(tvals), dtype=complex) # workspace
            term1 = np.dot(np.conj(F*Ylms),det_rholms)   # be very careful re how this multiplication is done: suitable to use this form of multiply
            term1 = np.real(term1) / (d_here/distMpcRef)


            lnL = term1+term2
            lnL_array[indx_ex] += lnL  #  copy into array.  Add, because we will get terms from other IFOs
            maxlnL = np.max(lnL)
            lnLmargOut[indx_ex] = maxlnL + np.log(my_simps(np.exp(lnL_array[indx_ex] - maxlnL), dx=deltaT))  # integrate term by term, minmize overflows

    return lnLmargOut

def  DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopOrig(tvals, P_vec, lookupNKDict, rholmsArrayDict, ctUArrayDict,ctVArrayDict,epochDict,Lmax=2,array_output=False):
    """
    DiscreteFactoredLogLikelihoodViaArray uses the array-ized data structures to compute the log likelihood,
    either as an array vs time *or* marginalized in time. 
    This generally is marginally faster, particularly if Lmax is large.
    The timeseries quantities are computed via discrete shifts of an existing grid
    Note 'P' must have the *sampling rate* set to correctly interpret the event time.
     Note arguments passed are NOW ARRAYS, in contrast to similar function which does not have 'Vector' postfix
    """
    global distMpcRef

    detectors = rholmsArrayDict.keys()
    npts = len(tvals)
    npts_extrinsic = len(P_vec.phi)

    # All arrays of length `npts_extrinsic`, except for `tref` which is a scalar
    RA = P_vec.phi
    DEC =  P_vec.theta
    tref = P_vec.tref # geocenter time, stored as a scalar
    phiref = P_vec.phiref
    incl = P_vec.incl
    psi = P_vec.psi
    dist = P_vec.dist
    distMpc = dist/(lal.PC_SI*1e6)
    invDistMpc = distMpcRef/distMpc


    deltaT = P_vec.deltaT # this is stored as a scalar

    # Array to use for work
    lnL = np.zeros(npts,dtype=np.float64)
    lnL_t_accum = np.zeros((npts_extrinsic,npts),dtype=np.float64)

    for det in detectors:  # strings right now - need to change to make ufunc-able
        # These do not depend on extrinsic params.
        # Arrays of shape (n_lms, n_lms).
        # Axis 0 corresponds to (l,m), and axis 1 corresponds to (l',m').
        U = ctUArrayDict[det]
        V = ctVArrayDict[det]

        n_lms = len(U)

        # These do depend on extrinsic params
        # Array of shape (npts_extrinsic, n_lms,)
        Ylms_vec = ComputeYlmsArrayVector(lookupNKDict[det], incl, -phiref).T
        # Array of shape (npts_extrinsic,)
        F_vec = lalF(det, RA, DEC, psi, tref)
        # Array of shape (npts_extrinsic,)
        invDistMpc = distMpcRef/distMpc

        # Scalar -- is constant for each IFO
        t_ref = epochDict[det]

        # This is the GPS time at the detector, an array of shape (npts_extrinsic,)
        t_det = lalT(det, RA, DEC, tref)

        tfirst = t_det + tvals[0]

        ifirst = (np.round((tfirst-t_ref) / P_vec.deltaT) + 0.5).astype(int)
        ilast = ifirst + npts

        # Note: Very inefficient, need to avoid making `Qlms` by doing the
        # inner product in a CUDA kernel.
        det_rholms = rholmsArrayDict[det]
        Qlms = np.empty((npts_extrinsic, npts, n_lms), dtype=np.complex128)
        for i in range(npts_extrinsic):
            Qlms[i] = det_rholms[..., ifirst[i]:ilast[i]].T

        # Has shape (npts_extrinsic,)
        term2 = ( (F_vec*np.conj(F_vec)).real *np.einsum(
                "...i,...j,ij",
                np.conj(Ylms_vec), Ylms_vec, U,
            ).real )
        term2 += (np.square(F_vec) *
            np.einsum(
                "...i,...j,ij",
                Ylms_vec, Ylms_vec, V,
            )
        ).real
        term2 *= -0.25 * np.square(distMpcRef / distMpc)

       # Has shape (npts_extrinsic, npts).
        # Starts as term1, and accumulates term2 after.

        # View into F with shape (npts_extrinsic, n_lms)
        F_vec_dummy_lm = F_vec[..., np.newaxis]
        # View into F * Ylm with shape (npts_extrinsic, npts, n_lms)
        FY_dummy_t = np.broadcast_to(
            (F_vec_dummy_lm * Ylms_vec)[:, np.newaxis],
            Qlms.shape,
        )

        lnL_t_accum += np.einsum(
            "...i,...i",
            np.conj(FY_dummy_t), Qlms,
        ).real * (distMpcRef/distMpc)[...,None]

        # Accumulate term2 into the time-dependent log likelihood.
        # Have to create a view with an extra axis so they broadcast.
        lnL_t_accum += term2[..., np.newaxis]

    # Take exponential of the log likelihood in-place.
    lnLmax  = np.max(lnL_t_accum)
    L_t = np.exp(lnL_t_accum - lnLmax, out=lnL_t_accum)
        
    # Integrate out the time dimension.  We now have an array of shape
    # (npts_extrinsic,)
    L = my_simps(L_t, dx=deltaT, axis=-1)
    # Compute log likelihood in-place.
    lnL = lnLmax+ np.log(L, out=L)


    return lnL


# TODO: This should be set based on the marginalization scheme being used.
# Right now it is hard-coded to not use any tricks, but later will be toggled
# to use (e.g.) pre-tabulated distance marginaliation dependance.
def _factored_lnL_helper(kappa_sq, rho_sq):
    return kappa_sq - 0.5 * rho_sq


def _cubic_Q_window_numpy(Q_block, start_indices, fractional_offsets, npts):
    """Return cubic-interpolated Q windows with zero extension.

    Q_block has shape (n_time, n_lm).  The returned array has shape
    (n_extrinsic, npts, n_lm), matching the CPU fallback layout used by NoLoop.
    The interpolation is a local four-point cubic Lagrange stencil; at integer
    offsets it reproduces the original samples exactly.
    """
    npts_extrinsic = len(start_indices)
    n_lms_det = Q_block.shape[1]
    Qlms = np.zeros((npts_extrinsic, npts, n_lms_det), dtype=np.complex128)
    tgrid = np.arange(npts)
    n_time = Q_block.shape[0]
    for i in range(npts_extrinsic):
        idxs = int(start_indices[i]) + tgrid
        u = float(fractional_offsets[i])
        u2 = u*u
        u3 = u2*u
        weights = (
            -u * (u - 1.0) * (u - 2.0) / 6.0,
            (u + 1.0) * (u - 1.0) * (u - 2.0) / 2.0,
            -(u + 1.0) * u * (u - 2.0) / 2.0,
            (u + 1.0) * u * (u - 1.0) / 6.0,
        )
        for offset, weight in zip((-1, 0, 1, 2), weights):
            idxs_here = idxs + offset
            valid = (idxs_here >= 0) & (idxs_here < n_time)
            if np.any(valid):
                Qlms[i, valid] += weight * Q_block[idxs_here[valid]]
    return Qlms


SINC_HALFWIDTH_DEFAULT = 8   # taps per side for time_interp='sinc' (stencil 2a); see
                             # _sinc_Q_window_numpy for the accuracy-vs-oversampling crossover


def _sinc_lanczos_weight_matrix(u, a=SINC_HALFWIDTH_DEFAULT, xpy=np):
    """Lanczos (windowed-sinc) interpolation weights for an ARRAY of fractional offsets.

    THIS IS THE SINGLE DEFINITION OF THE STENCIL.  The CPU window builder and the GPU kernel
    wrapper both come here for their weights, so the two paths cannot drift apart by someone
    re-deriving the formula in CUDA; a GPU/CPU parity failure is then unambiguously a kernel bug.

    ``xpy`` selects the array backend: pass cupy and the weights are built ON THE DEVICE, so the
    GPU path needs no host round trip for the per-sample offsets (at production n_extrinsic that
    round trip would move tens of MB per detector per likelihood call).  The arithmetic is the
    same source expression either way; only the underlying sin() differs, at the 1e-16 level.

    Returns (offsets, weights): offsets has shape (2a,) and holds the integer tap positions
    [-a+1, a] relative to the sample below the target; weights has shape (len(u), 2a).  Both are
    in the requested backend.

    L(x) = sinc(x) sinc(x/a) with the normalised sinc, so L(0)=1 and L(k)=0 at nonzero integer
    k: at u=0 this reduces to the identity and reproduces the original samples exactly, as the
    cubic stencil does.  Weights are renormalised to sum to unity, which is a no-op at u=0 and
    makes the interpolation exact for constants.
    """
    u = xpy.atleast_1d(xpy.asarray(u, dtype=float))
    k = xpy.arange(-a + 1, a + 1)
    x = u[:, None] - k[None, :]
    w = xpy.sinc(x) * xpy.sinc(x / float(a))
    w = xpy.where(xpy.abs(x) >= a, 0.0, w)
    total = w.sum(axis=1)
    # A zero row cannot happen for u in [0,1) (the u=0 row is a unit vector), but guard anyway
    # rather than emit NaNs into the likelihood.
    total = xpy.where(total == 0, 1.0, total)
    return k, w / total[:, None]


def _sinc_lanczos_weights(u, a=SINC_HALFWIDTH_DEFAULT):
    """Scalar-offset convenience wrapper over _sinc_lanczos_weight_matrix.

    Returns (offsets, weights) with weights of shape (2a,).
    """
    k, w = _sinc_lanczos_weight_matrix(u, a)
    return k, w[0]


def _sinc_Q_window_numpy(Q_block, start_indices, fractional_offsets, npts,
                         a=SINC_HALFWIDTH_DEFAULT):
    """Return band-limited-interpolated Q windows with zero extension.

    Same contract as _cubic_Q_window_numpy: Q_block has shape (n_time, n_lm), result has shape
    (n_extrinsic, npts, n_lm).  a is the number of taps per side (stencil 2a).

    WHEN THIS WINS, AND WHEN IT DOES NOT.  Q^a_lm(t) is band-limited to fmax, sampled at 1/deltaT,
    so what matters is the oversampling factor fNyq/fmax.  The two stencils fail differently:

      * 'cubic' is a four-point Lagrange polynomial.  Its error is O(h^4) and so falls FAST with
        oversampling -- but it is poor near Nyquist, where a cubic cannot follow the signal.
      * 'sinc' (this) is a Lanczos-windowed sinc.  Its error is set by the window, NOT by h, so it
        PLATEAUS: more oversampling does not help it, but neither does less hurt it.

    Measured max relative error on a synthetic band-limited signal (test_q_window_interp.py):

        fNyq/fmax     cubic      sinc a=8     sinc a=32
            1.5       6.2e-2      1.2e-3        9.9e-5
            2         2.7e-2      7.9e-4        4.7e-5
            4         2.2e-3      4.3e-4        2.8e-5
            8         9.0e-5      2.7e-4        2.0e-5
           16         1.0e-5      3.3e-4        2.2e-5

    Re-measured with 12 seeds per point, the crossover (cubic error = sinc error) sits at
    fNyq/fmax ~= 5.3, with the seed-to-seed spread bracketing 1.0 only over 5-6.  PRODUCTION RUNS
    ARE NEAR NYQUIST -- srate 4096 with fmax ~1700 is fNyq/fmax ~ 1.2 -- which is exactly where
    sinc is tens of times better.  A heavily oversampled configuration (the slow-rotation
    brute-force test runs fmax=512 at srate 16384, i.e. 16) is the regime where cubic already
    wins and this option should NOT be used.

    Because of that crossover the DEFAULT is deliberately left at 'cubic': the right choice
    depends on fNyq/fmax, which this function cannot see.  The PIPELINE can, and does:
    RIFT.likelihood.time_interp_choice.choose_time_interp_stencil applies the threshold, and
    helper_LDG_Events.py calls it when --internal-ile-interpolate-time is given without a value.

    COST, measured (not estimated from the tap count):
      CPU  ~4.2-4.5x cubic -- 2a=16 taps against 4, and this path IS tap-count bound.
      GPU  ~1.6-3.0x cubic -- Q_inner_sinc is bandwidth/latency bound, so it does far better
           than the naive 4x.  See Q_inner_product.Q_inner_product_sinc_cupy.
    """
    npts_extrinsic = len(start_indices)
    n_lms_det = Q_block.shape[1]
    Qlms = np.zeros((npts_extrinsic, npts, n_lms_det), dtype=np.complex128)
    tgrid = np.arange(npts)
    n_time = Q_block.shape[0]
    for i in range(npts_extrinsic):
        idxs = int(start_indices[i]) + tgrid
        offsets, weights = _sinc_lanczos_weights(float(fractional_offsets[i]), a)
        for offset, weight in zip(offsets, weights):
            if weight == 0.0:
                continue
            idxs_here = idxs + offset
            valid = (idxs_here >= 0) & (idxs_here < n_time)
            if np.any(valid):
                Qlms[i, valid] += weight * Q_block[idxs_here[valid]]
    return Qlms


TIME_INTERP_CHOICES = ('nearest', 'cubic', 'sinc')


def validate_time_interp(time_interp, on_gpu=False):
    """Reject unknown stencils loudly.

    All three stencils now have both a CPU and a GPU implementation ('sinc' via the Q_inner_sinc
    kernel added alongside Q_inner and Q_inner_cubic), so on_gpu no longer restricts the choice.
    It is kept in the signature because the callers pass it and because it documents, at each
    call site, that the stencil has to be legal on the backend actually in use.
    """
    if time_interp not in TIME_INTERP_CHOICES:
        raise ValueError("time_interp must be one of %r, got %r"
                         % (TIME_INTERP_CHOICES, time_interp))
    return time_interp


def _q_window_numpy_interp(Q_block, start_indices, fractional_offsets, npts, time_interp,
                           xpy=np):
    """CPU Q-window dispatch.  start_indices must already match the stencil: 'nearest' rounds,
    the interpolating stencils floor and carry the fractional part separately."""
    if time_interp == 'nearest':
        return _nearest_Q_window_numpy(Q_block, start_indices, npts, xpy=xpy)
    if time_interp == 'sinc':
        return _sinc_Q_window_numpy(Q_block, start_indices, fractional_offsets, npts)
    return _cubic_Q_window_numpy(Q_block, start_indices, fractional_offsets, npts)


def _q_inner_product_gpu(Q, A, start_indices, fractional_offsets, npts, time_interp):
    """GPU Q-product dispatch: the device-side counterpart of _q_window_numpy_interp.

    Same stencil contract and the same fallthrough structure as the CPU dispatch, deliberately:
    the four GPU call sites (here x2, plus _with_rotation and _freqresponse) all route through
    this one function so a new stencil cannot be wired into three of them and forgotten in the
    fourth.  Note this returns the CONTRACTED (n_extrinsic, npts) product, not the
    (n_extrinsic, npts, n_lm) window the CPU builder returns -- the device kernels fuse the
    lm contraction to avoid the large temporary."""
    if time_interp == 'nearest':
        return Q_inner_product.Q_inner_product_cupy(Q, A, start_indices, npts)
    if time_interp == 'sinc':
        return Q_inner_product.Q_inner_product_sinc_cupy(
            Q, A, start_indices, fractional_offsets, npts)
    return Q_inner_product.Q_inner_product_cubic_cupy(
        Q, A, start_indices, fractional_offsets, npts)


def _nearest_Q_window_numpy(Q_block, start_indices, npts, xpy=np):
    """Return nearest-grid Q windows with zero extension."""
    npts_extrinsic = len(start_indices)
    n_lms_det = Q_block.shape[1]
    Qlms = xpy.zeros((npts_extrinsic, npts, n_lms_det), dtype=np.complex128)
    tgrid = np.arange(npts)
    n_time = Q_block.shape[0]
    for i in range(npts_extrinsic):
        idxs = int(start_indices[i]) + tgrid
        valid = (idxs >= 0) & (idxs < n_time)
        Qlms[i, valid] = Q_block[idxs[valid]]
    return Qlms


def  DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(tvals, P_vec, lookupNKDict, rholmsArrayDict, ctUArrayDict,ctVArrayDict,epochDict,Lmax=2,array_output=False,xpy=np, loglikelihood=_factored_lnL_helper,return_lnLt=False,phase_marginalization=False,n_cal=1,cal_method='loop',cal_distmarg=None,cal_log_weights=None,return_cal_components=False,time_interp='nearest',ctUArrayDict_cal=None,ctVArrayDict_cal=None):
    """
    DiscreteFactoredLogLikelihoodViaArray uses the array-ized data structures to compute the log likelihood,
    either as an array vs time *or* marginalized in time.
    This generally is marginally faster, particularly if Lmax is large.
    The timeseries quantities are computed via discrete shifts of an existing grid
    Note 'P' must have the *sampling rate* set to correctly interpret the event time.
     Note arguments passed are NOW ARRAYS, in contrast to similar function which does not have 'Vector' postfix

    Calibration marginalization (n_cal>=1)
    --------------------------------------
    When n_cal>1, the rholm timeseries are assumed to hold ``n_cal`` contiguous
    calibration realizations, each of length N_window = npts_full/n_cal (built by
    ComputeModeIPTimeSeries with the calibration applied to the *data*).  The data
    term kappa = <h|C_c d> changes per realization, selected by shifting the window
    offset ifirst -> ifirst + c*N_window.

    The template self-term is NOT calibration-independent once the calibration is
    applied to the data: the statistically complete term is the per-realization
    rho_sq_c = <C_c h | C_c h> = <h| |C_c|^2 |h>.  When the |C_c|^2-weighted cross
    terms are supplied (ctUArrayDict_cal/ctVArrayDict_cal), this per-realization
    rho_sq_c is used in place of the shared, cal-independent rho_sq=<h|h> -- for
    every n_cal, INCLUDING n_cal==1 (a single applied draw still carries its own
    self-term).  Omitting them (ctUArrayDict_cal=None) reproduces the historical,
    cal-independent-<h|h> behavior byte-for-byte.  (The bare <h|h> shortcut broke
    the C->lambda*C distance-degeneracy invariance and biased lnZ; see
    ComputeModeCrossTermIPCal / BuildCalibrationSelfTermBasis.)

    We then Monte-Carlo marginalize:
        Z_cal(theta) = (1/n_cal) sum_c  integral dt exp( lnL_t(theta, c) )
    via a streaming log-sum-exp over the n_cal realizations (memory unchanged vs
    the n_cal==1 path; one extra GPU kernel launch per realization).

    cal_method selects the n_cal>1 reduction:
      'loop'  (default, Option B): Python loop over realizations reusing the
              existing Q_inner_product kernel; works on CPU and GPU and with any
              loglikelihood callback (distance/phase marginalization).
      'fused' (Option C): a single fused kernel (RIFT.likelihood.Q_fused_calmarg)
              does the Q-product, the loglikelihood, and the cal+time log-sum-exp in
              one launch -- a CUDA kernel on GPU, or pure numpy on CPU.  Supports
              phase marginalization (|kappa|).  With cal_distmarg=None it
              uses the default (distance-unmarginalized) helper kernel; with
              cal_distmarg=<table dict> it uses the separate distance-marginalization
              kernel (Q_fused_calmarg_distmarg) reproducing distmarg_loglikelihood.
              Raises NotImplementedError for unsupported combinations; the 'loop'
              path is the fallback for everything (incl. distmarg, on CPU and GPU).

    cal_distmarg : dict or None
        Distance-marginalization table+params for the fused distmarg kernel; see
        RIFT.likelihood.Q_fused_calmarg.Q_fused_calmarg_distmarg_cupy.

    time_interp : {'nearest', 'cubic', 'sinc'}
        Sub-sample stencil for the Q(t) lookup.  Which is best depends on the oversampling
        factor fNyq/fmax: 'cubic' (4-point Lagrange) has O(h^4) error so it wins when heavily
        oversampled, while 'sinc' (Lanczos) is window-limited so its error is flat in
        oversampling and it wins near Nyquist -- ~50x better at fNyq/fmax ~ 1.2, which is where
        production runs sit.  Crossover is around fNyq/fmax ~ 4-8.  See _sinc_Q_window_numpy.
        'sinc' is CPU-only for now.
        Detector-time sampling convention for the data term.  'nearest'
        preserves the historical NoLoop integer-bin gather.  'cubic' evaluates
        the precomputed Q_lm time series at the fractional detector arrival time
        using a four-sample cubic Lagrange stencil, with zero extension outside
        the precomputed buffer.
    """
    global distMpcRef

    validate_time_interp(time_interp, on_gpu=not (xpy is np))
    if time_interp != 'nearest' and cal_method == 'fused':
        raise NotImplementedError("time_interp='{}' is not implemented for cal_method='fused'".format(time_interp))

    detectors = rholmsArrayDict.keys()
    npts = len(tvals)
    npts_extrinsic = len(P_vec.phi)
    # npts_full = len(rholmsArrayDict[detectors[0]][0]) # all have same size
    # print " npts :", npts
    # print " npts_full:", npts_full

    # All arrays of length `npts_extrinsic`, except for `tref` which is a scalar
    RA = P_vec.phi
    DEC = P_vec.theta

    # geocenter time, stored as a scalar
    tref = P_vec.tref
    phiref = P_vec.phiref
    incl = P_vec.incl
    psi = P_vec.psi
    dist = P_vec.dist
    distMpc = dist/(lal.PC_SI*1e6)
    invDistMpc = distMpcRef/distMpc


    deltaT = float(P_vec.deltaT) # this is stored as a scalar


    # Convert tref to greenwich mean sidereal time
    greenwich_mean_sidereal_time_tref = xpy.asarray(
        lal.GreenwichMeanSiderealTime(tref)
    )

    # this is stored as a scalar
    deltaT = P_vec.deltaT

    # Array to accumulate lnL(t) summed across all detectors.
#    lnL_t_accum = xpy.zeros((npts_extrinsic, npts), dtype=np.float64)

    # Used to accumulate kappa^2 and rho^2 over all detectors.  They are just
    # the sum in quadrature of the individual detector contributions.
    kappa_sq = xpy.zeros((npts_extrinsic, npts), dtype=np.complex128)
    rho_sq = xpy.zeros((npts_extrinsic, npts), dtype=np.float64)

    # When marginalizing over calibration (n_cal>1), cache the per-detector data
    # term inputs here; the calibration-independent rho_sq is still accumulated
    # in the loop below, and kappa is recomputed per realization afterwards.
    cal_cache = {}

    # Fused-calmarg self-term fix (the calmarg self-term-bias analysis note).  When the
    # per-realization |C_c|^2-weighted cross terms are supplied, accumulate the
    # per-realization template self-term rho_sq_c = <C_c h | C_c h> (shape
    # (n_cal, npts_extrinsic); time-independent) alongside the shared rho_sq, and use
    # it PER REALIZATION in the reductions below in place of the cal-independent
    # rho_sq.  This restores the C -> lambda*C distance-degeneracy invariance that the
    # data-side shortcut (fixed <h|h>) broke.  When absent (ctUArrayDict_cal is None),
    # behavior is byte-for-byte identical to before.
    # Active whenever the |C_c|^2-weighted cross terms are supplied -- INCLUDING n_cal==1
    # (a single calibration draw is still applied to the data, so its self-term must be
    # <C h|C h>, not the cal-independent <h|h>; the n_cal==1 path below uses rho_sq_cal[0]).
    _use_rho_sq_cal = (ctUArrayDict_cal is not None) and (ctVArrayDict_cal is not None)
    rho_sq_cal = xpy.zeros((n_cal, npts_extrinsic), dtype=np.float64) if _use_rho_sq_cal else None

    if (xpy is np) or (optimized_gpu_tools is None):
        simps = my_simps
    elif not (xpy is np):
        simps = optimized_gpu_tools.simps
    else:
        raise NotImplementedError("Backend not supported: {}".format(xpy))

    # strings right now - need to change to make ufunc-able
    for det in detectors:
        # Compute the detector's location and response matrix
        detector = lalsim.DetectorPrefixToLALDetector(det)
        detector_location = xpy.asarray(detector.location)
        detector_response = xpy.asarray(detector.response)

        # These do not depend on extrinsic params.
        # Arrays of shape (n_lms, n_lms).
        # Axis 0 corresponds to (l,m), and axis 1 corresponds to (l',m').
        U = ctUArrayDict[det]
        V = ctVArrayDict[det]

        lms = lookupNKDict[det]
        n_lms = len(lms)

        # These do depend on extrinsic params
        # Array of shape (npts_extrinsic, n_lms,)
        Ylms_vec = SphericalHarmonicsVectorized(
            lms, incl, -phiref,
            xpy=xpy,
            l_max=Lmax,
        )

        # Array of shape (npts_extrinsic,)
#        F_vec_old = xpy.asarray(lalF(det, RA, DEC, psi, tref))
        F_vec = ComputeDetAMResponse(
            detector_response,
            RA, DEC, psi,
            greenwich_mean_sidereal_time_tref,
            xpy=xpy
        )

        # Scalar -- is constant for each IFO
        t_ref = epochDict[det]

        # This is the GPS time at the detector,
        # Note that to save on precision compared to ...NoLoopOrig, we CHANGE the t_det definition to be relative to the IFO statt time t_ref
        #    ... this means we don't keep a 1e9 out in front, so we have more significant digits in the event time (and can if needed reduce precision in GPU ops)
        # an array of shape (npts_extrinsic,)
        t_det = float(tref - float(t_ref)) + TimeDelayFromEarthCenter(
            detector_location, RA, DEC,
            float(greenwich_mean_sidereal_time_tref),
            xpy=xpy
        )
        tfirst = t_det + tvals[0]

        sample_first = tfirst / deltaT
        if time_interp == 'nearest':
            ifirst = (xpy.rint(sample_first) + 0.5).astype(np.int32)  # C uses 32 bit integers : be careful
            frac_first = None
        else:
            ifirst = xpy.floor(sample_first).astype(np.int32)
            frac_first = (sample_first - xpy.floor(sample_first)).astype(np.float64)
#        ilast = ifirst + npts


        # Has shape (npts_extrinsic,)
        rho_sq_det = (
            (F_vec*xpy.conj(F_vec)).real *
            xpy.einsum(
                "...i,...j,ij",
                xpy.conj(Ylms_vec), Ylms_vec, U,
            ).real
        )

        rho_sq_det += (
            xpy.square(F_vec) *
            xpy.einsum(
                "...i,...j,ij",
                Ylms_vec, Ylms_vec, V,
            )
        ).real
        # NOTE: Double-check this 0.5 is correct.  It used to be -0.25, but
        # we're slightly reorganizing the expressions from term2 = -0.5 rho_sq
        # into using rho_sq directly
        rho_sq_det *= 0.5 * xpy.square(distMpcRef / distMpc)

        # Fused-calmarg self-term fix: the per-realization template self-term
        # rho_sq_c = <C_c h | C_c h>, built EXACTLY like rho_sq_det above but from the
        # |C_c|^2-weighted cross terms U_cal,V_cal (shape (n_cal, n_lms, n_lms)).  The
        # extra leading axis is the calibration realization c.  MUST use the original
        # (un-phase-conjugated) Ylms_vec, matching rho_sq_det, so it is computed here
        # before the phase_marginalization conjugation below.
        if _use_rho_sq_cal:
            # Use the FIRST n_cal weighted blocks.  The supplied arrays may hold more
            # realizations than n_cal -- e.g. the zero-cal burn-in (--calibration-burn-in-neff)
            # runs with n_cal=1 while retaining the full N_cal cross-term arrays -- and without
            # this slice rho_sq_det_cal would be (N_cal, npts) and mismatch the (n_cal, npts)
            # accumulator.  Slicing keeps rho_sq_c consistent with the rholm blocks the n_cal
            # reduction actually reads (realizations 0..n_cal-1).
            U_cal = ctUArrayDict_cal[det][:n_cal]
            V_cal = ctVArrayDict_cal[det][:n_cal]
            rho_sq_det_cal = (
                (F_vec*xpy.conj(F_vec)).real *
                xpy.einsum("ei,ej,cij->ce", xpy.conj(Ylms_vec), Ylms_vec, U_cal).real
            )
            rho_sq_det_cal = rho_sq_det_cal + (
                xpy.square(F_vec) *
                xpy.einsum("ei,ej,cij->ce", Ylms_vec, Ylms_vec, V_cal)
            ).real
            rho_sq_det_cal *= 0.5 * xpy.square(distMpcRef / distMpc)
            rho_sq_cal += rho_sq_det_cal   # accumulate over detectors -> (n_cal, npts_extrinsic)

        # If phase_marginalization is turned on, the (2, -2) term should be
        # replaced by its complex conjugate before the absolute value of
        # kappa_sq is calculated
        if phase_marginalization:
            Ylms_vec[:, 1] = xpy.conj(Ylms_vec[:, 1])
            Q = xpy.ascontiguousarray(
                xpy.concatenate(
                    [
                        rholmsArrayDict[det][0][..., np.newaxis],
                        xpy.conj(rholmsArrayDict[det][1][..., np.newaxis])
                    ],
                    axis=1
                )
            )
            F_vec_dummy_lm = xpy.concatenate(
                [F_vec[..., np.newaxis], xpy.conj(F_vec[..., np.newaxis])],
                axis=1
            )
        else:
            Q = xpy.ascontiguousarray(rholmsArrayDict[det].T)
            # # Note: Very inefficient, need to avoid making `Qlms` by doing the
            # # inner product in a CUDA kernel.
            # det_rholms = xpy.asarray(rholmsArrayDict[det])
            # Qlms = xpy.empty((npts_extrinsic, npts, n_lms), dtype=complex)
            # for i in range(npts_extrinsic):
            #     Qlms[i] = det_rholms[...,ifirst[i]:ilast[i]].T

            # Has shape (npts_extrinsic, npts).
            # Starts as term1, and accumulates term2 after.

            # View into F with shape (npts_extrinsic, n_lms)
            F_vec_dummy_lm = F_vec[..., np.newaxis]
            # # View into F * Ylm with shape (npts_extrinsic, npts, n_lms)
            # FY_dummy_t = xpy.broadcast_to(
            #     (F_vec_dummy_lm * Ylms_vec)[:, np.newaxis],
            #     Qlms.shape,
            # )

            # lnL_t_accum += xpy.einsum(
            #     "...i,...i",
            #     xpy.conj(FY_dummy_t), Qlms,
            # ).real * (distMpcRef/distMpc)[...,None]

        if n_cal == 1:
          # ---- standard path (no calibration marginalization): UNCHANGED ----
          if not (xpy is np):
            FY_conj = xpy.conj(F_vec_dummy_lm * Ylms_vec)
            # Shape Q = (npts_time_full, nlms)
            # Shape A=FY_conj = (npts_extrinsic, nlms)
            # shape result = (npts_extrinsic, npts_time_*window* = npts)
            Q_prod_result = _q_inner_product_gpu(
                Q, FY_conj, ifirst, frac_first, npts, time_interp)
          else:
            # Use old code completely unchanged ... very wasteful on memory management!
            Q_block = rholmsArrayDict[det].T
            Qlms = _q_window_numpy_interp(Q_block, ifirst, frac_first, npts, time_interp,
                                          xpy=xpy)
            if phase_marginalization:
                Qlms[:, :, 1] = xpy.conj(Qlms[:, :, 1])

            FY_dummy_t = np.broadcast_to(
              (F_vec_dummy_lm * Ylms_vec)[:, np.newaxis],
              Qlms.shape,
              )

            Q_prod_result =  np.einsum(
              "...i,...i",
              np.conj(FY_dummy_t), Qlms,
              )

          kappa_sq += Q_prod_result * (distMpcRef/distMpc)[..., np.newaxis]
        else:
          # ---- calibration-marginalization path (Option B): cache pieces ----
          # The rholm timeseries hold n_cal contiguous realizations; realization c
          # is selected later via ifirst -> ifirst + c*N_window_block.  Q is
          # (npts_full, n_lms) and FY_conj is (npts_extrinsic, n_lms); both already
          # encode any phase-marginalization conjugation built above.
          npts_full_det = Q.shape[0]
          N_window_block = npts_full_det // n_cal
          FY_conj_cal = xpy.conj(F_vec_dummy_lm * Ylms_vec)
          cal_cache[det] = (Q, FY_conj_cal, ifirst, N_window_block, frac_first)
        # lnL_t_accum += Q_prod_result * (distMpcRef/distMpc)[...,None]

        # lnL_t_accum += Q_inner_product.Q_inner_product_cupy(
        #     FY_conj, Q,
        #     ifirst, npts,
        # ).real * (distMpcRef/distMpc)[...,None]


        # Accumulate term2 into the time-dependent log likelihood.
        # Have to create a view with an extra axis so they broadcast.
        rho_sq += rho_sq_det[..., np.newaxis]
        # lnL_t_accum += term2[..., np.newaxis]

#        print lnL_t_accum.shape, lnL_t.shape

#        lnL_t_accum += lnL_t


    if n_cal == 1:
        # Fused-calmarg self-term fix also applies to a SINGLE calibration draw: the data
        # carries C_0, so its self-term is rho_sq_c = <C_0 h|C_0 h> = rho_sq_cal[0], not the
        # cal-independent <h|h>.  Falls back to rho_sq for the ordinary (no-cal) likelihood.
        rho_sq_here = rho_sq if not _use_rho_sq_cal else xpy.broadcast_to(rho_sq_cal[0][:, np.newaxis], (npts_extrinsic, npts))
        if phase_marginalization:
            lnL_t = loglikelihood(xpy.abs(kappa_sq), rho_sq_here)
        else:
            lnL_t = loglikelihood(kappa_sq.real, rho_sq_here)

        # Take exponential of the log likelihood in-place.
        lnLmax  = xpy.max(lnL_t)
        if return_lnLt:
          return lnL_t  #- lnLmax    # we want the verbatim lnL_t values, no shift
        L_t = xpy.exp(lnL_t - lnLmax, out=lnL_t)

        L = simps(L_t, dx=deltaT, axis=-1)

        # Compute log likelihood in-place.
        lnL = lnLmax + xpy.log(L, out=L)

        return lnL

    # ---- calibration-marginalization reduction (Option B) ----
    # Monte-Carlo marginalize over the n_cal calibration draws:
    #   Z_cal(theta) = (1/n_cal) sum_c  \int dt exp( lnL_t(theta, c) )
    # rho_sq (the <h|h> U,V term) is calibration-independent and was accumulated
    # once above; only the data term kappa changes per realization, selected by
    # shifting the window offset into block c.  Accumulate a streaming log-sum-exp
    # over realizations for numerical stability (S holds sum_c exp(lnL_t - max)).
    # return_lnLt with calibration marginalization: return the cal-marginalized lnL(t)
    # timeseries (weighted log-sum-exp over realizations at each time bin) so downstream
    # time resampling/output works.  This is produced by the loop reduction below; the
    # fused kernel (which returns the time-integrated scalar) is skipped in that case.

    # Per-realization importance log-weights for the cal marginalization.  Default
    # (None) is uniform -> the plain (1/n_cal) average; non-uniform weights support
    # adaptive / importance cal sampling (w_c = prior(c)/proposal(c), with E_q[w]=1).
    # The cal-marginalized likelihood is the UNBIASED importance estimate of the prior
    # integral:  L_marg = integral pi(c) L(c) dc ~= (1/n_cal) sum_c w_c L_c.  So the
    # normalization is log(n_cal) -- NOT logsumexp(log_w) (the self-normalized/ratio
    # estimator), which is biased by log(sum w / n_cal) for non-uniform weights.  For
    # uniform weights the two coincide.
    if cal_log_weights is None:
        cal_log_w = xpy.zeros(n_cal, dtype=np.float64)
    else:
        cal_log_w = xpy.asarray(cal_log_weights, dtype=np.float64)
    cal_log_w_norm = float(np.log(n_cal))

    if cal_method == 'fused' and not return_lnLt and not return_cal_components:
        # ---- Option C: fused implementation (GPU CUDA kernel, or numpy on CPU) ----
        # (return_lnLt needs the per-time series, which the loop reduction produces, so
        #  the fused scalar kernel is bypassed when a timeseries is requested.)
        if cal_distmarg is None and loglikelihood is not _factored_lnL_helper:
            raise NotImplementedError("fused cal kernel needs either the default helper or a cal_distmarg table")
        import RIFT.likelihood.Q_fused_calmarg as Q_fused_calmarg
        dets = list(cal_cache.keys())
        # All detectors share modes/length; ifirst differs per detector (time delay)
        Q_stack = xpy.stack([cal_cache[d][0] for d in dets])               # (n_det, npts_full, n_lms)
        A_stack = xpy.stack([cal_cache[d][1] for d in dets])               # (n_det, npts_extrinsic, n_lms)
        ifirst_stack = xpy.stack([cal_cache[d][2] for d in dets]).astype(np.int32)  # (n_det, npts_extrinsic)
        N_window_block = cal_cache[dets[0]][3]
        # Simpson quadrature weight vector (incl. dx=deltaT), so time integration
        # matches the loop path's simps() exactly.  simps is linear -> weights = simps(I).
        w_t = simps(xpy.eye(npts, dtype=np.float64), dx=deltaT, axis=-1)
        # invDistMpc is a scalar when distance is marginalized (P.dist fixed at the
        # fiducial) and a vector when distance is sampled; the kernel wants one value
        # per extrinsic sample, so broadcast to (npts_extrinsic,).
        invDist_vec = xpy.asarray(invDistMpc, dtype=np.float64) * xpy.ones(npts_extrinsic, dtype=np.float64)
        # Fused-calmarg self-term fix: hand the per-realization rho_sq_c (n_cal,
        # npts_extrinsic) to the kernel, which indexes it by realization in place of
        # the shared (n_ext, npts) rho_sq.  None -> the kernel keeps the old behavior.
        if xpy is np:
            # CPU: pure-numpy fused (no CUDA); independent cross-check of the kernel
            return Q_fused_calmarg.Q_fused_calmarg_numpy(
                Q_stack, A_stack, ifirst_stack, invDist_vec, rho_sq, w_t,
                n_cal, N_window_block, distmarg=cal_distmarg, cal_log_weights=cal_log_weights,
                phase_marginalization=phase_marginalization, rho_sq_cal=rho_sq_cal)
        if cal_distmarg is None:
            return Q_fused_calmarg.Q_fused_calmarg_cupy(
                Q_stack, A_stack, ifirst_stack, invDist_vec, rho_sq, w_t,
                n_cal, N_window_block, cal_log_weights=cal_log_weights,
                phase_marginalization=phase_marginalization, rho_sq_cal=rho_sq_cal)
        else:
            return Q_fused_calmarg.Q_fused_calmarg_distmarg_cupy(
                Q_stack, A_stack, ifirst_stack, invDist_vec, rho_sq, w_t,
                n_cal, N_window_block, cal_distmarg, cal_log_weights=cal_log_weights,
                phase_marginalization=phase_marginalization, rho_sq_cal=rho_sq_cal)

    running_max = None
    S = xpy.zeros((npts_extrinsic, npts), dtype=np.float64)
    # Pilot/adaptive support: optionally collect the RAW (un-importance-weighted)
    # time-integrated log-likelihood per realization, one value per extrinsic point.
    # This is the cal posterior responsibility used by util_CalPilotFit to learn a
    # proposal.  (loop method only; the fused scalar path is bypassed above.)
    cal_components = xpy.zeros((npts_extrinsic, n_cal), dtype=np.float64) if return_cal_components else None
    for c in range(n_cal):
        kappa_sq_c = xpy.zeros((npts_extrinsic, npts), dtype=np.complex128)
        for det in detectors:
            Q_det, FY_conj_det, ifirst_det, N_window_block, frac_first_det = cal_cache[det]
            # Restrict to THIS realization block and use the within-block offset, so
            # the window is confined to [0, N_window): an over-running window then
            # zeros (the shared kernel / CPU fill below guard against N_window_block)
            # instead of bleeding into the neighbouring block or reading out of
            # bounds.  This matches the per-block n_cal==1 reference and the fused
            # kernel exactly.
            Q_block = Q_det[c*N_window_block:(c+1)*N_window_block]   # (N_window, n_lms)
            ifirst_within = ifirst_det.astype(np.int32)
            if not (xpy is np):
                Q_prod_result = _q_inner_product_gpu(
                    Q_block, FY_conj_det, ifirst_within, frac_first_det, npts, time_interp)
            else:
                Qlms = _q_window_numpy_interp(Q_block, ifirst_within, frac_first_det, npts,
                                              time_interp, xpy=xpy)
                # Q_det and FY_conj_det already encode any phase-marg conjugation
                Q_prod_result = np.einsum("ej,etj->et", FY_conj_det, Qlms)
            kappa_sq_c += Q_prod_result * invDistMpc[..., np.newaxis]

        # Fused-calmarg self-term fix: use this realization's rho_sq_c = <C_c h|C_c h>
        # instead of the shared cal-independent rho_sq.  Broadcast to the FULL
        # (npts_extrinsic, npts) shape (as the shared rho_sq is), so distance-
        # marginalization loglikelihoods that boolean-index rho_sq work unchanged.
        # Falls back to rho_sq when the fix is inactive.
        if _use_rho_sq_cal:
            rho_sq_here = xpy.broadcast_to(rho_sq_cal[c][:, np.newaxis], (npts_extrinsic, npts))
        else:
            rho_sq_here = rho_sq
        if phase_marginalization:
            lnL_t_c = loglikelihood(xpy.abs(kappa_sq_c), rho_sq_here)
        else:
            lnL_t_c = loglikelihood(kappa_sq_c.real, rho_sq_here)
        if return_cal_components:
            # RAW per-realization time-integrated log L (no importance weight), stable:
            #   log( simps_t exp(lnL_t,c) ) = m + log( simps_t exp(lnL_t,c - m) )
            m_raw = xpy.max(lnL_t_c, axis=-1, keepdims=True)
            cal_components[:, c] = m_raw[:, 0] + xpy.log(simps(xpy.exp(lnL_t_c - m_raw), dx=deltaT, axis=-1))
        # fold in this realization's importance log-weight
        lnL_t_c = lnL_t_c + cal_log_w[c]

        m_c = xpy.max(lnL_t_c)
        if running_max is None:
            running_max = m_c
        elif m_c > running_max:
            S *= xpy.exp(running_max - m_c)
            running_max = m_c
        S += xpy.exp(lnL_t_c - running_max)

    if return_cal_components:
        # (npts_extrinsic, n_cal): RAW per-realization integrated log-likelihood.  The
        # caller (util_CalPilot / ILE dump) accumulates over the harvested extrinsic
        # points to form the cal posterior responsibility, then fits a proposal.
        return cal_components

    if return_lnLt:
        # cal-marginalized lnL at each time bin:
        #   lnL_marg(t) = log( sum_c exp(log_w[c]) exp(lnL_t,c(t)) ) - log(n_cal)
        # (the time integral is NOT taken; downstream resamples this timeseries).
        return running_max + xpy.log(S) - cal_log_w_norm

    L = simps(S, dx=deltaT, axis=-1)
    # lnL = max + log( sum_c exp(log_w[c]) \int dt exp(lnL_t - max) ) - log(n_cal)
    lnL = running_max + xpy.log(L) - cal_log_w_norm

    return lnL


def ComputeYlmsArray(lookupNK, theta, phi):
    """
    Returns an array Ylm[k] where lookup(k) = l,m.  Only computes the LM values needed.
    theta, phi arguments are *scalars*

    SHOULD BE DEPRECATED
    """
    Ylms=None
    if isinstance(lookupNK, dict):
       pairs = lookupNK.keys()
       Ylms = np.zeros(len(pairs),dtype=complex)
       for indx in np.arange(len(pairs)):
            l = int(pairs[indx][0])
            m = int(pairs[indx][1])
            Ylms[ indx] = lal.SpinWeightedSphericalHarmonic(theta, phi,-2, l, m)
    elif isinstance(lookupNK, np.ndarray):
       Ylms = np.zeros(len(lookupNK),dtype=complex)
       for indx in np.arange(len(lookupNK)):
            l = int(lookupNK[indx][0])
            m = int(lookupNK[indx][1])
            Ylms[ indx] = lal.SpinWeightedSphericalHarmonic(theta, phi,-2, l, m)
    return Ylms


fallback=False

if not('RIFT_LOWLATENCY' in os.environ): 
  # numba is rarely used : we use GPU optimized almost always.  No point incurring these import costs.
 try:
        import numba
        from numba import vectorize, complex128, float64, int64
        numba_on = True
        if log_loud:
          print(" Numba on ")

        # Very inefficient : decorating
        # Problem - lately, compiler not correctly identifying return value of code
        # Should just use SphericalHarmonicsVectorized
        @vectorize([complex128(float64,float64,int64,int64,int64)])
        def lalylm(th,ph,s,l,m):
                return lal.SpinWeightedSphericalHarmonic(th,ph,s,l,m)
        # @vectorize
        # def lalF(det, RA,DEC,psi,tref):
        #         return ComplexAntennaFactor(det, RA, DEC, psi, tref)
        # @vectorize
        # def lalT(deta, RA, DEC, tref):
#                return ComputeArrivalTimeAtDetector(det, RA, DEC, tref)

        def lalF(det, RA, DEC,psi,tref): # note tref is a SCALAR
                F = np.zeros( len(RA), dtype=complex)
                for indx  in np.arange(len(RA)):
                        F[indx] = ComplexAntennaFactor(det, RA[indx],DEC[indx], psi[indx], tref)
                return F
        def lalT(det, RA, DEC,tref): # note tref is a SCALAR
                T = np.zeros( len(RA), dtype=float)
                for indx  in np.arange(len(RA)):
                        T[indx] = ComputeArrivalTimeAtDetector(det, RA[indx],DEC[indx],  tref)
                return T

 except:
   fallback=True

if fallback or ('RIFT_LOWLATENCY' in os.environ): 
        numba_on = False
        if log_loud:
          print(" Numba off ")
        # Very inefficient.  np.vectorize replicates the numba @vectorize elementwise
        # behavior (the plain scalar wrapper below breaks ComputeYlmsArrayVector, which
        # passes numpy ARRAYS for th,ph,s,l,m -> LAL raises 'argument 1 of type REAL8').
        lalylm = np.vectorize(lal.SpinWeightedSphericalHarmonic, otypes=[complex])
        def lalF(det, RA, DEC,psi,tref):
                if isinstance(RA, float):
                        return ComplexAntennaFactor(det, RA, DEC, psi,tref)
                F = np.zeros( len(RA), dtype=complex)
                for indx  in np.arange(len(RA)):
                        F[indx] = ComplexAntennaFactor(det, RA[indx],DEC[indx], psi[indx], tref)
                return F
        def lalT(det, RA, DEC,tref):
                if isinstance(RA, float):
                        return ComputeArrivalTimeAtDetector(det, RA, DEC,tref)
                T = np.zeros( len(RA), dtype=float)
                for indx  in np.arange(len(RA)):
                        T[indx] = ComputeArrivalTimeAtDetector(det, RA[indx],DEC[indx], tref)
                return T

#        lalF = ComplexAntennaFactor
#        lalT = ComputeArrivalTimeAtDetector

def ComputeYlmsArrayVector(lookupNK, theta, phi):
    """
    Returns an array Ylm[k] where lookup(k) = l,m.  Only computes the LM values needed.
    theta, phi arguments are *vectors*.  Shape is (len(th),len(lookup(NK)))

    Should be combined with the previous routine ComputeYlmsArray (redundant)

    Example:
       th = np.linspace(0,np.pi, 5); lookupNK =[[2,-2], [2,2]]  
       factored_likelihood.ComputeYlmsArrayVector(lookupNK,th,th)
    """

    # Allocate
    Ylms = np.zeros((len(lookupNK), len(theta)),dtype=complex)

    # Force cast to array. This should never be called, but can avoid some failures due to 'object' dtype failing through
    theta = np.array(theta,dtype=float)
    phi = np.array(phi,dtype=float)

    # Loop over l, m and evaluate.
    for indx in range(len(lookupNK)):
            l = int(lookupNK[indx][0])*np.ones(len(theta),dtype=int)   # use np.repeat instead for speed
            m = int(lookupNK[indx][1])*np.ones(len(theta),dtype=int)
            s = -2 * np.ones(len(theta),dtype=int)

            Ylms[indx] = lalylm(theta, phi, s, l, m)
    return Ylms
