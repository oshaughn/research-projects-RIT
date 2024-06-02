from __future__ import print_function

import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lsu  # problem of relative comprehensive import - dangerous due to package name
import numpy as np
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

from .SphericalHarmonics_gpu import SphericalHarmonicsVectorized


from scipy import integrate

import os
if 'PROFILE' not in os.environ:
   def profile(fn):
      return fn

has_GWS=False  # make sure defined in top-level scope
try:
        import RIFT.physics.GWSignal as rgws
        has_GWS=True
except:
        has_GWS=False

if not( 'RIFT_LOWLATENCY'  in os.environ):
  # Dont support external packages in low latency
 try:
        import NRWaveformCatalogManager3 as nrwf
        useNR =True
        print(" factored_likelihood.py : NRWaveformCatalogManager3 available ")
 except ImportError:
        useNR=False


 try:
        import RIFT.physics.ROMWaveformManager as romwf
        print(" factored_likelihood.py: ROMWaveformManager as romwf")
        useROM=True
        rom_basis_scale = 1.0*1e-21   # Fundamental problem: Inner products with ROM basis vectors/Sh are tiny. Need to rescale to avoid overflow/underflow and simplify comparisons
 except ImportError:
        useROM=False
        print(" factored_likelihood.py: - no ROM - ")
        rom_basis_scale =1

 try:
    hasEOB=True
    import RIFT.physics.EOBTidalExternalC as eobwf
 #    import EOBTidalExternal as eobwf
 except:
    hasEOB=False
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

        if rosDebugMessages:
                print("NR variant: Length check: ",hlms[(2,2)].data.length, first_data.data.length)
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
            print(" External EOB length check ", hlms[(2,2)].data.length, first_data.data.length, first_data.data.length*P.deltaT)
            print(" External EOB length check (in M) ", end=' ')
            print(" Comparison EOB duration check vs epoch vs window size (sec) ", wfP.estimateDurationSec(),  -hlms[(2,2)].epoch, 1./hlms[(2,2)].deltaF)
            assert hlms[(2,2)].data.length ==first_data.data.length
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
# LISA
def ComputeIPTimeSeries(IP, hf, data, N_shift, N_window, analyticPSD_Q=False,
        inv_spec_trunc_Q=False, T_spec=0., debug=True):
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
    assert data.deltaF == hf.deltaF
    assert data.data.length == hf.data.length


    rho, rhoTS, rhoIdx, rhoPhase = IP.ip(hf, data)
    rhoTS.epoch = data.epoch - hf.epoch
    tmp= lsu.DataRollBins(rhoTS, N_shift)  # restore functionality for bidirectional shifts: waveform need not start at t=0
    rho_time_series =lal.CutCOMPLEX16TimeSeries(rhoTS, 0, N_window)
    if debug:
        print(f"Max in the original series = {np.max(rhoTS.data.data)}, Max in the truncated series = {np.max(rho_time_series.data.data)}")
    return rho_time_series

def PrecomputeAlignedSpinLISA(tref, t_window, hlms, hlms_conj, data_dict, psd_dict, flow, fNyq, fhigh, deltaT,  beta, lamda, analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.):
   # GENERATE DETECTOR RESPONSE
   # Compute time truncation (this assumes no existing time shifts, so don't inlcude them)
   # Compute Qlm (order of entires in the IP, conj, t_ref)
   # Compute Ulm (sensitive to t_ref being used)

   # TO precompute innerproducts, we need tref for time shift, t_window to select what time values we need 
    # hlms, hlms_conj, data_dict, psd_dict, flow, fhigh, deltaT (integration range)
    # beta, lamdba for detector response

    # Let's work in indexes for time shifts
    # LISA response is time dependent, large shifts in time won't be good so your t_ref needs to be right. Assuming that, the peak in time series should be near the first index

    N_shift = int(t_window/deltaT)
    N_window = int(2 * t_window/deltaT)
    
    # first get 6 terms per mode, and multiply with detector response
    tf_dict, f_dict, amp_dict, phase_dict = get_tf_from_phase_dict(hlms, fNyq)  #here we need fmax, but RIFT has fhigh
    collect_mode_terms = {}
    modes = list(hlms.keys())

    collect_mode_terms["A"] = {}
    collect_mode_terms["E"] = {}
    collect_mode_terms["T"] = {}
    for mode in modes:
        print(mode)
        A_terms, E_terms, T_terms = Evaluate_Gslr_test_2(tf_dict[mode]+tref, f_dict[mode], beta, lamda) # NOTE: I added t_ref
        
        amp, phase = amp_dict[mode], phase_dict[mode]
        shifted_phase = (phase + 2*np.pi*f_dict[mode]*tref) #take care of convention
        tmp_mode_data = (amp * np.exp(1j*shifted_phase)).reshape(1, -1) #take care of convention

        # tmp_mode_data = hlms[mode].data.data
        collect_mode_terms["A"][mode] = {}
        tmp_mode_here = np.conj(A_terms * tmp_mode_data)
        collect_mode_terms["A"][mode][0] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[0])  # A_term (6,n), hlm (n,1)
        collect_mode_terms["A"][mode][1] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[1]) 
        collect_mode_terms["A"][mode][2] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[2]) 
        collect_mode_terms["A"][mode][3] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[3]) 
        collect_mode_terms["A"][mode][4] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[4]) 
        collect_mode_terms["A"][mode][5] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[5])

        
        collect_mode_terms["E"][mode] = {}
        tmp_mode_here = np.conj(E_terms * tmp_mode_data)
        collect_mode_terms["E"][mode][0] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[0])  # A_term (6,n), hlm (n,1)
        collect_mode_terms["E"][mode][1] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[1]) 
        collect_mode_terms["E"][mode][2] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[2]) 
        collect_mode_terms["E"][mode][3] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[3]) 
        collect_mode_terms["E"][mode][4] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[4]) 
        collect_mode_terms["E"][mode][5] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[5]) 

        
        collect_mode_terms["T"][mode] = {}
        tmp_mode_here = np.conj(T_terms * tmp_mode_data)
        collect_mode_terms["T"][mode][0] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[0])  # A_term (6,n), hlm (n,1)
        collect_mode_terms["T"][mode][1] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[1]) 
        collect_mode_terms["T"][mode][2] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[2]) 
        collect_mode_terms["T"][mode][3] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[3]) 
        collect_mode_terms["T"][mode][4] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[4]) 
        collect_mode_terms["T"][mode][5] = create_lal_frequency_series(f_dict[mode], tmp_mode_here[5])  
    # calculate <d|h_lm>
    IP_time_series= lsu.ComplexOverlap(flow, fhigh, fNyq, hlms[2,2].deltaF, psd_dict["A"], analyticPSD_Q, inv_spec_trunc_Q, T_spec, full_output =True)  # Incase the three arms have different PSDs. Assume all arms have same PSD for now and 2,2 mode is present

    Q_lm = {}
    for channel in ["A", "E", "T"]:
        inner_product= IP_time_series
        for mode in hlms.keys():
            l, m = mode[0], mode[1]
        
            Q_lm[f"{channel}_{l}_{m}_xx"] = ((ComputeIPTimeSeries(inner_product, data_dict[channel], collect_mode_terms[channel][mode][0], -N_shift, N_window)))
            Q_lm[f"{channel}_{l}_{m}_xy"] = ((ComputeIPTimeSeries(inner_product, data_dict[channel], collect_mode_terms[channel][mode][1], -N_shift, N_window)))
            Q_lm[f"{channel}_{l}_{m}_xz"] = ((ComputeIPTimeSeries(inner_product, data_dict[channel], collect_mode_terms[channel][mode][2], -N_shift, N_window)))
            Q_lm[f"{channel}_{l}_{m}_yy"] = ((ComputeIPTimeSeries(inner_product, data_dict[channel], collect_mode_terms[channel][mode][3], -N_shift, N_window)))
            Q_lm[f"{channel}_{l}_{m}_yz"] = ((ComputeIPTimeSeries(inner_product, data_dict[channel], collect_mode_terms[channel][mode][4], -N_shift, N_window)))
            Q_lm[f"{channel}_{l}_{m}_zz"] = ((ComputeIPTimeSeries(inner_product, data_dict[channel], collect_mode_terms[channel][mode][5], -N_shift, N_window)))

    # calculate <h_lm|h_pq>
    U_lm_pq = {}
    IP = lsu.ComplexIP(flow, fhigh, fNyq, hlms[2,2].deltaF, psd_dict["A"], analyticPSD_Q, inv_spec_trunc_Q, T_spec) # Incase the three arms have different PSDs. Assume all arms have same PSD for now and 2,2 mode is present
    for i in np.arange(len(modes)):
        for j in np.arange(len(modes))[i:]:
            # print(modes[i], modes[j])
            l, m, p, q = modes[i][0], modes[i][1], modes[j][0], modes[j][1]
            for channel in  ["A", "E", "T"]:
                inner_product= IP
                U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_xx"] = inner_product.ip(collect_mode_terms[channel][modes[i]][0], collect_mode_terms[channel][modes[j]][0])
                U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_xy"] = inner_product.ip(collect_mode_terms[channel][modes[i]][0], collect_mode_terms[channel][modes[j]][1])
                U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_xz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][0], collect_mode_terms[channel][modes[j]][2])
                U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_yy"] = inner_product.ip(collect_mode_terms[channel][modes[i]][0], collect_mode_terms[channel][modes[j]][3])
                U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_yz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][0], collect_mode_terms[channel][modes[j]][4])
                U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_zz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][0], collect_mode_terms[channel][modes[j]][5])

                U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_xx"] = inner_product.ip(collect_mode_terms[channel][modes[i]][1], collect_mode_terms[channel][modes[j]][0])
                U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_xy"] = inner_product.ip(collect_mode_terms[channel][modes[i]][1], collect_mode_terms[channel][modes[j]][1])
                U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_xz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][1], collect_mode_terms[channel][modes[j]][2])
                U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_yy"] = inner_product.ip(collect_mode_terms[channel][modes[i]][1], collect_mode_terms[channel][modes[j]][3])
                U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_yz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][1], collect_mode_terms[channel][modes[j]][4])
                U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_zz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][1], collect_mode_terms[channel][modes[j]][5])

                U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_xx"] = inner_product.ip(collect_mode_terms[channel][modes[i]][2], collect_mode_terms[channel][modes[j]][0])
                U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_xy"] = inner_product.ip(collect_mode_terms[channel][modes[i]][2], collect_mode_terms[channel][modes[j]][1])
                U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_xz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][2], collect_mode_terms[channel][modes[j]][2])
                U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_yy"] = inner_product.ip(collect_mode_terms[channel][modes[i]][2], collect_mode_terms[channel][modes[j]][3])
                U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_yz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][2], collect_mode_terms[channel][modes[j]][4])
                U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_zz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][2], collect_mode_terms[channel][modes[j]][5])

                U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_xx"] = inner_product.ip(collect_mode_terms[channel][modes[i]][3], collect_mode_terms[channel][modes[j]][0])
                U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_xy"] = inner_product.ip(collect_mode_terms[channel][modes[i]][3], collect_mode_terms[channel][modes[j]][1])
                U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_xz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][3], collect_mode_terms[channel][modes[j]][2])
                U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_yy"] = inner_product.ip(collect_mode_terms[channel][modes[i]][3], collect_mode_terms[channel][modes[j]][3])
                U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_yz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][3], collect_mode_terms[channel][modes[j]][4])
                U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_zz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][3], collect_mode_terms[channel][modes[j]][5])
                
                U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_xx"] = inner_product.ip(collect_mode_terms[channel][modes[i]][4], collect_mode_terms[channel][modes[j]][0])
                U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_xy"] = inner_product.ip(collect_mode_terms[channel][modes[i]][4], collect_mode_terms[channel][modes[j]][1])
                U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_xz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][4], collect_mode_terms[channel][modes[j]][2])
                U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_yy"] = inner_product.ip(collect_mode_terms[channel][modes[i]][4], collect_mode_terms[channel][modes[j]][3])
                U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_yz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][4], collect_mode_terms[channel][modes[j]][4])
                U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_zz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][4], collect_mode_terms[channel][modes[j]][5])

                U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_xx"] = inner_product.ip(collect_mode_terms[channel][modes[i]][5], collect_mode_terms[channel][modes[j]][0])
                U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_xy"] = inner_product.ip(collect_mode_terms[channel][modes[i]][5], collect_mode_terms[channel][modes[j]][1])
                U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_xz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][5], collect_mode_terms[channel][modes[j]][2])
                U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_yy"] = inner_product.ip(collect_mode_terms[channel][modes[i]][5], collect_mode_terms[channel][modes[j]][3])
                U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_yz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][5], collect_mode_terms[channel][modes[j]][4])
                U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_zz"] = inner_product.ip(collect_mode_terms[channel][modes[i]][5], collect_mode_terms[channel][modes[j]][5])

                if (l,m) != (p,q):
                    print( (l,m), (p,q))
                    U_lm_pq[f"{channel}_{p}_{q}_xx_{l}_{m}_xx"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_xx"])
                    U_lm_pq[f"{channel}_{p}_{q}_xx_{l}_{m}_xy"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_xx"])
                    U_lm_pq[f"{channel}_{p}_{q}_xx_{l}_{m}_xz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_xx"])
                    U_lm_pq[f"{channel}_{p}_{q}_xx_{l}_{m}_yy"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_xx"])
                    U_lm_pq[f"{channel}_{p}_{q}_xx_{l}_{m}_yz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_xx"])
                    U_lm_pq[f"{channel}_{p}_{q}_xx_{l}_{m}_zz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_xx"])

                    U_lm_pq[f"{channel}_{p}_{q}_xy_{l}_{m}_xx"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_xy"])
                    U_lm_pq[f"{channel}_{p}_{q}_xy_{l}_{m}_xy"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_xy"])
                    U_lm_pq[f"{channel}_{p}_{q}_xy_{l}_{m}_xz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_xy"])
                    U_lm_pq[f"{channel}_{p}_{q}_xy_{l}_{m}_yy"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_xy"])
                    U_lm_pq[f"{channel}_{p}_{q}_xy_{l}_{m}_yz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_xy"])
                    U_lm_pq[f"{channel}_{p}_{q}_xy_{l}_{m}_zz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_xy"])
        
                    U_lm_pq[f"{channel}_{p}_{q}_xz_{l}_{m}_xx"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_xz"])
                    U_lm_pq[f"{channel}_{p}_{q}_xz_{l}_{m}_xy"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_xz"])
                    U_lm_pq[f"{channel}_{p}_{q}_xz_{l}_{m}_xz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_xz"])
                    U_lm_pq[f"{channel}_{p}_{q}_xz_{l}_{m}_yy"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_xz"])
                    U_lm_pq[f"{channel}_{p}_{q}_xz_{l}_{m}_yz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_xz"])
                    U_lm_pq[f"{channel}_{p}_{q}_xz_{l}_{m}_zz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_xz"])

                    U_lm_pq[f"{channel}_{p}_{q}_yy_{l}_{m}_xx"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_yy"])
                    U_lm_pq[f"{channel}_{p}_{q}_yy_{l}_{m}_xy"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_yy"])
                    U_lm_pq[f"{channel}_{p}_{q}_yy_{l}_{m}_xz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_yy"])
                    U_lm_pq[f"{channel}_{p}_{q}_yy_{l}_{m}_yy"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_yy"])
                    U_lm_pq[f"{channel}_{p}_{q}_yy_{l}_{m}_yz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_yy"])
                    U_lm_pq[f"{channel}_{p}_{q}_yy_{l}_{m}_zz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_yy"])

                    U_lm_pq[f"{channel}_{p}_{q}_yz_{l}_{m}_xx"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_yz"])
                    U_lm_pq[f"{channel}_{p}_{q}_yz_{l}_{m}_xy"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_yz"])
                    U_lm_pq[f"{channel}_{p}_{q}_yz_{l}_{m}_xz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_yz"])
                    U_lm_pq[f"{channel}_{p}_{q}_yz_{l}_{m}_yy"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_yz"])
                    U_lm_pq[f"{channel}_{p}_{q}_yz_{l}_{m}_yz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_yz"])
                    U_lm_pq[f"{channel}_{p}_{q}_yz_{l}_{m}_zz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_yz"])

                    U_lm_pq[f"{channel}_{p}_{q}_zz_{l}_{m}_xx"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_zz"])
                    U_lm_pq[f"{channel}_{p}_{q}_zz_{l}_{m}_xy"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_zz"])
                    U_lm_pq[f"{channel}_{p}_{q}_zz_{l}_{m}_xz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_zz"])
                    U_lm_pq[f"{channel}_{p}_{q}_zz_{l}_{m}_yy"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_zz"])
                    U_lm_pq[f"{channel}_{p}_{q}_zz_{l}_{m}_yz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_zz"])
                    U_lm_pq[f"{channel}_{p}_{q}_zz_{l}_{m}_zz"] = np.conj(U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_zz"])
                
    return collect_mode_terms, Q_lm, U_lm_pq


def FactoredLogLikelihoodAlignedSpinLISA(Q_lm, U_lm_pq, beta, lam, psi, inclination, phi_ref, distance, modes, reference_distance):
    # Calculated marginalized likelihood, marginalized over time, psi, inlincation, phiref and distance.

    # call psi terms
    plus_terms = get_beta_lamda_psi_terms_Hp(beta, lam, psi)
    cross_terms =  get_beta_lamda_psi_terms_Hc(beta, lam, psi)

    # call spherical harmonics 
    factor = np.ones(modes.shape)
    factor[:,1] = -factor[:,1] 
    negative_m_modes = modes * factor
    spherical_harmonics  = SphericalHarmonicsVectorized(modes, inclination, -phi_ref)
    negative_m_harmonics = SphericalHarmonicsVectorized(negative_m_modes, inclination, -phi_ref)

    Qlm = {}
    Qlm["A"], Qlm["E"], Qlm["T"] = 0, 0, 0

    for i, mode in enumerate(modes):
        l, m = mode[0], mode[1]
        Sp_harm_lm, Sp_harm_lm_  = spherical_harmonics[:,i], negative_m_harmonics[:,i]
        Sp_harm_lm_conjugate = np.conj((Sp_harm_lm))
        for channel in ["A", "E", "T"]:
            Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_xx"].data.data.reshape(-1,1) * (0.5*(plus_terms[0] - 1j*cross_terms[0])*Sp_harm_lm_conjugate + 0.5*(-1)**(l)*(plus_terms[0] + 1j*cross_terms[0])*(Sp_harm_lm_))
            Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_xy"].data.data.reshape(-1,1) * (0.5*(plus_terms[1] - 1j*cross_terms[1])*Sp_harm_lm_conjugate + 0.5*(-1)**(l)*(plus_terms[1] + 1j*cross_terms[1])*(Sp_harm_lm_))
            Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_xz"].data.data.reshape(-1,1) * (0.5*(plus_terms[2] - 1j*cross_terms[2])*Sp_harm_lm_conjugate + 0.5*(-1)**(l)*(plus_terms[2] + 1j*cross_terms[2])*(Sp_harm_lm_))
            Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_yy"].data.data.reshape(-1,1) * (0.5*(plus_terms[3] - 1j*cross_terms[3])*Sp_harm_lm_conjugate + 0.5*(-1)**(l)*(plus_terms[3] + 1j*cross_terms[3])*(Sp_harm_lm_))
            Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_yz"].data.data.reshape(-1,1) * (0.5*(plus_terms[4] - 1j*cross_terms[4])*Sp_harm_lm_conjugate + 0.5*(-1)**(l)*(plus_terms[4] + 1j*cross_terms[4])*(Sp_harm_lm_))
            Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_zz"].data.data.reshape(-1,1) * (0.5*(plus_terms[5] - 1j*cross_terms[5])*Sp_harm_lm_conjugate + 0.5*(-1)**(l)*(plus_terms[5] + 1j*cross_terms[5])*(Sp_harm_lm_))

    Ulmpq = {}
    Ulmpq["A"], Ulmpq["E"], Ulmpq["T"] = 0, 0, 0
    for i, mode in enumerate(modes):
        l, m = modes[i][0], modes[i][1]
        
        Sp_harm_lm, Sp_harm_lm_ = spherical_harmonics[:,i], negative_m_harmonics[:,i]
        Sp_harm_lm__conjugate = np.conj(Sp_harm_lm_),
        for j, mode in enumerate(modes): 
            p, q =  modes[j][0], modes[j][1]

            Sp_harm_pq, Sp_harm_pq_  = spherical_harmonics[:,j], negative_m_harmonics[:,j]
            Sp_harm_pq_conjugate =  np.conj(Sp_harm_pq)

            for channel in ["A", "E", "T"]:
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_xx"]* (0.5*(plus_terms[0] + 1j*cross_terms[0])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[0] - 1j*cross_terms[0])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[0] - 1j*cross_terms[0])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[0] + 1j*cross_terms[0])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_xy"]* (0.5*(plus_terms[0] + 1j*cross_terms[0])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[0] - 1j*cross_terms[0])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[1] - 1j*cross_terms[1])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[1] + 1j*cross_terms[1])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_xz"]* (0.5*(plus_terms[0] + 1j*cross_terms[0])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[0] - 1j*cross_terms[0])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[2] - 1j*cross_terms[2])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[2] + 1j*cross_terms[2])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_yy"]* (0.5*(plus_terms[0] + 1j*cross_terms[0])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[0] - 1j*cross_terms[0])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[3] - 1j*cross_terms[3])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[3] + 1j*cross_terms[3])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_yz"]* (0.5*(plus_terms[0] + 1j*cross_terms[0])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[0] - 1j*cross_terms[0])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[4] - 1j*cross_terms[4])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[4] + 1j*cross_terms[4])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_zz"]* (0.5*(plus_terms[0] + 1j*cross_terms[0])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[0] - 1j*cross_terms[0])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[5] - 1j*cross_terms[5])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[5] + 1j*cross_terms[5])*(Sp_harm_pq_))

                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_xx"]* (0.5*(plus_terms[1] + 1j*cross_terms[1])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[1] - 1j*cross_terms[1])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[0] - 1j*cross_terms[0])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[0] + 1j*cross_terms[0])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_xy"]* (0.5*(plus_terms[1] + 1j*cross_terms[1])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[1] - 1j*cross_terms[1])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[1] - 1j*cross_terms[1])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[1] + 1j*cross_terms[1])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_xz"]* (0.5*(plus_terms[1] + 1j*cross_terms[1])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[1] - 1j*cross_terms[1])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[2] - 1j*cross_terms[2])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[2] + 1j*cross_terms[2])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_yy"]* (0.5*(plus_terms[1] + 1j*cross_terms[1])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[1] - 1j*cross_terms[1])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[3] - 1j*cross_terms[3])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[3] + 1j*cross_terms[3])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_yz"]* (0.5*(plus_terms[1] + 1j*cross_terms[1])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[1] - 1j*cross_terms[1])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[4] - 1j*cross_terms[4])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[4] + 1j*cross_terms[4])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_zz"]* (0.5*(plus_terms[1] + 1j*cross_terms[1])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[1] - 1j*cross_terms[1])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[5] - 1j*cross_terms[5])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[5] + 1j*cross_terms[5])*(Sp_harm_pq_))

                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_xx"]* (0.5*(plus_terms[2] + 1j*cross_terms[2])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[2] - 1j*cross_terms[2])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[0] - 1j*cross_terms[0])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[0] + 1j*cross_terms[0])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_xy"]* (0.5*(plus_terms[2] + 1j*cross_terms[2])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[2] - 1j*cross_terms[2])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[1] - 1j*cross_terms[1])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[1] + 1j*cross_terms[1])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_xz"]* (0.5*(plus_terms[2] + 1j*cross_terms[2])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[2] - 1j*cross_terms[2])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[2] - 1j*cross_terms[2])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[2] + 1j*cross_terms[2])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_yy"]* (0.5*(plus_terms[2] + 1j*cross_terms[2])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[2] - 1j*cross_terms[2])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[3] - 1j*cross_terms[3])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[3] + 1j*cross_terms[3])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_yz"]* (0.5*(plus_terms[2] + 1j*cross_terms[2])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[2] - 1j*cross_terms[2])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[4] - 1j*cross_terms[4])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[4] + 1j*cross_terms[4])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_zz"]* (0.5*(plus_terms[2] + 1j*cross_terms[2])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[2] - 1j*cross_terms[2])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[5] - 1j*cross_terms[5])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[5] + 1j*cross_terms[5])*(Sp_harm_pq_))

                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_xx"]* (0.5*(plus_terms[3] + 1j*cross_terms[3])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[3] - 1j*cross_terms[3])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[0] - 1j*cross_terms[0])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[0] + 1j*cross_terms[0])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_xy"]* (0.5*(plus_terms[3] + 1j*cross_terms[3])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[3] - 1j*cross_terms[3])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[1] - 1j*cross_terms[1])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[1] + 1j*cross_terms[1])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_xz"]* (0.5*(plus_terms[3] + 1j*cross_terms[3])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[3] - 1j*cross_terms[3])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[2] - 1j*cross_terms[2])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[2] + 1j*cross_terms[2])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_yy"]* (0.5*(plus_terms[3] + 1j*cross_terms[3])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[3] - 1j*cross_terms[3])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[3] - 1j*cross_terms[3])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[3] + 1j*cross_terms[3])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_yz"]* (0.5*(plus_terms[3] + 1j*cross_terms[3])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[3] - 1j*cross_terms[3])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[4] - 1j*cross_terms[4])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[4] + 1j*cross_terms[4])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_zz"]* (0.5*(plus_terms[3] + 1j*cross_terms[3])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[3] - 1j*cross_terms[3])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[5] - 1j*cross_terms[5])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[5] + 1j*cross_terms[5])*(Sp_harm_pq_))

                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_xx"]* (0.5*(plus_terms[4] + 1j*cross_terms[4])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[4] - 1j*cross_terms[4])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[0] - 1j*cross_terms[0])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[0] + 1j*cross_terms[0])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_xy"]* (0.5*(plus_terms[4] + 1j*cross_terms[4])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[4] - 1j*cross_terms[4])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[1] - 1j*cross_terms[1])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[1] + 1j*cross_terms[1])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_xz"]* (0.5*(plus_terms[4] + 1j*cross_terms[4])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[4] - 1j*cross_terms[4])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[2] - 1j*cross_terms[2])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[2] + 1j*cross_terms[2])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_yy"]* (0.5*(plus_terms[4] + 1j*cross_terms[4])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[4] - 1j*cross_terms[4])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[3] - 1j*cross_terms[3])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[3] + 1j*cross_terms[3])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_yz"]* (0.5*(plus_terms[4] + 1j*cross_terms[4])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[4] - 1j*cross_terms[4])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[4] - 1j*cross_terms[4])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[4] + 1j*cross_terms[4])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_zz"]* (0.5*(plus_terms[4] + 1j*cross_terms[4])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[4] - 1j*cross_terms[4])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[5] - 1j*cross_terms[5])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[5] + 1j*cross_terms[5])*(Sp_harm_pq_))

                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_xx"]* (0.5*(plus_terms[5] + 1j*cross_terms[5])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[5] - 1j*cross_terms[5])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[0] - 1j*cross_terms[0])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[0] + 1j*cross_terms[0])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_xy"]* (0.5*(plus_terms[5] + 1j*cross_terms[5])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[5] - 1j*cross_terms[5])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[1] - 1j*cross_terms[1])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[1] + 1j*cross_terms[1])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_xz"]* (0.5*(plus_terms[5] + 1j*cross_terms[5])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[5] - 1j*cross_terms[5])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[2] - 1j*cross_terms[2])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[2] + 1j*cross_terms[2])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_yy"]* (0.5*(plus_terms[5] + 1j*cross_terms[5])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[5] - 1j*cross_terms[5])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[3] - 1j*cross_terms[3])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[3] + 1j*cross_terms[3])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_yz"]* (0.5*(plus_terms[5] + 1j*cross_terms[5])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[5] - 1j*cross_terms[5])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[4] - 1j*cross_terms[4])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[4] + 1j*cross_terms[4])*(Sp_harm_pq_))
                Ulmpq[channel] += U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_zz"]* (0.5*(plus_terms[5] + 1j*cross_terms[5])*(Sp_harm_lm) + 0.5*(-1)**(l)*(plus_terms[5] - 1j*cross_terms[5])*Sp_harm_lm__conjugate) * (0.5*(plus_terms[5] - 1j*cross_terms[5])*Sp_harm_pq_conjugate + 0.5*(-1)**(p)*(plus_terms[5] + 1j*cross_terms[5])*(Sp_harm_pq_))
    total_lnL = 0
    for channel in ["A", "E", "T"]:
        total_lnL += np.real(reference_distance/distance * Qlm[channel] - ((reference_distance/distance)**2)*0.5*Ulmpq[channel])
    Q_lm_term = list(Q_lm.keys())[0]
    L_t = np.exp(total_lnL - np.max(total_lnL))
    L = integrate.simpson(L_t, dx = Q_lm[Q_lm_term].deltaT, axis=0) #P.deltaT
    lnL  = np.max(total_lnL) + L
    return lnL


