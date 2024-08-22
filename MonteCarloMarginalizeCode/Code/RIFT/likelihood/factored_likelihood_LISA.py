from __future__ import print_function
import sys
from RIFT.LISA.response.LISA_response import *
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

distGpcRef = 200 # a fiducial distance for the template source.
tWindowExplore = [-0.15, 0.15] # Not used in main code.  Provided for backward compatibility for ROS. Should be consistent with t_ref_wind in ILE.
rosDebugMessages = True
rosDebugMessagesDictionary = {}   # Mutable after import (passed by reference). Not clear if it can be used by caling routines
                                                  # BUT if every module has a `PopulateMessagesDictionary' module, I can set their internal copies
rosDebugMessagesDictionary["DebugMessages"] = False
rosDebugMessagesDictionary["DebugMessagesLong"] = False



###########################################################################################
# Main functions
###########################################################################################
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
    assert data.deltaF == hf.deltaF
    assert data.data.length == hf.data.length

    rho, rhoTS, rhoIdx, rhoPhase = IP.ip(hf, data)
    rhoTS.epoch = data.epoch - hf.epoch
    tmp= lsu.DataRollBins(rhoTS, N_shift)  # restore functionality for bidirectional shifts: waveform need not start at t=0
    rho_time_series =lal.CutCOMPLEX16TimeSeries(rhoTS, 0, N_window)
    if debug:
        print(f"Max in the original series = {np.max(rhoTS.data.data)}, max in the truncated series = {np.max(rho_time_series.data.data)}, max index in the original series = {np.argmax(rhoTS.data.data) + N_shift}, max occurs at time shift of {rhoIdx * rhoTS.deltaT}s")
    return rho_time_series

def PrecomputeAlignedSpinLISA(tref, fref, t_window, hlms, hlms_conj, data_dict, psd_dict, flow, fNyq, fhigh, deltaT,  beta, lamda, analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.):
    print(f"PrecomputeAlignedSpinLISA has been called with the following arguments: \n{locals()}")
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
    tf_dict, f_dict, amp_dict, phase_dict = get_tf_from_phase_dict(hlms, fNyq, fref)  #here we need fmax, but RIFT has fhigh
    index_at_fref = get_closest_index(f_dict[2,2], fref)
    
    collect_mode_terms = {}
    modes = list(hlms.keys())

    reference_phase = 0.0
    modes.remove((2,2))
    modes.insert(0, (2,2))

    collect_mode_terms["A"] = {}
    collect_mode_terms["E"] = {}
    collect_mode_terms["T"] = {}
    for mode in modes:
        A_terms, E_terms, T_terms = Evaluate_Gslr_test_2(tf_dict[mode]+tref, f_dict[mode], beta, lamda) # NOTE: I added t_ref
        
        amp, phase = amp_dict[mode], phase_dict[mode]
        shifted_phase = (phase + 2*np.pi*f_dict[mode]*tref) #take care of convention
        if mode == (2,2):
            phase_22_current = shifted_phase[index_at_fref]
            difference = reference_phase - phase_22_current
        shifted_phase = shifted_phase + mode[1]/2 * difference
        print(f"Precompute: {mode}, phase = {shifted_phase[index_at_fref]}, time = {tf_dict[mode][index_at_fref]+tref}")
        
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
    IP_time_series_A= lsu.ComplexOverlap(flow, fhigh, fNyq, hlms[2,2].deltaF, psd_dict["A"], analyticPSD_Q, inv_spec_trunc_Q, T_spec, full_output =True)  # Incase the three arms have different PSDs. Assume all arms have same PSD for now and 2,2 mode is present
    IP_time_series_E= lsu.ComplexOverlap(flow, fhigh, fNyq, hlms[2,2].deltaF, psd_dict["E"], analyticPSD_Q, inv_spec_trunc_Q, T_spec, full_output =True)
    IP_time_series_T= lsu.ComplexOverlap(flow, fhigh, fNyq, hlms[2,2].deltaF, psd_dict["T"], analyticPSD_Q, inv_spec_trunc_Q, T_spec, full_output =True)

    Q_lm = {}
    for channel in ["A", "E", "T"]:
        if channel == "A":
            inner_product= IP_time_series_A
        if channel == "E":
            inner_product= IP_time_series_E
        if channel == "T":
            inner_product= IP_time_series_T
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
    IP_A = lsu.ComplexIP(flow, fhigh, fNyq, hlms[2,2].deltaF, psd_dict["A"], analyticPSD_Q, inv_spec_trunc_Q, T_spec) # Incase the three arms have different PSDs. Assume all arms have same PSD for now and 2,2 mode is present
    IP_E = lsu.ComplexIP(flow, fhigh, fNyq, hlms[2,2].deltaF, psd_dict["E"], analyticPSD_Q, inv_spec_trunc_Q, T_spec)
    IP_T = lsu.ComplexIP(flow, fhigh, fNyq, hlms[2,2].deltaF, psd_dict["T"], analyticPSD_Q, inv_spec_trunc_Q, T_spec)
    for i in np.arange(len(modes)):
        for j in np.arange(len(modes))[i:]:
            # print(modes[i], modes[j])
            l, m, p, q = modes[i][0], modes[i][1], modes[j][0], modes[j][1]
            for channel in  ["A", "E", "T"]:
                if channel == "A":
                    inner_product= IP_A
                if channel == "E":
                    inner_product= IP_E
                if channel == "T":
                    inner_product= IP_T
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

    # Match RIFT's output
    guess_snr = None # right now, it is painful to evaluate this
    rholms_intp = None
    V_lm_pq = None
    rest = None
    # current RIFT output for precompute is: rholms_intp, cross_terms, cross_terms_V,  rholms,  guess_snr, rest     
    return rholms_intp, U_lm_pq, V_lm_pq, Q_lm, guess_snr, rest


def FactoredLogLikelihoodAlignedSpinLISA(Q_lm, U_lm_pq, beta, lam, psi, inclination, phi_ref, distance, modes, reference_distance, return_lnLt=False):
    # Calculated marginalized likelihood, marginalized over time, psi, inlincation, phiref and distance.
    # call psi terms
    plus_terms = get_beta_lamda_psi_terms_Hp(beta, lam, psi)
    cross_terms =  get_beta_lamda_psi_terms_Hc(beta, lam, psi)
    terms = 0.5*(plus_terms - 1j*cross_terms)
    conj_terms = 0.5*(plus_terms + 1j*cross_terms)

    # call spherical harmonics 
    factor = np.ones(modes.shape)
    factor[:,1] = -factor[:,1] 
    negative_m_modes = modes * factor
    spherical_harmonics  = SphericalHarmonicsVectorized(modes, inclination, -phi_ref)
    negative_m_harmonics = SphericalHarmonicsVectorized(negative_m_modes, inclination, -phi_ref)

    term_lm_conj_conjterm_lm__ = {}
    conjterm_lm_term_lm__conj = {}

    Qlm = {}
    Qlm["A"], Qlm["E"], Qlm["T"] = 0, 0, 0
    for i, mode in enumerate(modes):
        l, m = mode[0], mode[1]

        # This is used in calculating Ulmpq too, so we will save it now instead of calculating it again
        Sp_harm_lm, Sp_harm_lm_  = spherical_harmonics[:,i], negative_m_harmonics[:,i]
        Sp_harm_lm_conjugate, Sp_harm_lm__conjugate = np.conj((Sp_harm_lm)), np.conj(Sp_harm_lm_)

        term_lm_conj_conjterm_lm__[f"{l}_{m}"] = (terms)*Sp_harm_lm_conjugate + (-1)**(l)*(conj_terms)*(Sp_harm_lm_) 
        conjterm_lm_term_lm__conj[f"{l}_{m}"]  = conj_terms*(Sp_harm_lm) + (-1)**(l)*(terms)*Sp_harm_lm__conjugate

        for channel in ["A", "E", "T"]:
            Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_xx"].data.data.reshape(-1,1) * (term_lm_conj_conjterm_lm__[f"{l}_{m}"][0])
            Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_xy"].data.data.reshape(-1,1) * (term_lm_conj_conjterm_lm__[f"{l}_{m}"][1])
            Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_xz"].data.data.reshape(-1,1) * (term_lm_conj_conjterm_lm__[f"{l}_{m}"][2])
            Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_yy"].data.data.reshape(-1,1) * (term_lm_conj_conjterm_lm__[f"{l}_{m}"][3])
            Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_yz"].data.data.reshape(-1,1) * (term_lm_conj_conjterm_lm__[f"{l}_{m}"][4])
            Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_zz"].data.data.reshape(-1,1) * (term_lm_conj_conjterm_lm__[f"{l}_{m}"][5])
            # Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_xx"].data.data.reshape(-1,1) * (0.5*(plus_terms[0] - 1j*cross_terms[0])*Sp_harm_lm_conjugate + 0.5*(-1)**(l)*(plus_terms[0] + 1j*cross_terms[0])*(Sp_harm_lm_))
            # Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_xy"].data.data.reshape(-1,1) * (0.5*(plus_terms[1] - 1j*cross_terms[1])*Sp_harm_lm_conjugate + 0.5*(-1)**(l)*(plus_terms[1] + 1j*cross_terms[1])*(Sp_harm_lm_))
            # Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_xz"].data.data.reshape(-1,1) * (0.5*(plus_terms[2] - 1j*cross_terms[2])*Sp_harm_lm_conjugate + 0.5*(-1)**(l)*(plus_terms[2] + 1j*cross_terms[2])*(Sp_harm_lm_))
            # Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_yy"].data.data.reshape(-1,1) * (0.5*(plus_terms[3] - 1j*cross_terms[3])*Sp_harm_lm_conjugate + 0.5*(-1)**(l)*(plus_terms[3] + 1j*cross_terms[3])*(Sp_harm_lm_))
            # Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_yz"].data.data.reshape(-1,1) * (0.5*(plus_terms[4] - 1j*cross_terms[4])*Sp_harm_lm_conjugate + 0.5*(-1)**(l)*(plus_terms[4] + 1j*cross_terms[4])*(Sp_harm_lm_))
            # Qlm[channel] += Q_lm[f"{channel}_{l}_{m}_zz"].data.data.reshape(-1,1) * (0.5*(plus_terms[5] - 1j*cross_terms[5])*Sp_harm_lm_conjugate + 0.5*(-1)**(l)*(plus_terms[5] + 1j*cross_terms[5])*(Sp_harm_lm_))

    Ulmpq = {}
    Ulmpq["A"], Ulmpq["E"], Ulmpq["T"] = 0, 0, 0
    for i in np.arange(len(modes)):
        l, m = modes[i][0], modes[i][1]
        for j in np.arange(len(modes))[i:]:
            p, q =  modes[j][0], modes[j][1]
            for channel in ["A", "E", "T"]:
                temp_value = 0.0 # so we can exploit conjugate symmetry! (tested)
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_xx"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][0]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][0])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_xy"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][0]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][1])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_xz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][0]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][2])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_yy"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][0]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][3])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_yz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][0]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][4])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xx_{p}_{q}_zz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][0]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][5])

                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_xx"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][1]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][0])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_xy"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][1]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][1])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_xz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][1]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][2])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_yy"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][1]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][3])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_yz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][1]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][4])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xy_{p}_{q}_zz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][1]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][5])

                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_xx"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][2]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][0])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_xy"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][2]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][1])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_xz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][2]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][2])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_yy"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][2]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][3])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_yz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][2]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][4])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_xz_{p}_{q}_zz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][2]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][5])

                temp_value += U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_xx"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][3]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][0])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_xy"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][3]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][1])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_xz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][3]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][2])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_yy"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][3]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][3])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_yz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][3]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][4])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_yy_{p}_{q}_zz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][3]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][5])

                temp_value += U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_xx"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][4]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][0])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_xy"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][4]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][1])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_xz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][4]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][2])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_yy"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][4]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][3])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_yz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][4]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][4])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_yz_{p}_{q}_zz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][4]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][5])

                temp_value += U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_xx"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][5]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][0])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_xy"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][5]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][1])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_xz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][5]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][2])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_yy"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][5]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][3])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_yz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][5]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][4])
                temp_value += U_lm_pq[f"{channel}_{l}_{m}_zz_{p}_{q}_zz"]* (conjterm_lm_term_lm__conj[f"{l}_{m}"][5]) * (term_lm_conj_conjterm_lm__[f"{p}_{q}"][5])
                if (l,m) != (p,q):
                    temp_value += np.conj(temp_value)
                Ulmpq[channel] += temp_value
    total_lnL = 0
    for channel in ["A", "E", "T"]:
        total_lnL += np.real(reference_distance/distance * Qlm[channel] - ((reference_distance/distance)**2)*0.5*Ulmpq[channel])
    # for time sampling, return likelihood time series.
    if return_lnLt:
        return total_lnL
    # shape (time terms, extrinsic_params), integrating in time --> axis 0
    Q_lm_term = list(Q_lm.keys())[0]
    L_t = np.exp(total_lnL - np.max(total_lnL, axis=0))
    L = integrate.simpson(L_t, dx = Q_lm[Q_lm_term].deltaT, axis=0) #P.deltaT
    lnL  = np.max(total_lnL, axis=0) + np.log(L)
    return lnL


