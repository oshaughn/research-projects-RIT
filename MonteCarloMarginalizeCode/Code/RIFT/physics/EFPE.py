# Wrapper for EFPE models, for now just pyEFPE

# What is the plan?
# 1) Write a function that can output polarizations in time domain.
# 2) Use the PhenomPv2 approach to get the modes
# Need to be careful about epoch. Also the input frequency array should mimic that of time series.

# Functions 1) get_ISCO_fval 2) get_frequency_array 3) get_polarizations 4) complex_hoft 5) hoft 6) hlmoft 7) std_and_conj_hlmoff 
# What is this code doing? First, get_polarization gets the waveform polarizations from pyEFPE in FD, FFTs it to TD (while properly setting epoch) and returns it. I learned the hardway to have a cutoff on fmax so the fmax cutoff is ISCO. This function is utlized in complex_hoft which return hp-1jhc *exp(2j*psi). complex_hoft is used by hlmoft which is in turn used by std_and_conj_hlmoff (that is used in recovery). hoft uses get_polarizations to create injections.
# Potential sources of problems 1) How I construct hlms 2) Epoch 3) Injection creation. The lal detector series function adds more points to the time series. 

import lal
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import pyEFPE
import numpy as np
import RIFT.lalsimutils as lalsimutils
from astropy.time import Time

debug=True
use_ISCO_cutoff=False # FFT routines sometimes fail if I set this to True. Bizarre!

def get_ISCO_fval(P):
    """
    Compute the ISCO (innermost stable circular orbit) frequency for the binary system. This frequency serves as an upper frequency cutoff for waveform generation.

    Args:
        P: ChooseWaveformParams object.

    Returns:
        isco_fval (float): ISCO frequency in Hz.
    """
    # c^3/ (6^(3/2)*pi*G*M_tot)
    isco_fval = 4331.648896/(P.m1 + P.m2) * lal.MSUN_SI
    return isco_fval

def get_frequency_array(P, use_ISCO_cutoff=use_ISCO_cutoff):
    """
    Generate frequency arrays used for waveform generation
    Args:
        P: ChooseWaveformParams object
    Returns:
        fvals_wf (np.ndarray): Frequency array starting from the closest frequency to fmin, used for waveform generation. 
        fvals (np.ndarray): Full frequency array from 0 to Nyquist frequency (inclusive).
        index_fmin (int): Index in `fvals` corresponding to the first frequency, which is closes to fmin.
        index_fmax (int): Index in `fvals` corresponding to the first frequency, which is closes to f_isco.
    """
    # for real series the fvals go from 0 -> fNyq
    fvals = np.arange(0, 1/P.deltaT/2 + P.deltaF, P.deltaF) # 0 and fNyq included

    # find the fval closest to fmin
    index_fmin = np.argmin(np.abs(fvals - P.fmin))

    # fvals for waveform generation (NOTE: no upper cutoff here!)
    fvals_wf = fvals[index_fmin:]
    
    # generate the waveform only upto ISCO
    index_fmax = -1
    if use_ISCO_cutoff:
        isco_fval = get_ISCO_fval(P)
        index_fmax = np.argmin(np.abs(fvals - isco_fval))
        fvals_wf = fvals[index_fmin:index_fmax+1]

    if debug:
        print(f'Waveform being generated from {fvals_wf[0]} to {fvals_wf[-1]} Hz with deltaF {P.deltaF} Hz')

    return fvals_wf, fvals, index_fmin, index_fmax

def get_polarizations(P, return_in_FD=False, use_ISCO_cutoff=use_ISCO_cutoff):
    """
    Generate gravitational wave polarizations (h₊, h×) for a binary system using pyEFPE.
    Args:
        P: ChooseWaveformParams object
        return_in_FD (bool): If True, return frequency-domain polarizations; otherwise, return time-domain versions.
    Returns:
        hp, hc: 
            If return_in_FD is False:
                hp (lal.REAL8TimeSeries): Time-domain h₊ polarization.
                hc (lal.REAL8TimeSeries): Time-domain h× polarization.
            If return_in_FD is True:
                hp_f (lal.COMPLEX16FrequencySeries): Frequency-domain h₊ polarization.
                hc_f (lal.COMPLEX16FrequencySeries): Frequency-domain h× polarization.
    """
    params = {
    'mass1': P.m1/lal.MSUN_SI,       # Mass of companion 1 (solar masses)
    'mass2': P.m2/lal.MSUN_SI,       # Mass of companion 2 (solar masses)
    'e_start':P.eccentricity,        # Initial eccentricity
    'mean_anomaly_start': P.meanPerAno, # initial mean anomaly of quasi keplerian parametrization
    'spin1x': P.s1x,                 # Spin components of companion 1
    'spin1y': P.s1y,
    'spin1z': P.s1z,
    'spin2x': P.s2x,                 # Spin components of companion 2
    'spin2y': P.s2y,
    'spin2z': P.s2z,
    'inclination': P.incl,           # Initial binary inclination (radians)
    'phi_start': P.phiref,           # Initial orbital phase (radians)
    'f22_start': P.fmin,             # Starting (simulation) waveform frequency of GW 22 mode (Hz)
    'distance': P.dist/1e6/lal.PC_SI # Distance to the source, in Mpc
    }

    if debug:
        print(f'\nGenerating polarizations for params = {params}')
    
    # Initialize pyEFPE waveform model
    wf = pyEFPE.pyEFPE(params)

    # Define frequency array for waveform generation
    freqs, freqs_full, index_fmin, index_fmax = get_frequency_array(P, use_ISCO_cutoff=use_ISCO_cutoff)

    # Compute frequency-domain gravitational wave polarizations
    hpf, hcf = wf.generate_waveform(freqs)

    # Create lal objects so we can compute FFT
    # hp
    hp_f = lal.CreateCOMPLEX16FrequencySeries("Template hp(f)",
            0.0, 0.0, P.deltaF, lal.HertzUnit,  # epoch set to 0.0 and so is f0. Will set epoch when in TD
            len(freqs_full))
    hp_f.data.data *= 0.0
    if use_ISCO_cutoff:
        hp_f.data.data[index_fmin:index_fmax+1] = hpf
    else:
        hp_f.data.data[index_fmin:] = hpf
    
    # hc
    hc_f = lal.CreateCOMPLEX16FrequencySeries("Template hc(f)",
            0.0, 0.0, P.deltaF, lal.HertzUnit,  # epoch set to 0.0 and so is f0. Will set epoch when in TD
            len(freqs_full))
    hc_f.data.data *= 0.0
    if use_ISCO_cutoff:
        hc_f.data.data[index_fmin:index_fmax+1] = hcf
    else:
        hc_f.data.data[index_fmin:] = hcf

    # return FD polarizations
    if return_in_FD:
        return hp_f, hc_f

    # FFT
    hp_t, hc_t = lalsimutils.DataInverseFourierREAL8(hp_f), lalsimutils.DataInverseFourierREAL8(hc_f)
    
    # roll them so the polarizations epoch is positioned at center
    tchirp = lalsim.SimInspiralChirpTimeBound(P.fmin, P.m1,P.m2, P.s1z, P.s2z)
    s = lalsim.SimInspiralFinalBlackHoleSpinBound(P.s1z,P.s2z)
    t_merge = lalsim.SimInspiralMergeTimeBound(P.m1,P.m2) + lalsim.SimInspiralRingdownTimeBound(P.m1+P.m2,s)
    factor_roll = 0.98
    assert t_merge < (1-factor_roll)*hp_t.data.length*hp_t.deltaT, f'The waveform is being rolled by factor {factor_roll} and this is causing wraparound. The predicted merge time is {t_merge}s and time left for it is {(1-factor_roll)*hp_t.data.length*hp_t.deltaT}, tchirp = {tchirp} {factor_roll*hp_t.data.length*hp_t.deltaT}'

    hp_t.data.data = np.roll(hp_t.data.data, int(factor_roll*hp_t.data.length))
    hc_t.data.data = np.roll(hc_t.data.data, int(factor_roll*hc_t.data.length))
    
    # find epoch: based on the apporach in GWSignal.py
    amplitude = np.sqrt(hp_t.data.data**2 + hc_t.data.data**2)
    max_amp_index = np.argmax(amplitude)
    hp_t.epoch, hc_t.epoch = -max_amp_index * hp_t.deltaT, -max_amp_index * hc_t.deltaT
    if debug:
        print(f'Length in FD = {hp_f.data.length}, length in TD = {hp_t.data.length, hp_t.data.length * hp_t.deltaT}, epoch set to {hp_t.epoch}')


    return hp_t, hc_t


def complex_hoft(P, sgn=-1, use_ISCO_cutoff=use_ISCO_cutoff):
    """
    Generate the complex strain h(t) = h₊(t) - i·h×(t) using the pyEFPE waveform model.
    Args:
        P: ChooseWaveformParams object
        sgn (int, optional): Sign convention for constructing complex strain.
            Default is -1, corresponding to h(t) = h₊(t) - i·h×(t).
    Returns:
        hT (lal.COMPLEX16TimeSeries): Time-domain complex gravitational wave strain.
    """
    # Generate time-domain polarizations
    hp_t, hc_t = get_polarizations(P, return_in_FD=False, use_ISCO_cutoff=use_ISCO_cutoff)
 
    # Create complex strain time series h(t) = h₊(t) - i·h×(t)
    hT = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hp_t.epoch, hp_t.f0,
            hp_t.deltaT, lal.DimensionlessUnit, hp_t.data.length)
    hT.data.data *= 0.0 
    hT.data.data = np.real(hp_t.data.data) + 1j * sgn * np.real(hc_t.data.data)

    # Include polarization
    hT.data.data *= np.exp(2j*sgn*P.psi)
    
    return hT

def hoft(P, Fp=None, Fc=None, use_ISCO_cutoff=use_ISCO_cutoff, **kwargs):
    """
    Generate the detector strain time series h(t) from gravitational wave polarizations.
    Args:
        P: ChooseWaveformParams object
        Fp (float, optional): Antenna pattern projection factor for the plus polarization.
        Fc (float, optional): Antenna pattern projection factor for the cross polarization.
        **kwargs: Additional keyword arguments (currently unused).
    Returns:
        ht (lal.REAL8TimeSeries): Detector strain time series h(t).
    """
    # Call polarizations in TD
    P_copy = P.manual_copy()
    P_copy.deltaF = 1.01 * P.deltaF # Why? 1/DeltaF is T, but SimDetectorStrainREAL8TimeSeries adds more points so by increasing it, I am decreasing the length, allowing the addition of points to not break the code. This should be checked for signals where the signal length is almost the same as data length.
    hp, hc = get_polarizations(P_copy, use_ISCO_cutoff=use_ISCO_cutoff) 
    
    # Apply detector response
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
        # If astropy Time function, overwrite with GPS time, otherwise use normal addition
        if isinstance(hp.epoch, Time):
            dT = hp.epoch.to_value('gps','long')  # pull out the time
            hp.epoch = P.tref + dT
            hc.epoch = P.tref +dT
        else:
            hp.epoch = hp.epoch + P.tref
            hc.epoch = hc.epoch + P.tref
        ht = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc,
                P.phi, P.theta, P.psi,
                lalsim.DetectorPrefixToLALDetector(str(P.detector)))
    
    # lalsimulation tapering: not sufficient for high SNRs
    if P.taper != lalsimutils.lsu_TAPER_NONE:
        lalsim.SimInspiralREAL8WaveTaper(ht.data, P.taper)

    # Resize such that TDlen = 1/deltaF
    if P.deltaF is not None:
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        # hp and hc were already at length such that TDlen = 1/deltaF, but lalsim.SimDetectorStrainREAL8TimeSeries adds a few points. Removing this assert so it resizes to desired size
        assert TDlen >= ht.data.length, f"TDlen = {TDlen}, data_length = {ht.data.length}, 1/deltaT = {1/P.deltaT}, 1/deltaF = {1/P.deltaF}"
        if TDlen < ht.data.length:
            print(f'Data removed from {TDlen}:{ht.data.length}. Values = {ht.data.data[TDlen:ht.data.length]}')
        ht = lal.ResizeREAL8TimeSeries(ht, 0, TDlen)
    
    # Match lalsimutils tapering: Do we need to taper given it is a FD model?
    try:
        taper=False # Don't taper, it is already an FD model.
        if taper :
            ntaper = int(0.01*TDlen)
            if P.fmin > 0: # avoid failure if waveform start frequency 0 is nominally specified
                ntaper = np.max([ntaper, int(1./(P.fmin*P.deltaT))])
            vectaper= 0.5 - 0.5*np.cos(np.pi*np.arange(ntaper)/(1.*ntaper))
            # Taper at the start of the segment
            ht.data.data[:ntaper]*=vectaper
    except Exception as e:
        print("Couldn't apply tapering", e)
    return ht

def hlmoft(P, Lmax=2, use_ISCO_cutoff=use_ISCO_cutoff, **kwargs):
    mtxAngularForward = np.zeros((5,5),dtype=np.complex64)
    # Organize points so the (2,-2) and (2,2) mode clearly have contributions only from one point
    # Problem with points in equatorial plane: pure real signal in aligned-spin limit! Degeneracy!
    paramsForward =[ [np.pi,0], [np.pi/3,0], [np.pi/2, 0], [2*np.pi/3, 0], [0,0]];
    mvals = [-2,-1,0,1,2]
    for indx in np.arange(len(paramsForward)):
        for indx2 in np.arange(5):
            m = mvals[indx2]
            th,ph = paramsForward[indx]
            mtxAngularForward[indx,indx2] = lal.SpinWeightedSphericalHarmonic(th,-ph,-2,2,m)  # note phase sign
    mtx = np.linalg.inv(mtxAngularForward)


    # Now generate solutions at these values
    P_copy = P.manual_copy()
    P_copy.tref =0  # we do not need or want this offset when constructing hlm
    P_copy.psi = 0 # we want to force a certain polarization

    hTC_list = []
    for indx in np.arange(len(paramsForward)):
        th,ph = paramsForward[indx]
#        P_copy.assign_param('thetaJN', th)  # to be CONSISTENT with h(t) produced by ChooseTD, need to use incl here (!!)
        P_copy.incl = th  # to be CONSISTENT with h(t) produced by ChooseTD, need to use incl here (!!)
        P_copy.phiref = ph
        # Note argument change!  Hack to recover reasonable hlm modes for (2,+/-1), (2,0)
        # Something is funny about how SimInspiralFD/etc applies all these coordinate transforms
        if P_copy.fref ==0:
            P_copy.fref = P_copy.fmin

        hTC = complex_hoft(P_copy, use_ISCO_cutoff=use_ISCO_cutoff)
        hTC_list.append(hTC)

    # Now construct the hlm from this sequence
    hlmT = {}
    for indx2 in np.arange(5):
        m = mvals[indx2]
        hlmT[(2,m)] = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hTC.epoch, hTC.f0,
            hTC.deltaT, lal.DimensionlessUnit, hTC.data.length)
        hlmT[(2,m)].epoch = float(hTC_list[0].epoch)
        hlmT[(2,m)].data.data *=0   # this is needed, memory is not reliably cleaned
        for indx in np.arange(len(paramsForward)):
            hlmT[(2,m)].data.data += mtx[indx2,indx] * hTC_list[indx].data.data
    return hlmT

def std_and_conj_hlmoff(P, Lmax=2, use_ISCO_cutoff=use_ISCO_cutoff, **kwargs):
    """
    Generate frequency-domain spherical harmonic modes (hlm) and their complex conjugates for RIFT precomputation.

    Args:
        P : ChooseWaveformParams
        Lmax : int, optional
            Maximum spherical harmonic multipole order to include (default is 2).
        **kwargs : dict
            Additional keyword arguments passed to `hlmoft`.

    Returns:
        hlmsF : dict
            Dictionary mapping (l, m) mode tuples to the frequency-domain h_{lm}(f).
        hlms_conj_F : dict
            Dictionary mapping (l, m) mode tuples to the frequency-domain of the conjugated h_{lm}^*(f).
    """
    hlms = hlmoft(P, Lmax, use_ISCO_cutoff=use_ISCO_cutoff, **kwargs)
    hlmsF = {}
    hlms_conj_F = {}
    for mode in hlms:
        hlmsF[mode] = lalsimutils.DataFourier(hlms[mode])
        hlms[mode].data.data = np.conj(hlms[mode].data.data)
        hlms_conj_F[mode] = lalsimutils.DataFourier(hlms[mode])
    return hlmsF, hlms_conj_F
