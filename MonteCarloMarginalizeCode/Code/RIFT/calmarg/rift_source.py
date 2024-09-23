"""
Source functions to match how RIFT generates waveforms
"""

import numpy as np
import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lalsimutils
from RIFT.likelihood.factored_likelihood import internal_hlm_generator
import matplotlib
try:
    from bilby.core import utils
except:
    True

has_GWS=False  # make sure defined in top-level scope
try:
        import RIFT.physics.GWSignal as rgws
        has_GWS=True
except:
        has_GWS=False


def RIFT_lal_binary_black_hole_orig(
        frequency_array, mass_1, mass_2, luminosity_distance, spin_1x, spin_1y, spin_1z,
        spin_2x, spin_2y, spin_2z, lambda_1, lambda_2, iota, phase, **kwargs):

    waveform_kwargs = dict(
        waveform_approximant='SEOBNRv4PHM', reference_frequency=15.0,
        minimum_frequency=15.0, maximum_frequency=frequency_array[-1], Lmax=4,
        sampling_frequency=2*frequency_array[-1],
        extra_waveform_kwargs={})
    waveform_kwargs.update(kwargs)
    waveform_approximant = waveform_kwargs['waveform_approximant']
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']
    maximum_frequency = waveform_kwargs['maximum_frequency']
    sampling_frequency = waveform_kwargs['sampling_frequency']
    Lmax = waveform_kwargs['Lmax']
    waveform_dictionary = waveform_kwargs.get(
        'lal_waveform_dictionary', lal.CreateDict()
    )
    h_method = 'hlmoft'
    extra_waveform_kwargs = waveform_kwargs['extra_waveform_kwargs']
    if 'h_method' in kwargs:
        h_method = kwargs['h_method']

    approximant = lalsim.GetApproximantFromString(waveform_approximant)
    
    P = lalsimutils.ChooseWaveformParams()
    P.m1 = mass_1 * lal.MSUN_SI
    P.m2 = mass_2 * lal.MSUN_SI
    P.s1x = spin_1x; P.s1y = spin_1y; P.s1z = spin_1z
    P.s2x = spin_2x; P.s2y = spin_2y; P.s2z = spin_2z
    P.lambda1 = lambda_1; P.lambda2 = lambda_2
    P.deltaT = 1./(sampling_frequency)

    P.fmin = float(minimum_frequency)
    P.fmax = float(maximum_frequency)
    P.fref = float(reference_frequency)
    P.deltaF=frequency_array[1]-frequency_array[0]
    P.incl = iota
    P.phiref = phase
    P.dist=luminosity_distance*lal.PC_SI*1e6
    P.approx = approximant
    P.taper = lalsim.SIM_INSPIRAL_TAPER_START

    if h_method == 'hlmoft':
        # Waveform generator used internally in RIFT. ILE internally assumes phiref==0, so set this.
        # Note ILE also assumes L-frame waveforms, so this will not work as expected for J-frame output
        # Note several underlying interfaces like ChooseTDModes will enforce these conditions already, but not all. Better safe than sorry.
        P.phiref = 0
        P.incl = 0  # L direction frame
        hlmT = lalsimutils.hlmoft(P,Lmax=Lmax,extra_waveform_kwargs=extra_waveform_kwargs) # extra needed to control ChooseFDWaveform
        P.phiref = phase
        P.incl =  iota # restore
        h22T = hlmT[(2,2)]
        hT = lal.CreateCOMPLEX16TimeSeries("hoft", h22T.epoch, h22T.f0, h22T.deltaT, h22T.sampleUnits, h22T.data.length)
        hT.data.data = np.zeros(hT.data.length)

        # combine modes
        phase_offset = 0#np.pi/2 # TODO this could be a different value i.e. np.pi/2
        for mode in hlmT:
            hT.data.data += hlmT[mode].data.data * lal.SpinWeightedSphericalHarmonic(
                P.incl,phase_offset - 1.0*P.phiref, -2, int(mode[0]), int(mode[1]))
    elif h_method == 'gws_hlmoft':
        # gwsignal specific
        P.phiref = 0
        P.incl = 0  # L direction frame
        hlmT = rgws.hlmoft(P,Lmax=Lmax,approx_string=waveform_approximant,**extra_waveform_kwargs)
        P.phiref = phase
        P.incl =  iota # restore
        h22T = hlmT[(2,2)]
        hT = lal.CreateCOMPLEX16TimeSeries("hoft", h22T.epoch, h22T.f0, h22T.deltaT, h22T.sampleUnits, h22T.data.length)
        hT.data.data = np.zeros(hT.data.length)

        # combine modes
        phase_offset = 0#np.pi/2 # TODO this could be a different value i.e. np.pi/2
        for mode in hlmT:
            hT.data.data += hlmT[mode].data.data * lal.SpinWeightedSphericalHarmonic(
                P.incl,phase_offset - 1.0*P.phiref, -2, int(mode[0]), int(mode[1]))
    else:
        # Backstop waveform generator. Includes all L modes. Use cases where waveforms are J-frame calculations for hlm.
        hT = lalsimutils.complex_hoft(P)
    tvals = lalsimutils.evaluate_tvals(hT)
    t_max = tvals[np.argmax(np.abs(hT.data.data))]

    # end max is cutting the signal such that it ends 2s after merger
    n_max = np.argmax(np.abs(hT.data.data))

    hp = lal.CreateREAL8TimeSeries("h(t)", hT.epoch, hT.f0, hT.deltaT, hT.sampleUnits, hT.data.length)
    hp.data.data = np.real(hT.data.data)
    hc = lal.CreateREAL8TimeSeries("h(t)", hT.epoch, hT.f0, hT.deltaT, hT.sampleUnits, hT.data.length)
    hc.data.data = -np.imag(hT.data.data)

    lalsim.SimInspiralREAL8WaveTaper(hp.data, P.taper)
    lalsim.SimInspiralREAL8WaveTaper(hc.data, P.taper)

    h_plus = hp.data.data
    h_plus = np.concatenate([h_plus[n_max:], h_plus[:n_max]])
    h_cross = hc.data.data
    h_cross = np.concatenate([h_cross[n_max:], h_cross[:n_max]])

    hf_p, freqs = utils.nfft(h_plus, sampling_frequency)
    hf_c, freqs = utils.nfft(h_cross, sampling_frequency)

    #hf_p *= np.exp(2j*np.pi * freqs * tvals[0])
    #hf_c *= np.exp(2j*np.pi * freqs * tvals[0])

    return dict(plus=hf_p, cross=hf_c)


def RIFT_lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, spin_1x, spin_1y, spin_1z,
        spin_2x, spin_2y, spin_2z,iota, phase, **kwargs):

    waveform_kwargs = dict(
        waveform_approximant='SEOBNRv4PHM', reference_frequency=15.0,
        minimum_frequency=15.0, maximum_frequency=frequency_array[-1], Lmax=4,
        sampling_frequency=2*frequency_array[-1],
        extra_waveform_kwargs={})
    waveform_kwargs.update(kwargs)
    waveform_approximant = waveform_kwargs['waveform_approximant']
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']
    maximum_frequency = waveform_kwargs['maximum_frequency']
    sampling_frequency = waveform_kwargs['sampling_frequency']
    Lmax = waveform_kwargs['Lmax']
    waveform_dictionary = waveform_kwargs.get(
        'lal_waveform_dictionary', lal.CreateDict()
    )
    h_method = 'hoft'
    extra_waveform_kwargs = waveform_kwargs['extra_waveform_kwargs']
    if 'h_method' in kwargs:
        h_method = kwargs['h_method']

    approximant = lalsim.GetApproximantFromString(waveform_approximant)
    
    P = lalsimutils.ChooseWaveformParams()
    P.m1 = mass_1 * lal.MSUN_SI
    P.m2 = mass_2 * lal.MSUN_SI
    P.s1x = float(spin_1x); P.s1y = float(spin_1y); P.s1z = float(spin_1z)
    P.s2x = float(spin_2x); P.s2y = float(spin_2y); P.s2z = float(spin_2z)
    P.deltaT = 1./(sampling_frequency)

    P.fmin = float(minimum_frequency)
    P.fmax = float(maximum_frequency)
    P.fref = float(reference_frequency)
    P.deltaF=frequency_array[1]-frequency_array[0]
    P.incl = float(iota)
    P.phiref = float(phase)
    P.dist=luminosity_distance*lal.PC_SI*1e6
    P.approx = approximant
    P.taper = lalsim.SIM_INSPIRAL_TAPER_START
    
    if h_method == 'hlmoft':
        # Waveform generator used internally in RIFT. ILE internally assumes phiref==0, so set this.
        # Note ILE also assumes L-frame waveforms, so this will not work as expected for J-frame output
        # Note several underlying interfaces like ChooseTDModes will enforce these conditions already, but not all. Better safe than sorry.
        P.phiref = 0
        P.incl = 0  # L direction frame
        hlmT = lalsimutils.hlmoft(P,Lmax=Lmax,extra_waveform_kwargs=extra_waveform_kwargs) # extra needed to control ChooseFDWaveform
        P.phiref = phase
        P.incl =  iota # restore

        # combine modes
        hT = lalsimutils.hoft_from_hlm(hlmT, P, return_complex=True)
        
    elif h_method == 'gws_hlmoft':
        # gwsignal specific
        P.phiref = 0
        P.incl = 0  # L direction frame
        hlmT = rgws.hlmoft(P,Lmax=Lmax,approx_string=waveform_approximant,**extra_waveform_kwargs)
        P.phiref = phase
        P.incl =  iota # restore

        # combine modes
        hT = lalsimutils.hoft_from_hlm(hlmT, P, return_complex=True)

    elif h_method == 'internal_hlmoft':
        # gwsignal specific
        P.phiref = 0
        P.incl = 0  # L direction frame
        hlmF_1, _= factored_likelihood.internal_hlm_generator(P_copy, opts.Lmax, use_gwsignal=opts.use_gwsignal, use_gwsignal_approx=opts.approximant,ROM_group=opts.rom_group,ROM_param=opts.rom_param, extra_waveform_kwargs=extra_waveform_args, **extra_args)
        hlmT_1  = {}
        for mode in hlmF_1:
            #print(mode,hlmF_1[mode].data.data[0])
            hlmT_1[mode] = lalsimutils.DataInverseFourier(hlmF_1[mode])
        P.phiref = phase
        P.incl =  iota # restore

        # combine modes
        hT = lalsimutils.hoft_from_hlm(hlmT_1, P, return_complex=True)

        
    else:
        # Backstop waveform generator. Includes all L modes. Use cases where waveforms are J-frame calculations for hlm.
        hT = lalsimutils.complex_hoft(P)
    tvals = lalsimutils.evaluate_tvals(hT)
    t_max = tvals[np.argmax(np.abs(hT.data.data))]

    # end max is cutting the signal such that it ends 2s after merger
    n_max = np.argmax(np.abs(hT.data.data))

    hp = lal.CreateREAL8TimeSeries("h(t)", hT.epoch, hT.f0, hT.deltaT, hT.sampleUnits, hT.data.length)
    hp.data.data = np.real(hT.data.data)
    hc = lal.CreateREAL8TimeSeries("h(t)", hT.epoch, hT.f0, hT.deltaT, hT.sampleUnits, hT.data.length)
    hc.data.data = -np.imag(hT.data.data)

    lalsim.SimInspiralREAL8WaveTaper(hp.data, P.taper)
    lalsim.SimInspiralREAL8WaveTaper(hc.data, P.taper)

    h_plus = hp.data.data
    h_plus = np.concatenate([h_plus[n_max:], h_plus[:n_max]])
    h_cross = hc.data.data
    h_cross = np.concatenate([h_cross[n_max:], h_cross[:n_max]])

    hf_p, freqs = utils.nfft(h_plus, sampling_frequency)
    hf_c, freqs = utils.nfft(h_cross, sampling_frequency)

    #hf_p *= np.exp(2j*np.pi * freqs * tvals[0])
    #hf_c *= np.exp(2j*np.pi * freqs * tvals[0])

    # PROBLEM: now we need to interpolate to frequency_array : our grid does not correspond to frequency_array in general
    if frequency_array[0] >0:
        hf_p_out = np.zeros(frequency_array.shape,dtype=np.complex128)
        hf_c_out = np.zeros(frequency_array.shape,dtype=np.complex128)
        # interpolate in real, imag
        hf_p_out += np.interp(frequency_array, freqs, np.real(hf_p))
        hf_p_out += 1j* np.interp(frequency_array, freqs, np.imag(hf_p))
        hf_c_out += np.interp(frequency_array, freqs, np.real(hf_c))
        hf_c_out += 1j* np.interp(frequency_array, freqs, np.imag(hf_c))
        hf_p = hf_p_out
        hf_c = hf_c_out
    
    return dict(plus=hf_p, cross=hf_c)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # For testing purposes we evaluate one waveform and plot it here
    # test with two different maximum frequencies
    frequency_array = np.arange(15, 1024., 1./8)
    waveform_kwargs = {'Lmax':4, 'maximum_frequency':1024, 'minimum_frequency':10}
    waveform_polarizations = RIFT_lal_binary_black_hole(
        frequency_array, 60., 55., 400., 0.0, 0.0, 0.1,
        0.0, 0.0, 0.1, iota=np.pi/4, phase=np.pi/2, **waveform_kwargs)

    hf_p = waveform_polarizations['plus']
    hf_c = waveform_polarizations['cross']

    waveform_kwargs = {'Lmax':4, 'maximum_frequency':4096, 'minimum_frequency':10}
    waveform_polarizations = RIFT_lal_binary_black_hole(
        frequency_array, 60., 55., 400., 0.0, 0.0, 0.1,
        0.0, 0.0, 0.1, iota=np.pi/4, phase=np.pi/2, **waveform_kwargs)

    hf_p2 = waveform_polarizations['plus'][:int(len(waveform_polarizations['plus'])/4)]
    hf_c2 = waveform_polarizations['cross'][:int(len(waveform_polarizations['plus'])/4)]

    plt.axvline(15, color='k')

    plt.loglog(np.linspace(0,frequency_array[-1],len(hf_p)), np.abs(hf_p), color='C0')
    plt.loglog(np.linspace(0,frequency_array[-1],len(hf_p)), np.abs(hf_c), color='C0', linestyle='--')
    plt.loglog(np.linspace(0,frequency_array[-1],len(hf_p2)), np.abs(hf_p2), color='C1')
    plt.loglog(np.linspace(0,frequency_array[-1],len(hf_p2)), np.abs(hf_c2), color='C1', linestyle='--')
    plt.show()
    plt.clf()

    # be aware shifts by 2pi are common, plot so we can see that shift
    plt.semilogx(np.linspace(0,frequency_array[-1],len(hf_p)), np.unwrap(np.angle(hf_p)), color='C0')
    plt.semilogx(np.linspace(0,frequency_array[-1],len(hf_p)), np.unwrap(np.angle(hf_c)), color='C0', linestyle='--')
    plt.semilogx(np.linspace(0,frequency_array[-1],len(hf_p2)), np.unwrap(np.angle(hf_p2)), color='C1')
    plt.semilogx(np.linspace(0,frequency_array[-1],len(hf_p2)), np.unwrap(np.angle(hf_c2)), color='C1', linestyle='--')
    plt.show()
    plt.clf()
