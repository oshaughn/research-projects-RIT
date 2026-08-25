#
# gwsignal wrapper
#  
#  WHY IS THIS NOT IN lalsimutils
#     - gwsignal (via gwsurrogate) prints to stdout some stupid messages, which breaks several scripts which write to stdout
#     - gwsignal is a slow import with a lot of large dependencies, not well suited to our lowlatency use.


# References:
#   https://git.ligo.org/waveforms/new-waveforms-interface/-/blob/master/python_interface/docs/source/examples/example_usage.ipynb

import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lalsimutils
from RIFT.physics import teobresums_compat
import numpy as np
import astropy.units as u
from astropy.time import Time
from gwpy.timeseries import TimeSeries
import astropy.constants as ac
has_gws= False
try:
    # Warning: prints stupid messages to stdout
    try: # if hasattr(lalsim, 'gwsignal'):
        import lalsimulation.gwsignal as gws
        from lalsimulation.gwsignal.core import utils as ut
        from lalsimulation.gwsignal.core import waveform as wfm
    except: # else:
        import gwsignal as gws
        from gwsignal.core import utils as ut
        from gwsignal.core import waveform as wfm
    has_gws=True
except:
    has_gws=False
    print("GWsignal import failed")


def std_and_conj_hlmoff(P, Lmax=2,approx_string=None,**kwargs):
    """
    Generates both the Fourier-transformed harmonic modes and their complex conjugates.

    Args:
        P (ChooseWaveformParams): Waveform parameters.
        Lmax (int): Maximum l-mode to generate. Default is 2.
        approx_string (str): Approximant string. If None, P.approx is used.
        **kwargs: Additional arguments passed to hlmoft.

    Returns:
        tuple: (hlmsF, hlms_conj_F) where both are dictionaries mapping (l, m) to Fourier-transformed series.
    """
    hlms = hlmoft(P, Lmax,approx_string=approx_string,**kwargs)
    hlmsF = {}
    hlms_conj_F = {}
    for mode in hlms:
        hlmsF[mode] = lalsimutils.DataFourier(hlms[mode])
        hlms[mode].data.data = np.conj(hlms[mode].data.data)
        hlms_conj_F[mode] = lalsimutils.DataFourier(hlms[mode])
    return hlmsF, hlms_conj_F

def hlmoff(P, Lmax=2,approx_string=None,**kwargs):
    """
    Generates Fourier-transformed harmonic modes.

    Args:
        P (ChooseWaveformParams): Waveform parameters.
        Lmax (int): Maximum l-mode to generate. Default is 2.
        approx_string (str): Approximant string. If None, P.approx is used.
        **kwargs: Additional arguments passed to hlmoft.

    Returns:
        dict: A dictionary mapping (l, m) to Fourier-transformed series.
    """
    hlms = hlmoft(P, Lmax,approx_string=approx_string,**kwargs)
    hlmsF = {}
    for mode in hlms:
        hlmsF[mode] = lalsimutils.DataFourier(hlms[mode])
    return hlmsF


def hlmoft(P, Lmax=2,approx_string=None,no_trust_align_method=None,internal_phase_shift=np.pi/2, **kwargs):
    """
    Generates time-domain harmonic modes using the gwsignal library.

    Args:
        P (ChooseWaveformParams): Waveform parameters.
        Lmax (int): Maximum l-mode to generate. Default is 2.
        approx_string (str): Approximant string. If None, P.approx is used.
        no_trust_align_method (str): If 'peak', shifts epoch to the peak of the total signal power.
        internal_phase_shift (float): Phase shift applied to the modes. Default is pi/2.
        **kwargs: Additional arguments (e.g., 'lmax_nyquist', 'force_22_mode').
            force_22_mode (bool): If True, return only the (2,+-2) modes.

    Returns:
        dict: A dictionary mapping (l, m) to LAL COMPLEX16TimeSeries objects.
    """

    assert Lmax >= 2

    force_22_mode = kwargs.get('force_22_mode', False)

    # Check that masses are not nan!
    assert (not np.isnan(P.m1)) and (not np.isnan(P.m2)), " masses are NaN "
    taper=0
    if P.taper != lalsim.SIM_INSPIRAL_TAPER_NONE:
        taper = 1
    python_dict = {'mass1' : P.m1/lal.MSUN_SI * u.solMass,
              'mass2' : P.m2/lal.MSUN_SI * u.solMass,
              'spin1x' : P.s1x*u.dimensionless_unscaled,
              'spin1y' : P.s1y*u.dimensionless_unscaled,
              'spin1z' : P.s1z*u.dimensionless_unscaled,
              'spin2x' : P.s2x*u.dimensionless_unscaled,
              'spin2y' : P.s2y*u.dimensionless_unscaled,
              'spin2z' : P.s2z*u.dimensionless_unscaled,
              'deltaT' : P.deltaT*u.s,
              'f22_start' : P.fmin*u.Hz,
              'f22_ref': P.fref*u.Hz,
              'phi_ref' : P.phiref*u.rad,
              'distance' : P.dist/(1e6*lal.PC_SI)*u.Mpc,
              'inclination' : P.incl*u.rad,
              'eccentricity' : P.eccentricity*u.dimensionless_unscaled,
              'longAscNodes' : P.psi*u.rad,
              'meanPerAno' : P.meanPerAno*u.rad,
              'condition' : taper     }
    if 'lmax_nyquist' in kwargs:
        python_dict['lmax_nyquist'] = kwargs['lmax_nyquist']

    # if needed
#    lal_dict = gws.core.utils.to_lal_dict(python_dict)

    approx_string_here = approx_string
    if not(approx_string):
        approx_string_here = lalsim.GetStringFromApproximant(P.approx)

    # DO NOT remove this as cosmetic spin rounding.  TEOBResumS-DALI's C code
    # classifies sum(chi_perp) <= 1e-4 as aligned, while its GWSignal wrapper
    # requests inertial modes for any exactly nonzero transverse component.
    # That disagreement segfaults EOBRunPy in production DALI builds.  Zeroing
    # only the backend's own aligned interval makes both layers take the same
    # path; genuinely precessing spins above the boundary remain untouched.
    python_dict = teobresums_compat.guard_gwsignal_transverse_spins(
        python_dict, approx_string_here
    )

    # Fork on calling different generators
    gen = gws.models.gwsignal_get_waveform_generator(approx_string_here)
    # if "NRSur7dq4_gwsurr" == approx_string_here:
    #     gen =gws.NRSur7dq4_gwsurr()
    # elif approx_string_here == 'SEOBNRv5PHM':  # only available
    #     gen = gws.models.pyseobnr.SEOBNRv5PHM()
    # else:
    #     gen = wfm.LALCompactBinaryCoalescenceGenerator(approx_string_here)

    hlm = wfm.GenerateTDModes(python_dict,gen)
    tvals = hlm[(2,2)].times
    npts = len(tvals)
    epoch = float(tvals[0]/u.second)

    # Repack in conventional structure (typing)
    hlmT = {}
    for mode in hlm:
        if isinstance(mode, str):  # skip 'time_array'
            continue
        if mode[0] > Lmax:  # skip modes with L > Lmax
            continue
        # force_22_mode must actually produce a 22-only waveform here too, not
        # just on the lalsimutils path.  The restriction is applied to the
        # returned modes rather than to the generator arguments, because the
        # mode-restriction keyword is not uniformly supported by the generators
        # reachable through gwsignal_get_waveform_generator.
        if force_22_mode and not(mode[0] == 2 and abs(mode[1]) == 2):
            continue
        # 
        h = lal.CreateCOMPLEX16TimeSeries("hlm",
                lal.LIGOTimeGPS(0.), 0., P.deltaT, lal.DimensionlessUnit,
                npts)
        h.data.data = np.array(hlm[mode])
        h.epoch = epoch
        TDlen_orig = h.data.length  # size of data, pertinent for tapering!
        # Resize if needed
        if P.deltaF:
            TDlen = int(1./P.deltaF * 1./P.deltaT)
            if TDlen < h.data.length:   # Truncate the series to the desired length, removing data at the *start* (left)
                h = lal.ResizeCOMPLEX16TimeSeries(h,h.data.length-TDlen,TDlen)
            elif TDlen > h.data.length:   # Zero pad, extend at end
                h = lal.ResizeCOMPLEX16TimeSeries(h,0,TDlen)
        # WARNING:  realistically, the GWSignal mode output was NEVER tapered, oddly -- so do it by hand, following lalsimutils choices
        if taper:
            ntaper = int(0.01*np.min([TDlen_orig,h.data.length]) ) # DO NOT TAPER BASED ON RESIZING/EXTENDING, otherwise we taper due to zero pad!
            if P.fmin > 0: # avoid failure if waveform start frequency 0 is nominally specified
                ntaper = np.max([ntaper, int(1./(P.fmin*P.deltaT))]) 
            vectaper= 0.5 - 0.5*np.cos(np.pi*np.arange(ntaper)/(1.*ntaper))
            # Taper at the start of the segment
            h.data.data[:ntaper]*=vectaper
        # Apply phase shift
        h.data.data *= np.exp(1j*internal_phase_shift*mode[1])  # exp( i m phi_shift)
        # Add to structure
        hlmT[mode] = h

    # if no_trust_peak_method, we will change the epoch.  Standard option is 'peak', to find the peak value of |h|^2, summed over modes
    # Note there is *no interpolation* between samples, so the sampling rate will introduce some jitter.
    if no_trust_align_method == 'peak':
        rhosq = np.zeros(TDlen)
        for mode in hlmT:
            rhosq += np.abs(hlmT[mode].data.data)**2
        indx_max =np.argmax(rhosq)
        new_epoch = - indx_max*P.deltaT
        for mode in hlmT:
            hlmT[mode].epoch = new_epoch
    if approx_string_here == 'TEOBResumSDALI':
        nu = P.m1*P.m2/((P.m1+P.m2)**2)
        distance_rescaling = (
            (
                nu
                * (P.m1 + P.m2)
                / P.dist
                * ac.G
                / ac.c ** 2
            )
            .value
        )
        for mode in hlmT:
            # NOTE THE SIGN.  gwsignal's TEOBResumSDALI modes are MINUS the
            # polarization convention, not plus.  Getting this wrong is not
            # visible as a bad fit: a global sign on h is exactly
            # psi -> psi + pi/2, so it silently displaces the polarization
            # angle by a quarter turn and leaves every other parameter, and
            # the peak likelihood, looking fine.
            #
            # Separately, and NOT corrected here: TEOB's modes also do not
            # want the exp(i m phi_shift) applied above, which leaves RIFT's
            # reported `phase` for this approximant offset by pi/2 from the
            # polarization path.  That is a phase-convention difference, not
            # a sign error, and it is unchanged by this fix.
            hlmT[mode].data.data = -distance_rescaling*hlmT[mode].data.data
            
    return hlmT



#
# Functions to generate waveforms
#
def hoft(P, Fp=None, Fc=None,approx_string=None, **kwargs):
    """
    Generate a real-valued time-domain waveform from ChooseWaveformParams P.
    
    This function projects the h+ and hx polarizations onto a detector strain.
    If antenna patterns Fp and Fc are provided, they are used; otherwise, they are 
    computed from the information in P.

    Args:
        P (ChooseWaveformParams): Waveform parameters.
        Fp (float/array, optional): Antenna pattern F+.
        Fc (float/array, optional): Antenna pattern Fx.
        approx_string (str, optional): Approximant string. If None, P.approx is used.
        **kwargs: Additional arguments.

    Returns:
        lal.REAL8TimeSeries: The projected time-domain waveform.
    """

    # special sauce for EOB, because it is so finicky regarding
    if P.approx == lalsim.EOBNRv2HM and P.m1 == P.m2:
#        print " Using ridiculous tweak for equal-mass line EOB"
        P.m2 = P.m1*(1-1e-6)
    extra_params = P.to_lal_dict()


    assert (not np.isnan(P.m1)) and (not np.isnan(P.m2)), " masses are NaN "
    taper=0
    if P.taper != lalsim.SIM_INSPIRAL_TAPER_NONE:
        taper = 1
    python_dict = {'mass1' : P.m1/lal.MSUN_SI * u.solMass,
              'mass2' : P.m2/lal.MSUN_SI * u.solMass,
              'spin1x' : P.s1x*u.dimensionless_unscaled,
              'spin1y' : P.s1y*u.dimensionless_unscaled,
              'spin1z' : P.s1z*u.dimensionless_unscaled,
              'spin2x' : P.s2x*u.dimensionless_unscaled,
              'spin2y' : P.s2y*u.dimensionless_unscaled,
              'spin2z' : P.s2z*u.dimensionless_unscaled,
              'deltaT' : P.deltaT*u.s,
              'f22_start' : P.fmin*u.Hz,
              'f22_ref': P.fref*u.Hz,
              'phi_ref' : P.phiref*u.rad,
              'distance' : P.dist/(1e6*lal.PC_SI)*u.Mpc,
              'inclination' : P.incl*u.rad,
              'eccentricity' : P.eccentricity*u.dimensionless_unscaled,
              'longAscNodes' : P.psi*u.rad,
              'meanPerAno' : P.meanPerAno*u.rad,
              'condition' : taper}
    if 'lmax_nyquist' in kwargs:
        python_dict['lmax_nyquist'] = kwargs['lmax_nyquist']

    # if needed
#    lal_dict = gws.core.utils.to_lal_dict(python_dict)

    approx_string_here = approx_string
    if not(approx_string):
        approx_string_here = lalsim.GetStringFromApproximant(P.approx)

    # Apply the same native-backend safety boundary as hlmoft.  Keep this on
    # the polarization path too: callers may reach TEOBResumS through either
    # GWSignal entry point.
    python_dict = teobresums_compat.guard_gwsignal_transverse_spins(
        python_dict, approx_string_here
    )

    # Fork on calling different generators
    gen = gws.models.gwsignal_get_waveform_generator(approx_string_here)

    # gwsignal return values are sometimes gwsignal objects
    hp, hc = gws.core.waveform.GenerateTDWaveform(python_dict, gen)
    if not isinstance(hp, lal.REAL8TimeSeries):
        # gwpy.timeseries.timeseries.TimeSeries object
        hp_lal = lal.CreateREAL8TimeSeries("hp",
                lal.LIGOTimeGPS(0.), 0., P.deltaT, lal.DimensionlessUnit,
                len(hp.times))
        hc_lal = lal.CreateREAL8TimeSeries("hp",
                lal.LIGOTimeGPS(0.), 0., P.deltaT, lal.DimensionlessUnit,
                len(hc.times))
        hp_lal.data.data =hp.value
        hc_lal.data.data = hc.value
        if isinstance(hp.epoch, Time):
            dT = hp.epoch.to_value('gps','long')  # pull out the time
        else:
            dT = float(hp.epoch) # old-style
        hp_lal.epoch = dT
        hc_lal.epoch = dT
        hp = hp_lal
        hc = hc_lal

    if Fp!=None and Fc!=None:
        hp.data.data *= Fp
        hc.data.data *= Fc
        hp = lal.AddREAL8TimeSeries(hp, hc)
        ht = hp
    elif P.radec==False:
        fp = lalsimutils.Fplus(P.theta, P.phi, P.psi)
        fc = lalsimutils.Fcross(P.theta, P.phi, P.psi)
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
    if P.taper != lalsimutils.lsu_TAPER_NONE: # Taper if requested
        lalsim.SimInspiralREAL8WaveTaper(ht.data, P.taper)
    if P.deltaF is not None:
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen >= ht.data.length
        ht = lal.ResizeREAL8TimeSeries(ht, 0, TDlen)
    return ht


def complex_hoft(P, Fp=None, Fc=None,approx_string=None,sgn=-1, **kwargs):
    """
    Generate a complex-valued time-domain waveform (h+ + i * sgn * hx).

    Args:
        P (ChooseWaveformParams): Waveform parameters.
        Fp (float/array, optional): Antenna pattern F+.
        Fc (float/array, optional): Antenna pattern Fx.
        approx_string (str, optional): Approximant string. If None, P.approx is used.
        sgn (int): Sign for the imaginary part. Default is -1.
        **kwargs: Additional arguments.

    Returns:
        lal.COMPLEX16TimeSeries: The complex time-domain waveform.
    """

    # special sauce for EOB, because it is so finicky regarding
    if P.approx == lalsim.EOBNRv2HM and P.m1 == P.m2:
#        print " Using ridiculous tweak for equal-mass line EOB"
        P.m2 = P.m1*(1-1e-6)
    extra_params = P.to_lal_dict()


    assert (not np.isnan(P.m1)) and (not np.isnan(P.m2)), " masses are NaN "
    taper=0
    if P.taper != lalsim.SIM_INSPIRAL_TAPER_NONE:
        taper = 1
    python_dict = {'mass1' : P.m1/lal.MSUN_SI * u.solMass,
              'mass2' : P.m2/lal.MSUN_SI * u.solMass,
              'spin1x' : P.s1x*u.dimensionless_unscaled,
              'spin1y' : P.s1y*u.dimensionless_unscaled,
              'spin1z' : P.s1z*u.dimensionless_unscaled,
              'spin2x' : P.s2x*u.dimensionless_unscaled,
              'spin2y' : P.s2y*u.dimensionless_unscaled,
              'spin2z' : P.s2z*u.dimensionless_unscaled,
              'deltaT' : P.deltaT*u.s,
              'f22_start' : P.fmin*u.Hz,
              'f22_ref': P.fref*u.Hz,
              'phi_ref' : P.phiref*u.rad,
              'distance' : P.dist/(1e6*lal.PC_SI)*u.Mpc,
              'inclination' : P.incl*u.rad,
              'eccentricity' : P.eccentricity*u.dimensionless_unscaled,
              'longAscNodes' : P.psi*u.rad,
              'meanPerAno' : P.meanPerAno*u.rad,
              'condition' : taper}
    if 'lmax_nyquist' in kwargs:
        python_dict['lmax_nyquist'] = kwargs['lmax_nyquist']

    # if needed
#    lal_dict = gws.core.utils.to_lal_dict(python_dict)

    approx_string_here = approx_string
    if not(approx_string):
        approx_string_here = lalsim.GetStringFromApproximant(P.approx)

    # complex_hoft reaches the same GWSignal polarization generator as hoft;
    # keep its ResumS native call behind the same near-aligned safety boundary.
    python_dict = teobresums_compat.guard_gwsignal_transverse_spins(
        python_dict, approx_string_here
    )

    # Fork on calling different generators
    gen = gws.models.gwsignal_get_waveform_generator(approx_string_here)

    # gwsignal return values are sometimes gwsignal objects
    hp, hc = gws.core.waveform.GenerateTDWaveform(python_dict, gen)
    if not isinstance(hp, lal.REAL8TimeSeries):
        # gwpy.timeseries.timeseries.TimeSeries object
        hp_lal = lal.CreateREAL8TimeSeries("hp",
                lal.LIGOTimeGPS(0.), 0., P.deltaT, lal.DimensionlessUnit,
                len(hp.times))
        hc_lal = lal.CreateREAL8TimeSeries("hp",
                lal.LIGOTimeGPS(0.), 0., P.deltaT, lal.DimensionlessUnit,
                len(hc.times))
        hp_lal.data.data =hp.value
        hc_lal.data.data = hc.value
        if isinstance(hp.epoch, Time):
            dT = hp.epoch.to_value('gps','long')  # pull out the time
        else:
            dT = float(hp.epoch) # old-style
        hp_lal.epoch = dT
        hc_lal.epoch = dT
        hp = hp_lal
        hc = hc_lal


    ht = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hp.epoch, hp.f0, 
                                       hp.deltaT, lalsimutils.lsu_DimensionlessUnit, hp.data.length)
    ht.epoch = ht.epoch + P.tref
    ht.data.data = hp.data.data + 1j * sgn * hc.data.data
    # impose polarization directly, using precisely the conventions we demand
    ht.data.data*= np.exp(2j*sgn*P.psi)
    return ht
