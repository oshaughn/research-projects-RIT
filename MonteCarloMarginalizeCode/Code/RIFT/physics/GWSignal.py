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
import numpy as np
import astropy.units as u
from astropy.time import Time
from gwpy.timeseries import TimeSeries

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
    hlms = hlmoft(P, Lmax,approx_string=approx_string,**kwargs)
    hlmsF = {}
    hlms_conj_F = {}
    for mode in hlms:
        hlmsF[mode] = lalsimutils.DataFourier(hlms[mode])
        hlms[mode].data.data = np.conj(hlms[mode].data.data)
        hlms_conj_F[mode] = lalsimutils.DataFourier(hlms[mode])
    return hlmsF, hlms_conj_F

def hlmoff(P, Lmax=2,approx_string=None,**kwargs):
    hlms = hlmoft(P, Lmax,approx_string=approx_string,**kwargs)
    hlmsF = {}
    for mode in hlms:
        hlmsF[mode] = lalsimutils.DataFourier(hlms[mode])
    return hlmsF


def hlmoft(P, Lmax=2,approx_string=None,**kwargs):
    """
    gwsignal.  Note the call will use approx_string, NOT a lalsimulation mode ID.  If approx_string is none, use P.approx but convert to string
    """

    assert Lmax >= 2

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
              'condition' : taper}
    if 'lmax_nyquist' in kwargs:
        python_dict['lmax_nyquist'] = kwargs['lmax_nyquist']

    # if needed
#    lal_dict = gws.core.utils.to_lal_dict(python_dict)

    approx_string_here = approx_string
    if not(approx_string):
        approx_string_here = lalsim.GetStringFromApproximant(P.approx)

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
        # 
        h = lal.CreateCOMPLEX16TimeSeries("hlm",
                lal.LIGOTimeGPS(0.), 0., P.deltaT, lal.DimensionlessUnit,
                npts)
        h.data.data = np.array(hlm[mode])
        h.epoch = epoch
        # Resize if needed
        if P.deltaF:
            TDlen = int(1./P.deltaF * 1./P.deltaT)
            if TDlen < h.data.length:   # Truncate the series to the desired length, removing data at the *start* (left)
                h = lal.ResizeCOMPLEX16TimeSeries(h,h.data.length-TDlen,TDlen)
            elif TDlen > h.data.length:   # Zero pad, extend at end
                h = lal.ResizeCOMPLEX16TimeSeries(h,0,TDlen)
        # Add to structure
        hlmT[mode] = h

    return hlmT



#
# Functions to generate waveforms
#
def hoft(P, Fp=None, Fc=None,approx_string=None, **kwargs):
    """
    Generate a TD waveform from ChooseWaveformParams P
    Based on https://git.ligo.org/waveforms/reviews/newwfinterface/-/blob/main/example_usage/example_usage_using_gwsignal_in_lalsimulation.ipynb
    You may pass in antenna patterns Fp, Fc. If none are provided, they will
    be computed from the information in ChooseWaveformParams.

    Returns a REAL8TimeSeries object
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
    if P.taper != lalsimutils.lsu_TAPER_NONE: # Taper if requested
        lalsim.SimInspiralREAL8WaveTaper(ht.data, P.taper)
    if P.deltaF is not None:
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen >= ht.data.length
        ht = lal.ResizeREAL8TimeSeries(ht, 0, TDlen)
    return ht
