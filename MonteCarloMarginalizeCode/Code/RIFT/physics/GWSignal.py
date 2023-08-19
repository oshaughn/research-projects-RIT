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
