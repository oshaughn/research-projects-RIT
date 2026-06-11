"""Compatibility helpers for the standalone LISA ILE fork.

These routines are adapted from Aasim Z. Jan's LISA-RIFT
``lisa_rift_paper`` branch.  Keep LISA-only shims here while the normal
``RIFT.lalsimutils`` and ILE driver continue to evolve independently.
"""

import os
import sys

import h5py
import lal
import lalsimulation as lalsim
import numpy as np

import RIFT.lalsimutils as lalsimutils


def _filter_modes(hlms, modes):
    if modes is None:
        return hlms

    requested_modes = {tuple(map(int, mode)) for mode in modes}
    return {mode: hlm for mode, hlm in hlms.items() if mode in requested_modes}


def hlmoff_for_LISA(
    P,
    Lmax=4,
    modes=None,
    fd_standoff_factor=0.964,
    fd_alignment_postevent_time=None,
    path_to_NR_hdf5=None,
    **kwargs
):
    """Generate frequency-domain modes with the sizing expected by LISA ILE."""
    assert Lmax >= 2
    assert (not np.isnan(P.m1)) and (not np.isnan(P.m2)), "Masses are NaN."

    extra_waveform_args = {}
    if "extra_waveform_args" in kwargs:
        extra_waveform_args.update(kwargs["extra_waveform_args"])
    extra_params = P.to_lal_dict_extended(extra_args_dict=extra_waveform_args)

    fNyq = 0.5 / P.deltaT
    TDlen = int(1.0 / (P.deltaT * P.deltaF))
    if fd_alignment_postevent_time:
        if fd_alignment_postevent_time >= TDlen * P.deltaT / 2:
            print(
                " Warning: fd alignment postevent time requested incompatible with short duration ",
                file=sys.stderr,
            )

    if path_to_NR_hdf5 is not None:
        hlms_struct = lalsimutils.hlmoft_from_NRhdf5(
            path_to_NR_hdf5,
            P,
            Lmax,
            only_mode=modes,
            taper_percent=10,
            beta=8,
            verbose=True,
        )
        return {mode: lalsimutils.DataFourier(hlms_struct[mode]) for mode in hlms_struct}

    fd_mode_approximants = {
        lalsimutils.lalIMRPhenomHM,
        lalsimutils.lalIMRPhenomXPHM,
        lalsimutils.lalIMRPhenomXHM,
    }
    if P.approx in fd_mode_approximants:
        hlms_struct = lalsim.SimInspiralChooseFDModes(
            P.m1,
            P.m2,
            P.s1x,
            P.s1y,
            P.s1z,
            P.s2x,
            P.s2y,
            P.s2z,
            P.deltaF,
            P.fmin * fd_standoff_factor,
            fNyq,
            P.fref,
            P.phiref,
            P.dist,
            P.incl,
            extra_params,
            P.approx,
        )
        hlmsdict = lalsimutils.SphHarmFrequencySeries_to_dict(hlms_struct, Lmax)
        hlmsdict = _filter_modes(hlmsdict, modes)
        return {
            mode: lal.ResizeCOMPLEX16FrequencySeries(hlm, 0, TDlen)
            for mode, hlm in hlmsdict.items()
        }

    if P.approx in {lalsimutils.lalNRHybSur3dq8, lalsimutils.lalIMRPhenomD}:
        hlms_struct = lalsimutils.hlmoff(P, Lmax=Lmax)
        hlmsdict = lalsimutils.SphHarmFrequencySeries_to_dict(hlms_struct, Lmax)
        hlmsdict = _filter_modes(hlmsdict, modes)
        for mode, hlm in list(hlmsdict.items()):
            if not (1 / hlm.deltaF == P.deltaT * TDlen):
                print(
                    "WARNING: RESIZING IN FD DOMAIN "
                    f"(1/deltaF = {1 / hlm.deltaF}, deltaT*TDlen = {P.deltaT * TDlen}), "
                    "THIS SHOULD NOT BE HAPPENING."
                )
            hlmsdict[mode] = lal.ResizeCOMPLEX16FrequencySeries(hlm, 0, TDlen)
        return hlmsdict

    raise ValueError(f"Unsupported LISA FD mode approximant: {P.approx}")


def _cache_path_for_channel(fname, channel):
    cache_data = np.loadtxt(fname, dtype=str)
    if cache_data.ndim == 1:
        cache_data = cache_data.reshape(1, len(cache_data))

    channel_prefix = channel[0]
    for row in cache_data:
        if row[0] == channel_prefix:
            raw_path = row[-1]
            if "localhost" in raw_path:
                raw_path = raw_path.split("localhost", 1)[1]
            elif raw_path.startswith("file://"):
                raw_path = raw_path[len("file://"):]
            return os.path.expanduser(raw_path)

    raise ValueError(f"Could not find channel {channel!r} in cache {fname!r}")


def frame_h5_to_hoff(fname, channel, start=None, stop=None, verbose=True):
    """Read LISA frequency-domain HDF5 data from a cache entry."""
    if verbose:
        print(" ++ Loading from cache ", fname, channel)

    path_to_h5 = _cache_path_for_channel(fname, channel)
    if verbose:
        print(f"Reading h5 file {path_to_h5}")

    with h5py.File(path_to_h5, "r") as data:
        hoff = lal.CreateCOMPLEX16FrequencySeries(
            "hoff",
            data.attrs["epoch"],
            data.attrs["f0"],
            data.attrs["deltaF"],
            lalsimutils.lsu_HertzUnit,
            int(data.attrs["length"]),
        )
        hoff.data.data = data["data"][()]

    return hoff


def frame_h5_to_hoft(fname, channel, start=None, stop=None, verbose=True):
    """Read LISA time-domain HDF5 data from a cache entry."""
    if verbose:
        print(" ++ Loading from cache ", fname, channel)

    path_to_h5 = _cache_path_for_channel(fname, channel)
    if verbose:
        print(f"Reading h5 file {path_to_h5}")

    with h5py.File(path_to_h5, "r") as data:
        hoft = lal.CreateREAL8TimeSeries(
            "hoft",
            data.attrs["epoch"],
            data.attrs["f0"],
            data.attrs["deltaT"],
            lal.DimensionlessUnit,
            int(data.attrs["length"]),
        )
        hoft.data.data = data["data"][()]

    return hoft


def frame_data_to_non_herm_hoff(
    fname,
    channel,
    start=None,
    stop=None,
    TDlen=0,
    window_shape=0.0,
    verbose=True,
    deltaT=None,
    h5_frame=False,
):
    """Read frame/HDF5 time-domain data and FFT to two-sided frequency data."""
    if not h5_frame:
        return lalsimutils.frame_data_to_non_herm_hoff(
            fname,
            channel,
            start=start,
            stop=stop,
            TDlen=TDlen,
            window_shape=window_shape,
            verbose=verbose,
            deltaT=deltaT,
        )

    ht = frame_h5_to_hoft(fname, channel, start, stop, verbose)

    tmplen = ht.data.length
    if TDlen == -1:
        TDlen = tmplen
    elif TDlen == 0:
        TDlen = lalsimutils.nextPow2(tmplen)
    else:
        assert TDlen >= tmplen

    ht = lal.ResizeREAL8TimeSeries(ht, 0, TDlen)
    hoftC = lal.CreateCOMPLEX16TimeSeries(
        "h(t)", ht.epoch, ht.f0, ht.deltaT, ht.sampleUnits, TDlen
    )
    hoftC.data.data = ht.data.data + 0j
    fwdplan = lal.CreateForwardCOMPLEX16FFTPlan(TDlen, 0)
    hf = lal.CreateCOMPLEX16FrequencySeries(
        "Template h(f)",
        ht.epoch,
        ht.f0,
        1.0 / ht.deltaT / TDlen,
        lalsimutils.lsu_HertzUnit,
        TDlen,
    )
    lal.COMPLEX16TimeFreqFFT(hf, hoftC, fwdplan)
    if verbose:
        print(
            " ++ Loaded data h(f) of length n= ",
            hf.data.length,
            " (= ",
            len(hf.data.data) * ht.deltaT,
            "s) at sampling rate ",
            1.0 / ht.deltaT,
        )
    return hf


def print_params_lisa(self, show_system_frame=False):
    """LISA-flavored parameter printer used by the standalone ILE fork."""
    print("This ChooseWaveformParams has the following parameter values:")
    print(f"m1 = {self.m1 / lalsimutils.lsu_MSUN / 1e3:0.3f} x 1e3 (Msun)")
    print(f"m2 = {self.m2 / lalsimutils.lsu_MSUN / 1e3:0.3f} x 1e3 (Msun)")
    print("s1x =", self.s1x)
    print("s1y =", self.s1y)
    print("s1z =", self.s1z)
    print("s2x =", self.s2x)
    print("s2y =", self.s2y)
    print("s2z =", self.s2z)
    if show_system_frame:
        self.print_params(show_system_frame=show_system_frame)
    print("lambda1 =", self.lambda1)
    print("lambda2 =", self.lambda2)
    print("inclination =", self.incl)
    print("distance =", self.dist / 1.0e9 / lalsimutils.lsu_PC, "(Gpc)")
    print("reference orbital phase =", self.phiref)
    print("polarization angle =", self.psi)
    print("eccentricity = ", self.eccentricity)
    print("reference time = ", float(self.tref), "(s)")
    print("detector is: LISA")
    print("ecliptic latitude (beta):", self.theta, "(radians)")
    print("ecliptic longitude (lambda):", self.phi, "(radians)")
    print("starting frequency is =", self.fmin, "(Hz)")
    print("reference frequency is =", self.fref, "(Hz)")
    print("Max frequency is =", self.fmax, "(Hz)")
    print("time step =", self.deltaT, "(s) <==>", 1.0 / self.deltaT, "(Hz) sample rate")
    print("freq. bin size is =", self.deltaF, "(Hz)")


def install_choose_waveform_print_params_lisa():
    """Install the LISA printer on ChooseWaveformParams for the LISA fork."""
    lalsimutils.ChooseWaveformParams.print_params_lisa = print_params_lisa


__all__ = [
    "frame_data_to_non_herm_hoff",
    "frame_h5_to_hoff",
    "frame_h5_to_hoft",
    "hlmoff_for_LISA",
    "install_choose_waveform_print_params_lisa",
    "print_params_lisa",
]
