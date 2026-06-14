#!/usr/bin/env python

"""Generate tiny synthetic LISA inputs for the demo analysis surface."""

from argparse import ArgumentParser
import os

from igwn_ligolw import utils as ligolw_utils
import lal
import lal.series
import lalsimulation as lalsim
import numpy as np

import RIFT.LISA.lalsimutils_compat as lisa_lalsimutils_compat
import RIFT.lalsimutils as lalsimutils
from RIFT.LISA.response import LISA_response
from RIFT.LISA.psd_generation import generate_LISA_psd


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--output-directory", default=".")
    parser.add_argument("--mass1", type=float, default=1.0e5)
    parser.add_argument("--mass2", type=float, default=8.0e4)
    parser.add_argument("--spin1z", type=float, default=0.1)
    parser.add_argument("--spin2z", type=float, default=-0.05)
    parser.add_argument("--distance-mpc", type=float, default=1.0e3)
    parser.add_argument("--fmin", type=float, default=1.0e-3)
    parser.add_argument("--fref", type=float, default=5.0e-3)
    parser.add_argument("--fmax", type=float, default=0.125)
    parser.add_argument("--deltaT", type=float, default=4.0)
    parser.add_argument("--duration", type=float, default=4096.0)
    parser.add_argument("--ecliptic-latitude", type=float, default=0.3)
    parser.add_argument("--ecliptic-longitude", type=float, default=1.0)
    parser.add_argument("--psi", type=float, default=0.2)
    parser.add_argument("--inclination", type=float, default=0.4)
    parser.add_argument("--phiref", type=float, default=0.1)
    parser.add_argument("--psd-level", type=float, default=1.0e-40)
    return parser


def synthetic_params(opts):
    P = lalsimutils.ChooseWaveformParams()
    P.m1 = opts.mass1 * lal.MSUN_SI
    P.m2 = opts.mass2 * lal.MSUN_SI
    P.s1z = opts.spin1z
    P.s2z = opts.spin2z
    P.dist = opts.distance_mpc * 1.0e6 * lal.PC_SI
    P.fmin = opts.fmin
    P.fref = opts.fref
    P.fmax = opts.fmax
    P.deltaT = float(opts.deltaT)
    P.deltaF = 1.0 / float(opts.duration)
    P.approx = lalsim.IMRPhenomD
    P.theta = opts.ecliptic_latitude
    P.phi = opts.ecliptic_longitude
    P.psi = opts.psi
    P.incl = opts.inclination
    P.phiref = opts.phiref
    return P


def write_cache(output_directory):
    cache_path = os.path.join(output_directory, "lisa.cache")
    rows = [
        ("A", "A-fake_strain-1000000-10000.h5"),
        ("E", "E-fake_strain-1000000-10000.h5"),
        ("T", "T-fake_strain-1000000-10000.h5"),
    ]
    with open(cache_path, "w") as out:
        for channel, filename in rows:
            path = os.path.abspath(os.path.join(output_directory, filename))
            out.write(f"{channel} {channel} 0 1 file://localhost{path}\n")
    return cache_path


def _ldc_inference_psd(channel, fvals):
    """The PSD people actually use for INFERENCE: the LISA Data Challenge noise
    model (per A/E/T channel), via RIFT.LISA.utils.utils.get_ldc_psds.  Requires
    the optional `ldc` package.  Returns None (so the caller falls back to the
    analytic sensitivity curve) if ldc is unavailable -- the analytic curve is
    sky-averaged and strictly a mismatch tool, but it matches the SciRDv1 model
    to <~1% across the sensitive band, so it is an adequate CI proxy.  (Guidance
    from A. Jan; the production runs use the ldc PSD.)
    """
    try:
        import ldc.lisa.noise as _noise
    except Exception:
        return None
    nm = _noise.get_noise_model("SciRDv1", fvals[1:])
    out = np.empty(len(fvals))
    out[0] = np.inf
    out[1:] = nm.psd(fvals[1:], channel)
    return out


def write_lisa_psd_xml(channel, output_directory, deltaF, length, Tobs_years=0.5, NC=3):
    """Write a LISA PSD on the SAME frequency grid as the data (f0=0, this
    deltaF, this length).

    Prefers the ldc INFERENCE PSD (per-channel SciRDv1); falls back to the
    analytic sky-averaged sensitivity curve when `ldc` is not installed.

    A flat PSD is unphysical for LISA: at the SNRs of MBHBs the steeply-rising
    low-frequency noise must be modelled or the low-f content dominates and the
    posterior collapses to a delta (extrinsic eff_samp -> 1).  Keeping the PSD on
    the data's own grid also avoids ILE interpolation across a mismatched grid.
    """
    fvals = np.arange(length) * deltaF
    psd_values = _ldc_inference_psd(channel, fvals)
    if psd_values is None:
        R_exists, interp_func = generate_LISA_psd.response_interpolant(NC)
        psd_values = np.empty(length)
        # DC bin (f=0) is not physical for the sensitivity curve; set it huge so
        # it carries zero weight in the likelihood.
        psd_values[0] = np.inf
        psd_values[1:] = generate_LISA_psd.Sn(
            fvals[1:], Tobs_years * lal.YRSID_SI, NC, R_exists, interp_func)
    psd = lal.CreateREAL8FrequencySeries(
        channel,
        lal.LIGOTimeGPS(0),
        0.0,
        deltaF,
        lalsimutils.lsu_HertzUnit,
        length,
    )
    psd.data.data[:] = psd_values
    xmldoc = lal.series.make_psd_xmldoc({channel: psd})
    xmldoc.childNodes[0].attributes._attrs = {"Name": "psd"}
    path = os.path.join(output_directory, f"{channel}_psd.xml.gz")
    ligolw_utils.write_filename(xmldoc, path, compress="gz")
    return path


def main(argv=None):
    opts = build_parser().parse_args(argv)
    output_directory = os.path.abspath(opts.output_directory)
    os.makedirs(output_directory, exist_ok=True)

    P = synthetic_params(opts)
    modes = [(2, 2)]
    hlms = lisa_lalsimutils_compat.hlmoff_for_LISA(P, Lmax=2, modes=modes)
    data = LISA_response.create_lisa_injections(
        hlms,
        P.fmax,
        P.fref,
        P.theta,
        P.phi,
        P.psi,
        P.incl,
        P.phiref,
        tref=0.0,
    )
    LISA_response.create_h5_files_from_data_dict(data, output_directory)
    cache_path = write_cache(output_directory)

    psd_paths = {}
    for channel, channel_data in data.items():
        psd_paths[channel] = write_lisa_psd_xml(
            channel,
            output_directory,
            channel_data.deltaF,
            channel_data.data.length,
        )

    summary_path = os.path.join(output_directory, "synthetic-params.env")
    with open(summary_path, "w") as out:
        out.write(f"MASS1={opts.mass1}\n")
        out.write(f"MASS2={opts.mass2}\n")
        out.write(f"SPIN1Z={opts.spin1z}\n")
        out.write(f"SPIN2Z={opts.spin2z}\n")
        out.write(f"DISTANCE_MPC={opts.distance_mpc}\n")
        out.write(f"SRATE={1.0 / P.deltaT}\n")
        out.write(f"DELTA_T={P.deltaT}\n")
        out.write(f"DURATION={1.0 / P.deltaF}\n")
        out.write(f"DELTA_F={P.deltaF}\n")
        out.write(f"FMIN={P.fmin}\n")
        out.write(f"FREF={P.fref}\n")
        out.write(f"FMAX={P.fmax}\n")
        out.write(f"ECLIPTIC_LATITUDE={P.theta}\n")
        out.write(f"ECLIPTIC_LONGITUDE={P.phi}\n")
        out.write(f"PSI={P.psi}\n")
        out.write(f"INCLINATION={P.incl}\n")
        out.write(f"PHIREF={P.phiref}\n")
        out.write(f"CACHE_FILE={cache_path}\n")
        for channel in ["A", "E", "T"]:
            out.write(f"{channel}_PSD={psd_paths[channel]}\n")

    print(f"Wrote synthetic LISA inputs in {output_directory}")
    print(f"  cache: {cache_path}")
    print(f"  params: {summary_path}")


if __name__ == "__main__":
    main()
