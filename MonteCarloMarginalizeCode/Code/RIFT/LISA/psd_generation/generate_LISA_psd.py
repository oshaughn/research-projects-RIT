#!/usr/bin/env python

"""Generate an analytic LISA PSD text file and optional diagnostic products."""

from argparse import ArgumentParser
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

###########################################################################################
# CONSTANTS
###########################################################################################
fm = 3.168753575e-8
YRSID_SI = 31558149.763545603
C_SI = 299792458.0
e = 0.004824185218078991
a = 149597870700.0
Larm = 2 * np.sqrt(3) * a * e
fstar = C_SI / (2 * np.pi * Larm)

path_to_file = os.path.dirname(__file__)


###########################################################################################
# FUNCTIONS
# These functions were taken from LISA.py of LISA sensitivity
# (https://github.com/eXtremeGravityInstitute/LISA_Sensitivity)
###########################################################################################
def Pn(f):
    """Calculate the strain power spectral density."""
    P_oms = (1.5e-11) ** 2 * (1.0 + (2.0e-3 / f) ** 4)
    P_acc = (3.0e-15) ** 2 * (1.0 + (0.4e-3 / f) ** 2) * (1.0 + (f / (8.0e-3)) ** 4)
    return (P_oms + 2.0 * (1.0 + np.cos(f / fstar) ** 2) * P_acc / (2.0 * np.pi * f) ** 4) / Larm**2


def SnC(f, Tobs=0.5, NC=3):
    """
    Estimate galactic binary confusion noise.

    Tobs is provided in seconds. Supported fitted regimes correspond to roughly
    0.5 yr, 1 yr, 2 yr, and 4 yr observations.
    """
    if Tobs < 0.75 * YRSID_SI:
        est = 1
    elif 0.75 * YRSID_SI < Tobs and Tobs < 1.5 * YRSID_SI:
        est = 2
    elif 1.5 * YRSID_SI < Tobs and Tobs < 3.0 * YRSID_SI:
        est = 3
    else:
        est = 4

    if est == 1:
        alpha = 0.133
        beta = 243.0
        kappa = 482.0
        gamma = 917.0
        f_knee = 2.58e-3
    elif est == 2:
        alpha = 0.171
        beta = 292.0
        kappa = 1020.0
        gamma = 1680.0
        f_knee = 2.15e-3
    elif est == 3:
        alpha = 0.165
        beta = 299.0
        kappa = 611.0
        gamma = 1340.0
        f_knee = 1.73e-3
    else:
        alpha = 0.138
        beta = -221.0
        kappa = 521.0
        gamma = 1680.0
        f_knee = 1.13e-3

    A = 1.8e-44 / NC
    Sc = 1.0 + np.tanh(gamma * (f_knee - f))
    Sc *= np.exp(-(f**alpha) + beta * f * np.sin(kappa * f))
    Sc *= A * f ** (-7.0 / 3.0)
    return Sc


def Sn(f, Tobs=0.5, NC=3, R_exists=False, interp_func=None):
    """Calculate the sensitivity curve."""
    if R_exists:
        R = interpolate.splev(f, interp_func, der=0)
    else:
        R = 3.0 / 20.0 / (1.0 + 6.0 / 10.0 * (f / fstar) ** 2) * NC

    return Pn(f) / R + SnC(f, Tobs, NC)


def response_interpolant(NC):
    if os.path.exists(f"{path_to_file}/R.txt"):
        data = np.loadtxt(f"{path_to_file}/R.txt")
        R = data[:, 1] * NC
        f = data[:, 0] * fstar
        return True, interpolate.splrep(f, R, s=0)
    print("R.txt doesn't exist.")
    return False, None


def generate_psd(fmin=5.0e-5, fmax=1.0, Tobs_years=0.5, NC=3, npts=500001):
    """Return frequency and PSD arrays for the analytic LISA sensitivity curve."""
    R_exists, interp_func = response_interpolant(NC)
    f = np.linspace(fmin, fmax, npts)
    sens = Sn(f, Tobs_years * YRSID_SI, NC, R_exists, interp_func)
    return f, sens


def write_lisa_psd(output_dir, fmin=5.0e-5, fmax=1.0, Tobs_years=0.5, NC=3, npts=500001, write_xml=True):
    """Write LISA_psd.txt, LISA_psd_plot.png, and optionally A-psd.xml.gz."""
    f, sens = generate_psd(fmin=fmin, fmax=fmax, Tobs_years=Tobs_years, NC=NC, npts=npts)
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, "LISA_psd.txt")
    png_path = os.path.join(output_dir, "LISA_psd_plot.png")

    np.savetxt(txt_path, np.vstack([f, sens]).T)

    plt.figure()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Characteristic strain")
    plt.loglog(f, np.sqrt(f * sens))
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    if write_xml:
        subprocess.run(
            [
                "convert_psd_ascii2xml",
                "--fname-psd-ascii",
                txt_path,
                "--conventional-postfix",
                "--ifo",
                "A",
            ],
            check=True,
            cwd=output_dir,
        )
    return txt_path, png_path


def parse_args(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--NC", default=3, type=int, help="Number of channels.")
    parser.add_argument("--Tobs", default=0.5, type=float, help="Observation time in years.")
    parser.add_argument("--fmin", default=5.0e-5, type=float, help="Lowest PSD frequency.")
    parser.add_argument("--fmax", default=1, type=float, help="Highest PSD frequency.")
    parser.add_argument("--npts", default=500001, type=int, help="Number of PSD samples.")
    parser.add_argument("--output-dir", default=os.getcwd(), help="Directory for generated PSD products.")
    parser.add_argument("--skip-xml", action="store_true", help="Do not run convert_psd_ascii2xml.")
    return parser.parse_args(argv)


def main(argv=None):
    opts = parse_args(argv)
    print(f"Argument parser has the following arguments:\n{vars(opts)}")
    write_lisa_psd(
        opts.output_dir,
        fmin=opts.fmin,
        fmax=opts.fmax,
        Tobs_years=opts.Tobs,
        NC=opts.NC,
        npts=opts.npts,
        write_xml=not opts.skip_xml,
    )


if __name__ == "__main__":
    main()
