#!/usr/bin/env python

"""Create LISA A/E/T injection frames from a RIFT injection XML file."""

from argparse import ArgumentParser
import ast
import os

import lal
import lalsimulation
import numpy as np

import RIFT.lalsimutils as lalsimutils
from RIFT.LISA.injections.LISA_injections import (
    generate_lisa_TDI_dict,
    generate_lisa_injections,
)

__author__ = "A. Jan"


def parse_args(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--save-path", default=os.getcwd(), help="Path where h5 files should be written.")
    parser.add_argument("--psd-path", default=None, help="Directory containing A/E/T PSD XML files for SNR.")
    parser.add_argument("--inj", required=True, help="Inspiral XML file containing injection information.")
    parser.add_argument("--fNyq", default=0.125, type=float, help="Nyquist frequency for generated waveforms.")
    parser.add_argument("--deltaF", default=1 / (64 * 32768), type=float, help="Injection deltaF.")
    parser.add_argument(
        "--modes",
        default="[(2,2),(2,1),(3,3),(3,2),(3,1),(4,4),(4,3),(4,2),(5,5)]",
        help="List of modes to use in injection.",
    )
    parser.add_argument("--path-to-NR-hdf5", default=None, help="NRHDF5 path when using NR injection data.")
    parser.add_argument("--snr-fmin", default=0.0001, type=float, help="fmin while calculating SNR.")
    parser.add_argument("--skip-snr", action="store_true", help="Write frames without PSD/SNR products.")
    return parser.parse_args(argv)


def parameter_dict_from_xml(opts):
    P_inj = lalsimutils.xml_to_ChooseWaveformParams_array(str(opts.inj))[0]
    modes = np.array(ast.literal_eval(opts.modes))
    param_dict = {
        "m1": P_inj.m1 / lal.MSUN_SI,
        "m2": P_inj.m2 / lal.MSUN_SI,
        "s1z": P_inj.s1z,
        "s2z": P_inj.s2z,
        "dist": P_inj.dist / (1e6 * lal.PC_SI),
        "fmin": P_inj.fmin,
        "fmax": opts.fNyq,
        "deltaF": opts.deltaF,
        "deltaT": 0.5 / opts.fNyq,
        "fref": None,
        "wf-fref": P_inj.fref,
        "tref": float(P_inj.tref),
        "beta": P_inj.theta,
        "lambda": P_inj.phi,
        "psi": P_inj.psi,
        "phi_ref": P_inj.phiref,
        "inclination": P_inj.incl,
        "approx": lalsimulation.GetStringFromApproximant(P_inj.approx),
        "modes": modes,
        "save_path": opts.save_path or os.getcwd(),
        "path_to_NR_hdf5": opts.path_to_NR_hdf5,
        "snr_fmin": opts.snr_fmin,
        "snr_fmax": opts.fNyq,
    }
    if opts.psd_path:
        param_dict["psd_path"] = opts.psd_path
    return param_dict


def main(argv=None):
    opts = parse_args(argv)
    param_dict = parameter_dict_from_xml(opts)
    print(f"Saving frames in {param_dict['save_path']}")
    print("###############")
    if 1 / param_dict["deltaF"] / 60 / 60 / 24 > 0.5:
        print(f"Data length = {1 / param_dict['deltaF'] / 60 / 60 / 24} days.")
    else:
        print(f"Data length = {1 / param_dict['deltaF'] / 60 / 60} hrs.")
    print(
        f"\nWaveform is being generated with m1 = {param_dict['m1']}, "
        f"m2 = {param_dict['m2']}, s1z = {param_dict['s1z']}, s2z = {param_dict['s2z']}"
    )
    print(
        f"deltaF = {param_dict['deltaF']}, fmin = {param_dict['fmin']}, "
        f"fmax = {param_dict['fmax']}, deltaT = {param_dict['deltaT']}, "
        f"modes = {list(param_dict['modes'])}, tref = {param_dict['tref']}"
    )
    print(
        f"phiref = {param_dict['phi_ref']}, psi = {param_dict['psi']}, "
        f"inclination = {param_dict['inclination']}, beta = {param_dict['beta']}, "
        f"lambda = {param_dict['lambda']}"
    )
    print(f"path_to_NR_hdf5 = {param_dict['path_to_NR_hdf5']}, approx = {param_dict['approx']}\n")
    print("###############")

    data_dict = generate_lisa_TDI_dict(param_dict)
    generate_lisa_injections(data_dict, param_dict, get_snr=not opts.skip_snr and opts.psd_path is not None)


if __name__ == "__main__":
    main()
