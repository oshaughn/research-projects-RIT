#!/usr/bin/env python
"""Emit a minimal CEPP argument bundle for LISA-RIFT workflows.

This helper is intentionally separate from ``helper_LDG_Events.py``.  It does
not select observing-run data, skymaps, or astrophysical fit strategies; it
only writes the files consumed by ``create_event_parameter_pipeline_*`` for a
LISA-specific ILE run.
"""

from __future__ import print_function

import argparse
import os
import shlex

import numpy as np

from RIFT.misc import hyperpipeline_io


def _split_assignment(text, option_name):
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "{} entries must have the form NAME=value".format(option_name)
        )
    name, value = text.split("=", 1)
    if not name or not value:
        raise argparse.ArgumentTypeError(
            "{} entries must have the form NAME=value".format(option_name)
        )
    return name, value


def _quote_join(parts):
    return " ".join(shlex.quote(str(part)) for part in parts if str(part) != "")


def _write_arg_file(path, parts):
    with open(path, "w") as out:
        out.write("X " + _quote_join(parts) + "\n")


def _write_cip_list(path, lines):
    with open(path, "w") as out:
        out.write("\n".join(lines) + "\n")


def _write_transfer_file(path, names):
    with open(path, "w") as out:
        for name in names:
            out.write(str(name) + "\n")


def _write_initial_grid(path, opts):
    columns = hyperpipeline_io.build_column_list(use_sky=True)
    base = np.array([
        0.0,
        1.0,
        opts.mass1,
        opts.mass2,
        opts.spin1x,
        opts.spin1y,
        opts.spin1z,
        opts.spin2x,
        opts.spin2y,
        opts.spin2z,
        opts.ecliptic_longitude,
        opts.ecliptic_latitude,
    ])

    rows = []
    for idx in range(opts.grid_size):
        row = base.copy()
        if opts.grid_size > 1:
            offset = (idx - (opts.grid_size - 1) / 2.0) * opts.grid_fractional_width
            row[2] = opts.mass1 * (1.0 + offset)
            row[3] = opts.mass2 * (1.0 - offset)
        rows.append(row)
    hyperpipeline_io.write_table(path, columns, np.array(rows))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Write CEPP contract files for a standalone LISA-RIFT run."
    )
    parser.add_argument("--working-directory", default=".")
    parser.add_argument("--input-grid", default="proposed-grid.dat")
    parser.add_argument("--ile-args", default="args_ile.txt")
    parser.add_argument("--cip-args-list", default="args_cip_list.txt")
    parser.add_argument("--test-args", default="args_test.txt")
    parser.add_argument("--transfer-file-list", default="helper_transfer_files.txt")
    parser.add_argument("--cepp-command-file", default="command-cepp-lisa.sh")

    parser.add_argument("--cache-file", default="lisa.cache")
    parser.add_argument(
        "--channel-name",
        action="append",
        default=None,
        help="LISA channel assignment, e.g. A=fake_strain.",
    )
    parser.add_argument(
        "--psd-file",
        action="append",
        default=None,
        help="PSD assignment, e.g. A=A_psd.xml.gz.",
    )

    parser.add_argument("--mass1", type=float, default=1.0e5)
    parser.add_argument("--mass2", type=float, default=8.0e4)
    parser.add_argument("--spin1x", type=float, default=0.0)
    parser.add_argument("--spin1y", type=float, default=0.0)
    parser.add_argument("--spin1z", type=float, default=0.0)
    parser.add_argument("--spin2x", type=float, default=0.0)
    parser.add_argument("--spin2y", type=float, default=0.0)
    parser.add_argument("--spin2z", type=float, default=0.0)
    parser.add_argument("--ecliptic-longitude", type=float, default=1.0)
    parser.add_argument("--ecliptic-latitude", type=float, default=0.3)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--grid-fractional-width", type=float, default=1.0e-3)

    parser.add_argument("--approximant", default="IMRPhenomD")
    parser.add_argument("--fmin-template", type=float, default=1.0e-3)
    parser.add_argument("--fmax", type=float, default=0.125)
    parser.add_argument("--reference-freq", type=float, default=5.0e-3)
    parser.add_argument("--srate", type=float, default=0.25)
    parser.add_argument("--l-max", type=int, default=2)
    parser.add_argument("--modes", default="[(2,2)]")
    parser.add_argument("--lisa-reference-time", type=float, default=0.0)
    parser.add_argument("--lisa-reference-frequency", type=float, default=5.0e-3)
    parser.add_argument("--data-integration-window-half", type=float, default=8.0)
    parser.add_argument("--d-max", type=float, default=5000.0)
    parser.add_argument("--d-min", type=float, default=1.0)
    parser.add_argument("--event-time", type=float, default=0.0)

    parser.add_argument("--zero-likelihood", action="store_true")
    parser.add_argument("--n-eff", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=20)
    parser.add_argument("--n-chunk", type=int, default=10)
    parser.add_argument("--save-P", type=float, default=0.1)

    parser.add_argument("--cip-fit-method", default="quadratic")
    parser.add_argument("--cip-iterations", default="1")
    parser.add_argument("--cip-n-output-samples", type=int, default=100)
    parser.add_argument("--cip-lnL-offset", type=float, default=100.0)
    parser.add_argument("--test-threshold", type=float, default=0.02)
    parser.add_argument("--cepp-exe", default="create_event_parameter_pipeline_BasicIteration")
    parser.add_argument("--ile-exe", default="integrate_likelihood_extrinsic_batchmode_lisa")
    parser.add_argument("--n-samples-per-job", type=int, default=1)
    parser.add_argument("--n-iterations", type=int, default=1)
    parser.add_argument("--request-memory-ILE", type=int, default=4096)
    parser.add_argument("--request-memory-CIP", type=int, default=4096)
    return parser


def main(argv=None):
    opts = build_parser().parse_args(argv)
    if opts.grid_size < 1:
        raise ValueError("--grid-size must be positive")

    workdir = os.path.abspath(opts.working_directory)
    os.makedirs(workdir, exist_ok=True)
    if opts.channel_name is None:
        opts.channel_name = ["A=fake_strain", "E=fake_strain", "T=fake_strain"]
    if opts.psd_file is None:
        opts.psd_file = ["A=A_psd.xml.gz", "E=E_psd.xml.gz", "T=T_psd.xml.gz"]

    input_grid = os.path.join(workdir, opts.input_grid)
    ile_args = os.path.join(workdir, opts.ile_args)
    cip_args_list = os.path.join(workdir, opts.cip_args_list)
    test_args = os.path.join(workdir, opts.test_args)
    transfer_file_list = os.path.join(workdir, opts.transfer_file_list)
    cepp_command_file = os.path.join(workdir, opts.cepp_command_file)

    _write_initial_grid(input_grid, opts)

    channel_args = []
    transfer_files = [opts.cache_file]
    for assignment in opts.channel_name:
        name, value = _split_assignment(assignment, "--channel-name")
        channel_args.extend(["--channel-name", "{}={}".format(name, value)])
    psd_args = []
    for assignment in opts.psd_file:
        name, value = _split_assignment(assignment, "--psd-file")
        psd_args.extend(["--psd-file", "{}={}".format(name, value)])
        transfer_files.append(value)

    ile_parts = [
        "--LISA",
        "--h5-frame-FD",
        "--time-marginalization",
        "--lisa-fixed-sky", "1",
        "--ecliptic-longitude", opts.ecliptic_longitude,
        "--ecliptic-latitude", opts.ecliptic_latitude,
        "--lisa-reference-time", opts.lisa_reference_time,
        "--lisa-reference-frequency", opts.lisa_reference_frequency,
        "--data-integration-window-half", opts.data_integration_window_half,
        "--modes", opts.modes,
        "--cache-file", opts.cache_file,
        "--event-time", opts.event_time,
    ] + channel_args + psd_args + [
        "--fmin-template", opts.fmin_template,
        "--fmin-ifo", "A={}".format(opts.fmin_template),
        "--fmin-ifo", "E={}".format(opts.fmin_template),
        "--fmin-ifo", "T={}".format(opts.fmin_template),
        "--fmax", opts.fmax,
        "--reference-freq", opts.reference_freq,
        "--srate", opts.srate,
        "--l-max", opts.l_max,
        "--approx", opts.approximant,
        "--d-max", opts.d_max,
        "--d-min", opts.d_min,
        "--n-eff", opts.n_eff,
        "--n-max", opts.n_max,
        "--n-chunk", opts.n_chunk,
        "--save-P", opts.save_P,
        "--no-adapt",
        "--internal-use-lnL",
    ]
    if opts.zero_likelihood:
        ile_parts.append("--zero-likelihood")
    _write_arg_file(ile_args, ile_parts)

    cip_line = _quote_join([
        opts.cip_iterations,
        "--fit-method", opts.cip_fit_method,
        "--parameter", "mc",
        "--parameter", "eta",
        "--parameter", "ecliptic_longitude",
        "--parameter", "ecliptic_latitude",
        "--n-output-samples", opts.cip_n_output_samples,
        "--lnL-offset", opts.cip_lnL_offset,
        "--no-plots",
    ])
    _write_cip_list(cip_args_list, [cip_line])

    _write_arg_file(test_args, [
        "--method", "lame",
        "--parameter", "mc",
        "--parameter", "eta",
        "--iteration", "$(macroiteration)",
        "--threshold", opts.test_threshold,
        "--always-succeed",
    ])

    _write_transfer_file(transfer_file_list, transfer_files)

    cepp_parts = [
        "RIFT_HYPERPIPELINE_FORMAT=1",
        opts.cepp_exe,
        "--ile-n-events-to-analyze", 1,
        "--input-grid", input_grid,
        "--ile-exe", opts.ile_exe,
        "--ile-args", ile_args,
        "--cip-args-list", cip_args_list,
        "--test-args", test_args,
        "--working-directory", workdir,
        "--n-iterations", opts.n_iterations,
        "--n-samples-per-job", opts.n_samples_per_job,
        "--n-copies", 1,
        "--request-memory-ILE", opts.request_memory_ILE,
        "--request-memory-CIP", opts.request_memory_CIP,
        "--transfer-file-list", transfer_file_list,
    ]
    with open(cepp_command_file, "w") as out:
        out.write(_quote_join(cepp_parts) + "\n")

    print("Wrote LISA CEPP helper bundle in {}".format(workdir))
    print("  input grid: {}".format(input_grid))
    print("  ILE args: {}".format(ile_args))
    print("  CIP args list: {}".format(cip_args_list))
    print("  test args: {}".format(test_args))
    print("  CEPP command: {}".format(cepp_command_file))


if __name__ == "__main__":
    main()
