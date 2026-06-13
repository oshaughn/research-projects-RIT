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
    if opts.grid_size <= 1:
        rows = [base.copy()]
    else:
        # Build a 2-D lattice that varies m1 and m2 INDEPENDENTLY, so the grid
        # spans chirp mass AND symmetric mass ratio in two dimensions.  The old
        # 1-D line (m1*(1+off), m2*(1-off)) is collinear in (mc,eta) and is
        # degenerate for CIP's 2-D (mc,eta) quadratic/rf fit (-> NaN/no output).
        side = max(2, int(round(np.sqrt(opts.grid_size))))
        offs = np.linspace(-1.0, 1.0, side) * opts.grid_fractional_width
        sky_offs = np.linspace(-1.0, 1.0, side) * opts.sky_grid_width
        for i, oi in enumerate(offs):
            for j, oj in enumerate(offs):
                row = base.copy()
                row[2] = opts.mass1 * (1.0 + oi)
                row[3] = opts.mass2 * (1.0 + oj)
                if opts.vary_sky:
                    row[10] = opts.ecliptic_longitude + sky_offs[i]
                    row[11] = opts.ecliptic_latitude + sky_offs[j]
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
    parser.add_argument("--puff-args", default="args_puff.txt")
    parser.add_argument("--puff-factor", type=float, default=1.0,
                        help="util_ParameterPuffball --puff-factor: scale of the inter-iteration grid spread.")
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
    parser.add_argument("--vary-sky", action="store_true", help="Treat ecliptic sky location as an intrinsic grid parameter.")
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--grid-fractional-width", type=float, default=1.0e-3)
    parser.add_argument("--sky-grid-width", type=float, default=1.0e-3)

    parser.add_argument("--approximant", default="IMRPhenomD")
    parser.add_argument("--fmin-template", type=float, default=1.0e-4)  # in-band for LISA (1e-3 starts near top of band)
    parser.add_argument("--fmax", type=float, default=0.125)
    parser.add_argument("--reference-freq", type=float, default=5.0e-3)
    parser.add_argument("--srate", type=float, default=0.25)
    parser.add_argument("--l-max", type=int, default=2)
    parser.add_argument("--modes", default="[(2,2)]")
    parser.add_argument("--lisa-reference-time", type=float, default=0.0)
    parser.add_argument("--lisa-reference-frequency", type=float, default=5.0e-3)
    parser.add_argument("--data-integration-window-half", type=float, default=300.0)  # ~600s window: a 16s window mis-marginalizes the long LISA signal -> biased lnL
    parser.add_argument("--d-max", type=float, default=100000.0)  # LISA MBHBs reach cosmological distances
    parser.add_argument("--d-min", type=float, default=1000.0)
    parser.add_argument("--event-time", type=float, default=0.0)

    parser.add_argument("--zero-likelihood", action="store_true")
    parser.add_argument("--no-adapt", action="store_true",
                        help="Disable adaptive extrinsic sampling (uniform). Loud signals need adaptation, so default is OFF.")
    parser.add_argument("--ile-sampler-method", default="AV")
    parser.add_argument("--n-eff", type=int, default=20)
    parser.add_argument("--n-max", type=int, default=8000)
    parser.add_argument("--n-chunk", type=int, default=500)
    parser.add_argument("--save-P", type=float, default=0.1)

    parser.add_argument("--cip-fit-method", default="quadratic")
    parser.add_argument("--cip-sampler-method", default="AV")
    parser.add_argument("--cip-iterations", default="1")
    parser.add_argument("--cip-n-output-samples", type=int, default=100)
    parser.add_argument("--cip-lnL-offset", type=float, default=2000.0)  # keep all grid points for the fit (loud-signal lnL spread is large)
    parser.add_argument("--cip-n-eff", type=int, default=100)
    parser.add_argument("--cip-n-max", type=int, default=3000000)
    parser.add_argument("--cip-m-max-cut", default="1e8",
                        help="CIP --M-max-cut (Msun). LISA MBHBs need a large value.")
    parser.add_argument("--cip-sigma-cut", default="10.0",
                        help="CIP --sigma-cut. Relaxed for single-sample high-SNR demo integrals.")
    parser.add_argument("--cip-mass-range-frac", type=float, default=0.0,
                        help="Explicit half-width (fractional) of the CIP mc/mtot range; if 0, auto = 3x the grid fractional width (brackets the grid).")
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
    puff_args = os.path.join(workdir, opts.puff_args)
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
        "--sampler-method", opts.ile_sampler_method,
        "--internal-use-lnL",
    ]
    # Adaptive sampling is REQUIRED for loud LISA signals: with uniform sampling
    # (--no-adapt) the sharp extrinsic peak is single-sample-dominated (eff_samp
    # collapses to 1).  --force-adapt-all adapts every extrinsic dimension
    # (distance, angles) so the sampler concentrates on the peak.
    if opts.no_adapt:
        ile_parts.append("--no-adapt")
    else:
        ile_parts.append("--force-adapt-all")
    if not opts.vary_sky:
        ile_parts[3:3] = [
            "--lisa-fixed-sky", "1",
            "--ecliptic-longitude", opts.ecliptic_longitude,
            "--ecliptic-latitude", opts.ecliptic_latitude,
        ]
    if opts.zero_likelihood:
        ile_parts.append("--zero-likelihood")
    _write_arg_file(ile_args, ile_parts)

    # CIP fits only the parameters that actually VARY across the grid.  In
    # known-sky mode the ecliptic sky location is fixed (constant columns), so
    # fitting it is degenerate AND those coordinates are not understood by CIP's
    # waveform-parameter machinery (-> "No attribute ecliptic_longitude").  Only
    # add the sky as a fit parameter when it is varied (--vary-sky).
    cip_params = ["--parameter", "mc", "--parameter", "eta"]
    if opts.vary_sky:
        cip_params += ["--parameter", "ecliptic_longitude",
                       "--parameter", "ecliptic_latitude"]
    # CIP's posterior MC sampler defaults to a STELLAR-mass chirp-mass range
    # ([0.9, 250] Msun); for a LISA MBHB (mc ~ 1e4-1e7 Msun) the sampler would
    # never place a point near the signal -> eff_samp=nan.  Bracket mc and mtot
    # around the injected masses (analogue of the paper's force-mc-range).
    _mtot = opts.mass1 + opts.mass2
    _mc = (opts.mass1 * opts.mass2) ** 0.6 / _mtot ** 0.2
    # Bracket the GRID (plus margin): a range much wider than the grid samples
    # mostly where the fit extrapolates -> eff_samp=nan; one matched to the grid
    # keeps the CIP sampler where lnL is actually constrained.  The injected
    # mc/eta are measured to ~Fisher precision (<< grid), so grid-tied is also
    # tight enough to bracket the posterior.
    _w = max(opts.cip_mass_range_frac, 1.5 * opts.grid_fractional_width)
    _eta = (opts.mass1 * opts.mass2) / _mtot ** 2
    # Bracket eta too: with only mc/mtot bounded, the CIP posterior drifts to the
    # eta FLOOR (0.01) -> extreme q -> garbage m1 (mtot=mc/eta^0.6 blows up).  The
    # truth eta (~0.25 for near-equal MBHBs) is near the ceiling, so a tight
    # grid-tied eta window is essential (analogue of the paper's force-eta-range).
    cip_range_args = [
        "--mc-range", "[{},{}]".format(_mc * (1.0 - _w), _mc * (1.0 + _w)),
        "--mtot-range", "[{},{}]".format(_mtot * (1.0 - _w), _mtot * (1.0 + _w)),
        "--eta-range", "[{},{}]".format(_eta * (1.0 - _w), min(0.2499999, _eta * (1.0 + _w))),
        "--n-eff", str(opts.cip_n_eff), "--n-max", str(opts.cip_n_max),
    ]
    cip_line = _quote_join([
        opts.cip_iterations,
        "--fit-method", opts.cip_fit_method,
        # AV integrator works in lnL space automatically (no exp() overflow at
        # loud-signal lnL) and avoids the default sampler's lsoda CDF inversion
        # (which NaNs on the sharp high-SNR posterior).  --internal-use-lnL too.
        "--sampler-method", opts.cip_sampler_method,
        "--internal-use-lnL",
        *cip_params,
        *cip_range_args,
        "--n-output-samples", opts.cip_n_output_samples,
        "--lnL-offset", opts.cip_lnL_offset,
        # LISA sources are massive black-hole binaries (M ~ 1e4-1e8 Msun), far
        # above CIP's stellar-mass default (--M-max-cut 1e5) which would strip
        # every grid point as "too massive".  Likewise the synthetic high-SNR
        # demo gives single-sample (n_eff=1) integrals, so relax CIP's own
        # error cut (default 0.6) to keep those points.
        "--M-max-cut", opts.cip_m_max_cut,
        "--sigma-cut", opts.cip_sigma_cut,
        "--no-plots",
    ])
    _write_cip_list(cip_args_list, [cip_line])

    # Puffball: between iterations, perturb the (very tight) CIP posterior so the
    # next iteration's grid is not a near-degenerate cluster (which makes the CIP
    # refit ill-conditioned and diverge).  The CEPP wraps this with --inj-file /
    # --inj-file-out; we supply the perturbation parameters + physical bounds.
    _write_arg_file(puff_args, [
        "--parameter", "mc", "--parameter", "eta",
        "--puff-factor", opts.puff_factor,
        "--mc-range", "[{},{}]".format(_mc * (1.0 - _w), _mc * (1.0 + _w)),
        "--mtot-range", "[{},{}]".format(_mtot * (1.0 - _w), _mtot * (1.0 + _w)),
        "--eta-range", "[{},{}]".format(_eta * (1.0 - _w), min(0.2499999, _eta * (1.0 + _w))),
    ])

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
