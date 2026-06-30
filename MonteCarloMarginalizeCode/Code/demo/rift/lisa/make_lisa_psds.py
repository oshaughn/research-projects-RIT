#!/usr/bin/env python

"""Generate lightweight analytic LISA PSD products for demo/test workflows."""

from argparse import ArgumentParser
import os

from igwn_ligolw import utils as ligolw_utils
import lal
import lal.series
import numpy as np

from RIFT.LISA.psd_generation import generate_LISA_psd
import RIFT.lalsimutils as lalsimutils


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--output-directory", default=".")
    parser.add_argument("--channels", default="A,E,T")
    parser.add_argument("--fmax", type=float, default=0.125)
    parser.add_argument("--npts", type=int, default=513)
    parser.add_argument("--Tobs", type=float, default=0.5, help="Observation time in years.")
    parser.add_argument("--NC", type=int, default=3, help="Number of LISA channels.")
    parser.add_argument("--write-ascii", action="store_true")
    return parser


def write_psd_xml(channel, path, fvals, psd_values):
    deltaF = fvals[1] - fvals[0]
    psd = lal.CreateREAL8FrequencySeries(
        channel,
        lal.LIGOTimeGPS(0),
        fvals[0],
        deltaF,
        lalsimutils.lsu_HertzUnit,
        len(fvals),
    )
    psd.data.data[:] = psd_values
    xmldoc = lal.series.make_psd_xmldoc({channel: psd})
    xmldoc.childNodes[0].attributes._attrs = {"Name": "psd"}
    ligolw_utils.write_filename(xmldoc, path, compress="gz")


def main(argv=None):
    opts = build_parser().parse_args(argv)
    output_directory = os.path.abspath(opts.output_directory)
    os.makedirs(output_directory, exist_ok=True)

    if opts.npts < 2:
        raise ValueError("--npts must be at least 2")
    fvals = np.linspace(0.0, opts.fmax, opts.npts)
    positive_fvals, positive_psd = generate_LISA_psd.generate_psd(
        fmin=fvals[1],
        fmax=opts.fmax,
        Tobs_years=opts.Tobs,
        NC=opts.NC,
        npts=opts.npts - 1,
    )
    psd_values = np.zeros_like(fvals)
    psd_values[1:] = positive_psd

    for channel in [item.strip() for item in opts.channels.split(",") if item.strip()]:
        xml_path = os.path.join(output_directory, "{}_psd.xml.gz".format(channel))
        write_psd_xml(channel, xml_path, fvals, psd_values)
        print("Wrote {}".format(xml_path))

    if opts.write_ascii:
        ascii_path = os.path.join(output_directory, "LISA_psd.txt")
        np.savetxt(ascii_path, np.column_stack([positive_fvals, positive_psd]))
        print("Wrote {}".format(ascii_path))


if __name__ == "__main__":
    main()
