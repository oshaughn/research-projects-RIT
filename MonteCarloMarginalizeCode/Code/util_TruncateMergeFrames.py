#! /usr/bin/env python

from gwpy.timeseries import TimeSeries
#import glob
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start",type=float)
parser.add_argument("--end",type=float)
parser.add_argument("--channel")
parser.add_argument("--output")
parser.add_argument("fnames",nargs="+")
opts = parser.parse_args()

dat = TimeSeries.read(opts.fnames, opts.channel,start=opts.start,end=opts.end)

dat.write(opts.output)
