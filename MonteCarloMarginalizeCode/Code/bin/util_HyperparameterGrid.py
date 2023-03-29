#! /usr/bin/env python

import argparse
import sys
import numpy as np
import numpy.lib.recfunctions
import scipy
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lal
import functools
import itertools


parser = argparse.ArgumentParser()
parser.add_argument("--fname-out", default="output-grid.dat", help="Name of dat file")
parser.add_argument("--npts",default=100,type=int)
parser.add_argument("--random-parameter", action='append',help="These parameters are specified at random over the entire range, uncorrelated with the grid used for other parameters.  Use for variables which correlate weakly with others; helps with random exploration")
parser.add_argument("--random-parameter-range", action='append', type=str,help="Add a range (pass as a string evaluating to a python 2-element list): --parameter-range '[0.,1000.]'   MUST specify ALL parameter ranges (min and max) in order if used.  ")
opts=  parser.parse_args()

npts = opts.npts

# Extract parameter names
coord_names = opts.random_parameter # Used  in fit

param_ranges = list(map(eval, opts.random_parameter_range))  # do not randomize mass or distance parameters, not being rescaled

#if opts.parameter_nofit:
#    coord_names = coord_names + opts.parameter_nofit
if coord_names is None:
    sys.exit(0)


# Load data, keep parameter names
X= np.zeros((npts, 2+len(coord_names)))
for indx in np.arange(len(coord_names)):
        X[:,2+indx] = np.random.uniform(param_ranges[indx][0], param_ranges[indx][1],size=npts)

np.savetxt(opts.fname_out, X,header="lnL sigma_lnL " + " ".join(coord_names))
