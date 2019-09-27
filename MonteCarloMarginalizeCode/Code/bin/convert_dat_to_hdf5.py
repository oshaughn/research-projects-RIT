#! /usr/bin/env python
###############################################
#
# convert_dat_to_hdf5.py
# Convert LALInference results.dat file to
# a results.hdf5 file
#
# Provided by Simon Stevenson
#
# Used to convert for format compatible with PESummary: https://git.ligo.org/lscsoft/pesummary
###############################################

import numpy as np
import h5py as h5
import argparse

#-- Get command line arguments
parser = argparse.ArgumentParser(description='''Convert LALInference results.dat file to a results.hdf5 file''',
								epilog='''
								''')

parser.add_argument("--input-file", type=str, default='results.dat', help="Input file e.g. results.dat")
parser.add_argument("--output-file", type=str, default='results.hdf5', help="Output file e.g. results.hdf5")
parser.add_argument("--sampler", type=str, default='mcmc', help="Sampler used mcmc/nest")
args = parser.parse_args()

dat_file = np.genfromtxt(args.input_file, names=True)

h5_file  = h5.File(args.output_file, 'w')

lalinference_group = h5_file.create_group('lalinference')

sampler_group = lalinference_group.create_group('lalinference_' + args.sampler)

sampler_group.create_dataset('posterior_samples', data=dat_file)

h5_file.close()
