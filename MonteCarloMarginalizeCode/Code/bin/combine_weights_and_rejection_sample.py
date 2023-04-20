#! /usr/bin/env python
"""
Code to load and combine all the  weights and then rejection sample the results
"""

import os
import sys

import numpy as np
import pandas as pd
import pickle
from glob import glob

import bilby
import bilby_pipe

def sortKeyFunc(s):
    file = s.split('/')[-1]
    file_num = file.lstrip('weights_s')
    number = file_num.split('e')[0]
    return int(number)

# load in the arguments
sample_file = sys.argv[1]
weights_file_directory = sys.argv[2]

outdir = os.path.dirname(os.path.abspath(sample_file))

# Load the sample file
if sample_file.split(".")[-1] == 'json':
    result = bilby.core.result.read_in_result(sample_file)
elif (sample_file.split(".")[-1] == 'txt') or (sample_file.split(".")[-1] == 'dat'):
    result = bilby.core.result.Result()
    result.posterior = pd.DataFrame(np.genfromtxt(sample_file, names=True))
    result.posterior = result.posterior
    result.meta_data = {}

# create the weight list
weights_individual_list = []
weight_files = glob(f'{weights_file_directory}/*.dat')
weight_files.sort(key=sortKeyFunc)

for weight_file in weight_files:
    weights_individual_list.append(np.atleast_1d(np.genfromtxt(weight_file)))

weights = np.concatenate(weights_individual_list)
np.savetxt(f'{outdir}/weights.dat', weights)

# Truncate samples to weight size, if larger
npts_wts = len(weights)
npts_samples = len(result.posterior['m1'])
print(" Sizes ", npts_samples, npts_wts)
if npts_samples > len(weights):
    print(" Truncating input samples to weight size (=requested output size) ")
    result.posterior = result.posterior.truncate(after=npts_wts-1)

print(f'Importance sampling efficiency: {np.sum(weights)**2/np.sum(weights**2)/len(weights)}')
result.posterior = bilby.core.result.rejection_sample(result.posterior, weights)
result.save_posterior_samples(filename=outdir+'/reweighted_posterior_samples.dat', outdir=outdir)
