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
allow_alternate=False
allow_duplicates = False
if len(sys.argv) > 3:   # if 3 arguments passed, target the effective sample size as output size
    allow_alternate=True
if len(sys.argv) > 4:
    allow_duplicates=True

outdir = os.path.dirname(os.path.abspath(sample_file))

# test if cal samples are present,and of the correct length of filesa
fnames_extended_post = glob(weights_file_directory+"/weights*extended_posterior")
fnames_weights = glob(weights_file_directory+"/weights*dat")
have_extended = len(fnames_extended_post)>0
if have_extended and len(fnames_extended_post) ==len(fname_weights):
    # important to import them IN ORDER -- see above
    fnames_extended_post.sort(key=sortKeyFunc)
    # import all the data, then concatenate
    dat_individual =[]
    for name in fnames_extended_post:
        dat_individual.append(np.genfromtxt(name, names=True))

    result = bilby.core.result.Result()
    result.posterior = pd.DataFrame(np.concatenate(dat_individual))
    result.meta_data = {}
else:
    # Load the sample file. Note we are reading from a file that has the same entries as the extrinsic samples
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
name_0 = result.posterior.keys()[0]
npts_samples = len(result.posterior[name_0])
print(" Sizes ", npts_samples, npts_wts)
if npts_samples > len(weights):
    print(" Truncating input samples to weight size (=requested output size) ")
    result.posterior = result.posterior.truncate(after=npts_wts-1)

print(f'Importance sampling efficiency: {np.sum(weights)**2/np.sum(weights**2)/len(weights)}')
if not(allow_alternate):
    result.posterior = bilby.core.result.rejection_sample(result.posterior, weights)
else:
    n_est = int(len(weights)*np.sum(weights)**2/np.sum(weights**2)/len(weights))+1  # effective sample size used to draw
    result.posterior = result.posterior.sample(n=n_est,weights=weights/np.sum(weights),replace=allow_duplicates)
result.save_posterior_samples(filename=outdir+'/reweighted_posterior_samples.dat', outdir=outdir)
