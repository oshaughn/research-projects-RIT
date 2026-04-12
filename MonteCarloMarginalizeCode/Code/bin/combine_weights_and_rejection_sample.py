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
def name_to_key(s):
    file= os.path.basename(s)
    #print(s, file)
    return file.replace('.','_').split('_')[1]
def key_to_bounds(key):
    start_str,end_str = key.replace('s','').split('e')
    return [int(start_str),int(end_str)]
def index_of(sub_str, my_list):
    # find instance in list where sub_str present. There is only one
    # Horrible for loop implementation
    for indx in np.arange(len(my_list)):
        if sub_str in my_list[indx]:
            return int(indx)
saved_dtype=None
def check_valid_import(fname,size=None):
    try:
        #dat = np.genfromtxt(fname,names=True)
        my_in = pd.read_csv(fname, sep=' ')
        # size check
        if not(size is None):
            if len(my_in) == size:
                return True
            else:
                return False
    except:
        return False
    return True

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
fnames_extended_post = list(glob(weights_file_directory+"/weights*extended_posterior"))
fnames_weights = list(glob(weights_file_directory+"/weights*dat"))
have_extended = len(fnames_extended_post)>0
indx_result_to_downselect=None
k_low, k_high  = key_to_bounds(name_to_key(fnames_weights[0]))
n_stride = k_high - k_low # HARDCODE COMMON SIZE

if have_extended:
    # perform safety check, downselect
    keys_weights = [name_to_key(s) for s in fnames_weights] # key list
    keys_post = [name_to_key(s) for s in fnames_extended_post if check_valid_import(s,size=n_stride)] # key list
    v = set(keys_weights).intersection(set(keys_post)); print(v)
    keys_common = list(v) # remove duplicates
    print(keys_weights, keys_common, keys_post)
    print(len(keys_common), len(keys_weights), len(keys_post))
    indx_post    = [index_of(s, fnames_extended_post) for s in keys_common]
    indx_weights = [index_of(s, fnames_weights) for s in keys_common]
    #print(indx_post, indx_weights)
    # downselect the two lists
    fnames_weights       = np.array(fnames_weights)[indx_weights]
    fnames_extended_post = np.array(fnames_extended_post)[indx_post]
else:
    # Now, find integer ranges we need to preserve from the files, so it maps to our order
    # in case we have job failures or other corrpution 
    indx_net = []
    for fname in fnames_weights:
        start,end = key_to_bounds(name_to_key(fname))
        indx_net += range(start,end)
    indx_result_to_downselect = np.array(indx_net)


if have_extended and len(fnames_extended_post) ==len(fnames_weights):
    # important to import them IN ORDER -- see above
    #fnames_extended_post.sort(key=sortKeyFunc)
    # import all the data, then concatenate
    # BEWARE NUMPY MERGES OF RECORD ARRAYS: This requires care to do correctly!
    dat_individual =[]
    dtype_here=None
    dat_net = None
    for name in fnames_extended_post:
        dat_here = pd.read_csv(name,sep=' ').to_records() #np.genfromtxt(name, names=True)
        if dtype_here is None:
            dtype_here = dat_here.dtype
            dat_net = dat_here
        else:
            dat_here_reordered = dat_here[[  *dtype_here.names ]]
            dat_net = np.concatenate((dat_net, dat_here_reordered))
#        dat_individual.append(dat_here) #np.genfromtxt(name, names=True))

    result = bilby.core.result.Result()
    result.posterior = pd.DataFrame(dat_net) #   np.concatenate(dat_individual))
    if indx_result_to_downselect:
        result.posterior= result.posterior[indx_result_to_downselect] # force order change/downselect to match the files we really have
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
#weight_files = glob(f'{weights_file_directory}/*.dat')
#weight_files.sort(key=sortKeyFunc)
weight_files = fnames_weights

for weight_file in weight_files:
    weights_individual_list.append(np.atleast_1d(np.genfromtxt(weight_file)))

weights = np.concatenate(weights_individual_list)
np.savetxt(f'{outdir}/weights.dat', weights)

# Truncate samples to weight size, if larger
npts_wts = len(weights)
name_0 = result.posterior.keys()[0]
npts_samples = len(result.posterior[name_0])
print(" Sizes ", npts_samples, npts_wts)
weights_for_diagnostics = weights[np.isfinite(weights)]
if npts_samples > len(weights):
    print(" Truncating input samples to weight size (=requested output size) ")
    result.posterior = result.posterior.truncate(after=npts_wts-1)

print(f'Importance sampling efficiency: {np.sum(weights_for_diagnostics)**2/np.sum(weights_for_diagnostics**2)/len(weights_for_diagnostics)}')
if not(allow_alternate):
    result.posterior = bilby.core.result.rejection_sample(result.posterior, weights)
else:
    n_est = int(len(weights)*np.sum(weights_for_diagnostics)**2/np.sum(weights_for_diagnostics**2)/len(weights_for_diagnostics))+1  # effective sample size used to draw
    result.posterior = result.posterior.sample(n=n_est,weights=weights/np.sum(weights),replace=allow_duplicates)

# remove samples with nan or empty entries
result.posterior.dropna(inplace=True)
# field_names_with_cal = [x for x in list(result.posterior.columns) if 'recalib' in x and 'phase' in x]   # warning, hardcodes names
# one_field_name_with_cal = field_names_with_cal[0]
# vals = result.posterior[one_field_name_with_cal]
# bool_bad = np.logical_not(vals.notna())
# vals[bool_bad]= 0 # clean empty items so they process in 
# bool_bad = np.logical_or( bool_bad, np.isinf(vals))
# bool_bad = np.logical_or(bool_bad, np.isnan(vals))
# indx_bad = np.arange(len(bool_bad))[bool_bad]
# result.posterior.drop(indx_bad, inplace=True)

result.save_posterior_samples(filename=outdir+'/reweighted_posterior_samples.dat', outdir=outdir)
