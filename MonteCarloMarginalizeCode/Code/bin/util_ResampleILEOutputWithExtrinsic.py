#! /usr/bin/env python
#
# GOAL
#   - take output of concatenated 'flatfile_points' files (=concatennated output of convert_output_format_ile2inference)
#   - assume concatenation is a FAIR DRAW FROM THE EXTRINSIC POSTERIOR 
#   - resample to produce extrinsic parameters and intrinsic parameters
#   
#
# ORIGINAL REFERENCES
#    - postprocess_1d_cumulative.py (based on 'flatfile-points.dat', original format for storing output)
#    - util_ConstructIntrinsicPosterior_GenericCoordinates.py
#
# EXAMPLES
#    convert_output_format_ile2inference zero_noise.xml.gz > flatfile-points.dat   
#    util_ResampleILEOutputWithExtrinsic.py --fname flatfile-points.dat  --n-output-samples 5


# Setup. 
import numpy as np
import lal
import RIFT.lalsimutils as lalsimutils
import bisect


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fname", default=None,help="Target file to read. Output of convert_output_format_ile2inference. Can merge, but must retain columns!")
parser.add_argument("--fname-out", default="output-with-extrinsic.xml.gz",help="Target output file, including extrinsic variables. No cosmology or source-frame masses...user must do that themselves.")
parser.add_argument("--save-P",default=0.,type=float,help="Not currently used")
parser.add_argument("--n-output-samples",default=2000,type=int)
opts = parser.parse_args()

samples = np.genfromtxt(opts.fname,names=True,invalid_raise=False)
lnLmax = np.max(samples["lnL"])
p = samples["p"]     # only prior!
ps = samples["ps"]  # sampling prior
Npts = samples["Npts"] # how many points in this iteration
weights = np.exp( samples["lnL"]-lnLmax) * (p/ps)/Npts   # each point will produce Npts underlying samples. Treat each *experiment* equally (instead of each pt)...suboptimal 

# if save_P is implemented, we will need to sort the weights, so we can downselect the unimportant ones

# Draw random numbers
p_threshold_size = opts.n_output_samples #np.min([5*opts.n_output_samples,len(weights)])
#p_thresholds =  np.random.uniform(low=0.0,high=1.0,size=p_threshold_size)#opts.n_output_samples)
#cum_sum  = np.cumsum(weights)
#cum_sum = cum_sum/cum_sum[-1]
#indx_list = map(lambda x : np.sum(cum_sum < x),  p_thresholds)  # this can lead to duplicates
indx_list = np.random.choice(np.arange(len(weights)), size=p_threshold_size, p=weights/np.sum(weights))
#print indx_list, samples["lnL"][indx_list], samples["distance"][indx_list], weights[indx_list], np.max(weights), samples["lnL"], lnLmax
#print  np.mean(samples["lnL"][indx_list]), np.mean(samples["distance"][indx_list]), weights[indx_list]


# downselect
samples_out = np.empty( len(indx_list), samples.dtype)
for param in samples.dtype.names:
    samples_out[param] = samples[param][indx_list]


# Write output (ascii)
# Vastly superior data i/o with pandas, but note no # used
import pandas
dframe = pandas.DataFrame(samples_out)
fname_out_ascii= opts.fname_out.replace(".xml.gz", "")+".dat"
dframe.to_csv(fname_out_ascii,sep=' ',header=' # '+' '.join(samples_out.dtype.names),index=False) # # is not pre-appended...just boolean

    
