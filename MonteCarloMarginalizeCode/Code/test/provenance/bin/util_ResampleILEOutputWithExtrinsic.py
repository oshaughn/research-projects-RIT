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

print(" Input: ", opts.fname)
print(" Output: ", opts.fname_out)

