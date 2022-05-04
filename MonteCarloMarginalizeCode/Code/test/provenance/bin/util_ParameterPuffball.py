#! /usr/bin/env python
#
# GOAL
#   - read in parameter XML
#   - assess parameter covariance (in some named parameters)
#   - generate random numbers, based on that covariance, to add to those parameters, 'puffing up' the distribution
#
# USAGE
#   - tries to follow util_CIP_GC
#
# EXAMPLES
#    util_ManualOverlapGrid.py  --skip-overlap --parameter mc --parameter-range [1,2] --parameter eta --parameter-range [0.1,0.2] --parameter s1z --parameter-range [-1,1] --parameter s2z --parameter-range [-1,1] 
#    python util_ParameterPuffball.py  --parameter mc --parameter eta --no-correlation "['mc','eta']" --parameter s1z --parameter s2z --inj-file ./overlap-grid.xml.gz  --no-correlation "['mc','s1z']"
#   python util_ParameterPuffball.py  --parameter mc --parameter eta --parameter s1z --parameter s2z --inj-file ./overlap-grid.xml.gz   --force-away 0.4

# PROBLEMS
#    - if points are too dense (i.e, if the output size gets too large) then we will reject everything, even for uniform placement.  
#    - current implementation produces pairwise distance matrix, so can be memory-hungry for many points


import argparse
import sys
import numpy as np
import numpy.lib.recfunctions
import functools
import itertools




parser = argparse.ArgumentParser()
parser.add_argument("--inj-file", help="Name of XML file")
parser.add_argument("--inj-file-out", default="output-puffball", help="Name of XML file")
parser.add_argument("--puff-factor", default=1,type=float)
parser.add_argument("--force-away", default=0,type=float,help="If >0, uses the icov to compute a metric, and discards points which are close to existing points")
parser.add_argument("--approx-output",default="SEOBNRv2", help="approximant to use when writing output XML files.")
parser.add_argument("--fref",default=None,type=float, help="Reference frequency used for spins in the ILE output.  (Since I usually use SEOBNRv3, the best choice is 20Hz). Default is to use what is in the original overlap-grid.xml.gz file")
parser.add_argument("--fmin",type=float,default=None,help="Min frequency, default is to use what is in original file")
parser.add_argument("--parameter", action='append', help="Parameters used as fitting parameters AND varied at a low level to make a posterior")
parser.add_argument("--no-correlation", type=str,action='append', help="Pairs of parameters, in format [mc,eta]  The corresponding term in the covariance matrix is eliminated")
#parser.add_argument("--parameter-implied", action='append', help="Parameter used in fit, but not independently varied for Monte Carlo")
parser.add_argument("--random-parameter", action='append',help="These parameters are specified at random over the entire range, uncorrelated with the grid used for other parameters.  Use for variables which correlate weakly with others; helps with random exploration")
parser.add_argument("--random-parameter-range", action='append', type=str,help="Add a range (pass as a string evaluating to a python 2-element list): --parameter-range '[0.,1000.]'   MUST specify ALL parameter ranges (min and max) in order if used.  ")
parser.add_argument("--mc-range",default=None,help="Chirp mass range [mc1,mc2]. Important if we have a low-mass object, to avoid wasting time sampling elsewhere.")
parser.add_argument("--eta-range",default=None,help="Eta range. Important if we have a BNS or other item that has a strong constraint.")
parser.add_argument("--mtot-range",default=None,help="Chirp mass range [mc1,mc2]. Important if we have a low-mass object, to avoid wasting time sampling elsewhere.")
parser.add_argument("--downselect-parameter",action='append', help='Name of parameter to be used to eliminate grid points ')
parser.add_argument("--downselect-parameter-range",action='append',type=str)
parser.add_argument("--enforce-duration-bound",default=None,type=float,help="If present, enforce a duration bound. Used to prevent grid placement for obscenely long signals, when the window size is prescribed")
parser.add_argument("--regularize",action='store_true',help="Add some ad-hoc terms based on priors, to help with nearly-singular matricies")
opts=  parser.parse_args()

print(" Arguments: ", ' '.join(sys.argv[1:]))
print(" Inputs: ", opts.inj_file)
print(" Outputs: ", opts.inj_file_out)

# Export
import os
os.system(" touch "+opts.inj_file_out + ".xml.gz")

    

