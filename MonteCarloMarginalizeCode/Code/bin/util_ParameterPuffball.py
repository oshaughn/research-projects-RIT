#! /usr/bin/env python
#
# GOAL
#   - read in parameter XML
#   - assess parameter covariance (in some named parameters)
#   - generate random numbers, based on that covariance, to add to those parameters, 'puffing up' the distribution
#
# USAGE
#   - tries to follow util_CIP_GC

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


from glue.ligolw import lsctables, utils, ligolw
lsctables.use_in(ligolw.LIGOLWContentHandler)


parser = argparse.ArgumentParser()
parser.add_argument("--inj-file", help="Name of XML file")
parser.add_argument("--inj-file-out", default="output-puffball", help="Name of XML file")
parser.add_argument("--puff-factor", default=1,type=float)
parser.add_argument("--approx-output",default="SEOBNRv2", help="approximant to use when writing output XML files.")
parser.add_argument("--fref",default=20,type=float, help="Reference frequency used for spins in the ILE output.  (Since I usually use SEOBNRv3, the best choice is 20Hz)")
parser.add_argument("--fmin",type=float,default=20)
parser.add_argument("--parameter", action='append', help="Parameters used as fitting parameters AND varied at a low level to make a posterior")
parser.add_argument("--parameter-implied", action='append', help="Parameter used in fit, but not independently varied for Monte Carlo")
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

if opts.random_parameter is None:
    opts.random_parameter = []

# Extract parameter names
coord_names = opts.parameter # Used  in fit
#if opts.parameter_nofit:
#    coord_names = coord_names + opts.parameter_nofit
if coord_names is None:
    sys.exit(0)
print coord_names

downselect_dict = {}
if opts.downselect_parameter:
    dlist = opts.downselect_parameter
    dlist_ranges  = map(eval,opts.downselect_parameter_range)
else:
    dlist = []
    dlist_ranges = []
    opts.downselect_parameter =[]
if len(dlist) != len(dlist_ranges):
    print " downselect parameters inconsistent", dlist, dlist_ranges
for indx in np.arange(len(dlist_ranges)):
    downselect_dict[dlist[indx]] = dlist_ranges[indx]


# Load data
P_list = lalsimutils.xml_to_ChooseWaveformParams_array(opts.inj_file)

# extract parameters to measure the co
dat_out = []
for P in P_list:
    line_out = np.zeros(len(coord_names))
    for x in np.arange(len(coord_names)):
        fac=1
        if coord_names[x] in ['mc','m1','m2','mtot']:
            fac = lal.MSUN_SI
        line_out[x] = P.extract_param(coord_names[x])/fac
        
    dat_out.append(line_out)

# Scale out physical mass

# relabel data
dat_out = np.array(dat_out)
X =dat_out[:,0:len(coord_names)]


# Measure covariance matrix and generate random errors
if len(coord_names) >1:
    cov_in = np.cov(X.T)
    cov = cov_in*opts.puff_factor*opts.puff_factor

    # Check for singularities
    if np.min(np.linalg.eig(cov)[0])<1e-10:
        print " ===> WARNING: SINGULAR MATRIX: are you sure you varied this parameters? <=== "
        icov_pseudo = np.linalg.pinv(cov)
        # Prior range for each parameter is 1000, so icov diag terms are 10^(-6)
        # This is somewhat made up, but covers most things
        diag_terms = 1e-6*np.ones(len(cov))
        # 
        icov_proposed = icov_pseudo+np.diag(diag_terms)
        cov= np.linalg.inv(icov_proposed)

    # Compute errors
    rv = scipy.stats.multivariate_normal(mean=np.zeros(len(coord_names)), cov=cov,allow_singular=True)  # they are just complaining about dynamic range of parameters, usually
    delta_X = rv.rvs(size=len(X))
    X_out = X+delta_X
else:
    sigma = np.std(X)
    cov = sigma*sigma
    delta_X =np.random.normal(size=len(coord_names), scale=sigma)
    X_out = X+delta_X

# Sanity check parameters
for indx in np.arange(len(coord_names)):
    if coord_names[indx] == 'eta':
        X_out[:,indx] = np.minimum(X_out[:,indx], 0.25)
        X_out[:,indx] = np.maximum(X_out[:,indx], 0.01)
    if coord_names[indx] == 's1z' or coord_names[indx]=='s2z':
        X_out[:,indx] = np.minimum(X_out[:,indx], 0.99)
        X_out[:,indx] = np.maximum(X_out[:,indx], -0.99)


cov_out = np.cov(X_out.T)
print " Covariance change: The following two matrices should be (A) and (1+puff^2)A, where puff= ", opts.puff_factor
print cov
print  cov_out
if len(coord_names)>1:
    print " The one dimensional widths are ", np.sqrt(np.diag(cov_out))
else:
    print " The one dimensional width is", np.sqrt(cov_out)

# Copy parameters back in.  MAKE SURE THIS IS POSSIBLE
P_out = []
for indx_P in np.arange(len(P_list)):
    include_item=True
    P = P_list[indx_P]
    for indx in np.arange(len(coord_names)):
        fac=1
        # sanity check restrictions, which may cause problems with the coordinate converters
        if coord_names[indx] is 'eta' and (X_out[indx_P,indx]>0.25 or out[indx_P,indx]<0.001) :
            continue
        if coord_names[indx] is 'delta_mc' and (X_out[indx_P,indx]>1 or out[indx_P,indx]<0.) :
            continue
        if coord_names[indx] in ['mc','m1','m2','mtot']:
            fac = lal.MSUN_SI
        P_list[indx_P].assign_param( coord_names[indx], X_out[indx_P,indx]*fac)

    if not(opts.enforce_duration_bound is None):
      if lalsimutils.estimateWaveformDuration(P)> opts.enforce_duration_bound:
        include_item = False
    for param in downselect_dict:
        val = P.extract_param(param)
        if param in ['mc','m1','m2','mtot']:
            val = val/ lal.MSUN_SI
        if val < downselect_dict[param][0] or val > downselect_dict[param][1]:
            include_item =False
    if include_item:
        P_out.append(P)




# Randomize parameters that have been requested to be randomized
#   - note there is NO SANITY CHECKING if you do this
#   - target: tidal parameters, more efficiently hammer on low-tide corner if necessary
if len(opts.random_parameter) >0:
  random_ranges = {}   
  for indx in np.arange(len(opts.random_parameter)):
    param = opts.random_parameter[indx]
    random_ranges[param] = np.array(eval(opts.random_parameter_range[indx]))
  for P in P_out: 
    for param in opts.random_parameter:
        val = np.random.uniform( random_ranges[param][0], random_ranges[param][1])
        if param in ['mc','m1','m2','mtot']:
            val = val* lal.MSUN_SI
        P.assign_param(param,val)


# Export
lalsimutils.ChooseWaveformParams_array_to_xml(P_out,fname=opts.inj_file_out,fref=P.fref)

    

