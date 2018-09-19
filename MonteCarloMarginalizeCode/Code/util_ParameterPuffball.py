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
import lalsimutils
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
parser.add_argument("--mc-range",default=None,help="Chirp mass range [mc1,mc2]. Important if we have a low-mass object, to avoid wasting time sampling elsewhere.")
parser.add_argument("--eta-range",default=None,help="Eta range. Important if we have a BNS or other item that has a strong constraint.")
parser.add_argument("--mtot-range",default=None,help="Chirp mass range [mc1,mc2]. Important if we have a low-mass object, to avoid wasting time sampling elsewhere.")
parser.add_argument("--downselect-parameter",action='append', help='Name of parameter to be used to eliminate grid points ')
parser.add_argument("--downselect-parameter-range",action='append',type=str)
opts=  parser.parse_args()

# Extract parameter names
coord_names = opts.parameter # Used  in fit
#if opts.parameter_nofit:
#    coord_names = coord_names + opts.parameter_nofit
if coord_names is None:
    sys.exit(0)
print coord_names


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
cov_in = np.cov(X.T)
cov = cov_in*opts.puff_factor*opts.puff_factor
rv = scipy.stats.multivariate_normal(mean=np.zeros(len(coord_names)), cov=cov)
delta_X = rv.rvs(size=len(X))
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
print " The one dimensional widths are ", np.sqrt(np.diag(cov_out))


# Copy parameters back in.  MAKE SURE THIS IS POSSIBLE
for indx_P in np.arange(len(P_list)):
    for indx in np.arange(len(coord_names)):
        fac=1
        if coord_names[indx] in ['mc','m1','m2','mtot']:
            fac = lal.MSUN_SI
        P_list[indx_P].assign_param( coord_names[indx], X_out[indx_P,indx]*fac)


# Export
lalsimutils.ChooseWaveformParams_array_to_xml(P_list,fname=opts.inj_file_out,fref=P.fref)

    

