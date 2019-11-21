#! /usr/bin/env python
#
# Attempt at semi-automated PP plot.
# Boundary values not working yet

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm


def binomial_credible_interval(phat,n,z):
    offset = np.sqrt(phat*(1-phat)/n + z**2/(4*n**2))
    my_1 = np.ones((len(phat),2))
    return          1./(1 + z**2/n) *  (np.outer(phat,np.array([1,1])) +  my_1* z**2/(2* n) + np.outer(z *offset, np.array([-1,1])))
    
pvalFiducial = 0.9 # fiducial credible level for outermost CI.  (Will change depending on number of parameters)

def binomial_credible_interval_default(phat,n,nParams=2):
    z_fiducial = norm.ppf((  np.power(1-(1.-pvalFiducial)/2, 1./nParams)))
    print "fiducial z", z_fiducial
    return  binomial_credible_interval(phat,n, z_fiducial)

import sys

dat = np.genfromtxt(sys.argv[1],filling_values=0,invalid_raise=False)

for indx in np.arange(len(dat[0]) - 2):
    pvals = np.sort(dat[:,indx])
    pvals_emp = np.arange(len(dat))*1.0/len(dat) 
    plt.scatter(pvals,pvals_emp,label=indx)
    

xvals = np.linspace(0,1,100)
plt.plot(xvals,xvals,color='k')
pvals_lims = binomial_credible_interval_default(xvals, len(dat))
plt.plot(xvals,pvals_lims[:,0], color='k',ls=':')
plt.plot(xvals,pvals_lims[:,1], color='k',ls=':')
plt.legend()
plt.savefig("output_pp.png")
