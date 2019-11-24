#! /usr/bin/env python
#
# Attempt at semi-automated PP plot.  Shows empirical CDF (p value vs estimate), and 90% credible interval of *most extreme* of the N things being plotted
# 
# FIXME
#   - strip data without convergence test
#   - enable column names 
#
# EXAMPLE
#   python pp_plot.py net_pp.dat_clean 4 "['{\cal M}_c','q','\chi_{1,z}','\chi_{2,z}']"


import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

dpi_base=200
legend_font_base=16
rc_params = {'backend': 'ps',
             'axes.labelsize': 11,
             'axes.titlesize': 10,
             'font.size': 11,
             'legend.fontsize': legend_font_base,
             'xtick.labelsize': 11,
             'ytick.labelsize': 11,
             #'text.usetex': True,
             'font.family': 'Times New Roman'}#,
             #'font.sans-serif': ['Bitstream Vera Sans']}#,
plt.rcParams.update(rc_params)


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

dat = np.genfromtxt(sys.argv[1],filling_values=0,invalid_raise=False,usecols=tuple(np.arange(int(sys.argv[2])+2)))
if len(sys.argv) > 3:
    param_labels=eval(sys.argv[3])#sys.argv[3].split()  # assume a string is passed, with whitespace
else:
    param_labels = np.arange(sys.argv[2])

for indx in np.arange(len(dat[0]) - 2):
    pvals = np.sort(dat[:,indx])
    pvals_emp = np.arange(len(dat))*1.0/len(dat) 
    plt.scatter(pvals,pvals_emp,label='$'+param_labels[indx]+'$')
    

xvals = np.linspace(0,1,100)
plt.plot(xvals,xvals,color='k')
pvals_lims = binomial_credible_interval_default(xvals, len(dat))
plt.plot(xvals,pvals_lims[:,0], color='k',ls=':')
plt.plot(xvals,pvals_lims[:,1], color='k',ls=':')
plt.legend()
plt.xlabel(r"$P(x_{\rm inj})$")
plt.ylabel(r"$\hat{P}$")
plt.savefig("output_pp.pdf",dpi=dpi_base)
