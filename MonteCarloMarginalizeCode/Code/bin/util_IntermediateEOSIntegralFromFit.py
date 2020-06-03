

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

import joblib
from scipy.integrate import nquad

from glue.ligolw import lsctables, utils, ligolw
lsctables.use_in(ligolw.LIGOLWContentHandler)


parser = argparse.ArgumentParser()
parser.add_argument("--fit-file", help="Name of pkl file")
parser.add_argument("--out-file", default="intermediate_integrand.dat",help="Name of output file (prefix)")
parser.add_argument("--fit-parameter", action='append', help="Parameters altered in fit (assume mc, eta, chi_eff, LambaTilde)")
parser.add_argument("--mc-range",default=None,help="Chirp mass range [mc1,mc2]. Important if we have a low-mass object, to avoid wasting time sampling elsewhere.")
parser.add_argument("--eta-range",default=None,help="Chirp mass range [mc1,mc2]. Important if we have a low-mass object, to avoid wasting time sampling elsewhere.")
parser.add_argument("--integrate-spin",action='store_true')
parser.add_argument("--downselect-parameter",action='append', help='Name of parameter to be used to eliminate grid points ')
parser.add_argument("--downselect-parameter-range",action='append',type=str)
parser.add_argument("--chi-max", default=0.05,type=float,help="Maximum range of 'a' allowed.  Use when comparing to models that aren't calibrated to go to the Kerr limit.")
parser.add_argument("--verbose",action='store_true')
opts=  parser.parse_args()

chi_max = opts.chi_max
mc_range = None # impossible in prac
eta_range =[0.2,0.249999]
if opts.mc_range:
    mc_range = eval(opts.mc_range)  # used in integral
if mc_range == None:
    sys.exit(0)
if opts.eta_range:
    eta_range = eval(opts.eta_range) # needed
delta_mc_range = np.sqrt(1-4*np.array(eta_range))[::-1] # reverse order

gp = joblib.load(opts.fit_file)
my_fit = lambda x: gp.predict(x)


mc_max = mc_range[0]
mc_min =mc_range[1]
def mc_prior(x):
    return x/(mc_max-mc_min)
def eta_prior(x):
    return 1./np.power(x,6./5.)/np.power(1-4.*x, 0.5)/1.44
def delta_mc_prior(x):
    """
    delta_mc = sqrt(1-4eta)  <-> eta = 1/4(1-delta^2)
    Transform the prior above
    """
    eta_here = 0.25*(1 -x*x)
    return 2./np.power(eta_here, 6./5.)/1.44
def s_component_zprior(x,R=chi_max):
    # assume maximum spin =1. Should get from appropriate prior range
    # Integrate[-1/2 Log[Abs[x]], {x, -1, 1}] == 1
    val = -1./(2*R) * np.log( (np.abs(x)/R+1e-7).astype(float))
    return val



# Fiducial values

def integrate_at(eta_here, LambdaTildeHere):
 if not opts.integrate_spin:
  def int_func(mcv):
    etav = eta_here
    LambdaTildev = LambdaTildeHere
    if isinstance(mcv,float):
        m1,m2=lalsimutils.m1m2(mcv,etav)
        delta = (m1 - m2)/(m1+m2)
        return np.exp(my_fit([[mcv,etav,0, LambdaTildev]]))*mc_prior(mcv)
    else:
        m1,m2=lalsimutils.m1m2(mcv,etav)
        delta = (m1 - m2)/(m1+m2)
        chi_eff = (m1*chi1+m2*chi2)/(m1+m2)  # aligned spin
        packed_x = np.zeros((len(mcv),4))
        packed_x[:,0] = mcv
        packed_x[:,1]=etav
        packed_x[:,3]=LambdaTildev
        return np.exp(my_fit(x))*mc_prior(mcv)
  val =scipy.integrate.nquad(int_func, [mc_range])[1]
 else:
  def int_func(mcv,chi1,chi2):
    etav = eta_here
    LambdaTildev = LambdaTildeHere
    if isinstance(mcv,float):
        m1,m2=lalsimutils.m1m2(mcv,etav)
        delta = (m1 - m2)/(m1+m2)
        chi_eff = (m1*chi1+m2*chi2)/(m1+m2)  # aligned spin
        return np.exp(my_fit([[mcv,etav,chi_eff, LambdaTildev]]))*mc_prior(mcv)*s_component_zprior(chi1)*s_component_zprior(chi2)
    else:
        m1,m2=lalsimutils.m1m2(mcv,etav)
        delta = (m1 - m2)/(m1+m2)
        chi_eff = (m1*chi1+m2*chi2)/(m1+m2)  # aligned spin
        packed_x = np.zeros((len(mcv),4))
        packed_x[:,0] = mcv
        packed_x[:,1]=etav
        packed_x[:,2] =chi_eff
        packed_x[:,3]=LambdaTildev
        return np.exp(my_fit(x))*mc_prior(mcv)*s_component_zprior(chi1)*s_component_zprior(chi2)
  val =scipy.integrate.nquad(int_func, [mc_range,[-chi_max,chi_max],[-chi_max,chi_max] ])
 return val

# Test case
eta_here = 0.245
LambdaTildeHere = 200
print(integrate_at(eta_here, LambdaTildeHere))

dat_out = []
for d in np.linspace(delta_mc_range[0],delta_mc_range[1],50):
    for L in np.linspace(1,2000,200):  # need very dense sampling
        dat_out.append([d, L, np.log(integrate_at( 0.25*(1 - d*d), L))])
        if opts.verbose:
            print(dat_out[-1])

np.savetxt(opts.out_file, dat_out)
