#! /usr/bin/env python
#
#


import argparse
import numpy as np
from RIFT.misc.samples_utils import add_field

import sys

import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lalsimutils


remap_ILE_2_LI = {
 "s1z":"a1z", "s2z":"a2z", 
 "s1x":"a1x", "s1y":"a1y",
 "s2x":"a2x", "s2y":"a2y",
 "chi1_perp":"chi1_perp",
 "chi2_perp":"chi2_perp",
 "chi1":'a1',
 "chi2":'a2',
 "cos_phiJL": 'cos_phiJL',
 "sin_phiJL": 'sin_phiJL',
 "cos_theta1":'costilt1',
 "cos_theta2":'costilt2',
 "theta1":"tilt1",
 "theta2":"tilt2",
  "xi":"chi_eff", 
  "chiMinus":"chi_minus", 
  "delta":"delta", 
  "delta_mc":"delta", 
 "mtot":'mtotal', "mc":"mc", "eta":"eta","m1":"m1","m2":"m2",
  "cos_beta":"cosbeta",
  "beta":"beta",
  "LambdaTilde":"lambdat",
  "DeltaLambdaTilde": "dlambdat",
  "thetaJN":"theta_jn",
  "dist":"distance",'phiref':'phiorb'}
remap_LI_to_ILE = { "a1z":"s1z", "a2z":"s2z", "chi_eff":"xi", "lambdat":"LambdaTilde", 'mtotal':'mtot','distance':'dist', 'ra':'phi', 'dec':'theta','phiorb':'phiref','a1x':'s1x','a1y':'s1y', 'a2x':'s2x', 'a2y':'s2y'}




# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--posterior-file",action='append',help="filename of *.dat file [standard LI output]")
parser.add_argument("--flag-tides-in-composite",action='store_true')
parser.add_argument("--composite-file",type=str,help="filename of all.net file, needed to get peak lnL (i.e., log how much this event is pure noise")
parser.add_argument("--truth-file",type=str, help="file containing the true parameters")
parser.add_argument("--truth-event",type=int, default=0,help="file containing the true parameters")
parser.add_argument("--chi-max",type=int, default=1,help="re-impose chi-max")
parser.add_argument("--parameter", action='append',help="parameter name (ILE). Note source-frame masses are only natively supported for LI")
opts=  parser.parse_args()

# Load in truth file
P_ref = None
if opts.truth_file:
    print(" Loading true parameters from  ", opts.truth_file)
    truth_P_list  =lalsimutils.xml_to_ChooseWaveformParams_array(opts.truth_file)
    P_ref = truth_P_list[opts.truth_event]
else:
    sys.exit(0)

lnL_max = -np.inf
lnL_col = 9
if opts.flag_tides_in_composite:
    lnL_col = 11
if opts.composite_file:
    dat = np.loadtxt(opts.composite_file)
    lnL_max = np.max(dat[:,lnL_col])

# Load in posterior samples
fname = opts.posterior_file[0] # just do the one
samples = np.genfromtxt(fname,names=True)
if 'time' in samples.dtype.names:
    samples = add_field(samples, [('tref',float)]); samples['tref'] = samples['time']
if not 'mtotal' in samples.dtype.names and 'mc' in samples.dtype.names:  # raw LI samples use 
    q_here = samples['q']
    eta_here = q_here/(1+q_here)
    mc_here = samples['mc']
    mtot_here = mc_here / np.power(eta_here, 3./5.)
    m1_here = mtot_here/(1+q_here)
    samples = add_field(samples, [('mtotal', float)]); samples['mtotal'] = mtot_here
    samples = add_field(samples, [('eta', float)]); samples['eta'] = eta_here
    samples = add_field(samples, [('m1', float)]); samples['m1'] = m1_here
    samples = add_field(samples, [('m2', float)]); samples['m2'] = mtot_here * q_here/(1+q_here)

if (not 'theta1' in samples.dtype.names)  and ('a1x' in samples.dtype.names):  # probably does not have polar coordinates
    chiperp_here = np.sqrt( samples['a1x']**2+ samples['a1y']**2)
    if any(chiperp_here > 0):
     chi1_here = np.sqrt( samples['a1z']**2 + chiperp_here**2)
     theta1_here = np.arctan(chiperp_here/ samples['a1z'])
     phi1_here = np.angle(samples['a1x']+1j*samples['a1y'])
     samples = add_field(samples, [('chi1', float)]); samples['chi1'] = chi1_here
     samples = add_field(samples, [('theta1', float)]); samples['theta1'] = theta1_here
     samples = add_field(samples, [('phi1', float)]); samples['phi1'] = phi1_here

    chiperp_here = np.sqrt( samples['a2x']**2+ samples['a2y']**2)
    if any(chiperp_here > 0):
     
     chi2_here = np.sqrt( samples['a2z']**2 + chiperp_here**2)
     theta2_here = np.arctan(chiperp_here/ samples['a2z'])
     phi2_here = np.angle(samples['a2x']+1j*samples['a2y'])
     samples = add_field(samples, [('chi2', float)]); samples['chi2'] = chi1_here
     samples = add_field(samples, [('theta2', float)]); samples['theta2'] = theta1_here
     samples = add_field(samples, [('phi2', float)]); samples['phi2'] = phi1_here

elif "theta1" in samples.dtype.names:
    a1x_dat = samples["a1"]*np.sin(samples["theta1"])*np.cos(samples["phi1"])
    a1y_dat = samples["a1"]*np.sin(samples["theta1"])*np.sin(samples["phi1"])
    chi1_perp = samples["a1"]*np.sin(samples["theta1"])

    a2x_dat = samples["a2"]*np.sin(samples["theta2"])*np.cos(samples["phi2"])
    a2y_dat = samples["a2"]*np.sin(samples["theta2"])*np.sin(samples["phi2"])
    chi2_perp = samples["a2"]*np.sin(samples["theta2"])


    samples = add_field(samples, [('a1x', float)]);  samples['a1x'] = a1x_dat
    samples = add_field(samples, [('a1y', float)]); samples['a1y'] = a1y_dat
    samples = add_field(samples, [('a2x', float)]);  samples['a2x'] = a2x_dat
    samples = add_field(samples, [('a2y', float)]);  samples['a2y'] = a2y_dat
    samples = add_field(samples, [('chi1_perp',float)]); samples['chi1_perp'] = chi1_perp
    samples = add_field(samples, [('chi2_perp',float)]); samples['chi2_perp'] = chi2_perp
    if not 'chi_eff' in samples.dtype.names:
        samples = add_field(samples, [('chi_eff',float)]); samples['chi_eff'] = (samples["m1"]*samples["a1z"]+samples["m2"]*samples["a2z"])/(samples["m1"]+samples["m2"])

# add other derived parameters
if 'a1x' in samples.dtype.names:
    if not ('chi1_perp' in samples.dtype.names):
        chi1_perp = np.sqrt(samples['a1x']**2 + samples['a1y']**2)
        chi2_perp = np.sqrt(samples['a2x']**2 + samples['a2y']**2)
        samples = add_field(samples, [('chi1_perp',float)]); samples['chi1_perp'] = chi1_perp
        samples = add_field(samples, [('chi2_perp',float)]); samples['chi2_perp'] = chi2_perp
    if not 'chi_eff' in samples.dtype.names:
        samples = add_field(samples, [('chi_eff',float)]); samples['chi_eff'] = (samples["m1"]*samples["a1z"]+samples["m2"]*samples["a2z"])/(samples["m1"]+samples["m2"])
        samples = add_field(samples, [('xi',float)]); samples['chi_eff'] = (samples["m1"]*samples["a1z"]+samples["m2"]*samples["a2z"])/(samples["m1"]+samples["m2"])



if 'lambda1' in samples.dtype.names and not ('lambdat' in samples.dtype.names):
    Lt,dLt = lalsimutils.tidal_lambda_tilde(samples['m1'], samples['m2'],  samples['lambda1'], samples['lambda2'])
    samples = add_field(samples, [('lambdat', float)]); samples['lambdat'] = Lt
    samples = add_field(samples, [('dlambdat', float)]); samples['dlambdat'] = dLt


if 'chi1_perp' in samples.dtype.names:
    # impose Kerr limit, if neede
    npts = len(samples["m1"])
    indx_ok =np.arange(npts)
    chi1_squared = samples['chi1_perp']**2 + samples["a1z"]**2
    chi2_squared = samples["chi2_perp"]**2 + samples["a2z"]**2
    indx_ok = np.logical_and(chi1_squared < opts.chi_max ,chi2_squared < opts.chi_max)
    npts_out = np.sum(indx_ok)
    new_samples = np.recarray( (npts_out,), dtype=samples.dtype)
    for name in samples.dtype.names:
        new_samples[name] = samples[name][indx_ok]
    samples = new_samples

# Loop over labelled fields
for param in opts.parameter:
    param_here = param
    if param_here in list(remap_LI_to_ILE.keys()):
        param_here = remap_LI_to_ILE[param_here]
    val_here = P_ref.extract_param(param_here)
    if param in [ 'mc', 'm1', 'm2', 'mtotal']:
        val_here= val_here/lal.MSUN_SI
    if param in ['dist','distance']:
        val_here = val_here/(1e6*lal.PC_SI)
 
    if param in ['phiref','phiorb', 'psi']:   # BUG in injection file, this is workaround
        samples[param] = np.mod(samples[param],np.pi)

    n_ok = np.sum(samples[param] < val_here)
    print(n_ok/(1.*len(samples[param])), end=' ')

if opts.composite_file:
    print(lnL_max)
