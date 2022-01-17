#! /usr/bin/env python
#
# util_ManualOverlapGrid.py
#
# EXAMPLES
#   util_ManualOverlapGrid.py --inj inj.xml.gz --parameter LambdaTilde  # 1d grid in changing LambdaTilde
#   util_ManualOverlapGrid.py  --verbose --parameter LambdaTilde --parameter-range '[0,1000]'
#   util_ManualOverlapGrid.py  --verbose --parameter LambdaTilde --parameter-range '[0,1000]' --parameter eta --parameter-range '[0.23,0.25]' --grid-cartesian-npts 10
#   util_ManualOverlapGrid.py --parameter s1z --parameter-range '[-0.9,0.9]' --parameter s2z --parameter-range '[-0.9,0.9]' --downselect-parameter xi --downselect-parameter-range '[-0.1,0.1]' --skip-overlap --verbose
#
# EXAMPLES WITH RANDOM FIELDS
#   python util_ManualOverlapGrid.py  --parameter mc --parameter-range [2,2.1] --random-parameter s1z --random-parameter-range [-1,1] --skip-overlap --random-parameter lambda1 --random-parameter-range [3,500] --random-parameter lambda2 --random-parameter-range [3,500]
#
# EOB SOURCE EXAMPLES
#
#   util_ManualOverlapGrid.py --inj inj.xml.gz --parameter LambdaTilde  --parameter-range '[0,1000]' --grid-cartesian-npts 10 --use-external-EOB-source
#   util_ManualOverlapGrid.py --inj inj.xml.gz --parameter LambdaTilde  --parameter-range '[0,1000]' --grid-cartesian-npts 10 --use-external-EOB-source --use-external-EOB
#   python util_ManualOverlapGrid.py --parameter s1x --parameter-range [-1,1] --parameter s1y --parameter-range [-1,1] --parameter s1z --parameter-range [-1,1] --skip-overlap  --verbose # check Kerr bound is enforced
#  
#  util_ManualOverlapGrid.py --parameter s1z --parameter-range '[-0.5,0.5]'  --parameter eta --parameter-range '[0.2,0.2499]' --verbose --seglen 16 --mass1 10 --mass2 6 --use-fisher --match-val 0.95 --approx SpinTaylorT4

# EXAMPLES: Iterative Fisher matrix creation (converging on higher match)
#   python util_ManualOverlapGrid.py  --parameter eta --parameter-range '[0.2,0.2499]' --verbose --seglen 32 --mass1 10 --mass2 6 --use-fisher --match-val 0.97 --reset-grid-via-match
#  python util_ManualOverlapGrid.py --parameter s1z --parameter-range '[-0.1,0.1]'  --parameter eta --parameter-range '[0.2,0.2499]' --verbose --seglen 16 --mass1 10 --mass2 6 --use-fisher --match-val 0.90 --approx SpinTaylorT4 --grid-cartesian-npts 500 --reset-grid-via-match
#  python util_ManualOverlapGrid.py  --parameter eta --parameter-range '[0.2,0.2499]' --verbose --seglen 32 --mass1 10 --mass2 6 --use-fisher --match-val 0.97 --reset-grid-via-match --parameter mc --parameter-range '[6.6,6.8]'
# python util_ManualOverlapGrid.py --parameter s1z --parameter-range '[-0.3,0.3]'  --parameter eta --parameter-range '[0.2,0.2499]' --verbose --seglen 16 --mass1 10 --mass2 6 --use-fisher --match-val 0.97 --approx SpinTaylorT4 --grid-cartesian-npts 500 --reset-grid-via-match --parameter mc --parameter-range '[6.6,6.8]'
# python util_ManualOverlapGrid.py --parameter s1z --parameter-range '[-0.3,0.3]'  --parameter eta --parameter-range '[0.2,0.2499]' --verbose --seglen 8 --mass1 35 --mass2 32 --use-fisher --match-val 0.97 --approx SEOBNRv2 --grid-cartesian-npts 100 --reset-grid-via-match --parameter mc --parameter-range '[28,32]'
#
# IDEA
#    - pass a list of parameters and a list of ranges
#    - signal generated on a grid using these parameters. Default layout is cartesian grid; can use others
#    - 
#
#
# ISSUES
#    - default is to regenerate the signals as needed.  Give option to archive them (very memory-painful!)
#      May want a caching interface on disk?
#    - Option to load grid from file: standard xml (injection format)
#    - default is to work for an ALIGNED-SPIN BINARY, not for more generic sources.
#
#    - enable automatic rejection of systems that violate the kerr bound

import argparse
import sys
import numpy as np
import scipy
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lalframe
import lal
import functools
import itertools

from scipy.optimize import brentq


import RIFT.physics.effectiveFisher  as eff   # for the mesh grid generation
import RIFT.physics.PrecessingFisherMatrix   as pcf   # Superior tools to perform overlaps. Will need to standardize with Evans' approach in effectiveFisher.py

from multiprocessing import Pool
try:
    import os
    n_threads = int(os.environ['OMP_NUM_THREADS'])
    print(" Pool size : ", n_threads)
except:
    n_threads=1
    print(" - No multiprocessing - ")


try:
    from gwbench import network
except:
    print(" Requires gwbench to work")
    import sys
    sys.exit(1)


# Arguments: try to use same structure as MOG
parser = argparse.ArgumentParser()
# Parameters
parser.add_argument("--parameter", action='append')
parser.add_argument("--parameter-range", action='append', type=str,help="Add a range (pass as a string evaluating to a python 2-element list): --parameter-range '[0.,1000.]'   MUST specify ALL parameter ranges (min and max) in order if used")
parser.add_argument("--random-parameter", action='append',help="These parameters are specified at random over the entire range, uncorrelated with the grid used for other parameters.  Use for variables which correlate weakly with others; helps with random exploration")
parser.add_argument("--random-parameter-range", action='append', type=str,help="Add a range (pass as a string evaluating to a python 2-element list): --parameter-range '[0.,1000.]'   MUST specify ALL parameter ranges (min and max) in order if used.  ")
parser.add_argument("--latin-hypercube-sampling", action='store_true', help="use latin hypercube sampling from smt on all parameters called 'parameter'")
parser.add_argument("--amplitude-order",default=-1,type=int,help="Set ampO for grid. Used in PN")
parser.add_argument("--phase-order",default=7,type=int,help="Set phaseO for grid. Used in PN")
parser.add_argument("--downselect-parameter",action='append', help='Name of parameter to be used to eliminate grid points ')
parser.add_argument("--downselect-parameter-range",action='append',type=str)
parser.add_argument("--enforce-duration-bound",default=None,type=float,help="If present, enforce a duration bound. Used to prevent grid placement for obscenely long signals, when the window size is prescribed")
parser.add_argument("--parameter-value-list", action='append', type=str,help="Add an explicit list of parameter choices to use. ONLY those values will be used. Intended for NR simulations (e.g., q, a1, a2)")
# Use external EOB for source or template?
# Grid layout options
parser.add_argument("--fake-data", action='store_true', help="Use perfectly quadratic data. Test for fisher code.")
parser.add_argument("--uniform-spoked", action="store_true", help="Place mass pts along spokes uniform in volume (if omitted placement will be random and uniform in volume")
parser.add_argument("--linear-spoked", action="store_true", help="Place mass pts along spokes linear in radial distance (if omitted placement will be random and uniform in volume")
parser.add_argument("--grid-cartesian", action="store_true", help="Place mass points using a cartesian grid")
parser.add_argument("--grid-cartesian-npts", default=100, type=int)
parser.add_argument("--reset-grid-via-match",action='store_true',help="Reset the parameter_range results so each parameter's range is limited by  match_value.  Use this ONLY for estimating the fisher matrix quickly!")
# Cutoff options
parser.add_argument("--match-value", type=float, default=0.01, help="Use this as the minimum match value. Default is 0.01 (i.e., keep almost everything)")
# Overlap options
parser.add_argument("--fisher-psd",type=str,default="SimNoisePSDaLIGOZeroDetHighPower",help="psd name (attribute in lalsimulation).  SimNoisePSDiLIGOSRD, lalsim.SimNoisePSDaLIGOZeroDetHighPower, lalsimutils.Wrapper_AdvLIGOPsd, .SimNoisePSDiLIGOSRD... ")
parser.add_argument("--psd-file",  help="File name for PSD (assumed hanford). Overrides --fisher-psd if provided")
parser.add_argument("--srate",type=int,default=16384,help="Sampling rate")
parser.add_argument("--seglen", type=float,default=64., help="Default window size for processing.")
parser.add_argument("--fref",type=float,default=0.);
# Base point
parser.add_argument("--inj", dest='inj', default=None,help="inspiral XML file containing the base point.")
parser.add_argument("--inj-file-out", default="overlap-grid", help="Name of XML file")
parser.add_argument("--event",type=int, dest="event_id", default=None,help="event ID of injection XML to use.")
parser.add_argument("--fmin", default=35,type=float,help="Mininmum frequency in Hz, default is 40Hz to make short enough waveforms. Focus will be iLIGO to keep comutations short")
parser.add_argument("--fmax",default=2000,type=float,help="Maximum frequency in Hz, used for PSD integral.")
parser.add_argument("--mass1", default=1.50,type=float,help="Mass in solar masses")  # 150 turns out to be ok for Healy et al sims
parser.add_argument("--mass2", default=1.35,type=float,help="Mass in solar masses")
parser.add_argument("--s1z", default=0.,type=float,help="Spin1z")
#parser.add_argument("--lambda1",default=590,type=float)
#parser.add_argument("--lambda2", default=590,type=float)
parser.add_argument("--eff-lambda", type=float, help="Value of effective tidal parameter. Optional, ignored if not given")
parser.add_argument("--deff-lambda", type=float, help="Value of second effective tidal parameter. Optional, ignored if not given")
parser.add_argument("--lmax", default=2, type=int)
parser.add_argument("--approx",type=str,default=None)
# Output options
parser.add_argument("--fname", default="overlap-grid", help="Base output file for ascii text (.dat) and xml (.xml.gz)")
parser.add_argument("--verbose", action="store_true",default=False, help="Extra warnings")
parser.add_argument("--extra-verbose", action="store_true",default=False, help="Lots of messages")
parser.add_argument("--save-plots",default=False,action='store_true', help="Write plots to file (only useful for OSX, where interactive is default")
opts=  parser.parse_args()

if opts.verbose:
    True


downselect_dict = {}

# Add some pre-built downselects, to avoid common out-of-range-error problems
downselect_dict['chi1'] = [0,1]
downselect_dict['chi2'] = [0,1]
downselect_dict['eta'] = [0,0.25]
downselect_dict['m1'] = [0,1e10]
downselect_dict['m2'] = [0,1e10]


chi1_prior_limit = 1
chi2_prior_limit =1
if 's1z' in downselect_dict:
    chi1_prior_limit =np.max(downselect_dict['s1z'])
if 's2z' in downselect_dict:
    chi2_prior_limit =np.max(downselect_dict['s2z'])

if opts.downselect_parameter:
    dlist = opts.downselect_parameter
    dlist_ranges  = list(map(eval,opts.downselect_parameter_range))
else:
    dlist = []
    dlist_ranges = []
    opts.downselect_parameter =[]
if len(dlist) != len(dlist_ranges):
    print(" downselect parameters inconsistent", dlist, dlist_ranges)
for indx in np.arange(len(dlist_ranges)):
    downselect_dict[dlist[indx]] = dlist_ranges[indx]



# Handle PSD
# FIXME: Change to getattr call, instead of 'eval'
eff_fisher_psd = lalsim.SimNoisePSDiLIGOSRD
if not opts.psd_file:
    #eff_fisher_psd = eval(opts.fisher_psd)
    eff_fisher_psd = getattr(lalsim, opts.fisher_psd)   # --fisher-psd SimNoisePSDaLIGOZeroDetHighPower   now
    analyticPSD_Q=True
else:
    print(" Importing PSD file ", opts.psd_file)
    eff_fisher_psd = lalsimutils.load_resample_and_clean_psd(opts.psd_file, 'H1', 1./opts.seglen)
    analyticPSD_Q = False



P=lalsimutils.ChooseWaveformParams()
if opts.inj:
    from ligo.lw import lsctables, table, utils # check all are needed
    filename = opts.inj
    event = opts.event_id
    xmldoc = utils.load_filename(filename, verbose = True,contenthandler =lalsimutils.cthdler)
    sim_inspiral_table = table.get_table(xmldoc, lsctables.SimInspiralTable.tableName)
    P.copy_sim_inspiral(sim_inspiral_table[int(event)])
    P.fmin =opts.fmin
    if opts.approx:
        P.approx = lalsim.GetApproximantFromString(opts.approx)
        if not (P.approx in [lalsim.TaylorT1,lalsim.TaylorT2, lalsim.TaylorT3, lalsim.TaylorT4]):
            # Do not use tidal parameters in approximant which does not implement them
            print(" Do not use tidal parameters in approximant which does not implement them ")
            P.lambda1 = 0
            P.lambda2 = 0    
else:    
    P.m1 = opts.mass1 *lal.MSUN_SI
    P.m2 = opts.mass2 *lal.MSUN_SI
    P.s1z = opts.s1z
    P.dist = 150*1e6*lal.PC_SI
    if opts.eff_lambda and Psig:
        lambda1, lambda2 = 0, 0
        if opts.eff_lambda is not None:
            lambda1, lambda2 = lalsimutils.tidal_lambda_from_tilde(m1, m2, opts.eff_lambda, opts.deff_lambda or 0)
            Psig.lambda1 = lambda1
            Psig.lambda2 = lambda2

    P.fmin=opts.fmin   # Just for comparison!  Obviously only good for iLIGO
    P.ampO=opts.amplitude_order  # include 'full physics'
    P.phaseO = opts.phase_order
    if opts.approx:
        P.approx = lalsim.GetApproximantFromString(opts.approx)
        if not (P.approx in [lalsim.TaylorT1,lalsim.TaylorT2, lalsim.TaylorT3, lalsim.TaylorT4]):
            # Do not use tidal parameters in approximant which does not implement them
            print(" Do not use tidal parameters in approximant which does not implement them ")
            P.lambda1 = 0
            P.lambda2 = 0
    else:
        P.approx = lalsim.GetApproximantFromString("TaylorT4")
P.deltaT=1./16384
P.taper = lalsim.SIM_INSPIRAL_TAPER_START
P.deltaF = 1./opts.seglen #lalsimutils.findDeltaF(P)
P.fref = opts.fref
P.print_params()
Pbase = P.copy()




### CALCULATION

# choose the desired detectors
network_spec = ['aLIGO_H','aLIGO_L','aLIGO_V']
# initialize the network with the desired detectors
net = network.Network(network_spec)

# choose the desired waveform 
wf_model_name = 'tf2'
# pass the chosen waveform to the network for initialization
net.set_wf_vars(wf_model_name=wf_model_name)

# pick the desired frequency range
f = np.arange(opts.fmin,opts.fmax,1./opts.seglen)


# set the injection parameters
inj_params = {
    'Mc':    P.extract_param('mc')/lal.MSUN_SI,
    'eta':   P.extract_param('eta'),
    'chi1z': P.s1z,
    'chi2z': P.s2z,
    'DL':    P.dist/lal.PC_SI/1e6,
    'tc':    0,
    'phic':  0,
    'iota':  0,
    'ra':    np.pi/4,
    'dec':   np.pi/4,
    'psi':   np.pi/4,
    'gmst0': 0
    }

print(inj_params)

# assign with respect to which parameters to take derivatives
# should be based on PARAMETER values  ... but assume just Mc eta chi1z chi2z for now (univesal)
deriv_symbs_string = 'Mc eta chi1z chi2z tc phic psi'


# pass all these variables to the network
net.set_net_vars(
    f=f, inj_params=inj_params,
    deriv_symbs_string=deriv_symbs_string,
    use_rot=0
    )


# setup antenna patterns, location phase factors, and PSDs
net.setup_ant_pat_lpf_psds()

# Test if files are present
#   NOT CURRENTLY DONE

# compute the detector responses and their derivatives
net.load_det_responses_derivs_sym()
net.calc_det_responses_derivs_sym()

# compute the detector responses
net.calc_det_responses()

# calculate the network and detector SNRs
net.calc_snrs()

# calculate the network and detector Fisher matrices, condition numbers,
# covariance matrices, error estimates, and inversion errors
net.calc_errors()


print(net.snr)
print(net.fisher)

# Marginalize out 3d subspace
#  NO REGULARIZATION ATTEMPTED RIGHT NOW
fisher_extr = net.fisher[-3:,-3:]
A = net.fisher[:-3,:-3]
B = net.fisher[:-3,-3:]
print(fisher_extr)
C=fisher_extr_inv = np.linalg.pinv(fisher_extr)

fisher_marg = A - np.dot(B,np.dot(C,B.T))
fisher_marg  += np.diag([1,4,1./chi1_prior_limit**2, 1./chi2_prior_limit**2])   # regularize, important for second spin term AND if we have prior hard limits on spin
#print(fisher_marg)

#print(np.linalg.eig(fisher_marg))



# Generate random numbers
cov = np.linalg.pinv(fisher_marg)


# Add to base points
    # Compute errors

coord_names=['mc','eta','s1z','s2z']
X = np.zeros((opts.grid_cartesian_npts,len(coord_names)))
for indx in np.arange(len(coord_names)):
    X[:,indx] = P.extract_param(coord_names[indx])
    if indx==0:
        X[:,indx]*= 1./lal.MSUN_SI

rv = scipy.stats.multivariate_normal(mean=np.zeros(len(coord_names)), cov=cov,allow_singular=True)  # they are just complaining about dynamic range of parameters, usually
delta_X = rv.rvs(size=len(X))
X_out = X+delta_X


# Sanity check parameters
#   - eta check, done by hand
is_ok= np.logical_and(X_out[:,1] < 0.25, X_out[:,1]>0.01)
X_out = X_out[is_ok]
#   - other checks, force into range
for indx in np.arange(len(coord_names)):
    if coord_names[indx] == 'eta':
        X_out[:,indx] = np.minimum(X_out[:,indx], 0.25)
        X_out[:,indx] = np.maximum(X_out[:,indx], 0.01)
    if coord_names[indx] == 's1z' or coord_names[indx]=='s2z':
        X_out[:,indx] = np.minimum(X_out[:,indx], 0.99)
        X_out[:,indx] = np.maximum(X_out[:,indx], -0.99)

print(" Generated random points raw ", len(X_out))
#print(X_out, delta_X)



## SPECIAL OUTPUT CHANGES
#   - use a weight, so the parameters are drawn from the prior (np.random.choice of the points, using a weight function based on the prior)



# Convert to P list
P_out = []
for indx_P in np.arange(len(X_out)):
    include_item=True
    P_new = P.copy()
    for indx in np.arange(len(coord_names)):
        fac=1
        # sanity check restrictions, which may cause problems with the coordinate converters
        if coord_names[indx] == 'eta' and (X_out[indx_P,indx]>0.25 or X_out[indx_P,indx]<0.001) :
#            print(" Rej eta")
            continue
        if coord_names[indx] == 'delta_mc' and (X_out[indx_P,indx]>1 or X_out[indx_P,indx]<0.) :
 #           print(" Rej delta_mc")
            continue
        if coord_names[indx] in ['mc','m1','m2','mtot']:
            fac = lal.MSUN_SI
        P_new.assign_param( coord_names[indx], X_out[indx_P,indx]*fac)

    if np.isnan(P_new.m1) or np.isnan(P_new.m2):  # don't allow nan mass
        continue

    for param in downselect_dict:
        val = P_new.extract_param(param)
        if np.isnan(val):
            include_item=False   # includes check on m1,m2
            continue # stop trying to calculate with this parameter
        if param in ['mc','m1','m2','mtot']:
            val = val/ lal.MSUN_SI
        if val < downselect_dict[param][0] or val > downselect_dict[param][1]:
            include_item =False
#            print(" Rej downselect",param)

    if include_item:
        P_out.append(P_new)



print(" Retained random points  ", len(P_out))


# Final export
# Export
lalsimutils.ChooseWaveformParams_array_to_xml(P_out,fname=opts.inj_file_out,fref=P.fref)
