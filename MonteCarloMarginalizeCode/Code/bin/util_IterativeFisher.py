#! /usr/bin/env python
#
# util_IterativeFisher.py
#
# BASED ON
#    util_ManualOverlapGrid.py
#
# STRATEGY
#    - compute 1d diagonal terms in fisher matrix via 1d calculations
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
	import NRWaveformCatalogManager3 as nrwf
	hasNR =True
except:
	hasNR=False
try:
    hasEOB=True
    import EOBTidalExternal as eobwf
except:
    hasEOB=False


###
### Linear fits. Resampling a quadratic. (Export me)
###

import RIFT.interpolators.BayesianLeastSquares as BayesianLeastSquares

param_priors_gamma = {'s1z':0.01, 's2z': 0.01, 'xi':0.1}  # weak constraints on s1z, s2z

# def fit_quadratic(x,y,x0=None):
#     """
#     x = array so x[0] , x[1], x[2] are points.
#     """
#     x0_val = np.zeros(len(x[0]))
#     if not (x0 is None):
#         if opts.verbose:
#             print " Fisher: Using reference point ", x0
#         x0_val = x0

#     dim = len(x[0])   
#     npts = len(x)
# #    print x.shape, y.shape
#     if opts.verbose:
#         print " Fisher : dimension, npts = " ,dim, npts
#     # Constant, linear, quadratic functions. 
#     # Beware of lambda:  f_list = [(lambda x: k) for k in range(5)] does not  work, but this does
#     #     f_list = [(lambda x,k=k: k) for k in range(5)]
#     f0 = [lambda z: np.ones(len(z),dtype=np.float128)]
#     # indx_lookup_linear = {}   # protect against packing errors
#     # indx_here = len(f0)
#     # f_linear = []
#     # for k in np.arange(dim):
#     #     f_linear.append( (lambda z,k=k,x0V=x0_val: z.T[k] - x0V[k]))
#     #     indx_lookup_linear[k] =indx_here
#     #     indx_here+=1
#     f_linear = [(lambda z,k=k,x0V=x0_val: z.T[k] - x0V[k]) for k in np.arange(dim)]
#     f_quad = []
#     indx_lookup = {}
#     indx_here =len(f0)+len(f_linear) 
#     for k in np.arange(dim):
#         for q in range(k,dim):
#             f_quad.append( (lambda z,k=k,q=q: (z.T[k] - x0_val[k])*(z.T[q]-x0_val[q]))   )
#             indx_lookup[(k,q)] = indx_here
#             indx_here+=1
#     f_list=f0+f_linear + f_quad
#     n_params_model = len(f_list)
#     F = np.matrix(np.zeros((len(x), n_params_model),dtype=np.float128))
#     for q in np.arange(n_params_model):
#         fval = f_list[q](np.array(x,dtype=np.float128))
# #        print q, f_list[q], fval, fval.shape, len(x)
#         F[:,q] = np.reshape(fval, (len(x),1))
#     if opts.verbose:
#         print " ---- index pattern --- "
#         print indx_lookup
#     gamma = np.matrix( np.diag(np.ones(npts,dtype=np.float128)))
#     Gamma = F.T * gamma * F      # Fisher matrix for the fit
#     Sigma = scipy.linalg.inv(Gamma)  # Covariance matrix for the fit. WHICH CODE YOU USE HERE IS VERY IMPORTANT.
#     if opts.verbose:
#         print " -- should be identity (error here is measure of overall error) --- "
#         print "   Fisher: Matrix inversion/manipulation error ", np.linalg.norm(Sigma*Gamma - np.eye(len(Sigma))) , " which can be large if the fit coordinates are not centered near the peak"
#         print " --  --- "
#     lambdaHat =  np.array((Sigma* F.T*gamma* np.matrix(y).T))[:,0]  # point estimate for the fit parameters (i.e., fisher matrix and best fit point)
#     if opts.verbose:
#         print " Fisher: LambdaHat = ", lambdaHat
#     constant_term_est = lambdaHat[0]  # Constant term
#     linear_term_est = lambdaHat[1:dim+1]  # Coefficient of linear terms
#     my_fisher_est = np.zeros((dim,dim),dtype=np.float64)   #  A SIGNIFICANT LIMITATION...
#     for pair in indx_lookup:
#         k = pair[0]; q=pair[1];
#         indx_here = indx_lookup[pair]
#         my_fisher_est[k,q] += -lambdaHat[indx_here]
#         my_fisher_est[q,k] += -lambdaHat[indx_here]  # this will produce a factor of 2 if the two terms are identical
# #    peak_val_est = F*lambdaHat         # Peak value of the quadratic
#     if opts.verbose:
#         print "  Fisher: ", my_fisher_est
#         print "  Fisher: Sanity check (-0.5)*Fisher matrix vs components (diagonal only) : ", -0.5*my_fisher_est, "versus",  lambdaHat
#     my_fisher_est_inv = scipy.linalg.inv(my_fisher_est)   # SEE INVERSE DISCUSSION
#     if opts.verbose:
#         print " Fisher: Matrix inversion/manipulation error test 2", np.linalg.norm(np.dot(my_fisher_est,my_fisher_est_inv) - np.eye(len(my_fisher_est)))
#     # Peak value:   a - b cinv b/4
# #    print constant_term_est.shape, linear_term_est.shape, my_fisher_est_inv.shape
# #    print my_fisher_est_inv 
# #    print np.dot(my_fisher_est_inv,linear_term_est)
# #    print np.dot(linear_term_est, np.dot(my_fisher_est_inv,linear_term_est))
#     peak_val_est = float(constant_term_est) +np.dot(linear_term_est, np.dot(my_fisher_est_inv,linear_term_est))/2
#     best_val_est = x0_val +  np.dot(my_fisher_est_inv,linear_term_est)   # estimated peak location, including correction for reference point
#     if opts.verbose:
#         print " Fisher : Sanity check: peak value estimate = ", peak_val_est, " which arises as a delicate balance between ",  constant_term_est, " and ",  np.dot(linear_term_est, np.dot(my_fisher_est_inv,linear_term_est))/2
#         print " Fisher : Best coordinate estimate = ", best_val_est
#         print " Fisher : eigenvalues ", np.linalg.eig(my_fisher_est)
# #        print " Fisher : Sanity check: sizes and indexes ", dim*dim, -dim*dim+1, lambdaHat[-dim*dim+1:], len(lambdaHat[-dim*dim+1:])
#     print " WARNING: Constant offsets seen in recovery, tied to base point"
#     return [peak_val_est, best_val_est, my_fisher_est, linear_term_est]


###
### Load options
###

parser = argparse.ArgumentParser()
# Parameters
parser.add_argument("--parameter", action='append')
parser.add_argument("--parameter-range", action='append', type=str,help="Add a range (pass as a string evaluating to a python 2-element list): --parameter-range '[0.,1000.]'   MUST specify ALL parameter ranges (min and max) in order if used")
parser.add_argument("--random-parameter", action='append',help="These parameters are specified at random over the entire range, uncorrelated with the grid used for other parameters.  Use for variables which correlate weakly with others; helps with random exploration")
parser.add_argument("--random-parameter-range", action='append', type=str,help="Add a range (pass as a string evaluating to a python 2-element list): --parameter-range '[0.,1000.]'   MUST specify ALL parameter ranges (min and max) in order if used.  ")
parser.add_argument("--amplitude-order",default=-1,type=int,help="Set ampO for grid. Used in PN")
parser.add_argument("--phase-order",default=7,type=int,help="Set phaseO for grid. Used in PN")
parser.add_argument("--downselect-parameter",action='append', help='Name of parameter to be used to eliminate grid points ')
parser.add_argument("--downselect-parameter-range",action='append',type=str)
parser.add_argument("--enforce-duration-bound",default=None,type=float,help="If present, enforce a duration bound. Used to prevent grid placement for obscenely long signals, when the window size is prescribed")
parser.add_argument("--parameter-value-list", action='append', type=str,help="Add an explicit list of parameter choices to use. ONLY those values will be used. Intended for NR simulations (e.g., q, a1, a2)")
# Use external EOB for source or template?
parser.add_argument("--use-external-EOB-source",action="store_true",help="One external EOB call is performed to generate the reference signal")
parser.add_argument("--use-external-EOB",action="store_true",help="External EOB calls are performed for each template")
# Use external EOB for source or template?
parser.add_argument("--use-external-NR-source",action="store_true",help="One external NR call is performed to generate the reference signal")
parser.add_argument("--use-external-NR",action="store_true",help="External NR calls are performed for each template")
parser.add_argument("--NR-signal-group", default="Sequence-GT-Aligned-UnequalMass",help="Specific NR simulation group to use")
parser.add_argument("--NR-signal-param", default=(0.0,2.),help="Parameter value")
parser.add_argument("--NR-template-group", default=None,help="Specific NR simulation group to use")
parser.add_argument("--NR-template-param", default=None,help="Parameter value")
# Grid layout options
parser.add_argument("--use-fisher", action='store_true',help="Instead of just reporting the overlap results, perform a fit to them, and use the resulting Fisher matrix to select points. REVAMPING IMPLEMENTATION")
parser.add_argument("--fake-data", action='store_true', help="Use perfectly quadratic data. Test for fisher code.")
parser.add_argument("--uniform-spoked", action="store_true", help="Place mass pts along spokes uniform in volume (if omitted placement will be random and uniform in volume")
parser.add_argument("--linear-spoked", action="store_true", help="Place mass pts along spokes linear in radial distance (if omitted placement will be random and uniform in volume")
parser.add_argument("--grid-cartesian", action="store_true", help="Place mass points using a cartesian grid")
parser.add_argument("--grid-cartesian-npts", default=100, type=int)
parser.add_argument("--skip-overlap",action='store_true', help="If true, the grid is generated without actually performing overlaps. Very helpful for uncertain configurations or low SNR")
parser.add_argument("--reset-grid-via-match",action='store_true',help="Reset the parameter_range results so each parameter's range is limited by  match_value.  Use this ONLY for estimating the fisher matrix quickly!")
parser.add_argument("--no-reset-parameter",action='append',help="Don't reset the range of this parameter via tuning. Important for spin parameters, which can be over-tuned due to strong correlations")
parser.add_argument("--use-fisher-resampling",action='store_true',help="Resample the grid using the fisher matrix. Requires fisher matrix")
# Cutoff options
parser.add_argument("--match-value", type=float, default=0.01, help="Use this as the minimum match value. Default is 0.01 (i.e., keep almost everything)")
# Overlap options
parser.add_argument("--fisher-psd",type=str,default="SimNoisePSDaLIGOZeroDetHighPower",help="psd name (attribute in lalsimulation).  SimNoisePSDiLIGOSRD, lalsim.SimNoisePSDaLIGOZeroDetHighPower, lalsimutils.Wrapper_AdvLIGOPsd, .SimNoisePSDiLIGOSRD... ")
parser.add_argument("--psd-file",  help="File name for PSD (assumed hanford). Overrides --fisher-psd if provided")
parser.add_argument("--srate",type=int,default=16384,help="Sampling rate")
parser.add_argument("--seglen", type=float,default=256*2., help="Default window size for processing.")
parser.add_argument("--fref",type=float,default=0.);
# External grid
parser.add_argument("--use-eos", default=None, help="Equation of state to determine lambdas for given mass ranges. Filename, not EOS name (no internal database)")
parser.add_argument("--external-grid-xml", default=None,help="Inspiral XML file (injection form) for alternate grid")
parser.add_argument("--external-grid-txt", default=None, help="Cartesian grid. Must provide parameter names in header. Exactly like output of code. Last column not used.")
# Base point
parser.add_argument("--inj", dest='inj', default=None,help="inspiral XML file containing the base point.")
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
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
parser.add_argument("--save-plots",default=False,action='store_true', help="Write plots to file (only useful for OSX, where interactive is default")
opts=  parser.parse_args()

if opts.verbose:
    True
    #lalsimutils.rosDebugMessagesContainer[0]=True   # enable error logging inside lalsimutils


###
### Handle NR arguments
###
if hasNR and not ( opts.NR_signal_group in nrwf.internal_ParametersAvailable.keys()):
    if opts.NR_signal_group:
        print(" ===== UNKNOWN NR PARAMETER ====== ")
        print(opts.NR_signal_group, opts.NR_signal_param)
elif hasNR:
    if opts.NR_signal_param:
        opts.NR_signal_param = eval(str(opts.NR_signal_param)) # needs to be evaluated
    if not ( opts.NR_signal_param in nrwf.internal_ParametersAvailable[opts.NR_signal_group]):
        print(" ===== UNKNOWN NR PARAMETER ====== ")
        print(opts.NR_signal_group, opts.NR_signal_param)
if hasNR and not ( opts.NR_template_group in nrwf.internal_ParametersAvailable.keys()):
    if opts.NR_template_group:
        print(" ===== UNKNOWN NR PARAMETER ====== ")
        print(opts.NR_template_group, opts.NR_template_param)
elif hasNR:
    if opts.NR_template_param:
        opts.NR_template_param = eval(opts.NR_template_param) # needs to be evaluated
    if not ( opts.NR_template_param in nrwf.internal_ParametersAvailable[opts.NR_template_group]):
        print(" ===== UNKNOWN NR PARAMETER ====== ")
        print(opts.NR_template_group, opts.NR_template_param)




###
### Define grid overlap functions
###   - Python's 'multiprocessing' module seems to cause process lock
###

use_external_EOB=opts.use_external_EOB
Lmax = 2

def eval_overlap(grid,P_list, IP,indx):
    global opts
#    if opts.verbose: 
#        print " Evaluating for ", indx
    global use_external_EOB
    global Lmax
    global opts
    P2 = P_list[indx]
    T_here = 1./IP.deltaF
    P2.deltaF=1./T_here
#    P2.print_params()
    if not opts.skip_overlap:
        if not use_external_EOB:
            hf2 = lalsimutils.complex_hoff(P2)
        else:
            print("  Waiting for EOB waveform ....", indx, " with duration  ", T_here)
            wfP = eobwf.WaveformModeCatalog(P2,lmax=Lmax)  # only include l=2 for us.
            hf2 = wfP.complex_hoff(force_T=T_here)
        nm2 = IP.norm(hf2);  hf2.data.data *= 1./nm2
#    if opts.verbose:
#        print " Waveform normalized for ", indx
        ip_val = IP.ip(hfBase,hf2)
    line_out = []
    line_out = list(grid[indx])
    if not opts.skip_overlap:
        line_out.append(ip_val)
    else:
        line_out.append(-1)
    if opts.verbose:
        print(" Answer ", indx, line_out)
    return line_out

def calc_lambda_from_m(m, eos_fam):
    if m<10**15:
       m=m*lal.MSUN_SI

    #eos=et.read_eos(eos)
    #eos_fam=lalsim.CreateSimNeutronStarFamily(eos)

    k2=lalsim.SimNeutronStarLoveNumberK2(m, eos_fam)
    r=lalsim.SimNeutronStarRadius(m, eos_fam)

    m=m*lal.G_SI/lal.C_SI**2
    lam=2./(3*lal.G_SI)*k2*r**5
    dimensionless_lam=lal.G_SI*lam*(1/m)**5

    return dimensionless_lam


def evaluate_overlap_on_grid(hfbase,param_names, grid):
    global downselect_dict
    # Validate grid is working: Create a loop and print for each one.
    # WARNING: Assumes grid for mass-unit variables hass mass units (!)
    P_list = []
    grid_revised = []
    for line in grid:
        Pgrid = P.manual_copy()
        Pgrid.ampO=opts.amplitude_order  # include 'full physics'
        Pgrid.phaseO = opts.phase_order

        # Set attributes that are being changed as necessary, leaving all others fixed
        for indx in np.arange(len(param_names)):
            Pgrid.assign_param(param_names[indx], line[indx])

        # Downselect
        include_item =True
        if not(opts.enforce_duration_bound is None):
            if lalsimutils.estimateWaveformDuration(Pgrid)> opts.enforce_duration_bound:
                include_item = False
        for param in downselect_dict:
            if Pgrid.extract_param(param) < downselect_dict[param][0] or Pgrid.extract_param(param) > downselect_dict[param][1]:
                include_item =False
        if include_item:
         grid_revised.append(line)
         if Pgrid.m2 <= Pgrid.m1:  # do not add grid elements with m2> m1, to avoid possible code pathologies !
            P_list.append(Pgrid)
         else:
            Pgrid.swap_components()  # IMPORTANT.  This should NOT change the physical functionality FOR THE PURPOSES OF OVERLAP (but will for PE - beware phiref, etc!)
            P_list.append(Pgrid)
        else:
#            print "skipping"
#            Pgrid.print_params()
            True
#            print " skipping "
#    print "Length check", len(P_list), len(grid)
    ###
    ### Loop over grid and make overlaps : see effective fisher code for wrappers
    ###
    #  FIXME: More robust multiprocessing implementation -- very heavy!
#    p=Pool(n_threads)
    # PROBLEM: Pool code doesn't work in new configuration.
    if len(grid_revised) ==0 :
        return [],[]
    grid_out = np.array(map(functools.partial(eval_overlap, grid_revised, P_list,IP), np.arange(len(grid_revised))))
    # Remove mass units at end
    for p in ['mc', 'm1', 'm2', 'mtot']:
        if p in param_names:
            indx = param_names.index(p)
            grid_out[:,indx] /= lal.MSUN_SI
    # remove distance units at end
    for p in ['distance', 'dist']:
        if p in param_names:
            indx = param_names.index(p)
            grid_out[:,indx] /= lal.PC_SI*1e6
    # Truncate grid so overlap with the base point is > opts.min_match. Make sure to CONSISTENTLY truncate all lists (e.g., the P_list)
    grid_out_new = []
    P_list_out_new = []
    for indx in np.arange(len(grid_out)):
        if opts.skip_overlap or grid_out[indx,-1] > opts.match_value:
            grid_out_new.append(grid_out[indx])
            P_list_out_new.append(P_list[indx])
    grid_out = np.array(grid_out_new)
    return grid_out, P_list_out_new



###
### Define base point 
###


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

#    from matplotlib import pyplot as plt
#    plt.plot(eff_fisher_psd.f0+np.arange(eff_fisher_psd.data.length)*eff_fisher_psd.deltaF,np.log10(eff_fisher_psd.data.data))
#    plt.show()



P=lalsimutils.ChooseWaveformParams()
if opts.inj:
    from glue.ligolw import lsctables, table, utils # check all are needed
    filename = opts.inj
    event = opts.event_id
    xmldoc = utils.load_filename(filename, verbose = True,contenthandler =lalsimutils.cthdler)
    sim_inspiral_table = table.get_table(xmldoc, lsctables.SimInspiralTable.tableName)
    P.copy_sim_inspiral(sim_inspiral_table[int(event)])
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

# Define base COMPLEX signal.  ASSUME length long enough via seglen for this  to work always
# Define base COMPLEX overlap 

hfBase = None
if opts.skip_overlap:
    print(" ---- NO WAVEFORM GENERATION ---- ")
    hfBase = None
    IP=lalsimutils.InnerProduct()  # Default, so IP.deltaF code etc does not need to be wrapped
else:
 if hasEOB and opts.use_external_EOB_source:
    print("    -------INTERFACE ------")
    print("    Using external EOB interface (Bernuzzi)   with window  ", opts.seglen)
    # Code WILL FAIL IF LAMBDA=0
    if P.lambda1<1:
        P.lambda1=1
    if P.lambda2<1:
        P.lambda2=1
    if P.deltaT > 1./16384:
        print()
    wfP = eobwf.WaveformModeCatalog(P,lmax=Lmax)  # only include l=2 for us.
    if opts.verbose:
        print(" Duration of stored signal (cut if necessary) ", wfP.estimateDurationSec())
    hfBase = wfP.complex_hoff(force_T=opts.seglen)
    print("EOB waveform length ", hfBase.data.length)
    print("EOB waveform duration", -hfBase.epoch)
 elif opts.use_external_EOB_source and not hasEOB:
    # do not do something else silently!
    print(" Failure: EOB requested but impossible ")
    sys.exit(0)
 elif opts.use_external_NR_source and hasNR:
    m1Msun = P.m1/lal.MSUN_SI;     m2Msun = P.m2/lal.MSUN_SI
    if m1Msun < 50 or m2Msun < 50:
        print(" Invalid NR mass ")
        sys.exit(0)
    print(" Using NR ", opts.NR_signal_group, opts.NR_signal_param)
    T_window = 16. # default 
    wfP = nrwf.WaveformModeCatalog(opts.NR_signal_group, opts.NR_signal_param, clean_initial_transient=True,clean_final_decay=True, shift_by_extraction_radius=True, lmax=opts.lmax,align_at_peak_l2_m2_emission=True,build_strain_and_conserve_memory=True)
    q = wfP.P.m2/wfP.P.m1
    print(" NR q  (overrides anything)", q)
    mtotOrig  =(wfP.P.m1+wfP.P.m2)/lal.MSUN_SI
    wfP.P.m1 *= (m1Msun+m2Msun)/mtotOrig
    wfP.P.m2 *= (m1Msun+m2Msun)/mtotOrig

    wfP.P.deltaT = 1./opts.srate
    print(" NR duration (in s) of simulation at this mass = ", wfP.estimateDurationSec())
    print(" NR starting 22 mode frequency at this mass = ", wfP.estimateFminHz())
    T_window = max([16, 2**int(np.log(wfP.estimateDurationSec())/np.log(2)+1)])
    wfP.P.deltaF = 1./T_window
    print(" Final T_window ", T_window)
    wfP.P.radec = False  # use a real source with a real instrument
    wfP.P.fmin = 10
    print("  ---- NR interface: Overriding parameters to match simulation requested ---- ")
    wfP.P.print_params()
    hfBase = wfP.complex_hoff(force_T=T_window)
 elif opts.use_external_NR_source and not hasNR:
    print(" Failure: NR requested but impossible ")
    sys.exit(0)
 else:
    print("    -------INTERFACE ------")
    print("    Using lalsuite   ", hasEOB, opts.use_external_EOB_source)
    hfBase = lalsimutils.complex_hoff(P)
 IP = lalsimutils.CreateCompatibleComplexOverlap(hfBase,analyticPSD_Q=analyticPSD_Q,psd=eff_fisher_psd,fMax=opts.fmax,interpolate_max=True)
 nmBase = IP.norm(hfBase)
 hfBase.data.data *= 1./nmBase
 if opts.verbose:
    print(" ------  SIGNAL DURATION ----- ")
    print(hfBase.data.length*P.deltaT)

###
### Define parameter ranges to be changed
###

if not opts.parameter:
    param_names =[ 'eta', 'LambdaTilde']  # override options for now
    param_ranges =[ [0.23, 0.25], [0, 1000]]
    pts_per_dim = [ 10,10]
else:
    param_names = opts.parameter
    for param in param_names:
        # Check if in the valid list
        if not(param in lalsimutils.valid_params):
            print(' Invalid param ', param, ' not in ', lalsimutils.valid_params)
            sys.exit(0)
    npts_per_dim = int(np.power(opts.grid_cartesian_npts, 1./len(param_names)))+1
    pts_per_dim = npts_per_dim*np.ones(len(param_names))  # ow!
    param_ranges = []
    if len(param_names) == len(opts.parameter_range):
        param_ranges = map(eval, opts.parameter_range)
        # Rescale hand-specified ranges to SI units
        for p in ['mc', 'm1', 'm2', 'mtot']:
          if p in param_names:
            indx = param_names.index(p)
            #print p, param_names[indx], param_ranges[indx]
            param_ranges[indx]= np.array(param_ranges[indx])* lal.MSUN_SI
        # Rescale hand-specified ranges to SI units
        for p in ['distance','dist']:
          if p in param_names:
            indx = param_names.index(p)
            #print p, param_names[indx], param_ranges[indx]
            param_ranges[indx]= np.array(param_ranges[indx])* lal.PC_SI*1e6
    else:
     for param in param_names:
        if param == 'mc':
            val_center = P.extract_param(param)
            param_ranges.append( [val_center*0.99, val_center*1.01])
        elif param == 'eta':
            val_center = P.extract_param(param)
            mcval = P.extract_param('mc')
            # Use range so limited to smaller body > 1 Msun: HARDCODED
            eta_min = lalsimutils.eta_crit(mcval, 1)
            param_ranges.append( [eta_min, 0.25])
        elif param == 'delta_mc':
            param_ranges.append( [0, 1])
        # FIXME: Implement more parameter ranges, using a lookup-based method
        elif param == 'LambdaTilde':
            #val_center = P.extract_param(param)
            param_ranges.append( [0, 1000]) # HARDCODED
        else:
            print(" Parameter not implemented ", param)
            sys.exit(0)

    if opts.verbose:
        print(" ----- PARAMETER RANGES ----- ")
        for indx in np.arange(len(param_names)):
            print(param_names[indx], param_ranges[indx], pts_per_dim[indx])


template_min_freq = opts.fmin
ip_min_freq = opts.fmin


###
### Auto-tune parameter range based on match threshold
###
if not(opts.skip_overlap) and opts.reset_grid_via_match and opts.match_value <1:
    # Based on the original effective fisher code: 'find_effective_Fisher_region'
    TOL=(1-opts.match_value)*1e-2
    maxit=50
#    TOL=1e-5
    for indx in np.arange(len(param_names)):
        PT = Pbase.copy()  # same as the grid, but we will reset all its parameters
        param_now = param_names[indx]
        if not(opts.no_reset_parameter is None):
          if param_now in opts.no_reset_parameter:
            print(" ==> not retuning range for ", param_now)
            continue
        param_peak = Pbase.extract_param(param_now)
        fac_print =1.0
        if param_now in ['m1', 'm2', 'mc']:
            fac_print = lal.MSUN_SI
        if opts.verbose:
            print(" Optimizing for ", param_now, " with peak expected at value = ", param_peak)
            PT.print_params()
        def ip_here(x):
            PT.assign_param(param_now,x)
            PT.deltaF = IP.deltaF
            hf_now = lalsimutils.complex_hoff(PT)
            nm_now = IP.norm(hf_now)
            val = IP.ip(hfBase,hf_now)/nm_now
            if opts.verbose:
                print(param_now, x, val)
            return val - opts.match_value
        def ip_here_squared(x):
            val= ip_here(x)
            return val**2
        try:
            print(" Minimum: looking between ",param_ranges[indx][0]/fac_print,param_peak/fac_print, " delta ", np.abs(param_peak - param_ranges[indx][0])/fac_print)
#            if np.abs(param_peak - param_ranges[indx][0])/fac_print < 1e-2:  # very narrow placement range. Want to avoid jumping to the wrong side1
#                print " Using minimization code ...  "
#                
#            else:
            param_min = brentq(ip_here,param_ranges[indx][0],param_peak ,xtol=TOL,maxiter=maxit)
            if param_min > param_peak:
                print(" Ordering problem, using minimization code as backup ")
                param_min = brent(ip_here_squared, brack=(param_ranges[indx][0],param_peak, param_ranges[indx][1]), tol=TOL)
        except:
            print("  Range retuning: minimum for ", param_now)
            param_min = param_ranges[indx][0]
        try:
            print(" Maximum: looking between ",param_peak/fac_print, param_ranges[indx][1]/fac_print)
            param_max = brentq(ip_here,param_peak, param_ranges[indx][1],xtol=TOL)
        except:
            print("  Range retuning: maximum for ", param_now)
            param_max = param_ranges[indx][1]
        if np.abs(param_max - param_min)/(np.abs(param_max)+np.abs(param_min)) < 1e-6:  # override if we have catastrophically close placement
            print(" Override: tuned parameters got too close, returning to original range ")
            param_min = param_ranges[indx][0]
            param_max = param_ranges[indx][1]
        print(" Revised range for parameter ", np.array(param_ranges[indx])/fac_print, " to ", [param_min/fac_print,param_max/fac_print], " around ", param_peak/fac_print)
        param_ranges[indx][0] = param_min
        param_ranges[indx][1] = param_max


###
### Downselect parameters
###

downselect_dict = {}
if opts.downselect_parameter:
    dlist = opts.downselect_parameter
    dlist_ranges  = map(eval,opts.downselect_parameter_range)
else:
    dlist = []
    dlist_ranges = []
if len(dlist) != len(dlist_ranges):
    print(" downselect parameters inconsistent", dlist, dlist_ranges)
for indx in np.arange(len(dlist_ranges)):
    downselect_dict[dlist[indx]] = dlist_ranges[indx]

# Enforce Kerr bound
downselect_dict['chi1'] = [0,1]
downselect_dict['chi2'] = [0,1]
for param in ['s1z', 's2z', 's1x','s2x', 's1y', 's2y']:
    downselect_dict[param] = [-1,1]
# Enforce definition of eta
downselect_dict['eta'] = [0,0.25]

print(" Downselect dictionary ", downselect_dict)

# downselection procedure: fix units so I can downselect on mass parameters
for p in ['mc', 'm1', 'm2', 'mtot']:
    if p in downselect_dict.keys():
        downselect_dict[p] = np.array(downselect_dict[p],dtype=np.float64)
        downselect_dict[p] *= lal.MSUN_SI  # add back units


###
### Prior dictionary on coordinates (mainly for Fisher resampling).
###
prior_dict = {}
prior_dict['s1z'] =  2  # provide std dev. Don't want to allow arbitrarily large spins
prior_dict['s2z'] =  2  # provide std dev. Don't want to allow arbitrarily large spins
prior_dict['xi']  = 2    
prior_dict['chieff_aligned']  = 2    
prior_dict['eta'] = 1    # provide st dev. Don't want to allow arbitrary eta.



# Base ranges
print(param_ranges,pts_per_dim)


# Evaluate fisher prototype
n_dim = len(param_ranges)
gamma_fisher_diag = np.zeros((n_dim,n_dim))
for indx in np.arange(n_dim):
    npts_1d=20
    xvals = np.linspace(param_ranges[indx][0], param_ranges[indx][1],npts_1d)
    grid = np.zeros((npts_1d,n_dim))
    for indx_param in np.arange(param_names):
        param_now = param_names[indx_param]
        grid[:,indx_param] = Pbase.extract_param(param_now)*np.ones(npts_1d)
    grid[:,indx] = xvals   # replace this with range
    grid_out, P_list = evaluate_overlap_on_grid(hfBase, param_names, grid)
    # Perform 1d fit
    z = np.polyfit( xvals, grid_out[:,-1],2)
    print(" Fisher term for ", param_now, " = ", -2*z[0])
    gamma_fisher_diag[indx,indx] = -2*z[0]

print(gamma_fisher_diag)

sys.exit(0)

grid_out, P_list = evaluate_overlap_on_grid(hfBase, param_names, grid)
if len(grid_out)==0:
    print(" No points survive....")

if opts.use_fisher:
    rho_fac = 8
    if opts.fake_data:
        grid_out[:,-1] = np.ones(len(grid_out))
        for k in np.arange(len(param_names)):
            grid_out[:,-1] += -1.0*np.power(grid_out[:,k] - np.mean(grid_out[:,k]),2)/np.power(np.std(grid_out[:,k]),2)   # simple quadratic sum in all data. Change to rhange

    # Save data for fisher
    headline = ' '.join(param_names + ['ip'])
    if int(np.version.short_version.split('.')[1])>=7:  # http://docs.scipy.org/doc/numpy/reference/generated/numpy.savetxt.html
        np.savetxt(opts.fname+"_fisher_input.dat", grid_out, header=headline)
    else:
        np.savetxt(opts.fname+"_fisher_input.dat", grid_out)   # 


    # Reference point for fit should NOT MATTER
    x0_val_here =grid_out[0,:len(param_names)]
#    print grid_out[0], x0_val_here
    print(" Generating Fisher matrix using N = ", len(grid_out), " surviving points with match > ", opts.match_value)
    print(" Creating nominal prior on parameters ")
    prior_x_gamma = np.zeros( (len(param_names),len(param_names)) )
    for indx in np.arange(len(param_names)):
        if param_names[indx] in prior_dict:
            prior_x_gamma[indx,indx] = 1./prior_dict[param_names[indx]]**2 /(rho_fac*rho_fac) # must divide, because we work with the scaled Fisher
    the_quadratic_results = BayesianLeastSquares.fit_quadratic( grid_out[:,:len(param_names)], grid_out[:,len(param_names)],x0=x0_val_here,prior_x_gamma=prior_x_gamma)#x0=None)#x0_val_here)
    print("Fisher matrix results (raw) :", the_quadratic_results)
    peak_val_est, best_val_est, my_fisher_est, linear_term_est,fn_estimate = the_quadratic_results
    np.savetxt("fisher_reference.dat",x0_val_here,header=' '.join(param_names)) 
    np.savetxt("fisher_peakval.dat",[peak_val_est])   # generally not very useful
    np.savetxt("fisher_bestpt.dat",best_val_est,header=' '.join(param_names))  
    np.savetxt("fisher_gamma.dat",my_fisher_est,header=' '.join(param_names))
    np.savetxt("fisher_linear.dat",linear_term_est)

    my_eig= scipy.linalg.eig(my_fisher_est)
    if any(np.real(my_eig[0]) < 0) : 
        print(" Negative eigenvalues COULD preclude resampling ! Use a prior to regularize")
        print(" Eigenvalue report ", my_eig)
        print(" HOPE that priors help !")
#        sys.exit(0)





###
### Optional: Write grid to XML file (ONLY if using cutoff option)
###
lalsimutils.ChooseWaveformParams_array_to_xml(P_list, fname=opts.fname, fref=P.fref)

###
### Write output to text file:  p1 p2 p3 ... overlap, only including named params
###
headline = ' '.join(param_names + ['ip'])
if int(np.version.short_version.split('.')[1])>=7:  # http://docs.scipy.org/doc/numpy/reference/generated/numpy.savetxt.html
    np.savetxt(opts.fname+".dat", grid_out, header=headline)
else:
    np.savetxt(opts.fname+".dat", grid_out)   # 


###
### Optional: Scatterplot
###
if opts.save_plots and opts.verbose and len(param_names)==1 and len(grid_out)>0:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(grid_out[:,0], grid_out[:,1])
    plt.savefig("fig-grid2d.png")
#    plt.show()

if opts.save_plots and opts.verbose and len(param_names)==2:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(grid_out[:,0], grid_out[:,1], grid_out[:,2])
    plt.savefig("fig-grid3d.png")
#    plt.show()

print(" ---- DONE ----")
