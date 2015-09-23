#! /usr/bin/env python
#
# util_ManualOverlapGrid.py
#
# EXAMPLES
#   util_ManualOverlapGrid.py --inj inj.xml.gz --parameter LambdaTilde  # 1d grid in changing LambdaTilde
#   util_ManualOverlapGrid.py  --verbose --parameter LambdaTilde --parameter-range '[0,1000]'
#   util_ManualOverlapGrid.py  --verbose --parameter LambdaTilde --parameter-range '[0,1000]' --parameter eta --parameter-range '[0.23,0.25]' --grid-cartesian-npts 10
#
# EOB SOURCE EXAMPLES
#
#   util_ManualOverlapGrid.py --inj inj.xml.gz --parameter LambdaTilde  --parameter-range '[0,1000]' --grid-cartesian-npts 10 --use-external-EOB-source
#   util_ManualOverlapGrid.py --inj inj.xml.gz --parameter LambdaTilde  --parameter-range '[0,1000]' --grid-cartesian-npts 10 --use-external-EOB-source --use-external-EOB
#  

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


import argparse
import sys
import numpy as np
import lalsimutils
import lalsimulation as lalsim
import lalframe
import lal
import functools

import effectiveFisher  as eff   # for the mesh grid generation
import PrecessingFisherMatrix   as pcf   # Superior tools to perform overlaps. Will need to standardize with Evans' approach in effectiveFisher.py

from multiprocessing import Pool
try:
    import os
    n_threads = int(os.environ['OMP_NUM_THREADS'])
    print " Pool size : ", n_threads
except:
    n_threads=1
    print " - No multiprocessing - "

try:
	import NRWaveformCatalogManager as nrwf
	hasNR =True
except:
	hasNR=False
try:
    hasEOB=True
    import EOBTidalExternal as eobwf
except:
    hasEOB=False


###
### Load options
###

parser = argparse.ArgumentParser()
# Parameters
parser.add_argument("--parameter", action='append')
parser.add_argument("--parameter-range", action='append', type=str,help="Add a range (pass as a string evaluating to a python 2-element list): --parameter-range '[0.,1000.]'   MUST specify ALL parameter ranges (min and max) in order if used")
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
parser.add_argument("--uniform-spoked", action="store_true", help="Place mass pts along spokes uniform in volume (if omitted placement will be random and uniform in volume")
parser.add_argument("--linear-spoked", action="store_true", help="Place mass pts along spokes linear in radial distance (if omitted placement will be random and uniform in volume")
parser.add_argument("--grid-cartesian", action="store_true", help="Place mass points using a cartesian grid")
parser.add_argument("--grid-cartesian-npts", default=100, type=int)
# Cutoff options
parser.add_argument("--match-value", type=float, default=0.01, help="Use this as the minimum match value. Default is 0.01 (i.e., keep almost everything)")
# Overlap options
parser.add_argument("--fisher-psd",type=str,default="SimNoisePSDaLIGOZeroDetHighPower",help="psd name (attribute in lalsimulation).  SimNoisePSDiLIGOSRD, lalsim.SimNoisePSDaLIGOZeroDetHighPower, lalsimutils.Wrapper_AdvLIGOPsd, .SimNoisePSDiLIGOSRD... ")
parser.add_argument("--psd-file",  help="File name for PSD (assumed hanford). Overrides --fisher-psd if provided")
parser.add_argument("--srate",type=int,default=16384,help="Sampling rate")
parser.add_argument("--seglen", type=float,default=256*2., help="Default window size for processing.")
parser.add_argument("--fref",type=float,default=0.);
# External grid
parser.add_argument("--external-grid-xml", default=None,help="Inspiral XML file (injection form) for alternate grid")
parser.add_argument("--external-grid-txt", default=None, help="Cartesian grid. Must provide parameter names in header. Exactly like output of code. Last column not used.")
# Base point
parser.add_argument("--inj", dest='inj', default=None,help="inspiral XML file containing the base point.")
parser.add_argument("--event",type=int, dest="event_id", default=None,help="event ID of injection XML to use.")
parser.add_argument("--fmin", default=35,type=float,help="Mininmum frequency in Hz, default is 40Hz to make short enough waveforms. Focus will be iLIGO to keep comutations short")
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
        print " ===== UNKNOWN NR PARAMETER ====== "
        print opts.NR_signal_group, opts.NR_signal_param
elif hasNR:
    if opts.NR_signal_param:
        opts.NR_signal_param = eval(str(opts.NR_signal_param)) # needs to be evaluated
    if not ( opts.NR_signal_param in nrwf.internal_ParametersAvailable[opts.NR_signal_group]):
        print " ===== UNKNOWN NR PARAMETER ====== "
        print opts.NR_signal_group, opts.NR_signal_param
if hasNR and not ( opts.NR_template_group in nrwf.internal_ParametersAvailable.keys()):
    if opts.NR_template_group:
        print " ===== UNKNOWN NR PARAMETER ====== "
        print opts.NR_template_group, opts.NR_template_param
elif hasNR:
    if opts.NR_template_param:
        opts.NR_template_param = eval(opts.NR_template_param) # needs to be evaluated
    if not ( opts.NR_template_param in nrwf.internal_ParametersAvailable[opts.NR_template_group]):
        print " ===== UNKNOWN NR PARAMETER ====== "
        print opts.NR_template_group, opts.NR_template_param




###
### Define grid overlap functions
###   - Python's 'multiprocessing' module seems to cause process lock
###

use_external_EOB=opts.use_external_EOB
Lmax = 2

def eval_overlap(grid,P_list, IP,indx):
#    if opts.verbose: 
#        print " Evaluating for ", indx
    global use_external_EOB
    global Lmax
    P2 = P_list[indx]
    T_here = 1./IP.deltaF
    P2.deltaF=1./T_here
#    P2.print_params()
    if not use_external_EOB:
        hf2 = lalsimutils.complex_hoff(P2)
    else:
        print "  Waiting for EOB waveform ....", indx, " with duration  ", T_here
        wfP = eobwf.WaveformModeCatalog(P2,lmax=Lmax)  # only include l=2 for us.
        hf2 = wfP.complex_hoff(force_T=T_here)
    nm2 = IP.norm(hf2);  hf2.data.data *= 1./nm2
#    if opts.verbose:
#        print " Waveform normalized for ", indx
    ip_val = IP.ip(hfBase,hf2)
    line_out = []
    line_out = list(grid[indx])
    line_out.append(ip_val)
    if opts.verbose:
        print " Answer ", indx, line_out
    return line_out

def evaluate_overlap_on_grid(hfbase,param_names, grid):
    # Validate grid is working: Create a loop and print for each one.
    # WARNING: Assumes grid for mass-unit variables hass mass units (!)
    P_list = []
    for line in grid:
        Pgrid = P.manual_copy()
        # Set attributes that are being changed as necessary, leaving all others fixed
        for indx in np.arange(len(param_names)):
            Pgrid.assign_param(param_names[indx], line[indx])
        P_list.append(Pgrid)
#    print "Length check", len(P_list), len(grid)
    ###
    ### Loop over grid and make overlaps : see effective fisher code for wrappers
    ###
    #  FIXME: More robust multiprocessing implementation -- very heavy!
#    p=Pool(n_threads)
    # PROBLEM: Pool code doesn't work in new configuration.
    grid_out = np.array(map(functools.partial(eval_overlap, grid, P_list,IP), np.arange(len(grid))))
    # Remove mass units at end
    for p in ['mc', 'm1', 'm2', 'mtot']:
        if p in param_names:
            indx = param_names.index(p)
            grid_out[:,indx] /= lal.MSUN_SI
    # Truncate grid so overlap with the base point is > opts.min_match. Make sure to CONSISTENTLY truncate all lists (e.g., the P_list)
    grid_out_new = []
    P_list_out_new = []
    for indx in np.arange(len(grid_out)):
        if grid_out[indx,-1] > opts.match_value:
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
    eff_fisher_psd = lalsimutils.load_resample_and_clean_psd(opts.psd_file, 'H1', 1./opts.seglen)
    analyticPSD_Q = False



P=lalsimutils.ChooseWaveformParams()
if opts.inj:
    from glue.ligolw import lsctables, table, utils # check all are needed
    filename = opts.inj
    event = opts.event_id
    xmldoc = utils.load_filename(filename, verbose = True,contenthandler =lalsimutils.cthdler)
    sim_inspiral_table = table.get_table(xmldoc, lsctables.SimInspiralTable.tableName)
    P.copy_sim_inspiral(sim_inspiral_table[int(event)])
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
    P.ampO=-1  # include 'full physics'
    if opts.approx:
        P.approx = lalsim.GetApproximantFromString(opts.approx)
        if not (P.approx in [lalsim.TaylorT1,lalsim.TaylorT2, lalsim.TaylorT3, lalsim.TaylorT4]):
            # Do not use tidal parameters in approximant which does not implement them
            print " Do not use tidal parameters in approximant which does not implement them "
            P.lambda1 = 0
            P.lambda2 = 0
    else:
        P.approx = lalsim.GetApproximantFromString("TaylorT4")
P.deltaT=1./16384
P.taper = lalsim.SIM_INSPIRAL_TAPER_START
P.deltaF = 1./opts.seglen #lalsimutils.findDeltaF(P)
P.fref = opts.fref
P.print_params()


# Define base COMPLEX signal.  ASSUME length long enough via seglen for this  to work always
# Define base COMPLEX overlap 

if hasEOB and opts.use_external_EOB_source:
    print "    -------INTERFACE ------"
    print "    Using external EOB interface (Bernuzzi)   with window  ", opts.seglen
    # Code WILL FAIL IF LAMBDA=0
    if P.lambda1<1:
        P.lambda1=1
    if P.lambda2<1:
        P.lambda2=1
    if P.deltaT > 1./16384:
        print 
    wfP = eobwf.WaveformModeCatalog(P,lmax=Lmax)  # only include l=2 for us.
    if opts.verbose:
        print " Duration of stored signal (cut if necessary) ", wfP.estimateDurationSec()
    hfBase = wfP.complex_hoff(force_T=opts.seglen)
    print "EOB waveform length ", hfBase.data.length
    print "EOB waveform duration", -hfBase.epoch
elif opts.use_external_EOB_source and not hasEOB:
    # do not do something else silently!
    print " Failure: EOB requested but impossible "
    sys.exit(0)
elif opts.use_external_NR_source and hasNR:
    m1Msun = P.m1/lal.MSUN_SI;     m2Msun = P.m2/lal.MSUN_SI
    if m1Msun < 50 or m2Msun < 50:
        print " Invalid NR mass "
        sys.exit(0)
    print " Using NR ", opts.NR_signal_group, opts.NR_signal_param
    T_window = 16. # default 
    wfP = nrwf.WaveformModeCatalog(opts.NR_signal_group, opts.NR_signal_param, clean_initial_transient=True,clean_final_decay=True, shift_by_extraction_radius=True, lmax=opts.lmax,align_at_peak_l2_m2_emission=True,build_strain_and_conserve_memory=True)
    q = wfP.P.m2/wfP.P.m1
    print " NR q  (overrides anything)", q
    mtotOrig  =(wfP.P.m1+wfP.P.m2)/lal.MSUN_SI
    wfP.P.m1 *= (m1Msun+m2Msun)/mtotOrig
    wfP.P.m2 *= (m1Msun+m2Msun)/mtotOrig

    wfP.P.deltaT = 1./opts.srate
    print " NR duration (in s) of simulation at this mass = ", wfP.estimateDurationSec()
    print " NR starting 22 mode frequency at this mass = ", wfP.estimateFminHz()
    T_window = max([16, 2**int(np.log(wfP.estimateDurationSec())/np.log(2)+1)])
    wfP.P.deltaF = 1./T_window
    print " Final T_window ", T_window
    wfP.P.radec = False  # use a real source with a real instrument
    wfP.P.fmin = 10
    print "  ---- NR interface: Overriding parameters to match simulation requested ---- "
    wfP.P.print_params()
    hfBase = wfP.complex_hoff(force_T=T_window)
elif opts.use_external_NR_source and not hasNR:
    print " Failure: NR requested but impossible "
    sys.exit(0)
else:
    print "    -------INTERFACE ------"
    print "    Using lalsuite   ", hasEOB, opts.use_external_EOB_source
    hfBase = lalsimutils.complex_hoff(P)
IP = lalsimutils.CreateCompatibleComplexOverlap(hfBase,analyticPSD_Q=analyticPSD_Q,psd=eff_fisher_psd)
nmBase = IP.norm(hfBase)
hfBase.data.data *= 1./nmBase
if opts.verbose:
    print " ------  SIGNAL DURATION ----- "
    print hfBase.data.length*P.deltaT

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
            print ' Invalid param ', param, ' not in ', lalsimutils.valid_params
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
        # FIXME: Implement more parameter ranges, using a lookup-based method
        elif param == 'LambdaTilde':
            #val_center = P.extract_param(param)
            param_ranges.append( [0, 1000]) # HARDCODED
        else:
            print " Parameter not implemented ", param
            sys.exit(0)

    if opts.verbose:
        print " ----- PARAMETER RANGES ----- "
        for indx in np.arange(len(param_names)):
            print param_names[indx], param_ranges[indx], pts_per_dim[indx]



template_min_freq = opts.fmin
ip_min_freq = opts.fmin



###
### Lay out grid, currently CARTESIAN.   OPTIONAL: Load grid from file
###
# WARNINGS: The code will NOT enforce sanity, and can produce lambda's < 0. This may cause some codes to barf.
# FIXME: Use seed cartesian grid to help lay out an effective Fisher grid


# Base Cartesian grid
grid_tuples = eff.make_regular_1d_grids(param_ranges, pts_per_dim)
# Strip unphysical parameters
print "  NEED TO IMPLEMENT: Stripping of unphysical parameters "
grid = eff.multi_dim_grid(*grid_tuples)  # eacy line in 'grid' is a set of parameter values

# If external grid provided, erase this grid and set of names, and replace it with the new one.
if opts.external_grid_txt:
    tmp = np.genfromtxt(opts.external_grid_txt, names=True)
    raw_names = tmp.dtype.names
#    print tmp, tmp['eta'], tmp['ip'], raw_names
    param_names = np.array(list(set(raw_names) - set(['ip'])))
    print param_names
    grid = np.array(tmp[param_names])
#    print grid, grid[1][0]
#    sys.exit(0)

grid_out, P_list = evaluate_overlap_on_grid(hfBase, param_names, grid)
if len(grid_out)==0:
    print " No points survive...."

###
### (Fisher matrix-based grids): 
###     - Use seed cartesian grid to compute the effective fisher matrix
###     - Loop *again* to evaluate overlap on that grid
###
if opts.linear_spoked or opts.uniform_spoked:
    print " Effective fisher report. GRID NOT YET IMPLEMENTED "
    if len(param_names)==2:
        fitgamma = eff.effectiveFisher(eff.residuals2d, grid_out[:,-1], *grid_out[:,0:len(param_names)-1])
        gam = eff.array_to_symmetric_matrix(fitgamma)
        evals, evecs, rot = eff.eigensystem(gam)
        # Print information about the effective Fisher matrix
        # and its eigensystem
        print "Least squares fit finds ", fitgamma
        print "\nFisher matrix:"
        print "eigenvalues:", evals
        print "eigenvectors:"
        print evecs
        print "rotation "
        print rot

    else:
        print " Higher-dimensional grids not yet implemented "
        sys.exit(0)
    print "Fisher grid not yet implemented"
    sys.exit(0)
                             

###
### Write output to text file:  p1 p2 p3 ... overlap, only including named params
###
headline = ' '.join(param_names + ['ip'])
np.savetxt(opts.fname+".dat", grid_out, header=headline)

###
### Optional: Write grid to XML file (ONLY if using cutoff option)
###
lalsimutils.ChooseWaveformParams_array_to_xml(P_list, fname=opts.fname, fref=P.fref)


###
### Optional: Scatterplot
###
if opts.verbose and len(param_names)==1 and len(grid_out)>0:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(grid_out[:,0], grid_out[:,1])
    plt.show()

if opts.verbose and len(param_names)==2:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(grid_out[:,0], grid_out[:,1], grid_out[:,2])
    plt.show()

print " ---- DONE ----"
