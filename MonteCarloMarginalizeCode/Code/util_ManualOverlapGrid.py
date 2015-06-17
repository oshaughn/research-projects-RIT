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
# Use external EOB for source or template?
parser.add_argument("--use-external-EOB-source",action="store_true",help="One external EOB call is performed to generate the reference signal")
parser.add_argument("--use-external-EOB",action="store_true",help="External EOB calls are performed for each template")
# Grid layout options
parser.add_argument("--uniform-spoked", action="store_true", help="Place mass pts along spokes uniform in volume (if omitted placement will be random and uniform in volume")
parser.add_argument("--linear-spoked", action="store_true", help="Place mass pts along spokes linear in radial distance (if omitted placement will be random and uniform in volume")
parser.add_argument("--grid-cartesian", action="store_true", help="Place mass points using a cartesian grid")
parser.add_argument("--grid-cartesian-npts", default=100, type=int)
# Cutoff options
parser.add_argument("--match-value", type=float, default=0.97, help="Use this as the minimum match value. Default is 0.97")
# Overlap options
parser.add_argument("--fisher-psd",type=str,default="lalsim.SimNoisePSDiLIGOSRD",help="psd name ('eval'). lalsim.SimNoisePSDaLIGOZeroDetHighPower, lalsim.SimNoisePSDaLIGOZeroDetHighPower, lalsimutils.Wrapper_AdvLIGOPsd, .SimNoisePSDiLIGOSRD... ")
parser.add_argument("--psd-file",  help="File name for PSD (assumed hanford). Overrides --fisher-psd if provided")
parser.add_argument("--srate",type=int,default=16384,help="Sampling rate")
parser.add_argument("--seglen", type=float,default=256*2., help="Default window size for processing.")
parser.add_argument("--fref",type=float,default=0.);
# External grid
parser.add_argument("--external-grid-xml", default=None,help="Inspiral XML file (injection form) for alternate grid")
# Base point
parser.add_argument("--inj", dest='inj', default=None,help="inspiral XML file containing the base point.")
parser.add_argument("--event",type=int, dest="event_id", default=None,help="event ID of injection XML to use.")
parser.add_argument("--fmin", default=35,type=float,help="Mininmum frequency in Hz, default is 40Hz to make short enough waveforms. Focus will be iLIGO to keep comutations short")
parser.add_argument("--mass1", default=1.50,type=float,help="Mass in solar masses")  # 150 turns out to be ok for Healy et al sims
parser.add_argument("--mass2", default=1.35,type=float,help="Mass in solar masses")
parser.add_argument("--lambda1",default=590,type=float)
parser.add_argument("--lambda2", default=590,type=float)
parser.add_argument("--lmax", default=2, type=int)
# Output options
parser.add_argument("--fname", default="overlap-grid", help="Base output file for ascii text (.dat) and xml (.xml.gz)")
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
parser.add_argument("--save-plots",default=False,action='store_true', help="Write plots to file (only useful for OSX, where interactive is default")
opts=  parser.parse_args()

if opts.verbose:
    True
    #lalsimutils.rosDebugMessagesContainer[0]=True   # enable error logging inside lalsimutils


###
### Define grid overlap functions
###   - Python's 'multiprocessing' module seems to cause process lock
###

def eval_overlap(grid,P_list, IP,indx):
#    if opts.verbose: 
#        print " Evaluating for ", indx
    P2 = P_list[indx]
    hf2 = lalsimutils.complex_hoff(P2); nm2 = IP.norm(hf2);  hf2.data.data *= 1./nm2
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
    ###
    ### Loop over grid and make overlaps : see effective fisher code for wrappers
    ###
    #  FIXME: More robust multiprocessing implementation -- very heavy!
#    p=Pool(n_threads)
    # PROBLEM: Pool code doesn't work in new configuration.
    grid_out = np.array(map(functools.partial(eval_overlap, grid, P_list,IP), np.arange(len(grid))))
    # Remove mass units at end
    for p in ['mc', 'm1', 'm2']:
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
    eff_fisher_psd = eval(opts.fisher_psd)
    analyticPSD_Q=True
else:
    sys.exit(0)



P=lalsimutils.ChooseWaveformParams()
if opts.inj:
    from glue.ligolw import lsctables, table, utils # check all are needed
    filename = opts.inj
    event = opts.event_id
    xmldoc = utils.load_filename(filename, verbose = True)
    sim_inspiral_table = table.get_table(xmldoc, lsctables.SimInspiralTable.tableName)
    P.copy_sim_inspiral(sim_inspiral_table[int(event)])
else:    
    P.m1 = opts.mass1 *lal.MSUN_SI
    P.m2 = opts.mass2 *lal.MSUN_SI
    P.dist = 150*1e6*lal.PC_SI
    P.lambda1  = 500
    P.lambda2  = 500
    P.fmin=opts.fmin   # Just for comparison!  Obviously only good for iLIGO
    P.ampO=-1  # include 'full physics'
P.deltaT=1./16384
P.taper = lalsim.SIM_INSPIRAL_TAPER_START
P.deltaF = 1./opts.seglen #lalsimutils.findDeltaF(P)
P.fref = opts.fref
P.print_params()


# Define base COMPLEX signal.  ASSUME length long enough via seglen for this  to work always
# Define base COMPLEX overlap 

if hasEOB and opts.use_external_EOB_source:
    print "    Using external EOB interface (Bernuzzi)    "
    # Code WILL FAIL IF LAMBDA=0
    if P.lambda1<1:
        P.lambda1=1
    if P.lambda2<1:
        P.lambda2=1
    if P.deltaT > 1./16384:
        print 
    wfP = eobwf.WaveformModeCatalog(P,lmax=2)  # only include l=2 for us.
    hfBase = wfP.complex_hoff(P,force_T=True)
    print "EOB waveform length ", hfBase.data.length
else:
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
        for p in ['mc', 'm1', 'm2']:
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

grid_out, P_list = evaluate_overlap_on_grid(hfBase, param_names, grid)


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
if opts.verbose and len(param_names)==1:
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
