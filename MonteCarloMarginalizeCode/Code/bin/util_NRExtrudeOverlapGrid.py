#! /usr/bin/env python
#
# util_NRExtrudeOverlapGrid.py
#
#   This takes the whole NR catalog (aligned spin only!) and extrudes it in mtot (only)
#   Comparisons for cutoff are done ONLY using the 'approx' in LAL, for convenience
#
# EXAMPLES
#
#     # grid targeted to a single simulation
#        python util_NRExtrudeOverlapGrid.py --inj mdc.xml.gz  --group Sequence-SXS-All --param 1 --event 0 --verbose --skip-overlap
#     # extend the grid, to include the rest of the simlations
#        python util_NRExtrudeOverlapGrid.py --inj overlap-grid.xml.gz  --group Sequence-SXS-All  --event 0 --verbose --skip-overlap --insert-missing-spokes
#
# WARNINGS: i
#   xi_factor: meaning of grid is CHANGED from original interpretation
#


import argparse
import sys
import numpy as np
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lalframe
import lal
import functools

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
### Load options
###

parser = argparse.ArgumentParser()
# Parameters
parser.add_argument("--mtot-range",default='[50,110]')
parser.add_argument("--mc-range",default=None, help='If specified, overrides the total mass range [24,38]')
parser.add_argument("--grid-cartesian-npts",default=30)
parser.add_argument("--group",default=None)
parser.add_argument("--param", action='append', help='Explicit list of parameters to use')
parser.add_argument("--insert-missing-spokes", action='store_true')
parser.add_argument("--aligned-only",action='store_true')
parser.add_argument("--nospin-only",action='store_true')
parser.add_argument("--eccentricity",action='store_true')
parser.add_argument("--eta-range",default='[0.1,0.25]')
parser.add_argument("--mass-xi-factor",default=0.0,type=float, help="The mass ranges are assumed to apply at ZERO SPIN. For other values of xi, the mass ranges map to m_{used} = m(1+xi *xi_factor+(1/4-eta)*eta_factor).  Note that to be stable, xi_factor<1. Default value 0.6, based on relevant mass region")
parser.add_argument("--mass-eta-factor",default=0,type=float, help="The mass ranges are assumed to apply at ZERO SPIN. For other values of xi, the mass ranges map to m_{used} = m(1+xi *xi_factor+(1/4-eta)*eta_factor).  Note that to be stable, xi_factor<1. Default value 0.6, based on relevant mass region")
# Cutoff options
parser.add_argument("--skip-overlap",action='store_true', help="If true, the grid is generated without actually performing overlaps. Very helpful if the grid is just in mtot, for the purposes of reproducing a specific NR simulation")
parser.add_argument("--match-value", type=float, default=0.01, help="Use this as the minimum match value. Default is 0.01 (i.e., keep almost everything)")
# Overlap options
parser.add_argument("--fisher-psd",type=str,default="SimNoisePSDaLIGOZeroDetHighPower",help="psd name ('eval'). lalsim., lalsim.SimNoisePSDaLIGOZeroDetHighPower, lalsimutils.Wrapper_AdvLIGOPsd, .SimNoisePSDiLIGOSRD... ")
parser.add_argument("--psd-file",  help="File name for PSD (assumed hanford). Overrides --fisher-psd if provided")
parser.add_argument("--srate",type=int,default=16384,help="Sampling rate")
parser.add_argument("--seglen", type=float,default=64., help="Default window size for processing. Short for NR waveforms")
parser.add_argument("--fref",type=float,default=0.);
# External grid
parser.add_argument("--external-grid-xml", default=None,help="Inspiral XML file (injection form) for alternate grid")
parser.add_argument("--external-grid-txt", default=None, help="Cartesian grid. Must provide parameter names in header. Exactly like output of code. Last column not used.")
# Base point
parser.add_argument("--inj", dest='inj', default=None,help="inspiral XML file containing the base point (OR containing the reference XML grid to check against a database)")
parser.add_argument("--event",type=int, dest="event_id", default=None,help="event ID of injection XML to use.")
parser.add_argument("--fmin", default=10,type=float,help="Mininmum frequency in Hz, default is 40Hz to make short enough waveforms. Focus will be iLIGO to keep comutations short")
parser.add_argument("--require-fmin-above-NR-start",action='store_true')
parser.add_argument("--fmax",default=2000,type=float,help="Maximum frequency in Hz, used for PSD integral.")
parser.add_argument("--mass1", default=35,type=float,help="Mass in solar masses")  # 150 turns out to be ok for Healy et al sims
parser.add_argument("--mass2", default=35,type=float,help="Mass in solar masses")
parser.add_argument("--s1z", default=0.1,type=float,help="Spin1z")
#parser.add_argument("--lambda1",default=590,type=float)
#parser.add_argument("--lambda2", default=590,type=float)
parser.add_argument("--eff-lambda", type=float, help="Value of effective tidal parameter. Optional, ignored if not given")
parser.add_argument("--deff-lambda", type=float, help="Value of second effective tidal parameter. Optional, ignored if not given")
parser.add_argument("--lmax", default=2, type=int)
parser.add_argument("--approx",type=str,default="SEOBNRv2")
# Output options
parser.add_argument("--fname", default="overlap-grid", help="Base output file for ascii text (.dat) and xml (.xml.gz)")
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
parser.add_argument("--save-plots",default=False,action='store_true', help="Write plots to file (only useful for OSX, where interactive is default")
opts=  parser.parse_args()

xi_factor = opts.mass_xi_factor
eta_factor0 = opts.mass_eta_factor

if opts.verbose:
    True
    #lalsimutils.rosDebugMessagesContainer[0]=True   # enable error logging inside lalsimutils



###
### Define grid overlap functions
###   - Python's 'multiprocessing' module seems to cause process lock
###

Lmax = 2

def eval_overlap(grid,P_list, IP,indx):
#    if opts.verbose: 
#        print " Evaluating for ", indx
    global Lmax
    global opts
    P2 = P_list[indx]
#    P2.print_params()
    line_out = []
    line_out = list(grid[indx])
    if not opts.skip_overlap:
        T_here = 1./IP.deltaF
        P2.deltaF=1./T_here
        hf2 = lalsimutils.complex_hoff(P2)
        nm2 = IP.norm(hf2);  hf2.data.data *= 1./nm2
        ip_val = IP.ip(hfBase,hf2)
        line_out.append(ip_val)
    else:
        line_out.append(-1)
    if opts.verbose:
            print(" Answer ", indx, line_out)
    return line_out

def evaluate_overlap_on_grid(hfbase,param_names, grid):
    # Validate grid is working: Create a loop and print for each one.
    # WARNING: Assumes grid for mass-unit variables hass mass units (!)
    global xi_factor
    P_list = []
    for line in grid:
        Pgrid = P.manual_copy()
        # Set attributes that are being changed as necessary, leaving all others fixed
        for indx in np.arange(len(param_names)):
                Pgrid.assign_param(param_names[indx], line[indx])   # DANGER: need to keep overlap values
        # Rescale mass parameters using the xi factor
        xi_now = Pgrid.extract_param('xi')
        eta_now = Pgrid.extract_param('eta')
            # THIS CAN BE NEGATIVE: PATHOLOGICAL
        Pgrid.m1 *= (1.+xi_now*xi_factor - eta_factor0*(0.25-eta_now))
        Pgrid.m2 *= (1.+xi_now*xi_factor - eta_factor0*(0.25-eta_now))
        P_list.append(Pgrid)
#    print "Length check", len(P_list), len(grid)
    ###
    ### Loop over grid and make overlaps : see effective fisher code for wrappers
    ###
    #  FIXME: More robust multiprocessing implementation -- very heavy!
#    p=Pool(n_threads)
    # PROBLEM: Pool code doesn't work in new configuration.
    grid_out = np.array(list(map(functools.partial(eval_overlap, grid, P_list,IP), np.arange(len(grid)))))
    # Remove mass units at end
    for p in ['mc', 'm1', 'm2', 'mtot']:
        if p in param_names:
            indx = param_names.index(p)
            grid_out[:,indx] /= lal.MSUN_SI
    # Truncate grid so overlap with the base point is > opts.min_match. Make sure to CONSISTENTLY truncate all lists (e.g., the P_list)
    grid_out_new = []
    P_list_out_new = []
    for indx in np.arange(len(grid_out)):
      if P_list[indx].m1 >0 and P_list[indx].m2>0:   # reject insane cases where the masses were scaled to be <0
        if (opts.skip_overlap) or (grid_out[indx,-1] > opts.match_value):
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
    from ligo.lw import lsctables, table, utils # check all are needed
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


print("    -------INTERFACE ------")
hfBase =None
IP=None
if not opts.skip_overlap:
    hfBase = lalsimutils.complex_hoff(P)
    IP = lalsimutils.CreateCompatibleComplexOverlap(hfBase,analyticPSD_Q=analyticPSD_Q,psd=eff_fisher_psd,fMax=opts.fmax)
    nmBase = IP.norm(hfBase)
    hfBase.data.data *= 1./nmBase
    if opts.verbose:
        print(" ------  SIGNAL DURATION ----- ")
        print(hfBase.data.length*P.deltaT)



###
### If we are testing an existing grid to see if simulations are missing
###

# Step 1: Load in the grid, identify NR simulations

P_list_nr = []
gp_list = []
if opts.inj and opts.insert_missing_spokes:
    import spokes
    sdHere = spokes.LoadSpokeXML(opts.inj)
    for key in sdHere:
        P_here =sdHere[key][0]
        P_list_nr.append(P_here) 
        compare_dict = {}
        compare_dict['q'] = P_here.m2/P_here.m1 # Need to match the template parameter. NOTE: VERY IMPORTANT that P is updated with the event params
        compare_dict['s1z'] = P_here.s1z
        compare_dict['s1x'] = P_here.s1x
        compare_dict['s1y'] = P_here.s1y
        compare_dict['s2z'] = P_here.s2z
        compare_dict['s2x'] = P_here.s2x
        compare_dict['s2y'] = P_here.s2y
        good_sim_list = nrwf.NRSimulationLookup(compare_dict,valid_groups=[opts.group])
        if len(good_sim_list)< 1:
                        print(" ------- NO MATCHING SIMULATIONS FOUND ----- ")
                        import sys
                        sys.exit(0)
                        print(" Identified set of matching NR simulations ", good_sim_list)
        try:
                        print("   Attempting to pick longest simulation matching  the simulation  ")
                        MOmega0  = 1
                        good_sim = None
                        for key in good_sim_list:
                                print(key, nrwf.internal_EstimatePeakL2M2Emission[key[0]][key[1]])
                                if nrwf.internal_WaveformMetadata[key[0]][key[1]]['Momega0'] < MOmega0:
                                        good_sim = key
                                        MOmega0 = nrwf.internal_WaveformMetadata[key[0]][key[1]]['Momega0']
                                print(" Picked  ",key,  " with MOmega0 ", MOmega0, " and peak duration ", nrwf.internal_EstimatePeakL2M2Emission[key[0]][key[1]])
        except:
            good_sim  = good_sim_list[0] # pick the first one.  Note we will want to reduce /downselect the lookup process
            group = good_sim[0]
            param = good_sim[1] 
            gp_list.append( (group,param))



###
### Load in the NR simulation array metadata
###
P_list_NR = []
omega_list_NR=[]

glist = []
if opts.group:
    glist = [opts.group]
else:
    glist = nrwf.internal_ParametersAvailable.keys()

eta_range = eval(opts.eta_range)

print(opts.group, opts.param)
for group in glist:
  if not opts.param:
    for param in nrwf.internal_ParametersAvailable[group]:
     print("  -> ", group, param)
     try:
        wfP = nrwf.WaveformModeCatalog(group,param,metadata_only=True)
        wfP.P.deltaT = P.deltaT
        wfP.P.deltaF = P.deltaF
        wfP.P.fmin = P.fmin
#        wfP.P.print_params()
        # Add parameters. Because we will compare with SEOB, we need an ALIGNED waveform, so we fake it
        if wfP.P.extract_param('eta') >= eta_range[0] and wfP.P.extract_param('eta')<=eta_range[1] and (not np.isnan(wfP.P.s1z) and not np.isnan(wfP.P.s2z)):
            if (not opts.skip_overlap) and wfP.P.SoftAlignedQ():
                print(" Adding aligned sim ", group, param,  " and changing spin values, so we can compute overlaps ")
                wfP.P.approx = lalsim.GetApproximantFromString(opts.approx)  # Make approx consistent and sane
                wfP.P.m2 *= 1. - 1e-6  # Prevent failure for exactly equal!
            # Satisfy error checking condition for lal
                wfP.P.s1x = 0
                wfP.P.s2x = 0
                wfP.P.s1y = 0
                wfP.P.s2y = 0
                P_list_NR = P_list_NR + [wfP.P]
                try:
                    omega_list_NR += [nrwf.internal_WaveformMetadata[group][param]["Momega0"]]
                except:
                    omega_list_NR+=[-1]
            elif opts.skip_overlap:
                if opts.nospin_only:
                    if wfP.P.s1x ==0.0 and wfP.P.s1y==0.0 and wfP.P.s1z==0.0 and wfP.P.s2x==0.0 and wfP.P.s2y == 0.0 and wfP.P.s2z == 0.0 and not opts.eccentricity:
                        print(" Adding non-spinning simulation ", group, param)
                        P_list_NR = P_list_NR + [wfP.P]
                        try:
                            omega_list_NR += [nrwf.internal_WaveformMetadata[group][param]["Momega0"]]
                        except:
                            omega_list_NR+=[-1]
                    elif wfP.P.s1x ==0.0 and wfP.P.s1y==0.0 and wfP.P.s1z==0.0 and wfP.P.s2x==0.0 and wfP.P.s2y == 0.0 and wfP.P.s2z == 0.0 and opts.eccentricity:
                        wfP.P.print_params()
                        if not wfP.P.eccentricity==0:
                            P_list_NR = P_list_NR + [wfP.P]
                            try:
                                omega_list_NR += [nrwf.internal_WaveformMetadata[group][param]["Momega0"]]
                            except:
                                omega_list_NR+=[-1]
                elif not opts.aligned_only:
                    print(" Adding generic sim; for layout only ", group, param)
                    P_list_NR = P_list_NR + [wfP.P]
                    try:
                        omega_list_NR += [nrwf.internal_WaveformMetadata[group][param]["Momega0"]]
                    except:
                        omega_list_NR+=[-1]
                elif opts.aligned_only and  wfP.P.SoftAlignedQ():
                    print(" Adding aligned spin simulation; for layout only", group, param, " and fixing transverse spins accordingly")
                    wfP.P.s1x = 0
                    wfP.P.s2x = 0
                    wfP.P.s1y = 0
                    wfP.P.s2y = 0
                    P_list_NR = P_list_NR + [wfP.P]
                    try:
                        omega_list_NR += [nrwf.internal_WaveformMetadata[group][param]["Momega0"]]
                    except:
                        omega_list_NR+=[-1]
            else:
                print(" Skipping non-aligned simulation because overlaps active (=SEOBNRv2 comparison usually)", group, param)
#                wfP.P.print_params()
#                print nrwf.internal_WaveformMetadata[group][param]
     except:
        print(" Failed to add ", group, param)
        
  else: # target case if a single group and parameter sequence are specified
        print("Looping over list ", opts.param)
        for paramKey in opts.param:
            print(" Adding specific simulation ", opts.group, paramKey)
            if nrwf.internal_ParametersAreExpressions[group]:
                param = eval(paramKey)
            else:
                param = paramKey
        wfP = nrwf.WaveformModeCatalog(group,param,metadata_only=True)
        wfP.P.deltaT = P.deltaT
        wfP.P.deltaF = P.deltaF
        wfP.P.fmin = P.fmin
        P_list_NR = P_list_NR + [wfP.P]
        try:
            omega_list_NR += [nrwf.internal_WaveformMetadata[group][param]["Momega0"]]
        except:
            omega_list_NR+=[-1]


if len(P_list_NR)<1:
    print(" No simulations")
    sys.exit(0)


###
### Define parameter ranges to be changed
###

template_min_freq = opts.fmin
ip_min_freq = opts.fmin



###
### Lay out grid, currently CARTESIAN.   OPTIONAL: Load grid from file
###


# For now, we just extrude in these parameters
if not opts.skip_overlap:  # aligned only!  Compare to SEOB
    if opts.mc_range:
        param_names = ['mc', 'eta','s1z','s2z']
    else:
        param_names = ['mtot', 'q','s1z','s2z']
else:
    if opts.mc_range:
        param_names = ['mc', 'eta', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z']
    else:
        param_names = ['mtot', 'q', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z']
    if opts.eccentricity:
        param_names.append('eccentricity')

mass_range =[]
if opts.mc_range:
    mass_range = np.array(eval(opts.mc_range))*lal.MSUN_SI
else:
    mass_range = np.array(eval(opts.mtot_range))*lal.MSUN_SI
mass_grid =np.linspace( mass_range[0],mass_range[1],int(opts.grid_cartesian_npts))

# Loop over simulations and mass grid
grid = []
indx=-1
print("Length check", len(P_list_NR), len(omega_list_NR))
for P in P_list_NR:
    indx+=1
    Momega0_here = omega_list_NR[indx]
#    P.print_params()
    eta0 = P.extract_param('eta')   # prevent drift in mass ratio across the set if I change mc
    for M in mass_grid:
        if opts.mc_range:
            P.assign_param('mc', M)
            P.assign_param('eta', eta0)  # prevent drift in mass ratio
        else:
            P.assign_param('mtot', M)
        newline = []
        for param in param_names:
            newline = newline + [P.extract_param(param)]
#        print newline
        fstart_NR = Momega0_here/((P.extract_param('mtot')/(lal.MSUN_SI) ) *lalsimutils.MsunInSec*(np.pi))
        if   fstart_NR < P.fmin or not opts.require_fmin_above_NR_start :
            # only add mass point if a valid mass choice, given NR starting frequency.
            grid = grid+[ newline]

print(" ---- DONE WITH GRID SETUP --- ")
print(" grid points # = " ,len(grid))


grid_out, P_list = evaluate_overlap_on_grid(hfBase, param_names, grid)
if len(grid_out)==0:
    print(" No points survive....")

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

print(" ---- DONE ----")
