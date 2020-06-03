#! /usr/bin/env python
#
#
# EXAMPLE
#    python util_NRNearestSimulationTo.py --approx SEOBNRv2 --nr-group Sequence-RIT-Generic  --verbose --srate 4096 --fname overlap-grid.xml.gz  --force-aligned-interpret
#    python util_NRNearestSimulationTo.py --fname-fisher fisher_gamma.dat --fname overlap-grid.xml.gz  --npts 3 --approx SEOBNRv2  --npts 3 
#
# WHERE TO GET FISHER MATRICIES
#   + Match calculations
# 
#     python util_ManualOverlapGrid.py --inj mdc.xml.gz --event 0 --parameter mtot --parameter-range '[50,70]' --use-fisher --seglen 8 --verbose --reset-grid-via-match
#     python util_ManualOverlapGrid.py --inj mdc.xml.gz --event 0 --parameter mtot --parameter-range '[50,70]' --use-fisher --seglen 8 --verbose --reset-grid-via-match --parameter eta --parameter-range '[0.1,0.2499]'
#
#   + Real overlap

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



import NRWaveformCatalogManager3 as nrwf
import scipy.optimize as optimize

parser = argparse.ArgumentParser()
# Parameters
parser.add_argument("--cut",default=0.001,type=float,help="Reject points if nothing is closer than this")
parser.add_argument('--nr-group', default=None,help="NR group to search (otherwise, will loop over all)")
parser.add_argument("--force-aligned-interpret",default=False,action='store_true')
parser.add_argument('--fname-fisher', default=None,help="Fisher name")
parser.add_argument('--fname', default=None,help="Name for XML file")
parser.add_argument('--fname-output', default="suggested_placement_from_posterior",help="Name for XML file for output")
parser.add_argument('--npts', default=None,type=int,help="Number of elements of the XML file to use (truncation)")
parser.add_argument("--approx",type=str,default=None,help="If supplied, the overlaps are done using this approximant, instead of a Fisher matrix")
parser.add_argument("--fisher-psd",type=str,default="SimNoisePSDaLIGOZeroDetHighPower",help="psd name (attribute in lalsimulation).  SimNoisePSDiLIGOSRD, lalsim.SimNoisePSDaLIGOZeroDetHighPower, lalsimutils.Wrapper_AdvLIGOPsd, .SimNoisePSDiLIGOSRD... ")
parser.add_argument("--psd-file",default=None,help="PSD file (assumed for hanford)")
parser.add_argument("--match-tolerance",default=3e-3,type=float)
parser.add_argument("--srate",type=int,default=16384,help="Sampling rate")
parser.add_argument("--seglen", type=int,default=16., help="Default window size for processing.")
parser.add_argument("--fmin", default=20,type=float,help="Mininmum frequency in Hz, default is 20Hz.")
parser.add_argument("--fmax", default=1700,type=float,help="Mininmum frequency in Hz, default is 20Hz.")
parser.add_argument("--verbose", action="store_true",default=False, help="Spam")
opts=  parser.parse_args()

T_window = int(opts.seglen)
srate = opts.srate
deltaT = 1./srate
###
### Key routine
###

def nr_closest_to(P,distance_function_P1P2,nr_group,mass_ref=None,return_list=False):


    mass_start_msun = 70
    if not(mass_ref is None):
        mass_start_msun = mass_ref
    result_list = []
    for param  in nrwf.internal_ParametersAvailable[nr_group]:
        acat = nrwf.WaveformModeCatalog(nr_group,param,metadata_only=True)
        P1 = acat.P
        P1.deltaF = P.deltaF
        P1.deltaT = P.deltaT
        P1.fmin =P.fmin= opts.fmin
        if opts.force_aligned_interpret:
            P.s1x = P.s1y =0
            P.s2x = P.s2y=0
            P1.s1x = P1.s1y =0
            P1.s2x = P1.s2y=0
        # print " ---> preparing for loop "
        # P.print_params()
        # P1.print_params()
        def distance_at_mass(m_msun):
            if m_msun < 20:
                return 1e5
            if opts.verbose:
                print(" Trying ", nr_group, param, m_msun)
            P1.assign_param('mtot', m_msun*lal.MSUN_SI)
            return distance_function_P1P2(P,P1)
        
        res = optimize.minimize(distance_at_mass, mass_start_msun,bounds=[(20,200)],tol=opts.match_tolerance,method='Nelder-Mead',options={'maxiter':15}) # loose tolerance because quadratic, often noisy calculations due to fmin, etc
        if opts.verbose:
            print(" ===> search result <=== ")
            P.print_params(); print("  ", nr_group, param,  res.x[0])
        val = distance_at_mass(res.x[0])
        result_list.append( (param,res.x[0],val))
        mass_start_msun = res.x  # use guess for next point, to speed up things
        
    # Full output: param, M, value for the whole group
    if opts.verbose:
        print(result_list)
    if return_list:
        return result_list

    # short output
    
    xvals = np.array([x for a,b,x in result_list])
    indx = np.argmin(xvals)
    return [result_list[indx][0],result_list[indx][-1]]
    


def make_distance_for_fisher(mtx, param_names):
    def my_distance(P1,P2):
        # extract parameters for P1
        vec1 = np.zeros(len(param_names))
        for indx in np.arange(len(param_names)):
            vec1[indx] = P1.extract_param(param_names[indx])
            if param_names[indx] in ['mc', 'mtot','m1', 'm2']:
                vec1[indx] /= lal.MSUN_SI

        # extract parameters for P2
        vec2 = np.zeros(len(param_names))
        for indx in np.arange(len(param_names)):
            vec2[indx] = P2.extract_param(param_names[indx])
            if param_names[indx] in ['mc', 'mtot','m1', 'm2']:
                vec2[indx] /= lal.MSUN_SI


        deltaV = vec1-vec2
        return np.dot(deltaV, np.dot(mtx,deltaV))

    return my_distance


dist_here = None
if opts.fname_fisher:
    ###
    ### Load in Fisher. Create default distance function (WARNING: needs parameter labels to be appended!)
    ###
    datFisher = np.genfromtxt(opts.fname_fisher,names=True)
    print(" Fisher matrix ", datFisher, " with shape ",  datFisher.shape , " and parameters ", datFisher.dtype.names)

    # parameter names
    param_names = list(datFisher.dtype.names)

    # FAIL if Fisher matrix does not include a mass-like dimension
    if not ( 'mc' in param_names  or 'mtot' in param_names or 'm1' in param_names):
        print(" You must have a mass parameter in your Fisher matrix, otherwise you cannot optimize in mass ")
        sys.exit(0)

    # parameter matrix
    mtx =[]
    if len(param_names) > 1:
        mtx = np.zeros( (len(param_names),len(param_names)) )
        for indx1 in np.arange(len(param_names)):
            for indx2 in np.arange(len(param_names)):
                mtx = datFisher[indx1][indx2]
    else:
        mtx = np.zeros( (1,1))
        mtx[0,0] = datFisher[param_names[0]] #datFisher[0]

    print(" Fisher in matrix form is ", mtx)

    # make distance function
    dist_here = make_distance_for_fisher(mtx,param_names)
elif opts.approx:

    if not opts.psd_file:
    #eff_fisher_psd = eval(opts.fisher_psd)
        eff_fisher_psd = getattr(lalsim, opts.fisher_psd)   # --fisher-psd SimNoisePSDaLIGOZeroDetHighPower   now
        analyticPSD_Q=True
    else:
        print(" Importing PSD file ", opts.psd_file)
        eff_fisher_psd = lalsimutils.load_resample_and_clean_psd(opts.psd_file, 'H1', 1./opts.seglen)
        analyticPSD_Q = False

    ###
    ###  Create the  inner product function, etc needed (distance =match)
    ###
    P=lalsimutils.ChooseWaveformParams()
    P.m1 = P.m2 = 50*lal.MSUN_SI
    P.approx = lalsim.GetApproximantFromString(opts.approx)
    P.deltaT = 1./srate
    P.deltaF = 1./opts.seglen
    hfBase = lalsimutils.complex_hoff(P)
    IP = lalsimutils.CreateCompatibleComplexOverlap(hfBase,analyticPSD_Q=analyticPSD_Q,psd=eff_fisher_psd,fMax=opts.fmax,interpolate_max=True)
    def my_distance(P1,P2):
        global IP
        global opts
        P1.approx = P2.approx = lalsim.GetApproximantFromString(opts.approx);
        P1.fmin = P2.fmin = opts.fmin
        P1.deltaF = P2.deltaF = 1./T_window

        # if opts.verbose:
        #     print " ---> Inside distance function < "
        #     P1.print_params()
        #     P2.print_params()
        dist = 1e5
        try:
            hF1 = lalsimutils.complex_hoff(P1)
            hF2 = lalsimutils.complex_hoff(P2)
            rho1 = IP.norm(hF1)
            rho2 = IP.norm(hF2)
            dist = 1- np.abs( IP.ip(hF1,hF2)/rho1/rho2)
        except:
            print(" Distance evaluation failure ")
        if np.isnan(dist):
            return 1e5
        return dist

    dist_here = my_distance

    # print " --->  Testing <--- "
    # print my_distance(P,P)


    # print " --->  Testing <--- "
    # P2 = P.manual_copy()
    # P2.assign_param('mtot', 70*lal.MSUN_SI)
    # print my_distance(P,P2)


###
### Load in xml
###
if not opts.fname:
    print(" No data provided ")
    sys.exit(0)
P_list = lalsimutils.xml_to_ChooseWaveformParams_array(opts.fname)
if not (opts.npts==None) and opts.npts>0 and opts.npts < len(P_list):
    print(" Truncating list of sample points to examine to size ", opts.npts)
    P_list = P_list[:opts.npts]


###
### Loop over XML
###
if opts.nr_group:
    glist = [opts.nr_group]
else:
    glist = nrwf.internal_ParametersAvailable.keys()
best_fits =[]
single_best_fits = []
P_list_retain = []
dist_list_retain =[]
for P in P_list:
    if P.m1/lal.MSUN_SI < 1 or P.m2 < lal.MSUN_SI:
        # Skip crazy things that cause problems
        continue
    P.fmin = opts.fmin # we change this. FOR PRECESSING SYSTEMS THAT CAN CHANGE THE MEANING, BE CAREFUL
    if opts.verbose:
        print(" Trying next point in XML")
        P.print_params()
    P.approx = lalsim.GetApproximantFromString(opts.approx)
    P.fmin = opts.fmin
    P.deltaT = 1./srate
    P.deltaF = 1./T_window

    # Loop over everything, and find best point
    best_fits_here = []
    closest_so_far_d = 1e10
    closest_so_far = ()
    for group in glist:
#        if opts.verbose:
#            print " Checking ", group
        res = nr_closest_to(P, dist_here,group)  # pair of values returned
        if res[1] < closest_so_far_d:
            closest_so_far  = (group,res[0])
            closest_so_far_d = res[1]
            if opts.verbose:
                print(" NEW CLOSEST SIMULATION FOR THIS POSTERIOR POINT", closest_so_far, closest_so_far_d)

    if closest_so_far_d> opts.cut:
        P_list_retain.append(P)
        dist_list_retain.append(closest_so_far_d)
    else:
        print(" Skipping point shown below as too close to simulations because the closest has distance", closest_so_far_d, " which is less than ", opts.cut)
        P.print_params()
        

lalsimutils.ChooseWaveformParams_array_to_xml(P_list_retain, fname=opts.fname_output, fref=P.fref)
np.savetxt(opts.fname_output+"_distances.dat", np.array([dist_list_retain]).T)

