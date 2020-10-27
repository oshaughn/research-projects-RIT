#
# spokes.py
# $Id$
#

# GOAL:
#    - library to manage two arrays: *results* and *XML grids*
#    - provides
#         - i/o
#         - consolidating (=joining entries from duplicate masses)
#         - fitting
#         - maximum finding
#         - grid refinement
#
#   - should NOT need NR lookup: works entirely with XML and reported ILE output

import numpy as np
import lal
import RIFT.lalsimutils as lalsimutils
from . import weight_simulations

rosDebug=False

default_deltaLogL = 10 # 10  # Standard refinement interval in lnL.
n_refine_min =3           # minimum number of points needed for refinement


##
## Spoke fitting
##
def FitSpokeNearPeak(xvals,lnLVals,deltaLogL=default_deltaLogL,refinement_scale_min=0.2,**kwargs):
    # Find the peak
    lnLmax = np.max(lnLVals)
    lnLmin = np.min(lnLVals)
    xmax =np.max(xvals)
    xmin = np.min(xvals)

    # Find x values "sufficiently near" the peak (assume ONE set, for now).  Use top 30%, PLUS anything within 5
    pts_vals = np.array([xvals,np.real(lnLVals)]).T
    pts_sorted = pts_vals[pts_vals[:,1].argsort()] #  np.sort(pts_vals.T,axis=-1)   # usual sort was NOT reliable, oddly!!
#    print " Sorted array ", pts_sorted
    indx_max_base = int(len(pts_sorted)*refinement_scale_min)   # maximum index, based on fraction to keep
    indx_max_delta = np.sum(1 for x in lnLVals if  x > lnLmax-deltaLogL) # len( [x for x,lnL in pts_sorted if lnL> lnLMax-deltaLogL])  # maximum index based on deltaLogL
#    print indx_max_base, indx_max_delta, np.max([indx_max_base,indx_max_delta])
    indx_max = np.max([indx_max_base,indx_max_delta])
    pts_sorted_reduced = pts_sorted[-indx_max:]  # Reduce the number. Minimum length is 3

    xmin_here = np.min(pts_sorted_reduced[:,0])
    xmax_here = np.max(pts_sorted_reduced[:,0])
    z = np.polyfit(pts_sorted_reduced[:,0],pts_sorted_reduced[:,1],2)   # note z[0] is coefficient of x^2, z[2] is coefficient of constant
    if z[0]<0:
        fn = np.poly1d(z)
        def fnReturn(x):
            return np.piecewise(x, [np.logical_and(x>xmin_here, x<xmax_here),np.logical_not(np.logical_and(x>xmin_here , x<xmax_here))] ,[ fn, 0])
        return fnReturn

    if z[0]>0:
        z = np.polyfit(pts_sorted_reduced[:,0],pts_sorted_reduced[:,1],1) 
        fn = np.poly1d(z)
        def fnReturn(x):
            return np.piecewise(x, [np.logical_and(x>xmin_here, x<xmax_here),np.logical_not(np.logical_and(x>xmin_here , x<xmax_here))] ,[ fn, 0])
        return fnReturn
    
    return None

##
## Refinement
##
def Refine(xvals,lnLVals,xmin=None,deltaLogL=default_deltaLogL,npts=10,refinement_scale_min=0.2,**kwargs):
    """
    Refine(xvals,lnLVals) takes a 1d grid of lnL(x) and returns a simple grid refinement guess.
    If the maximum is at or very near an edge (1 or two samples).
    Refinement by default fits to top 30%
    """
    return_code = None
    len_min = 5
    if len(xvals) < len_min:
#        if rosDebug:
        print(" FAILED REFINEMENT: Too few points for safety", len(xvals))
        return 'fail', None
    if rosDebug and len(xvals)*refinement_scale_min > npts:
        print(" WARNING: Refinement will not increase point density")


    # Find the peak
    # warning: x values are NOT necessarily sorted
    lnLmax = np.max(lnLVals)
    lnLmin = np.min(lnLVals)
    xmax =np.max(xvals)
    xmin = np.min(xvals)
    indx_first = np.argmin(xvals)
    indx_last = np.argmin(xvals)


    # If we are at an edge, extend, with the same resolution as the CLOSEST FEW POINTS
    if lnLVals[indx_first]==lnLmax:
        return_code = 'extended'
        dx = np.std( xvals[-len_min:])/len_min   # reduces chance of numerical catastrophe. Note this is a  large stride
        xvals_new = xmin - np.arange(npts)*dx
        return 'extended',xvals_new
    if lnLVals[indx_last]==lnLmax:
        return_code = 'extended'
        dx = np.std( xvals[-len_min:])/len_min   # reduces chance of numerical catastrophe. Note this is a large stride
        xvals_new = xmax + np.arange(npts)*dx
        return 'extended',xvals_new

    # Find x values "sufficiently near" the peak (assume ONE set, for now).  Use top 30%, PLUS anything within 5
    pts_vals = np.array([xvals,np.real(lnLVals)]).T
    pts_sorted = pts_vals[pts_vals[:,1].argsort()] #  np.sort(pts_vals.T,axis=-1)   # usual sort was NOT reliable, oddly!!
#    print " Sorted array ", pts_sorted
    indx_max_base = int(len(pts_sorted)*refinement_scale_min)   # maximum index, based on fraction to keep
    indx_max_delta = np.sum(1 for x in lnLVals if  x > lnLmax-deltaLogL) # len( [x for x,lnL in pts_sorted if lnL> lnLMax-deltaLogL])  # maximum index based on deltaLogL
    #print indx_max_base, indx_max_delta, np.max([indx_max_base,indx_max_delta])
    indx_max = np.max([n_refine_min,indx_max_base,indx_max_delta])
    pts_sorted_reduced = pts_sorted[-indx_max:]  # Reduce the number. Minimum length is 3

    # Add on elements which are immediately outside the interval, on either side
    #   - important in case the peak is asymmetrically sampled (i.e., only one side of a hill is well-explored)
    xc_min = np.min(pts_sorted_reduced[:,0])
    xc_max = np.min(pts_sorted_reduced[:,0])
    tmp =  pts_sorted[pts_sorted[:,0]<  xc_min ]
    if len(tmp) >0:
        best_before = tmp[-1]
        pts_sorted_reduced = np.append(pts_sorted_reduced,[best_before],axis=0)
    tmp = pts_sorted[pts_sorted[:,0]>  xc_max ]
    if len(tmp)>0:
        best_after = tmp[-1]
        pts_sorted_reduced = np.append(pts_sorted_reduced,[best_after],axis=0)

    # Fail if length too small (should not happen)
    if len(pts_sorted_reduced) < n_refine_min: 
        print("PROBLEM: Reduced length ")  # should never happen
        deltaLogLNew = (n_refine_min+1)*1.0/len(xvals) * np.max(lnLVals)  # top few elements are autoselected
        return Refine(xvals,lnLVals, xmin=xmin,deltaLogL=deltaLogLNew,npts=npts,refinement_scale_min=refinement_scale_min)
#        return 'fail', None
    # Return no refinement possible if the array is long (i.e., lnLVals is nearly constant)
#    if len(pts_sorted_reduced) > 0.8*len(xvals):
#        return 'no-refine',None

    # Refine
    # OPTION 1: Quadratic fit refinement
    #    - quadratic fit replacement will not intelligently refine if we are VERY CLOSE to a grid boundary...but it could
    #    - 'refined' grid may be LESS DENSE THAN ORIGINAL GRID
    z = np.polyfit(pts_sorted_reduced[:,0],pts_sorted_reduced[:,1],2)   # note z[0] is coefficient of x^2, z[2] is coefficient of constant
    if z[0]<0:
        fn = np.poly1d(z)
        predicted_xmax = -z[1]/(2*z[0] )
        xmin_here = np.min(pts_sorted_reduced[:,0])
        xmax_here = np.max(pts_sorted_reduced[:,0])
        lnLmin_here =np.min(pts_sorted_reduced[:,1])
        z2 = z - np.array([0,0,np.min([lnLmin_here,lnLmax-deltaLogL])])   # predicted roots, based on 2* delta log L (safety)
        predicted_roots = np.sort(np.roots(z2))
        dx = np.std( xvals[-len_min:])/npts   # reduces chance of numerical catastrophe
        if predicted_roots[0] <xmin_here-len_min*dx:
            predicted_roots[0] = xmin_here
        if predicted_roots[1] > xmax_here+len_min*dx:
            predicted_roots[1] = xmax_here
        if (not (xmin == None)) and predicted_roots[0]<xmin:
            predicted_roots[0] = xmin   # prevent negative total masses
        xvals_new = np.linspace(predicted_roots[0],predicted_roots[1], npts)
        return 'refined', xvals_new
    # OPTION 2: Literal refinement based on surviving points
    xmin_here = np.min(pts_sorted_reduced[:,0])
    xmax_here = np.max(pts_sorted_reduced[:,0])
    return 'refined', np.linspace(xmin_here, xmax_here, npts)


def ChooseWaveformParams_to_spoke_label(P,digits=4):
    m1 = P.m1/lal.MSUN_SI
    m2 = P.m2/lal.MSUN_SI
    s1x = P.s1x
    s1y = P.s1y
    s1z = P.s1z
    s2x = P.s2x
    s2y = P.s2y
    s2z = P.s2z

    line = np.array([m2/m1,s1x,s1y,s1z,s2x,s2y,s2z ])
    return np.around(line, decimals=digits)

def Line_to_spoke_label(line,digits=4):
    m1,m2,s1x,s1y, s1z, s2x,s2y,s2z = line[1:9]
    line = np.array([m2/m1,s1x,s1y,s1z,s2x,s2y,s2z ])
    return np.around(line, decimals=digits)
    
def Line_to_spoke_entry(line,digits=4):
    m1 = line[1]
    m2 = line[2]
    lnL = line[9]
    deltaLogL = line[10]
    return np.around(np.array([m1+m2,lnL,deltaLogL]), decimals=digits)

def ChooseWaveformParams_to_spoke_mass(P,digits=4):
    m1 = P.m1
    m2 = P.m2
    return np.around((m1+m2)/lal.MSUN_SI, decimals=4)


def LoadSpokeDAT(fname):
    # load the *.dat concatenated file 
    dat = np.loadtxt(fname)

    # group by spokes
    sdHere = {}
    for line in dat:
#        spoke_id = str([round(elem, 3) for elem in Line_to_spoke_label(line)])
        spoke_id = str(Line_to_spoke_label(line))
        spoke_contents = Line_to_spoke_entry(line)
        if spoke_id in sdHere:
            sdHere[spoke_id].append(spoke_contents)
        else:
            sdHere[spoke_id] = [spoke_contents]
    # return 
    return sdHere

def LoadSpokeXML(fname):
    # load the xml file
    P_list = lalsimutils.xml_to_ChooseWaveformParams_array(fname)

    # group by spokes
    sdHere = {}
    for P in P_list:
#        spoke_id = str( [round(elem, 3) for elem in ChooseWaveformParams_to_spoke_label(P)])
        spoke_id = str(ChooseWaveformParams_to_spoke_label(P))
        if sdHere.has_key(spoke_id):
            sdHere[spoke_id].append(P)
        else:
            sdHere[spoke_id] = [P]
    # return 
    return sdHere



##
## Clean spoke entries: remove duplicate masses
##
def CleanSpokeEntries(spoke_entries,digits=4):
    data_at_intrinsic = {}
    # Group entries by their total mass (data in spoke: M, lnL, deltalnL
    for line in spoke_entries:
        mtot, lnL, sigmaOverlnL = line
        if data_at_intrinsic.has_key(mtot):
            data_at_intrinsic[mtot].append( [lnL,sigmaOverlnL])
        else:
            data_at_intrinsic[mtot] = [[lnL,sigmaOverlnL]]

    spoke_entries_out = []
    for key in data_at_intrinsic:
        lnL, sigmaOverL =   np.transpose(data_at_intrinsic[key])
        lnLmax = np.max(lnL)
        sigma = sigmaOverL*np.exp(lnL-lnLmax)  # remove overall Lmax factor, which factors out from the weights constructed from \sigma
        wts = weight_simulations.AverageSimulationWeights(None, None,sigma)   
        lnLmeanMinusLmax = np.log(np.sum(np.exp(lnL - lnLmax)*wts))
        sigmaNetOverL = (np.sqrt(1./np.sum(1./sigma/sigma)))/np.exp(lnLmeanMinusLmax)
        spoke_entries_out.append([key,lnLmeanMinusLmax+lnLmax, sigmaNetOverL])

    return np.around(np.array(spoke_entries_out),decimals=digits)

