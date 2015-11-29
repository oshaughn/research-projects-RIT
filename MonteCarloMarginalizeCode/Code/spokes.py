#
# spokes.py
# $Id$
#

# GOAL:
#    - library to manage two arrays: *results* and *XML grids*
#    - provides
#         - i/o
#         - fitting
#         - maximum finding
#         - grid refinement

import numpy as np



default_deltaLogL = 5  # Standard de

def Refine(xvals,lnLVals,deltaLogL=default_deltaLogL,npts=10,refinement_scale_min=0.2,**kwargs):
    """
    Refine(xvals,lnLVals) takes a 1d grid of lnL(x) and returns a simple grid refinement guess.
    If the maximum is at or very near an edge (1 or two samples).
    Refinement by default fits to top 30%
    """
    return_code = None
    len_min = 10
    if len(xvals) < len_min:
        print " FAILED REFINEMENT: Too few points for safety"
        return 'fail', None
    if len(xvals)*refinement_scale_min > npts:
        print " WARNING: Refinement will not increase point density"


    # Find the peak
    lnLmax = np.max(lnLVals)
    lnLmin = np.min(lnLVals)
    xmax =np.max(xvals)
    xmin = np.min(xvals)

    # If we are at an edge, extend, with the same resolution as the CLOSEST FEW POINTS
    if lnLVals[0]==lnLmax:
        return_code = 'extended'
        dx = np.std( xvals[-len_min:])   # reduces chance of numerical catastrophe. Note this is a  large stride
        xvals_new = xvals[0] - np.arange(npts)*dx
        return 'extended',xvals_new
    if lnLVals[-1]==lnLmax:
        return_code = 'extended'
        dx = np.std( xvals[-len_min:])   # reduces chance of numerical catastrophe. Note this is a large stride
        xvals_new = xvals[-1] + np.arange(npts)*dx
        return 'extended',xvals_new

    # Find x values "sufficiently near" the peak (assume ONE set, for now).  Use top 30%, PLUS anything within 5
    pts_sorted = np.sort(np.array([xvals,lnLVals]).T)
    indx_max_base = int(len(pts_sorted)*refinement_scale_min)   # maximum index, based on fraction to keep
    indx_max_delta = np.sum(1 for x in lnLVals if  x > lnLmax-deltaLogL) # len( [x for x,lnL in pts_sorted if lnL> lnLMax-deltaLogL])  # maximum index based on deltaLogL
#    print indx_max_base, indx_max_delta, np.max([indx_max_base,indx_max_delta])
    indx_max = np.max([indx_max_base,indx_max_delta])
    pts_sorted_reduced = pts_sorted[-indx_max:]  # Reduce the number. Minimum length is 3
    # Fail if length too small (should not happen)
    if len(pts_sorted_reduced) < 3: 
        print "FAILURE: Reduced length "
        return 'fail', None
    # Return no refinement possible if the array is long.
    if len(pts_sorted_reduced) > 0.8*len(xvals):
        return 'no-refine',None

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
        xvals_new = np.linspace(predicted_roots[0],predicted_roots[1], npts)
        return 'refined', xvals_new
    # OPTION 2: Literal refinement based on surviving points
    xmin_here = np.min(pts_sorted_reduced[:,0])
    xmax_here = np.max(pts_sorted_reduced[:,0])
    return 'refined', np.linspace(xmin_here, xmax_here, npts)

