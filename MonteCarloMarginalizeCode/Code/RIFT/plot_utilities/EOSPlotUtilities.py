#
# GOAL
#    Library to facilitate making various plots of equation-of-state-related quantities.


# from matplotlib import pyplot as plt
# import EOSManager
# import EOSPlotUtilities
# my_eos = EOSManager.EOSLALSimulation('SLy')
# EOSPlotUtilities.render_eos(my_eos.eos,'rest_mass_density', 'pressure')

import numpy as np
import RIFT.physics.EOSManager as EOSManager
import lalsimulation as lalsim
import lal

from matplotlib import pyplot as plt
try:
    from tqdm import tqdm
except:
    tqdm = lambda x:x

from scipy.interpolate import UnivariateSpline, PchipInterpolator


def render_eos(eos, xvar='energy_density', yvar='pressure',units='cgs',npts=100,label=None,logscale=True,verbose=False,**kwargs):

    min_pseudo_enthalpy = 0.005
    max_pseudo_enthalpy = lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(eos)
    hvals = max_pseudo_enthalpy* 10**np.linspace( np.log10(min_pseudo_enthalpy/max_pseudo_enthalpy),  -1e-4,num=npts)
    if verbose:
        print(hvals,min_pseudo_enthalpy, max_pseudo_enthalpy)

    qry = EOSManager.QueryLS_EOS(eos)

    xvals = qry.extract_param(xvar,hvals)
    yvals = qry.extract_param(yvar,hvals)
    if verbose:
        print(np.c_[xvals,yvals])

    if logscale:
        plt.loglog(xvals, yvals,label=label,**kwargs)
    else:
        plt.plot(xvals, yvals,label=label,**kwargs)
    return None


def eval_eos_list_vs(eos_list, xvar='energy_density', xgrid=None,yvar='pressure', units='cgs',use_monotonic=True):
    if xgrid is None:
        raise Exception(" EOSPlotUtilities: none passed for grid")
    n_eos = len(eos_list)
    npts = len(xgrid)
    print(npts,n_eos)
    # LARGE ALLOCATION potentially, so watch out -- usually I just need quantiles
    outvals  = np.zeros( (npts,n_eos))
    # loop and compute -- ideally parallelize! Silly to do serialy
    for indx in tqdm(np.arange(n_eos)):
        eos = eos_list[indx]
        # Pull out on grid
        min_pseudo_enthalpy = 0.005
        max_pseudo_enthalpy = lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(eos)
        hvals = max_pseudo_enthalpy* 10**np.linspace( np.log10(min_pseudo_enthalpy/max_pseudo_enthalpy),  -1e-4,num=npts)
        qry = EOSManager.QueryLS_EOS(eos)   
        xvals = qry.extract_param(xvar,hvals)
        yvals = qry.extract_param(yvar,hvals)   
        # interpolate to target grid.   Usually interpolate log x to log y.  Assume INCREASING sample array. LINEAR interpolation
        #log_ygrid = np.interp(np.log(xgrid), np.log(xvals), np.log(yvals))
        if use_monotonic:
            intp_func = PchipInterpolator(np.log(xvals),np.log(yvals))
        else:
            intp_func = UnivariateSpline(np.log(xvals),np.log(yvals))
        ygrid = intp_func(xgrid)
        outvals[:,indx] = ygrid


    return outvals


def render_eos_list_quantiles_vs(eos_list, quantile_bounds=None, xvar='energy_density', xgrid=None,yvar='pressure', units='cgs',use_monotonic=True,use_log=True,return_outvals=False,input_outvals=None,plot_kwargs={}, fill_kwargs={}):
    outvals_here=None
    if input_outvals is None:
        outvals_here  = eval_eos_list_vs(eos_list, xvar=xvar , xgrid=xgrid, yvar=yvar, units=units, use_monotonic=use_monotonic)
        print(outvals_here[:,-1])
    else:
        outvals_here = input_outvals

    if outvals_here is None:
        raise Exception(" failure generating eval list, should never happen this way")

    xgrid_here = np.array(xgrid)
    upper_vals = np.percentile(outvals_here,quantile_bounds[0]*100,1)
    lower_vals = np.percentile(outvals_here,quantile_bounds[1]*100,1)
    if use_log:
        xgrid_here = np.log10(xgrid_here)
        upper_vals = np.log10(upper_vals)
        lower_vals = np.log10(lower_vals)


    plt.plot(xgrid_here, upper_vals, **plot_kwargs)
    plt.plot(xgrid_here, lower_vals, **plot_kwargs)
    plt.fill_between(xgrid, lower_vals,upper_vals,**fill_kwargs)
    if return_outvals:
        return outvals
    return None
