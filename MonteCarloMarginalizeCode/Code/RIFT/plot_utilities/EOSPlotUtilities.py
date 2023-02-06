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



def render_eos(eos, xvar='energy_density', yvar='pressure',units='cgs',npts=100,label=None,logscale=True,**kwargs):

    min_pseudo_enthalpy = 0.005
    max_pseudo_enthalpy = lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(eos)
    hvals = max_pseudo_enthalpy* 10**np.linspace( np.log10(min_pseudo_enthalpy/max_pseudo_enthalpy),  -1e4,num=npts)
    print(hvals,min_pseudo_enthalpy, max_pseudo_enthalpy)

    qry = EOSManager.QueryLS_EOS(eos)

    xvals = qry.extract_param(xvar,hvals)
    yvals = qry.extract_param(yvar,hvals)
    print(np.c_[xvals,yvals])

    if logscale:
        plt.loglog(xvals, yvals,label=label,**kwargs)
    else:
        plt.plot(xvals, yvals,label=label,**kwargs)
    return None
