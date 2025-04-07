#! /usr/bin/env python
# Objective
#    call bilby  make_pp_plot on RIFT results : use same formatting, error bars, and overall p-value
# Compare to
#     https://git.ligo.org/pe/O4/bilby_o4_review/-/blob/main/pp-test-templates/make_pp_plot.py


import glob
import json
import sys
import argparse

import lal
import lalsimulation as lalsim

from collections import namedtuple

import bilby
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import RIFT.lalsimutils as lalsimutils
from RIFT.misc.samples_utils import add_field

from itertools import product
import scipy.stats
from bilby.core.utils import logger
from bilby.core.utils import safe_save_figure

param_names = ['mc','q', 'ra', 'dec','incl', 'psi', 'phiorb','distance']
remap_export_to_P = {'mc':'mc', 'q':'q','ra':'phi', 'dec':'theta', 'distance':'dist','phiorb':'phiref','psi':'psi','incl':'incl'}
for i in ['1','2']:
    for j in ['x','y','z']:
        str_in = 's'+i+j
        str_out = 'a'+i+j
        remap_export_to_P[str_out]=str_in
# Omit phases, they are too hard to get right in a quick pass
param_names_alt = ['mc','q','chi1', 'chi2', 'theta1','theta2', 'thetaJN','phiorb', 'distance','psi'] # 'psiJ', 'phiJL', 'phi12'

for name in ['chi1', 'chi2', 'theta1','theta2', 'thetaJN', 'psiJ', 'phiJL', 'phi12']:
   remap_export_to_P[name]=name


fref=20

def load_result(fname,alt_coords=False):
#    print(fname)
    samples = np.genfromtxt(fname,names=True)
    samples['psi'] = np.mod(samples['psi'],np.pi) # deal with range issue
    # Coordinate conversion
    if alt_coords:
        samples = add_field(samples, [('chi1',float),('chi2',float), ('theta1',float), ('theta2',float),('thetaJN',float),('phiJL',float),('phi12',float),('psiJ',float)])
        for indx in np.arange(len(samples['psi'])):
            theta_jn, phi_JL, theta_1, theta_2, phi_12, a_1, a_2 = lalsim.SimInspiralTransformPrecessingWvf2PE( samples['incl'][indx], samples['a1x'][indx], samples['a1y'][indx], samples['a1z'][indx], samples['a2x'][indx], samples['a2y'][indx], samples['a2z'][indx], samples['m1'][indx], samples['m2'][indx],20, samples['phiorb'][indx])
            samples['thetaJN'][indx] = theta_jn
            samples['phiJL'][indx] = np.mod(phi_JL, 2*np.pi)
            samples['theta1'][indx]= theta_1
            samples['theta2'][indx] = theta_2
            samples['chi1'][indx] = a_1
            samples['chi2'][indx] = a_2
            samples['phi12'][indx] = np.mod(phi_12, 2*np.pi)


    # drop nonessential fields
    
    # Convert to dataframe
    samples_df = pd.DataFrame(samples)
    bilby_result = bilby.core.result.Result(label=fname,posterior=samples_df)
    return bilby_result

def load_priors(basedir):
    priors = bilby.gw.prior.BBHPriorDict(f"/home/colm.talbot/O4/pe_review/prior-files/{basedir}.prior")

    del priors["mass_1"], priors["mass_2"]
    for key in list(priors.keys()):
        if "time" in key:
            del priors[key]
        elif isinstance(priors[key], (float, int, bilby.core.prior.DeltaFunction)):
            del priors[key]

    if "a_1" in priors:
        priors["a_1"].latex_label = "$a_{1}$"
        priors["a_2"].latex_label = "$a_{2}$"
    if "lambda_1" in list(priors.keys()):
        priors["lambda_1"].latex_label = "$\\lambda_{1}$"
    if "lambda_2" in priors:
        priors["lambda_2"].latex_label = "$\\lambda_{2}$"
    return priors

def make_pp_plot(results, filename=None, save=True, confidence_interval=[0.68, 0.95, 0.997],
                 lines=None, legend_fontsize='x-small', keys=None, title=True,
                 confidence_interval_alpha=0.1, weight_list=None,
                 **kwargs):
    """
    Make a P-P plot for a set of runs with injected signals.

    Parameters
    ==========
    results: list
        A list of Result objects, each of these should have injected_parameters
    filename: str, optional
        The name of the file to save, the default is "outdir/pp.png"
    save: bool, optional
        Whether to save the file, default=True
    confidence_interval: (float, list), optional
        The confidence interval to be plotted, defaulting to 1-2-3 sigma
    lines: list
        If given, a list of matplotlib line formats to use, must be greater
        than the number of parameters.
    legend_fontsize: float
        The font size for the legend
    keys: list
        A list of keys to use, if None defaults to search_parameter_keys
    title: bool
        Whether to add the number of results and total p-value as a plot title
    confidence_interval_alpha: float, list, optional
        The transparency for the background condifence interval
    weight_list: list, optional
        List of the weight arrays for each set of posterior samples.
    kwargs:
        Additional kwargs to pass to matplotlib.pyplot.plot

    Returns
    =======
    fig, pvals:
        matplotlib figure and a NamedTuple with attributes `combined_pvalue`,
        `pvalues`, and `names`.
    """
    import matplotlib.pyplot as plt

    if keys is None:
        keys = results[0].search_parameter_keys

    if weight_list is None:
        weight_list = [None] * len(results)

    credible_levels = list()
    for i, result in enumerate(results):
        credible_levels.append(
            result.get_all_injection_credible_levels(keys, weights=weight_list[i])
        )
    credible_levels = pd.DataFrame(credible_levels)

    if lines is None:
        colors = ["C{}".format(i) for i in range(8)]
        linestyles = ["-", "--", ":"]
        lines = ["{}{}".format(a, b) for a, b in product(linestyles, colors)]
    if len(lines) < len(credible_levels.keys()):
        raise ValueError("Larger number of parameters than unique linestyles")

    x_values = np.linspace(0, 1, 1001)

    N = len(credible_levels)
    fig, ax = plt.subplots()

    if isinstance(confidence_interval, float):
        confidence_interval = [confidence_interval]
    if isinstance(confidence_interval_alpha, float):
        confidence_interval_alpha = [confidence_interval_alpha] * len(confidence_interval)
    elif len(confidence_interval_alpha) != len(confidence_interval):
        raise ValueError(
            "confidence_interval_alpha must have the same length as confidence_interval")

    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1. - ci) / 2.
        lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(x_values, lower, upper, alpha=alpha, color='k')

    pvalues = []
    logger.info("Key: KS-test p-value")
    for ii, key in enumerate(credible_levels):
        pp = np.array([sum(credible_levels[key].values < xx) /
                       len(credible_levels) for xx in x_values])
        pvalue = scipy.stats.kstest(credible_levels[key], 'uniform').pvalue
        pvalues.append(pvalue)
        logger.info("{}: {}".format(key, pvalue))

        #try:
        #    name = results[0].priors[key].latex_label
        #except AttributeError:
        name = key
        label = "{} ({:2.3f})".format(name, pvalue)
        plt.plot(x_values, pp, lines[ii], label=label, **kwargs)

    Pvals = namedtuple('pvals', ['combined_pvalue', 'pvalues', 'names'])
    pvals = Pvals(combined_pvalue=scipy.stats.combine_pvalues(pvalues)[1],
                  pvalues=pvalues,
                  names=list(credible_levels.keys()))
    logger.info(
        "Combined p-value: {}".format(pvals.combined_pvalue))

    if title:
        ax.set_title("N={}, p-value={:2.4f}".format(
            len(results), pvals.combined_pvalue))
    ax.set_xlabel("C.I.")
    ax.set_ylabel("Fraction of events in C.I.")
    ax.legend(handlelength=2, labelspacing=0.25, fontsize=legend_fontsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if save:
        if filename is None:
            filename = 'outdir/pp.png'
        safe_save_figure(fig=fig, filename=filename, dpi=500)

    return fig, pvals


parser = argparse.ArgumentParser()
parser.add_argument("--base-dir",default='my_set_16s',type=str)
parser.add_argument("--alt-coords",action='store_true')
opts = parser.parse_args()


# Can't use external code
#   - different param names
#   - coord system change too annoying
#base_name_short = opts.base_dir.split('_')[-1].replace('/','')
#priors = load_priors(base_name_short)
#print(priors)

if opts.alt_coords:
    param_names = param_names_alt

results = []
fname_list = glob.glob(opts.base_dir+'/analysis_*/rundir/extrinsic_posterior_samples.dat')
#print(fname_list)
indx_name =0
for fname in fname_list:
    indx_event = int(fname.split('/')[-3].replace('analysis_event_',''))
    print(fname,indx_event)
    # load in data
    result = load_result(fname,alt_coords=opts.alt_coords)
    #result.priors = priors
    
    # Add injection
    fname_inj = opts.base_dir+"/mdc.xml.gz"#fname.replace('rundir/extrinsic_posterior_samples.dat', 'mdc.xml.gz')
    #print(fname_inj)
    P = lalsimutils.xml_to_ChooseWaveformParams_array(fname_inj)[indx_event]
    P.fref = fref
    inj_vals = {}
    for name in param_names:
        scale = 1
        if name =='mc':
            scale = lal.MSUN_SI
        if name =='distance':
            scale =lal.PC_SI*1e6
        if name != 'phi12':
            inj_vals[name ] = P.extract_param(remap_export_to_P[name])/scale
        else:
            inj_vals[name] = P.extract_param('phi2') - P.extract_param('phi1')
    #inj_vals = [P.extract_param(remap_export_to_P[x]) for x in param_names]

    # add parameters to file
    result.injection_parameters = inj_vals

    #  augment list
    results.append(result)

#bilby.core.result.
make_pp_plot(results,"output_pp_bilby.pdf",keys=param_names)
