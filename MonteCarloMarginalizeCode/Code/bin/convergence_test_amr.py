#! /usr/bin/env python



import os
import sys
import glob
import argparse
#import json
#import bisect
#import re
#from collections import defaultdict
#from argparse import ArgumentParser
#from copy import copy

from matplotlib import pyplot as plt

import scipy.stats

from ligo.lw import utils, ligolw, lsctables # , ilwd
lsctables.use_in(ligolw.LIGOLWContentHandler)
from ligo.lw.utils import process

import h5py
import numpy
np = numpy
from sklearn.neighbors import BallTree


import lal
import lalsimulation
from RIFT.misc import amrlib
from RIFT import lalsimutils

from scipy.special import logsumexp


parser = argparse.ArgumentParser()
parser.add_argument("--fname-hdf",default="intrinsic_grid_all_iterations.hdf")
parser.add_argument("--working-directory",default=".")
parser.add_argument("--test-ks",action='store_true')
opts=  parser.parse_args()

# ass in util_AMRGrid.py.  Should be in 'amrlib'
def get_evidence_grid(points, res_pts, intr_prms, exact=False):
    """
    Associate the "z-axis" value (evidence, overlap, etc...) res_pts with its
    corresponding point in the template bank (points). If exact is True, then
    the poit must exactly match the point in the bank.
    """
    grid_tree = BallTree(selected)
    grid_idx = []
    # Reorder the grid points to match their weight indices
    for res in res_pts:
        dist, idx = grid_tree.query(numpy.atleast_2d(res), k=1)
        # Stupid floating point inexactitude...
        #print res, selected[idx[0][0]]
        #assert numpy.allclose(res, selected[idx[0][0]])
        grid_idx.append(idx[0][0])
    return points[grid_idx]

def load_composite(base, indx):
    fname_composite = base + "/consolidated_{}.composite".format(indx)
    use_composite=True
    tempfile = "temp_convert_file"
    cmd = "convert_output_format_allnet2xml --fname {} --fname-output-samples temp_convert_file ".format(fname_composite)
    os.system(cmd)
    fname_composite = "temp_convert_file.xml.gz"
    results = []
    xmldoc = utils.load_filename(fname_composite, contenthandler=ligolw.LIGOLWContentHandler)
    results.extend(lsctables.SimInspiralTable.get_table(xmldoc))
    # Space grid
    res_pts = numpy.array([tuple(getattr(t, a) for a in intr_prms) for t in results]) # can fill spin components if present

    # log Evidence values
#    maxlnevid = numpy.max([s.alpha3 for s in results])
#    total_evid = numpy.exp([s.alpha3 - maxlnevid for s in results]).sum()
#    for res in results:
#        res.alpha3 = numpy.exp(res.alpha3 - maxlnevid)/total_evid  # (a) convert lnL -> exp(lnL), but also rescale
    results = numpy.array([res.alpha3 for res in results])  # convert to scalar array

    return res_pts, results   # results are log evidences, with correct normalization


# Load AMR


# currently hardcoded here, will make more flexible
intr_prms = ["mchirp","eta", "spin1z", "spin2z"]




# Load results file
#  - in principle, this has everything we need, IF results are already gridded.
# fname_composite = "all.net"
# if '.composite' in fname_composite  or '.net' in fname_composite:
#         use_composite=True
#         tempfile = "temp_convert_file"
#         cmd = "convert_output_format_allnet2xml --fname {} --fname-output-samples temp_convert_file ".format(fname_composite)
#         os.system(cmd)
#         fname_composite = "temp_convert_file.xml.gz"
# results = []
# xmldoc = utils.load_filename(fname_composite, contenthandler=ligolw.LIGOLWContentHandler)
# results.extend(lsctables.SimInspiralTable.get_table(xmldoc))

fname_hdf = opts.fname_hdf # "intrinsic_grid_all_iterations.hdf"


# count the number of composite files
fnames_composite = list(glob.glob(opts.working_directory+"/consolidated_*.composite"))

log_Zl = []
results_saved=[]
for indx in range(0,len(fnames_composite)):
    # Note this only works with a *complete* sample at every level .. if I start truncating, it fails
    # Evidence values
    res_pts, results = load_composite(opts.working_directory,indx)

    dim = 4  # default; effective dimension count
#    print(res_pts[:10], results[:10])

    # this is dumb, we can just get the spacing from top level
    amr_current_level = indx
    (prev_cells, spacing), level, _ = amrlib.load_grid_level(fname_hdf, amr_current_level, True)


    # Perform Riemann sum
    #   - assumes we are all ok
    #    print(spacing)
    dv = np.prod(spacing)
    log_Zl_here = logsumexp(results) + np.log(dv)
    log_Zl.append(log_Zl_here)

    if not(opts.test_ks):
        print(indx, log_Zl_here)
    else:
        # 1d analysis
        #  - nearest neihbor spacing
        vals_sorted = results[np.argsort(results)]
        mean_diff = np.mean(np.diff(vals_sorted))  # which is just of course delta X/N, performing the sum
    
        lnLmax = np.max(vals_sorted)
        
        def my_cdf(lnL):
            return 1.-scipy.stats.chi2.cdf((lnLmax-lnL)*2,df=dim)

        val_ks = scipy.stats.kstest(vals_sorted, my_cdf)

        plt.clf()
        plt.scatter( vals_sorted , np.arange(len(vals_sorted))/(1.0*len(vals_sorted)))
        plt.xlabel(r'$w=2(\ln L_{\rm max} - \ln L)$')
        plt.ylabel(r'$P(<w)$')
        plt.plot( vals_sorted, my_cdf(vals_sorted))
        plt.savefig("my_fig_{}.pdf".format(indx))
    

        print(indx, log_Zl_here, mean_diff, val_ks)


# Convergence log (only works if not throttled)
print(log_Zl)




## EXTRA CODE
def superfluous_operations():
    # Load hdf5 file
    #  - copied verbatim from util_AMRGrid.py
    init_region, region_labels = amrlib.load_init_region(fname_hdf, get_labels=True)
    reindex = numpy.array([list(region_labels).index(l) for l in intr_prms])
    intr_prms = list(region_labels)

    # Remap/reorder loaded grid points to have desired parameter name order
    res_pts = res_pts[:,reindex]
    # now overwrite eta values (all filled with 0 since no attr) with correct value
    eta_indx = intr_prms.index('eta')
    mass_pts = numpy.array([tuple(getattr(t, a) for a in ['mass1','mass2']) for t in results])
    res_pts[:,eta_indx] = lalsimutils.symRatio(mass_pts[:,0],mass_pts[:,1])

    # Raw coordinates as currently constructed
    print(res_pts[:10])




