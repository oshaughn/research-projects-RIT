#! /usr/bin/env python

import argparse
import sys
import numpy as np
import numpy.lib.recfunctions
import scipy
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lal
import functools
import itertools


parser = argparse.ArgumentParser()
parser.add_argument("--inj-file", help="Name of dat file")
parser.add_argument("--inj-file-out", default="output-puffball.dat", help="Name of dat file")
parser.add_argument("--puff-factor", default=1,type=float)
parser.add_argument("--force-away", default=0,type=float,help="If >0, uses the icov to compute a metric, and discards points which are close to existing points")
parser.add_argument("--parameter", action='append', help="Parameters used as fitting parameters AND varied at a low level to make a posterior")
parser.add_argument("--no-correlation", type=str,action='append', help="Pairs of parameters, in format [mc,eta]  The corresponding term in the covariance matrix is eliminated")
#parser.add_argument("--parameter-implied", action='append', help="Parameter used in fit, but not independently varied for Monte Carlo")
parser.add_argument("--random-parameter", action='append',help="These parameters are specified at random over the entire range, uncorrelated with the grid used for other parameters.  Use for variables which correlate weakly with others; helps with random exploration")
parser.add_argument("--random-parameter-range", action='append', type=str,help="Add a range (pass as a string evaluating to a python 2-element list): --parameter-range '[0.,1000.]'   MUST specify ALL parameter ranges (min and max) in order if used.  ")
parser.add_argument("--downselect-parameter",action='append', help='Name of parameter to be used to eliminate grid points ')
parser.add_argument("--downselect-parameter-range",action='append',type=str)
parser.add_argument("--regularize",action='store_true',help="Add some ad-hoc terms based on priors, to help with nearly-singular matricies")
# ---- Optional coordinate-convert plugin (additive; legacy path byte-identical when omitted) ----
# When set, the puff lane operates in the PLUGIN basis: it forward-transforms the
# file columns named by --supplementary-coordinate-input-parameter into the
# basis named by --parameter, draws the puff displacement there, then
# INVERSE-transforms back to the file basis and writes those file columns.
# The plugin must implement both convert_coordinates AND inverse_convert_coordinates.
parser.add_argument("--supplementary-coordinate-code", default=None, type=str,
                    help="Coordinate plugin spec: 'rift_default', a .py path, or an importable dotted name. "
                         "See RIFT.misc.coordinate_plugin for the contract.")
parser.add_argument("--supplementary-coordinate-function", default=None, type=str,
                    help="Entry-point callable name. Defaults to 'convert_coordinates'.")
parser.add_argument("--supplementary-coordinate-ini", default=None, type=str,
                    help="Optional ini file handed to the plugin's prepare() hook.")
parser.add_argument("--supplementary-coordinate-chart", default=None, type=str,
                    help="Which chart in the plugin's CHARTS dict to use.")
parser.add_argument("--supplementary-coordinate-input-parameter", action='append', default=None,
                    help="File-column name to feed the plugin as an input dimension. Repeat per column. "
                         "If omitted, the plugin's CHARTS[chart] input_parameters / INPUT_PARAMETERS is used.")
opts=  parser.parse_args()

if opts.random_parameter is None:
    opts.random_parameter = []

# Extract parameter names
coord_names = opts.parameter # Used  in fit
#if opts.parameter_nofit:
#    coord_names = coord_names + opts.parameter_nofit
if coord_names is None:
    sys.exit(0)

# match up pairs in --no-correlation
corr_list = None
if not(opts.no_correlation is None):
    corr_list = []
    corr_name_list = list(map(eval,opts.no_correlation))
#    print opts.no_correlation, corr_name_list
    for my_pair in corr_name_list:
        
        i1 = coord_names.index(my_pair[0])
        i2 = coord_names.index(my_pair[1])

        if i1>-1 and i2 > -1:
            corr_list.append([i1,i2])
#        else:
#            print i1, i2
#    print opts.no_correlation, coord_names, corr_list

downselect_dict = {}



if opts.downselect_parameter:
    dlist = opts.downselect_parameter
    dlist_ranges  = list(map(eval,opts.downselect_parameter_range))
else:
    dlist = []
    dlist_ranges = []
    opts.downselect_parameter =[]
if len(dlist) != len(dlist_ranges):
    print(" downselect parameters inconsistent", dlist, dlist_ranges)
for indx in np.arange(len(dlist_ranges)):
    downselect_dict[dlist[indx]] = dlist_ranges[indx]




# ---- Optional coordinate-convert plugin ---------------------------------- #
#
# When --supplementary-coordinate-code is set, the puff displacement runs in
# the PLUGIN basis (coord_names) instead of directly in the file columns.
# This lets the user puff in a basis where the covariance is well-conditioned
# (e.g. axis-aligned with the underlying physics) even when the data file
# stores everything in a different basis.  The legacy code path is
# byte-identical when no plugin is supplied: _coord_plugin_converter stays
# None, the existing `X[:, i] = dat_raw[p]` extraction runs untouched, and
# the file-column write-back at the bottom runs untouched too.
_coord_plugin_converter = None
_coord_plugin_inverse   = None
_coord_plugin_in_names  = None  # the FILE-basis names we feed the plugin
if opts.supplementary_coordinate_code:
    from RIFT.misc.coordinate_plugin import load_coordinate_converter
    _coord_plugin_converter, _coord_plugin_module = load_coordinate_converter(
        spec=opts.supplementary_coordinate_code,
        function_name=opts.supplementary_coordinate_function,
        ini_path=opts.supplementary_coordinate_ini,
        coord_names=coord_names,
        low_level_coord_names=opts.supplementary_coordinate_input_parameter,
        chart=opts.supplementary_coordinate_chart,
        opts=opts,
        prior_map=None,
        prior_range_map=None,
    )
    # Resolve the FILE-basis input names: explicit CLI override > chart's
    # input_parameters > module's INPUT_PARAMETERS.
    _chart_spec = (
        getattr(_coord_plugin_module, "CHARTS", {}).get(opts.supplementary_coordinate_chart)
        if opts.supplementary_coordinate_chart else None
    )
    if _chart_spec is None:
        _charts = getattr(_coord_plugin_module, "CHARTS", None) or {}
        if len(_charts) == 1:
            _chart_spec = next(iter(_charts.values()))
    _coord_plugin_in_names = list(
        opts.supplementary_coordinate_input_parameter
        or (_chart_spec.get("input_parameters") if _chart_spec else None)
        or getattr(_coord_plugin_module, "INPUT_PARAMETERS", [])
    )
    if not _coord_plugin_in_names:
        sys.exit("util_HyperparameterPuffball: plugin loaded but no file-basis "
                 "input columns are declared; pass --supplementary-coordinate-input-parameter "
                 "or define INPUT_PARAMETERS / CHARTS[chart].input_parameters in the plugin.")
    # Round-trip requires an inverse.  Bail out with a clear message if the
    # plugin doesn't provide one -- silently using a pseudo-inverse here
    # would produce subtly wrong puff displacements.
    _coord_plugin_inverse = getattr(_coord_plugin_module, "inverse_convert_coordinates", None)
    if not callable(_coord_plugin_inverse):
        sys.exit("util_HyperparameterPuffball: --supplementary-coordinate-code set, but "
                 "the plugin does not define inverse_convert_coordinates.  The puff lane "
                 "needs to round-trip through the plugin basis -- add an inverse or run "
                 "without the plugin.")
    print(" util_HyperparameterPuffball: puffing in plugin basis {!r} (file columns {!r}).".format(
        list(coord_names), _coord_plugin_in_names,
    ))

# Load data, keep parameter names
dat_raw = np.genfromtxt(opts.inj_file,names=True)
X= np.zeros((len(dat_raw), len(coord_names)))
if _coord_plugin_converter is None:
    # Legacy path: --parameter names are file columns; copy directly.
    for p in coord_names:
        indx_in = coord_names.index(p)
        X[:,indx_in] = dat_raw[p]
else:
    # Plugin path: forward-transform the file's input-basis columns into
    # the puff basis (coord_names).  --parameter names need not exist as
    # file columns at all.
    missing_in = [n for n in _coord_plugin_in_names if n not in dat_raw.dtype.names]
    if missing_in:
        sys.exit("util_HyperparameterPuffball: plugin input column(s) {!r} not present in {!r}; "
                 "headers seen: {!r}".format(missing_in, opts.inj_file, list(dat_raw.dtype.names)))
    X_in = np.column_stack([np.asarray(dat_raw[n], dtype=float)
                            for n in _coord_plugin_in_names])
    X = _coord_plugin_converter(X_in,
                                coord_names=coord_names,
                                low_level_coord_names=_coord_plugin_in_names)
    X = np.asarray(X, dtype=float)
    if X.shape != (len(dat_raw), len(coord_names)):
        sys.exit("util_HyperparameterPuffball: plugin forward returned shape {!r}, "
                 "expected {!r}".format(X.shape, (len(dat_raw), len(coord_names))))


# Measure covariance matrix and generate random errors
if len(coord_names) >1:
    cov_in = np.cov(X.T)
    cov = cov_in*opts.puff_factor*opts.puff_factor

    # Check for singularities
    if np.min(np.linalg.eig(cov)[0])<1e-10:
        print(" ===> WARNING: SINGULAR MATRIX: are you sure you varied this parameters? <=== ")
        icov_pseudo = np.linalg.pinv(cov)
        # Prior range for each parameter is 1000, so icov diag terms are 10^(-6)
        # This is somewhat made up, but covers most things
        diag_terms = 1e-6*np.ones(len(cov))
        # 
        icov_proposed = icov_pseudo+np.diag(diag_terms)
        cov= np.linalg.inv(icov_proposed)

    cov_orig = np.array(cov)  # force copy
    # Remove targeted covariances
    if not(corr_list is None):
      for my_pair in corr_list:
        if my_pair[0] != my_pair[1]:
            cov[my_pair[0],my_pair[1]]=0
            cov[my_pair[1],my_pair[0]]=0
            

    # Compute errors
    rv = scipy.stats.multivariate_normal(mean=np.zeros(len(coord_names)), cov=cov,allow_singular=True)  # they are just complaining about dynamic range of parameters, usually
    delta_X = rv.rvs(size=len(X))
    X_out = X+delta_X
else:
    sigma = np.std(X)
    cov = sigma*sigma
    delta_X =np.random.normal(size=len(coord_names), scale=sigma)
    X_out = X+delta_X


# Downselect
names_downselect = list(downselect_dict.keys())
# no conversion needed
indx_ok = np.ones(len(X_out),dtype=bool)
for indx, name in enumerate(names_downselect):
    indx_ok = np.logical_and(indx_ok,  np.logical_not(np.isnan(X_out[:,indx])))
    indx_ok = np.logical_and(indx_ok,  X_out[:,indx]<= downselect_dict[name][1] )
    indx_ok = np.logical_and(indx_ok,  X_out[:,indx]>= downselect_dict[name][0] )
    print('   Increment downselect : {} {} '.format(name, np.sum(indx_ok) ))
X_out = X_out[indx_ok]
dat_raw = dat_raw[indx_ok] # must downselect here as well!
    
# Write data back into correct format and save.
#
# Legacy path: --parameter names are file columns; the puffed X_out columns
# go straight back into the matching dat_raw fields.
#
# Plugin path: X_out lives in the puff (plugin output) basis.  Inverse-
# transform it to the file basis, then write each file column from the
# matching inverse-transformed column.  --parameter names (coord_names)
# need not appear in dat_raw at all in this branch.
if _coord_plugin_converter is None:
    for p in coord_names:
        indx_in = coord_names.index(p)
        dat_raw[p] = X_out[:,indx_in]
else:
    X_in_out = _coord_plugin_inverse(
        np.asarray(X_out, dtype=float),
        coord_names=coord_names,
        low_level_coord_names=_coord_plugin_in_names,
    )
    X_in_out = np.asarray(X_in_out, dtype=float)
    if X_in_out.shape != (len(X_out), len(_coord_plugin_in_names)):
        sys.exit("util_HyperparameterPuffball: plugin inverse returned shape {!r}, "
                 "expected {!r}".format(X_in_out.shape, (len(X_out), len(_coord_plugin_in_names))))
    for j, name in enumerate(_coord_plugin_in_names):
        dat_raw[name] = X_in_out[:, j]

np.savetxt(opts.inj_file_out, dat_raw,header=" ".join(dat_raw.dtype.names))
