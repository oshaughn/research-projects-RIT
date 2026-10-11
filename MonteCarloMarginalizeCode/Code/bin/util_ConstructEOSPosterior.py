#!/usr/bin/env python
#
#  util_ConstructEOSPosterior.py
#     - takes in *generic-format* hyperparameter likelihood data
#     - uses *uniform* prior on hyperparameters.  [non-uniform priors can  be applied by the user with a supplementary function]
#     - generates posterior distribution by weighted Monte Carlo
#
# EXAMPLE:
#   python `which util_ConstructEOSPosterior.py` --fname fake_int_grid.dat  --parameter gamma1 --parameter gamma2 --lnL-offset 50

import RIFT.interpolators.BayesianLeastSquares as BayesianLeastSquares

import argparse
import sys
import numpy as np
import numpy.lib.recfunctions
import scipy
import scipy.stats
import functools
import itertools

import joblib  # http://scikit-learn.org/stable/modules/model_persistence.html

# GPU acceleration: NOT YET, just do usual
xpy_default=numpy  # just in case, to make replacement clear and to enable override
identity_convert = lambda x: x  # trivial return itself
cupy_success=False

no_plots = True
internal_dtype = np.float32  # only use 32 bit storage! Factor of 2 memory savings for GP code in high dimensions

 
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.lines as mlines
    import corner

    no_plots=False
except ImportError:
    print(" - no matplotlib - ")


from sklearn.preprocessing import PolynomialFeatures
if True:
#try:
    import RIFT.misc.ModifiedScikitFit as msf  # altenative polynomialFeatures
else:
#except:
    print(" - Faiiled ModifiedScikitFit : No polynomial fits - ")
from sklearn import linear_model

from igwn_ligolw import lsctables, utils, ligolw
lsctables.use_in(ligolw.LIGOLWContentHandler)

import RIFT.integrators.mcsampler as mcsampler
try:
    import RIFT.integrators.mcsamplerEnsemble as mcsamplerEnsemble
    mcsampler_gmm_ok = True
except:
    print(" No mcsamplerEnsemble ")
    mcsampler_gmm_ok = False
try:
    import RIFT.integrators.mcsamplerGPU as mcsamplerGPU
    mcsampler_gpu_ok = True
    mcsamplerGPU.xpy_default =xpy_default  # force consistent, in case GPU present
    mcsamplerGPU.identity_convert = identity_convert
except:
    print( " No mcsamplerGPU ")
    mcsampler_gpu_ok = False
try:
    import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAdaptiveVolume
    mcsampler_AV_ok = True
except:
    print(" No mcsamplerAV ")
    mcsampler_AV_ok = False
try:
    import RIFT.integrators.mcsamplerPortfolio as mcsamplerPortfolio
    mcsampler_Portfolio_ok = True
except:
    print(" No mcsamplerPortolfio ")





def add_field(a, descr):
    """Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError("`A' must be a structured numpy array")
    b = numpy.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b


parser = argparse.ArgumentParser()
parser.add_argument("--fname",help="filename of *.dat file (EOS-format: lnL sigma_lnL p1 p2 ... .  ASSUME any stacking over events already performed.")
parser.add_argument("--fname-output-samples",default="output-EOS-samples",help="output grid")
parser.add_argument("--fname-output-integral",default="output-EOS-integral",help="for evidencees and pipeline compatibility")
parser.add_argument("--n-output-samples",default=2000,type=int,help="output posterior samples (default 3000)")
parser.add_argument("--eos-param", type=str, default=None, help="parameterization of equation of state [spectral only, for now]")
parser.add_argument("--parameter", action='append', help="Parameter used BOTH as a fit dimension (GP/RF input) AND as a Monte Carlo sampling dimension. Adds to coord_names AND low_level_coord_names. Must be a column in --fname unless --supplementary-coordinate-code is supplied. IF NEITHER --parameter NOR --parameter-implied IS PROVIDED, coord_names defaults to the data file's column list; IF NEITHER --parameter NOR --parameter-nofit IS PROVIDED, low_level_coord_names also defaults to the data file's column list.")
parser.add_argument("--parameter-implied", action='append', help="Parameter used as a fit dimension only -- added to coord_names but NOT sampled independently. The coordinate plugin is responsible for producing it from the data file's columns. Useful for fitting in a different basis (e.g. coord_names=[u,v,w]) than the data is stored in (e.g. dat_orig_names=[x,y,z]).")
#parser.add_argument("--no-adapt-parameter",action='append',help="Disable adaptive sampling in a parameter. Useful in cases where a parameter is not well-constrained, and the a prior sampler is well-chosen.")
parser.add_argument("--parameter-nofit", action='append', help="Parameter used as a sampling dimension only -- added to low_level_coord_names but NOT to the fit basis. Useful when the MC samples in the data-file basis (e.g. dat_orig_names=[x,y,z]) but the fit lives in a transformed basis routed through the coordinate plugin.")
parser.add_argument("--integration-parameter-range",action='append', help="Integration parameter ranges. Syntax is name:[a,b]")
parser.add_argument("--downselect-parameter",action='append', help='Name of parameter to be used to eliminate grid points ')
parser.add_argument("--downselect-parameter-range",action='append',type=str)
parser.add_argument("--no-downselect",action='store_true')
parser.add_argument("--aligned-prior", default="uniform",help="Options are 'uniform', 'volumetric', and 'alignedspin-zprior'")
parser.add_argument("--cap-points",default=-1,type=int,help="Maximum number of points in the sample, if positive. Useful to cap the number of points ued for GP. See also lnLoffset. Note points are selected AT RANDOM")
parser.add_argument("--lambda-max", default=4000,type=float,help="Maximum range of 'Lambda' allowed.  Minimum value is ZERO, not negative.")
parser.add_argument("--lnL-shift-prevent-overflow",default=None,type=float,help="Define this quantity to be a large positive number to avoid overflows. Note that we do *not* define this dynamically based on sample values, to insure reproducibility and comparable integral results. BEWARE: If you shift the result to be below zero, because the GP relaxes to 0, you will get crazy answers.")
parser.add_argument("--lnL-offset",type=float,default=np.inf,help="lnL offset")
parser.add_argument("--lnL-cut",type=float,default=None,help="lnL cut [MANUAL]")
parser.add_argument("--sigma-cut",type=float,default=0.6,help="Eliminate points with large error from the fit.")
parser.add_argument("--ignore-errors-in-data",action='store_true',help='Ignore reported error in lnL. Helpful for testing purposes (i.e., if the error is zero)')
parser.add_argument("--lnL-peak-insane-cut",type=float,default=np.inf,help="Throw away lnL greater than this value. Should not be necessary")
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
parser.add_argument("--save-plots",default=False,action='store_true', help="Write plots to file (only useful for OSX, where interactive is default")
parser.add_argument("--n-max",default=3e5,type=float)
parser.add_argument("--n-step",default=1e5,type=int)
parser.add_argument("--n-eff",default=3e3,type=int)
parser.add_argument("--pool-size",default=3,type=int,help="Integer. Number of GPs to use (result is averaged)")
parser.add_argument("--fit-method",default="rf",help="rf (default) : rf|gp|gp-matern|dslice-amp.  These are the only four this driver builds; dslice-amp is for distance-export grids (a physical per-point distance model, RIFT.interpolators.dslice_amplitude_model); util_ConstructIntrinsicPosterior_GenericCoordinates.py implements quadratic|polynomial|gp_hyper|gp_lazy|cov and more.")
parser.add_argument("--gp-predict-backend",default="sklearn",choices=["sklearn","cupy"],help="gp-matern only: evaluate the fitted Matern-5/2 mean with exact cached float64 CuPy blocks (sklearn: CPU). Applies to fresh and reloaded fits, whichever backend trained them.")
parser.add_argument("--gp-predict-batch-size",default=4096,type=int,help="Maximum queries per cached CuPy GP prediction block. Used only with --gp-predict-backend cupy.")
parser.add_argument("--gp-matern-max-train-points",default=4800,type=int,help="gp-matern: deterministic balanced training bound (rho1 x lnL strata when the data carry in-plane spin; otherwise lnL strata). Exact float64 fit; see --gp-matern-fit-backend.")
parser.add_argument("--gp-matern-optimizer-maxiter",default=25,type=int,help="gp-matern: bounded single L-BFGS-B start; numerical convergence is not interpolation validation.")
parser.add_argument("--gp-matern-fit-backend",default="sklearn",choices=["sklearn","cupy"],help="gp-matern: fit hyperparameters with sklearn on CPU (default) or with the same objective in float64 CuPy on a GPU (RIFT.interpolators.cupy_matern_fit), which affords larger --gp-matern-max-train-points. Prediction uses --gp-predict-backend (numpy by default).")
parser.add_argument("--dslice-amp-decompose",action='store_true',help="dslice-amp: write lnL as an interpolated distance-marginal M(x) plus a normalized conditional in d, so distance-shape interpolation error cannot move intrinsic weights.")
parser.add_argument("--dslice-amp-loss",default="linear",help="dslice-amp: scipy least_squares loss for the per-point fits (linear|soft_l1|cauchy|huber).")
parser.add_argument("--dslice-amp-loss-scale",default=1.0,type=float,help="dslice-amp: robust-loss scale in nats.")
parser.add_argument("--dslice-amp-point-fit",default="scipy",choices=["scipy","batched"],help="dslice-amp: per-point fits by one scipy least_squares call each (default), or all points at once by a batched Levenberg-Marquardt on the same objective (on the GPU under --fit-device gpu). batched is experimental and not for production: it reaches a worse local minimum than scipy on ~0.5%% of points, enough to degrade the distance posterior on some grids.")
parser.add_argument("--dslice-amp-point-fit-jobs",default=1,type=int,help="dslice-amp, scipy point fits: number of worker processes. Fits are deterministic, so the result equals the serial one.")
parser.add_argument("--rf-seed",default=None,type=int,help="rf: random_state for the ExtraTrees forest (sklearn or cupy), so a forest can be regrown and scored twice. The same seed on the same data gives the same trees; the cupy grower reproduces predictions bit for bit (integer node sums), sklearn to float roundoff (it sums trees in thread order). Default: unseeded, as before.")
parser.add_argument("--rf-dslice-tails",default="none",choices=["none","near","far","both"],help="rf on a distance-slice grid only: beyond each nearest grid point's slice range (near: below its closest slice; far: above its farthest), continue from the fit's edge value with the averaged-amplitude distance shape fitted to that point (RIFT.interpolators.dslice_rf_tails) instead of the tree's flat extrapolation. Requires a 'dist' fit coordinate.")
parser.add_argument("--rf-dslice-tails-fmin",default="shared",choices=["shared","free"],help="--rf-dslice-tails (tail shapes fit as set by --dslice-amp-point-fit/-jobs): share one f_min across the event (median of per-point fits; default) or fit it per point.")
parser.add_argument("--rf-fit-backend",default="sklearn",choices=["sklearn","cupy"],help="rf: grow the ExtraTrees forest with sklearn on CPU (default) or on the GPU with the same algorithm (RIFT.interpolators.cupy_extratrees; same distribution of forests, different random stream). cupy requires --fit-device gpu.")
parser.add_argument("--fit-device",default="cpu",choices=["cpu","gpu"],help="rf and dslice-amp only. gpu: copy the fitted trees to the GPU and evaluate the fit there inside the sampler (RIFT.interpolators.cupy_forest; same predictions as sklearn to float64 roundoff, checked at startup), keep sample batches on the device, and convert coordinates as --coordinate-convert-xpy. Needs cupy and a visible device.")
parser.add_argument("--coordinate-convert-xpy",action='store_true',help="With --supplementary-coordinate-code: convert the data file and the sampler's batches with the vectorized RIFT.misc.waveform_coordinates_xpy instead of the plugin, after checking the two agree on data rows and on draws from the integration ranges (the plugin is kept if they do not). Avoids convert_waveform_coordinates' per-row fallthrough.")
parser.add_argument("--gp-matern-seed",default=25062842,type=int,help="gp-matern: deterministic row selection and sklearn seed, independent of sampler randomness.")
parser.add_argument("--fit-load-gp",default=None,type=str,help="Filename of GP fit to load. Overrides fitting process, but user MUST correctly specify coordinate system to interpret the fit with.  Does not override loading and converting the data.")
parser.add_argument("--fit-save-gp",default=None,type=str,help="Filename of GP fit to save. ")
parser.add_argument("--fit-order",type=int,default=2,help="Fit order (polynomial case: degree)")
parser.add_argument("--fit-distance-tail",action='store_true',help="Distance-export (.dslice) runs ONLY, i.e. runs that carry an explicit distance fit coordinate. Beyond each intrinsic point's exported distance support, make the fitted lnL decay to zero as d->infinity instead of holding its edge value. An RF/ExtraTrees fit is piecewise constant outside its training envelope, so without this it holds lnL flat while the volumetric prior keeps growing like d^2, and the recovered distance posterior comes out ~18 percent too wide. Changes nothing on the support, so it is a strict addition. It is an error to request this without a distance fit coordinate.")
parser.add_argument("--no-plots",action='store_true')
parser.add_argument("--using-eos-type", type=str, default=None, help="Name of EOS parameterization (must match what is used for inputs). Will use EOS parameterization to identify appropriate field headers")
parser.add_argument("--sampler-method",default="adaptive_cartesian",help="adaptive_cartesian|GMM|AV|adaptive_cartesian_gpu|portfolio")
parser.add_argument("--sampler-portfolio",default=None,action='append',type=str,help="Portfolio member, one NAME per flag (repeat the flag): AV|GMM|adaptive_cartesian_gpu|NFlow.  Not comma-separated -- a comma-joined 'AV,GMM' matches no sampler.  Requires --sampler-method portfolio.")
parser.add_argument("--sampler-portfolio-args",default=None, action='append', type=str, help='eval-able dictionary of extra setup() arguments for the corresponding --sampler-portfolio member.  One per flag, in the same order and the same COUNT as --sampler-portfolio.')
parser.add_argument("--sampler-portfolio-allow-stratified-density",action='store_true',help="Accept a portfolio whose members cannot form the balance-heuristic mixture density q_mix, i.e. run the legacy stratified per-member estimator even when a member reports its sampling density on a non-normalized scale.  THE EVIDENCE IS THEN BIASED (measured: 0.753772 on a constant integrand whose exact ln Z is 1.386294).  Without this the portfolio refuses at setup.  Exists so an unusual member combination is recoverable without editing RIFT; do not use it for production evidence.")
parser.add_argument("--internal-use-lnL",action='store_true',help="integrator internally manipulates lnL..   ")
parser.add_argument("--internal-correlate-parameters",default=None,type=str,help="comman-separated string indicating parameters that should be sampled allowing for correlations. Must be sampling parameters. Only implemented for gmm.  If string is 'all', correlate *all* parameters")
parser.add_argument("--internal-n-comp",default=1,type=int,help="number of components to use for GMM sampling. Default is 1, because we expect a unimodal posterior in well-adapted coordinates.  If you have crappy coordinates, use more")
parser.add_argument("--force-no-adapt",action='store_true',help="Disable adaptation, both of the tempering exponent *and* the individual sampling prior(s)")
parser.add_argument("--tripwire-fraction",default=0.05,type=float,help="Fraction of nmax of iterations after which n_eff needs to be greater than 1+epsilon for a small number epsilon")

# Supplemental likelihood factors: convenient way to effectively change the mass/spin prior in arbitrary ways for example
# Note this supplemental factor is passed the *fitting* arguments, directly.  Use with extreme caution, since we often change the dimension in a DAG 
parser.add_argument("--supplementary-likelihood-factor-code", default=None,type=str,help="Import a module (in your pythonpath!) containing a supplementary factor for the likelihood.  Used to impose supplementary external priors of arbitrary complexity and external dependence (e.g., imposing alternate EOS priors)")
parser.add_argument("--supplementary-likelihood-factor-function", default=None,type=str,help="With above option, specifies the specific function used as an external likelihood. EXPERTS ONLY")
parser.add_argument("--supplementary-likelihood-factor-ini", default=None,type=str,help="With above option, specifies an ini file that is parsed (here) and passed to the preparation code, called when the module is first loaded, to configure the module. EXPERTS ONLY")
parser.add_argument("--supplementary-coordinate-code", default=None,type=str,help="Coordinate conversion/prior code. Accepts: the literal 'rift_default' (use RIFT.lalsimutils.convert_waveform_coordinates plus RIFT-standard priors); a filesystem path ending in .py (loaded as a plugin); or any importable dotted module name. See RIFT.misc.coordinate_plugin for the interface plugins must implement.")
parser.add_argument("--supplementary-coordinate-function", default=None, type=str, help="Name of the entry-point callable inside the module named by --supplementary-coordinate-code. Defaults to 'convert_coordinates'.")
parser.add_argument("--supplementary-coordinate-ini", default=None, type=str, help="Optional ini file parsed and handed to the coordinate plugin's prepare() hook so it can read its own configuration block(s).")
parser.add_argument("--supplementary-coordinate-chart", default=None, type=str, help="Which chart (coordinate system) defined by the plugin to use for this run. Required when the plugin's CHARTS dict has more than one entry; ignored when the plugin doesn't define CHARTS. Different charts can share parameter names but imply different priors -- the chart name disambiguates which (name -> prior) mapping is installed.")
parser.add_argument("--get-range-from-external", action='store_true', help="Ask the coordinate plugin for integration ranges: calls get_bounds(low_level_coord_names, ranges, **kwargs), which returns {name: [lo,hi]} in the SAMPLING basis. Replaces data-derived and chart ranges; --integration-parameter-range wins. Errors raise.")
parser.add_argument("--external-range-args", action='append', type=str, help="key=value passed to the plugin's get_bounds. Values are parsed as python literals when possible.")
opts=  parser.parse_args()
if opts.fit_method != 'gp-matern':
    if opts.gp_predict_backend != "sklearn":
        parser.error("--gp-predict-backend cupy requires --fit-method gp-matern")
    if opts.gp_matern_fit_backend != "sklearn":
        parser.error("--gp-matern-fit-backend cupy requires --fit-method gp-matern")
if opts.fit_device == "gpu" and opts.fit_method not in ("rf", "dslice-amp"):
    parser.error("--fit-device gpu supports --fit-method rf and dslice-amp")
# options a fit method would silently ignore are refused
_defaults = {a.dest: a.default for a in parser._actions}
def _set(*names):
    return [n for n in names if getattr(opts, n) != _defaults[n]]
if opts.fit_method != 'dslice-amp':
    # --rf-dslice-tails fits its tail shapes with the same per-point fitter, so it uses the point-fit options
    _tails = opts.fit_method == 'rf' and opts.rf_dslice_tails != 'none'
    _bad = _set('dslice_amp_decompose', 'dslice_amp_loss', 'dslice_amp_loss_scale',
                *(() if _tails else ('dslice_amp_point_fit', 'dslice_amp_point_fit_jobs')))
    if _bad:
        parser.error("{} apply to --fit-method dslice-amp only".format(", ".join("--" + b.replace("_", "-") for b in _bad)))
else:
    _bad = _set('fit_distance_tail', 'cap_points', 'ignore_errors_in_data')
    if _bad:
        parser.error("{} are not used by --fit-method dslice-amp".format(", ".join("--" + b.replace("_", "-") for b in _bad)))
if opts.fit_method != 'gp-matern':
    _bad = _set('gp_matern_max_train_points', 'gp_matern_optimizer_maxiter', 'gp_matern_seed', 'gp_predict_batch_size')
    if _bad:
        parser.error("{} apply to --fit-method gp-matern only".format(", ".join("--" + b.replace("_", "-") for b in _bad)))
if opts.fit_method != 'rf':
    _bad = _set('rf_dslice_tails', 'rf_dslice_tails_fmin')
    if _bad:
        parser.error("{} apply to --fit-method rf only".format(", ".join("--" + b.replace("_", "-") for b in _bad)))
elif opts.rf_dslice_tails == 'none' and _set('rf_dslice_tails_fmin'):
    parser.error("--rf-dslice-tails-fmin needs --rf-dslice-tails near|far|both")
if opts.fit_method == 'rf' and opts.rf_dslice_tails != 'none':
    # the rest is checked after the coordinates are known; fail before data loading where possible
    if opts.ignore_errors_in_data:
        parser.error("--rf-dslice-tails needs the per-row sigma column (drop --ignore-errors-in-data)")
    if (opts.parameter or opts.parameter_implied) and 'dist' not in (opts.parameter or []) + (opts.parameter_implied or []):
        parser.error("--rf-dslice-tails needs 'dist' as a fit coordinate (--parameter dist or --parameter-implied dist)")
if opts.rf_seed is not None and opts.fit_method not in ('rf', 'dslice-amp'):
    parser.error("--rf-seed applies to --fit-method rf and dslice-amp (its field forest)")
if opts.fit_method == 'gp-matern' and 'cupy' in (opts.gp_matern_fit_backend, opts.gp_predict_backend):
    try:    # fail before data loading, not after it
        import cupy as _cupy_probe
        _cupy_probe.zeros(1)
        _dev_total = _cupy_probe.cuda.Device().mem_info[1]
    except Exception as _err:
        parser.error("--gp-matern-fit-backend/--gp-predict-backend cupy need cupy and a visible CUDA device ({})".format(_err))
    if opts.gp_matern_fit_backend == 'cupy' and 48.0 * opts.gp_matern_max_train_points ** 2 > 0.8 * _dev_total:
        parser.error("--gp-matern-max-train-points {} needs ~{:.0f} GB on the device for the CuPy fit (6 n^2 float64); {:.0f} GB available".format(
            opts.gp_matern_max_train_points, 48.0 * opts.gp_matern_max_train_points ** 2 / 1e9, _dev_total / 1e9))
if opts.dslice_amp_point_fit_jobs < 1:
    parser.error("--dslice-amp-point-fit-jobs must be at least 1")
if opts.dslice_amp_point_fit_jobs > 1 and opts.dslice_amp_point_fit != "scipy":
    parser.error("--dslice-amp-point-fit-jobs applies to scipy point fits only")
if opts.rf_fit_backend == "cupy" and (opts.fit_method != "rf" or opts.fit_device != "gpu"):
    parser.error("--rf-fit-backend cupy requires --fit-method rf and --fit-device gpu")
if opts.fit_device == "gpu":
    if opts.fit_distance_tail:
        parser.error("--fit-distance-tail is host-only; it is not available with --fit-device gpu")
    try:    # fail before the fit, not after it
        import cupy as _cupy_probe
        _cupy_probe.zeros(1)
    except Exception as _err:
        parser.error("--fit-device gpu needs cupy and a visible CUDA device ({})".format(_err))

#print(" WARNING: Always use internal_use_lnL for now ")
#opts.internal_use_lnL=True

no_plots = no_plots |  opts.no_plots
lnL_shift = 0
lnL_default_large_negative = -500
# value the rf fit returns for rows with a nonfinite or |x| > 1e37 coordinate, on CPU and GPU alike
# (a floor; it was +500, a ceiling, until rift_O4d fixed the sign)
rf_nonfinite_fill = lnL_default_large_negative
if opts.lnL_shift_prevent_overflow:
    lnL_shift  = opts.lnL_shift_prevent_overflow



### Comparison data (from LI)
###

downselect_dict = {}
dlist = []
dlist_ranges=[]
if opts.downselect_parameter:
    dlist = opts.downselect_parameter
    dlist_ranges  = map(eval,opts.downselect_parameter_range)
else:
    dlist = []
    dlist_ranges = []
if len(dlist) != len(dlist_ranges):
    print(" downselect parameters inconsistent", dlist, dlist_ranges)
for indx in np.arange(len(dlist_ranges)):
    downselect_dict[dlist[indx]] = dlist_ranges[indx]

if opts.no_downselect:
    downselect_dict={}


test_converged={}

###
### Retrieve data
###
#  int_sig sigma/L gamma1 gamma2 ...
col_lnL = 0
dat_orig = dat = np.loadtxt(opts.fname)
dat_orig = dat[dat[:,col_lnL].argsort()] # sort  http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
print(" Original data size = ", len(dat), dat.shape)
dat_orig_names = None
with open(opts.fname,'r') as f:
    header_str = f.readline()
    header_str = header_str.rstrip()
dat_orig_names = header_str.replace('#','').split()[2:]

###
### Parameters in use
###

# Decoupled fit basis vs Monte Carlo sampling basis -- mirrors the
# convention established by util_ConstructIntrinsicPosterior_GenericCoordinates.py
# and required for the new coordinate-plugin path:
#
#   --parameter X        -> X is BOTH a fit (GP/RF) and a sampling (MC) dim
#   --parameter-implied X-> X is a fit dim ONLY (the plugin produces it from
#                           dat_orig_names; the MC integrator never sees it)
#   --parameter-nofit X  -> X is a sampling dim ONLY (the MC integrates over
#                           it; the fit never sees it).  Typical use: MC in
#                           the data-file basis while the fit lives in a
#                           transformed basis routed through the plugin.
#
# Legacy fallback (preserves the pre-decoupling default): if the user
# supplies neither --parameter nor --parameter-implied, coord_names
# defaults to the data file's column list.  Likewise low_level_coord_names
# defaults to the data file's columns when --parameter and --parameter-nofit
# are both absent.  That way a bare invocation -- no flags -- still does
# "fit on every column in the file, MC sample in the same basis", which
# is what every existing hyperpipe / EOS-posterior demo relies on.
_user_params  = list(opts.parameter)         if opts.parameter         else []
_user_implied = list(opts.parameter_implied) if opts.parameter_implied else []
_user_nofit   = list(opts.parameter_nofit)   if opts.parameter_nofit   else []

if not _user_params and not _user_implied:
    coord_names = list(dat_orig_names)             # legacy default
else:
    coord_names = _user_params + _user_implied      # fit basis

if not _user_params and not _user_nofit:
    low_level_coord_names = list(dat_orig_names)   # legacy default
else:
    low_level_coord_names = _user_params + _user_nofit  # MC basis

# The "easy case": every fit coordinate is also a sampling coordinate (the
# user supplied only --parameter calls, in the plugin's output basis).  The
# MC samples are then ALREADY in the fit basis, so the per-sample
# convert_coords is a pure column selection/permutation -- the plugin is
# needed only for (a) the one-time conversion of the input grid into the
# fit basis and (b) the inverse transform of the output samples back to the
# fiducial (data-file) coordinates.  When this is False (--parameter-implied
# present), every MC sample must be routed through the plugin.
_per_sample_needs_plugin = not all(name in low_level_coord_names for name in coord_names)

error_factor = len(coord_names)
name_index_dict ={}
for name in dat_orig_names:
    try:
        name_index_dict[name] = 2+dat_orig_names.index(name)
    except:
        raise Exception(" Currently fitting parameter names must match columns in data file ")
# TeX dictionary
print(" Coordinate names for fit :, ", coord_names, " from ", dat_orig_names, " indexed as ", name_index_dict)
print(" Coordinate names for Monte Carlo :, ", low_level_coord_names)


###
### Integration ranges
###

param_ranges = {}
for range_code  in (opts.integration_parameter_range or []):
    name, range_str  = range_code.split(':')
    range_expr =     eval(range_str)  # define. Better to split on , for example
    param_ranges[name]  = np.array(range_expr)

# Add in integration range for everything else, if nothing specified
for name in dat_orig_names:
    if not name in param_ranges:
        vals = dat_orig[:,name_index_dict[name]]
        param_ranges[name] = [np.min(vals), np.max(vals)]

###
### Prior functions : default is UNIFORM, since it is unmodeled and generic
###
#
# PRIORS AND THE CHANGE-OF-VARIABLES JACOBIAN.
# We do NOT apply the Jacobian |det d(coord)/d(low_level_coord)| of the
# coordinate transform here: the sampling-basis prior is taken to be
# uniform (or whatever the plugin's chart installs), full stop.  The
# assumption is that the user knows what they are doing and folds any
# desired measure -- the transform Jacobian, a physically-motivated EOS
# prior, anything non-uniform -- into a --supplementary-likelihood-factor
# function, which is evaluated in the fitting/sampling coordinates and so
# can express an arbitrary prior exactly.  For a linear/affine transform
# the Jacobian is constant and a uniform input prior maps to a uniform
# output prior, so the default is already correct there.  A nonlinear
# plugin that wants the induced prior must supply that factor (or, in
# future, populate the plugin's jacobian() hook -- see the TODO in
# RIFT.misc.coordinate_plugin / the tracking issue on falling back to the
# util_ConstructIntrinsicPosterior_GenericCoordinates default-prior
# conventions).

def uniform_prior(x):
    return np.ones(x.shape)

prior_map = {}
for name in low_level_coord_names:
    prior_map[name] = uniform_prior
# NOTE: range validation (every sampled name must have an integration range)
# is deferred until AFTER the coordinate plugin is loaded, below.  The plugin
# can install ranges for the names it produces (CHARTS[chart]['ranges']), and
# for the remaining names we can auto-derive ranges by forward-transforming
# the input grid.  Validating here -- as this script used to -- made the
# chart-declared ranges unreachable dead code and forced the user to repeat
# every range on the command line.


prior_range_map = param_ranges

# prior_map  = { 'gamma1':eos_param_uniform_prior, 'gamma2':eos_param_uniform_prior,
# }
# # Les: somewhat more aggressive: 
# #    gamma1: 0.2,2
# #    gamma2: -1.67, 1.7
# prior_range_map = { 'gamma1':  [0.707899,1.31], 'gamma2':[-1.6,1.7], 'gamma3':[-0.6,0.6], 'gamma4':[-0.02,0.02]
# }


###
### Supplemental likelihood: load (as in ILE)
###
supplemental_ln_likelihood= None
supplemental_ln_likelihood_prep =None
supplemental_ln_likelihood_parsed_ini=None
supplemental_ln_likelihood_offset_fn=None
supplemental_ln_likelihood_offset=0.0
# Supplemental likelihood factor. Must have identical call sequence to 'likelihood_function'. Called with identical raw inputs (including cosines/etc)
if opts.supplementary_likelihood_factor_code and opts.supplementary_likelihood_factor_function:
  print(" EXTERNAL SUPPLEMENTARY LIKELIHOOD FACTOR : {}.{} ".format(opts.supplementary_likelihood_factor_code,opts.supplementary_likelihood_factor_function))
  __import__(opts.supplementary_likelihood_factor_code)
  external_likelihood_module = sys.modules[opts.supplementary_likelihood_factor_code]
  supplemental_ln_likelihood = getattr(external_likelihood_module,opts.supplementary_likelihood_factor_function)
  name_prep = "prepare_"+opts.supplementary_likelihood_factor_function
  # Optional <function>_offset hook, same naming convention as prepare_<function>.  A plugin whose
  # contribution is large must return a CENTRED lnL -- the default path here exponentiates it
  # (likelihood_function*np.exp(supplemental)), and float64 overflows past ~709, which a single
  # loud-event quadratic (lnL_peak ~ SNR^2/2) exceeds on its own.  That centring is a constant
  # multiplicative factor: harmless in the posterior, but it would otherwise leak into the ABSOLUTE
  # evidence written below and make it wrong by exactly that constant.  The plugin reports what it
  # removed; we add it back at the write site.  Queried after integration, not here: the value is
  # only known once the plugin has been prepared/loaded.
  name_offset = opts.supplementary_likelihood_factor_function+"_offset"
  if hasattr(external_likelihood_module,name_offset):
    supplemental_ln_likelihood_offset_fn=getattr(external_likelihood_module,name_offset)
  if hasattr(external_likelihood_module,name_prep):
    supplemental_ln_likelihood_prep=getattr(external_likelihood_module,name_prep)
    # Check for and load in ini file associated with external library
    if opts.supplementary_likelihood_factor_ini:
      import configparser as ConfigParser
      config = ConfigParser.ConfigParser()
      config.optionxform=str # force preserve case!
      config.read(opts.supplementary_likelihood_factor_ini)
      supplemental_ln_likelihood_parsed_ini=config

    # Prepare the plugin, telling it what coordinates we are using by name.  Called whether or not
    # an ini was supplied (config=None then): a plugin configured entirely by environment still
    # needs the basis, and without it the arrays it is handed are anonymous -- same count, same
    # order, no error, wrong coordinates.
    # coords MUST be low_level_coord_names, not coord_names: the sampler integrates over
    # low_level_coord_names and so calls supplemental_ln_likelihood(*x) with one array per
    # SAMPLING coordinate, in that order.  With --parameter-implied/--parameter-nofit the two
    # lists differ, and handing over the fit basis would make the plugin label those arrays
    # wrongly -- evaluating at the wrong coordinates without any error.
    supplemental_ln_likelihood_prep(config=supplemental_ln_likelihood_parsed_ini,coords=low_level_coord_names)

supplemental_coordinate_convert = None
supplemental_coordinate_inverse = None
_coord_plugin_in_names = None
if opts.supplementary_coordinate_code:
    # Resolve the user-supplied coordinate-convert plugin.  The loader
    # accepts three forms in --supplementary-coordinate-code: the literal
    # 'rift_default', a filesystem path to a .py file, or an importable
    # dotted module name.  The plugin must expose a callable named by
    # --supplementary-coordinate-function (default 'convert_coordinates')
    # with the signature (x_in, coord_names, low_level_coord_names, **kwargs)
    # returning a 2-D ndarray of shape (N, len(coord_names)).  Plugins may
    # optionally define prepare() (one-shot setup, gets the parsed ini and
    # the active coord-name lists) and register_priors() (mutate prior_map
    # in place).  See RIFT.misc.coordinate_plugin for the full contract.
    from RIFT.misc.coordinate_plugin import load_coordinate_converter, resolve_input_parameters
    # Tell the loader (and the plugin's prepare hook) which basis the plugin
    # will actually be fed as input.  In the easy case the per-sample path
    # bypasses the plugin entirely, so the only inputs it ever sees are the
    # data file's columns; declaring low_level_coord_names (= the plugin's
    # OUTPUT basis in that case) would make a strict plugin reject its own
    # documented usage.
    _plugin_fed_input_names = low_level_coord_names if _per_sample_needs_plugin else dat_orig_names
    supplemental_coordinate_convert, _coord_plugin_module = load_coordinate_converter(
        spec=opts.supplementary_coordinate_code,
        function_name=opts.supplementary_coordinate_function,
        ini_path=opts.supplementary_coordinate_ini,
        coord_names=coord_names,
        low_level_coord_names=_plugin_fed_input_names,
        chart=opts.supplementary_coordinate_chart,
        opts=opts,
        prior_map=prior_map,
        prior_range_map=prior_range_map,
    )
    # Optional inverse (plugin basis -> file basis), same hook the puff lane
    # (util_HyperparameterPuffball.py) uses.  Needed to write the final
    # posterior samples in fiducial coordinates when the sampling basis is
    # not a subset of the data file's columns.
    supplemental_coordinate_inverse = getattr(_coord_plugin_module, "inverse_convert_coordinates", None)
    _coord_plugin_in_names = resolve_input_parameters(
        _coord_plugin_module, chart=opts.supplementary_coordinate_chart
    ) or list(dat_orig_names)

# Ranges from the plugin's get_bounds hook, in the sampling basis.  They
# replace data-derived and chart ranges; --integration-parameter-range wins.
# Failures raise: a range in the wrong frame must not be used silently.
if opts.get_range_from_external:
    if supplemental_coordinate_convert is None:
        raise Exception(" --get-range-from-external needs --supplementary-coordinate-code ")
    from RIFT.misc.coordinate_plugin import call_get_bounds
    _cli_range_names = set(name for name, _ in (r.split(':', 1) for r in (opts.integration_parameter_range or [])))
    for _name, _rng in call_get_bounds(_coord_plugin_module, low_level_coord_names, param_ranges, opts.external_range_args).items():
        if _name in low_level_coord_names and _name not in _cli_range_names:
            param_ranges[_name] = _rng
            print(" Integration range for {} from get_bounds : {} ".format(_name, _rng))

# Auto-derive integration ranges for sampled names that are still missing one:
# forward-transform the input grid into the sampling basis and use the
# column-wise min/max.  Explicit --integration-parameter-range and
# chart-declared ranges (installed by the loader above) always win; this is
# only a fallback so the easy case needs no per-name range flags at all.
if supplemental_coordinate_convert is not None:
    _names_missing_range = [p for p in low_level_coord_names if p not in param_ranges]
    if _names_missing_range:
        try:
            _dat_sampling_basis = supplemental_coordinate_convert(
                dat[:, 2:],
                coord_names=_names_missing_range,
                low_level_coord_names=dat_orig_names,
            )
            for _k, _name in enumerate(_names_missing_range):
                _vals = np.asarray(_dat_sampling_basis)[:, _k]
                param_ranges[_name] = [np.min(_vals), np.max(_vals)]
                print(" Integration range for {} auto-derived from transformed input grid : {} ".format(_name, param_ranges[_name]))
        except Exception as _err:
            print(" Could not auto-derive integration ranges for {} via the coordinate plugin ({}); supply --integration-parameter-range ".format(_names_missing_range, _err))

# Deferred range validation (see note at the prior_map seeding above).
for name in low_level_coord_names:
    if not (name in param_ranges):
        raise Exception(" {} not provided a parameter range ".format(name))

# Vectorized coordinate conversion (--coordinate-convert-xpy; implied by --fit-device gpu).  Used only
# where it reproduces the plugin: on data rows, and on draws from the integration box when the
# per-sample path goes through the plugin.  NaN rows (e.g. |cos_theta| > 1) must match as well.
_xpy_convert = None
if supplemental_coordinate_convert is not None and (opts.coordinate_convert_xpy or opts.fit_device == 'gpu'):
    from RIFT.misc.waveform_coordinates_xpy import convert_waveform_coordinates_xpy, check_against
    _rows = dat[np.linspace(0, len(dat) - 1, min(len(dat), 4096)).astype(int), 2:]
    _checks = []
    try:
        _checks.append(("data",) + check_against(supplemental_coordinate_convert, coord_names, dat_orig_names, _rows))
        if _per_sample_needs_plugin:
            _box = np.array([param_ranges[p] for p in low_level_coord_names], dtype=float)
            _draw = np.random.default_rng(0).uniform(_box[:, 0], _box[:, 1], (4096, len(low_level_coord_names)))
            _checks.append(("samples",) + check_against(supplemental_coordinate_convert, coord_names, low_level_coord_names, _draw))
    except NotImplementedError as _err:
        _checks.append(("unsupported", False, np.inf, str(_err)))
    print(" COORDINATE-CONVERT-XPY check (max error in units of rtol 1e-10): {}".format(_checks))
    if all(c[1] for c in _checks):
        _xpy_convert = convert_waveform_coordinates_xpy
    else:
        print(" COORDINATE-CONVERT-XPY : vectorized converter does not reproduce the plugin; keeping the plugin")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

def adderr(y):
    val,err = y
    return val+error_factor*err

def fit_gp(x,y,x0=None,symmetry_list=None,y_errors=None,hypercube_rescale=False,fname_export="gp_fit"):
    """
    x = array so x[0] , x[1], x[2] are points.
    """

    # If we are loading a fit, override everything else
    if opts.fit_load_gp:
        print(" WARNING: Do not re-use fits across architectures or versions : pickling is not transferrable ")
        my_gp=joblib.load(opts.fit_load_gp)
        return lambda x:my_gp.predict(x)

    # Amplitude: 
    #   - We are fitting lnL.  
    #   - We know the scale more or less: more than 2 in the log is bad
    # Scale
    #   - because of strong correlations with chirp mass, the length scales can be very short
    #   - they are rarely very long, but at high mass can be long
    #   - I need to allow for a RANGE

    length_scale_est = []
    length_scale_bounds_est = []
    for indx in np.arange(len(x[0])):
        # These length scales have been tuned by expereience
        length_scale_est.append( 2*np.std(x[:,indx])  )  # auto-select range based on sampling retained
        length_scale_min_here= np.max([1e-3,0.2*np.std(x[:,indx]/np.sqrt(len(x)))])
        length_scale_bounds_est.append( (length_scale_min_here , 5*np.std(x[:,indx])   ) )  # auto-select range based on sampling *RETAINED* (i.e., passing cut).  Note that for the coordinates I usually use, it would be nonsensical to make the range in coordinate too small, as can occasionally happens

    print(" GP: Input sample size ", len(x), len(y))
    print(" GP: Estimated length scales ")
    print(length_scale_est)
    print(length_scale_bounds_est)

    if not (hypercube_rescale):
        # These parameters have been hand-tuned by experience to try to set to levels comparable to typical lnL Monte Carlo error
        kernel = WhiteKernel(noise_level=0.1,noise_level_bounds=(1e-2,1))+C(0.5, (1e-3,1e1))*RBF(length_scale=length_scale_est, length_scale_bounds=length_scale_bounds_est)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8)

        gp.fit(x,y)

        print(" Fit: std: ", np.std(y - gp.predict(x)),  "using number of features ", len(y))

        if opts.fit_save_gp:
            print(" Attempting to save fit ", opts.fit_save_gp+".pkl")
            joblib.dump(gp,opts.fit_save_gp+".pkl")
        
        return lambda x: gp.predict(x)
    else:
        x_scaled = np.zeros(x.shape)
        x_center = np.zeros(len(length_scale_est))
        x_center = np.mean(x)
        print(" Scaling data to central point ", x_center)
        for indx in np.arange(len(x)):
            x_scaled[indx] = (x[indx] - x_center)/length_scale_est # resize

        kernel = WhiteKernel(noise_level=0.1,noise_level_bounds=(1e-2,1))+C(0.5, (1e-3,1e1))*RBF( len(x_center), (1e-3,1e1))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8)
        
        gp.fit(x_scaled,y)
        print(" Fit: std: ", np.std(y - gp.predict(x_scaled)),  "using number of features ", len(y))  # should NOT be perfect

        return lambda x,x0=x_center,scl=length_scale_est: gp.predict( (x-x0 )/scl)

def fit_gp_matern(x,y,y_errors,rho1=None):
    """Bounded standardized Matern-5/2 recipe of RIFT.interpolators.matern_gp, as in the generic CIP."""
    import json
    from RIFT.interpolators.matern_gp import fit_matern_gp
    prediction_offset = 0.
    if opts.fit_load_gp:
        model = joblib.load(opts.fit_load_gp)
        record = getattr(model,"rift_matern_provenance",None)
        if record is None or "lnL_shift" not in record.get("provenance",{}):
            raise ValueError("gp-matern reload requires saved coordinate/lnL-shift provenance")
        if record.get("feature_names") != list(coord_names):
            raise ValueError("Saved gp-matern feature coordinates differ from this run")
        prediction_offset = float(record["provenance"]["lnL_shift"]) - lnL_shift
    else:
        bound = opts.gp_matern_max_train_points
        if opts.cap_points > 0:
            bound = min(bound,opts.cap_points)
        fitter = fit_matern_gp
        if opts.gp_matern_fit_backend == "cupy":
            from RIFT.interpolators.cupy_matern_fit import fit_matern_gp_cupy as fitter
        model,record = fitter(x,y,y_errors,
            max_train_points=bound,optimizer_maxiter=opts.gp_matern_optimizer_maxiter,
            seed=opts.gp_matern_seed,rho1=rho1,feature_names=list(coord_names),
            provenance={"lnL_shift":float(lnL_shift),"fit_method":"gp-matern",
                        "fit_backend":opts.gp_matern_fit_backend})
        model.rift_matern_provenance = record
        print("GP-MATERN-RECORD",json.dumps({k:v for k,v in record.items() if k!="selected_indices"}))
        if opts.fit_save_gp:
            joblib.dump(model,opts.fit_save_gp+".pkl")
            with open(opts.fit_save_gp+".meta.json","w") as stream:
                json.dump(record,stream,indent=2)
    from RIFT.interpolators.cached_matern_gp import CachedMaternMean
    if isinstance(model,CachedMaternMean):
        # CuPy-fitted model: the stored backend is a training-host detail, not a request
        model.backend = "cupy" if opts.gp_predict_backend == "cupy" else "numpy"
        model.batch_size = int(opts.gp_predict_batch_size)
        model._device_state = None
        print(" Matern means via CachedMaternMean: backend ", model.backend, " batch ", model.batch_size)
    elif opts.gp_predict_backend == "cupy":
        from RIFT.interpolators.cached_matern_gp import from_sklearn
        model = from_sklearn(model,backend="cupy",batch_size=opts.gp_predict_batch_size)
        print(" Cached CUDA Matern means: float64, prediction batch ", opts.gp_predict_batch_size)
    def fn_return(x_in):
        # Same guard as fit_rf: a nonfinite coordinate round-trip gets the default floor, not a crash
        x_in = np.asarray(x_in,dtype=np.float64)
        f_out = lnL_default_large_negative*np.ones(len(x_in))
        ok = np.all(np.isfinite(x_in),axis=-1) & np.all(np.abs(x_in) < 1e37,axis=-1)
        if np.any(ok):
            f_out[ok] = model.predict(x_in[ok]) + prediction_offset
        return f_out
    return fn_return

def map_funcs(func_list,obj):
    return [func(obj) for func in func_list]
def fit_gp_pool(x,y,n_pool=10,**kwargs):
    """
    Split the data into 10 parts, and return a GP that averages them
    """
    x_copy = np.array(x)
    y_copy = np.array(y)
    indx_list =np.arange(len(x_copy))
    np.random.shuffle(indx_list) # acts in place
    partition_list = np.array_split(indx_list,n_pool)
    gp_fit_list =[]
    for part in partition_list:
        print(" Fitting partition ")
        gp_fit_list.append(fit_gp(x[part],y[part],**kwargs))
    fn_out =  lambda x: np.mean( map_funcs( gp_fit_list,x), axis=0)
    print(" Testing ", fn_out([x[0]]))
    return fn_out


def _cupy():
    try:
        import cupy
        cupy.zeros(1)
        return cupy
    except Exception as err:
        raise RuntimeError("--fit-device gpu needs cupy and a visible CUDA device ({})".format(err))


def _device_forest_fit(rf, x_check, fill):
    """Device evaluation of a fitted forest with fn_return's row guard: rows with a nonfinite or
    |x| > 1e37 coordinate get `fill`. Returns cupy for cupy input and numpy for numpy input. rf is a
    fitted sklearn forest, checked at startup against rf.predict on training rows x_check, or an
    already built CupyForest (x_check None)."""
    cp = _cupy()
    from RIFT.interpolators.cupy_forest import CupyForest
    if isinstance(rf, CupyForest):
        forest, msg = rf, "grown on device"
    else:
        forest = CupyForest(rf)
        xc = np.asarray(x_check, dtype=float)
        xc = xc[np.all(np.isfinite(xc), axis=-1) & np.all(~(np.abs(xc) > 1e37), axis=-1)]   # rows fn_return evaluates
        if len(xc) == 0:
            raise RuntimeError("--fit-device gpu: no training row has finite coordinates to check the device forest on")
        xc = xc[np.linspace(0, len(xc) - 1, min(len(xc), 4096)).astype(int)]
        err = float(np.max(np.abs(cp.asnumpy(forest.predict(xc)) - rf.predict(xc))))
        msg = "max |gpu - sklearn| on {} training rows = {:.3g}".format(len(xc), err)
        if not err < 1e-9:
            raise RuntimeError("--fit-device gpu: device forest disagrees with sklearn ({:.3g})".format(err))
    print(" FIT-DEVICE gpu : forest of {} trees, {} nodes, {:.2f} GB on device; {}".format(
        forest.n_trees, forest.n_nodes, forest.device_bytes / 1e9, msg))

    def fn_return(x_in, forest=forest):
        on_host = not isinstance(x_in, cp.ndarray)
        x_in = cp.asarray(x_in, dtype=cp.float64)
        ok = cp.all(cp.isfinite(x_in), axis=-1) & cp.all(~(cp.abs(x_in) > 1e37), axis=-1)
        f_out = cp.where(ok, forest.predict(cp.where(ok[:, None], x_in, 0.0)), fill)
        return cp.asnumpy(f_out) if on_host else f_out
    return fn_return


def _rf_fingerprint(trees, seed):
    """One log line identifying a forest: tree count, node count, threshold and leaf-value sums."""
    nodes = sum(int(t.node_count) for t in trees)
    thr = sum(float(np.sum(t.threshold[t.children_left >= 0])) for t in trees)
    leaf = sum(float(np.sum(t.value[t.children_left < 0])) for t in trees)
    print(" RF-FOREST seed={} trees={} nodes={} thr_sum={:.17g} leaf_sum={:.12g}".format(seed, len(trees), nodes, thr, leaf))


def fit_rf(x,y,y_errors=None,fname_export='nn_fit',device='cpu',backend='sklearn',seed=None):
#    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    # Instantiate model. Usually not that many structures to find, don't overcomplicate
    #   - should scale like number of samples
    if backend == 'cupy':
        from RIFT.interpolators.cupy_extratrees import CupyExtraTreesRegressor
        sw = None if y_errors is None else 1./np.maximum(np.asarray(y_errors,dtype=float),1e-3)**2
        keep = np.all(np.isfinite(x), axis=-1) & np.all(~(np.abs(x) > 1e37), axis=-1)
        if not np.all(keep):
            # sklearn fits such rows as missing values; the GPU grower does not support them
            print(" rf cupy fit: dropping {} of {} rows with nonfinite or |x| > 1e37 coordinates".format(int(np.sum(~keep)), len(keep)))
        model = CupyExtraTreesRegressor(n_estimators=100, verbose=True, random_state=seed).fit(
            x[keep], y[keep], sample_weight=None if sw is None else sw[keep])
        if seed is not None:
            _rf_fingerprint([e.tree_ for e in model.estimators_], seed)
        fn_return = _device_forest_fit(model.forest(release=True), None, fill=rf_nonfinite_fill)
        residuals = fn_return(x[keep]) - y[keep]
        print( " Demonstrating RF (cupy fit)")
        print( "    std ", np.std(residuals), np.max(y), np.max(fn_return(x)))
        return fn_return
    rf = ExtraTreesRegressor(n_estimators=100, verbose=True,n_jobs=-1,random_state=seed)
    if y_errors is None:
        rf.fit(x,y)
    else:
        # floor sigma so a zero error (placeholder rows can leak into the
        # accumulated marg net with sigma=0) doesn't make sample_weight=1/sigma^2
        # infinite (sklearn rejects inf sample_weight).
        rf.fit(x,y,sample_weight=1./np.maximum(np.asarray(y_errors,dtype=float),1e-3)**2)
    if seed is not None:
        _rf_fingerprint([e.tree_ for e in rf.estimators_], seed)

    ### reject points with infinities : problems for inputs
    def fn_return(x_in,rf=rf):
        f_out = rf_nonfinite_fill*np.ones(len(x_in))
        # remove infinity or Nan
        indx_ok = np.all(np.isfinite(np.array(x_in,dtype=float)),axis=-1)
        # rf internally uses float32, so we need to remove points > 10^37 or so ! 
        #    ... this *should* never happen due to bounds constraints, but ...
        indx_ok_size = np.all( np.logical_not(np.greater(np.abs(x_in),1e37)), axis=-1)
        indx_ok = np.logical_and(indx_ok, indx_ok_size)
        if np.any(indx_ok):      # sklearn rejects an empty batch
            f_out[indx_ok] = rf.predict(x_in[indx_ok])
        return f_out
#    fn_return = lambda x_in: rf.predict(x_in) 
    if device == 'gpu':
        fn_return = _device_forest_fit(rf, x, fill=rf_nonfinite_fill)

    print( " Demonstrating RF")   # debugging
    residuals = (fn_return(x) if device == 'gpu' else rf.predict(x))-y
    print( "    std ", np.std(residuals), np.max(y), np.max(fn_return(x)))
    return fn_return





# initialize
dat_mass  = [] 
weights = []
n_params = -1


 ###
 ### Convert data.   RIGHT NOW JUST DOWNSELECTING, no intermediate fitting parameters defined
 ###

# Naive convert: no downselect.
if (supplemental_coordinate_convert ==None):

    # The identity convert_coords below only makes sense when the fit
    # basis equals the MC sampling basis equals a permutation of the
    # data file's columns.  Catch the new "split-basis" misconfiguration
    # early, otherwise the integrator silently feeds samples in
    # low_level_coord_names through an identity into a fit built on
    # coord_names.
    if list(low_level_coord_names) != list(coord_names):
        raise ValueError(
            " EOSPosterior: --parameter-implied / --parameter-nofit make "
            "the fit basis ({coord!r}) differ from the MC sampling basis "
            "({low!r}), but no --supplementary-coordinate-code was "
            "supplied.  The integrator cannot translate between the two "
            "bases without a converter.".format(
                coord=list(coord_names),
                low=list(low_level_coord_names),
            )
        )

    indx_of_orig_names =  np.array([ dat_orig_names.index(coord_names[k]) for k in range(len(coord_names))])
    dat_out = []
    for line in dat:
        dat_here= np.zeros(len(coord_names)+2)
        if line[col_lnL+1] > opts.sigma_cut:
            print("skipping", line)
            continue
        dat_here[:-2] = line[indx_of_orig_names+2]#line[2:len(coord_names)+2]  # modify to use names!
        dat_here[-2] = line[0]
        dat_here[-1] = line[1]
        dat_out.append(dat_here)
    dat_out= np.array(dat_out)

    # Repack data, WHOLE SET
    X =dat_out[:,0:len(coord_names)]
    Y = dat_out[:,-2]
    if np.max(Y)<0 and lnL_shift ==0: 
        lnL_shift  = -100 - np.max(Y)   # force it to be offset/positive -- may help some configurations. Remember our adaptivity is silly.
    Y_err = dat_out[:,-1]
    def convert_coords(x):
        return x

else:
    # Pack data, using coordinate converter. Note later calculations MUST use the converter.
    #
    # Two distinct call sites for the converter, with two different
    # input bases -- this is the change that decouples the fit from
    # the MC sampling basis:
    #
    #   (1) The initial dat->X conversion below feeds rows whose
    #       columns are ordered by dat_orig_names (the data file's
    #       header).  So we pass low_level_coord_names=dat_orig_names
    #       at this site.
    #
    #   (2) The convert_coords closure is what the integrator calls
    #       on every Monte Carlo sample.  The sampler operates in
    #       low_level_coord_names (we add_parameter() over that list
    #       below), so the closure must claim its inputs are in
    #       low_level_coord_names -- NOT dat_orig_names.  Pre-fix this
    #       was hardcoded to dat_orig_names, which only happened to
    #       work when low_level_coord_names == dat_orig_names (i.e.
    #       the legacy case).  For any non-trivial plugin where the
    #       MC samples in a different basis than the file's columns,
    #       the old behaviour applied the rotation an extra time and
    #       silently mis-evaluated lnL.
    if _xpy_convert is not None:
        X = _xpy_convert(dat[:,2:], coord_names, dat_orig_names)
    else:
        X = supplemental_coordinate_convert(dat[:,2:], coord_names=coord_names, low_level_coord_names=dat_orig_names) # convert and generate X
    Y = dat[:,0]
    Y_err = dat[:,1]
    if np.max(Y)<0 and lnL_shift ==0:
        lnL_shift  = -100 - np.max(Y)   # force it to be offset/positive -- may help some configurations. Remember our adaptivity is silly.
    if not _per_sample_needs_plugin:
        # Easy case: the sampling basis contains every fit coordinate, so a
        # Monte Carlo sample is already in the fit basis (up to column
        # selection/order).  Do NOT route per-sample batches through the
        # plugin -- its forward map expects file-basis inputs
        # (INPUT_PARAMETERS), not its own outputs, and would either raise
        # or, worse, silently apply the transform a second time.
        _fit_col_of_sample = np.array([ low_level_coord_names.index(name) for name in coord_names ])
        def convert_coords(x_in, _idx=_fit_col_of_sample):
            return np.asarray(x_in)[:, _idx]
        if opts.fit_device == 'gpu':
            def convert_coords(x_in, _idx=_fit_col_of_sample):
                return x_in[:, _idx]
    elif _xpy_convert is not None:
        def convert_coords(x_in, _low=list(low_level_coord_names), _coord=list(coord_names)):
            return _xpy_convert(x_in, _coord, _low)
    else:
        def convert_coords(x_in, _low=low_level_coord_names, _coord=coord_names):
            # _low / _coord captured as defaults so the closure stays correct
            # even if either list mutates later in the script.
            return supplemental_coordinate_convert(x_in, coord_names=_coord, low_level_coord_names=_low)
        if opts.fit_device == 'gpu':
            # the plugin is host-only: convert a device batch on the host, evaluate the fit on the device
            print(" FIT-DEVICE gpu : coordinate conversion runs on the host (plugin); the fit is evaluated on the device")
            def convert_coords(x_in, _low=low_level_coord_names, _coord=coord_names):
                import cupy as _cp
                if isinstance(x_in, _cp.ndarray):
                    return _cp.asarray(supplemental_coordinate_convert(_cp.asnumpy(x_in), coord_names=_coord, low_level_coord_names=_low))
                return supplemental_coordinate_convert(x_in, coord_names=_coord, low_level_coord_names=_low)
# Save copies for later (plots)
X_orig = X.copy()
Y_orig = Y.copy()



# Eliminate values with Y too small
max_lnL = np.max(Y)
if np.isinf(opts.lnL_offset):
    indx_ok= np.ones(len(Y),dtype=bool)  # default case, we preserve all the data
else:
    indx_ok = np.array(Y>np.max(Y)-opts.lnL_offset,dtype=bool)  # force cast : sometimes indx_ok is a mappable object?
n_ok = np.sum(indx_ok)
# Provide some random points, to insure reasonable tapering behavior away from the sample
print(" Points used in fit : ", n_ok, " out of ", len(indx_ok), " given max lnL ", max_lnL)
if max_lnL < 10 and np.mean(Y) > -10: # second condition to allow synthetic tests not to fail, as these often have maxlnL not large
    print(" Resetting to use ALL input data -- beware ! ")
    # nothing matters, we will reject it anyways
    indx_ok = np.ones(len(Y),dtype=bool)
elif n_ok < 10: # and max_lnL > 30:
    # mark the top 10 elements and use them for fits
    # this may be VERY VERY DANGEROUS if the peak is high and poorly sampled
    idx_sorted_index = np.lexsort((np.arange(len(Y)), Y))  # Sort the array of Y, recovering index values
    indx_list = np.array( [[k, Y[k]] for k in idx_sorted_index])     # pair up with the weights again
    indx_list = indx_list[::-1]  # reverse, so most significant are first
    indx_ok = list(map(int,indx_list[:10,0]))
    print(" Revised number of points for fit: ", np.sum(indx_ok), len(indx_ok), indx_list[:10])
X_raw = X.copy()



my_fit= None
if opts.fit_method =='gp':
    print(" FIT METHOD : GP")
    # some data truncation IS used for the GP, but beware
    print(" Truncating data set used for GP, to reduce memory usage needed in matrix operations")
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    # Cap the total number of points retained, AFTER the threshold cut
    if opts.cap_points< len(Y) and opts.cap_points> 100:
        n_keep = opts.cap_points
        indx = np.random.choice(np.arange(len(Y)),size=n_keep,replace=False)
        Y=Y[indx]
        X=X[indx]
        Y_err=Y_err[indx]
    if opts.ignore_errors_in_data:
        Y_err=None
    my_fit = fit_gp(X,Y,y_errors=Y_err)
elif opts.fit_method == 'rf':
    print( " FIT METHOD ", opts.fit_method, " IS RF ")
    # NO data truncation for NN needed?  To be *consistent*, have the code function the same way as the others
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    # Cap the total number of points retained, AFTER the threshold cut
    if opts.cap_points< len(Y) and opts.cap_points> 100:
        n_keep = opts.cap_points
        indx = np.random.choice(np.arange(len(Y)),size=n_keep,replace=False)
        Y=Y[indx]
        X=X[indx]
        Y_err=Y_err[indx]
    if opts.ignore_errors_in_data:
        Y_err=None
    my_fit = fit_rf(X,Y,y_errors=Y_err,device=opts.fit_device,backend=opts.rf_fit_backend,seed=opts.rf_seed)
    if opts.rf_dslice_tails != "none":
        if 'dist' not in list(coord_names):
            raise ValueError("--rf-dslice-tails needs 'dist' as a fit coordinate (distance-slice grid)")
        if Y_err is None:
            raise ValueError("--rf-dslice-tails needs the per-row sigma column (not --ignore-errors-in-data)")
        from RIFT.interpolators.dslice_rf_tails import RFDistanceTails
        _xp_tails = np
        if opts.fit_device == "gpu":
            import cupy as _xp_tails
        # coord_names: the tail wrapper groups the slices by the intrinsic coordinates, so it needs the
        # names to keep 'inv_dist' (the same axis as 'dist') out of that grouping
        my_fit = RFDistanceTails(my_fit, list(coord_names).index('dist'), sides=opts.rf_dslice_tails,
                                 shared_fmin=(opts.rf_dslice_tails_fmin == "shared"), xp=_xp_tails,
                                 point_fit=opts.dslice_amp_point_fit, n_jobs=opts.dslice_amp_point_fit_jobs,
                                 coord_names=list(coord_names)).fit(X, Y + lnL_shift, Y_err)
        print(" RF-DSLICE-TAILS ", my_fit.report)

elif opts.fit_method == 'gp-matern':
    print(" FIT METHOD gp-matern: bounded standardized Matern5/2")
    # rho1 from the data file's native in-plane spin, aligned with X before any cut
    from RIFT.interpolators.matern_gp import native_rho1
    # rows with nonfinite or negative native spin geometry cannot be stratified; they are dropped below
    _ok_rows = np.all(np.isfinite(dat[:,2:]), axis=1)
    rho1 = native_rho1(dat[_ok_rows,2:],dat_orig_names)
    if rho1 is not None:
        _r = np.full(len(dat), np.nan); _r[_ok_rows] = rho1
        rho1 = _r[indx_ok]
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    if opts.ignore_errors_in_data:
        Y_err = np.zeros_like(Y_err)
    # Rows the coordinate converter cannot map cannot be fit; drop them visibly
    finite = np.all(np.isfinite(X),axis=1) & np.isfinite(Y) & np.isfinite(Y_err)
    if rho1 is not None:
        finite &= np.isfinite(rho1)
    if not np.all(finite):
        print(" gp-matern: dropping {} of {} training rows with nonfinite fit coordinates/targets".format(int(np.sum(~finite)),len(finite)))
        X, Y, Y_err = X[finite], Y[finite], Y_err[finite]
        if rho1 is not None:
            rho1 = rho1[finite]
    my_fit = fit_gp_matern(X,Y,Y_err,rho1=rho1)

elif opts.fit_method == 'dslice-amp':
    print(" FIT METHOD dslice-amp: per-point averaged-amplitude distance model, fields interpolated by ExtraTrees")
    if 'dist' not in list(coord_names):
        raise ValueError("--fit-method dslice-amp needs 'dist' as a fit coordinate (distance-export grid)")
    from RIFT.interpolators.dslice_amplitude_model import DistanceAmplitudeModel
    X=X[indx_ok]
    Y=Y[indx_ok]            # UNshifted: the model's d -> infinity limit lnL -> 0 is physical
    Y_err = Y_err[indx_ok]
    finite = np.all(np.isfinite(X),axis=1) & np.isfinite(Y) & np.isfinite(Y_err)
    if opts.fit_load_gp:
        dslice_model = joblib.load(opts.fit_load_gp)
        _saved = getattr(dslice_model, 'coord_names', None)
        if _saved is not None and list(_saved) != list(coord_names):
            raise ValueError("--fit-load-gp: dslice-amp model was fit in coordinates {} but this run uses {}".format(list(_saved), list(coord_names)))
    else:
        _mi = list(coord_names).index('mtot') if 'mtot' in list(coord_names) else None   # mass-scaled distance field when mtot is a fit coordinate
        _kw = {}
        if opts.dslice_amp_decompose:
            # marginal/conditional split under the run's own distance prior and integration range
            _kw = dict(prior=prior_map['dist'], d_range=param_ranges['dist'])
        if opts.dslice_amp_point_fit == 'batched':
            _kw.update(point_fit='batched', xp=_cupy() if opts.fit_device == 'gpu' else np)
        elif opts.dslice_amp_point_fit_jobs > 1:
            _kw.update(point_fit_jobs=opts.dslice_amp_point_fit_jobs)
        dslice_model = DistanceAmplitudeModel(list(coord_names).index('dist'),mass_index=_mi,
                                              loss=opts.dslice_amp_loss,f_scale=opts.dslice_amp_loss_scale,
                                              random_state=opts.rf_seed,
                                              **_kw).fit(X[finite],Y[finite],Y_err[finite])
        dslice_model.coord_names = list(coord_names)
        if opts.fit_save_gp:
            joblib.dump(dslice_model,opts.fit_save_gp+".pkl")
    print(" dslice-amp report ", dslice_model.report)
    Y = Y - lnL_shift
    def my_fit(x_in, _m=dslice_model, _shift=lnL_shift):
        x_in = np.asarray(x_in,dtype=np.float64)
        f_out = lnL_default_large_negative*np.ones(len(x_in))
        ok = np.all(np.isfinite(x_in),axis=-1) & np.all(np.abs(x_in) < 1e37,axis=-1)
        if np.any(ok):
            f_out[ok] = _m.predict(x_in[ok]) - _shift
        return f_out
    if opts.fit_device == 'gpu':
        _cp = _cupy()
        _xc = X[finite][np.linspace(0, int(np.sum(finite)) - 1, min(int(np.sum(finite)), 4096)).astype(int)]
        _ok, _rep = dslice_model.check_device(_xc)
        print(" FIT-DEVICE gpu : dslice-amp device predict vs cpu on training rows {}".format(_rep))
        if not _ok:
            raise RuntimeError("--fit-device gpu: dslice-amp device predict disagrees with the CPU model ({})".format(_rep))
        _host_fit = my_fit
        def my_fit(x_in, _m=dslice_model, _shift=lnL_shift, _host_fit=_host_fit):
            if not isinstance(x_in, _cp.ndarray):
                return _host_fit(x_in)
            x_in = x_in.astype(_cp.float64)
            ok = _cp.all(_cp.isfinite(x_in),axis=-1) & _cp.all(_cp.abs(x_in) < 1e37,axis=-1)
            return _cp.where(ok, _m.predict_device(_cp.where(ok[:, None], x_in, 1.0)) - _shift, float(lnL_default_large_negative))

if my_fit is None:
    # This driver builds only 'gp', 'rf', 'gp-matern' and 'dslice-amp'.  The --fit-method help was copied from
    # util_ConstructIntrinsicPosterior_GenericCoordinates.py, which implements a dozen more, so
    # asking for one of those here left my_fit at None and the run continued: nothing referenced
    # it until the sampler evaluated the integrand, which then died with
    # "TypeError: 'NoneType' object is not callable" from inside log_likelihood_function, after
    # the whole setup had been paid for and with nothing naming --fit-method.
    print(" OPTION MISMATCH : --fit-method {} is not implemented in this driver; it builds only 'gp', 'rf', 'gp-matern' and 'dslice-amp'.  (util_ConstructIntrinsicPosterior_GenericCoordinates.py implements the others.)".format(opts.fit_method))
    sys.exit(99)

### Distance tail: make the fit decay beyond each intrinsic point's exported distance support
###
### Only meaningful for a distance-export (.dslice) run, where `dist` is a FIT coordinate and the
### training set is ~50 discrete distances per intrinsic point.  An RF/ExtraTrees fit is piecewise
### constant outside its training envelope, so past a point's outermost exported slice it returns
### that slice's lnL forever.  The distance prior is volumetric and keeps growing like d^2, so the
### integrand grows instead of dying and the recovered distance posterior comes out ~18% too wide
### in every quantile span, median untouched.  See RIFT/interpolators/distance_tail.py.
###
### This wraps whatever fit was just built and is a no-op on the support, so nothing that currently
### works changes.  It is applied AFTER the cap/threshold cuts above so the tail is built from
### exactly the rows the fit itself saw.
if opts.fit_distance_tail:
    if my_fit is None:
        raise ValueError("--fit-distance-tail: no fit was built (--fit-method %s)" % opts.fit_method)
    if 'dist' not in list(coord_names):
        # Fail rather than silently do nothing: a run that asked for this and did not get it would
        # carry the very bias the option exists to remove, with no sign of it in the log.
        raise ValueError("--fit-distance-tail requires a distance fit coordinate, but coord_names "
                         "is %s. This option is for distance-export (.dslice) runs." % (list(coord_names),))
    from RIFT.interpolators.distance_tail import wrap_distance_tail
    tail_report = {}
    # The production decay law is the parameter-free chord (ratio = u), so the wrapper's tuning
    # arguments -- the per-slice edge-slope fit and its `outer_frac`, the alternate laws -- reach
    # only the diagnostic branches described in RIFT/interpolators/distance_tail.py. They are
    # deliberately not exposed here: a CLI knob that cannot change the posterior is worse than none.
    my_fit = wrap_distance_tail(my_fit, X, Y, coord_names, y_errors=Y_err,
                                lnL_offset=lnL_shift, report=tail_report)
    print(" DISTANCE TAIL : decay beyond exported support enabled ", tail_report)


# Sort for later convenience (scatterplots, etc)
indx = Y.argsort()#[::-1]
X=X[indx]
Y=Y[indx]



###
### Integrate posterior
###


sampler = mcsampler.MCSampler()
if opts.sampler_method == "adaptive_cartesian_gpu":
    sampler = mcsamplerGPU.MCSampler()
    sampler.xpy = xpy_default
    sampler.identity_convert=identity_convert
    mcsampler  = mcsamplerGPU  # force use of routines in that file, for properly configured GPU-accelerated code as needed

    # if opts.sampler_xpy == "numpy":
    #   mcsampler.set_xpy_to_numpy()
    #   sampler.xpy= numpy
    #   sampler.identity_convert= lambda x: x
if opts.sampler_method == "GMM":
    sampler = mcsamplerEnsemble.MCSampler()
elif opts.sampler_method == "AV":
    sampler = mcsamplerAdaptiveVolume.MCSampler()
    opts.internal_use_lnL= True  # required!
elif opts.sampler_method == "portfolio":
    use_portfolio=True
    opts.internal_use_lnL=True  # required: mcsamplerPortfolio.integrate() refuses to run without use_lnL
    sampler = None
    sampler_list = []
    sampler_types = opts.sampler_portfolio
    if not sampler_types:
        # --sampler-method portfolio with no --sampler-portfolio at all.  The argparse default is
        # None, not [], so without this the next line raises
        # "TypeError: 'NoneType' object is not iterable" -- which names neither option.
        print(" OPTION MISMATCH : --sampler-method portfolio requires at least one --sampler-portfolio NAME (AV|GMM|NFlow|adaptive_cartesian_gpu).")
        sys.exit(99)
    accepted_indices = []   # index into sampler_types of each member that actually got built
    for indx, name in enumerate(sampler_types):
        # Clear the carry-over BEFORE dispatching on the name.  Without this, the
        # "if sampler is None: continue" below cannot do what its comment says: after one
        # recognized name, `sampler` stays bound to that member, so every LATER unrecognized
        # name appends THE SAME OBJECT again -- a portfolio holding one sampler twice, sharing
        # all its adaptation state, silently and with no message.
        sampler = None
        if name =='AV':
            sampler = mcsamplerAdaptiveVolume.MCSampler()
        if name =='GMM':
            sampler = mcsamplerEnsemble.MCSampler()
            opts.sampler_method = 'GMM'  # this will force the creation/parsing of GMM-specific arguments below, so they are properly passed
        if name == "adaptive_cartesian_gpu":
            sampler = mcsamplerGPU.MCSampler()
            sampler.xpy = xpy_default
            sampler.identity_convert=identity_convert
        if name == 'NFlow':
            # expensive import, only do if requested
            try:
                import RIFT.integrators.mcsamplerNFlow as mcsamplerNFlow
                mcsampler_NF_ok = True
            except:
                print(" No mcsamplerNFlow ")
                continue
            sampler = mcsamplerNFlow.MCSampler()
            sampler.xpy = xpy_default
            sampler.identity_convert=identity_convert
        if sampler is None:
            # Don't add unknown type.  Say so: a single mistyped name among several good ones
            # leaves the run to proceed with a SMALLER portfolio than the user asked for, and
            # the guards above only fire when NOTHING matched.
            print(" PORTFOLIO : WARNING, ignoring unrecognized --sampler-portfolio {!r} (known: AV, GMM, NFlow, adaptive_cartesian_gpu)".format(name))
            continue
        print('PORTFOLIO: adding {} '.format(name))
        sampler_list.append(sampler)
        accepted_indices.append(indx)
    if not sampler_list:
        # Every name was unrecognized (note --sampler-portfolio takes ONE name per flag; a
        # comma-joined "AV,GMM" matches nothing).  Same reasoning as the guard above.
        print(" OPTION MISMATCH : --sampler-portfolio matched no known sampler in {}.  Pass one name per flag, e.g. --sampler-portfolio AV --sampler-portfolio GMM.".format(sampler_types))
        sys.exit(99)
    # MIXED-BACKEND PORTFOLIO.  The warning that stood here is gone because the condition it
    # warned about is now checked in mcsamplerPortfolio.setup() -- earlier than here, and for
    # every caller rather than this driver alone.  mcsamplerGPU implements sampling_density(),
    # so the mixed case this driver could produce is correct rather than merely flagged; a
    # member that still lacks one is refused at setup when any member has not DECLARED that its
    # joint_p_s is a normalized density -- mcsamplerAdaptiveVolume declares False, and a member
    # that declares nothing (a plugin, a new sampler) counts as unknown rather than safe.  A pool
    # in which every member declares True is allowed and warned about.
    #
    # The measurements that motivated the warning, re-taken as a BEFORE/AFTER pair on this
    # branch and its base with identical settings (the absolute numbers differ from the earlier
    # revision of this comment because n/neff differ; only a same-settings pair is comparable).
    # Constant integrand, non-square box, prior integral 5, exact ln Z = 1.609438, adaptation
    # running, 16 seeds, ldas-grid numpy backend:
    #
    #     portfolio          before (94f352ad8)        after
    #     [AV]               1.6094 (sd 0.0000)        1.6094 (sd 0.0000)    unchanged
    #     [GPU]              1.6134 (sd 0.1158)        1.6134 (sd 0.1158)    unchanged
    #     [GMM]              1.6207 (sd 0.0093)        1.6207 (sd 0.0093)    unchanged
    #     [GPU,GPU]          1.5891 (sd 0.0691)        1.5891 (sd 0.0691)    unchanged
    #     [AV,GMM]           1.6092 (sd 0.0026)        1.6092 (sd 0.0026)    unchanged
    #     [GPU,GMM]          1.6165 (sd 0.0093)        1.6108 (sd 0.0151)    err +0.0071 -> +0.0013
    #     [AV,GPU]           0.4458 (sd 0.0326)        1.6105 (sd 0.0027)    err -1.16 -> +0.001
    #     [GPU,AV]           0.4330 (sd 0.0231)        1.6090 (sd 0.0028)    err -1.18 -> -0.000
    #
    # Read the [GPU,GMM] row as a wash, not a win: both members already reported a normalized
    # joint_p_s, so the stratified fallback was unbiased there and the earlier warning
    # over-fired.  What changes is that q_mix is now formed, which moves the mean 0.006 closer
    # to exact and raises the seed-to-seed spread from 0.009 to 0.015.  Only the two AV+GPU
    # orderings were actually broken.
    #
    # The earlier comment recorded that AV was the outlier on scale -- E[prior/p_s] = 0.2 where
    # mcsamplerGPU and mcsamplerEnsemble both report 5 -- and said the reason was not
    # established.  It is mcsamplerAdaptiveVolume.draw_simplified reporting ps = V_s/V with V
    # the live volume FRACTION (1 before any contraction), so the reported value is V_s**2 times
    # the density the points actually come from, 1/(V_s*V).  AV's own integrate_log is written
    # against that scale, so it is left alone; its sampling_density() already returns the
    # density, and that method is the portfolio contract.  Pinned by
    # test/integrators/test_portfolio_member_density.py::test_AV_reported_p_s_is_left_alone.
    #
    # Two things the old warning noted as gaps are closed by the refusal rather than by a better
    # predicate: a portfolio in which EVERY member lacks sampling_density took the same fallback
    # without warning, and q_mix not being formed also left credit_per_sample identically zero,
    # silently killing the draw-allocation signal under portfolio_adaptive_alloc=True with
    # portfolio_quality_signal='credit'.
    sampler = mcsamplerPortfolio.MCSampler(portfolio=sampler_list)


# Does this sampler hand back ln(integral), or the integral itself?
#
# --internal-use-lnL alone does NOT answer that.  'use_lnL'/'return_lnI' ride in as
# **kwargs to integrate(), and several samplers ignore one or both of them:
#   mcsamplerAdaptiveVolume, mcsamplerNFlow, mcsamplerPortfolio
#                     -- integrate() ALWAYS delegates to integrate_log(); the return is
#                        ln(integral) whether or not the flags were passed
#   mcsamplerGPU      -- delegates to integrate_log() only when use_lnL is set
#   mcsamplerEnsemble -- the only sampler that reads return_lnI
#   mcsampler         -- the legacy 'adaptive_cartesian' backend: reads NEITHER flag,
#                        and always returns the linear integral
# Key off the class actually constructed above rather than off --sampler-method: the
# portfolio branch rebinds sampler, and an unrecognized --sampler-method silently falls
# through to the default mcsampler.
_sampler_module = type(sampler).__module__.split('.')[-1]
if _sampler_module in ['mcsamplerAdaptiveVolume', 'mcsamplerNFlow', 'mcsamplerPortfolio']:
    sampler_returns_ln_integral = True
elif _sampler_module in ['mcsamplerGPU', 'mcsamplerEnsemble']:
    sampler_returns_ln_integral = bool(opts.internal_use_lnL)  # exactly when use_lnL/return_lnI are set below
else:  # mcsampler, the legacy adaptive_cartesian backend
    sampler_returns_ln_integral = False
    if opts.internal_use_lnL:
        # This backend ignores use_lnL, so it would integrate lnL as though it were L:
        # the returned value is not an evidence in either convention, and neither
        # res nor log(res) can repair it.  Refuse rather than write a meaningless
        # number to --fname-output-integral.  Compare ok_lnL_methods/bad_lnL_methods
        # in util_ConstructIntrinsicPosterior_GenericCoordinates.py.
        print(" OPTION MISMATCH : --internal-use-lnL needs a sampler that honors it; --sampler-method {} builds {}, which ignores it.  Use GMM, AV, adaptive_cartesian_gpu, or portfolio.".format(opts.sampler_method, _sampler_module))
        sys.exit(99)


##
## Loop over param names
##
# IMPORTANT: iterate over low_level_coord_names, not coord_names.  The
# sampler operates in the MC basis.  coord_names is the FIT basis, which
# only the GP/RF and the convert_coords closure see.  Pre-decoupling this
# loop used coord_names because the two lists were forced to be equal.
for p in low_level_coord_names:
    prior_here = prior_map[p]
    range_here = prior_range_map[p]

    sampler.add_parameter(p, pdf=np.vectorize(lambda x:1), prior_pdf=prior_here,left_limit=range_here[0],right_limit=range_here[1],adaptive_sampling=True)

likelihood_function = None
log_likelihood_function = None
def log_likelihood_function(*args):
    return my_fit(convert_coords(np.array([*args]).T ))
if opts.fit_device == 'gpu':
    # sample batches stay on the device the sampler hands them in (AV: cupy when a GPU is visible)
    _cp_mod = _cupy()
    def log_likelihood_function(*args):
        _xp = _cp_mod if any(isinstance(a, _cp_mod.ndarray) for a in args) else np
        return my_fit(convert_coords(_xp.stack([_xp.asarray(a, dtype=_xp.float64) for a in args], axis=1)))

# Fixed-arity wrappers around log_likelihood_function / likelihood_function.
#
# mcsampler's adaptive code introspects the wrapped function's argument
# count, so we generate one definition per supported dimensionality.  The
# arity that matters is the MC SAMPLING dimensionality
# (len(low_level_coord_names)), not the fit dimensionality
# (len(coord_names)) -- the sampler passes one positional per
# low_level_coord_name and then convert_coords maps that batch into the
# fit basis.  Pre-decoupling, the dispatch keyed on len(coord_names),
# which was only correct when low_level_coord_names == coord_names.
#
# Scalar branches also go through convert_coords so a non-trivial
# converter does not silently get bypassed when an internal caller hands
# in a single scalar sample.
def _scalar_to_lnL_input(args_tuple):
    # Wrap an N-tuple of scalars into a (1, N) row, push it through the
    # converter, and return -- shape (1, len(coord_names)) -- ready for my_fit.
    return convert_coords(np.array([args_tuple], dtype=float))

_LN_LOW_DIM = len(low_level_coord_names)
if _LN_LOW_DIM == 1:
    def likelihood_function(x):
        if isinstance(x, float):
            return np.exp(my_fit(_scalar_to_lnL_input((x,))))
        return np.exp(my_fit(convert_coords(np.c_[x])))
elif _LN_LOW_DIM == 2:
    def likelihood_function(x, y):
        if isinstance(x, float):
            return np.exp(my_fit(_scalar_to_lnL_input((x, y))))
        return np.exp(my_fit(convert_coords(np.c_[x, y])))
elif _LN_LOW_DIM == 3:
    def likelihood_function(x, y, z):
        if isinstance(x, float):
            return np.exp(my_fit(_scalar_to_lnL_input((x, y, z))))
        return np.exp(my_fit(convert_coords(np.c_[x, y, z])))
elif _LN_LOW_DIM == 4:
    def likelihood_function(x, y, z, a):
        if isinstance(x, float):
            return np.exp(my_fit(_scalar_to_lnL_input((x, y, z, a))))
        return np.exp(my_fit(convert_coords(np.c_[x, y, z, a])))
elif _LN_LOW_DIM == 5:
    def likelihood_function(x, y, z, a, b):
        if isinstance(x, float):
            return np.exp(my_fit(_scalar_to_lnL_input((x, y, z, a, b))))
        return np.exp(my_fit(convert_coords(np.c_[x, y, z, a, b])))
elif _LN_LOW_DIM == 6:
    def likelihood_function(x, y, z, a, b, c):
        if isinstance(x, float):
            return np.exp(my_fit(_scalar_to_lnL_input((x, y, z, a, b, c))))
        return np.exp(my_fit(convert_coords(np.c_[x, y, z, a, b, c])))
elif _LN_LOW_DIM == 7:
    def likelihood_function(x, y, z, a, b, c, d):
        if isinstance(x, float):
            return np.exp(my_fit(_scalar_to_lnL_input((x, y, z, a, b, c, d))))
        return np.exp(my_fit(convert_coords(np.c_[x, y, z, a, b, c, d])))
elif _LN_LOW_DIM == 8:
    def likelihood_function(x, y, z, a, b, c, d, e):
        if isinstance(x, float):
            return np.exp(my_fit(_scalar_to_lnL_input((x, y, z, a, b, c, d, e))))
        return np.exp(my_fit(convert_coords(np.c_[x, y, z, a, b, c, d, e])))
elif _LN_LOW_DIM == 9:
    def likelihood_function(x, y, z, a, b, c, d, e, f):
        if isinstance(x, float):
            return np.exp(my_fit(_scalar_to_lnL_input((x, y, z, a, b, c, d, e, f))))
        return np.exp(my_fit(convert_coords(np.c_[x, y, z, a, b, c, d, e, f])))
elif _LN_LOW_DIM == 10:
    def likelihood_function(x, y, z, a, b, c, d, e, f, g):
        if isinstance(x, float):
            return np.exp(my_fit(_scalar_to_lnL_input((x, y, z, a, b, c, d, e, f, g))))
        return np.exp(my_fit(convert_coords(np.c_[x, y, z, a, b, c, d, e, f, g])))
else:
    raise NotImplementedError(
        " EOSPosterior currently only ships fixed-arity likelihood_function "
        "wrappers for 1..10 sampling dimensions; got "
        "{} (low_level_coord_names={!r}).".format(_LN_LOW_DIM, low_level_coord_names)
    )




n_step = opts.n_step
my_exp = np.min([1,0.8*np.log(n_step)/np.max(Y)])   # target value : scale to slightly sublinear to (n_step)^(0.8) for Ymax = 200. This means we have ~ n_step points, with peak value wt~ n_step^(0.8)/n_step ~ 1/n_step^(0.2), limiting contrast
if np.max(Y_orig) < 0:   # for now, don't use a weight exponent if we are negative: can't use guess based from GW experience
    my_exp = 1
if not (my_exp > 0):
    # The ratio above divides by max(Y), which is lnL_shift-shifted; the Y_orig test is not.
    # --lnL-shift-prevent-overflow above the peak lnL makes max(Y)<0 and the exponent negative,
    # which adapts AWAY from the peak.  Fall back to the same "don't use it" value as above.
    my_exp = 1
#my_exp = np.max([my_exp,  1/np.log(n_step)]) # do not allow extreme contrast in adaptivity, to the point that one iteration will dominate
print(" Weight exponent ", my_exp, " and peak contrast (exp)*lnL = ", my_exp*np.max(Y), "; exp(ditto) =  ", np.exp(my_exp*np.max(Y)), " which should ideally be no larger than of order the number of trials in each epoch, to insure reweighting doesn't select a single preferred bin too strongly.  Note also the floor exponent also constrains the peak, de-facto")


extra_args={}
if opts.sampler_method == "GMM":
    n_max_blocks = ((1.0*int(opts.n_max))/n_step) 
    n_comp = opts.internal_n_comp # default
    def parse_corr_params(my_str):
        """
        Takes a string with no spaces, and returns a tuple
        """
        corr_param_names = my_str.replace(',',' ').split()
        corr_param_indexes = []
        for param in corr_param_names:
            try:
                indx = low_level_coord_names.index(param)
                corr_param_indexes.append(indx)
            except:
                continue
        return tuple(corr_param_indexes)
    if opts.internal_correlate_parameters == 'all':
        gmm_dict = {tuple(range(len(low_level_coord_names))):None} # integrate *jointly* in all parameters together
    elif not (opts.internal_correlate_parameters is None):
        # Correlate identified parameters
        my_blocks = opts.internal_correlate_parameters.split()
        my_tuples = list(map( parse_corr_params, my_blocks))
        gmm_dict = {x:None for x in my_tuples}
        print(" GMM: Proposed correlated ", gmm_dict)
        # What about un-labelled parameters? Make a null tuple for them as well
        correlated_params = set(); correlated_params = correlated_params.union( *list(map(set,my_tuples)))
        uncorrelated_params = set(np.arange(len(low_level_coord_names))); 
        uncorrelated_params = uncorrelated_params.difference(correlated_params)
        for x in uncorrelated_params:
            gmm_dict[(x,)] = None
        print( " Using correlated GMM sampling on sampling variable indexes " , gmm_dict, " out of ", low_level_coord_names)
    else:
        param_indexes = range(len(low_level_coord_names))
        gmm_dict  = {(k,):None for k in param_indexes} # no correlations
#    lnL_offset_saving = opts.lnL_offset
    lnL_offset_saving = -20  # for simplicity, hardcode for now for preserving points
    print("GMM ", gmm_dict)
    extra_args = {'n_comp':n_comp,'max_iter':n_max_blocks,'L_cutoff': None,'gmm_dict':gmm_dict,'max_err':50, 'lnw_failure_cut':-np.inf}  # made up for now, should adjust
extra_args.update({
    "n_adapt": 100, # Number of chunks to allow adaption over
    "history_mult": 10, # Multiplier on 'n' - number of samples to estimate marginalized 1D histograms with, 
    "force_no_adapt":opts.force_no_adapt,
    "tripwire_fraction":opts.tripwire_fraction
})

fn_passed = likelihood_function
if supplemental_ln_likelihood:
    fn_passed =  lambda *x: likelihood_function(*x)*np.exp(supplemental_ln_likelihood(*x))
if opts.internal_use_lnL:
    fn_passed = log_likelihood_function   # helps regularize large values
    if supplemental_ln_likelihood:
        fn_passed =  lambda *x: log_likelihood_function(*x) + supplemental_ln_likelihood(*x)
        if opts.fit_device == 'gpu':
            # the supplementary factor is host code: add it on the host
            def fn_passed(*x):
                _h = [a.get() if hasattr(a, 'get') else a for a in x]
                _l = log_likelihood_function(*x)
                return (_l.get() if hasattr(_l, 'get') else _l) + supplemental_ln_likelihood(*_h)
    extra_args.update({"use_lnL":True,"return_lnI":True})


# PORTFOLIO: setup() is MANDATORY, not an optimization.  It is the only place that
#   (a) initializes self.portfolio_breakpoints -- left at None by __init__, so the first
#       draw() evaluates `None <= iteration` and dies with
#       "TypeError: '<=' not supported between instances of 'NoneType' and 'int'"; and
#   (b) calls setup() on each MEMBER, which is what builds AV's my_ranges/dx/V_s and the
#       GMM integrator.  A member that never got setup() is cold and cannot draw.
# Compare util_ConstructIntrinsicPosterior_GenericCoordinates.py, which calls setup() at the
# same point in its own flow.  Key off the CLASS actually constructed rather than off
# --sampler-method: the portfolio branch above rebinds opts.sampler_method to 'GMM' whenever a
# GMM member is requested, so `opts.sampler_method == "portfolio"` is False by the time we get
# here for exactly the configurations that need this most.
if _sampler_module == 'mcsamplerPortfolio':
    print(" PORTFOLIO : setup")
    portfolio_args = None
    if opts.sampler_portfolio_args:
        # One eval-able dict per portfolio member, in --sampler-portfolio order.
        portfolio_args = list(map(eval, opts.sampler_portfolio_args))
        for indx, arg in enumerate(portfolio_args):
            if not isinstance(arg, dict):
                print(" OPTION MISMATCH : --sampler-portfolio-args entry {} is not a dict: {}".format(indx, arg))
                sys.exit(99)
        # Check the count against the NAMES THE USER ASKED FOR, not against the members that
        # survived.  An optional member can drop out for reasons that are not the user's fault --
        # the NFlow branch above `continue`s when nflows is not installed -- and counting
        # survivors would turn a portfolio that runs on one host into an exit 99 on another,
        # blaming the argument count for a missing dependency.
        if len(portfolio_args) != len(sampler_types):
            print(" OPTION MISMATCH : {} --sampler-portfolio-args entries for {} --sampler-portfolio names; they must correspond one-to-one.".format(len(portfolio_args), len(sampler_types)))
            sys.exit(99)
        # Now drop the entries belonging to members that were not built, so what reaches setup()
        # still lines up with the member list.  setup() only prints "PORTFOLIO - format ERROR"
        # and silently discards ALL of them on a length mismatch, so an unnoticed misalignment
        # would leave every member untuned with the run continuing.
        portfolio_args = [portfolio_args[i] for i in accepted_indices]
        print(" PORTFOLIO ARGS ", portfolio_args)
    # NOTE the spelling: mcsamplerPortfolio.setup() reads kwargs['portfolio_args'].  It takes
    # **kwargs, so a misspelled name is accepted and silently ignored rather than raising.
    sampler.setup(portfolio_args=portfolio_args, portfolio_allow_stratified_density=opts.sampler_portfolio_allow_stratified_density, **extra_args)


res, var, neff, dict_return = sampler.integrate(fn_passed, *low_level_coord_names,  verbose=True,nmax=int(opts.n_max),n=n_step,neff=opts.n_eff, save_intg=True,tempering_adapt=True, floor_level=1e-3,igrand_threshold_p=1e-3,convergence_tests=test_converged,tempering_exp=my_exp,no_protect_names=True,**extra_args)  # MC integrates in the SAMPLING basis (low_level_coord_names); convert_coords routes each sample into the fit basis (coord_names) before evaluating the GP/RF

# result value:  be careful, if the sampler returns lnI, then must not take log twice!
# See sampler_returns_ln_integral where the sampler is constructed: this is a property of
# the backend, NOT of --internal-use-lnL.
ln_integrand_value = None
if sampler_returns_ln_integral:
    ln_integrand_value = res
else:
    ln_integrand_value = np.log(res)

# Save result -- needed for odds ratios, etc.
# Absolute scale: a supplementary-likelihood plugin may subtract a constant from its own lnL to keep
# the exponentiated integrand in float64 range (see the <function>_offset note at the import above).
# That constant divides out of the posterior but not out of an evidence -- which is exactly what
# this file is read as -- so it is restored here.  Queried now rather than next to the prepare call
# because a plugin configured entirely by environment prepares itself lazily on its first
# evaluation.  Stays 0.0 for plugins that do not centre and for runs with no supplementary factor.
if supplemental_ln_likelihood_offset_fn:
    supplemental_ln_likelihood_offset = float(supplemental_ln_likelihood_offset_fn())
    print(" EXTERNAL SUPPLEMENTARY LIKELIHOOD FACTOR : restoring offset {} in reported evidence ".format(supplemental_ln_likelihood_offset))
# Every fit evaluates lnL - lnL_shift (--lnL-shift-prevent-overflow, or the shift applied automatically
# when every lnL is negative), so the integral is ln Z - lnL_shift. Restore it, as CIP does.
ln_integrand_value_absolute = ln_integrand_value + supplemental_ln_likelihood_offset + lnL_shift
np.savetxt(opts.fname_output_integral, [ln_integrand_value_absolute])

if neff < len(coord_names):
    print(" PLOTS WILL FAIL ")
    print(" Not enough independent Monte Carlo points to generate useful contours")


samples = sampler._rvs
print(samples.keys())
# sampler._rvs is keyed by the SAMPLING basis (low_level_coord_names).
# Look up sample arrays by names that actually exist in the dict.
n_params = len(low_level_coord_names)
dat_mass = np.zeros((len(samples[low_level_coord_names[0]]),n_params+3))
if not(opts.internal_use_lnL):
    dat_logL = np.log(samples["integrand"])
else:
    if 'log_integrand' in samples:
        dat_logL = samples['log_integrand']
    else:
        dat_logL = samples["integrand"]
lnLmax = np.max(dat_logL[np.isfinite(dat_logL)])
print(" Max lnL ", np.max(dat_logL), " (fit scale: lnL - lnL_shift, lnL_shift = {})".format(lnL_shift))

n_ESS = -1
if True:
    # Compute n_ESS.  Should be done by integrator!
    if 'log_joint_s_prior' in  samples:
        weights_scaled = np.exp(dat_logL - lnLmax + samples["log_joint_prior"] - samples["log_joint_s_prior"])
        # dictionary, write this to enable later use of it
        samples["joint_s_prior"] = np.exp(samples["log_joint_s_prior"])
        samples["joint_prior"] = np.exp(samples["log_joint_prior"])
    else:
        weights_scaled = np.exp(dat_logL - lnLmax)*sampler._rvs["joint_prior"]/sampler._rvs["joint_s_prior"]
    weights_scaled = weights_scaled/np.max(weights_scaled)  # try to reduce dynamic range
    n_ESS = np.sum(weights_scaled)**2/np.sum(weights_scaled**2)
    print(" n_eff n_ESS ", neff, n_ESS)


# Throw away stupid points that don't impact the posterior
indx_ok = np.ones(len(dat_logL),dtype=bool)
if not('log_joint_s_prior' in samples):
    indx_ok=samples["joint_s_prior"]>0
indx_ok = np.logical_and(dat_logL > np.max(dat_logL)-opts.lnL_offset ,indx_ok)
# Mask in the sampling basis -- samples dict is keyed by low_level_coord_names.
for p in low_level_coord_names:
    samples[p] = samples[p][indx_ok]
dat_logL  = dat_logL[indx_ok]
print(samples.keys())
samples["joint_prior"] =samples["joint_prior"][indx_ok]
samples["joint_s_prior"] =samples["joint_s_prior"][indx_ok]



###
### 1d posteriors of the coordinates used for sampling  [EQUALLY WEIGHTED, BIASED because physics cuts aren't applied]
###

p = samples["joint_prior"]
ps =samples["joint_s_prior"]
lnL = dat_logL
lnLmax = np.max(lnL)
weights = np.exp(lnL-lnLmax)*p/ps



print(" ---- Subset for posterior samples (and further corner work) --- ")


p_norm = (weights/np.sum(weights))
indx_list = np.random.choice(np.arange(len(weights)), p=p_norm.astype(np.float64),size=opts.n_output_samples)


dat_out = np.zeros( (opts.n_output_samples,2+len(dat_orig_names)) )

# The output file is ALWAYS in the fiducial coordinates dat_orig_names (the
# data file's own basis), regardless of what basis we fit or sampled in.
# Each output column is one of three kinds:
#
#   (1) directly sampled    : its name is in low_level_coord_names -- write
#       the weighted posterior draws for it.
#   (2) transform-covered   : the MC sampled in the plugin's output basis
#       (names NOT in dat_orig_names); apply the plugin's
#       inverse_convert_coordinates to the drawn samples to recover the
#       fiducial columns the transform spans.
#   (3) non-sampled extras  : global constants, derived quantities, nuisance
#       parameters carried in the data file -- fill from input-grid rows
#       selected AT RANDOM (row-coherently, so derived quantities stay
#       consistent across columns within one output row).
_sampled_file_cols   = [name for name in low_level_coord_names if name in name_index_dict]
_sampled_plugin_cols = [name for name in low_level_coord_names if name not in name_index_dict]
_covered_cols = set(_sampled_file_cols)

# (1) directly sampled columns
for name in _sampled_file_cols:
    dat_out[:, name_index_dict[name]] = samples[name][indx_list]

# (2) inverse-transform plugin-basis draws back to fiducial coordinates
if _sampled_plugin_cols:
    if supplemental_coordinate_inverse is None:
        print(" WARNING: sampled coordinate(s) {!r} are not data-file columns and the "
              "coordinate plugin does not define inverse_convert_coordinates; their "
              "posterior information CANNOT be written to the fiducial-coordinate "
              "output file.  Add an inverse to the plugin.".format(_sampled_plugin_cols))
    else:
        _S_plugin = np.column_stack([ samples[name][indx_list] for name in _sampled_plugin_cols ])
        _X_fiducial = np.asarray(
            supplemental_coordinate_inverse(
                _S_plugin,
                coord_names=_sampled_plugin_cols,
                low_level_coord_names=_coord_plugin_in_names,
            ), dtype=float)
        for _j, name in enumerate(_coord_plugin_in_names):
            if name in name_index_dict and name not in _covered_cols:
                dat_out[:, name_index_dict[name]] = _X_fiducial[:, _j]
                _covered_cols.add(name)

# (3) non-sampled extra columns: random input-grid rows (matches the
# rift_O4d capability; random selection rather than truncation avoids the
# bias of taking the first n_output_samples rows of a structured grid, and
# replaces the old duplicate-fill bookkeeping when len(dat) is short).
_extra_cols = [name for name in dat_orig_names if name not in _covered_cols]
if _extra_cols:
    print("  Not sampled:", _extra_cols, "; filling output from input-grid rows selected at random.")
    _idx_fill = np.random.choice(np.arange(len(dat)), size=opts.n_output_samples,
                                 replace=(len(dat) < opts.n_output_samples))
    for name in _extra_cols:
        outidx = name_index_dict[name]
        dat_out[:, outidx] = dat[_idx_fill, outidx]

# NOTE: if m1 or m2 is "constant" (i.e., not in samples), the possibility for m2 > m1 arises! Re-sort masses here to avoid; use below code.
#if ("m1" not in coord_names) or ("m2" not in coord_names):
#    print(" NOTE: re-sorting masses so m1 > m2 (precaution)")
#    m1dx = name_index_dict["m1"]
#    m1 = np.maximum(dat_out[:,m1dx], dat_out[:,m1dx+1]) #N.B.: assumes m2 col index after m1 col
#    m2 = np.minimum(dat_out[:,m1dx], dat_out[:,m1dx+1])
#    dat_out[:,m1dx] = m1
#    dat_out[:,m1dx+1] = m2

print(" Saving to ", opts.fname_output_samples+".dat")
np.savetxt(opts.fname_output_samples+".dat",dat_out,header=" lnL sigma_lnL " + ' '.join(dat_orig_names))
