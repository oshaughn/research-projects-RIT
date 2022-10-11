#! /usr/bin/env python

# plot_posterior_corner.py
#
#  --posterior-file f1 --posterior-label n1 --posterior-file f2 --posterior-label n2 ...
#  --parameter p1 --parameter p2 ...
#
# EXAMPLE
#    python plot_posterior_corner.py --posterior-file downloads/TidalP4.dat --parameter lambda1 --parameter lambda2 --parameter mc
#    python plot_posterior_corner.py --parameter mc --parameter eta --posterior-file G298048/production_C00_cleaned_TaylorT4/posterior-samples.dat  --parameter lambdat
#    plot_posterior_corner.py --posterior-file ejecta.dat --parameter mej_dyn --parameter mej_wind --parameter-log-scale mej_dyn --parameter-log-scale mej_wind --change-parameter-label "mej_dyn=m_{\rm ej,d}"  --change-parameter-label "mej_wind=m_{\rm ej,w}"

#
# USAGE
#    - hardcoded list of colors, used in order, for multiple plots
#

import RIFT.lalsimutils as lalsimutils
import RIFT.misc.samples_utils as samples_utils
from RIFT.misc.samples_utils import add_field, extract_combination_from_LI, standard_expand_samples
import lal
import numpy as np
import argparse

eos_param_names = ['logp1', 'gamma1','gamma2', 'gamma3', 'R1_km', 'R2_km']


try:
    import matplotlib
    print(" Matplotlib backend ", matplotlib.get_backend())
    if matplotlib.get_backend() == 'agg':
        fig_extension = '.png'
        bNoInteractivePlots=True
    else:
        matplotlib.use('agg')
        fig_extension = '.png'
        bNoInteractivePlots =True
    from matplotlib import pyplot as plt
    bNoPlots=False
except:
    print(" Error setting backend")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines
import corner

import RIFT.misc.our_corner as our_corner
try:
    import RIFT.misc.bounded_kde as bounded_kde
except:
    print(" -No 1d kdes- ")

dpi_base=200
legend_font_base=16
rc_params = {'backend': 'ps',
             'axes.labelsize': 11,
             'axes.titlesize': 10,
             'font.size': 11,
             'legend.fontsize': legend_font_base,
             'xtick.labelsize': 11,
             'ytick.labelsize': 11,
             #'text.usetex': True,
             'font.family': 'Times New Roman'}#,
             #'font.sans-serif': ['Bitstream Vera Sans']}#,
plt.rcParams.update(rc_params)
plt.rc('axes',unicode_minus=False)

print(" WARNINGS : BoundedKDE class can oversmooth.  Need to edit options for using this class! ")

def render_coord(x,logscale=False):
    if x in lalsimutils.tex_dictionary.keys():
        mystr= lalsimutils.tex_dictionary[x]
        if logscale:
            mystr=mystr.lstrip('$')
            mystr = "$\log_{10}"+mystr
            return mystr
        else:
            return mystr
    if 'product(' in x:
        a=x.replace(' ', '') # drop spaces
        a = a[:len(a)-1] # drop last
        a = a[8:]
        terms = a.split(',')
        exprs =list(map(render_coord, terms))
        exprs = list(map( lambda x: x.replace('$', ''), exprs))
        my_label = ' '.join(exprs)
        return '$'+my_label+'$'
    else:
        if logscale:
            return "log10 "+str(x)
        return x

def render_coordinates(coord_names,logparams=[]):
    print("log params ",logparams)
    return list(map(lambda x: render_coord(x,logscale=(x in logparams)), coord_names))




remap_ILE_2_LI = samples_utils.remap_ILE_2_LI
remap_LI_to_ILE = samples_utils.remap_LI_to_ILE

# def extract_combination_from_LI(samples_LI, p):
#     """
#     extract_combination_from_LI
#       - reads in known columns from posterior samples
#       - for selected known combinations not always available, it will compute them from standard quantities
#     Unike version in ConstructIntrinsicPosterior, this code does not rely on ChooseWaveformParams to perform coordinate changes...
#     """
#     if p in samples_LI.dtype.names:  # e.g., we have precomputed it
#         return samples_LI[p]
#     if p in remap_ILE_2_LI.keys():
#        if remap_ILE_2_LI[p] in samples_LI.dtype.names:
#          return samples_LI[ remap_ILE_2_LI[p] ]
#     # Return cartesian components of spin1, spin2.  NOTE: I may already populate these quantities in 'Add important quantities'
#     if (p == 'chi_eff' or p=='xi') and 'a1z' in samples_LI.dtype.names:
#         m1 = samples_LI['m1']
#         m2 = samples_LI['m2']
#         a1z = samples_LI['a1z']
#         a2z = samples_LI['a2z']
#         return (m1 * a1z + m2*a2z)/(m1+m2)
#     if p == 'chiz_plus':
#         print(" Transforming ")
#         if 'a1z' in samples_LI.dtype.names:
#             return (samples_LI['a1z']+ samples_LI['a2z'])/2.
#         if 'theta1' in samples_LI.dtype.names:
#             return (samples_LI['a1']*np.cos(samples_LI['theta1']) + samples_LI['a2']*np.cos(samples_LI['theta2']) )/2.
# #        return (samples_LI['a1']+ samples_LI['a2'])/2.
#     if p == 'chiz_minus':
#         print(" Transforming ")
#         if 'a1z' in samples_LI.dtype.names:
#             return (samples_LI['a1z']- samples_LI['a2z'])/2.
#         if 'theta1' in samples_LI.dtype.names:
#             return (samples_LI['a1']*np.cos(samples_LI['theta1']) - samples_LI['a2']*np.cos(samples_LI['theta2']) )/2.
# #        return (samples_LI['a1']- samples_LI['a2'])/2.
#     if  'theta1' in samples_LI.dtype.names:
#         if p == 's1x':
#             return samples_LI["a1"]*np.sin(samples_LI[ 'theta1']) * np.cos( samples_LI['phi1'])
#         if p == 's1y' :
#             return samples_LI["a1"]*np.sin(samples_LI[ 'theta1']) * np.sin( samples_LI['phi1'])
#         if p == 's2x':
#             return samples_LI["a2"]*np.sin(samples_LI[ 'theta2']) * np.cos( samples_LI['phi2'])
#         if p == 's2y':
#             return samples_LI["a2"]*np.sin(samples_LI[ 'theta2']) * np.sin( samples_LI['phi2'])
#         if p == 'chi1_perp' :
#             return samples_LI["a1"]*np.sin(samples_LI[ 'theta1']) 
#         if p == 'chi2_perp':
#             return samples_LI["a2"]*np.sin(samples_LI[ 'theta2']) 
#     elif  'tilt1' in samples_LI.dtype.names:
#         if p == 'chi1_perp' :
#             return samples_LI["a1"]*np.sin(samples_LI[ 'tilt1']) 
#         if p == 'chi2_perp':
#             return samples_LI["a2"]*np.sin(samples_LI[ 'tilt2']) 
#     else:  # aligned
#         if p == 'chi1_perp' :
#             return np.zeros(len(samples_LI["m1"])) 
#         if p == 'chi2_perp':
#             return np.zeros(len(samples_LI["m1"])) 

#     if 'lambdat' in samples_LI.dtype.names:  # LI does sampling in these tidal coordinates
#         lambda1, lambda2 = lalsimutils.tidal_lambda_from_tilde(samples_LI["m1"], samples_LI["m2"], samples_LI["lambdat"], samples_LI["dlambdat"])
#         if p == "lambda1":
#             return lambda1
#         if p == "lambda2":
#             return lambda2
#     if p == 'delta' or p=='delta_mc':
#         return (samples_LI['m1']  - samples_LI['m2'])/((samples_LI['m1']  + samples_LI['m2']))
#     # Return cartesian components of Lhat
#     if p == 'product(sin_beta,sin_phiJL)':
#         return np.sin(samples_LI[ remap_ILE_2_LI['beta'] ]) * np.sin(  samples_LI['phi_jl'])
#     if p == 'product(sin_beta,cos_phiJL)':
#         return np.sin(samples_LI[ remap_ILE_2_LI['beta'] ]) * np.cos(  samples_LI['phi_jl'])

#     if p == 'mc':
#         m1v= samples_LI["m1"]
#         m2v = samples_LI["m2"]
#         return lalsimutils.mchirp(m1v,m2v)
#     if p == 'eta':
#         m1v= samples_LI["m1"]
#         m2v = samples_LI["m2"]
#         return lalsimutils.symRatio(m1v,m2v)

#     if p == 'phi1':
#         return np.angle(samples_LI['a1x']+1j*samples_LI['a1y'])
#     if p == 'chi_pavg':
#         samples = np.array([samples_LI["m1"], samples_LI["m2"], samples_LI["a1x"], samples_LI["a1y"], samples_LI["a1z"], samples_LI["a2x"], samples_LI["a2y"], samples_LI["a2z"]]).T
#         with Pool(12) as pool:   
#             chipavg = np.array(pool.map(fchipavg, samples))          
#         return chipavg

#     if p == 'chi_p':
#         samples = np.array([samples_LI["m1"], samples_LI["m2"], samples_LI["a1x"], samples_LI["a1y"], samples_LI["a1z"], samples_LI["a2x"], samples_LI["a2y"], samples_LI["a2z"]]).T
#         with Pool(12) as pool:   
#             chip = np.array(pool.map(fchip, samples))          
#         return chip

#     # Backup : access lambdat if not present
#     if (p == 'lambdat' or p=='dlambdat') and 'lambda1' in samples.dtype.names:
#         Lt,dLt = lalsimutils.tidal_lambda_tilde(samples['m1'], samples['m2'],  samples['lambda1'], samples['lambda2'])
#         if p=='lambdat':
#             return Lt
#         if p=='dlambdat':
#             return dLt

#     if p == "q"  and 'm1' in samples.dtype.names:
#         return samples["m2"]/samples["m1"]

#     print(" No access for parameter ", p, " in ", samples.dtype.names)
#     return np.zeros(len(samples_LI['m1']))  # to avoid causing a hard failure


################################################

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--posterior-file",action='append',help="filename of *.dat file [standard LI output]")
parser.add_argument("--truth-file",type=str, help="file containing the true parameters")
parser.add_argument("--posterior-distance-factor",action='append',help="Sequence of factors used to correct the distances")
parser.add_argument("--truth-event",type=int, default=0,help="file containing the true parameters")
parser.add_argument("--composite-file",action='append',help="filename of *.dat file [standard ILE intermediate]")
parser.add_argument("--use-all-composite-but-grayscale",action='store_true',help="Composite")
parser.add_argument("--flag-tides-in-composite",action='store_true',help='Required, if you want to parse files with tidal parameters')
parser.add_argument("--flag-eos-index-in-composite",action='store_true',help='Required, if you want to parse files with EOS index in composite (and tides)')
parser.add_argument("--posterior-label",action='append',help="label for posterior file")
parser.add_argument("--posterior-color",action='append',help="color and linestyle for posterior. PREPENDED onto default list, so defaults exist")
parser.add_argument("--posterior-linestyle",action='append',help="color and linestyle for posterior. PREPENDED onto default list, so defaults exist")
parser.add_argument("--parameter", action='append',help="parameter name (ILE). Note source-frame masses are only natively supported for LI")
parser.add_argument("--parameter-log-scale",action='append',help="Put this parameter in log scale")
parser.add_argument("--change-parameter-label", action='append',help="format name=string. Will be wrapped in $...$")
parser.add_argument("--use-legend",action='store_true')
parser.add_argument("--use-title",default=None,type=str)
parser.add_argument("--use-smooth-1d",action='store_true')
parser.add_argument("--plot-1d-extra",action='store_true')
parser.add_argument("--pdf",action='store_true',help="Export PDF plots")
#option deprecated by bind-param and param-bound
#parser.add_argument("--mc-range",default=None,help='List for mc range. Default is None')
parser.add_argument("--bind-param",default=None,action="append",help="a parameter to impose a bound on, with corresponding --param-bound arg in respective order")
parser.add_argument("--param-bound",action="append",help="respective bounds for above params")
parser.add_argument("--ci-list",default=None,help='List for credible intervals. Default is 0.95,0.9,0.68')
parser.add_argument("--quantiles",default=None,help='List for 1d quantiles intervals. Default is 0.95,0.05')
parser.add_argument("--chi-max",default=1,type=float)
parser.add_argument("--lambda-plot-max",default=2000,type=float)
parser.add_argument("--lnL-cut",default=None,type=float)
parser.add_argument("--sigma-cut",default=0.4,type=float)
parser.add_argument("--eccentricity", action="store_true", help="Read sample files in format including eccentricity")
opts=  parser.parse_args()
if opts.posterior_file is None:
    print(" No input files ")
    import sys
    sys.exit(0)
if opts.pdf:
    fig_extension='.pdf'

truth_P_list = None
P_ref = None
if opts.truth_file:
    print(" Loading true parameters from  ", opts.truth_file)
    truth_P_list  =lalsimutils.xml_to_ChooseWaveformParams_array(opts.truth_file)
    P_ref = truth_P_list[opts.truth_event]
#    P_ref.print_params()

if opts.change_parameter_label:
  for name, new_str in map( lambda c: c.split("="),opts.change_parameter_label):
      if name in lalsimutils.tex_dictionary:
          lalsimutils.tex_dictionary[name] = "$"+new_str+"$"
      else:
          print(" Assigning new variable string",name,new_str)
          lalsimutils.tex_dictionary[name] = "$"+new_str+"$"  # should be able to ASSIGN NEW NAMES, not restrict

special_param_ranges = {
  'q':[0,1],
  'eta':[0,0.25],
  'a1z':[-opts.chi_max,opts.chi_max],
  'a2z':[-opts.chi_max,opts.chi_max],
  'chi_eff': [-opts.chi_max,opts.chi_max],  # this can backfire for very narrow constraints
  'lambda1':[0,4000],
  'lambda2':[0,4000],
  'chi_pavg':[0,2],
  'chi_p':[0,1],
  'lambdat':[0,4000],
  'eccentricity':[0,1]
}

#mc_range deprecated by generic bind_param
#if opts.mc_range:
#    special_param_ranges['mc'] = eval(opts.mc_range)
#    print(" mc range ", special_param_ranges['mc'])
    
if opts.bind_param:
     for i,par in enumerate(opts.bind_param):
         special_param_ranges[par]=eval(opts.param_bound[i])
         print(par +" range ",special_param_ranges[par])


# Parameters
param_list = opts.parameter

# Legend
color_list=['black', 'red', 'green', 'blue','yellow','C0','C1','C2','C3']
if opts.posterior_color:
    color_list  =opts.posterior_color + color_list
else:
    color_list += len(opts.posterior_file)*['black']
linestyle_list = ['-' for k in color_list]
if opts.posterior_linestyle:
    linestyle_list = opts.posterior_linestyle + linestyle_list
#linestyle_remap_contour  = {":", 'dotted', '-'
posterior_distance_factors = np.ones(len(opts.posterior_file))
if opts.posterior_distance_factor:
    for indx in np.arange(len(opts.posterior_file)):
        posterior_distance_factors[indx] = float(opts.posterior_distance_factor[indx])

line_handles = []
corner_legend_location=None; corner_legend_prop=None
if opts.use_legend and opts.posterior_label:
    n_elem = len(opts.posterior_file)
    for indx in np.arange(n_elem):
        my_line = mlines.Line2D([],[],color=color_list[indx],linestyle=linestyle_list[indx],label=opts.posterior_label[indx])
        line_handles.append(my_line)

    corner_legend_location=(0.7, 1.0)
    corner_legend_prop = {'size':8}
# https://stackoverflow.com/questions/7125009/how-to-change-legend-size-with-matplotlib-pyplot
#params = {'legend.fontsize': 20, 'legend.linewidth': 2}
#plt.rcParams.update(params)


# Import
posterior_list = []
posteriorP_list = []
label_list = []
# Load posterior files
if opts.posterior_file:
 for fname in opts.posterior_file:
    samples = np.genfromtxt(fname,names=True,replace_space=None)  # don't replace underscores in names
    samples = standard_expand_samples(samples)
    # if not 'mtotal' in samples.dtype.names and 'mc' in samples.dtype.names:  # raw LI samples use 
    #     q_here = samples['q']
    #     eta_here = q_here/(1+q_here)
    #     mc_here = samples['mc']
    #     mtot_here = mc_here / np.power(eta_here, 3./5.)
    #     m1_here = mtot_here/(1+q_here)
    #     samples = add_field(samples, [('mtotal', float)]); samples['mtotal'] = mtot_here
    #     samples = add_field(samples, [('eta', float)]); samples['eta'] = eta_here
    #     samples = add_field(samples, [('m1', float)]); samples['m1'] = m1_here
    #     samples = add_field(samples, [('m2', float)]); samples['m2'] = mtot_here * q_here/(1+q_here)

    # if (not 'theta1' in samples.dtype.names)  and ('a1x' in samples.dtype.names):  # probably does not have polar coordinates
    #     chiperp_here = np.sqrt( samples['a1x']**2+ samples['a1y']**2)
    #     chi1_here = np.sqrt( samples['a1z']**2 + chiperp_here**2)
    #     theta1_here = np.arctan( samples['a1z']/chiperp_here)
    #     phi1_here = np.angle(samples['a1x']+1j*samples['a1y'])
    #     samples = add_field(samples, [('chi1', float)]); samples['chi1'] = chi1_here
    #     samples = add_field(samples, [('theta1', float)]); samples['theta1'] = theta1_here
    #     samples = add_field(samples, [('phi1', float)]); samples['phi1'] = phi1_here
        
    #     # we almost certainly use standard
    #     chi1_perp = np.sqrt(samples['a1x']**2 + samples['a1y']**2)
    #     chi2_perp = np.sqrt(samples['a2x']**2 + samples['a2y']**2)
    #     samples = add_field(samples, [('chi1_perp',float)]); samples['chi1_perp'] = chi1_perp
    #     samples = add_field(samples, [('chi2_perp',float)]); samples['chi2_perp'] = chi2_perp
        
    # elif "theta1" in samples.dtype.names:
    #     a1x_dat = samples["a1"]*np.sin(samples["theta1"])*np.cos(samples["phi1"])
    #     a1y_dat = samples["a1"]*np.sin(samples["theta1"])*np.sin(samples["phi1"])
    #     chi1_perp = samples["a1"]*np.sin(samples["theta1"])

    #     a2x_dat = samples["a2"]*np.sin(samples["theta2"])*np.cos(samples["phi2"])
    #     a2y_dat = samples["a2"]*np.sin(samples["theta2"])*np.sin(samples["phi2"])
    #     chi2_perp = samples["a2"]*np.sin(samples["theta2"])

                                      
    #     samples = add_field(samples, [('a1x', float)]);  samples['a1x'] = a1x_dat
    #     samples = add_field(samples, [('a1y', float)]); samples['a1y'] = a1y_dat
    #     samples = add_field(samples, [('a2x', float)]);  samples['a2x'] = a2x_dat
    #     samples = add_field(samples, [('a2y', float)]);  samples['a2y'] = a2y_dat
    #     samples = add_field(samples, [('chi1_perp',float)]); samples['chi1_perp'] = chi1_perp
    #     samples = add_field(samples, [('chi2_perp',float)]); samples['chi2_perp'] = chi2_perp
    #     if not 'chi_eff' in samples.dtype.names:
    #         samples = add_field(samples, [('chi_eff',float)]); samples['chi_eff'] = (samples["m1"]*samples["a1z"]+samples["m2"]*samples["a2z"])/(samples["m1"]+samples["m2"])
 
    # elif 'a1x' in samples.dtype.names:
    #     chi1_perp = np.sqrt(samples['a1x']**2 + samples['a1y']**2)
    #     chi2_perp = np.sqrt(samples['a2x']**2 + samples['a2y']**2)
    #     samples = add_field(samples, [('chi1_perp',float)]); samples['chi1_perp'] = chi1_perp
    #     samples = add_field(samples, [('chi2_perp',float)]); samples['chi2_perp'] = chi2_perp

    # if 'lambda1' in samples.dtype.names and not ('lambdat' in samples.dtype.names):
    #     Lt,dLt = lalsimutils.tidal_lambda_tilde(samples['m1'], samples['m2'],  samples['lambda1'], samples['lambda2'])
    #     samples = add_field(samples, [('lambdat', float)]); samples['lambdat'] = Lt
    #     samples = add_field(samples, [('dlambdat', float)]); samples['dlambdat'] = dLt


    if 'chi1_perp' in samples.dtype.names:
        # impose Kerr limit, if neede
        npts = len(samples["m1"])
        indx_ok =np.arange(npts)
        chi1_squared = samples['chi1_perp']**2 + samples["a1z"]**2
        chi2_squared = samples["chi2_perp"]**2 + samples["a2z"]**2
        indx_ok = np.logical_and(chi1_squared < opts.chi_max ,chi2_squared < opts.chi_max)
        npts_out = np.sum(indx_ok)
        new_samples = np.recarray( (npts_out,), dtype=samples.dtype)
        for name in samples.dtype.names:
            new_samples[name] = samples[name][indx_ok]
        samples = new_samples


    # Save samples
    posterior_list.append(samples)

    # Continue ... rest not used at present
    continue

    # Populate a P_list with the samples, so I can perform efficient conversion for plots
    # note only the DETECTOR frame properties are stored here
    P_list = []
    P = lalsimutils.ChooseWaveformParams()
    for indx in np.arange(len(samples["m1"])):
        P.m1 = samples["m1"][indx]*lal.MSUN_SI
        P.m2 = samples["m2"][indx]*lal.MSUN_SI
        P.s1x = samples["a1x"][indx]
        P.s1y = samples["a1y"][indx]
        P.s1z = samples["a1z"][indx]
        P.s2x = samples["a2x"][indx]
        P.s2y = samples["a2y"][indx]
        P.s2z = samples["a2z"][indx]
        if "lnL" in samples.keys():
            P.lnL = samples["lnL"][indx]   # creates a new field !
        else:
            P.lnL = -1
        # Populate other parameters as needed ...
        P_list.append(P)
    posteriorP_list.append(P_list)

for indx in np.arange(len(posterior_list)):
    samples = posterior_list[indx]
    fac = posterior_distance_factors[indx]
    if 'dist' in samples.dtype.names:
        samples["dist"]*= fac
    if 'distance' in samples.dtype.names:
        samples["distance"]*= fac

# Import
composite_list = []
composite_full_list = []
field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","lnL", "sigmaOverL", "ntot", "neff")
if opts.flag_tides_in_composite:
    if opts.flag_eos_index_in_composite:
        print(" Reading composite file, assumingtide/eos-index-based format ")
        field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","lambda1", "lambda2", "eos_table_index","lnL", "sigmaOverL", "ntot", "neff")
    else:
        print(" Reading composite file, assuming tide-based format ")
        field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","lambda1", "lambda2", "lnL", "sigmaOverL", "ntot", "neff")
if opts.eccentricity:
    print(" Reading composite file, assuming eccentricity-based format ")
    field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","eccentricity", "lnL", "sigmaOverL", "ntot", "neff")
field_formats = [np.float32 for x in field_names]
composite_dtype = [ (x,float) for x in field_names] #np.dtype(names=field_names ,formats=field_formats)
# Load posterior files
if opts.composite_file:
 print(opts.composite_file)
 for fname in opts.composite_file[:1]:  # Only load the first one!
    print(" Loading ... ", fname)
    samples = np.loadtxt(fname,dtype=composite_dtype)  # Names are not always available
    samples = samples[ ~np.isnan(samples["lnL"])] # remove nan likelihoods -- they can creep in with poor settings/overflows
    if opts.sigma_cut >0:
        npts = len(samples["m1"])
        # strip NAN
        sigma_vals = samples["sigmaOverL"]
        good_sigma = sigma_vals < opts.sigma_cut
        npts_out = np.sum(good_sigma)
        if npts_out < npts:
            new_samples = np.recarray( (npts_out,), dtype=samples.dtype)
            for name in samples.dtype.names:
                new_samples[name] = samples[name][good_sigma]
            samples = new_samples

#    samples = np.recarray(samples.T,names=field_names,dtype=field_formats) #,formats=field_formats)
    # If no record names
    # Add mtotal, q, 
    samples=add_field(samples,[('mtotal',float)]); samples["mtotal"]= samples["m1"]+samples["m2"]; 
    samples=add_field(samples,[('q',float)]); samples["q"]= samples["m2"]/samples["m1"]; 
    samples=add_field(samples,[('mc',float)]); samples["mc"] = lalsimutils.mchirp(samples["m1"], samples["m2"])
    samples=add_field(samples,[('eta',float)]); samples["eta"] = lalsimutils.symRatio(samples["m1"], samples["m2"])
    samples=add_field(samples,[('chi_eff',float)]); samples["chi_eff"]= (samples["m1"]*samples["a1z"]+samples["m2"]*samples["a2z"])/(samples["mtotal"]); 
    chi1_perp = np.sqrt(samples['a1x']*samples["a1x"] + samples['a1y']**2)
    chi2_perp = np.sqrt(samples['a2x']**2 + samples['a2y']**2)
    samples = add_field(samples, [('chi1_perp',float)]); samples['chi1_perp'] = chi1_perp
    samples = add_field(samples, [('chi2_perp',float)]); samples['chi2_perp'] = chi2_perp

    if ('lambda1' in samples.dtype.names):
        Lt,dLt = lalsimutils.tidal_lambda_tilde(samples['m1'], samples['m2'],  samples['lambda1'], samples['lambda2'])
        samples= add_field(samples, [('LambdaTilde',float), ('DeltaLambdaTilde',float),('lambdat',float),('dlambdat',float)])
        samples['LambdaTilde'] = samples['lambdat']= Lt
        samples['DeltaLambdaTilde'] = samples['dlambdat']= dLt

    samples_orig = samples
    if opts.lnL_cut:
        npts = len(samples["m1"])
        # strip NAN
        lnL_vals = samples["lnL"]
        not_nan = np.logical_not(np.isnan(lnL_vals))
        npts_out = np.sum(not_nan)
        if npts_out < npts:
            new_samples = np.recarray( (npts_out,), dtype=samples.dtype)
            for name in samples.dtype.names:
                new_samples[name] = samples[name][not_nan]
            samples = new_samples
        
        # apply cutoff
        indx_ok =np.arange(npts)
        lnL_max = np.max(samples["lnL"])
        print(" lnL_max = ", lnL_max)
        indx_ok = samples["lnL"]>lnL_max  -opts.lnL_cut
        npts_out = np.sum(indx_ok)
        new_samples = np.recarray( (npts_out,), dtype=samples.dtype)
        for name in samples.dtype.names:
            new_samples[name] = samples[name][indx_ok]
        samples = new_samples


    print(" Loaded samples from ", fname , len(samples["m1"]))
    if True:
        # impose Kerr limit
        npts = len(samples["m1"])
        indx_ok =np.arange(npts)
        chi1_squared = samples["chi1_perp"]**2 + samples["a1z"]**2
        chi2_squared = samples["chi2_perp"]**2 + samples["a2z"]**2
        indx_ok = np.logical_and(chi1_squared < opts.chi_max ,chi2_squared < opts.chi_max)
        npts_out = np.sum(indx_ok)
        if npts_out < npts:
            print(" Ok systems ", npts_out)
            new_samples = np.recarray( (npts_out,), dtype=samples.dtype)
            for name in samples.dtype.names:
                new_samples[name] = samples[name][indx_ok]
            samples = new_samples
            print(" Stripped samples  from ", fname , len(samples["m1"]))


    composite_list.append(samples)
    composite_full_list.append(samples_orig)

    continue


## Plot posterior files

CIs = [0.95,0.9, 0.68]
if opts.ci_list:
    CIs = eval(opts.ci_list)  # force creation
quantiles_1d = [0.05,0.95]
if opts.quantiles:
    quantiles_1d=eval(opts.quantiles)

# Generate labels
if opts.parameter_log_scale is None:
    opts.parameter_log_scale = []
labels_tex = render_coordinates(opts.parameter,logparams=opts.parameter_log_scale)#map(lambda x: tex_dictionary[x], coord_names)

fig_base= None
# Create figure workspace for 1d plots
fig_1d_list = []
fig_1d_list_cum = []
#fig_1d_list_ids = []
if opts.plot_1d_extra:
    for indx in np.arange(len(opts.parameter))+5:
        fig_1d_list.append(plt.figure(indx))
        fig_1d_list_cum.append(plt.figure(indx+len(opts.parameter)))
#        fig_1d_list_ids.append(indx)
    plt.figure(1)


# Find parameter ranges
x_range = {}
range_list = []
if opts.posterior_file:
 for param in opts.parameter:
    xmax_list = []
    xmin_list = []
    for indx in np.arange(len(posterior_list)):
        dat_here = None
        samples = posterior_list[indx]
        if param in samples.dtype.names:
            dat_here = samples[param]
        else:
            dat_here = extract_combination_from_LI(samples, param)
        if param in opts.parameter_log_scale:
            indx_ok = dat_here > 0
            dat_here= np.log10(dat_here[indx_ok])
        if len(dat_here) < 1:
            print(" Failed to etract data ", param,  " from ", opts.posterior_file[indx])

        # extend the limits, so we have *extremal* limits 
        xmax_list.append(np.max(dat_here))
        xmin_list.append(np.min(dat_here))
    x_range[param] = np.array([np.min(xmin_list), np.max(xmax_list)])  # give a small buffer
#    if param == 'chi_eff':
#        x_range[param] -= 0.1*np.sign([-1,1])*(x_range[param]+np.array([-1,1]))
            
    if param in special_param_ranges:
        x_range[param] = special_param_ranges[param]

    if param in ['lambda1', 'lambda2', 'lambdat']:
        x_range[param][1] = opts.lambda_plot_max

    range_list.append(x_range[param])
    print(param, x_range[param])


my_cmap_values=None
for pIndex in np.arange(len(posterior_list)):
    samples = posterior_list[pIndex]
    sample_names = samples.dtype.names; sample_ref_name  = sample_names[0]
    # Create data for corner plot
    dat_mass = np.zeros( (len(list(samples[sample_ref_name])), len(list(labels_tex))) )
    my_cmap_values = color_list[pIndex]
    plot_range_list = []
    smooth_list =[]
    truths_here= None
    if opts.truth_file:
        truths_here = np.zeros(len(opts.parameter))
    for indx in np.arange(len(opts.parameter)):
        param = opts.parameter[indx]
        if param in samples.dtype.names:
            dat_mass[:,indx] = samples[param]
        else:
            dat_mass[:,indx] = extract_combination_from_LI(samples, param)

        if param in opts.parameter_log_scale:
            dat_mass[:,indx] = np.log10(dat_mass[:,indx])

        # Parameter ranges (duplicate)
        dat_here = np.array(dat_mass[:,indx])  # force copy ! I need to sort
        weights = np.ones(len(dat_here))*1.0/len(dat_here)
        if 'weights' in samples.dtype.names:
            weights = samples['weights']
        indx_sort= dat_here.argsort()
        dat_here = dat_here[indx_sort]
        weights =weights[indx_sort]
#
        dat_here.sort() # sort it
        xmin, xmax = x_range[param]
#            xmin = np.min([np.min( posterior_list[x][param]) for x in np.arange(len(posterior_list)) ]) # loop over all
        xmin  = np.min([xmin, np.mean(dat_here)  -4*np.std(dat_here)])
        xmax = np.max([xmax, np.mean(dat_here)  +4*np.std(dat_here)])
#            xmax  = np.max(dat_here)
        if param in special_param_ranges:
                xmin,xmax = special_param_ranges[param]
        plot_range_list.append((xmin,xmax))

        # smoothing list
        smooth_list.append(np.std(dat_here)/np.power(len(dat_here), 1./3))
        
        # truths
        if opts.truth_file:
            param_to_extract = param
            if param in remap_LI_to_ILE.keys():
                param_to_extract  = remap_LI_to_ILE[param]
            if param in eos_param_names:
                continue
            if param == 'time':
                truths_here[indx] = P_ref.tref
                continue
            truths_here[indx] = P_ref.extract_param(param_to_extract)
            if param in [ 'mc', 'm1', 'm2', 'mtotal']:
                truths_here[indx] = truths_here[indx]/lal.MSUN_SI
            if param in ['dist', 'distance']:
                truths_here[indx] = truths_here[indx]/lal.PC_SI/1e6
#            print param, truths_here[indx]

        # if 1d plots needed, make them
        if opts.plot_1d_extra:
            range_here = range_list[indx]
            # 1d PDF
            # Set range based on observed results in ALL sets of samples, by default
            fig =fig_1d_list[indx]
            ax = fig.gca()
            ax.set_xlabel(labels_tex[indx])
            ax.set_ylabel('$dP/d'+labels_tex[indx].replace('$','')+"$")
            try:
                my_kde = bounded_kde.BoundedKDE(dat_here,low=xmin,high=xmax)
                xvals = np.linspace(range_here[0],range_here[1],1000)
                yvals = my_kde.evaluate(xvals)
                ax.plot(xvals,yvals,color=my_cmap_values,linestyle= linestyle_list[pIndex])
                if opts.truth_file:
                    ax.axvline(truths_here[indx], color='k',linestyle='dashed')
            except:
                print(" Failed to plot 1d KDE for ", labels_tex[indx])

            # 1d CDF
            fig =fig_1d_list_cum[indx]
            ax = fig.gca()
            ax.set_xlabel(labels_tex[indx])
            ax.set_ylabel('$P(<'+labels_tex[indx].replace('$','')+")$")
            xvals = dat_here
            #yvals = np.arange(len(dat_here))*1.0/len(dat_here)
            yvals = np.cumsum(weights)
            yvals = yvals/yvals[-1]
            ax.plot(xvals,yvals,color=my_cmap_values,linestyle= linestyle_list[pIndex] )
            if opts.truth_file:
                ax.axvline(truths_here[indx], color='k',linestyle='dashed')
            ax.set_xlim(xmin,xmax)

    # Add weight columns (unsorted) for overall unsorted plot
    weights = np.ones(len(dat_mass))*1.0/len(dat_mass)
    if 'weights' in samples.dtype.names:
        weights= samples['weights']
        weights = weights/np.sum(weights)
    # plot corner
#    smooth=smooth_list
    smooth1d=None
#    if opts.use_smooth_1d:
#        smooth1d=smooth_list
#        print smooth1d
    fig_base = corner.corner(dat_mass,smooth1d=smooth1d, range=range_list,weights=weights, labels=labels_tex, quantiles=quantiles_1d, plot_datapoints=False, plot_density=False, no_fill_contours=True, contours=True, levels=CIs,fig=fig_base,color=my_cmap_values ,hist_kwargs={'linestyle': linestyle_list[pIndex]}, linestyle=linestyle_list[pIndex],contour_kwargs={'linestyles':linestyle_list[pIndex]},truths=truths_here)


if opts.plot_1d_extra:
    for indx in np.arange(len(opts.parameter)):
        fig = fig_1d_list[indx]
        param = opts.parameter[indx]
        ax = fig.gca()
        # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
        if opts.use_legend:
            ax.legend(handles=line_handles, bbox_to_anchor=corner_legend_location, prop=corner_legend_prop,loc=2) 
        fig.savefig(param+fig_extension,dpi=dpi_base)

        fig = fig_1d_list_cum[indx]
        param = opts.parameter[indx]
        ax = fig.gca()
        # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
        if opts.use_legend:
            ax.legend(handles=line_handles, prop=corner_legend_prop, loc=4) # bbox_to_anchor=corner_legend_location, 
        fig.savefig(param+"_cum"+fig_extension,dpi=dpi_base)


if composite_list:
  for pIndex in [0]: # np.arange(len(composite_list)):  # should NEVER have more than one
    samples = composite_list[pIndex]
    samples_orig = composite_full_list[pIndex]
    # Create data for corner plot
    dat_mass = np.zeros( (len(samples["m1"]), len(labels_tex)) )
    dat_mass_orig = np.zeros( (len(samples_orig["m1"]), len(labels_tex)) )
    lnL = samples["lnL"]
    indx_sorted = lnL.argsort()
    if len(lnL)<1:
        print(" Failed to retrieve lnL for composite file ", composite_list[0])

    cm = plt.cm.get_cmap('rainbow') #'RdYlBu_r')
    y_span = lnL.max() - lnL.min()
    print(" Composite file : lnL span ", y_span)
    y_min = lnL.min()
    cm2 = lambda x: cm(  (x - y_min)/y_span)
    my_cmap_values = cm(    (lnL-y_min)/y_span) 
#    print my_cmap_values[:10]


    truths_here=None
    if opts.truth_file:
        truths_here =np.zeros(len(opts.parameter))
    for indx in np.arange(len(opts.parameter)):
        param = opts.parameter[indx]
        if param in field_names:
            dat_mass[:,indx] = samples[param]
            dat_mass_orig[:,indx] = samples_orig[param]
        else:
            print(" Trying alternative access for ", param)
            dat_mass[:,indx] = extract_combination_from_LI(samples, param)
            dat_mass_orig[:,indx] = extract_combination_from_LI(samples_orig, param)
        # truths
        if opts.truth_file:
            param_to_extract = param
            if param in remap_LI_to_ILE.keys():
                param_to_extract  = remap_LI_to_ILE[param]
            truths_here[indx] = P_ref.extract_param(param_to_extract)
            if param in [ 'mc', 'm1', 'm2', 'mtotal']:
                truths_here[indx] = truths_here[indx]/lal.MSUN_SI
#            print param, truths_here[indx]

    print(" Truths here ", truths_here)
        

    # fix ranges
    if range_list == [] :
        range_list=None

    # reverse order ... make sure largest plotted last
    dat_mass = dat_mass[indx_sorted]   # Sort by lnL
    my_cmap_values = my_cmap_values[indx_sorted]
#    my_cmap_values = my_cmap_values[::-1]
#    dat_mass = dat_mass[::-1]
            
    # We will need to rewrite 'corner' to do what we want: see the source
    # https://github.com/dfm/corner.py/blob/master/corner/corner.py
    # Grayscale, using all points
    if opts.use_all_composite_but_grayscale:
        fig_base = our_corner.corner(dat_mass_orig,range=range_list, plot_datapoints=True,weights=np.ones(len(dat_mass_orig))*1.0/len(dat_mass_orig), plot_density=False, no_fill_contours=True, plot_contours=False,contours=False,levels=None,fig=fig_base,data_kwargs={'color':'0.5','s':1})
    # Color scale with colored points
    fig_base = our_corner.corner(dat_mass,range=range_list, plot_datapoints=True,weights=np.ones(len(dat_mass))*1.0/len(dat_mass), plot_density=False, no_fill_contours=True, plot_contours=False,contours=False,levels=None,fig=fig_base,data_kwargs={'color':my_cmap_values, 's':1}, truths=truths_here)

    # Create colorbar mappable
#    ax=plt.figure().gca()
#    ax.contourf(lnL, cm)

if opts.use_legend and opts.posterior_label:
    plt.legend(handles=line_handles, bbox_to_anchor=corner_legend_location, prop=corner_legend_prop,loc=4)
#plt.colorbar()  # Will fail, because colors not applied

# title
if opts.use_title:
    print(" Addding title ", opts.use_title)
    plt.title(opts.use_title)

param_postfix = "_".join(opts.parameter)
res_base = len(opts.parameter)*dpi_base
matplotlib.rcParams.update({'font.size': 11+int(len(opts.parameter)), 'legend.fontsize': legend_font_base+int(1.3*len(opts.parameter))})   # increase font size if I have more panels, to keep similar aspect
plt.savefig("corner_"+param_postfix+fig_extension,dpi=res_base)        # use more resolution, to make sure each image remains of consistent quality
