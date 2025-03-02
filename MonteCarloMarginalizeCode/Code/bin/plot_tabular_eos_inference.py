#! /usr/bin/env python

# Author: Askold Vilkha (https://github.com/askoldvilkha), av7683@rit.edu, askoldvilkha@gmail.com

# This script is used to plot the results of the tabular EOS inference. 
# Minimal input is the path to the tabular EOS file and posterior samples files the user wants to have on the plot.
# The user will have to specify which plots to make by setting the corresponding flags to True.
# The script has a functionality to choose custom labels and colors for the posterior samples.
# If multiple tabular EOS files are provided, the user will have to specify which ones to plot and to use for the posterior samples.

import numpy as np
import matplotlib.pyplot as plt
import h5py
import RIFT.physics.EOSManager as EOSManager
#from natsort import natsorted
import warnings
import argparse

import ast
import textwrap

import RIFT.plot_utilities.TabularEOSPlotUtilities as tabplot

# store long help messages in a dictionary to avoid cluttering the parser code below
argparse_help_dict = {
    'help': textwrap.dedent('''\
    This script is used to plot the results of the tabular EOS inference obtained with RIFT code.
    The user can plot pressure vs. density and mass vs. radius plots for the tabular EOS priors and posterior samples.
    The user can specify the tabular EOS files, posterior samples files, labels, and colors for the plots.
    The user has to provide the path to the tabular EOS files and the posterior samples files.
    Basic usage: 
    plot_tabular_eos_inference.py --tabular-eos-file <path_to_tabular_eos_file> --tabular-eos-label <tabular_eos_file_label> --posterior-file <path_to_posterior_samples_file> 
    --posterior-label <posterior_samples_file_label> --plot-p-vs-rho --plot-m-vs-r --use-bgcgb-colormap'''),

    'tabular-eos-file': textwrap.dedent('''\
    Path to the tabular EOS file. If multiple files are provided only the first one will be used for EOS posteriors unless specified otherwise by the user. 
    See --posterior-tabular-map for more information.','''),

    'plot-tabular-eos-prior': textwrap.dedent('''\
    Specify which tabular EOS files to plot priors for. 
    Should be specified by the label of the tabular EOS file'''),

    'posterior-tabular-map': textwrap.dedent('''\
    Specify which tabular EOS file to use for the posterior samples. 
    Should be specified by the label of the tabular EOS files and the posterior samples files. 
    The input format should be a dictionary in the form of a string:
    '{'tabular_eos_file_1': ['posterior_file_1', 'posterior_file_2'] 'tabular_eos_file_2': 'posterior_file_3', ...}'. 
    Use either labels instead of the file names.'''),

    'posterior-histogram-range-r': textwrap.dedent('''\
    Set the range for the Radius at the posterior samples histogram plot. 
    The input format should be a list of two numbers defining a range for the Radius. (e.g. [10, 15])'''),

    'posterior-histogram-range-lambda': textwrap.dedent('''\
    Set the range for the Tidal Deformability Lambda at the posterior samples histogram plot.
    The input format should be a list of two numbers defining a range for the Tidal Deformability Lambda. (e.g. [400, 1600])''')
}

parser = argparse.ArgumentParser(description = argparse_help_dict['help'])

# basic arguments
parser.add_argument('--tabular-eos-file', action = 'append', help = argparse_help_dict['tabular-eos-file'], required = True)
parser.add_argument('--posterior-file', action = 'append', help = 'Path to the posterior samples file. If none are provdided, only priors will be plotted.')
parser.add_argument('--plot-p-vs-rho', action = 'store_true', help = 'Plot pressure vs. density')
parser.add_argument('--plot-m-vs-r', action = 'store_true', help = 'Plot mass vs. radius')
parser.add_argument('--tabular-eos-label', action = 'append', help = 'Label for the tabular EOS file')
parser.add_argument('--posterior-label', action = 'append', help = 'Label for the posterior samples file')
parser.add_argument('--color', action = 'append', help = 'Colors for the plot. If not provided, colors will be chosen automatically')
parser.add_argument('--use-bgcgb-colormap', action = 'store_true', help = 'Use the BlackGreyCyanGreenBlue colormap for the plots')
parser.add_argument('--verbose', action = 'store_true', help = 'Print information on the progress of the code')
parser.add_argument('--plot-p-vs-rho-title', action = 'store', help = 'Title for the pressure vs. density plot')
parser.add_argument('--plot-m-vs-r-title', action = 'store', help = 'Title for the mass vs. radius plot')

# extra arguments for more advanced plotting (plot posteriors for multiple tabular EOS files, plot tabular EOS priors, etc.)
parser.add_argument('--plot-tabular-eos-prior', action = 'append', help = argparse_help_dict['plot-tabular-eos-prior'])
parser.add_argument('--posterior-tabular-map', action = 'store', help = argparse_help_dict['posterior-tabular-map'])
parser.add_argument('--plot-posterior-histogram', action = 'store_true', help = 'Plot the posterior samples histogram')
parser.add_argument('--plot-lambda-tilde-ratio', action = 'store_true', help = 'Plot the lambda_tilde vs ordering statistics S ratio')
parser.add_argument('--posterior-histogram-range-r', action = 'store', help = 'Range for Radius at the posterior samples histogram plot')
parser.add_argument('--posterior-histogram-range-lambda', action = 'store', help = 'Range for Tidal Deformability Lambda at the posterior samples histogram plot')

args = parser.parse_args()

tabular_eos_labels = [None] * len(args.tabular_eos_file)

# the code can operate with less labels than files, it will use the default labels for the rest of the files
if args.tabular_eos_label is not None:
    if len(args.tabular_eos_label) < len(args.tabular_eos_file):
        warnings.warn('Number of tabular EOS labels is less than the number of tabular EOS files. The code will use the default labels after the last provided label.')
    elif len(args.tabular_eos_label) > len(args.tabular_eos_file):
        raise ValueError('Number of tabular EOS labels is greater than the number of tabular EOS files. Please check if you have provided the correct number of labels or files.')

    tabular_eos_labels[:len(args.tabular_eos_label)] = args.tabular_eos_label # fill in the provided labels, use the default labels for the rest

    if args.verbose:
        print('\nChecked the tabular EOS file labels, proceeding with the data loading...')

# posterior histogram and lambda_tilde ratio plots require saving the EOS manager object and/or the posterior samples (not just eos_indx column)
save_eos_manager = False
save_posterior_samples = False
if args.plot_posterior_histogram or args.plot_lambda_tilde_ratio:
    if args.posterior_file is None:
        raise ValueError('You have chosen to plot the posterior samples histogram or lambda_tilde ratio without providing the posterior samples files. Please provide the posterior samples files.')
    save_eos_manager = True
if args.plot_lambda_tilde_ratio:
    save_posterior_samples = True

# load the tabular EOS files
tabular_eos_data = {}
for i, tabular_eos_file in enumerate(args.tabular_eos_file):
    eos_data_i = tabplot.EOS_data_loader(tabular_eos_file, tabular_eos_labels[i], args.verbose, save_eos_manager)
    eos_data_i_label = eos_data_i['data_label'] # if the label was not provided, the EOS_data_loader function will assign a default label
    tabular_eos_data[eos_data_i_label] = eos_data_i
    tabular_eos_labels[i] = eos_data_i_label # update the label in case the default label was assigned

if args.verbose:
    print('\nLoaded the tabular EOS data.')
    for i in range(len(tabular_eos_labels)):
        print(f'Tabular EOS file: {args.tabular_eos_file[i]} with label: {tabular_eos_labels[i]}')

if args.posterior_file is not None:
    priors_labels = []
    posterior_samples_labels = [None] * len(args.posterior_file)

    if args.posterior_label is not None:
        if len(args.posterior_label) < len(args.posterior_file):
            warnings.warn('Number of posterior samples labels is less than the number of posterior samples files. The code will use the default labels after the last provided label.')
        elif len(args.posterior_label) > len(args.posterior_file):
            raise ValueError('Number of posterior samples labels is greater than the number of posterior samples files. Please check if you have provided the correct number of labels or files.')

        posterior_samples_labels[:len(args.posterior_label)] = args.posterior_label 
    
        if args.verbose:
            print('\nChecked the posterior samples labels, proceeding with the data loading...')

    # load the posterior samples files
    posterior_samples_data = {}
    for i, posterior_file in enumerate(args.posterior_file):
        posterior_data_i = tabplot.posterior_data_loader(posterior_file, posterior_samples_labels[i], args.verbose, save_posterior_samples)
        posterior_data_i_label = posterior_data_i['data_label']
        posterior_samples_data[posterior_data_i_label] = posterior_data_i
        posterior_samples_labels[i] = posterior_data_i_label

    if args.verbose:
        print('\nLoaded the posterior samples data.')
        for i in range(len(posterior_samples_labels)):
            print(f'Posterior samples file: {args.posterior_file[i]} with label: {posterior_samples_labels[i]}')

    # make the posterior-tabular map if not provided by the user
    if args.posterior_tabular_map is None:
        # if the user did not provide the posterior-tabular map, the code will only use first tabular EOS file for the posterior samples
        posterior_tabular_map = {tabular_eos_labels[0]: posterior_samples_labels}

    # check if the map is provided in the correct format and all the labels are valid
    else:
        if args.verbose:
            print('\nChecking the posterior-tabular map...')
        posterior_tabular_map = ast.literal_eval(args.posterior_tabular_map)
        if not isinstance(posterior_tabular_map, dict):
            raise ValueError('The posterior-tabular map should be a dictionary. Please check the format of the input.')
        for tabular_label in posterior_tabular_map.keys():
            if tabular_label not in tabular_eos_labels:
                raise ValueError(f'Tabular EOS label: {tabular_label} from your posterior-tabular map is not found in the tabular EOS labels. Please check the labels you provided.')
            if isinstance(posterior_tabular_map[tabular_label], list):
                for posterior_label in posterior_tabular_map[tabular_label]:
                    if posterior_label not in posterior_samples_labels:
                        raise ValueError(f'Posterior samples label: {posterior_label} from your posterior-tabular map is not found in the posterior samples labels. Please check the labels you provided.')
            else:
                if posterior_tabular_map[tabular_label] not in posterior_samples_labels:
                    raise ValueError(f'Posterior samples label: {posterior_tabular_map[tabular_label]} from your posterior-tabular map is not found in the posterior samples labels. Please check the labels you provided.')

    if args.verbose:
        print('\nCreated the posterior-tabular map.')
        for tabular_label in posterior_tabular_map.keys():
            print(f'Tabular EOS label: {tabular_label} with posterior samples labels: {posterior_tabular_map[tabular_label]}')
        print('\nMoving on to priors...')
else:
    posterior_samples_data = None
    posterior_tabular_map = None
    priors_labels = tabular_eos_labels # plot priors for all tabular EOS files if no posterior samples are provided
    warnings.warn('No posterior samples files provided. The code will only plot the tabular EOS priors.')
    
# if the user specified the tabular EOS files to plot priors for, the code will plot priors for those files
if args.plot_tabular_eos_prior is not None:
    for tabular_label in args.plot_tabular_eos_prior:
        if tabular_label not in tabular_eos_labels:
            raise ValueError(f'Tabular EOS label for prior plot: {tabular_label} is not found in the tabular EOS labels. Please check the labels you provided.')
    priors_labels = args.plot_tabular_eos_prior
elif len(tabular_eos_labels) > 1 and posterior_samples_data is not None:
    priors_labels = tabular_eos_labels[1:] # plot priors for all tabular EOS files except the first one

if args.verbose:
    print('\nChecked the tabular EOS labels for the priors.')
    for i in range(len(priors_labels)):
        print(f'Priors will be plotted for the tabular EOS file with label: {priors_labels[i]}')
    print('\nLinking the tabular EOS data with the posterior samples data according to the posterior-tabular map...')

plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['font.size'] = 12

def EOS_plotter(args, tabular_eos_data: dict, posterior_samples_data: dict, posterior_tabular_map: dict, priors_labels: list, plot_type: str) -> None:
    """
    Function to plot the EOS inference results for the tabular EOS priors and posterior samples.
    
    Parameters:
    ----------
    args : argparse.Namespace
        The argparse namespace object with the script arguments.
    tabular_eos_data : dict
        The dictionary with the tabular EOS data generated earlier in the script. 
        Format: {tabular_label: tabular_eos_data}, tabular_eos_data should be generated by the EOS_data_loader function.
    posterior_samples_data : dict
        The dictionary with the posterior samples data generated earlier in the script. 
        Format: {posterior_label: posterior_samples_data}, posterior_samples_data should be generated by the posterior_data_loader function.
    posterior_tabular_map : dict
        The dictionary with the posterior-tabular map generated earlier in the script. 
        Format: {tabular_label: posterior_samples_label}, posterior_samples_label can be a list of labels or a single string label.
    priors_labels : list
        The list with the tabular EOS labels to plot priors for.
    plot_type : str
        The type of the plot to make. Can be 'pressure_density' or 'mass_radius'.

    Returns:
    -------
    None

    Raises:
    -------
    NOTE!
    Errors will not be raised if the script has not been changed from the initial version. 
    They are here to ensure future modifications do not break the code.

    ValueError
        If the plot type is not 'pressure_density' or 'mass_radius'. 
    ValueError
        If the posterior samples data is provided without the posterior-tabular map or vice versa.
    """
    if plot_type not in ['pressure_density', 'mass_radius']:
        raise ValueError('The plot type should be either pressure_density or mass_radius. Please check the input.')
    if (posterior_samples_data is None) != (posterior_tabular_map is None):
        raise ValueError('The posterior samples data and the posterior-tabular map should be provided together. Please check the input.')

    raw_plot_data = []

    for tabular_label in priors_labels:
        raw_plot_data.append(tabplot.link_eos_data_to_posterior(tabular_eos_data[tabular_label], plot_type))
    
    if args.verbose:
        print(f'\nCollected the data for priors for the {plot_type} plot. Moving on to posteriors...')
    
    # link the tabular EOS data with the posterior samples data according to the posterior-tabular map
    if posterior_samples_data is not None:
        for tabular_label in posterior_tabular_map.keys():
            if isinstance(posterior_tabular_map[tabular_label], list):
                for posterior_label in posterior_tabular_map[tabular_label]:
                    raw_plot_data.append(tabplot.link_eos_data_to_posterior(tabular_eos_data[tabular_label], plot_type, posterior_samples_data[posterior_label]))
            else:
                raw_plot_data.append(tabplot.link_eos_data_to_posterior(tabular_eos_data[tabular_label], plot_type, posterior_samples_data[posterior_tabular_map[tabular_label]]))

        if args.verbose:
            print(f'\nCollected the data for posterior samples for the {plot_type} plot. Processing...')
    
    # choose the colors for the plots
    colors = [None] * len(raw_plot_data)
    if args.color is not None:
        if len(args.color) < len(raw_plot_data):
            warnings.warn('Number of colors is less than the number of posterior samples files. The code will use the default colors for the rest of the files.')
            colors[:len(args.color)] = args.color
        else: 
            colors = args.color 
    if args.use_bgcgb_colormap:
        colors[:5] = ['black', 'grey', 'cyan', 'green', 'blue']
        if args.color is not None:
            warnings.warn('You have chosen to use the BlackGreyCyanGreenBlue colormap. The code will ignore the first 5 colors you provided.')
    if args.verbose and any(colors is not None for c in colors):
        print(f'\nThe colors for the plots are: {colors}. Any of the None values will be replaced with the default colors. (tab10 colormap)')
    
    plot_data = []
    if plot_type == 'pressure_density':
        for i, raw_data in enumerate(raw_plot_data):
            plot_data.append(tabplot.pressure_density_plot_data_gen(**raw_data, custom_color = colors[i]))
        
        if args.verbose:
            print(f'\nProcessed the data for the {plot_type} plot. Plotting...')

        tabplot.pressure_density_plot(*plot_data, title = args.plot_p_vs_rho_title)
    
    elif plot_type == 'mass_radius':
        for i, raw_data in enumerate(raw_plot_data):
            plot_data.append(tabplot.mass_radius_plot_data_gen(**raw_data, custom_color = colors[i]))

        if args.verbose:
            print(f'\nProcessed the data for the {plot_type} plot. Plotting...')

        tabplot.mass_radius_plot(*plot_data, title = args.plot_m_vs_r_title)

    return None
    
# plot the P vs rho EOS inference
if args.plot_p_vs_rho:
    EOS_plotter(args, tabular_eos_data, posterior_samples_data, posterior_tabular_map, priors_labels, 'pressure_density')

# plot the M vs R EOS inference
if args.plot_m_vs_r:
    EOS_plotter(args, tabular_eos_data, posterior_samples_data, posterior_tabular_map, priors_labels, 'mass_radius')

# plot the posterior samples histogram
if args.plot_posterior_histogram:
    
    posterior_hist_data = []
    for tabular_label in posterior_tabular_map.keys():
        if isinstance(posterior_tabular_map[tabular_label], list):
            for posterior_label in posterior_tabular_map[tabular_label]:
                posteriors_eos = posterior_samples_data[posterior_label]['posterior_samples_eos']
                eos_manager = tabular_eos_data[tabular_label]['eos_manager']
                posterior_hist_data.append(tabplot.posterior_hist_data_gen(posteriors_eos, eos_manager, posterior_label))
        else:
            posteriors_eos = posterior_samples_data[posterior_tabular_map[tabular_label]]['posterior_samples_eos']
            eos_manager = tabular_eos_data[tabular_label]['eos_manager']
            posterior_label = posterior_tabular_map[tabular_label]
            posterior_hist_data.append(tabplot.posterior_hist_data_gen(posteriors_eos, eos_manager, posterior_label))
    
    if args.verbose:
        print('\nProcessed the data for the posterior samples histogram. Plotting...')


    r_lim = (8, 17)
    lambda_lim = (0, 1000)
    if args.posterior_histogram_range_r is not None:
        r_lim_input = ast.literal_eval(args.posterior_histogram_range_r)
        r_lim = (r_lim_input[0], r_lim_input[1])
    if args.posterior_histogram_range_lambda is not None:
        lambda_lim_input = ast.literal_eval(args.posterior_histogram_range_lambda)
        lambda_lim = (lambda_lim_input[0], lambda_lim_input[1])

    tabplot.posterior_hist_plot(*posterior_hist_data, r_lim = r_lim, lambda_lim = lambda_lim)

# plot the lambda_tilde vs S ratio
if args.plot_lambda_tilde_ratio:
    
    lambda_tilde_ratio_data = []
    for tabular_label in posterior_tabular_map.keys():
        if isinstance(posterior_tabular_map[tabular_label], list):
            for posterior_label in posterior_tabular_map[tabular_label]:
                posteriors_samples_all = posterior_samples_data[posterior_label]['posterior_samples_all']
                eos_manager = tabular_eos_data[tabular_label]['eos_manager']
                lambda_tilde_ratio_data.append(tabplot.LambdaTilderatio_data_gen(posteriors_samples_all, eos_manager, posterior_label))
        else:
            posteriors_samples_all = posterior_samples_data[posterior_tabular_map[tabular_label]]['posterior_samples_all']
            eos_manager = tabular_eos_data[tabular_label]['eos_manager']
            posterior_label = posterior_tabular_map[tabular_label]
            lambda_tilde_ratio_data.append(tabplot.LambdaTilderatio_data_gen(posteriors_samples_all, eos_manager, posterior_label))
    
    if args.verbose:
        print('\nProcessed the data for the lambda_tilde vs S ratio plot. Plotting...')
    
    tabplot.LambdaTilderatio_plot(*lambda_tilde_ratio_data)
    tabplot.LambdaTilderatio_plot(*lambda_tilde_ratio_data, ylim = (0.8, 1.2))
