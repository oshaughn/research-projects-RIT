# Author: Askold Vilkha (https://github.com/askoldvilkha), av7683@rit.edu, askoldvilkha@gmail.com

# This code is a library of functions for plotting tabular EOS data along with the EOS inference from posterior samples obtained using the RIFT code.
# Current functionality includes:
# 1. Pressure vs Density plots
# 2. Mass vs Radius plots

# Possible future functionality:
# 1. Radius and Tidal deformability (LambdaTilde) histograms at a given mass (1.4 Msun by default)
# 2. Pressure vs Energy density plots
# 3. Pressure vs Density plots in the SI units

import numpy as np
import matplotlib.pyplot as plt
import h5py
import RIFT.physics.EOSManager as EOSManager
import RIFT.lalsimutils as lalsimutils
from natsort import natsorted
import warnings
# import seaborn as sns

def EOS_data_loader(tabular_eos_file: str, data_label: str = None, verbose: bool = False, save_eos_manager: bool = False) -> dict:
    """
    Load tabular EOS data from a HDF5 file.

    Parameters
    ----------
    tabular_eos_file : str
        The path to the tabular EOS file. The file should be in the .h5 format.
        The function will automatically get the data label from the file if the format is 'LCEHL_EOS_posterior_samples_<data_label>.h5'.
    data_label : str, optional
        The label of the tabular EOS data that is loaded. If not provided the label will be generated from the file name automatically.
    verbose : bool, optional
        If True, the function will print information about the progress.
    save_eos_manager : bool, optional
        If True, the function will save the EOSManager object. Default is False.

    Returns
    -------
    EOS_data : dict
        A dictionary containing the necessary tabular EOS data for plotting functions.
        The dictionary contains the following keys:
        - 'eos_tables': the tabular EOS data (interpolated by default)
        - 'tov_tables': the TOV data (not interpolated)
        - 'eos_names': the names of the EOS tables
        - 'data_label': the label of the tabular EOS data
        - 'eos_manager': the EOSManager object (only if `save_eos_manager` is True)
    """

    # load tabular EOS data
    EOSManager_Local = EOSManager.EOSSequenceLandry(fname = tabular_eos_file, load_ns = True, load_eos = True, verbose = verbose, eos_tables_units = 'cgs')
    eos_tables = EOSManager_Local.interpolate_eos_tables('baryon_density')
    tov_tables = EOSManager_Local.eos_ns_tov
    eos_names = EOSManager_Local.eos_names

    if verbose:
        print(f"Loaded tabular EOS data from the file: {tabular_eos_file}")
    EOS_data = {'eos_tables': eos_tables, 'tov_tables': tov_tables, 'eos_names': eos_names}

    if data_label is None:
        data_label = tabular_eos_file.split('_')[-1].split('.')[0]
        
    if data_label.upper() not in ['PSR', 'PSR+GW', 'PSR+GW+NICER']:
        warnings.warn(f"Data label for the tabular EOS file is not one of the commonly used: {data_label}. Please make sure the data label is correct.")
    
    EOS_data['data_label'] = data_label

    if save_eos_manager:
        EOS_data['eos_manager'] = EOSManager_Local

    if verbose:
        print(f'Loaded tabular EOS data from {tabular_eos_file} with data label: {data_label}')

    return EOS_data

def posterior_data_loader(posterior_file: str, data_label: str = None, verbose: bool = False, out_all: bool = False) -> dict:
    """
    Load the posterior data from a .dat file

    Parameters
    ----------
    posterior_file : str
        The path to the posterior samples file. The file should be in the .dat format.
        The function will automatically get the data label from the file if the format is 'posterior_samples_<data_label>.dat'.
    data_label : str
        The label of the data. If not provided, the function will try to get it from the file name.
    verbose : bool
        If True, the function will print information about the progress.
    out_all : bool
        If True, the function will return all the posterior samples. If False, the function will return only the EOS indices posteriors.

    Returns
    -------
    posterior_data : dict
        A dictionary containing the posterior samples.
        The dictionary contains the following keys:
        - 'posterior_samples_eos' : The posterior samples for the EOS indices.
        - 'data_label' : The label of the data.
        - 'posterior_samples_all' : The posterior samples for all the parameters. (Only if `out_all` is True)

    """

    # load the posterior samples
    posterior_samples = np.genfromtxt(posterior_file, names = True, replace_space = None)
    posterior_samples_eos = posterior_samples['eos_indx']
    posterior_data = {'posterior_samples_eos': posterior_samples_eos, 'data_label': data_label}
    if out_all:
        posterior_data['posterior_samples_all'] = posterior_samples

    if data_label is None:
        data_label = posterior_file.split('_')[-1].split('.')[0]
        posterior_data['data_label'] = data_label
    if data_label.upper() not in ['PSR', 'PSR+GW', 'PSR+GW+NICER']:
        warnings.warn(f"Data label for the posterior samples file is not one of the commonly used: {data_label}. Please make sure the data label is correct.")
    
    if verbose:
        print(f'Posterior samples loaded successfully from {posterior_file} with data label: {data_label}')
    
    return posterior_data

def link_eos_data_to_posterior(eos_data: dict, plot_type: str, posterior_data: dict = None) -> dict:
    """
    This function links the EOS data to the posterior data and outputs a dictionary with the linked data suited for the corresponding plot type.

    Parameters
    ----------
    eos_data : dict
        Dictionary containing the EOS data. Should be generated using the function EOS_data_loader. 
        See TabularEOSPlotUtilities.EOS_data_loader for more information on the format of the dictionary.
    plot_type : str
       String specifying the type of plot to be generated. Currently supported options are: 'pressure_density' and 'mass_radius'.
    posterior_data : dict
        Dictionary containing the posterior data. Should be generated using the function posterior_data_loader. 
        See TabularEOSPlotUtilities.posterior_data_loader for more information on the format of the dictionary.
        If posterior_data is not provided, the function will generate prior data from the EOS data.
    
    Returns
    -------
    plot_data_gen_dict: dict
        Dictionary containing the linked EOS and posterior data suited for the corresponding plot type.
        The format will fit the input requirements of the plotting functions in TabularEOSPlotUtilities.
    """
    # check if the plot type is supported
    if plot_type.lower() not in ['pressure_density', 'mass_radius']:
        raise ValueError('The plot type is not supported. Currently supported options are: "pressure_density" and "mass_radius".')
    
    # generate the dictionary to store the linked data
    plot_data_gen_dict = {}
    # generate input data for the pressure_density plot
    if plot_type.lower() == 'pressure_density':
        plot_data_gen_dict['eos_tables'] = eos_data['eos_tables']

    # generate input data for the mass_radius plot
    elif plot_type.lower() == 'mass_radius':
        plot_data_gen_dict['tov_tables'] = eos_data['tov_tables']
    
    plot_data_gen_dict['eos_names'] = eos_data['eos_names']

    # if the posterior data is provided, link the posterior data to the EOS data, if not, generate prior data
    if posterior_data is not None:
        plot_data_gen_dict['data_label'] = posterior_data['data_label']
        plot_data_gen_dict['posteriors_eos'] = posterior_data['posterior_samples_eos']
        plot_data_gen_dict['posterior'] = True
    else:
        plot_data_gen_dict['data_label'] = eos_data['data_label']
        plot_data_gen_dict['posteriors_eos'] = np.arange(len(eos_data['eos_names']))
        plot_data_gen_dict['posterior'] = False
    
    return plot_data_gen_dict

def generate_posterior_eos_param_grid(posteriors_eos: np.ndarray, eos_tables: dict, eos_names: list, param: str = 'pressure', verbose: bool = False) -> np.ndarray:
    """
    This function generates a grid of the format (N, M), 
    where N is the number of points in the EOS table, and M is the number of posterior samples.
    Each column of the grid will be the EOS table for a given posterior sample. 
    Each row will be the values of the parameter of interest (e.g. pressure, energy density, etc.) 
    for a given point of the parameter used as the x-axis of the plot (and the interpolation base) respectively.
    By default, the parameter of interest is the pressure and the interpolation base is the baryon density.
    Thus, each row of the grid will correspond to the pressure values at each posterior sample for a given baryon density value.

    Parameters
    ----------
    posteriors_eos : np.ndarray
        Array of EOS table indices for each posterior sample. Should be the 'eos_indx' column of the posterior samples.
    eos_tables : dict
        Dictionary of EOS tables. The keys should be the EOS names and the values should be the EOS tables.
        Should be generated using the EOSManager.interpolate_eos_tables() method.
        IMPORTANT! The EOS parameter used as the x-axis of the plot should be the same for all EOS tables.
        If this is not true, the percentiles will not make any sense!
    eos_names : list
        List of EOS names. Should be generated using the EOSManager.eos_names attribute.
    param : str, optional
        Parameter of interest. The default is 'pressure'. Another options are 'energy_density' and 'baryon_density'.
    verbose : bool, optional
        If True, print the progress of the function. The default is False.

    Returns
    -------
    posteriors_eos_param_grid : np.ndarray
        Grid of the format (N, M) where N is the number of points in the EOS table and M is the number of posterior samples.
    
    Raises
    ------
    ValueError
        If the parameter of interest is not present in the EOS tables.
    ValueError
        If the number of points in the EOS table for any EOS is different from the number of points in the EOS tables for other EOSs.
        
    """

    if param not in eos_tables[eos_names[0]].keys():
        raise ValueError(f'The parameter {param} is not present in the EOS tables! Please choose one of the following parameters: {eos_tables[eos_names[0]].keys()} If the parameters listed are other than "pressure, energy_density, baryon_density", please double-check the EOS tables!')

    # get the number of points in the EOS table (N) and the number of posterior samples (M)
    N = len(eos_tables[eos_names[0]][param])
    M = len(posteriors_eos)
    if verbose:
        print(f'The EOS parameter grid will have {N} points and there are {M} posterior samples.')

    # initialize the grid
    posteriors_eos_param_grid = np.zeros((N, M))

    # fill the grid
    for i, eos_indx in enumerate(posteriors_eos):
        if verbose:
            print(f'Processing posterior sample {i} with EOS index {int(eos_indx)}')
        eos_name = eos_names[int(eos_indx)]
        eos_table = eos_tables[eos_name]
        N_i = len(eos_table[param])
        if N_i != N:
            raise ValueError(f'The number of points in the EOS table for {eos_name} is different from the number of points in the EOS tables for other EOSs! Please interpolate the tables before passing them to this function!')
        posteriors_eos_param_grid[:, i] = eos_table[param]
    
    if verbose:
        print('The EOS parameter grid has been successfully generated!')
    
    return posteriors_eos_param_grid

def pressure_density_plot_data_gen(eos_tables: dict, posteriors_eos: np.ndarray, eos_names: list, data_label: str, posterior: bool = True, perc_values: list = [5, 95], custom_color: str = None, use_energy_dens: bool = False) -> dict:
    """
    This function generates the data dictionary for the pressure-density plot. 
    It is designed as an auxiliary function for the pressure_density_plot() function, and should be used as input for that function. 

    The dictionary will contain the following keys:
    - 'x_dens_grid': the grid of baryon densities for the x-axis
    - 'y_pres_perc': percentiles of the pressure for each baryon density. 
    This will be a np.ndarray of shape (2, N), where N is the number of points in the EOS table.
    - 'posterior': True if the data is for the posterior samples, False if it is for the prior samples.
    - 'perc_values': the percentiles used. Default is [5, 95].
    - 'data_label': the label for the data (e.g. 'PSR+GW+NICER', 'PSR', etc.)
    - 'custom_color': the color for the plot. If None, the color will be chosen automatically.

    The function will return the data dictionary.

    Parameters
    ----------
    eos_tables : dict
        The dictionary of EOS tables. Should be generated using the EOSManager.interpolate_eos_tables() method.
    posteriors_eos : np.ndarray
        The array of posterior samples. Should be the 'eos_indx' column from the posterior samples file. 
        If the user wants to generate the data for the prior samples, np.arange(int(len(eos_names))) should be used.
    eos_names : list
        The list of EOS names. Should be generated using the EOSManager.eos_names attribute.
    data_label : str
        The label for the data (e.g. 'PSR+GW+NICER', 'PSR', etc.). There are no restrictions on this parameter.
    posterior : bool, optional
        True if the data is for the posterior samples, False if it is for the prior samples. Default is True.
    perc_values : list, optional
        The percentiles used. Default is [5, 95]. 
        NOTE! These are percentile values, not quantile values. Should be in the range [0, 100]. 
        The code will raise warning if the values are in [0, 1] range, but will still run, please be aware of this.
    custom_color : str, optional
        The color for the plot. If None, the color will be chosen automatically. Default is None.
    use_energy_dens : bool, optional
        If True, the energy density will be used as the x-axis parameter instead of the baryon density. Default is False.
    
    Returns
    -------
    density_pressure_data : dict
        The dictionary with the data for the pressure-density plot.
    """

    # generate the grid of pressure values based on the posterior samples
    posteriors_eos_pressure_grid = generate_posterior_eos_param_grid(posteriors_eos, eos_tables, eos_names, 'pressure')

    # calculate the percentiles of the pressure for each baryon density
    pressure_percentiles = np.percentile(posteriors_eos_pressure_grid, perc_values, axis = 1)

    # collect and fill the data in the dictionary
    density_pressure_data = {}

    if use_energy_dens:
        raise NotImplementedError('The energy density is not yet implemented as the x-axis parameter! Please use the baryon density instead.')
    else:
        density_pressure_data['x_dens_grid'] = eos_tables[eos_names[0]]['baryon_density']
    density_pressure_data['y_pres_perc'] = pressure_percentiles
    density_pressure_data['posterior'] = posterior
    density_pressure_data['perc_values'] = perc_values
    density_pressure_data['data_label'] = data_label
    if custom_color is not None:
        density_pressure_data['custom_color'] = custom_color
    
    return density_pressure_data

def pressure_density_plot(*density_pressure_data: dict, use_energy_dens: bool = False, units: str = 'cgs', xlim: tuple = (1e14, 1e16), ylim: tuple = (1e32, 1e38), title: str = None) -> None:
    """
    This function generates a plot of pressure vs. baryon density for the given EOS data.

    Parameters
    ----------
    density_pressure_data : dict
        A dictionary with the following structure:

            {'x_dens_grid': np.ndarray,
             'y_pres_perc': np.ndarray,
             'posterior': bool,
             'perc_values': list,
             'data_label': str,
             'custom_color': str}

            `x_dens_grid`: np.ndarray - grid of baryon densities. 
            If the EOS tables were interpolated, one can use the `baryon_density` column of any EOS table (`eos_tables['eos_0']['baryon_density']`); 

            `y_pres_perc`: np.ndarray - percentiles of pressure values for each density point.
            IMPORTANT! Make sure that the percentiles were calculated of the pressure grids interpolated over the same `x_dens_grid` values.
            Must be of the format (2, N), where N is the number of points in the `x_dens_grid`.
            `y_pres_perc[0, :]` is the lower percentile, `y_pres_perc[1, :]` is the upper percentile.

            `posterior`: bool - if True, the data is considered to be posterior samples. If False, the data is considered to be prior samples.

            `perc_values`: list of 2 percentiles corresponding to the `y_pres_perc` array. If none is provided, the default values are [5, 95].

            `data_label`: str - label for the data used, i. e. 'PSR+GW+NICER', 'PSR+GW', etc.

            `custom_color`: color for the percentiles provided in this data entry. If none is provided, the default colormap is used.
    energy_dens_bool : bool
        If you are using energy density instead of baryon density, set this to True. Default is False.
    units : str
        Units of the data. Default is 'cgs'.
    xlim : tuple
        X-axis limits. Default is (1e14, 1e16). Corresponds the the CGS units.
    ylim : tuple
        Y-axis limits. Default is (1e32, 1e38). Corresponds the the CGS units.
    title: str
        Title of the plot. Default is None. If no title has been provided, the title will be set to generic Pressure vs Baryon Density. 
    
    Returns
    -------
    None
    """
    if use_energy_dens:
        raise NotImplementedError('The energy density is not yet implemented as the x-axis parameter! Please use the baryon density instead.')
        # warnings.warn('The energy density as the x-axis parameter is not fully tested yet! Please be cautious of that when analyzing the resulting plots.')

    colors_def = plt.cm.get_cmap('tab10')
    density_units = {'cgs': 'g/cm$^3$', 'SI': 'kg/m$^3$'}
    pressure_units = {'cgs': 'dyn/cm$^2$', 'SI': 'N/m$^2$'}

    fig, ax = plt.subplots()
    for i, data in enumerate(density_pressure_data):
        x_dens_grid = data['x_dens_grid']
        y_pres_perc = data['y_pres_perc']
        posterior = data['posterior']
        data_label = data['data_label']

        try: 
            perc_values = data['perc_values']
        except KeyError:
            perc_values = [5, 95]
        try: 
            colors = data['custom_color']
        except KeyError:
            colors = colors_def(i)

        if posterior:
            # posteriors
            ax.loglog(x_dens_grid, y_pres_perc[0], color = colors, label = f'{data_label} posterior {perc_values[1] - perc_values[0]}% CI')
            ax.loglog(x_dens_grid, y_pres_perc[1], color = colors)
            ax.fill_between(x_dens_grid, y_pres_perc[0], y_pres_perc[1], color = colors, alpha = 0.3)
        else:
            # priors
            ax.loglog(x_dens_grid, y_pres_perc[0], color = colors, linestyle = '--', label = f'{data_label} {perc_values[1] - perc_values[0]}% CI')
            ax.loglog(x_dens_grid, y_pres_perc[1], color = colors, linestyle = '--')
            ax.fill_between(x_dens_grid, y_pres_perc[0], y_pres_perc[1], color = colors, alpha = 0.3)

    ax.plot((2.5e14, 2.5e14), (1e32, 1e38), color = 'black', linestyle = '--')
    ax.text(2.5e14, 2e32, '$\\rho_{\\text{nuc}}$', ha = 'right', va = 'center', rotation = 'vertical', fontsize = 14)

    ax.plot((5e14, 5e14), (1e32, 1e38), color = 'black', linestyle = '--')
    ax.text(5e14, 2e32, '$2\\rho_{\\text{nuc}}$', ha = 'right', va = 'center', rotation = 'vertical', fontsize = 14)

    ax.plot((1.5e15, 1.5e15), (1e32, 1e38), color = 'black', linestyle = '--')
    ax.text(1.5e15, 2e32, '$6\\rho_{\\text{nuc}}$', ha = 'right', va = 'center', rotation = 'vertical', fontsize = 14)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    ax.set_xlabel(f'Baryon density, $\\rho$ [{density_units[units]}]', fontsize = 16)
    ax.set_ylabel(f'Pressure, $P$ [{pressure_units[units]}]', fontsize = 16)
    ax.legend()
    if title is None:
        ax.set_title('EOS Inference (Pressure vs Baryon Density)', fontsize = 16)
        plt.savefig('Pressure_Density_plot.pdf')
    else:
        ax.set_title(title, fontsize = 16)
        plt.savefig(f'{title}.pdf')
    plt.show()

    return None

def tov_tables_interpolator(tov_tables: dict, m_interp_range: list = [1, 1.9], n_points: int = 500, m_extremum_breakpoint: float = 1.9, verbose: bool = False) -> dict:
    """
    This function interpolates inserted TOV tables and removes all the point from each table if the change in gradient occurs after a breakpoint (`m_extremum_breakpoint`).
    This is done to prevent the double counting in percentiles.

    Parameters
    ----------
    tov_tables : dict
        The dictionary of TOV tables. Should be generated using the EOSManager.tov_tables attribute.
    m_interp_range : list
        The range of the interpolated mass values. Default is [1, 1.9].
    n_points : int
        The number of points in the interpolated mass values. Default is 500.
    m_extremum_breakpoint : float
        The breakpoint value for the extremum in the mass-radius curve. Default is 1.9.
        The value 1.9 was established from the previous analysis of the TOV tables. 
        Many EOS have extremums below 1 Msun, but they do not produce double M-R curves in the region 1 - 1.9 Msun.
        EOS that have extremums above the 1.9 Msun (in our previous analysis) would often have the M-R curve that returns to the region 1 - 1.9 Msun.
        This would cause the double counting of the R values in the percentiles. 
        Therefore, we delete the points from the tov_tables after the extremum if it is above 1.9 Msun.
        Only a few EOS (~20 of 10000) have extremums in the region 1 - 1.9 Msun, and their impact on the percentiles is negligible.
        NOTE! This value may be different for different EOS tables. Please analyze your EOS tables before setting this parameter.
        See an example of the analysis in the notebook: 
        https://github.com/oshaughn/RIT-matters/tree/master/communications/20230613-ROSAV-TabularEOSInference/plotting/EOS_plots.ipynb
    verbose : bool
        If True, the function will print the progress of the interpolation. Default is False.
    
    Returns
    -------
    tov_interp_tables : dict
        The dictionary of interpolated TOV tables.
    """

    tov_interp_tables = {}
    # we modify the TOV tables to avoid the M-R curve double counting problem
    if verbose:
        print(f'Removing the points after the extremum if the extremum is above the breakpoint: {m_extremum_breakpoint} Msun')
        
    for name in natsorted(tov_tables.keys()):
        # to find the extremum, we find the points where the gradient changes sign
        grad = np.gradient(tov_tables[name]['M'])
        grad_prods = grad[:-1] * grad[1:]
        extremum_indx = np.where(grad_prods < 0)

        # we remove the points after the extremum if the extremum is above the breakpoint
        # NOTE! We assume the extremums before the breakpoint to not be significant for the percentiles. 
        # Please make sure to analyze your EOS tables before setting the breakpoint value.
        extremum_mass = tov_tables[name]['M'][extremum_indx]
        mask = extremum_mass > m_extremum_breakpoint
        extremum_indx = extremum_indx[0][mask]
        if len(extremum_indx) == 0:
            continue
        breakpoint_extremum = extremum_indx[0]
        tov_tables[name] = tov_tables[name][:breakpoint_extremum + 1]
    
    if verbose:
        print(f'All the points after the extremum above the breakpoint are removed. Moving to the interpolation.')

    # we interpolate the TOV tables
    m_interp_base = np.linspace(m_interp_range[0], m_interp_range[1], n_points)
    for name in natsorted(tov_tables.keys()):
        tov_interp_tables[name] = {}
        r_interp_vals = np.interp(m_interp_base, tov_tables[name]['M'], tov_tables[name]['R'])
        Lambda_interp_vals_log = np.interp(m_interp_base, tov_tables[name]['M'], np.log(tov_tables[name]['Lambda']))
        tov_interp_tables[name]['M'] = m_interp_base
        tov_interp_tables[name]['R'] = r_interp_vals
        tov_interp_tables[name]['Lambda'] = np.exp(Lambda_interp_vals_log)

    if verbose:
        print(f'The interpolation is completed. Returning the interpolated TOV tables.')

    return tov_interp_tables
    
def mass_radius_plot_data_gen(tov_tables: dict, posteriors_eos: np.ndarray, eos_names: list, data_label: str, posterior: bool = True, tov_interp: bool = False, perc_values: list = [5, 95], custom_color: str = None) -> dict:
    """
    This function generates the data dictionary for the mass-radius plot

    Parameters
    ----------
    tov_tables : dict
        The dictionary of TOV tables. Should be generated using the EOSManager.tov_tables attribute.
    posteriors_eos : np.ndarray
        The array of posterior samples. Should be the 'eos_indx' column from the posterior samples file. 
        If the user wants to generate the data for the prior samples, np.arange(int(len(eos_names))) should be used.
    eos_names : list
        The list of EOS names. Should be generated using the EOSManager.eos_names attribute.
    data_label : str
        The label for the data (e.g. 'PSR+GW+NICER', 'PSR', etc.). There are no restrictions on this parameter.
    posterior : bool
        True if the data is for the posterior samples, False if it is for the prior samples. Default is True.
    tov_interp : bool
        True if the inserted TOV tables are already interpolated. Default is False.
        If False, the code will interpolate the tables using the default settings (linear interpolation, 500 points between 1 and 1.9 Msun, extremum breakpoint at 1.9 Msun).
        See the tov_tables_interpolator() function documentation for more details.
    perc_values : list
        The percentiles used. Default is [5, 95]. 
        NOTE! These are percentile values, not quantile values. Should be in the range [0, 100]. 
        The code will raise warning if the values are in [0, 1] range, but will still run, please be aware of this.
    custom_color : str
        The color for the plot. If None, the color will be chosen automatically. Default is None.
    
    Returns
    -------
    mass_radius_data : dict
        The dictionary with the data for the mass-radius plot.
    
    """

    if not tov_interp:
        tov_tables = tov_tables_interpolator(tov_tables)

    # generate the grid of radius values based on the posterior samples
    posteriors_eos_radius_grid = generate_posterior_eos_param_grid(posteriors_eos, tov_tables, eos_names, 'R')

    # calculate the percentiles of the radius for each mass point
    radius_percentiles = np.percentile(posteriors_eos_radius_grid, perc_values, axis = 1)

    # collect and fill the data in the dictionary
    mass_radius_data = {}
    mass_radius_data['x_rad_perc'] = radius_percentiles
    mass_radius_data['y_mass_grid'] = tov_tables[list(tov_tables.keys())[0]]['M']
    mass_radius_data['posterior'] = posterior
    mass_radius_data['perc_values'] = perc_values
    mass_radius_data['data_label'] = data_label
    if custom_color is not None:
        mass_radius_data['custom_color'] = custom_color

    return mass_radius_data

def mass_radius_plot(*mass_radius_data: dict, xlim: tuple = (8, 17), ylim: tuple = (1, 1.9), title: str = None) -> None:
    """
    This function generates a plot of mass vs. radius for the given EOS data.

    Parameters:
    -----------
    mass_radius_data : dict
        A dictionary with the following structure:

            {'x_rad_perc': np.ndarray,
             'y_mass_grid': np.ndarray,
             'posterior': bool,
             'perc_values': list,
             'data_label': str,
             'custom_color': str}

            `x_rad_perc` : np.ndarray - the percentiles of the radius for each mass point
            IMPORTANT! Make sure that the percentiles were calculated of the radius grids interpolated over the same `y_mass_grid` values.
            Must be of the format (2, N), where N is the number of points in the `y_mass_grid`.
            `x_rad_perc[0, :]` is the lower percentile, `x_rad_perc[1, :]` is the upper percentile.

            `y_mass_grid` : np.ndarray - the grid of mass values. 
            If the TOV tables were interpolated, one can use the `M` values from any of the interpolated tables. (tov_interp_tables['eos_0']['M'])

            `posterior` : bool - True if the data is for the posterior samples, False if it is for the prior samples.

            `perc_values` : list - the percentiles used in `x_rad_perc`. Default is [5, 95].

            `data_label` : str - the label for the data (e.g. 'PSR+GW+NICER', 'PSR', etc.). There are no restrictions on this parameter.

            `custom_color` : color for the percentiles provided in this data entry. If none is provided, the default colormap is used.
            
    xlim : tuple
        X-axis limits. Default is (8, 17). Corresponds the the km units.
    ylim : tuple
        Y-axis limits. Default is (1, 1.9). Corresponds the the solar mass units.
    title: str
        Title of the plot. Default is None. If no title has been provided, the title will be set to generic Mass vs Radius.
    
    Returns:
    --------
    None
    """

    colors_def = plt.cm.get_cmap('tab10')
    mass_units = 'M$_\odot$'
    radius_units = 'km'

    fig, ax = plt.subplots()
    for i, data in enumerate(mass_radius_data):
        x_rad_perc = data['x_rad_perc']
        y_mass_grid = data['y_mass_grid']
        posterior = data['posterior']
        data_label = data['data_label']
        perc_values = data['perc_values']
        try:
            colors = data['custom_color']
        except KeyError:
            colors = colors_def(i)

        if posterior:
            # posteriors
            ax.plot(x_rad_perc[0], y_mass_grid, color = colors, label = f'{data_label} posterior {perc_values[1] - perc_values[0]}% CI')
            ax.plot(x_rad_perc[1], y_mass_grid, color = colors)
            ax.fill_betweenx(y_mass_grid, x_rad_perc[0], x_rad_perc[1], color = colors, alpha = 0.3)
        else:
            # priors
            ax.plot(x_rad_perc[0], y_mass_grid, color = colors, linestyle = '--', label = f'{data_label} {perc_values[1] - perc_values[0]}% CI')
            ax.plot(x_rad_perc[1], y_mass_grid, color = colors, linestyle = '--')
            ax.fill_betweenx(y_mass_grid, x_rad_perc[0], x_rad_perc[1], color = colors, alpha = 0.3)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    ax.set_xlabel(f'Radius, $R$ [{radius_units}]', fontsize = 16)
    ax.set_ylabel(f'Mass, $M$ [{mass_units}]', fontsize = 16)
    ax.legend()
    if title is None:
        ax.set_title('EOS Inference (Mass vs Radius)', fontsize = 16)
        plt.savefig('Mass_Radius_plot.pdf')
    else:
        ax.set_title(title, fontsize = 16)
        plt.savefig(f'{title}.pdf')
    plt.show()

def posterior_hist_data_gen(posteriors_eos: np.ndarray, EOSManager_Local: EOSManager.EOSSequenceLandry, data_label: str, m_ref: float = 1.4) -> dict:
    """
    This function generates the data for the posterior histogram plots. 
    It is designed as an auxiliary function to the `posterior_hist_plot` function and should be used as an input to that function.
    The user has to provide the posterior samples in terms of the EOS table index, the EOSManager object, the data label, and the reference mass (default is 1.4 M_sun).

    Parameters
    ----------
    posteriors_eos: np.ndarray
        The posterior samples in terms of the EOS table index.
    EOSManager_Local: EOSManager.EOSSequenceLandry
        The EOSManager object. Load the tabular EOS data separately and provide it as an input to avoid loading it multiple times.
    data_label: str
        The data label for the plot.
    m_ref: float
        The reference mass for the EOS. Default is 1.4 M_sun. 
        The code will use functions `R_of_m_indx` and `lambda_of_m_indx` from the EOSManager object to calculate the radius and tidal deformability for a given mass m_ref.
        Should be in range [1.0, 2.5]
    
    Returns
    -------
    posterior_hist_data: dict
        A dictionary containing the data for the posterior histogram plot.
        The dictionary contains the following keys:
        - 'R': the radius data of each of the posterior samples
        - 'Lambda_': the tidal deformability data of each of the posterior samples
        - 'R_mean_std': the mean and standard deviation of radius of the posterior samples (tuple) of format (mean, std)
        - 'Lambda_mean_std': the mean and standard deviation of tidal deformability of the posterior samples (tuple) of format (mean, std)
        - 'data_label': the data label for the plot
        - 'm_ref': the reference mass used for the EOS calculations
    
    Raises
    ------
    ValueError
        If the reference mass is not in range [1.0, 2.5]
    """
    if m_ref < 1.0 or m_ref > 2.5:
        raise ValueError("The reference mass should be in range [1.0, 2.5]")

    posterior_hist_data = {}
    R = np.zeros(len(posteriors_eos))
    Lambda_ = np.zeros(len(posteriors_eos))
    for i, eos_indx in enumerate(posteriors_eos):
        R[i] = EOSManager_Local.R_of_m_indx(m_ref, int(eos_indx))
        Lambda_[i] = EOSManager_Local.lambda_of_m_indx(m_ref, int(eos_indx))
    
    R_mean_std = (np.mean(R), np.std(R))
    Lambda_mean_std = (np.mean(Lambda_), np.std(Lambda_))

    posterior_hist_data['R'] = R
    posterior_hist_data['Lambda_'] = Lambda_
    posterior_hist_data['R_mean_std'] = R_mean_std
    posterior_hist_data['Lambda_mean_std'] = Lambda_mean_std
    posterior_hist_data['data_label'] = data_label
    posterior_hist_data['m_ref'] = m_ref

    return posterior_hist_data

def posterior_hist_plot(*posterior_hist_data: dict, r_lim: tuple = (8, 17), lambda_lim: tuple = (0, 1000), title: str = None) -> None:
    """
    This function generates the posterior histogram plots.

    Parameters
    ----------
    posterior_hist_data: dict
        The data for the posterior histogram plot. 
        The user can provide multiple dictionaries as input, each containing the data for a different posterior sample.
        The dictionaries should be generated using the `posterior_hist_data_gen` function and should contain the following keys:
        - 'R': the radius data of each of the posterior samples
        - 'Lambda_': the tidal deformability data of each of the posterior samples
        - 'R_mean_std': the mean and standard deviation of radius of the posterior samples (tuple) of format (mean, std)
        - 'Lambda_mean_std': the mean and standard deviation of tidal deformability of the posterior samples (tuple) of format (mean, std)
        - 'data_label': the data label for the plot
        - 'm_ref': the reference mass used for the EOS calculations

    r_lim: tuple, optional
        The limits for the radius axis. Default is (8, 17).
    lambda_lim: tuple, optional
        The limits for the tidal deformability axis. Default is (0, 1000).
    title: str, optional
        The title for the plots. Default is None.

    Returns
    -------
    None  

    Raises
    ------
    ValueError
        If the reference masses for the different data entries are not the same.
    """
    if len(posterior_hist_data) > 1:
        for i in range(len(posterior_hist_data) - 1):
            if posterior_hist_data[i]['m_ref'] != posterior_hist_data[i + 1]['m_ref']:
                raise ValueError("The reference masses for the EOS calculations are not the same.")

    colors = plt.cm.get_cmap('tab10')
    m_ref = posterior_hist_data[0]['m_ref']

    fig, ax1 = plt.subplots()
    for i, data in enumerate(posterior_hist_data):
        R_data = data['R']
        label_i = data['data_label']
        alpha_i = 1 - 0.1 * i

        bins = np.linspace(r_lim[0], r_lim[1], 100)
        ax1.hist(R_data, bins = bins, color = colors(i), label = label_i, alpha = alpha_i, density = True)
        # sns.kdeplot(R_data, color = colors(i))
        ax1.axvline(x = data['R_mean_std'][0], color = colors(i), linestyle = '--')
        ax1.axvspan(data['R_mean_std'][0] - data['R_mean_std'][1], data['R_mean_std'][0] + data['R_mean_std'][1], alpha = 0.1, color = colors(i))
        print(f'{label_i} R at {m_ref} M_sun: {data["R_mean_std"][0]:.2f} +/- {data["R_mean_std"][1]:.2f} km')

    ax1.set_xlabel(f'Radius at $M = {m_ref} M_\\odot$ [km]', fontsize = 16)
    ax1.set_ylabel('Probability Density', fontsize = 16)
    ax1.set_xlim(r_lim)
    ax1.legend()
    ax1.grid(True)
    if title is not None:
        ax1.set_title(title, fontsize = 16)
        plt.savefig(f'{title}_radius.pdf')
    else:
        ax1.set_title('Posterior distribution of $R_{' + str(m_ref) + '}$', fontsize = 16)
        plt.savefig(f'R_{m_ref}_posterior.pdf')
    plt.show()
    
    fig, ax2 = plt.subplots()
    for i, data in enumerate(posterior_hist_data):
        Lambda_data = data['Lambda_']
        label_i = data['data_label']
        alpha_i = 1 - 0.1 * i

        bins = np.linspace(lambda_lim[0], lambda_lim[1], 200)
        ax2.hist(Lambda_data, bins = bins, color = colors(i), label = label_i, alpha = alpha_i, density = True)
        # sns.kdeplot(Lambda_data, color = colors(i))
        ax2.axvline(x = data['Lambda_mean_std'][0], color = colors(i), linestyle = '--')
        ax2.axvspan(data['Lambda_mean_std'][0] - data['Lambda_mean_std'][1], data['Lambda_mean_std'][0] + data['Lambda_mean_std'][1], alpha = 0.1, color = colors(i))
        print(f'{label_i} Lambda at {m_ref} M_sun: {data["Lambda_mean_std"][0]:.2f} +/- {data["Lambda_mean_std"][1]:.2f}')

    ax2.set_xlabel(f'Tidal Deformability at $M = {m_ref} M_\\odot$', fontsize = 16)
    ax2.set_ylabel('Probability Density', fontsize = 16)
    ax2.set_xlim(lambda_lim)
    ax2.legend()
    ax2.grid(True)
    if title is not None:
        ax2.set_title(title, fontsize = 16)
        plt.savefig(f'{title}_lambda.pdf')
    else:
        ax2.set_title('Posterior distribution of ${{\\Lambda}}_{' + str(m_ref)+ '}$', fontsize = 16)
        plt.savefig(f'Lambda_{m_ref}_posterior.pdf')
    plt.show()

    return None

def LambdaTilderatio_data_gen(posterior_samples: np.ndarray, EOSManager_Local: EOSManager.EOSSequenceLandry, data_label: str) -> dict:
    """
    This function generates the data for the Lambda tilde ratio plot. This is a ratio of the tidal deformability Lambda tilde to the ordering statistics S.
    Tidal deformability Lambda tilde is calculated from the individual masses and lambdas of the binary components.
    The ordering statistics S is calculated from the reference mass of the binary and the EOS table index.
    This ratio serves as a test for the credibility of the EOS inference ordering statistics S.
    The data will be generated based on the posterior samples and the EOS tables.
    The posterior samples should be loaded from the posterior samples file. 
    The user can use the 'posterior_samples_all' key from the `posterior_data_loader` if `out_all` is set to True in the `posterior_data_loader` function.

    Parameters
    ----------
    posterior_samples : np.ndarray
        The posterior samples. Should be loaded from the posterior samples file. Must contain the following columns: 'm1', 'm2', 'mc', 'q', 'lambda1', 'lambda2', 'eos_indx'.
    EOSManager_Local : EOSManager.EOSSequenceLandry
        The EOS manager object. Load the tabular EOS data separately and provide it as an input to avoid loading it multiple times.
    data_label : str
        The label for the data. There are no restrictions on this parameter.
    
    Returns
    -------
    LambdaTilderatio_data : dict
        The dictionary with the data for the Lambda tilde ratio plot.
        It will contain the following keys:
        - 'q': the mass ratio of the binary
        - 'Lambda_tilde_ratio': the ratio of the tidal deformability Lambda tilde to the ordering statistics S
        - 'data_label': the label for the data (e.g. 'PSR+GW+NICER', 'PSR', etc.)
    """
    # load the proper columns from the posterior samples
    m1 = posterior_samples['m1']
    m2 = posterior_samples['m2']
    mc = posterior_samples['mc']
    q = posterior_samples['q']
    Lambda1 = posterior_samples['lambda1']
    Lambda2 = posterior_samples['lambda2']
    eos_indx = posterior_samples['eos_indx']

    # calculate the real Lambda tilde values
    Lambda_tilde = lalsimutils.tidal_lambda_tilde(m1, m2, Lambda1, Lambda2)[0]

    # calculate the ordering statistics S values
    S = np.zeros(len(mc))
    for i in range(len(mc)):
        m_ref = mc[i] * np.power(2, 1/5)
        S[i] = EOSManager_Local.lambda_of_m_indx(m_ref, int(eos_indx[i]))
    
    # calculate the Lambda tilde ratio
    Lambda_tilde_ratio = Lambda_tilde / S

    # collect the data in the dictionary
    LambdaTilderatio_data = {}
    LambdaTilderatio_data['q'] = q
    LambdaTilderatio_data['Lambda_tilde_ratio'] = Lambda_tilde_ratio
    LambdaTilderatio_data['data_label'] = data_label

    return LambdaTilderatio_data

def LambdaTilderatio_plot(*LambdaTilderatio_data: dict, qlim: tuple = None, ylim: tuple = (0, 2), title: str = None) -> None:
    """
    This function generates a plot of the Lambda tilde ratio vs. mass ratio for the given EOS data.

    Parameters
    ----------
    LambdaTilderatio_data : np.ndarray
        The data for the Lambda tilde ratio plot. Should be generated using the LambdaTilderatio_data_gen() function.
    qlim : tuple
        The mass ratio limits. Default is None. If None, the limits will be set to (q_min, q_max). 
        If the user provides the limits that do not fit in [0, 1] range, the code will raise a ValueError.
    ylim : tuple
        Y-axis limits. Default is (0, 2).
    title : str
        Title of the plot. Default is None. If no title has been provided, the title will be set to generic Lambda tilde ratio vs Mass ratio. 
    
    Returns
    -------
    None
    """
    if qlim is not None:
        if qlim[0] < 0 or qlim[1] > 1:
            raise ValueError('The mass ratio limits must be in the range [0, 1]! Please provide the limits that fit in this range.')

    colors = plt.cm.get_cmap('tab10')
    q_min = 1
    q_max = 0

    fig, ax1 = plt.subplots()
    for i, data in enumerate(LambdaTilderatio_data):
        q = data['q']
        Lambda_tilde_ratio = data['Lambda_tilde_ratio']
        ax1.plot(q, Lambda_tilde_ratio, 'o', markersize = 1, color = colors(i), label = data['data_label'])
        q_min = min(q_min, np.min(q))
        q_max = max(q_max, np.max(q))
    
    ax1.set_xlabel('Mass ratio, $q$', fontsize = 16)
    ax1.set_ylabel('Lambda tilde ratio, $\\tilde{\Lambda}/S$', fontsize = 16)
    if qlim is None:
        qlim = (q_min, q_max)
    ax1.set_xlim(qlim)
    ax1.set_ylim(ylim)
    ax1.legend()
    ax1.grid(True)
    if title is None:
        ax1.set_title('Tidal deformability ratio vs Mass ratio', fontsize = 16)
        plt.savefig(f'Lambda_tilde_ratio_{ylim[0]}-{ylim[1]}.pdf')
    else:
        ax1.set_title(title, fontsize = 16)
        plt.savefig(f'{title}.pdf')
    plt.show()

    return None